"""Self-tests for the V3 model-candidate dump bridge.

These are plain Python tests because this checkout does not rely on pytest.
Run:
  trm\Scripts\python.exe scripts\test_model_dump_bridge.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from model_candidate_dump import load_model_dump, write_model_dump
import eval_arc_agi1
import verify_and_select_candidates
import dump_model_candidates


def _make_combined(root: Path) -> Path:
    combined = root / "combined"
    combined.mkdir(parents=True, exist_ok=True)
    challenges = {
        "toy_task": {
            "train": [
                {"input": [[0]], "output": [[0]]},
                {"input": [[1]], "output": [[1]]},
            ],
            "test": [{"input": [[0]]}],
        }
    }
    solutions = {"toy_task": [[[2]]]}
    (combined / "arc-agi_evaluation_challenges.json").write_text(json.dumps(challenges), encoding="utf-8")
    (combined / "arc-agi_evaluation_solutions.json").write_text(json.dumps(solutions), encoding="utf-8")
    return combined


def _flat_token(raw_colour: int, side: int = 30) -> np.ndarray:
    arr = np.zeros((side, side), dtype=np.int64)
    arr[0, 0] = raw_colour + eval_arc_agi1.COLOR_OFFSET
    return arr.reshape(-1)


def test_dump_round_trip_and_eval_floor_safe() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        combined = _make_combined(root)
        dump_path = root / "topk.npz"
        write_model_dump(
            dump_path,
            [
                {
                    "task_id": "toy_task",
                    "test_index": 0,
                    "candidates": [_flat_token(1), _flat_token(2)],
                    "vote_counts": [9, 3],
                }
            ],
        )

        loaded = load_model_dump(dump_path)
        assert ("toy_task", 0) in loaded
        assert loaded[("toy_task", 0)]["candidates"].shape == (2, 900)

        res = eval_arc_agi1.evaluate("evaluation", combined=combined, model_dump=dump_path, quiet=True)
        assert res["n"] == 1
        assert res["model_floor"] == 0, "TRM majority/floor candidate is intentionally wrong in this fixture"
        assert res["exact"] == 1, "top-K bridge must allow attempt1 to rescue a non-floor correct candidate"
        assert res["exact"] >= res["model_floor"], "2-attempt selector must remain floor-safe"


def test_dump_validation_rejects_bad_shape() -> None:
    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.npz"
        np.savez_compressed(
            bad,
            task_ids=np.array(["x"], dtype=object),
            test_indices=np.array([0], dtype=np.int64),
            candidates=np.zeros((1, 2, 899), dtype=np.int64),
            vote_counts=np.ones((1, 2), dtype=np.float32),
        )
        try:
            load_model_dump(bad)
        except ValueError as exc:
            assert "900" in str(exc)
        else:
            raise AssertionError("bad candidate length must be rejected")


def test_dump_kind_filter_and_floor_order_are_stable() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dump_path = root / "typed.npz"
        floor = _flat_token(0)
        fold_candidate = _flat_token(4)
        test_candidate = _flat_token(7)
        write_model_dump(
            dump_path,
            [
                {
                    "task_id": "toy",
                    "test_index": 0,
                    "record_kind": "fold",
                    "candidates": [floor, fold_candidate],
                    # The non-floor intentionally has the higher score; candidate
                    # order still defines candidate0=floor and must not be sorted.
                    "vote_counts": [1, 9],
                },
                {
                    "task_id": "toy",
                    "test_index": 0,
                    "record_kind": "test",
                    "candidates": [floor, test_candidate],
                    "vote_counts": [1, 8],
                },
            ],
        )

        try:
            load_model_dump(dump_path)
        except ValueError as exc:
            assert "record_kind" in str(exc)
        else:
            raise AssertionError("typed dumps require an explicit kind to avoid fold/test key collisions")

        folds = load_model_dump(dump_path, kind="fold")
        tests = load_model_dump(dump_path, kind="test")
        assert ("toy", 0) in folds and ("toy", 0) in tests
        assert np.array_equal(folds[("toy", 0)]["candidates"][0], floor), "fold floor must stay row 0"
        assert np.array_equal(folds[("toy", 0)]["candidates"][1], fold_candidate)
        assert np.array_equal(tests[("toy", 0)]["candidates"][1], test_candidate)


def test_typed_dump_rejects_mixed_record_kinds() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dump_path = root / "mixed.npz"
        try:
            write_model_dump(
                dump_path,
                [
                    {
                        "task_id": "toy",
                        "test_index": 0,
                        "record_kind": "fold",
                        "candidates": [_flat_token(0)],
                        "vote_counts": [1],
                    },
                    {
                        "task_id": "toy",
                        "test_index": 0,
                        "candidates": [_flat_token(1)],
                        "vote_counts": [1],
                    },
                ],
            )
        except ValueError as exc:
            assert "record_kind" in str(exc)
        else:
            raise AssertionError("mixed typed/untyped records must be rejected")

        try:
            write_model_dump(
                root / "null_kind.npz",
                [
                    {
                        "task_id": "toy",
                        "test_index": 0,
                        "record_kind": None,
                        "candidates": [_flat_token(0)],
                        "vote_counts": [1],
                    },
                ],
            )
        except ValueError as exc:
            assert "record_kind" in str(exc)
        else:
            raise AssertionError("record_kind=None must be rejected")


def test_fold_jobs_do_not_pass_heldout_output_as_model_labels() -> None:
    task = {
        "train": [
            {"input": [[0]], "output": [[9]]},
            {"input": [[1]], "output": [[8]]},
        ],
        "test": [{"input": [[0]]}],
    }
    jobs = dump_model_candidates._episode_jobs_for_task(
        task_id="leak_check",
        pid=1,
        task=task,
        context_limit=2,
        blank_pid=0,
        blank_fold_pid=True,
        include_folds=True,
        include_tests=False,
    )
    assert jobs and all(j["record_kind"] == "fold" for j in jobs)
    for job in jobs:
        assert np.array_equal(job["labels"], job["inputs"]), (
            "fold model inference must not receive the held-out support output as labels"
        )


def test_offline_lodo_model_dump_candidate() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dump_path = root / "folds.npz"
        # Two-demo LODO task. Fold 0 has a model non-floor candidate that exactly
        # matches demo 0's output; fold 1 likewise for demo 1.
        d0_in, d0_out = _flat_token(0), _flat_token(2)
        d1_in, d1_out = _flat_token(1), _flat_token(3)
        write_model_dump(
            dump_path,
            [
                {"task_id": "toy", "test_index": 0, "candidates": [_flat_token(0), d0_out], "vote_counts": [5, 1]},
                {"task_id": "toy", "test_index": 1, "candidates": [_flat_token(1), d1_out], "vote_counts": [5, 1]},
            ],
        )
        dump = load_model_dump(dump_path)
        res = verify_and_select_candidates.evaluate_task(
            [(torch_from_np(d0_in), torch_from_np(d0_out)), (torch_from_np(d1_in), torch_from_np(d1_out))],
            side=30,
            task_id="toy",
            model_dump=dump,
        )
        assert res["scores"]["model_dump"][0] == 1.0
        assert res["selector_exact"] >= res["floor_exact"]


def test_compose_test_uses_fold_ranked_model_test_candidates() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        challenges = {
            "toy_task": {
                "train": [
                    {"input": [[0]], "output": [[2]]},
                    {"input": [[1]], "output": [[3]]},
                ],
                "test": [{"input": [[0]]}],
            }
        }
        solutions = {"toy_task": [[[7]]]}
        ch_path = root / "challenges.json"
        sol_path = root / "solutions.json"
        ch_path.write_text(json.dumps(challenges), encoding="utf-8")
        sol_path.write_text(json.dumps(solutions), encoding="utf-8")

        dump_path = root / "typed_model_dump.npz"
        write_model_dump(
            dump_path,
            [
                {
                    "task_id": "toy_task",
                    "test_index": 0,
                    "record_kind": "fold",
                    "candidates": [_flat_token(0), _flat_token(2)],
                    "vote_counts": [9, 1],
                },
                {
                    "task_id": "toy_task",
                    "test_index": 1,
                    "record_kind": "fold",
                    "candidates": [_flat_token(1), _flat_token(3)],
                    "vote_counts": [9, 1],
                },
                {
                    "task_id": "toy_task",
                    "test_index": 0,
                    "record_kind": "test",
                    "candidates": [_flat_token(0), _flat_token(7)],
                    "vote_counts": [9, 1],
                },
            ],
        )

        buf = StringIO()
        with redirect_stdout(buf):
            summary = verify_and_select_candidates.run_compose_test(
                side=30,
                challenges=ch_path,
                solutions=sol_path,
                categories_csv=root / "missing_categories.csv",
                include_committed=False,
                model_dump_path=dump_path,
            )
        out = buf.getvalue()
        assert summary["total_solved_at2"] == 1
        assert summary["winning_family"]["model_dump"] == 1
        assert "model_dump" in out


def test_compose_test_keeps_deterministic_floor_attempt_when_model_ranks_first() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # The support demos conflict under the modal floor, so model_dump wins
        # LOOCV. On the real test, the deterministic floor is correct and the
        # model non-floor is wrong; floor-safe selection must still solve @2.
        challenges = {
            "toy_floor": {
                "train": [
                    {"input": [[0]], "output": [[2]]},
                    {"input": [[0]], "output": [[3]]},
                ],
                "test": [{"input": [[0]]}],
            }
        }
        solutions = {"toy_floor": [[[2]]]}
        ch_path = root / "challenges.json"
        sol_path = root / "solutions.json"
        ch_path.write_text(json.dumps(challenges), encoding="utf-8")
        sol_path.write_text(json.dumps(solutions), encoding="utf-8")
        dump_path = root / "model_first_floor_safe.npz"
        write_model_dump(
            dump_path,
            [
                {
                    "task_id": "toy_floor",
                    "test_index": 0,
                    "record_kind": "fold",
                    "candidates": [_flat_token(0), _flat_token(2)],
                    "vote_counts": [9, 1],
                },
                {
                    "task_id": "toy_floor",
                    "test_index": 1,
                    "record_kind": "fold",
                    "candidates": [_flat_token(0), _flat_token(3)],
                    "vote_counts": [9, 1],
                },
                {
                    "task_id": "toy_floor",
                    "test_index": 0,
                    "record_kind": "test",
                    "candidates": [_flat_token(8), _flat_token(7)],
                    "vote_counts": [9, 1],
                },
            ],
        )

        summary = verify_and_select_candidates.run_compose_test(
            side=30,
            challenges=ch_path,
            solutions=sol_path,
            categories_csv=root / "missing_categories.csv",
            include_committed=False,
            model_dump_path=dump_path,
        )
        assert summary["total_solved_at2"] == 1


def test_eval_rejects_typed_dump_without_test_records() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        combined = _make_combined(root)
        fold_only = root / "fold_only.npz"
        write_model_dump(
            fold_only,
            [
                {
                    "task_id": "toy_task",
                    "test_index": 0,
                    "record_kind": "fold",
                    "candidates": [_flat_token(0)],
                    "vote_counts": [1],
                }
            ],
        )
        try:
            eval_arc_agi1.evaluate("evaluation", combined=combined, model_dump=fold_only, quiet=True)
        except ValueError as exc:
            assert "zero target-test records" in str(exc)
        else:
            raise AssertionError("eval must reject fold-only typed dumps instead of silently reporting DSL-only")


def torch_from_np(arr: np.ndarray):
    import torch

    return torch.from_numpy(arr.astype(np.int64))


if __name__ == "__main__":
    test_dump_round_trip_and_eval_floor_safe()
    test_dump_validation_rejects_bad_shape()
    test_dump_kind_filter_and_floor_order_are_stable()
    test_typed_dump_rejects_mixed_record_kinds()
    test_fold_jobs_do_not_pass_heldout_output_as_model_labels()
    test_offline_lodo_model_dump_candidate()
    test_compose_test_uses_fold_ranked_model_test_candidates()
    test_compose_test_keeps_deterministic_floor_attempt_when_model_ranks_first()
    test_eval_rejects_typed_dump_without_test_records()
    print("test_model_dump_bridge PASS")
