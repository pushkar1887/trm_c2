"""Regression tests for verifier/selector safety bugs.

Plain Python tests; run with:
  trm\Scripts\python.exe scripts\test_verify_selector_regressions.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import model_candidate_dump
import solve
import verify_and_select_candidates as vsc


def _flat(raw_colour: int, side: int = 30) -> torch.Tensor:
    arr = torch.zeros(side * side, dtype=torch.long)
    arr[0] = raw_colour + vsc.COLOR_OFFSET
    return arr


def test_score_lodo_treats_wrong_shape_as_failed_fold() -> None:
    def wrong_shape_predict(_sin, _sout, _tin, _side):
        return torch.zeros(899, dtype=torch.long)

    demos = [(_flat(0), _flat(1)), (_flat(2), _flat(3))]
    score = vsc.score_lodo(demos, wrong_shape_predict, side=30)

    assert score == (0.0, 0.0), "wrong-shape candidate must fail the fold, not crash or score cells"


def test_model_dump_missing_fold_scores_zero_not_none() -> None:
    with np_temp_dump() as dump:
        d0_in, d0_out = _flat(0), _flat(2)
        d1_in, d1_out = _flat(1), _flat(3)
        d2_in, d2_out = _flat(4), _flat(5)
        model_candidate_dump.write_model_dump(
            dump,
            [
                {
                    "task_id": "toy",
                    "test_index": 0,
                    "candidates": [d0_in.numpy(), d0_out.numpy()],
                    "vote_counts": [5, 1],
                },
                {
                    "task_id": "toy",
                    "test_index": 1,
                    "candidates": [d1_in.numpy(), d1_out.numpy()],
                    "vote_counts": [5, 1],
                },
            ],
        )
        loaded = model_candidate_dump.load_model_dump(dump)
        score = vsc.score_model_dump_lodo(
            [(d0_in, d0_out), (d1_in, d1_out), (d2_in, d2_out)],
            side=30,
            task_id="toy",
            model_dump=loaded,
        )

    assert score is not None
    assert abs(score[0] - (2.0 / 3.0)) < 1e-6, "missing fold should contribute exact=0, not drop candidate"


class np_temp_dump:
    def __enter__(self):
        import tempfile

        self._td = tempfile.TemporaryDirectory()
        self.path = Path(self._td.name) / "dump.npz"
        return self.path

    def __exit__(self, exc_type, exc, tb):
        self._td.cleanup()


def test_relocate_second_candidate_is_scored_independently() -> None:
    import models.recursive_reasoning.object_rule_bank as orb

    original = orb.rearrange_candidates

    def fake_rearrange_candidates(_sin, _sout, tin, _side, k=2):
        correct = _flat(8 if int(tin[0]) == vsc.COLOR_OFFSET else 9)
        wrong = _flat(7)
        return [(correct, ("first",)), (wrong, ("second",))][:k]

    try:
        orb.rearrange_candidates = fake_rearrange_candidates
        demos = [(_flat(0), _flat(8)), (_flat(1), _flat(9))]

        first = vsc.score_relocate_k_lodo(demos, side=30, k_index=0)
        second = vsc.score_relocate_k_lodo(demos, side=30, k_index=1)
    finally:
        orb.rearrange_candidates = original

    assert first == (1.0, 1.0)
    assert second is not None
    assert second[0] == 0.0, "relocate@2 must not borrow relocate@1's exact LODO score"
    assert second[1] < first[1], "relocate@2 must carry its own similarity score"


def test_hole_filler_recolor_contains_no_dead_ground_truth_mutation() -> None:
    src = inspect.getsource(solve._hole_filler_recolor_solve)

    assert "gout[h]" not in src, "dead guarded function must not retain held-out-output mutation code"


def test_committed_predict_threads_rule_library_argument() -> None:
    import solve as solve_mod

    sentinel = object()
    original = solve_mod._committed_solve

    def fake_committed_solve(_gin, _gout, _sup, _h, _geo_ops, _labels, rule_lib=None, return_trace=False):
        assert rule_lib is sentinel
        return None, None, {"attempt1_source": "fake"}

    try:
        solve_mod._committed_solve = fake_committed_solve
        pred, trace = vsc.committed_predict_with_trace(
            torch.stack([_flat(0), _flat(1)]),
            torch.stack([_flat(2), _flat(3)]),
            _flat(4),
            30,
            rule_lib=sentinel,
        )
    finally:
        solve_mod._committed_solve = original

    assert pred.shape == (900,)
    assert trace["attempt1_source"] == "fake"


if __name__ == "__main__":
    test_score_lodo_treats_wrong_shape_as_failed_fold()
    test_model_dump_missing_fold_scores_zero_not_none()
    test_relocate_second_candidate_is_scored_independently()
    test_hole_filler_recolor_contains_no_dead_ground_truth_mutation()
    test_committed_predict_threads_rule_library_argument()
    print("test_verify_selector_regressions PASS")
