"""Regression tests for the minimal relational-color offline probe.

Plain script by design: this workspace does not consistently have pytest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.recursive_reasoning.relational_color_probe import (  # noqa: E402
    evaluate_lodo_task,
    infer_rule_from_support,
    pairdelta_intent_summary,
)
from scripts.run_relational_color_probe import select_target_pool, summarize_numeric  # noqa: E402


def _toy_pair(src_colour: int, dst_colour: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((4, 4), 3, dtype=np.int64)
    inp[1:3, 1:3] = src_colour
    out = inp.copy()
    out[1:3, 1:3] = dst_colour
    return inp, out


def test_infers_where_and_value_from_support() -> None:
    support = [_toy_pair(1, 2), _toy_pair(1, 2)]
    rule = infer_rule_from_support([p[0] for p in support], [p[1] for p in support])
    assert rule.valid, rule.reason
    assert rule.where.name.startswith("input_color=1"), rule.where.name
    assert rule.value_map[1] == 2, rule.value_map

    test_in, test_out = _toy_pair(1, 2)
    pred, diag = rule.apply(test_in, expected_output=test_out)
    assert diag["exact"] == 1.0, diag
    assert np.array_equal(pred, test_out)


def test_lodo_reports_exact_and_copy_preservation() -> None:
    pairs = [_toy_pair(1, 2), _toy_pair(1, 2), _toy_pair(1, 2)]
    result = evaluate_lodo_task("toy", [p[0] for p in pairs], [p[1] for p in pairs])
    assert result["lodo_folds"] == 3, result
    assert result["lodo_exact"] == 1.0, result
    assert result["unchanged_acc"] == 1.0, result
    assert result["where_f1"] == 1.0, result
    assert result["value_acc"] == 1.0, result


def test_value_accuracy_is_scored_on_selected_changed_cells() -> None:
    support = [_toy_pair(1, 2), _toy_pair(1, 2)]
    rule = infer_rule_from_support([p[0] for p in support], [p[1] for p in support])
    test_in, test_out = _toy_pair(1, 2)
    test_out[0, 0] = 4
    _pred, diag = rule.apply(test_in, expected_output=test_out)
    assert diag["exact"] == 0.0, diag
    assert diag["value_acc"] == 1.0, diag
    assert diag["changed_acc"] < 1.0, diag


def test_invalid_rule_reports_copy_preservation() -> None:
    support_in = [np.zeros((2, 2), dtype=np.int64)]
    support_out = [np.zeros((3, 3), dtype=np.int64)]
    rule = infer_rule_from_support(support_in, support_out)
    assert not rule.valid
    test_in = np.array([[1, 1], [0, 0]], dtype=np.int64)
    test_out = test_in.copy()
    test_out[0, 0] = 2
    _pred, diag = rule.apply(test_in, expected_output=test_out)
    assert diag["where_fpr"] == 0.0, diag
    assert diag["unchanged_acc"] == 1.0, diag
    assert diag["changed_acc"] == 0.0, diag


def test_pairdelta_intent_summary_sees_sparse_shape_preserved_recolor() -> None:
    pairs = [_toy_pair(1, 2), _toy_pair(1, 2)]
    summary = pairdelta_intent_summary([p[0] for p in pairs], [p[1] for p in pairs])
    assert summary["shape_preserved"] == 1.0, summary
    assert 0.0 < summary["changed_rate"] < 0.5, summary
    assert summary["dominant_source_color"] == 1, summary
    assert summary["dominant_target_color"] == 2, summary
    assert summary["conditional_recolor_score"] > 0.0, summary


def test_report_helpers_select_pool_and_summarize() -> None:
    rows = [
        {"task_id": "a", "bucket": "both_fail", "close_miss": "1", "where_f1": "0.5"},
        {"task_id": "b", "bucket": "both_fail", "close_miss": "0", "where_f1": "0.0"},
        {"task_id": "c", "bucket": "both_pass", "close_miss": "1", "where_f1": "1.0"},
    ]
    selected = select_target_pool(rows)
    assert [r["task_id"] for r in selected] == ["a"], selected
    assert summarize_numeric(selected, "where_f1") == 0.5


def main() -> None:
    tests = [
        test_infers_where_and_value_from_support,
        test_lodo_reports_exact_and_copy_preservation,
        test_value_accuracy_is_scored_on_selected_changed_cells,
        test_invalid_rule_reports_copy_preservation,
        test_pairdelta_intent_summary_sees_sparse_shape_preserved_recolor,
        test_report_helpers_select_pool_and_summarize,
    ]
    failures: list[str] = []
    for test in tests:
        try:
            test()
            print(f"{test.__name__}: PASS")
        except Exception as exc:
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
            print(f"{test.__name__}: FAIL - {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
