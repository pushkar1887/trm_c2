"""Offline probe for the minimal relational-colour lane.

This script is intentionally checkpoint-free. It tests whether existing
relational maps/object evidence can select WHERE, and whether a CTBank-style
changed transition can bind WHAT colour, before any C2/trainable integration.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.recursive_reasoning.relational_color_probe import (  # noqa: E402
    evaluate_lodo_task,
    evaluate_target_task,
    pairdelta_intent_summary,
)


DEFAULT_BASELINE = Path(r"C:\Users\PUSHKAR\Downloads\baseline_ledger_correct_pid_v3.csv")
DEFAULT_CURRENT = Path(
    r"reports\v3_fullsystem_step2158_pid401_17col_eval\v3_fullsystem_step2158_pid401_17col_ledger.csv"
)
DEFAULT_DATASET = Path(r"data\arc-agi-evaluation-full400-seed0-pid401aligned\test_puzzles.json")
DEFAULT_OUT = Path(r"reports\relational_color_probe")


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def select_target_pool(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """The preregistered pool: baseline both_fail and close_miss."""
    return [
        row
        for row in rows
        if row.get("bucket") == "both_fail" and int(_to_float(row.get("close_miss"), 0.0)) == 1
    ]


def summarize_numeric(rows: Sequence[dict[str, object]], key: str) -> float:
    vals = [_to_float(row.get(key)) for row in rows if key in row]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _as_grid(grid: Sequence[Sequence[int]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def _task_arrays(task: dict[str, object]) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    train = task.get("train", [])
    test = task.get("test", [])
    if not isinstance(train, list) or not isinstance(test, list) or not test:
        raise ValueError("Task must have train list and at least one test example")
    train_inputs = [_as_grid(ex["input"]) for ex in train]  # type: ignore[index]
    train_outputs = [_as_grid(ex["output"]) for ex in train]  # type: ignore[index]
    test_input = _as_grid(test[0]["input"])  # type: ignore[index]
    test_output = _as_grid(test[0]["output"])  # type: ignore[index]
    return train_inputs, train_outputs, test_input, test_output


def _shuffled_outputs(outputs: Sequence[np.ndarray]) -> list[np.ndarray]:
    if len(outputs) <= 1:
        return [np.zeros_like(outputs[0])] if outputs else []
    return list(outputs[1:]) + [outputs[0]]


def evaluate_pool(
    baseline_rows: Sequence[dict[str, str]],
    current_rows: Sequence[dict[str, str]],
    tasks: dict[str, object],
    *,
    limit: int | None = None,
) -> list[dict[str, object]]:
    current_by_task = {row["task_id"]: row for row in current_rows if row.get("task_id")}
    pool = select_target_pool(baseline_rows)
    if limit is not None:
        pool = pool[:limit]

    result_rows: list[dict[str, object]] = []
    for idx, base_row in enumerate(pool, start=1):
        task_id = base_row["task_id"]
        if task_id not in tasks:
            result_rows.append({"task_id": task_id, "error": "missing_task"})
            continue
        train_inputs, train_outputs, test_input, test_output = _task_arrays(tasks[task_id])  # type: ignore[index]

        lodo_rel = evaluate_lodo_task(task_id, train_inputs, train_outputs, mode="all")
        lodo_color = evaluate_lodo_task(task_id, train_inputs, train_outputs, mode="input_color")
        target_rel = evaluate_target_task(task_id, train_inputs, train_outputs, test_input, test_output, mode="all")
        target_color = evaluate_target_task(task_id, train_inputs, train_outputs, test_input, test_output, mode="input_color")

        intent = pairdelta_intent_summary(train_inputs, train_outputs)
        shuffled_intent = pairdelta_intent_summary(train_inputs, _shuffled_outputs(train_outputs))
        zero_intent = pairdelta_intent_summary([], [])
        current = current_by_task.get(task_id, {})

        result_rows.append(
            {
                "idx": idx,
                "task_id": task_id,
                "baseline_exact": _to_float(base_row.get("exact_accuracy")),
                "baseline_close_miss": _to_float(base_row.get("close_miss")),
                "baseline_content": _to_float(base_row.get("content_accuracy")),
                "baseline_inside_color": _to_float(base_row.get("inside_color_token_accuracy")),
                "current_exact": _to_float(current.get("exact_accuracy")),
                "current_close_miss": _to_float(current.get("close_miss")),
                "current_content": _to_float(current.get("content_accuracy")),
                "current_inside_color": _to_float(current.get("inside_canvas_color_acc")),
                "where_f1": lodo_rel["where_f1"],
                "where_fpr": lodo_rel["where_fpr"],
                "where_f1_input_color": lodo_color["where_f1"],
                "where_fpr_input_color": lodo_color["where_fpr"],
                "lodo_exact": lodo_rel["lodo_exact"],
                "lodo_exact_input_color": lodo_color["lodo_exact"],
                "value_acc": lodo_rel["value_acc"],
                "changed_acc": lodo_rel["changed_acc"],
                "unchanged_acc": lodo_rel["unchanged_acc"],
                "target_exact": target_rel["target_exact"],
                "target_exact_input_color": target_color["target_exact"],
                "target_where_f1": target_rel["target_where_f1"],
                "target_value_acc": target_rel["target_value_acc"],
                "target_changed_acc": target_rel["target_changed_acc"],
                "target_unchanged_acc": target_rel["target_unchanged_acc"],
                "top_predicate": lodo_rel["top_predicate"],
                "target_predicate": target_rel["target_predicate"],
                "pairdelta_conditional": intent["conditional_recolor_score"],
                "pairdelta_global": intent["global_recolor_score"],
                "pairdelta_changed_rate": intent["changed_rate"],
                "pairdelta_shape_preserved": intent["shape_preserved"],
                "pairdelta_correct_minus_shuffle": _to_float(intent["conditional_recolor_score"])
                - _to_float(shuffled_intent["conditional_recolor_score"]),
                "pairdelta_correct_minus_zero": _to_float(intent["conditional_recolor_score"])
                - _to_float(zero_intent["conditional_recolor_score"]),
            }
        )
    return result_rows


def summarize_probe(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    converted = sum(1 for r in rows if _to_float(r.get("target_exact")) >= 1.0)
    converted_input = sum(1 for r in rows if _to_float(r.get("target_exact_input_color")) >= 1.0)
    current_exact = sum(1 for r in rows if _to_float(r.get("current_exact")) >= 1.0)
    rel_better = summarize_numeric(rows, "where_f1") - summarize_numeric(rows, "where_f1_input_color")
    summary: dict[str, object] = {
        "tasks": len(rows),
        "where_f1": summarize_numeric(rows, "where_f1"),
        "where_fpr": summarize_numeric(rows, "where_fpr"),
        "where_f1_input_color": summarize_numeric(rows, "where_f1_input_color"),
        "where_fpr_input_color": summarize_numeric(rows, "where_fpr_input_color"),
        "where_f1_gain_vs_input_color": rel_better,
        "lodo_exact": summarize_numeric(rows, "lodo_exact"),
        "value_acc": summarize_numeric(rows, "value_acc"),
        "changed_acc": summarize_numeric(rows, "changed_acc"),
        "unchanged_acc": summarize_numeric(rows, "unchanged_acc"),
        "target_exact_converted": converted,
        "target_exact_input_color": converted_input,
        "current_exact_in_pool": current_exact,
        "pairdelta_conditional": summarize_numeric(rows, "pairdelta_conditional"),
        "pairdelta_correct_minus_shuffle": summarize_numeric(rows, "pairdelta_correct_minus_shuffle"),
        "pairdelta_correct_minus_zero": summarize_numeric(rows, "pairdelta_correct_minus_zero"),
    }
    summary["gate_where_f1_pass"] = int(_to_float(summary["where_f1"]) >= 0.45)
    summary["gate_where_fpr_pass"] = int(_to_float(summary["where_fpr"]) <= 0.20)
    summary["gate_beats_input_color_pass"] = int(_to_float(summary["where_f1_gain_vs_input_color"]) > 0.0)
    summary["gate_value_pass"] = int(_to_float(summary["value_acc"]) >= 0.70)
    summary["gate_unchanged_pass"] = int(_to_float(summary["unchanged_acc"]) >= 0.98)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ledger", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--current-ledger", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = read_csv_rows(args.baseline_ledger)
    current = read_csv_rows(args.current_ledger) if args.current_ledger.exists() else []
    with args.dataset.open(encoding="utf-8") as f:
        tasks = json.load(f)

    rows = evaluate_pool(baseline, current, tasks, limit=args.limit)
    summary = summarize_probe(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.out_dir / "per_task.csv", rows)
    write_csv_rows(args.out_dir / "summary.csv", [summary])

    print("[relational-color-probe]")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"wrote: {args.out_dir / 'per_task.csv'}")
    print(f"wrote: {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
