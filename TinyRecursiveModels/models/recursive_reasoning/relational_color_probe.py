"""Offline relational-colour probe built from existing V3 evidence.

This module intentionally does not call the TRM forward path. It answers a
narrow question before adding learned C2 changes:

    Can existing relmaps/object facts explain WHERE to recolour, and can a
    CTBank-style changed transition bind WHAT colour to write?

All execution is colour-only on the raw ARC grid. Structure is copied from the
input/floor by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from models.recursive_reasoning.object_bank import relational_maps, REL_MAP_CHANNELS


ARC_SIDE = 30
N_COLORS = 10
_RELMAP_CACHE: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}


@dataclass(frozen=True)
class PredicateScore:
    name: str
    precision: float
    recall: float
    f1: float
    fpr: float
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True)
class WherePredicate:
    name: str
    score: PredicateScore

    def mask(self, grid: np.ndarray) -> np.ndarray:
        return predicate_mask(self.name, grid)


@dataclass(frozen=True)
class RelationalColorRule:
    valid: bool
    where: WherePredicate
    value_map: dict[int, int]
    reason: str = ""
    input_color_baseline_f1: float = 0.0

    def apply(
        self,
        input_grid: np.ndarray,
        *,
        expected_output: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        candidate = np.asarray(input_grid, dtype=np.int64).copy()
        where = np.zeros_like(candidate, dtype=bool)
        if self.valid:
            where = self.where.mask(candidate)
            for src, dst in self.value_map.items():
                candidate[where & (input_grid == src)] = dst

        diag: dict[str, float] = {
            "selected_frac": float(where.mean()) if where.size else 0.0,
            "where_f1": 0.0,
            "where_fpr": 0.0,
            "value_acc": 0.0,
            "changed_acc": 0.0,
            "unchanged_acc": 0.0,
            "exact": 0.0,
        }
        if expected_output is None or expected_output.shape != input_grid.shape:
            return candidate, diag

        changed = input_grid != expected_output
        metrics = binary_metrics(where, changed)
        diag["where_f1"] = metrics.f1
        diag["where_fpr"] = metrics.fpr
        diag["exact"] = float(np.array_equal(candidate, expected_output))

        selected_changed = where & changed
        if selected_changed.any():
            diag["value_acc"] = float((candidate[selected_changed] == expected_output[selected_changed]).mean())
        else:
            diag["value_acc"] = 1.0
        if changed.any():
            diag["changed_acc"] = float((candidate[changed] == expected_output[changed]).mean())
        else:
            diag["changed_acc"] = 1.0
        unchanged = ~changed
        if unchanged.any():
            diag["unchanged_acc"] = float((candidate[unchanged] == expected_output[unchanged]).mean())
        else:
            diag["unchanged_acc"] = 1.0
        return candidate, diag


def _as_grid(grid: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D ARC grid, got shape {arr.shape}")
    if arr.shape[0] > ARC_SIDE or arr.shape[1] > ARC_SIDE:
        raise ValueError(f"Grid exceeds {ARC_SIDE}x{ARC_SIDE}: {arr.shape}")
    return arr


def grid_to_tokens(grid: np.ndarray) -> torch.Tensor:
    """Raw ARC colours [H,W] -> TRM tokens [900] with PAD/EOS conventions."""
    arr = _as_grid(grid)
    h, w = arr.shape
    canvas = np.zeros((ARC_SIDE, ARC_SIDE), dtype=np.int64)
    canvas[:h, :w] = arr + 2
    if h < ARC_SIDE:
        canvas[h, :w] = 1
    if w < ARC_SIDE:
        canvas[:h, w] = 1
    return torch.from_numpy(canvas.reshape(-1))


def relmap_crop(grid: np.ndarray) -> np.ndarray:
    arr = _as_grid(grid)
    key = (arr.shape, arr.tobytes())
    cached = _RELMAP_CACHE.get(key)
    if cached is not None:
        return cached
    tokens = grid_to_tokens(arr).view(1, -1)
    maps = relational_maps(tokens, side=ARC_SIDE)[0].cpu().numpy().reshape(ARC_SIDE, ARC_SIDE, REL_MAP_CHANNELS)
    cropped = maps[: arr.shape[0], : arr.shape[1], :]
    _RELMAP_CACHE[key] = cropped
    return cropped


def _relation_masks(grid: np.ndarray) -> dict[str, np.ndarray]:
    arr = _as_grid(grid)
    rm = relmap_crop(arr)
    comp_size = np.expm1(rm[..., 2])
    # 13-channel layout: distance_to_edge is now 4 DIRECTIONAL clearances (idx 6..9); the old single
    # nearest-edge distance == their per-cell min. solidity/inside_container/nearest_colour: 7/8/9 -> 10/11/12.
    dist_edge = np.minimum(np.minimum(rm[..., 6], rm[..., 7]), np.minimum(rm[..., 8], rm[..., 9]))
    dist_nearest = rm[..., 12]
    masks: dict[str, np.ndarray] = {
        "is_background": rm[..., 1] > 0.5,
        "is_largest": rm[..., 3] > 0.5,
        "is_singleton": rm[..., 4] > 0.5,
        "on_boundary": rm[..., 5] > 0.5,
        "edge_dist_zero": dist_edge <= 1e-6,
        "edge_dist_low": dist_edge <= 0.16,
        "edge_dist_high": dist_edge >= 0.33,
        "solid": rm[..., 10] >= 0.99,
        "not_solid": (rm[..., 10] > 0.0) & (rm[..., 10] < 0.99),
        "inside_container": rm[..., 11] > 0.5,
        "nearest_low": dist_nearest <= 0.10,
        "nearest_high": dist_nearest >= 0.25,
        "comp_small": (comp_size > 0.0) & (comp_size <= 4.0),
        "comp_medium": (comp_size >= 5.0) & (comp_size <= 12.0),
        "comp_large": comp_size >= 13.0,
    }
    return masks


def predicate_names(grids: Sequence[np.ndarray], *, mode: str = "all") -> list[str]:
    names: set[str] = set()
    colours = set(range(N_COLORS))
    for grid in grids:
        colours.update(int(x) for x in np.unique(_as_grid(grid)))
    rel_names = sorted(_relation_masks(_as_grid(grids[0])).keys()) if grids else []

    for colour in sorted(c for c in colours if 0 <= c < N_COLORS):
        names.add(f"input_color={colour}")
        if mode == "all":
            for rel in rel_names:
                names.add(f"input_color={colour}&{rel}")
    if mode == "all":
        names.update(rel_names)
    return sorted(names)


def predicate_mask(name: str, grid: np.ndarray) -> np.ndarray:
    arr = _as_grid(grid)
    if "&" in name:
        left, right = name.split("&", 1)
        return predicate_mask(left, arr) & predicate_mask(right, arr)
    if name.startswith("input_color="):
        colour = int(name.split("=", 1)[1])
        return arr == colour
    masks = _relation_masks(arr)
    if name not in masks:
        raise KeyError(f"Unknown predicate {name!r}")
    return masks[name]


def binary_metrics(pred: np.ndarray, truth: np.ndarray, name: str = "") -> PredicateScore:
    p = np.asarray(pred, dtype=bool)
    t = np.asarray(truth, dtype=bool)
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall / max(precision + recall, 1e-12)) if (precision + recall) else 0.0
    fpr = fp / max(fp + tn, 1)
    return PredicateScore(name=name, precision=precision, recall=recall, f1=f1, fpr=fpr, tp=tp, fp=fp, fn=fn, tn=tn)


def score_predicate(name: str, inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray]) -> PredicateScore:
    tp = fp = fn = tn = 0
    for inp, out in zip(inputs, outputs):
        inp_arr, out_arr = _as_grid(inp), _as_grid(out)
        if inp_arr.shape != out_arr.shape:
            continue
        m = predicate_mask(name, inp_arr)
        changed = inp_arr != out_arr
        s = binary_metrics(m, changed, name=name)
        tp += s.tp
        fp += s.fp
        fn += s.fn
        tn += s.tn
    return binary_metrics_from_counts(name, tp, fp, fn, tn)


def binary_metrics_from_counts(name: str, tp: int, fp: int, fn: int, tn: int) -> PredicateScore:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2.0 * precision * recall / max(precision + recall, 1e-12)) if (precision + recall) else 0.0
    fpr = fp / max(fp + tn, 1)
    return PredicateScore(name=name, precision=precision, recall=recall, f1=f1, fpr=fpr, tp=tp, fp=fp, fn=fn, tn=tn)


def rank_predicates(inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray], *, mode: str = "all") -> list[PredicateScore]:
    names = predicate_names([_as_grid(x) for x in inputs], mode=mode)
    scores = [score_predicate(name, inputs, outputs) for name in names]
    scores = [s for s in scores if (s.tp + s.fp) > 0]
    return sorted(
        scores,
        key=lambda s: (
            s.f1 - 0.25 * s.fpr,
            s.f1,
            -s.fpr,
            int("input_color=" in s.name),
            s.precision,
        ),
        reverse=True,
    )


def _fit_value_map(
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    where_name: str,
) -> dict[int, int]:
    selected_counts = np.zeros((N_COLORS, N_COLORS), dtype=np.int64)
    global_changed_counts = np.zeros((N_COLORS, N_COLORS), dtype=np.int64)
    for inp, out in zip(inputs, outputs):
        inp_arr, out_arr = _as_grid(inp), _as_grid(out)
        if inp_arr.shape != out_arr.shape:
            continue
        selected = predicate_mask(where_name, inp_arr)
        changed = inp_arr != out_arr
        for src, dst in zip(inp_arr[selected & changed].reshape(-1), out_arr[selected & changed].reshape(-1)):
            selected_counts[int(src), int(dst)] += 1
        for src, dst in zip(inp_arr[changed].reshape(-1), out_arr[changed].reshape(-1)):
            global_changed_counts[int(src), int(dst)] += 1

    value_map: dict[int, int] = {}
    for src in range(N_COLORS):
        row = selected_counts[src]
        if row.sum() <= 0:
            row = global_changed_counts[src]
        if row.sum() > 0:
            value_map[src] = int(row.argmax())
        else:
            value_map[src] = src
    return value_map


def infer_rule_from_support(
    support_inputs: Sequence[np.ndarray],
    support_outputs: Sequence[np.ndarray],
    *,
    mode: str = "all",
) -> RelationalColorRule:
    inputs = [_as_grid(x) for x in support_inputs]
    outputs = [_as_grid(y) for y in support_outputs]
    if not inputs or len(inputs) != len(outputs):
        empty = WherePredicate("invalid", binary_metrics_from_counts("invalid", 0, 0, 0, 0))
        return RelationalColorRule(False, empty, {}, "empty or mismatched support")
    if any(x.shape != y.shape for x, y in zip(inputs, outputs)):
        empty = WherePredicate("invalid", binary_metrics_from_counts("invalid", 0, 0, 0, 0))
        return RelationalColorRule(False, empty, {}, "support contains shape-changing pair")

    ranked = rank_predicates(inputs, outputs, mode=mode)
    baseline = rank_predicates(inputs, outputs, mode="input_color")
    baseline_f1 = baseline[0].f1 if baseline else 0.0
    if not ranked:
        empty = WherePredicate("invalid", binary_metrics_from_counts("invalid", 0, 0, 0, 0))
        return RelationalColorRule(False, empty, {}, "no non-empty predicate", baseline_f1)
    best = ranked[0]
    rule = WherePredicate(best.name, best)
    value_map = _fit_value_map(inputs, outputs, best.name)
    return RelationalColorRule(True, rule, value_map, input_color_baseline_f1=baseline_f1)


def evaluate_lodo_task(
    task_id: str,
    train_inputs: Sequence[np.ndarray],
    train_outputs: Sequence[np.ndarray],
    *,
    mode: str = "all",
) -> dict[str, float | str | int]:
    folds = 0
    exact = []
    where_f1 = []
    where_fpr = []
    value_acc = []
    changed_acc = []
    unchanged_acc = []
    selected_frac = []
    pred_names: list[str] = []

    for heldout in range(len(train_inputs)):
        support_idx = [i for i in range(len(train_inputs)) if i != heldout]
        if not support_idx:
            continue
        support_inputs = [train_inputs[i] for i in support_idx]
        support_outputs = [train_outputs[i] for i in support_idx]
        rule = infer_rule_from_support(support_inputs, support_outputs, mode=mode)
        _pred, diag = rule.apply(_as_grid(train_inputs[heldout]), expected_output=_as_grid(train_outputs[heldout]))
        folds += 1
        exact.append(diag["exact"])
        where_f1.append(diag["where_f1"])
        where_fpr.append(diag["where_fpr"])
        value_acc.append(diag["value_acc"])
        changed_acc.append(diag["changed_acc"])
        unchanged_acc.append(diag["unchanged_acc"])
        selected_frac.append(diag["selected_frac"])
        pred_names.append(rule.where.name if rule.valid else "invalid")

    return {
        "task_id": task_id,
        "lodo_folds": folds,
        "lodo_exact": _mean(exact),
        "where_f1": _mean(where_f1),
        "where_fpr": _mean(where_fpr),
        "value_acc": _mean(value_acc),
        "changed_acc": _mean(changed_acc),
        "unchanged_acc": _mean(unchanged_acc),
        "selected_frac": _mean(selected_frac),
        "top_predicate": _mode_string(pred_names),
    }


def evaluate_target_task(
    task_id: str,
    train_inputs: Sequence[np.ndarray],
    train_outputs: Sequence[np.ndarray],
    test_input: np.ndarray,
    test_output: np.ndarray,
    *,
    mode: str = "all",
) -> dict[str, float | str | int]:
    rule = infer_rule_from_support(train_inputs, train_outputs, mode=mode)
    _pred, diag = rule.apply(_as_grid(test_input), expected_output=_as_grid(test_output))
    return {
        "task_id": task_id,
        "target_exact": diag["exact"],
        "target_where_f1": diag["where_f1"],
        "target_where_fpr": diag["where_fpr"],
        "target_value_acc": diag["value_acc"],
        "target_changed_acc": diag["changed_acc"],
        "target_unchanged_acc": diag["unchanged_acc"],
        "target_selected_frac": diag["selected_frac"],
        "target_predicate": rule.where.name if rule.valid else "invalid",
        "target_rule_valid": int(rule.valid),
        "input_color_baseline_f1": rule.input_color_baseline_f1,
    }


def pairdelta_intent_summary(
    train_inputs: Sequence[np.ndarray],
    train_outputs: Sequence[np.ndarray],
) -> dict[str, float | int]:
    inputs = [_as_grid(x) for x in train_inputs]
    outputs = [_as_grid(y) for y in train_outputs]
    same_shape = [float(x.shape == y.shape) for x, y in zip(inputs, outputs)]
    changed_rates = []
    counts = np.zeros((N_COLORS, N_COLORS), dtype=np.int64)
    total_valid = 0
    total_changed = 0
    for inp, out in zip(inputs, outputs):
        if inp.shape != out.shape:
            continue
        changed = inp != out
        total_valid += int(inp.size)
        total_changed += int(changed.sum())
        changed_rates.append(float(changed.mean()))
        for src, dst in zip(inp[changed].reshape(-1), out[changed].reshape(-1)):
            counts[int(src), int(dst)] += 1
    dominant_src = int(counts.sum(axis=1).argmax()) if counts.sum() > 0 else -1
    dominant_dst = int(counts[dominant_src].argmax()) if dominant_src >= 0 else -1
    changed_rate = _mean(changed_rates)
    shape_preserved = _mean(same_shape)
    non_empty = float(total_changed > 0)
    sparse = float(0.0 < changed_rate < 0.5)
    row_peaks = counts.max(axis=1).sum()
    global_consistency = float(row_peaks / max(counts.sum(), 1))
    conditional_score = shape_preserved * non_empty * sparse * (1.0 - min(changed_rate, 1.0))
    global_score = shape_preserved * non_empty * global_consistency
    return {
        "shape_preserved": shape_preserved,
        "changed_rate": changed_rate,
        "dominant_source_color": dominant_src,
        "dominant_target_color": dominant_dst,
        "conditional_recolor_score": conditional_score,
        "global_recolor_score": global_score,
        "changed_cells": total_changed,
        "valid_cells": total_valid,
    }


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _mode_string(values: Sequence[str]) -> str:
    if not values:
        return ""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
