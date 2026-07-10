"""Self-tests for the verifier -> refiner -> memory-writer loop.

Run:
  trm\Scripts\python.exe scripts\test_lodo_refiner.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lodo_refiner
from lodo_refiner import refine_lodo_recipes
from rule_library import RuleLibrary
from solve import _CropOp, _DihedralOp, _ScaleOp, _TileOp, _TranslateOp, _execute_recipe_solve
from parse import COLOR_OFFSET, _parse_same
import verify_and_select_candidates


def _scale_recolor_task():
    gin, gout = {}, {}
    for i, pat in enumerate([[[3, 4], [4, 3]], [[3, 3], [4, 4]], [[4, 3], [3, 4]]]):
        g = torch.tensor(pat, dtype=torch.long) + COLOR_OFFSET
        up = g.repeat_interleave(2, 0).repeat_interleave(2, 1)
        o = up.clone()
        o[up == 3 + COLOR_OFFSET] = 7 + COLOR_OFFSET
        o[up == 4 + COLOR_OFFSET] = 8 + COLOR_OFFSET
        gin[i], gout[i] = g, o
    return gin, gout


def _seq(g: torch.Tensor, side: int) -> torch.Tensor:
    out = torch.zeros(side, side, dtype=torch.long)
    h, w = g.shape
    out[:h, :w] = g
    return out.reshape(-1)


def _set_cover_task():
    """Two-rule task: border-touching objects -> 7 and frame-contained objects -> 8."""
    oc = 4 + COLOR_OFFSET

    def frame(g, r0, r1, c0, c1):
        g[r0, c0:c1 + 1] = 3 + COLOR_OFFSET
        g[r1, c0:c1 + 1] = 3 + COLOR_OFFSET
        g[r0:r1 + 1, c0] = 3 + COLOR_OFFSET
        g[r0:r1 + 1, c1] = 3 + COLOR_OFFSET

    def mk(a, f, b, n):
        g = torch.full((14, 14), COLOR_OFFSET, dtype=torch.long)
        frame(g, *f)
        for r, c in (a, b, n):
            g[r:r + 2, c:c + 2] = oc
        o = g.clone()
        o[a[0]:a[0] + 2, a[1]:a[1] + 2] = 7 + COLOR_OFFSET
        o[b[0]:b[0] + 2, b[1]:b[1] + 2] = 8 + COLOR_OFFSET
        return g, o

    specs = [
        ((0, 0), (4, 9, 4, 9), (6, 6), (11, 11)),
        ((12, 0), (3, 8, 6, 11), (5, 8), (10, 2)),
        ((0, 12), (5, 10, 2, 7), (7, 4), (11, 10)),
    ]
    gin, gout = {}, {}
    for i, spec in enumerate(specs):
        gin[i], gout[i] = mk(*spec)
    return gin, gout


def _largest_colour_copy_task():
    """Relation task: every smaller object copies the largest object's colour."""

    def mk(big_colour, big_box, smalls):
        g = torch.full((12, 12), COLOR_OFFSET, dtype=torch.long)
        g[big_box[0]:big_box[1] + 1, big_box[2]:big_box[3] + 1] = big_colour + COLOR_OFFSET
        for box, col in smalls:
            g[box[0]:box[1] + 1, box[2]:box[3] + 1] = col + COLOR_OFFSET
        o = g.clone()
        for box, _ in smalls:
            o[box[0]:box[1] + 1, box[2]:box[3] + 1] = big_colour + COLOR_OFFSET
        return g, o

    specs = [
        (4, (1, 4, 1, 4), [((1, 2, 7, 8), 5), ((7, 8, 2, 3), 6)]),
        (8, (6, 10, 6, 10), [((1, 2, 1, 2), 5), ((1, 2, 9, 10), 3)]),
        (3, (2, 6, 2, 6), [((9, 10, 9, 10), 5), ((9, 10, 1, 2), 7)]),
    ]
    gin, gout = {}, {}
    for i, (bc, bb, sm) in enumerate(specs):
        gin[i], gout[i] = mk(bc, bb, sm)
    return gin, gout


def test_refiner_stores_only_lodo_exact_stable_recipe() -> None:
    gin, gout = _scale_recolor_task()
    lib = RuleLibrary()
    geo_ops = [_DihedralOp(), _TranslateOp(), _TileOp(), _CropOp(), _ScaleOp()]

    report = refine_lodo_recipes(gin, gout, list(gin), geo_ops, rule_lib=lib, family="synthetic")

    assert report["stored"] == [["scale", "recolor"]]
    assert ["scale", "recolor"] in lib
    assert report["folds"] and all(f["refined_exact"] for f in report["folds"])
    assert all(f["recipe"] == ["scale", "recolor"] for f in report["folds"])


def test_refiner_does_not_store_unstable_recipe() -> None:
    gin, gout = _scale_recolor_task()
    # Break one fold so the same recipe cannot be LODO-exact across the task.
    gout[2] = gout[2].clone()
    gout[2][0, 0] = 9 + COLOR_OFFSET

    lib = RuleLibrary()
    geo_ops = [_DihedralOp(), _TranslateOp(), _TileOp(), _CropOp(), _ScaleOp()]
    report = refine_lodo_recipes(gin, gout, list(gin), geo_ops, rule_lib=lib, family="synthetic")

    assert report["stored"] == []
    assert len(lib) == 0
    assert any(not f["refined_exact"] for f in report["folds"])


def test_offline_evaluate_task_can_refine_and_store() -> None:
    side = 8
    gin, gout = _scale_recolor_task()
    demos = [(_seq(gin[i], side), _seq(gout[i], side)) for i in sorted(gin)]
    lib = RuleLibrary()

    res = verify_and_select_candidates.evaluate_task(
        demos,
        side,
        task_id="synthetic-scale",
        rule_lib=lib,
        family="synthetic",
        refine_close_miss=True,
    )

    assert res["refinement"]["stored"] == [["scale", "recolor"]]
    assert ["scale", "recolor"] in lib
    assert res["scores"]["refined_recipe"][0] == 1.0


def test_set_cover_refiner_stores_and_committed_reuses_macro() -> None:
    gin, gout = _set_cover_task()
    lib = RuleLibrary()
    geo_ops = [_DihedralOp(), _TranslateOp(), _TileOp(), _CropOp(), _ScaleOp()]
    valid = [0, 2]

    report = refine_lodo_recipes(gin, gout, valid, geo_ops, rule_lib=lib, family="synthetic")

    assert report["stored"] == [["set_cover", "clause_union"]]
    assert ["set_cover", "clause_union"] in lib
    assert _execute_recipe_solve(gin, gout, [0], 2, ["set_cover", "clause_union"], geo_ops) is True


def test_failure_diagnostics_name_where_copy_failed() -> None:
    inp = torch.tensor([[2, 3], [3, 2]], dtype=torch.long) + COLOR_OFFSET
    target = torch.tensor([[7, 3], [3, 8]], dtype=torch.long) + COLOR_OFFSET
    pred = torch.tensor([[2, 9], [3, 8]], dtype=torch.long) + COLOR_OFFSET

    diag = lodo_refiner.diagnose_failure(pred, target, inp)

    assert diag["labels"]["missed_change"] == 1
    assert diag["labels"]["false_edit"] == 1
    assert diag["labels"]["wrong_value"] == 0
    assert diag["exact"] is False


def test_relation_refiner_stores_and_reuses_largest_colour_copy() -> None:
    gin, gout = _largest_colour_copy_task()
    lib = RuleLibrary()
    geo_ops = [_DihedralOp(), _TranslateOp(), _TileOp(), _CropOp(), _ScaleOp()]

    report = refine_lodo_recipes(gin, gout, list(gin), geo_ops, rule_lib=lib, family="synthetic")

    assert report["stored"] == [["object_relrecolor", "largest"]]
    assert ["object_relrecolor", "largest"] in lib
    assert all(f["recipe"] == ["object_relrecolor", "largest"] for f in report["folds"])
    assert _execute_recipe_solve(gin, gout, [0, 1], 2, ["object_relrecolor", "largest"], geo_ops) is True


def test_refiner_abstains_on_single_demo_task() -> None:
    g = torch.tensor([[2, 3], [3, 2]], dtype=torch.long) + COLOR_OFFSET
    gin = {0: g}
    gout = {0: g.clone()}
    lib = RuleLibrary()
    geo_ops = [_DihedralOp(), _TranslateOp(), _TileOp(), _CropOp(), _ScaleOp()]

    report = refine_lodo_recipes(gin, gout, [0], geo_ops, rule_lib=lib, family="synthetic")

    assert report["stored"] == []
    assert report["stable_recipe"] is None
    assert report["exact_folds"] == 0
    assert len(lib) == 0


if __name__ == "__main__":
    test_refiner_stores_only_lodo_exact_stable_recipe()
    test_refiner_does_not_store_unstable_recipe()
    test_offline_evaluate_task_can_refine_and_store()
    test_set_cover_refiner_stores_and_committed_reuses_macro()
    test_failure_diagnostics_name_where_copy_failed()
    test_relation_refiner_stores_and_reuses_largest_colour_copy()
    test_refiner_abstains_on_single_demo_task()
    print("test_lodo_refiner PASS")
