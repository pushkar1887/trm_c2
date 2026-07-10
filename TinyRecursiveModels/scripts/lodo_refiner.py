"""LODO verifier -> bounded refiner -> rule-memory writer.

This is deliberately conservative. It does not commit close-miss candidates.
It uses close-miss/failure evidence only to run a bounded refinement search,
then stores a recipe only when every support fold verifies the same recipe
exactly.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import torch

from rule_library import RuleLibrary
from parse import _parse_same
from solve import _compose2_solve, _object_relrecolor_predict, _set_cover_predict


RELATION_RECIPES = ("nearest", "container", "contained", "aligned", "between", "largest", "smallest")


def _grid_exact(pred: torch.Tensor | None, target: torch.Tensor | None) -> bool:
    return pred is not None and target is not None and pred.shape == target.shape and bool((pred == target).all())


def _shape_string(g: torch.Tensor | None) -> str:
    if g is None:
        return "None"
    return "x".join(str(int(v)) for v in g.shape)


def _flat_similarity(pred: torch.Tensor | None, target: torch.Tensor | None) -> float:
    if pred is None or target is None or pred.shape != target.shape:
        return 0.0
    return float((pred == target).float().mean())


def _floor_copy(gin_h: torch.Tensor | None) -> torch.Tensor | None:
    return None if gin_h is None else gin_h.clone()


def diagnose_failure(
    pred: torch.Tensor | None,
    target: torch.Tensor | None,
    inp: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Name the cell-level failure mode of a candidate.

    The verifier still stores nothing from a near miss. These labels are only
    search guidance: missed changed cells, false edits on copy cells, and wrong
    changed-cell values point the bounded refiner at the next family to try.
    """
    labels: Counter[str] = Counter()
    if pred is None or target is None:
        labels["missing_grid"] = 1
        return {"exact": False, "shape_match": False, "labels": dict(labels)}
    if pred.shape != target.shape:
        labels["shape_error"] = 1
        return {"exact": False, "shape_match": False, "labels": dict(labels)}

    exact = bool((pred == target).all())
    if inp is None or inp.shape != target.shape:
        labels["value_mismatch"] = int((pred != target).sum().item())
        return {"exact": exact, "shape_match": True, "labels": dict(labels)}

    should_change = inp != target
    did_change = pred != inp
    labels["missed_change"] = int((should_change & ~did_change).sum().item())
    labels["false_edit"] = int((~should_change & did_change).sum().item())
    labels["wrong_value"] = int((should_change & did_change & (pred != target)).sum().item())
    labels["copy_destruction"] = int((~should_change & (pred != target)).sum().item())
    return {"exact": exact, "shape_match": True, "labels": dict(labels)}


def _recipe_key(recipe: list[str] | None) -> tuple[str, ...] | None:
    return tuple(recipe) if recipe is not None else None


def _stable_exact_recipe(folds: list[dict[str, Any]]) -> list[str] | None:
    exact_recipes = [
        _recipe_key(f["recipe"])
        for f in folds
        if bool(f.get("refined_exact")) and f.get("recipe") is not None
    ]
    if len(exact_recipes) != len(folds) or not exact_recipes:
        return None
    counts = Counter(exact_recipes)
    recipe, n = counts.most_common(1)[0]
    if n != len(folds):
        return None
    return list(recipe)


def _try_refinement(
    gin: dict[int, torch.Tensor | None],
    gout: dict[int, torch.Tensor | None],
    sup: list[int],
    h: int,
    geo_ops,
    labels: dict[int, torch.Tensor],
) -> tuple[list[str] | None, str | None]:
    compose_recipe = _compose2_solve(gin, gout, sup, h, geo_ops)
    if compose_recipe is not None:
        return compose_recipe, "compose2"

    pred = _set_cover_predict(gin, gout, sup, h, labels)
    target = gout.get(h)
    if _grid_exact(pred, target):
        return ["set_cover", "clause_union"], "set_cover"

    for relation in RELATION_RECIPES:
        pred = _object_relrecolor_predict(gin, gout, sup, h, labels, relation)
        if _grid_exact(pred, target):
            return ["object_relrecolor", relation], f"object_relrecolor:{relation}"

    return None, None


def refine_lodo_recipes(
    gin: dict[int, torch.Tensor | None],
    gout: dict[int, torch.Tensor | None],
    valid: list[int],
    geo_ops,
    rule_lib: RuleLibrary | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    """Run exact LODO refinement for already-approved [SHAPE_OP, recolor] recipes.

    For each held-out demo h:
      - support = all other demos
      - inspect the floor/copy failure class
      - run compose2 refinement on support
      - require the refined prediction to exactly reconstruct h

    A recipe is written to RuleLibrary only if every fold verifies the same
    recipe exactly. Parameters are never stored; only op names are stored.
    """
    folds: list[dict[str, Any]] = []
    labels = {m: _parse_same(gin[m]) for m in valid if gin.get(m) is not None}
    for h in valid:
        sup = [m for m in valid if m != h]
        target = gout.get(h)
        floor = _floor_copy(gin.get(h))
        floor_exact = _grid_exact(floor, target)
        floor_sim = _flat_similarity(floor, target)
        reason = "floor_exact"
        if not floor_exact:
            if floor is None or target is None:
                reason = "missing_grid"
            elif floor.shape != target.shape:
                reason = "shape_mismatch"
            else:
                reason = "value_mismatch"
        failure_diag = diagnose_failure(floor, target, gin.get(h))

        if not sup:
            recipe, refine_source = None, "no_support"
        else:
            recipe, refine_source = _try_refinement(gin, gout, sup, h, geo_ops, labels)
        refined_exact = recipe is not None
        folds.append(
            {
                "holdout": int(h),
                "support": [int(m) for m in sup],
                "floor_exact": bool(floor_exact),
                "floor_sim": float(floor_sim),
                "failure": reason,
                "failure_labels": failure_diag["labels"],
                "input_shape": _shape_string(gin.get(h)),
                "target_shape": _shape_string(target),
                "recipe": recipe,
                "refine_source": refine_source,
                "refined_exact": bool(refined_exact),
            }
        )

    stable = _stable_exact_recipe(folds)
    stored: list[list[str]] = []
    if stable is not None and rule_lib is not None:
        before = len(rule_lib)
        rule_lib.add(stable, family=family)
        # Report the recipe as stored even if it was already present; the point
        # is that this task supplied exact proof for this recipe.
        if len(rule_lib) >= before:
            stored.append(stable)

    return {
        "folds": folds,
        "stable_recipe": stable,
        "stored": stored,
        "exact_folds": sum(1 for f in folds if f["refined_exact"]),
        "n_folds": len(folds),
    }
