"""Regression checks for V3 floor/candidate split.

Plain script style matches the local test harness: pytest is not assumed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_COMPILE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models.losses_fvr import _health_metrics


def _logits_from_pred(pred: torch.Tensor, vocab: int = 12) -> torch.Tensor:
    logits = torch.full((*pred.shape, vocab), -8.0)
    logits.scatter_(-1, pred.long().unsqueeze(-1), 8.0)
    return logits


def test_selector_never_scores_below_floor_and_prefers_better_candidate() -> None:
    labels = torch.tensor(
        [
            [2, 3, 1, -100],
            [4, 5, 1, -100],
        ],
        dtype=torch.long,
    )
    inputs = torch.tensor(
        [
            [2, 9, 1, 0],
            [4, 8, 1, 0],
        ],
        dtype=torch.long,
    )

    floor_pred = torch.tensor(
        [
            [2, 9, 1, 0],  # wrong changed colour at cell 1
            [4, 8, 1, 0],  # wrong changed colour at cell 1
        ],
        dtype=torch.long,
    )
    cand_pred = torch.tensor(
        [
            [2, 3, 1, 0],  # candidate solves row 0
            [4, 8, 1, 0],  # candidate ties floor on row 1
        ],
        dtype=torch.long,
    )

    metrics = _health_metrics(
        labels=labels,
        main_preds=floor_pred,
        main_inputs=inputs,
        aux_logits=_logits_from_pred(cand_pred),
        aux_floor_logits=_logits_from_pred(floor_pred),
        aux_labels=labels,
        aux_inputs=inputs,
    )

    assert float(metrics["main_strict_exact_pct"]) == 0.0, "MAIN must describe the floor path"
    assert float(metrics["lodo_color_exact_pct"]) == 50.0, "LODO must describe the candidate path"
    assert float(metrics["lodo_select_color_exact_pct"]) == 50.0
    assert float(metrics["lodo_floor_color_exact_pct"]) == 0.0
    assert float(metrics["lodo_select_ge_floor_pct"]) == 100.0, "selector must be floor-safe"
    assert float(metrics["lodo_candidate_chosen_pct"]) == 50.0, "candidate should fire only on the better row"


def main() -> None:
    tests = [test_selector_never_scores_below_floor_and_prefers_better_candidate]
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
