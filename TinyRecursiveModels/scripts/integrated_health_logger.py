"""Training-time introspection + pre-registered health dashboard.

This is the component whose absence let the v1/v2 verifier waste full train+eval
cycles before AUROC=0.506 surfaced. Every trainable component reports its health
every N steps; a consolidated dashboard prints OK/RED verdicts against
PRE-REGISTERED thresholds and returns alarms the trainer can act on.

Dependency-light on purpose (no pandas / sklearn): a heavy import that fails
mid-run would defeat the point. AUROC is computed via the Mann-Whitney rank
statistic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------
def auroc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via Mann-Whitney U (no sklearn). labels in {0,1}."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    # (cheap tie handling: group equal scores)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0  # average rank per unique value
    ranks = avg[inv]
    sum_pos = ranks[labels == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Per-component metric aggregators
# ---------------------------------------------------------------------------
def lodo_metrics(buffer: List[dict]) -> Dict[str, float]:
    """buffer entries: {'exact':Tensor[B], 'cell_acc':Tensor[B],
                        'change_overlap':Tensor[B], 'demo_idx':int}."""
    if not buffer:
        return {}
    exact = torch.cat([b["exact"].reshape(-1) for b in buffer]).float()
    cell = torch.cat([b["cell_acc"].reshape(-1) for b in buffer]).float()
    chg = torch.cat([b["change_overlap"].reshape(-1) for b in buffer]).float()
    return {
        "loo_exact_rate": float(exact.mean()),
        "loo_cell_accuracy": float(cell.mean()),
        "loo_change_overlap": float(chg.mean()),
    }


def rule_bank_metrics(rule_bank_module) -> Dict[str, float]:
    out = {
        "active_codes": float(rule_bank_module.active_codes()),
        "codebook_size": float(rule_bank_module.V),
        "code_entropy_ratio": float(rule_bank_module.code_entropy_ratio()),
    }
    lb = getattr(rule_bank_module, "last_batch_code_ids", None)
    if lb is not None and lb.numel() > 0:
        out["batch_codes_used"] = float(lb.unique().numel())
    return out


def verifier_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        probs = torch.sigmoid(logits.detach().float()).cpu().numpy()
        y = labels.detach().cpu().numpy().astype(int)
    if len(y) == 0:
        return {}
    preds = (probs > 0.5).astype(int)
    pos = y == 1
    neg = y == 0
    pos_acc = float((preds[pos] == 1).mean()) if pos.any() else 0.0
    neg_acc = float((preds[neg] == 0).mean()) if neg.any() else 0.0
    gap = (float(probs[pos].mean()) - float(probs[neg].mean())) if (pos.any() and neg.any()) else 0.0
    return {
        "balanced_acc": (pos_acc + neg_acc) / 2.0,
        "auroc": auroc_rank(probs, y),
        "score_gap": gap,
    }


# ---------------------------------------------------------------------------
# Pre-registered pass criteria  (metric -> (comparator, threshold, label))
# comparator: 'ge' = value must be >= threshold ; 'le' = value must be <= threshold
# ---------------------------------------------------------------------------
PASS_CRITERIA: Dict[str, Dict[str, Tuple[str, float, str]]] = {
    "generator": {
        "greedy_exact_frac": ("ge", 0.0, "tracked (baseline 125/400=0.3125)"),
    },
    "LODO": {
        "loo_exact_rate": ("ge", 0.10, ">0.10"),
        "loo_change_overlap": ("ge", 0.50, ">0.50"),
    },
    "RuleBank": {
        "active_codes": ("ge", 32.0, ">=32"),
        "code_entropy_ratio": ("ge", 0.50, ">0.50"),
    },
    "Verifier": {
        "balanced_acc": ("ge", 0.70, ">0.70"),
        "auroc": ("ge", 0.75, ">0.75"),
        "score_gap": ("ge", 0.25, ">0.25"),
    },
}


def _verdict(metric: str, value: float, crit: Dict[str, Tuple[str, float, str]]) -> Tuple[str, bool]:
    if metric not in crit:
        return ("", True)
    comp, thr, _label = crit[metric]
    if value != value:  # NaN
        return ("n/a", True)
    ok = (value >= thr) if comp == "ge" else (value <= thr)
    return ("OK" if ok else "RED", ok)


def print_health_dashboard(
    metrics_by_component: Dict[str, Dict[str, float]],
    step: int,
    min_step_for_alarm: int = 2000,
) -> List[str]:
    """Print the consolidated dashboard. Return list of alarm strings (RED rows).

    Alarms are only raised once `step >= min_step_for_alarm` (early-training noise
    shouldn't trip the halt logic)."""
    lines = [
        f"================ TRAINING HEALTH @ step {step} ================",
        f"{'component':<14}{'metric':<24}{'value':<11}verdict",
        "-" * 61,
    ]
    alarms: List[str] = []
    for comp, metrics in metrics_by_component.items():
        crit = PASS_CRITERIA.get(comp, {})
        for metric, value in metrics.items():
            verdict, ok = _verdict(metric, value, crit)
            label = ""
            if metric in crit:
                label = f"({crit[metric][2]})"
            vstr = f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(f"{comp:<14}{metric:<24}{vstr:<11}{verdict} {label}".rstrip())
            if verdict == "RED" and step >= min_step_for_alarm:
                alarms.append(f"{comp}.{metric}={value:.4f} fails {crit[metric][0]} {crit[metric][1]}")
    lines.append("=" * 61)
    lines.append("ALARMS: " + ("none" if not alarms else "; ".join(alarms)))
    lines.append("=" * 61)
    print("\n".join(lines))
    return alarms


def _self_test() -> None:
    # AUROC sanity: perfectly separable -> 1.0, inverted -> 0.0, random ~0.5
    s = np.array([0.1, 0.2, 0.8, 0.9]); y = np.array([0, 0, 1, 1])
    assert abs(auroc_rank(s, y) - 1.0) < 1e-9, auroc_rank(s, y)
    assert abs(auroc_rank(s, 1 - y) - 0.0) < 1e-9
    assert math.isnan(auroc_rank(s, np.zeros(4, dtype=int)))

    lodo_buf = [
        {"exact": torch.tensor([1.0, 0.0]), "cell_acc": torch.tensor([0.9, 0.8]),
         "change_overlap": torch.tensor([0.6, 0.7]), "demo_idx": 0},
    ]
    lm = lodo_metrics(lodo_buf)
    assert abs(lm["loo_exact_rate"] - 0.5) < 1e-6

    vm = verifier_metrics(torch.tensor([2.0, -2.0, 1.5, -1.0]), torch.tensor([1, 0, 1, 0]))
    assert vm["auroc"] == 1.0, vm

    dash = print_health_dashboard(
        {
            "LODO": {"loo_exact_rate": 0.18, "loo_change_overlap": 0.65},
            "RuleBank": {"active_codes": 12.0, "code_entropy_ratio": 0.71},  # active_codes RED
            "Verifier": {"balanced_acc": 0.78, "auroc": 0.82, "score_gap": 0.41},
        },
        step=5000,
    )
    assert any("RuleBank.active_codes" in a for a in dash), dash
    print("\nhealth_logger self-test PASS")


if __name__ == "__main__":
    _self_test()
