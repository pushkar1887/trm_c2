"""C2 cross-demo introspection — read-only probes on a FROZEN model.

Answers the user's literal question: "in cross-demo, what rule are they extracting
and how well are they doing that" — with hard numbers, no training.

Three probe families, all leak-safe (demo pairs only; test pair never read):

  1. real-vs-shuffle  — run LODO with (a) the task's REAL demos and (b) demos shuffled
     ACROSS tasks in the batch. If shuffle_loss - real_loss <= 0, the model ignores
     demo content and the whole C2 pathway is decoration (the master falsifier).

  2. context-ablation — run LODO with ALL demos masked (zero context). ablation_loss
     - real_loss > 0 means real demos genuinely help.

  3. color-transition — per task, build the 10x10 input->output color-transition
     histogram over CHANGED cells, per demo; measure cross-demo top-1 agreement
     (do the demos agree on one dominant recolor rule?) and entropy (is it sharp?).

These reuse the existing LODOTrainingHead (with blank_pid=True) for (1)+(2) so the
reconstruction path is identical to training. (3) is pure tensor stats on the demos.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

PAD_TOKEN = 0
EOS_TOKEN = 1
COLOR_OFFSET = 2          # token = color + 2 ; color in 0..9
IGNORE_LABEL_ID = 0


# ---------------------------------------------------------------------------
# (1)+(2) reconstruction probes via the LODO head
# ---------------------------------------------------------------------------
@torch.no_grad()
def lodo_reconstruction_metrics(lodo_head, batch: Dict[str, torch.Tensor],
                                generator: Optional[torch.Generator] = None) -> Dict[str, float]:
    """Run the (blank-pid) LODO head once and pull aggregated metrics from its buffer.

    Returns mean loss + the per-holdout reconstruction metrics the head records."""
    lodo_head.metrics_buffer.clear()
    loss = lodo_head(batch, grad=False, generator=generator)
    buf = lodo_head.metrics_buffer
    if not buf:
        return {"loss": float(loss.item()), "exact": float("nan"),
                "cell_acc": float("nan"), "changed_cell_acc": float("nan")}
    exact = torch.cat([b["exact"].reshape(-1) for b in buf]).float().mean().item()
    cell = torch.cat([b["cell_acc"].reshape(-1) for b in buf]).float().mean().item()
    chg = torch.cat([b["change_overlap"].reshape(-1) for b in buf]).float().mean().item()
    return {"loss": float(loss.item()), "exact": exact,
            "cell_acc": cell, "changed_cell_acc": chg}


def _shuffle_context_across_batch(batch: Dict[str, torch.Tensor],
                                  generator: Optional[torch.Generator]) -> Dict[str, torch.Tensor]:
    """Return a batch whose context demos are taken from OTHER tasks in the batch.

    A derangement of the batch index is applied to context_inputs/outputs/mask, so each
    row keeps its own held-out target but receives a different task's demos."""
    B = batch["context_inputs"].shape[0]
    if B < 2:
        return batch
    # derangement: roll by a random offset 1..B-1 (guarantees no row keeps its own demos)
    # RNG draw must live on the generator's device (callers may pass a CUDA generator).
    gdev = generator.device if generator is not None else "cpu"
    offset = 1 + int(torch.randint(0, B - 1, (1,), generator=generator, device=gdev).item())
    perm = (torch.arange(B, device=batch["context_inputs"].device) + offset) % B
    out = dict(batch)
    out["context_inputs"] = batch["context_inputs"][perm].clone()
    out["context_outputs"] = batch["context_outputs"][perm].clone()
    out["context_mask"] = batch["context_mask"][perm].clone()
    if "context_input_visual_features" in batch and torch.is_tensor(batch["context_input_visual_features"]):
        out["context_input_visual_features"] = batch["context_input_visual_features"][perm].clone()
    return out


def _ablate_context(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return a batch with ALL demos masked out (zero context)."""
    out = dict(batch)
    out["context_mask"] = torch.zeros_like(batch["context_mask"])
    return out


@torch.no_grad()
def real_vs_shuffle_vs_ablation(lodo_head, batch, generator=None, seed: int = 0) -> Dict[str, float]:
    # CRITICAL: all three conditions must hold out the SAME demo index, else the
    # losses aren't comparable (different targets). The LODO head picks holdouts
    # via its `generator` arg, so we reset to the SAME seed before each call.
    # (Shuffling demos across tasks changes context_inputs/outputs but the holdout
    # target is selected from each row's own context, so identical seeds => identical
    # holdout choice => a clean real-vs-shuffle-vs-ablation comparison.)
    def fresh():
        return torch.Generator().manual_seed(int(seed))
    real = lodo_reconstruction_metrics(lodo_head, batch, fresh())
    shuf = lodo_reconstruction_metrics(lodo_head, _shuffle_context_across_batch(batch, fresh()), fresh())
    abl = lodo_reconstruction_metrics(lodo_head, _ablate_context(batch), fresh())
    return {
        "real_loss": real["loss"],
        "shuffle_loss": shuf["loss"],
        "ablation_loss": abl["loss"],
        "real_shuffle_delta": shuf["loss"] - real["loss"],
        "context_ablation_delta": abl["loss"] - real["loss"],
        "loo_exact_rate": real["exact"],
        "loo_cell_acc": real["cell_acc"],
        "loo_changed_cell_acc": real["changed_cell_acc"],
    }


# ---------------------------------------------------------------------------
# (3) color-transition extraction — what recolor rule do the demos imply?
# ---------------------------------------------------------------------------
@torch.no_grad()
def transition_histograms(context_inputs: torch.Tensor, context_outputs: torch.Tensor,
                          context_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Per (task, demo) 10x10 color-transition histogram over CHANGED cells.

    tokens: PAD=0, EOS=1, color = token-2. We only count cells where both in and out
    are real colors (token>=2) and the color changed. Returns:
        hist        [B, N, 100]  normalized per demo
        top1        [B, N]       argmax transition id per demo (in 0..99), -1 if empty
        changed_rate[B, N]
    """
    x = context_inputs.long()
    y = context_outputs.long()
    cm = context_mask.to(torch.bool)                      # [B, N]
    real = (x >= COLOR_OFFSET) & (y >= COLOR_OFFSET)      # both ends are colors
    changed = real & (x != y) & cm.unsqueeze(-1)          # [B, N, L]
    xc = (x - COLOR_OFFSET).clamp(0, 9)
    yc = (y - COLOR_OFFSET).clamp(0, 9)
    pair = (xc * 10 + yc).clamp(0, 99)                    # [B, N, L]
    B, N, L = pair.shape
    onehot = F.one_hot(pair, num_classes=100).float()     # [B, N, L, 100]
    hist = (onehot * changed.unsqueeze(-1).float()).sum(dim=2)   # [B, N, 100]
    denom = hist.sum(dim=-1, keepdim=True)
    hist_norm = hist / (denom + 1e-8)
    top1 = torch.where(denom.squeeze(-1) > 0, hist.argmax(dim=-1),
                       torch.full((B, N), -1, dtype=torch.long, device=hist.device))
    changed_rate = changed.float().mean(dim=-1)           # [B, N]
    return {"hist": hist_norm, "top1": top1, "changed_rate": changed_rate}


@torch.no_grad()
def cross_demo_agreement(context_inputs, context_outputs, context_mask) -> Dict[str, float]:
    """How well do a task's demos agree on ONE dominant color transition?

    Returns batch-mean:
        top1_agreement  — fraction of valid demos sharing the task's modal top-1 transition
        transition_entropy — mean entropy of per-demo histograms (low = sharp rule)
        n_recolor_tasks — how many tasks had any changed colored cells (the recolor subset)
    """
    th = transition_histograms(context_inputs, context_outputs, context_mask)
    top1 = th["top1"]            # [B, N]
    hist = th["hist"]            # [B, N, 100]
    cm = context_mask.to(torch.bool)
    B, N = top1.shape
    agreements = []
    entropies = []
    n_recolor = 0
    for b in range(B):
        valid = [int(top1[b, n].item()) for n in range(N)
                 if bool(cm[b, n]) and int(top1[b, n].item()) >= 0]
        if len(valid) < 2:
            continue
        n_recolor += 1
        # modal transition among demos, fraction that match it
        vals, counts = np.unique(np.array(valid), return_counts=True)
        agreements.append(float(counts.max()) / float(len(valid)))
        # mean entropy of the valid demos' histograms
        for n in range(N):
            if bool(cm[b, n]) and int(top1[b, n].item()) >= 0:
                p = hist[b, n]
                ent = float(-(p * (p + 1e-12).log()).sum().item())
                entropies.append(ent)
    return {
        "cross_demo_top1_agreement": float(np.mean(agreements)) if agreements else float("nan"),
        "transition_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "n_recolor_tasks": float(n_recolor),
    }


@torch.no_grad()
def top_transitions_readable(context_inputs, context_outputs, context_mask, row: int = 0,
                             k: int = 5) -> List[str]:
    """Human-readable top-k color transitions for one task (for the rule dump)."""
    th = transition_histograms(context_inputs[row:row+1], context_outputs[row:row+1],
                               context_mask[row:row+1])
    hist = th["hist"][0]                     # [N, 100]
    agg = hist.sum(dim=0)                    # [100]
    if agg.sum() <= 0:
        return ["(no changed colored cells)"]
    topk = agg.topk(min(k, 100)).indices.tolist()
    out = []
    for pid in topk:
        if agg[pid] <= 0:
            continue
        out.append(f"color {pid // 10} -> {pid % 10}  (mass {agg[pid].item():.2f})")
    return out


def _self_test() -> None:
    torch.manual_seed(0)
    B, N, L = 3, 3, 900
    # craft demos with a clear 3->7 recolor in task 0
    ci = torch.full((B, N, L), PAD_TOKEN)
    co = torch.full((B, N, L), PAD_TOKEN)
    ci[0, :, :10] = 3 + COLOR_OFFSET
    co[0, :, :10] = 7 + COLOR_OFFSET          # consistent 3->7 across all demos
    ci[1, :, :10] = 4 + COLOR_OFFSET
    co[1, :, :10] = torch.randint(COLOR_OFFSET, COLOR_OFFSET + 10, (N, 10))  # noisy
    cm = torch.ones(B, N, dtype=torch.bool)

    agree = cross_demo_agreement(ci, co, cm)
    assert agree["cross_demo_top1_agreement"] == agree["cross_demo_top1_agreement"]  # not nan
    rules = top_transitions_readable(ci, co, cm, row=0)
    assert any("3 -> 7" in r for r in rules), rules
    # task 0 demos perfectly agree -> agreement should be high when averaged with task1
    print(f"c2_introspection self-test PASS  "
          f"(task0 top rule: {rules[0]}; batch top1_agreement={agree['cross_demo_top1_agreement']:.2f}, "
          f"n_recolor={agree['n_recolor_tasks']:.0f})")


if __name__ == "__main__":
    _self_test()
