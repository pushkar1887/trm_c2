"""ColorTransitionBank (Component C) — explicit, interpretable, task-SPECIFIC rule tokens.

Motivation (grounded in the D0 gate probe):
    The learned C2 cross-demo module produces a representation that is INVARIANT to which
    task's demos it sees (real_shuffle_delta == 0.0 even with the output gate forced to 3.0,
    while context_ablation_delta exploded to ~16). I.e. C2 encodes "demos are present /
    generic format", not "THIS task's transformation rule". A learned VQ codebook has the
    same risk (gradient descent can collapse it to generic content).

    The ColorTransitionBank avoids that by being a FIXED function: per demo it counts the
    10x10 input->output color-transition histogram over CHANGED cells. This is:
      - interpretable by construction:  hist[3*10+7] = "color 3 became color 7"
      - provably task-specific:         task A's recolor != task B's recolor
      - non-collapsible:                deterministic, no learned pooling to degenerate

    A small learned head then compresses the 100 transition bins into R rule tokens that
    downstream components (recursion conditioning / verifier) can consume. The LEARNED part
    is only the projection 100-bins -> R x D; the task-discriminative content is baked in by
    the histogram and cannot be trained away.

Token convention (matches the rest of the repo): PAD=0, EOS=1, color = token - 2 (0..9).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_TOKEN = 0
EOS_TOKEN = 1
COLOR_OFFSET = 2
N_TRANSITIONS = 100  # 10 input colors x 10 output colors


class ColorTransitionBank(nn.Module):
    """Demos -> R rule tokens via an explicit color-transition histogram.

    forward(context_inputs, context_outputs, context_mask) -> dict:
        rule_tokens   [B, R, D]   compressed rule tokens for downstream conditioning
        hist          [B, 100]    task-level normalized transition histogram (interpretable)
        per_demo_top1 [B, M]      argmax transition id per demo (-1 if empty)
        metrics       dict        entropy / peak / changed_rate / cross_demo_agreement
    """

    def __init__(self, hidden_dim: int = 512, rule_tokens: int = 16,
                 transition_embed_dim: int = 128, n_heads: int = 8):
        super().__init__()
        # NOTE: n_heads is currently unused (this bank pools + query-projects, no multi-head attention);
        # kept in the signature for construction/config compatibility. Do not read self.n_heads.
        self.D = hidden_dim
        self.R = rule_tokens
        self.transition_embed = nn.Embedding(N_TRANSITIONS, transition_embed_dim)
        # pooled feature per transition bin = [embed(weighted by mass) | 4 scalar stats]
        self.pool = nn.Sequential(
            nn.Linear(transition_embed_dim + 4, 256),
            nn.GELU(),
            nn.Linear(256, hidden_dim),
        )
        self.rule_queries = nn.Parameter(torch.randn(rule_tokens, hidden_dim) * 0.02)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    @torch.no_grad()
    def _per_demo_hist(context_inputs, context_outputs, context_mask):
        """[B, M, 100] normalized per-demo transition histogram over changed colored cells."""
        x = context_inputs.long()
        y = context_outputs.long()
        cm = context_mask.to(torch.bool)
        real = (x >= COLOR_OFFSET) & (y >= COLOR_OFFSET)
        changed = real & (x != y) & cm.unsqueeze(-1)            # [B, M, L]
        xc = (x - COLOR_OFFSET).clamp(0, 9)
        yc = (y - COLOR_OFFSET).clamp(0, 9)
        pair = (xc * 10 + yc).clamp(0, 99)                      # [B, M, L]
        # scatter_add into [B,M,100] instead of materialising one_hot [B,M,L,100] (~GBs at L=900).
        Bp, Mp, _ = pair.shape
        hist = torch.zeros(Bp, Mp, N_TRANSITIONS, device=pair.device, dtype=torch.float32)
        hist.scatter_add_(2, pair, changed.float())             # [B, M, 100]
        denom = hist.sum(dim=-1, keepdim=True)
        hist_norm = hist / (denom + 1e-8)
        return hist_norm, denom.squeeze(-1), changed.float().mean(dim=-1)

    @staticmethod
    @torch.no_grad()
    def _per_source_agreement(context_inputs, context_outputs, context_mask):
        """Per-source consensus: for each input colour a, how many demos agree on dst."""
        x = context_inputs.long()
        y = context_outputs.long()
        cm = context_mask.to(torch.bool)
        real = (x >= COLOR_OFFSET) & (y >= COLOR_OFFSET)
        changed = real & (x != y) & cm.unsqueeze(-1)
        xc = (x - COLOR_OFFSET).clamp(0, 9)
        yc = (y - COLOR_OFFSET).clamp(0, 9)
        pair = (xc * 10 + yc).clamp(0, N_TRANSITIONS - 1)
        onehot = F.one_hot(pair, num_classes=N_TRANSITIONS).float()
        counts = (onehot * changed.unsqueeze(-1).float()).sum(dim=2).view(
            x.shape[0], x.shape[1], 10, 10
        )  # [B, M, src, dst]
        src_mass = counts.sum(dim=-1)                    # [B, M, src]
        src_valid = (src_mass > 0) & cm.unsqueeze(-1)     # demo contains changed src colour
        top_dst = counts.argmax(dim=-1)                   # [B, M, src]
        votes = F.one_hot(top_dst, num_classes=10).float() * src_valid.unsqueeze(-1).float()
        vote_sum = votes.sum(dim=1)                       # [B, src, dst]
        valid_count = src_valid.float().sum(dim=1)         # [B, src]
        agreement = vote_sum.max(dim=-1).values / valid_count.clamp_min(1.0)
        agreement = torch.where(valid_count > 0, agreement, torch.zeros_like(agreement))
        demo_count = cm.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        support = valid_count / demo_count
        mode_dst = vote_sum.argmax(dim=-1)
        return agreement, support, mode_dst

    def forward(self, context_inputs: torch.Tensor, context_outputs: torch.Tensor,
                context_mask: torch.Tensor, compute_metrics: bool = True,
                compute_rule_tokens: bool = True) -> Dict[str, torch.Tensor]:
        B, M, L = context_inputs.shape
        per_demo_hist, per_demo_mass, per_demo_changed = self._per_demo_hist(
            context_inputs, context_outputs, context_mask)        # [B,M,100], [B,M], [B,M]
        src_agreement, src_support, src_mode_dst = self._per_source_agreement(
            context_inputs, context_outputs, context_mask)        # [B,10], [B,10], [B,10]

        # Task-level histogram = mass-weighted sum over demos (more changes -> more weight),
        # then renormalized. This is the task's aggregate recolor signature (CHANGED cells only;
        # used for the rule tokens + the dpcc extraction metric).
        task_hist = (per_demo_hist * per_demo_mass.unsqueeze(-1)).sum(dim=1)   # [B,100]
        task_hist = task_hist / (task_hist.sum(dim=-1, keepdim=True) + 1e-8)

        # IDENTITY-AWARE conditional P(out|in=a) over ALL aligned colour cells (changed AND
        # unchanged). This carries the a->a (copy) mass that the changed-only histogram drops,
        # so the direct prior leaves copy cells alone instead of force-recolouring them. Fixed
        # (no grad): it is a deterministic prior; only the strength gate downstream is learned.
        with torch.no_grad():
            xi = context_inputs.long()
            yo = context_outputs.long()
            cmb = context_mask.to(torch.bool).unsqueeze(-1)
            real_all = (xi >= COLOR_OFFSET) & (yo >= COLOR_OFFSET) & cmb        # [B,M,L]
            xc_a = (xi - COLOR_OFFSET).clamp(0, 9)
            yc_a = (yo - COLOR_OFFSET).clamp(0, 9)
            pair_a = (xc_a * 10 + yc_a).clamp(0, N_TRANSITIONS - 1)
            # scatter_add into [B,100] instead of one_hot [B,M,L,100] (~2.3 GB at B=16,M=4 fp32).
            cooc_flat = torch.zeros(B, N_TRANSITIONS, device=context_inputs.device, dtype=torch.float32)
            cooc_flat.scatter_add_(1, pair_a.reshape(B, -1), real_all.reshape(B, -1).float())
            cooc = cooc_flat.view(B, 10, 10)                                    # [B, in, out]
            cond_inout = cooc / cooc.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # P(out|in=a)
            # 0.3: CHANGED-ONLY conditional P(out | in=a, CHANGED) = per-row-normalised changed
            # histogram. Unlike cond_inout it DROPS the a->a copy mass, so on a changed cell it
            # points at the recolour target instead of biasing toward copy. The correct VALUE prior
            # for the colour head (identity-aware cond_inout scored HEAD changed 27 < dpcc 52).
            cond_changed = task_hist.view(B, 10, 10)                            # [B, in, out]
            cond_changed = cond_changed / cond_changed.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # Per-bin features -> R learned rule tokens. Finding A: when compute_rule_tokens=False
        # (set by the model when c2_color_repair_head is ON), these feed only the gated-off old
        # delta branch -> skip the learned pool/attention (dead compute). Code kept for the
        # alternative path (head OFF => old branch consumes them again).
        rule_tokens = None
        if compute_rule_tokens:
            tids = torch.arange(N_TRANSITIONS, device=context_inputs.device)
            emb = self.transition_embed(tids)                         # [100, E]
            weighted = task_hist.unsqueeze(-1) * emb.unsqueeze(0)      # [B,100,E]
            changed_rate = per_demo_changed.mean(dim=1, keepdim=True)             # [B,1]
            peak = task_hist.max(dim=-1, keepdim=True).values                    # [B,1]
            var = task_hist.var(dim=-1, keepdim=True)                            # [B,1]
            nnz = (task_hist > 1e-6).float().sum(dim=-1, keepdim=True) / N_TRANSITIONS  # [B,1]
            stats = torch.cat([changed_rate, peak, var, nnz], dim=-1)            # [B,4]
            stats = stats.unsqueeze(1).expand(B, N_TRANSITIONS, 4)               # [B,100,4]
            feats = torch.cat([weighted, stats], dim=-1)                          # [B,100,E+4]
            rule_base = self.pool(feats)                                          # [B,100,D]
            q = self.rule_queries.unsqueeze(0).expand(B, -1, -1)                  # [B,R,D]
            attn = torch.softmax(
                torch.einsum("brd,bnd->brn", q, rule_base) / math.sqrt(self.D), dim=-1)  # [B,R,100]
            rule_tokens = torch.einsum("brn,bnd->brd", attn, rule_base)          # [B,R,D]
            rule_tokens = self.out_proj(rule_tokens)

        # interpretable diagnostics. The model hot path never reads these (only rule_tokens /
        # hist / cond_inout / src_agreement / src_support), and the cross-demo loop forces
        # B*M GPU->CPU syncs every forward -- so it is skipped unless explicitly requested
        # (probes / self-test pass compute_metrics=True; the model passes False).
        per_demo_top1: torch.Tensor | None = None
        metrics: Dict[str, float] = {}
        if compute_metrics:
            with torch.no_grad():
                ent = -(task_hist * (task_hist + 1e-12).log()).sum(dim=-1).mean()
                per_demo_top1 = torch.where(
                    per_demo_mass > 0, per_demo_hist.argmax(dim=-1),
                    torch.full_like(per_demo_mass, -1, dtype=torch.long))
                # cross-demo agreement: fraction of valid demos sharing the task modal top-1
                agree_vals = []
                cmb = context_mask.to(torch.bool)
                for b in range(B):
                    tops = [int(per_demo_top1[b, mm].item()) for mm in range(M)
                            if bool(cmb[b, mm]) and int(per_demo_top1[b, mm].item()) >= 0]
                    if len(tops) >= 2:
                        vals, counts = torch.unique(torch.tensor(tops), return_counts=True)
                        agree_vals.append(float(counts.max().item()) / float(len(tops)))
                agreement = float(sum(agree_vals) / len(agree_vals)) if agree_vals else float("nan")
            metrics = {
                "transition_entropy": float(ent.item()),
                "transition_peak": float(task_hist.max(dim=-1).values.mean().item()),
                "changed_rate": float(per_demo_changed.mean().item()),
                "cross_demo_top1_agreement": agreement,
            }

        return {
            "rule_tokens": rule_tokens,
            "hist": task_hist,
            "cond_inout": cond_inout,
            "cond_changed": cond_changed,
            "src_agreement": src_agreement,
            "src_support": src_support,
            "src_mode_dst": src_mode_dst,
            "per_demo_top1": per_demo_top1,
            "metrics": metrics,
        }

    @staticmethod
    def readable_top(task_hist_row: torch.Tensor, k: int = 5):
        """Human-readable top-k transitions from one [100] histogram row."""
        if task_hist_row.sum() <= 0:
            return ["(no changed colored cells)"]
        topk = task_hist_row.topk(min(k, N_TRANSITIONS)).indices.tolist()
        out = []
        for pid in topk:
            if task_hist_row[pid] <= 0:
                continue
            out.append(f"color {pid // 10} -> {pid % 10}  (mass {task_hist_row[pid].item():.2f})")
        return out


def _self_test() -> None:
    torch.manual_seed(0)
    B, M, L, D, R = 3, 3, 900, 64, 8
    ci = torch.full((B, M, L), PAD_TOKEN)
    co = torch.full((B, M, L), PAD_TOKEN)
    # task 0: clean 3->7 across all demos
    ci[0, :, :10] = 3 + COLOR_OFFSET
    co[0, :, :10] = 7 + COLOR_OFFSET
    # task 1: 8->1
    ci[1, :, :10] = 8 + COLOR_OFFSET
    co[1, :, :10] = 1 + COLOR_OFFSET
    # task 2: noisy
    ci[2, :, :10] = 4 + COLOR_OFFSET
    co[2, :, :10] = torch.randint(COLOR_OFFSET, COLOR_OFFSET + 10, (M, 10))
    cm = torch.ones(B, M, dtype=torch.bool)

    bank = ColorTransitionBank(hidden_dim=D, rule_tokens=R)
    out = bank(ci, co, cm)
    assert out["rule_tokens"].shape == (B, R, D), out["rule_tokens"].shape
    assert out["hist"].shape == (B, N_TRANSITIONS)
    # task 0 top transition must be 3->7 (id 37)
    assert int(out["hist"][0].argmax().item()) == 37, int(out["hist"][0].argmax().item())
    assert int(out["hist"][1].argmax().item()) == 81, int(out["hist"][1].argmax().item())  # 8->1
    # gradient flows through the learned projection
    loss = out["rule_tokens"].pow(2).mean()
    loss.backward()
    assert bank.rule_queries.grad is not None
    r0 = ColorTransitionBank.readable_top(out["hist"][0])
    assert any("3 -> 7" in s for s in r0), r0
    print(f"color_transition_bank self-test PASS  (task0 top: {r0[0]}; "
          f"task0!=task1 hist: {not torch.equal(out['hist'][0], out['hist'][1])}; "
          f"metrics={out['metrics']})")


if __name__ == "__main__":
    _self_test()
