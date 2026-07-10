"""PairDeltaEncoder + RuleConditionedDecoder (Phase A).

Design (embodies the measured diagnosis):
  - C2's learned pooling of RAW demo images was shuffle-invariant (encodes format, not
    rule). So this encoder consumes ONLY input->output DIFFERENCES per demo:
      * raw 10x10 color-transition histogram over changed cells  (D1-proven task-specific)
      * input color histogram[10], output color histogram[10]
      * scalar deltas: changed_rate, |dH|, |dW|, area_ratio, in/out nonbg rate
  - per-demo MLP -> masked mean across demos -> K rule slots + a scalar rule_confidence.
  - A small RuleConditionedDecoder (embed query input + cross-attend to rule slots ->
    token logits) is the Phase-A probe head; it isolates whether the ENCODER carries a
    task-specific rule, independent of the big TRM.

Token convention: PAD=0, EOS=1, color = token-2 (colors 0..9). Grid is 30x30=900 tokens.
No DSL, no program search — pure differentiable low-level transformation statistics.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_TOKEN = 0
EOS_TOKEN = 1
COLOR_OFFSET = 2
N_COLORS = 10
N_TRANSITIONS = N_COLORS * N_COLORS  # 100
GRID_SIDE = 30
GRID_LEN = GRID_SIDE * GRID_SIDE     # 900
VOCAB = 12                            # PAD, EOS, 10 colors


# ---------------------------------------------------------------------------
# Explicit per-demo delta features (no learned pooling of raw images)
# ---------------------------------------------------------------------------
def _color_grid(tokens: torch.Tensor) -> torch.Tensor:
    """token grid -> color id (0..9) where colored, else -1 (pad/eos)."""
    is_color = tokens >= COLOR_OFFSET
    return torch.where(is_color, (tokens - COLOR_OFFSET).clamp(0, 9),
                       torch.full_like(tokens, -1))


def demo_delta_features(context_inputs: torch.Tensor, context_outputs: torch.Tensor,
                        context_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-demo explicit transformation features.

    Returns:
        feats [B, M, F]   (F = 100 transition + 10 in-hist + 10 out-hist + 6 scalars = 126)
        valid [B, M]      bool: demo has at least one changed colored cell
    """
    x = context_inputs.long()
    y = context_outputs.long()
    cm = context_mask.to(torch.bool)
    xs = x >= COLOR_OFFSET
    ys = y >= COLOR_OFFSET
    real = xs & ys
    changed = real & (x != y) & cm.unsqueeze(-1)              # [B,M,L]
    xc = (x - COLOR_OFFSET).clamp(0, 9)
    yc = (y - COLOR_OFFSET).clamp(0, 9)

    # 1) transition histogram (changed cells)
    pair = (xc * N_COLORS + yc).clamp(0, N_TRANSITIONS - 1)
    trans = (F.one_hot(pair, N_TRANSITIONS).float() * changed.unsqueeze(-1).float()).sum(2)
    trans = trans / (trans.sum(-1, keepdim=True) + 1e-6)      # [B,M,100]

    # 2) input/output color histograms (over colored cells, masked)
    in_oh = F.one_hot(xc, N_COLORS).float() * (xs & cm.unsqueeze(-1)).unsqueeze(-1).float()
    out_oh = F.one_hot(yc, N_COLORS).float() * (ys & cm.unsqueeze(-1)).unsqueeze(-1).float()
    in_hist = in_oh.sum(2); in_hist = in_hist / (in_hist.sum(-1, keepdim=True) + 1e-6)
    out_hist = out_oh.sum(2); out_hist = out_hist / (out_hist.sum(-1, keepdim=True) + 1e-6)

    # 3) scalar deltas
    in_area = (xs & cm.unsqueeze(-1)).float().sum(-1)         # [B,M]
    out_area = (ys & cm.unsqueeze(-1)).float().sum(-1)
    changed_rate = changed.float().sum(-1) / (real & cm.unsqueeze(-1)).float().sum(-1).clamp_min(1)
    area_ratio = out_area / in_area.clamp_min(1)
    in_nonbg = in_area / GRID_LEN
    out_nonbg = out_area / GRID_LEN
    # crude bbox-width proxy via per-row/col presence is expensive; use area-derived shape
    # delta surrogate: sqrt(area) difference (cheap, differentiable-free)
    dshape = (out_area.sqrt() - in_area.sqrt()).abs() / GRID_SIDE
    add_rate = ((~xs) & ys & cm.unsqueeze(-1)).float().sum(-1) / GRID_LEN   # cells that became colored
    del_rate = (xs & (~ys) & cm.unsqueeze(-1)).float().sum(-1) / GRID_LEN   # cells that became blank
    scalars = torch.stack([changed_rate, area_ratio.clamp(0, 5) / 5.0,
                           in_nonbg, out_nonbg, add_rate, del_rate], dim=-1)  # [B,M,6]

    feats = torch.cat([trans, in_hist, out_hist, scalars], dim=-1)           # [B,M,126]
    valid = changed.any(dim=-1) & cm                                          # [B,M]
    return feats, valid


def pairdelta_intent_features(
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
    context_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Cheap PairDelta intent diagnostics, not a solver.

    Returns per-task scalars used as evidence/router hints. They are computed
    directly from support input->output deltas and do not write logits.
    """
    with torch.no_grad():
        x = context_inputs.long()
        y = context_outputs.long()
        cm = context_mask.to(torch.bool)
        xs = x >= COLOR_OFFSET
        ys = y >= COLOR_OFFSET
        valid = xs & ys & cm.unsqueeze(-1)
        changed = valid & (x != y)
        active = cm.float().sum(dim=1).clamp_min(1.0)
        valid_counts = valid.float().sum(dim=(1, 2)).clamp_min(1.0)
        changed_counts = changed.float().sum(dim=(1, 2))
        changed_rate = changed_counts / valid_counts

        x_struct = torch.where(xs, torch.full_like(x, 2), torch.where(x == EOS_TOKEN, torch.ones_like(x), torch.zeros_like(x)))
        y_struct = torch.where(ys, torch.full_like(y, 2), torch.where(y == EOS_TOKEN, torch.ones_like(y), torch.zeros_like(y)))
        shape_same_per_demo = ((x_struct == y_struct).float().mean(dim=-1) * cm.float()).sum(dim=1) / active
        non_empty = (changed_counts > 0).float()
        sparse = ((changed_rate > 0.0) & (changed_rate < 0.5)).float()

        B = x.shape[0]
        counts = torch.zeros(B, N_COLORS, N_COLORS, device=x.device, dtype=torch.float32)
        xc = (x - COLOR_OFFSET).clamp(0, 9)
        yc = (y - COLOR_OFFSET).clamp(0, 9)
        pair = (xc * N_COLORS + yc).clamp(0, N_TRANSITIONS - 1)
        pair_oh = F.one_hot(pair, N_TRANSITIONS).float() * changed.unsqueeze(-1).float()
        counts = pair_oh.sum(dim=(1, 2)).view(B, N_COLORS, N_COLORS)
        total = counts.sum(dim=(1, 2))
        row_peaks = counts.max(dim=2).values.sum(dim=1)
        global_consistency = row_peaks / total.clamp_min(1.0)
        src_mass = counts.sum(dim=2)
        dominant_source = src_mass.argmax(dim=1)
        dominant_target = counts[torch.arange(B, device=x.device), dominant_source].argmax(dim=1)

        conditional_score = shape_same_per_demo * non_empty * sparse * (1.0 - changed_rate.clamp(0.0, 1.0))
        global_score = shape_same_per_demo * non_empty * global_consistency
        feature = conditional_score.unsqueeze(-1)
        return {
            "feature": feature,
            "shape_preserved": shape_same_per_demo,
            "changed_rate": changed_rate,
            "dominant_source_color": dominant_source.to(torch.float32),
            "dominant_target_color": dominant_target.to(torch.float32),
            "conditional_recolor_score": conditional_score,
            "global_recolor_score": global_score,
            "changed_cells": changed_counts,
            "valid_cells": valid_counts,
        }


FEATURE_DIM = N_TRANSITIONS + N_COLORS + N_COLORS + 6  # 126


class PairDeltaEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, n_slots: int = 8, n_heads: int = 4):
        super().__init__()
        self.D = hidden_dim
        self.K = n_slots
        self.pair_mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.slot_queries = nn.Parameter(torch.randn(n_slots, hidden_dim) * 0.02)
        self.slot_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.confidence = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, 1))

    def forward(self, context_inputs, context_outputs, context_mask) -> Dict[str, torch.Tensor]:
        feats, valid = demo_delta_features(context_inputs, context_outputs, context_mask)
        B, M, _ = feats.shape
        d = self.pair_mlp(feats)                                  # [B,M,D]
        # masked slot attention: slots (query) attend over per-demo features (key/value)
        q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)      # [B,K,D]
        kpm = ~valid                                             # True = mask out
        all_masked = kpm.all(dim=-1, keepdim=True)
        kpm = torch.where(all_masked, torch.zeros_like(kpm), kpm)
        slots, _ = self.slot_attn(q, d, d, key_padding_mask=kpm)  # [B,K,D]
        slots = self.slot_norm(slots)
        # rule vector = mean slot; confidence from it
        rule_vec = slots.mean(dim=1)                              # [B,D]
        conf = torch.sigmoid(self.confidence(rule_vec)).squeeze(-1)  # [B]
        # The kpm all-masked flip above only avoids the attention NaN; the slots it produces are over
        # a fully-masked (garbage) key set. Zero the rule + confidence for those rows so a consumer
        # cannot silently read a confident-looking rule for an empty task (emptiness is now explicit
        # in BOTH rule_confidence==0 AND demo_valid, not demo_valid alone).
        empty = all_masked.squeeze(-1)                           # [B]
        slots = torch.where(empty.view(B, 1, 1), torch.zeros_like(slots), slots)
        rule_vec = torch.where(empty.unsqueeze(-1), torch.zeros_like(rule_vec), rule_vec)
        conf = torch.where(empty, torch.zeros_like(conf), conf)
        return {"rule_slots": slots, "rule_vec": rule_vec,
                "rule_confidence": conf, "demo_valid": valid}


class RuleConditionedDecoder(nn.Module):
    """Phase-A probe decoder: predict y_j from x_j conditioned on rule_slots."""

    def __init__(self, hidden_dim: int = 256, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.tok_embed = nn.Embedding(VOCAB, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, GRID_LEN, hidden_dim) * 0.02)
        self.cross = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        enc = nn.TransformerEncoderLayer(hidden_dim, n_heads, dim_feedforward=hidden_dim * 2,
                                         batch_first=True, activation="gelu")
        self.body = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(hidden_dim, VOCAB)

    def forward(self, query_input: torch.Tensor, rule_slots: Optional[torch.Tensor]) -> torch.Tensor:
        assert query_input.shape[-1] == self.pos_embed.shape[1], (
            f"RuleConditionedDecoder.pos_embed is fixed at length {self.pos_embed.shape[1]} (GRID_LEN); "
            f"got query_input length {query_input.shape[-1]}. Rebuild the decoder for the new grid length.")
        h = self.tok_embed(query_input.long()) + self.pos_embed          # [B,GRID_LEN,D]
        if rule_slots is not None:
            crossed, _ = self.cross(h, rule_slots, rule_slots)
            h = self.cross_norm(h + crossed)
        h = self.body(h)
        return self.head(h)                                              # [B,900,VOCAB]


def _self_test() -> None:
    torch.manual_seed(0)
    B, M, L, D = 4, 3, GRID_LEN, 64
    # task t recolors color (t+2) -> color (t+5); build demos accordingly
    ci = torch.full((B, M, L), PAD_TOKEN)
    co = torch.full((B, M, L), PAD_TOKEN)
    for t in range(B):
        src = (t % 8) + COLOR_OFFSET
        dst = ((t + 3) % 8) + COLOR_OFFSET
        ci[t, :, :40] = src
        co[t, :, :40] = dst
    cm = torch.ones(B, M, dtype=torch.bool)

    enc = PairDeltaEncoder(hidden_dim=D, n_slots=8)
    dec = RuleConditionedDecoder(hidden_dim=D)
    out = enc(ci, co, cm)
    assert out["rule_slots"].shape == (B, 8, D)
    assert out["rule_confidence"].shape == (B,)

    # different tasks -> different rule vectors (cosine well below 1)
    rv = F.normalize(out["rule_vec"], dim=-1)
    cross = (rv @ rv.t())
    off = cross - torch.eye(B)
    assert off.max().item() < 0.99, f"rule vecs not task-specific: {off.max().item()}"

    # decoder: real rule should beat a shuffled (wrong-task) rule on CE for demo 0 as target
    xq = ci[:, 0]; yq = co[:, 0]
    real_logits = dec(xq, out["rule_slots"])
    shuf_slots = out["rule_slots"].roll(1, dims=0)               # other task's rule
    shuf_logits = dec(xq, shuf_slots)
    ce_real = F.cross_entropy(real_logits.reshape(-1, VOCAB), yq.reshape(-1).long())
    ce_shuf = F.cross_entropy(shuf_logits.reshape(-1, VOCAB), yq.reshape(-1).long())
    # (untrained decoder: just check it runs + shapes; separation appears after training)
    assert real_logits.shape == (B, L, VOCAB)
    loss = ce_real + enc(ci, co, cm)["rule_confidence"].mean()
    loss.backward()
    print(f"pair_delta_encoder self-test PASS  (feature_dim={FEATURE_DIM}, "
          f"max off-diag rule cos={off.max().item():.3f}, ce_real={ce_real.item():.3f}, "
          f"ce_shuffle={ce_shuf.item():.3f})")


if __name__ == "__main__":
    _self_test()
