"""DemoConsistencyVerifierV3 — per-cell attribution + 6 scalar features.

Why a v3 (vs scripts/verifier_head.py's v2):
  The v2 verifier collapsed 900 candidate positions to one mean-pooled vector
  before scoring (`cand_pooled = crossed.mean(dim=1)`), then produced a single
  scalar. A code review identified this as the reason v2 scored at chance
  (AUROC 0.506): a 1-3 cell error is invisible after averaging over 900 cells.

  v3 fixes this two ways:
    1. A per-cell consistency head produces [B, 900] scores (INSPECTABLE) —
       you can see WHICH cells the verifier thinks violate the rule.
    2. The scalar decision is BUILT FROM the per-cell scores (their pooled
       statistic is added to the global term), so per-cell evidence is
       causally part of the judgement, not a discarded side output.

  It also consumes BOTH the rich continuous rule_bank AND the discrete VQ
  rule_tokens, and takes 6 scalar features (vs v2's effectively-constant 4).

Inputs (all per candidate):
    candidate_tokens [B, 900] long
    rule_bank        [B, R, D] float   continuous per-cell demo encoding (C2)
    rule_mask        [B, R]    bool
    rule_tokens      [B, K, D] float   discrete VQ rule codes (RuleBank), optional
    scalar_features  [B, F]    float   F defaults to 6
Outputs:
    consistency_logit   [B]
    per_cell_scores     [B, 900]   INSPECTABLE attribution
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

TOKEN_GRID_SIDE = 30
TOKEN_GRID_LEN = TOKEN_GRID_SIDE * TOKEN_GRID_SIDE
VOCAB = 12


class DemoConsistencyVerifierV3(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 512,
        n_heads: int = 8,
        scalar_dim: int = 6,
        n_encoder_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scalar_dim = scalar_dim

        self.cand_token_embed = nn.Embedding(VOCAB, hidden_dim)
        self.cand_pos_embed = nn.Parameter(torch.randn(1, TOKEN_GRID_LEN, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            hidden_dim, n_heads, dim_feedforward=hidden_dim * 2,
            batch_first=True, activation="gelu",
        )
        self.cand_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)

        self.cross_to_demos = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.cross_to_rules = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm_d = nn.LayerNorm(hidden_dim)
        self.norm_r = nn.LayerNorm(hidden_dim)

        # Per-cell consistency head (INSPECTABLE).
        self.per_cell_score = nn.Linear(hidden_dim, 1)

        # Scalar-feature projection.
        self.scalar_proj = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
        )
        # Global pooled score head; its input is [global_feat || percell_stats(3)].
        self.pool_score = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_tokens: torch.Tensor,
        rule_bank: torch.Tensor,
        rule_mask: torch.Tensor,
        scalar_features: torch.Tensor,
        rule_tokens: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        assert candidate_tokens.shape[-1] == TOKEN_GRID_LEN
        rule_bank = rule_bank.to(torch.float32)
        if rule_tokens is not None:
            rule_tokens = rule_tokens.to(torch.float32)

        cand = self.cand_token_embed(candidate_tokens.long()) + self.cand_pos_embed
        cand = self.cand_encoder(cand)  # [B, 900, D]

        # Cross-attend candidate -> continuous rule_bank (per-cell, no pooling).
        kpm = ~rule_mask.to(torch.bool)
        all_masked = kpm.all(dim=-1, keepdim=True)
        kpm = torch.where(all_masked, torch.zeros_like(kpm), kpm)
        crossed_d, attn_d = self.cross_to_demos(
            cand, rule_bank, rule_bank, key_padding_mask=kpm,
            need_weights=return_attention, average_attn_weights=False,
        )
        cand_d = self.norm_d(cand + crossed_d)

        # Cross-attend to discrete rule tokens (if provided).
        if rule_tokens is not None:
            crossed_r, attn_r = self.cross_to_rules(
                cand_d, rule_tokens, rule_tokens, need_weights=return_attention,
            )
            cand_r = self.norm_r(cand_d + crossed_r)
        else:
            cand_r, attn_r = cand_d, None

        # Per-cell consistency scores (INSPECTABLE).
        per_cell_scores = self.per_cell_score(cand_r).squeeze(-1)  # [B, 900]

        # Build the scalar decision FROM per-cell evidence (fixes the C1 bottleneck):
        # global feature + 3 statistics of the per-cell score distribution.
        global_feat = cand_r.mean(dim=1)  # [B, D]
        scalar_emb = self.scalar_proj(scalar_features.to(global_feat.dtype))
        fused = global_feat + scalar_emb
        pc_mean = per_cell_scores.mean(dim=1, keepdim=True)
        pc_min = per_cell_scores.min(dim=1, keepdim=True).values   # worst (most-wrong) cell
        pc_std = per_cell_scores.std(dim=1, keepdim=True)
        pooled_in = torch.cat([fused, pc_mean, pc_min, pc_std], dim=-1)
        consistency_logit = self.pool_score(pooled_in).squeeze(-1)  # [B]

        out = {
            "consistency_logit": consistency_logit,
            "per_cell_scores": per_cell_scores,
        }
        if return_attention:
            out["demo_attn"] = attn_d
            out["rule_attn"] = attn_r
        return out


def count_parameters(m: nn.Module) -> int:
    return sum(int(p.numel()) for p in m.parameters() if p.requires_grad)


def _self_test() -> None:
    torch.manual_seed(0)
    B, D, R, K = 4, 64, 50, 8
    v = DemoConsistencyVerifierV3(hidden_dim=D, n_heads=4, scalar_dim=6)
    tokens = torch.randint(0, VOCAB, (B, TOKEN_GRID_LEN))
    rule_bank = torch.randn(B, R, D)
    rule_mask = torch.ones(B, R, dtype=torch.bool)
    rule_mask[0, 25:] = False
    rule_tokens = torch.randn(B, K, D)
    scalars = torch.randn(B, 6)

    out = v(tokens, rule_bank, rule_mask, scalars, rule_tokens=rule_tokens)
    assert out["consistency_logit"].shape == (B,), out["consistency_logit"].shape
    assert out["per_cell_scores"].shape == (B, TOKEN_GRID_LEN), out["per_cell_scores"].shape

    # Works without discrete rule_tokens too.
    out2 = v(tokens, rule_bank, rule_mask, scalars, rule_tokens=None)
    assert out2["consistency_logit"].shape == (B,)

    # Gradient flows.
    loss = out["consistency_logit"].sum() + out["per_cell_scores"].sum()
    loss.backward()

    # bf16 rule_bank is accepted (cast at boundary).
    out3 = v(tokens, rule_bank.to(torch.bfloat16), rule_mask, scalars,
             rule_tokens=rule_tokens.to(torch.bfloat16))
    assert out3["consistency_logit"].shape == (B,)

    print(f"verifier_v3 self-test PASS (params={count_parameters(v):,}, "
          f"per_cell shape OK, bf16 boundary OK)")


if __name__ == "__main__":
    _self_test()
