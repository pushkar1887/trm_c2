"""DemoConsistencyVerifier — a learned scorer for K-candidate ARC outputs.

Per the verifier flow specification:
  - Reuses C2's demo encoder (rule_bank from expose_demo_encoding) — does NOT
    re-encode demos.
  - Takes a candidate's 30x30 token grid, encodes it via a 2-layer transformer,
    cross-attends to the demo rule_bank, pools, fuses with scalar features and
    optional struct/shape features, and scores it.
  - ~4M params (small enough for RTX 4060 batch 32).

Inputs to forward():
    candidate_tokens:  (B, 900) int token ids 0..11
    rule_bank:         (B, R, D) from c2.expose_demo_encoding()
    rule_mask:         (B, R)    boolean valid-token mask
    scalar_feats:      (B, F_s)  [logit_margin, shape_conf, lodo_stab, change_match]
    struct_features:   (B, M, 10) optional from c2.expose_demo_encoding()
                                  pooled mean over M demos before fusion
    shape_features:    (B, shape_dim) optional 32-dim shape head output

Output:
    score logit per candidate: (B,)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


VOCAB_SIZE = 12  # 0=PAD, 1=EOS, 2-11=colours
TOKEN_GRID_SIDE = 30
TOKEN_GRID_LEN = TOKEN_GRID_SIDE * TOKEN_GRID_SIDE  # 900


class DemoConsistencyVerifier(nn.Module):
    """~4M-param verifier that scores K candidate ARC outputs per task."""

    def __init__(
        self,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_cand_layers: int = 2,
        scalar_feat_dim: int = 4,
        use_struct_features: bool = False,
        struct_feat_dim: int = 10,
        use_shape_features: bool = False,
        shape_feat_dim: int = 32,
        ffn_expansion: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_struct_features = bool(use_struct_features)
        self.use_shape_features = bool(use_shape_features)

        # Candidate token embedding. Separate from TRM's embed_tokens because the
        # verifier is a distinct module with its own optimizer / checkpoint.
        self.cand_token_embed = nn.Embedding(VOCAB_SIZE, self.hidden_dim)
        # Learnable 2D positional encoding (30x30 grid → 900 positions).
        self.cand_pos_embed = nn.Parameter(torch.zeros(1, TOKEN_GRID_LEN, self.hidden_dim))
        nn.init.trunc_normal_(self.cand_pos_embed, std=0.02)

        # Candidate encoder: lightweight 2-layer transformer.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(n_heads),
            dim_feedforward=self.hidden_dim * int(ffn_expansion),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cand_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_cand_layers))

        # Cross-attention: candidate queries ↔ demo rule_bank keys/values.
        # rule_bank dimensionality is hidden_dim (C2's hidden_size).
        self.cross_cand_to_demo = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=int(n_heads),
            batch_first=True,
        )
        self.post_cross_norm = nn.LayerNorm(self.hidden_dim)

        # Scalar features fusion (logit_margin, shape_conf, lodo_stab, change_match).
        self.scalar_feature_mlp = nn.Sequential(
            nn.Linear(int(scalar_feat_dim), self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Optional struct-features projection. Pools over M demos, then projects.
        if self.use_struct_features:
            self.struct_input_proj = nn.Sequential(
                nn.Linear(int(struct_feat_dim), self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        # Optional shape-features projection (32-dim shape head output).
        if self.use_shape_features:
            self.shape_input_proj = nn.Sequential(
                nn.Linear(int(shape_feat_dim), self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        # Final score head — single logit per candidate.
        self.score = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        candidate_tokens: torch.Tensor,           # (B, 900) int
        rule_bank: torch.Tensor,                  # (B, R, D)
        rule_mask: torch.Tensor,                  # (B, R) bool
        scalar_feats: torch.Tensor,               # (B, F_s) float
        struct_features: Optional[torch.Tensor] = None,  # (B, M, 10) or (B, 10)
        shape_features: Optional[torch.Tensor] = None,   # (B, shape_dim)
    ) -> torch.Tensor:
        assert candidate_tokens.dtype in (torch.int32, torch.int64), (
            f"candidate_tokens must be int; got {candidate_tokens.dtype}"
        )
        assert candidate_tokens.shape[-1] == TOKEN_GRID_LEN, (
            f"Expected {TOKEN_GRID_LEN} tokens, got {candidate_tokens.shape[-1]}"
        )

        # Encode candidate.
        cand = self.cand_token_embed(candidate_tokens.long()) + self.cand_pos_embed
        cand = self.cand_encoder(cand)

        # Cross-attend candidate -> demo rule bank.
        # `key_padding_mask` expects True for positions to MASK OUT.
        key_padding_mask = ~rule_mask.to(torch.bool)
        # Guard against rows with all-True padding (would NaN attention).
        has_key = (~key_padding_mask).any(dim=-1, keepdim=True)
        safe_mask = torch.where(has_key, key_padding_mask, torch.zeros_like(key_padding_mask))
        crossed, _ = self.cross_cand_to_demo(
            query=cand,
            key=rule_bank,
            value=rule_bank,
            key_padding_mask=safe_mask,
            need_weights=False,
        )
        crossed = self.post_cross_norm(crossed + cand)
        cand_pooled = crossed.mean(dim=1)  # (B, hidden_dim)

        # Fuse scalar features.
        scalar_emb = self.scalar_feature_mlp(scalar_feats.to(cand_pooled.dtype))
        fused = cand_pooled + scalar_emb

        # Optional struct features.
        if self.use_struct_features and struct_features is not None:
            if struct_features.dim() == 3:
                # (B, M, 10) → mean over M
                struct_pooled = struct_features.to(cand_pooled.dtype).mean(dim=1)
            else:
                struct_pooled = struct_features.to(cand_pooled.dtype)
            fused = fused + self.struct_input_proj(struct_pooled)

        # Optional shape features.
        if self.use_shape_features and shape_features is not None:
            fused = fused + self.shape_input_proj(shape_features.to(cand_pooled.dtype))

        return self.score(fused).squeeze(-1)  # (B,)


def count_parameters(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters() if p.requires_grad)


def _selftest() -> None:
    """Smoke test: construct + one forward pass + parameter count."""
    torch.manual_seed(0)
    model = DemoConsistencyVerifier(
        hidden_dim=512,
        n_heads=8,
        n_cand_layers=2,
        scalar_feat_dim=4,
        use_struct_features=True,
        use_shape_features=True,
    )
    B, R, F_s = 3, 32, 4
    tokens = torch.randint(0, VOCAB_SIZE, (B, TOKEN_GRID_LEN), dtype=torch.long)
    rule_bank = torch.randn(B, R, 512)
    rule_mask = torch.ones(B, R, dtype=torch.bool)
    scalar = torch.randn(B, F_s)
    struct_feats = torch.randn(B, 3, 10)  # 3 demos, 10 geom features each
    shape_feats = torch.randn(B, 32)
    score = model(tokens, rule_bank, rule_mask, scalar, struct_feats, shape_feats)
    assert score.shape == (B,), f"score shape {score.shape}; expected ({B},)"
    n_params = count_parameters(model)
    print(f"[verifier_head] params = {n_params:,} (target ~4M)")
    assert 2_000_000 < n_params < 12_000_000, f"param count out of expected band: {n_params}"
    print(f"[verifier_head] score sample: {score.tolist()}")
    print("[verifier_head self-test] PASS")


if __name__ == "__main__":
    _selftest()
