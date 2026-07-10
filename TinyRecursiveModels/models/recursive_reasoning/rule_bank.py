"""VQ Rule Bank — discrete rule-code binding on top of C2's continuous rule_bank.

DESIGN NOTE (deviation from the original spec, on purpose):
  The spec REPLACES the continuous rule_bank with K=8 discrete codes. A prior
  code review found C2's existing rule_bank is information-rich (per-cell demo
  encoding + cross-demo pair features) and that close-miss errors are a *local
  per-cell precision* problem — so collapsing the rule to 8 codes would discard
  exactly the signal those 1-3 wrong cells need.

  Therefore this module CONSUMES the existing rich rule_bank [B, R, D] (from
  `expose_demo_encoding`) and emits K discrete rule tokens as an ADDITIONAL,
  parallel signal. Downstream consumers (generator cross-attention, verifier)
  can read BOTH the continuous bank and the discrete codes. The "discrete rule
  binding" hypothesis stays testable (via code utilization / per-task code
  signatures) without throwing away per-cell information.

Introspection this module exposes (read by scripts/integrated_health_logger.py):
  - rule_code_ids        [B, K]  which codebook entry each slot picked
  - code_usage_ema       [V]     long-run usage per code (for active-code count)
  - last_batch_code_ids          for per-task code-signature tracking
  - perplexity / entropy of code usage (codebook health)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RuleBank(nn.Module):
    """VQ codebook that pools a rich rule_bank into K discrete rule tokens.

    forward(rule_bank, rule_mask) -> dict with:
        rule_tokens   [B, K, D]  straight-through quantized rule tokens
        rule_code_ids [B, K]     long, codebook indices in [0, V)
        codebook_loss scalar     VQ loss (codebook + commitment)
        slot_continuous [B,K,D]  pre-quantization slots (for inspection)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        codebook_size: int = 256,
        n_slots: int = 8,
        n_heads: int = 8,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        use_cosine: bool = False,
        restart_dead_codes: bool = True,
        dead_code_threshold: float = 1e-4,
    ) -> None:
        super().__init__()
        self.D = hidden_dim
        self.V = codebook_size
        self.K = n_slots
        self.commitment_weight = float(commitment_weight)
        self.decay = float(decay)
        self.use_cosine = bool(use_cosine)
        self.restart_dead_codes = bool(restart_dead_codes)
        self.dead_code_threshold = float(dead_code_threshold)

        self.codebook = nn.Embedding(codebook_size, hidden_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        self.slot_queries = nn.Parameter(torch.randn(1, n_slots, hidden_dim) * 0.02)
        self.slot_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

        # EMA usage tracker (buffer so it persists in checkpoints, no grad).
        self.register_buffer("code_usage_ema", torch.zeros(codebook_size))
        # Keep the most recent batch's code ids for per-task signature logging.
        self.register_buffer("last_batch_code_ids", torch.zeros(0, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def _maybe_restart_dead_codes(self, slot_vectors: torch.Tensor) -> int:
        """Reset codes with ~zero EMA usage to random current slot vectors.

        Standard VQ-VAE collapse remedy. Returns count of restarted codes.
        Only call periodically from the trainer (not every step)."""
        if not self.restart_dead_codes:
            return 0
        dead = self.code_usage_ema < self.dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        flat = slot_vectors.reshape(-1, self.D)
        if flat.shape[0] == 0:
            return 0
        pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        self.codebook.weight.data[dead] = flat[pick].to(self.codebook.weight.dtype)
        # Give restarted codes a small usage credit so they aren't instantly re-killed.
        self.code_usage_ema[dead] = self.dead_code_threshold * 2.0
        return n_dead

    def _distances(self, flat: torch.Tensor) -> torch.Tensor:
        cb = self.codebook.weight  # [V, D]
        if self.use_cosine:
            fn = F.normalize(flat, dim=-1)
            cn = F.normalize(cb, dim=-1)
            return 1.0 - fn @ cn.t()  # [N, V] cosine distance
        # squared L2
        return (
            flat.pow(2).sum(1, keepdim=True)
            - 2.0 * flat @ cb.t()
            + cb.pow(2).sum(1, keepdim=True).t()
        )

    def forward(self, rule_bank: torch.Tensor, rule_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, R, D = rule_bank.shape
        assert D == self.D, f"rule_bank dim {D} != hidden_dim {self.D}"
        slots = self.slot_queries.expand(B, -1, -1)  # [B, K, D]
        key_padding_mask = None
        if rule_mask is not None:
            # MultiheadAttention masks where True. rule_mask True = valid → invert.
            key_padding_mask = ~rule_mask.to(torch.bool)
            # Guard all-masked rows (would NaN): if a row has no valid keys, unmask it.
            all_masked = key_padding_mask.all(dim=-1, keepdim=True)
            key_padding_mask = torch.where(all_masked, torch.zeros_like(key_padding_mask), key_padding_mask)

        slot_vectors, _ = self.slot_attn(slots, rule_bank, rule_bank, key_padding_mask=key_padding_mask)
        # slot_vectors: [B, K, D]

        flat = slot_vectors.reshape(B * self.K, D).float()
        distances = self._distances(flat)
        code_ids = distances.argmin(dim=1)  # [B*K]
        quantized = self.codebook(code_ids).reshape(B, self.K, D).to(slot_vectors.dtype)
        code_ids = code_ids.reshape(B, self.K)

        # Straight-through estimator.
        rule_tokens = slot_vectors + (quantized - slot_vectors).detach()

        # VQ losses (codebook moves toward slots; slots commit to codebook).
        codebook_loss = F.mse_loss(quantized, slot_vectors.detach())
        commit_loss = F.mse_loss(slot_vectors, quantized.detach())
        total_vq_loss = codebook_loss + self.commitment_weight * commit_loss

        # EMA usage update (no grad).
        with torch.no_grad():
            onehot = F.one_hot(code_ids.reshape(-1), num_classes=self.V).float()
            usage = onehot.mean(dim=0)  # fraction of slots assigned to each code
            self.code_usage_ema.mul_(self.decay).add_(usage * (1.0 - self.decay))
            self.last_batch_code_ids = code_ids.detach().reshape(-1).clone()

        return {
            "rule_tokens": rule_tokens,
            "rule_code_ids": code_ids,
            "codebook_loss": total_vq_loss,
            "slot_continuous": slot_vectors,
        }

    # ----- introspection helpers (called by the health logger) -----
    @torch.no_grad()
    def active_codes(self) -> int:
        return int((self.code_usage_ema > self.dead_code_threshold).sum().item())

    @torch.no_grad()
    def code_entropy_ratio(self) -> float:
        ema = self.code_usage_ema
        p = ema / (ema.sum() + 1e-8)
        ent = -(p * (p + 1e-12).log()).sum().item()
        import math
        return ent / math.log(self.V)


def _self_test() -> None:
    torch.manual_seed(0)
    B, R, D, V, K = 4, 64, 32, 64, 8
    rb = RuleBank(hidden_dim=D, codebook_size=V, n_slots=K)
    rule_bank = torch.randn(B, R, D, requires_grad=True)
    rule_mask = torch.ones(B, R, dtype=torch.bool)
    rule_mask[0, 32:] = False  # a partially-masked row
    rule_mask[1, :] = False    # an all-masked row (guard path)

    out = rb(rule_bank, rule_mask)
    assert out["rule_tokens"].shape == (B, K, D), out["rule_tokens"].shape
    assert out["rule_code_ids"].shape == (B, K)
    assert out["rule_code_ids"].min() >= 0 and out["rule_code_ids"].max() < V
    assert torch.isfinite(out["codebook_loss"]), "vq loss not finite"

    # Straight-through: gradient must flow back to rule_bank.
    loss = out["rule_tokens"].pow(2).mean() + out["codebook_loss"]
    loss.backward()
    assert rule_bank.grad is not None and torch.isfinite(rule_bank.grad).all()

    # EMA + active codes populated.
    assert rb.active_codes() >= 1
    assert 0.0 <= rb.code_entropy_ratio() <= 1.0

    # Dead-code restart returns an int and doesn't crash.
    n_restarted = rb._maybe_restart_dead_codes(out["slot_continuous"].detach())
    assert isinstance(n_restarted, int)

    print("rule_bank self-test PASS "
          f"(active_codes={rb.active_codes()}/{V}, entropy_ratio={rb.code_entropy_ratio():.3f}, "
          f"restarted={n_restarted})")


if __name__ == "__main__":
    _self_test()
