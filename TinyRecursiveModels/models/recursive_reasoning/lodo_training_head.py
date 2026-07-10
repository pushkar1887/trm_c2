"""LODO Training Head — Leave-One-Demo-Out supervised loss + rule-binding diagnostic.

Holds out one demo at a time, predicts its OUTPUT from the remaining demos using the
(same) generator, and scores that prediction. Two uses:
  - Phase 1 (generator frozen): runs under no_grad as a pure DIAGNOSTIC. The
    loo_exact_rate / loo_change_overlap signals tell you LIVE whether the model's
    rule extraction generalizes across demo subsets.
  - Phase 2 (generator unfrozen, lr~1e-6): the held-out demo's GT output is a SECOND
    supervised CE target — gradient flows into the generator, forcing robust rules.

Adapted to THIS repo's interfaces (verified):
  - inner forward: `carry, out = inner(carry=carry, batch=batch)`, out["logits"] [B,L,V]
  - `inner.initial_carry(batch)`
  - batch keys: inputs, labels, puzzle_identifiers, context_inputs/outputs/mask
  - holding out demo i = flip context_mask[:, i] = False (masking is honored in
    _demo_tokens), and put demo i's grids in the inputs/labels slots.

Memory: a full halt_max_steps unroll WITH grad would OOM the 8 GB card, so we use
TRUNCATED BPTT — first (n_steps-1) increments under no_grad, final increment with grad.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_LABEL_ID = 0


class LODOTrainingHead(nn.Module):
    def __init__(self, inner_model: nn.Module, n_steps: int = 16, max_loo: int = 4,
                 blank_pid: bool = True):
        super().__init__()
        # Reference to the generator's inner model (NOT a copy). Not registered as a
        # submodule to avoid double-registration in the optimizer.
        object.__setattr__(self, "inner", inner_model)
        self.n_steps = int(n_steps)
        self.max_loo = int(max_loo)
        # blank_pid=True routes held-out-demo prediction through the blank puzzle
        # embedding so loo_exact measures rule induction, not puzzle-id memorization.
        self.blank_pid = bool(blank_pid)
        self.metrics_buffer: List[dict] = []

    @staticmethod
    def _reconstruct_loo_batch(batch: Dict[str, torch.Tensor], i: int,
                               blank_pid: bool = True) -> Dict[str, torch.Tensor]:
        """Batch where demo i is the 'test' pair and is masked out of the context.

        blank_pid (default True) — CRITICAL for a valid LODO diagnostic. The
        generator's puzzle_emb is trained per-puzzle-id and (for aug1000 on the
        original TRM) effectively MEMORIZES each puzzle's answer. If we leave
        puzzle_identifiers untouched, the model reconstructs the held-out demo by
        looking up its memorized embedding, NOT by inferring the rule from the
        remaining demos -> loo_exact_rate is a vacuous ~1.0 even with the model
        frozen. Setting puzzle_identifiers=0 (the blank row) forces the model to
        derive demo i's output from the OTHER demos only, which is the actual
        rule-binding signal we want to watch during training.
        """
        ci = batch["context_inputs"]
        co = batch["context_outputs"]
        lodo = dict(batch)  # shallow copy; we replace the keys we change
        lodo["inputs"] = ci[:, i].clone()
        lodo["labels"] = co[:, i].clone()
        cm = batch["context_mask"].clone()
        cm[:, i] = False
        lodo["context_mask"] = cm
        if blank_pid and "puzzle_identifiers" in batch and torch.is_tensor(batch["puzzle_identifiers"]):
            # Route through the blank/<pad> puzzle embedding (row 0) so the model
            # cannot recall the memorized answer for this puzzle id.
            lodo["puzzle_identifiers"] = torch.zeros_like(batch["puzzle_identifiers"])
        # If per-test visual features exist, swap in demo i's; else leave as-is.
        if "context_input_visual_features" in batch and "input_visual_features" in batch:
            lodo["input_visual_features"] = batch["context_input_visual_features"][:, i].clone()
        return lodo

    def _forward_generator(self, lodo_batch: Dict[str, torch.Tensor], grad: bool) -> torch.Tensor:
        """Run the generator on a LODO batch. Truncated BPTT: grad only on final step."""
        # initial_carry builds carry tensors under the ambient device context (the TRM
        # relies on this — see eval_with_verifier). Without the context they default to
        # CPU and reset_carry hits a cuda/cpu mismatch.
        ref = lodo_batch.get("inputs")
        dev_type = ref.device.type if torch.is_tensor(ref) else "cpu"
        with torch.device(dev_type):
            carry = self.inner.initial_carry(lodo_batch)
        if self.n_steps > 1:
            with torch.no_grad():
                for _ in range(self.n_steps - 1):
                    carry, _out = self.inner(carry=carry, batch=lodo_batch)
        if grad:
            carry, out = self.inner(carry=carry, batch=lodo_batch)
        else:
            with torch.no_grad():
                carry, out = self.inner(carry=carry, batch=lodo_batch)
        return out["logits"]

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        grad: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Return mean LODO CE loss; append per-holdout metrics to metrics_buffer.

        `grad=False` → diagnostic only (Phase 1). `grad=True` → supervised (Phase 2)."""
        ci = batch["context_inputs"]
        co = batch["context_outputs"]
        cm = batch["context_mask"].to(torch.bool)
        B, N = ci.shape[0], ci.shape[1]
        if N < 2:
            return ci.new_zeros((), dtype=torch.float32)

        # Prefer holding out demos that are (a) present in the context for most rows and
        # (b) have valid output cells for most rows — otherwise the holdout yields no
        # learnable target and the LODO loss silently becomes 0 (no generator gradient).
        out_valid = (co != IGNORE_LABEL_ID) & cm.unsqueeze(-1)  # [B, N, L]
        demo_quality = out_valid.any(dim=-1).float().mean(dim=0)  # [N] fraction of rows usable
        # Candidate demos: usable in at least one row; fall back to all if none qualify.
        usable = (demo_quality > 0).nonzero(as_tuple=False).flatten().tolist()
        if not usable:
            usable = list(range(N))
        # Order usable demos by quality (desc), then take up to max_loo, with a little
        # randomness among ties via a shuffled tiebreak.
        # RNG draw must live on the generator's device (callers may pass a CUDA generator).
        gdev = generator.device if generator is not None else ci.device
        order = sorted(usable, key=lambda d: (-float(demo_quality[d]),
                                              float(torch.rand((), generator=generator, device=gdev))))
        n_holdouts = min(len(order), self.max_loo)
        perm = order[:n_holdouts]

        losses = []
        for i in perm:
            lodo_batch = self._reconstruct_loo_batch(batch, int(i), blank_pid=self.blank_pid)
            logits = self._forward_generator(lodo_batch, grad=grad)  # [B, L, V]
            target = lodo_batch["labels"]                            # [B, L]
            valid = (target != IGNORE_LABEL_ID).reshape(-1)
            if int(valid.sum().item()) == 0:
                # Held-out demo had no valid target cells (all padding/EOS after masking).
                # cross_entropy would return NaN (0/0); skip this holdout entirely so a
                # NaN can never enter the loss (and corrupt the generator in Phase 2).
                continue
            loss_i = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                target.reshape(-1).long(),  # CUDA nll_loss requires int64 target
                ignore_index=IGNORE_LABEL_ID,
            )
            if not torch.isfinite(loss_i):
                continue  # defensive: never propagate a non-finite LODO loss
            losses.append(loss_i)

            with torch.no_grad():
                pred = logits.argmax(-1)
                mask = target != IGNORE_LABEL_ID
                # per-sample exact over the valid (canvas) cells
                eq = (pred == target) | (~mask)
                exact = eq.reshape(B, -1).all(dim=-1).float()
                cell_acc = ((pred == target) & mask).reshape(B, -1).sum(-1).float() / (
                    mask.reshape(B, -1).sum(-1).float() + 1e-6
                )
                test_in = lodo_batch["inputs"]
                input_changed = (test_in != target) & mask
                pred_changed = (test_in != pred) & mask
                change_overlap = (input_changed & pred_changed).reshape(B, -1).sum(-1).float() / (
                    input_changed.reshape(B, -1).sum(-1).float() + 1e-6
                )
                # Only record metrics for samples with valid target cells, so the
                # diagnostic isn't polluted by all-padding holdouts.
                row_valid = mask.reshape(B, -1).sum(-1) > 0
                if row_valid.any():
                    self.metrics_buffer.append({
                        "demo_idx": int(i),
                        "exact": exact[row_valid].detach().cpu(),
                        "cell_acc": cell_acc[row_valid].detach().cpu(),
                        "change_overlap": change_overlap[row_valid].detach().cpu(),
                    })

        if not losses:
            # No valid holdout this batch — return a real zero that still carries grad
            # in Phase 2 (so backward never sees a detached constant / empty stack).
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


def _self_test() -> None:
    """Self-test with a tiny stub inner model (no GPU / real TRM needed)."""
    torch.manual_seed(0)
    B, N, L, V = 2, 3, 12, 12

    class StubCarry:
        pass

    class StubInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, 8)
            self.proj = nn.Linear(8, V)

        def initial_carry(self, batch):
            return StubCarry()

        def forward(self, carry, batch):
            x = self.emb(batch["inputs"].long())  # [B, L, 8]
            return carry, {"logits": self.proj(x)}  # [B, L, V]

    inner = StubInner()
    head = LODOTrainingHead(inner, n_steps=3, max_loo=2)
    batch = {
        "inputs": torch.randint(0, V, (B, L)),
        "labels": torch.randint(0, V, (B, L)),
        "context_inputs": torch.randint(0, V, (B, N, L)),
        "context_outputs": torch.randint(0, V, (B, N, L)),
        "context_mask": torch.ones(B, N, dtype=torch.bool),
        "puzzle_identifiers": torch.zeros(B, dtype=torch.long),
    }

    # Diagnostic (no grad)
    loss_d = head(batch, grad=False)
    assert torch.isfinite(loss_d)
    assert len(head.metrics_buffer) == 2, len(head.metrics_buffer)
    for m in head.metrics_buffer:
        assert m["exact"].shape == (B,)

    # Supervised (grad) — gradient reaches the generator
    head.metrics_buffer.clear()
    loss_g = head(batch, grad=True)
    loss_g.backward()
    assert any(p.grad is not None for p in inner.parameters()), "no grad reached generator"

    # held-out demo really is masked from context
    lodo_batch = head._reconstruct_loo_batch(batch, 1)
    assert lodo_batch["context_mask"][:, 1].any().item() is False
    assert torch.equal(lodo_batch["inputs"], batch["context_inputs"][:, 1])

    from scripts.integrated_health_logger import lodo_metrics
    head.metrics_buffer.clear()
    head(batch, grad=False)
    lm = lodo_metrics(head.metrics_buffer)
    assert "loo_exact_rate" in lm and "loo_change_overlap" in lm

    print(f"lodo_training_head self-test PASS "
          f"(diag_loss={loss_d.item():.3f}, grad reaches generator, mask honored, "
          f"loo_metrics={ {k: round(v,3) for k,v in lm.items()} })")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    _self_test()
