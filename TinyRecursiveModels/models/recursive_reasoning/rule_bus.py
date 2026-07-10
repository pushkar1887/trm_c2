"""S2 RULE BUS -- merge the 3 redundant cross-demo extractors into ONE fused rule descriptor.

Why (whole-session finding, ARCHITECTURE_V2.md):
  The bank (ColorTransitionBank) and the encoder (PairDeltaEncoder) compute the SAME 10x10 transition
  histogram, then DIVERGE: the bank makes a DETERMINISTIC colour-transform map (cond_inout, src_agree),
  the encoder makes a LEARNED rule_vec. Today the colour head reaches into BOTH separately (the lookup
  reads cond_inout; rule_proj reads rule_vec), and the struct branch feeds z_H. Three extractors, three
  injection points, overlapping. The Rule Bus fuses them ONCE with the right roles:

    FLOOR  = the deterministic bank descriptor (cond_inout + src_agree). Always present, reliable, the
             colour-transform the demos literally show. A learned projection of it -> a rule vector.
    SOLVER = the learned encoder rule_vec. The relational lift the floor cannot express. ZERO-INIT and
             AGREEMENT-GATED: it contributes only where the demos AGREE (mean src_agree), and starts as
             a pure no-op the LODO loss must earn -- so the fused rule can never do WORSE than the floor.
    STRUCT = optional 3rd facet (the relational-structure vector), added when supplied.

  OUT = ONE fused rule vector [B, out_dim], the head's SINGLE rule input (replaces the scattered
  cond/rule_vec injection). Off-flag (the bus not built) => the head reads the raw rule_vec, unchanged.

GATE (the check, pre-registered): at init the fused rule == FLOOR alone (solver zero-init); the gain
  from fusing the solver must BEAT the bank-alone floor on LODO -- else the encoder adds nothing and the
  bus reduces to the deterministic prior (a real, honest result). `floor_only()` exposes the baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_COLORS = 10
_FLOOR_IN = N_COLORS * N_COLORS + N_COLORS          # cond_inout(100) + src_agree(10)


class RuleBus(nn.Module):
    def __init__(self, rule_vec_dim: int, out_dim: int = 256, struct_dim: int = 0):
        super().__init__()
        self.out_dim = int(out_dim)
        # FLOOR: project the deterministic colour-transform map to a rule vector.
        self.floor_proj = nn.Linear(_FLOOR_IN, self.out_dim)
        # SOLVER: the learned relational lift. ZERO-INIT => fused == floor at start (floor-respecting:
        # the bus can only ADD over the deterministic prior, never start by corrupting it).
        self.solver_proj = nn.Linear(int(rule_vec_dim), self.out_dim)
        nn.init.zeros_(self.solver_proj.weight)
        nn.init.zeros_(self.solver_proj.bias)
        self.struct_proj = nn.Linear(int(struct_dim), self.out_dim) if struct_dim and struct_dim > 0 else None
        if self.struct_proj is not None:
            nn.init.zeros_(self.struct_proj.weight)
            nn.init.zeros_(self.struct_proj.bias)

    def _floor_in(self, cond_inout: torch.Tensor, src_agree: torch.Tensor) -> torch.Tensor:
        B = cond_inout.shape[0]
        dt = self.floor_proj.weight.dtype
        return torch.cat([cond_inout.reshape(B, -1).to(dt), src_agree.to(dt)], dim=-1)   # [B,110]

    def forward(self, cond_inout: torch.Tensor, src_agree: torch.Tensor,
                rule_vec: torch.Tensor, struct_vec: torch.Tensor | None = None) -> torch.Tensor:
        """cond_inout [B,10,10], src_agree [B,10], rule_vec [B,rule_vec_dim] -> fused [B,out_dim]."""
        dt = self.floor_proj.weight.dtype
        floor = self.floor_proj(self._floor_in(cond_inout, src_agree))                    # FLOOR
        agree = src_agree.to(dt).mean(dim=-1, keepdim=True).clamp(0.0, 1.0)               # [B,1] trust
        fused = floor + agree * self.solver_proj(rule_vec.to(dt))                         # + gated SOLVER
        if self.struct_proj is not None and struct_vec is not None:
            fused = fused + self.struct_proj(struct_vec.to(dt))
        return fused

    @torch.no_grad()
    def floor_only(self, cond_inout: torch.Tensor, src_agree: torch.Tensor) -> torch.Tensor:
        """The FLOOR projection alone (solver excluded) -- the gate-check baseline the fused rule must beat."""
        return self.floor_proj(self._floor_in(cond_inout, src_agree))


def _self_test():
    """build-the-check-first: at init fused == floor (solver no-op); grad reaches the solver; gate in [0,1]."""
    torch.manual_seed(0)
    B, D = 4, 256
    bus = RuleBus(rule_vec_dim=D, out_dim=128)
    cond = torch.rand(B, N_COLORS, N_COLORS); cond = cond / cond.sum(-1, keepdim=True)
    agree = torch.rand(B, N_COLORS)
    rule_vec = torch.randn(B, D, requires_grad=True)
    fused = bus(cond, agree, rule_vec)
    floor = bus.floor_only(cond, agree)
    assert torch.allclose(fused, floor, atol=1e-6), "init must equal FLOOR (solver is zero-init)"
    # after a tiny solver perturbation, the solver path must move the output (grad reaches it)
    fused.sum().backward()
    assert rule_vec.grad is None or rule_vec.grad.abs().sum() == 0, \
        "solver zero-init => no grad to rule_vec yet (expected at init)"
    g = bus.solver_proj.weight.grad
    assert g is not None and g.abs().sum() > 0, "solver_proj must receive gradient"
    assert fused.shape == (B, 128)
    print("rule_bus self-test PASS (init==floor, solver gets grad, shape ok).")


if __name__ == "__main__":
    _self_test()
