import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from models.recursive_reasoning.color_transition_bank import ColorTransitionBank
from models.recursive_reasoning.color_repair_head import ColorRepairHead

torch.manual_seed(0)
B, M, L, H, COFF = 2, 3, 900, 512, 2
ci = torch.zeros(B, M, L, dtype=torch.long); co = torch.zeros(B, M, L, dtype=torch.long)
ci[0, :, :60] = 3 + COFF; co[0, :, :60] = 7 + COFF
ci[1, :, :60] = 8 + COFF; co[1, :, :60] = 1 + COFF
cm = torch.ones(B, M, dtype=torch.bool); ti = ci[:, 0].clone()

print("== Finding A: compute_rule_tokens gate ==")
ctb = ColorTransitionBank(hidden_dim=H, rule_tokens=16)
on = ctb(ci, co, cm, compute_metrics=False, compute_rule_tokens=True)
off = ctb(ci, co, cm, compute_metrics=False, compute_rule_tokens=False)
print("  ON  rule_tokens:", tuple(on["rule_tokens"].shape))
print("  OFF rule_tokens:", off["rule_tokens"], "(skipped = dead compute avoided)")
print("  OFF still has cond_inout%s hist%s src_agree%s" % (tuple(off["cond_inout"].shape), tuple(off["hist"].shape), tuple(off["src_agreement"].shape)))
print("  head signal (cond_inout) identical ON vs OFF:", bool(torch.equal(on["cond_inout"], off["cond_inout"])))
assert off["rule_tokens"] is None and torch.equal(on["cond_inout"], off["cond_inout"])

print("== Finding B: head consumes rule_vec / hist / both ==")
cond, sa, hist = off["cond_inout"], off["src_agreement"], off["hist"]
rv = torch.randn(B, 256)
gz = torch.randn(B, L, H); base = torch.randn(B, L, 10)
for src, dim, feat in [("rule_vec", 256, rv), ("hist", 100, hist), ("both", 356, torch.cat([rv, hist], -1))]:
    head = ColorRepairHead(hidden_dim=H, grid_side=30, rule_vec_dim=dim)
    nc, gl, rc, ap = head(gz, base, ti, cond, sa, rule_vec=feat)
    assert head.rule_proj.in_features == dim and feat.shape[1] == dim and nc.shape == (B, L, 10)
    (nc.pow(2).mean() + gl.pow(2).mean()).backward()
    assert head.rule_proj.weight.grad is not None
    print("  src=%-8s rule_proj.in=%d feat=%s out=%s init_noop_applied=%.4f grad_ok=True"
          % (src, head.rule_proj.in_features, tuple(feat.shape), tuple(nc.shape), ap.mean().item()))
print("ALL FINDINGS CHECKS PASS")