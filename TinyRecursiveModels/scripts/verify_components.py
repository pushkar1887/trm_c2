"""CPU component-IO verification: do inputs go in, outputs come out, wiring match, usable?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from models.recursive_reasoning.color_transition_bank import ColorTransitionBank
from models.recursive_reasoning.pair_delta_encoder import PairDeltaEncoder
from models.recursive_reasoning.color_repair_head import ColorRepairHead

torch.manual_seed(0)
B, M, L, H, COFF = 2, 3, 900, 512, 2
# task0: recolour 3->7 on a block; task1: recolour 8->1. Identity elsewhere (copy).
ci = torch.zeros(B, M, L, dtype=torch.long); co = torch.zeros(B, M, L, dtype=torch.long)
ci[0, :, :60] = 3 + COFF; co[0, :, :60] = 7 + COFF
ci[0, :, 60:120] = 5 + COFF; co[0, :, 60:120] = 5 + COFF      # copy cells (colour 5 -> 5)
ci[1, :, :60] = 8 + COFF; co[1, :, :60] = 1 + COFF
cm = torch.ones(B, M, dtype=torch.bool)
target_inputs = ci[:, 0].clone()                              # test grid = first demo input

print("="*70)
print("[1] ColorTransitionBank  input: context_in/out/mask", tuple(ci.shape))
ctb = ColorTransitionBank(hidden_dim=H, rule_tokens=16)
o = ctb(ci, co, cm, compute_metrics=False)
cond, sa, rt = o["cond_inout"], o["src_agreement"], o["rule_tokens"]
print("    out cond_inout", tuple(cond.shape), "src_agreement", tuple(sa.shape), "rule_tokens", tuple(rt.shape))
print("    cond P(out|in=3) task0 argmax =", int(cond[0,3].argmax()), "(expect 7)  | in=8 task1 =", int(cond[1,8].argmax()), "(expect 1)")
print("    cond P(out|in=5) task0 argmax =", int(cond[0,5].argmax()), "(expect 5 = copy preserved)")
print("    src_agree range [%.2f, %.2f] (expect within [0,1])" % (sa.min().item(), sa.max().item()))
print("    -> cond_inout + src_agree CONSUMED by ColorRepairHead. rule_tokens -> dead branch (gated off when head on): NOT consumed.")

print("="*70)
print("[2] PairDeltaEncoder  input: context_in/out/mask")
enc = PairDeltaEncoder(hidden_dim=256, n_slots=8)
eo = enc(ci, co, cm)
rv, rs = eo["rule_vec"], eo["rule_slots"]
cos = F.cosine_similarity(rv[0:1], rv[1:2]).item()
print("    out rule_vec", tuple(rv.shape), "rule_slots", tuple(rs.shape), "finite=", bool(torch.isfinite(rv).all()))
print("    task0-vs-task1 rule_vec cosine = %.3f (lower => more task-specific)" % cos)
print("    -> rule_vec CONSUMED by ColorRepairHead via B1 (when flag on). rule_slots -> dead branch.")

print("="*70)
print("[3] ColorRepairHead (B1 on)  inputs: grid_z, base, target_inputs, cond, src_agree, rule_vec")
head = ColorRepairHead(hidden_dim=H, grid_side=30, rule_vec_dim=256)
gz = torch.randn(B, L, H); base = torch.randn(B, L, 10)
nc, gl, rc, ap = head(gz, base, target_inputs, cond, sa, rule_vec=rv)
print("    out new_color", tuple(nc.shape), "gate", tuple(gl.shape), "repair", tuple(rc.shape))
print("    WIRING: head.rule_proj.in_features=%d  ==  PDE rule_vec dim=%d  -> %s"
      % (head.rule_proj.in_features, rv.shape[1], head.rule_proj.in_features == rv.shape[1]))
print("    init no-op: applied.mean=%.4f (expect ~0 => warm-start safe)" % ap.mean().item())
m3 = (target_inputs[0] == 3 + COFF)
print("    VALUE on task0 colour-3 cells -> argmax==7 frac = %.2f (prior should already point right)"
      % (rc[0][m3].argmax(-1) == 7).float().mean().item())
# does the head actually DRIVE the output when the gate opens?
with torch.no_grad(): head.gate_head.bias.fill_(6.0)
nc2, _, _, ap2 = head(gz, base, target_inputs, cond, sa, rule_vec=rv)
diff = (nc2.argmax(-1) != base.argmax(-1)).float().mean().item()
print("    gate-open: output differs from base on frac=%.2f cells, applied.mean=%.2f (proves head CAN reach output)"
      % (diff, ap2.mean().item()))
with torch.no_grad(): head.gate_head.bias.fill_(-6.0)
# does rule_vec gradient actually reach rule_proj (is B1 trainable)?
nc3, gl3, rc3, _ = head(gz, base, target_inputs, cond, sa, rule_vec=rv)
(nc3.pow(2).mean() + gl3.pow(2).mean() + rc3.pow(2).mean()).backward()
g = head.rule_proj.weight.grad
print("    B1 trainable: grad reaches rule_proj = %s (|grad|=%.4f)" % (g is not None and g.abs().sum().item() > 0, 0.0 if g is None else g.abs().sum().item()))
# backward-compat: head WITHOUT rule_vec_dim ignores rule_vec
head0 = ColorRepairHead(hidden_dim=H, grid_side=30)
nc0, _, _, ap0 = head0(gz, base, target_inputs, cond, sa, rule_vec=rv)
print("    prior-only head (no rule_proj): rule_proj is None =", head0.rule_proj is None, "| still no-op applied.mean=%.4f" % ap0.mean().item())
print("="*70)
print("ALL COMPONENT I/O CHECKS COMPLETE")