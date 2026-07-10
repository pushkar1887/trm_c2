"""Standalone check: does the S4 canvas geometry reproduce the generator's output
structure (PAD/EOS/VALID) when fed the TRUE (H,W)? Mirrors np_grid_to_seq_translational_augment
(build_arc_dataset.py) for grid construction and the canvas block in trm_fvr_c2._output_logits
for the recovery+box+EOS logic. No model, no dataset -- pure geometry."""
import numpy as np
import torch

ARC = 30
side = 30


def gen_grid(in_hw, out_hw, seed):
    """One (pad_r,pad_c) for BOTH grids, EOS L-border -- exactly the generator."""
    rng = np.random.RandomState(seed)
    ih, iw = in_hw
    oh, ow = out_hw
    pad_r = rng.randint(0, ARC - max(ih, oh) + 1)
    pad_c = rng.randint(0, ARC - max(iw, ow) + 1)
    grids = []
    for (nrow, ncol) in (in_hw, out_hw):
        g = np.zeros((ARC, ARC), dtype=np.int64)
        g[pad_r:pad_r + nrow, pad_c:pad_c + ncol] = 5            # a colour token (>=2)
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARC:
            g[eos_row, pad_c:eos_col] = 1
        if eos_col < ARC:
            g[pad_r:eos_row, eos_col] = 1
        grids.append(g.flatten())
    return grids[0], grids[1], (pad_r, pad_c)


cases = [(3, 4, 3, 4), (2, 2, 5, 5), (6, 3, 2, 7), (1, 1, 4, 4), (5, 5, 5, 5),
         (4, 4, 1, 1), (7, 2, 7, 9), (2, 8, 8, 2)]
ins, outs, Hs, Ws, origins = [], [], [], [], []
for i, c in enumerate(cases):
    inp, out, org = gen_grid((c[0], c[1]), (c[2], c[3]), seed=i)
    ins.append(inp); outs.append(out); Hs.append(c[2]); Ws.append(c[3]); origins.append(org)

inputs = torch.tensor(np.stack(ins))
outputs = torch.tensor(np.stack(outs))
Hs = torch.tensor(Hs); Ws = torch.tensor(Ws)
B = inputs.shape[0]

# ---- canvas geometry (mirror of trm_fvr_c2._output_logits S4 block), TRUE (H,W) ----
si = inputs.long().view(B, side, side)
is_col = si >= 2
row_has = is_col.any(dim=2)
col_has = is_col.any(dim=1)
r0 = row_has.float().argmax(dim=1).view(-1, 1, 1)
c0 = col_has.float().argmax(dim=1).view(-1, 1, 1)
Hpred = Hs.view(-1, 1, 1)
Wpred = Ws.view(-1, 1, 1)
rr = torch.arange(side).view(1, side, 1)
cc = torch.arange(side).view(1, 1, side)
in_box_r = (rr >= r0) & (rr < r0 + Hpred)
in_box_c = (cc >= c0) & (cc < c0 + Wpred)
out_col = in_box_r & in_box_c
out_eos = ((rr == r0 + Hpred) & in_box_c) | ((cc == c0 + Wpred) & in_box_r)
out_pad = ~(out_col | out_eos)
pred_class = torch.where(out_col, torch.tensor(2), torch.where(out_eos, torch.tensor(1), torch.tensor(0))).reshape(B, -1)

true = outputs.view(B, -1)
true_class = torch.where(true >= 2, torch.tensor(2), torch.where(true == 1, torch.tensor(1), torch.tensor(0)))

# origin recovery check
rec_origin = list(zip(r0.flatten().tolist(), c0.flatten().tolist()))
print("origin true vs recovered:")
for b in range(B):
    ok = tuple(rec_origin[b]) == origins[b]
    print(f"  case {cases[b]}: gen={origins[b]} recovered={rec_origin[b]} {'OK' if ok else 'MISMATCH'}")

match = (pred_class == true_class).all(dim=1)
print("\nper-case structure exact:", match.tolist())
for b in range(B):
    if not match[b]:
        mm = (pred_class[b] != true_class[b]).sum().item()
        print(f"  case {cases[b]} mismatched cells = {mm}")
print("\nALL ORIGINS OK:", all(tuple(rec_origin[b]) == origins[b] for b in range(B)))
print("ALL STRUCTURE EXACT:", bool(match.all()))
