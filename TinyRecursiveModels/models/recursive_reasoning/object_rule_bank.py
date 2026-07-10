"""S2 + S5 substrate: OBJECT-SLOT rule extraction + ANALOGY retrieval.

This is the relational reasoning the histogram (ColorTransitionBank) cannot do, built as a
standalone, no-op-safe, self-tested module BEFORE it touches the live forward (project rule:
"build the check before the feature").

WHY (whole-session diagnosis):
  * The colour lookup cond_inout[a]=P(out|in=a) emits ONE answer per input colour. It CANNOT
    represent a CONDITIONAL recolour -- colour a -> b in a big shape, a -> c as a singleton --
    because the output depends on the cell's OBJECT/RELATIONAL role, not just its colour.
    (key-ranking R3: no per-cell key beats cell-colour; no-invention probe: 97% of conditional
    out-colours are COPIED from an existing element BY RELATION.)
  * So the conditional family (~60% of the benchmark) needs RETRIEVAL/ANALOGY: for each TEST
    object, find its analogous DEMO object and copy that object's OUTPUT colour. Attention is the
    mechanism; it LEARNS which relational role matters per task -- general, not a hand-coded key.

What this module provides:
  extract_object_slots(...)  deterministic, no-grad. Each grid's 4-connected colour components ->
      per-object slots {input-colour 1-hot, size, centroid} (+ the object's modal OUTPUT colour
      and a changed flag when an output grid is given). Same extraction for demos and the test.
  analogy_recolour(...)      deterministic FLOOR. test-object query -> cosine over demo-object
      keys -> retrieved output-colour distribution -> scattered back to cells. Solves conditional
      cases the histogram provably cannot (asserted in the self-test).
  ObjectRuleBank(nn.Module)  learned lift, no-op at init: a relational self-attention encoder over
      demo slots -> a pooled RULE vector for z_H injection (the S3 forcing target, zero-init ->
      contributes nothing until trained), plus a zero-init key refinement on top of the cosine
      retrieval (falls back to the deterministic floor at init).

Token convention (repo-wide): PAD=0, EOS=1, colour = token-2 (0..9). Grid is side x side.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from models.recursive_reasoning.object_bank import connected_components  # noqa: E402

VOCAB = 12
COLOR_OFFSET = 2
N_COLORS = 10
SLOT_FEAT = N_COLORS + 9            # in-colour 1-hot (10) + 9 core-knowledge geometry features:
#   size, centroid_r, centroid_c, n_same_colour, size_rank, solidity, touches_border, top, bottom
# (Chollet core-knowledge priors: counting + topology + gravity, beyond bare objectness/size.)


def _grid_slots(g_in: torch.Tensor, g_out: torch.Tensor | None, side: int, K: int):
    """One grid -> its top-K colour-object slots.

    g_in [B,L] tokens; g_out [B,L] tokens or None.
    Returns:
      feats   [B,K,SLOT_FEAT]  slot key features (in-colour 1-hot, size_norm, centroid r/c)
      in_col  [B,K] long       slot input colour idx (0..9)
      out_col [B,K] long       slot modal OUTPUT colour idx (0..9), -1 if no output / invalid
      valid   [B,K] bool       slot has >=1 colour cell
      labels  [B,L] long       per-cell component label (for cell->slot mapping)
      topk_label [B,K] long    the label id behind each slot
    """
    B, L = g_in.shape
    S = side
    dev = g_in.device
    grid = g_in.long().clamp(0, VOCAB - 1).view(B, S, S)
    labels = connected_components(grid).view(B, L).long()                     # [B,L]
    colour = (g_in.long() - COLOR_OFFSET).clamp(0, N_COLORS - 1)              # [B,L] colour idx
    is_colour = (g_in.long() >= COLOR_OFFSET).float()                        # [B,L] mask

    # component size = number of COLOUR cells per label (PAD/EOS contribute 0 -> ranked last)
    size_per_label = torch.zeros(B, L, device=dev).scatter_add(1, labels, is_colour)   # [B,L]
    topk_size, topk_label = size_per_label.topk(K, dim=1)                     # [B,K]
    valid = topk_size > 0

    # input colour per label (constant within a same-colour component). scatter_reduce(amax) is
    # order-independent -> deterministic on CUDA, unlike scatter_ (overwrite) whose winner among
    # duplicate indices is undefined. amax is exact here: all cells of a mono-colour component share
    # one colour, and colour idx >= 0 so max over the zero-init is that colour.
    col_per_label = torch.zeros(B, L, device=dev).scatter_reduce(
        1, labels, colour.float(), reduce="amax")
    in_col = col_per_label.gather(1, topk_label).round().long().clamp(0, N_COLORS - 1)  # [B,K]

    # centroid (row,col) over colour cells, normalised to [0,1]
    rows = torch.arange(S, device=dev).view(1, S, 1).expand(B, S, S).reshape(B, L).float()
    cols = torch.arange(S, device=dev).view(1, 1, S).expand(B, S, S).reshape(B, L).float()
    rsum = torch.zeros(B, L, device=dev).scatter_add(1, labels, rows * is_colour)
    csum = torch.zeros(B, L, device=dev).scatter_add(1, labels, cols * is_colour)
    sz = size_per_label.clamp_min(1.0)
    cr = (rsum / sz).gather(1, topk_label) / max(S - 1, 1)                    # [B,K]
    cc = (csum / sz).gather(1, topk_label) / max(S - 1, 1)
    size_norm = topk_size / float(L)
    # bbox per component (for solidity + border-touch): label-level reduce, then gather to slots
    rmin = torch.full((B, L), 1e9, device=dev).scatter_reduce(1, labels, rows, reduce='amin', include_self=True)
    rmax = torch.full((B, L), -1e9, device=dev).scatter_reduce(1, labels, rows, reduce='amax', include_self=True)
    cmin = torch.full((B, L), 1e9, device=dev).scatter_reduce(1, labels, cols, reduce='amin', include_self=True)
    cmax = torch.full((B, L), -1e9, device=dev).scatter_reduce(1, labels, cols, reduce='amax', include_self=True)
    s_rmin = rmin.gather(1, topk_label); s_rmax = rmax.gather(1, topk_label)
    s_cmin = cmin.gather(1, topk_label); s_cmax = cmax.gather(1, topk_label)

    # modal OUTPUT colour per label
    if g_out is not None:
        outc = (g_out.long() - COLOR_OFFSET).clamp(0, N_COLORS - 1)           # [B,L]
        out_is_colour = (g_out.long() >= COLOR_OFFSET).float()
        wv = is_colour * out_is_colour                                        # cells coloured in BOTH
        flat = (labels * N_COLORS + outc).clamp(0, L * N_COLORS - 1)          # [B,L]
        out_hist = torch.zeros(B, L * N_COLORS, device=dev).scatter_add(1, flat, wv).view(B, L, N_COLORS)
        out_hist_k = out_hist.gather(1, topk_label.unsqueeze(-1).expand(-1, -1, N_COLORS))  # [B,K,10]
        has_out = out_hist_k.sum(-1) > 0
        out_col = torch.where(has_out, out_hist_k.argmax(-1), torch.full_like(in_col, -1))
    else:
        out_col = torch.full((B, K), -1, dtype=torch.long, device=dev)

    # ---- core-knowledge slot features (Chollet priors): counting / topology / gravity ----
    bbox_h = (s_rmax - s_rmin + 1.0).clamp_min(1.0)
    bbox_w = (s_cmax - s_cmin + 1.0).clamp_min(1.0)
    solidity = (topk_size / (bbox_h * bbox_w)).clamp(0.0, 1.0)                # 1=solid rect (TOPOLOGY)
    big = torch.full_like(rows, 1e9)
    reg_rmin = torch.where(is_colour.bool(), rows, big).min(1, keepdim=True).values   # colour-region bbox
    reg_rmax = torch.where(is_colour.bool(), rows, -big).max(1, keepdim=True).values  # (border = grid edge,
    reg_cmin = torch.where(is_colour.bool(), cols, big).min(1, keepdim=True).values   #  not the 30x30 pad)
    reg_cmax = torch.where(is_colour.bool(), cols, -big).max(1, keepdim=True).values
    touches = ((s_rmin <= reg_rmin) | (s_rmax >= reg_rmax) |
               (s_cmin <= reg_cmin) | (s_cmax >= reg_cmax)).float()          # TOPOLOGY: touches border
    same_col = (in_col.unsqueeze(2) == in_col.unsqueeze(1)) & valid.unsqueeze(1)      # [B,K,K]
    n_same = same_col.sum(-1).float()                                        # COUNTING: # same-colour objects
    n_same_norm = (n_same / 4.0).clamp(0.0, 1.0)
    sz_k = topk_size.unsqueeze(2); sz_j = topk_size.unsqueeze(1)
    size_rank = (((sz_j > sz_k) & same_col).sum(-1).float()
                 / (n_same - 1.0).clamp_min(1.0)).clamp(0.0, 1.0)            # COUNTING: 0=largest..1=smallest
    cr_lo = torch.where(valid, cr, torch.full_like(cr, 1e9))
    cr_hi = torch.where(valid, cr, torch.full_like(cr, -1e9))
    ext_top = (cr <= cr_lo.min(1, keepdim=True).values + 1e-6).float() * valid.float()   # GRAVITY: topmost
    ext_bot = (cr >= cr_hi.max(1, keepdim=True).values - 1e-6).float() * valid.float()   # GRAVITY: bottommost
    in_oh = F.one_hot(in_col, N_COLORS).float()                              # [B,K,10]
    geom = torch.stack([size_norm * 5.0, cr, cc, n_same_norm, size_rank,
                        solidity, touches, ext_top, ext_bot], dim=-1)        # [B,K,9]
    feats = torch.cat([in_oh, geom], dim=-1)                                 # [B,K,SLOT_FEAT]
    return feats, in_col, out_col, valid, labels, topk_label


def _cell_slot_idx(labels: torch.Tensor, topk_label: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[B,L] labels + [B,K] slot label-ids -> [B,L] long: which slot k each cell belongs to (-1 none)."""
    B, L = labels.shape
    K = topk_label.shape[1]
    idx = torch.full((B, L), -1, dtype=torch.long, device=labels.device)
    for k in range(K):
        m = (labels == topk_label[:, k:k + 1]) & valid[:, k:k + 1]
        idx = torch.where(m, torch.full_like(idx, k), idx)
    return idx


def extract_object_slots(context_inputs: torch.Tensor, context_outputs: torch.Tensor | None,
                         context_mask: torch.Tensor | None, side: int, slots_per_grid: int = 6):
    """[B,M,L] demos -> stacked DEMO object slots across all demos.

    Returns dict with:
      feats   [B, M*K, SLOT_FEAT]   demo slot key features
      in_col  [B, M*K] long
      out_col [B, M*K] long  (-1 if the slot has no output colour)
      valid   [B, M*K] bool  (slot has colour cells AND its demo is unmasked)
      changed [B, M*K] bool  (out_col != in_col, both valid)
    """
    B, M, L = context_inputs.shape
    K = slots_per_grid
    feats_l, inc_l, outc_l, val_l = [], [], [], []
    for m in range(M):
        g_in = context_inputs[:, m]
        g_out = context_outputs[:, m] if context_outputs is not None else None
        feats, in_col, out_col, valid, _labels, _tl = _grid_slots(g_in, g_out, side, K)
        if context_mask is not None:
            valid = valid & context_mask[:, m].bool().unsqueeze(-1)
        feats_l.append(feats); inc_l.append(in_col); outc_l.append(out_col); val_l.append(valid)
    feats = torch.cat(feats_l, dim=1)                                         # [B, M*K, F]
    in_col = torch.cat(inc_l, dim=1)
    out_col = torch.cat(outc_l, dim=1)
    valid = torch.cat(val_l, dim=1)
    changed = valid & (out_col >= 0) & (out_col != in_col)
    return {"feats": feats, "in_col": in_col, "out_col": out_col, "valid": valid, "changed": changed}


def extract_target_slots(target_input: torch.Tensor, side: int, slots_per_grid: int = 6):
    """[B,L] test input -> its object slots + the per-cell slot index (for scatter-back).

    Returns dict: feats [B,K,F], in_col [B,K], valid [B,K], cell_idx [B,L] (-1 = no slot).
    """
    feats, in_col, _out, valid, labels, topk_label = _grid_slots(target_input, None, side, slots_per_grid)
    cell_idx = _cell_slot_idx(labels, topk_label, valid)
    return {"feats": feats, "in_col": in_col, "valid": valid, "cell_idx": cell_idx}


def analogy_recolour(demo_feats: torch.Tensor, demo_out_col: torch.Tensor, demo_valid: torch.Tensor,
                     test_feats: torch.Tensor, test_cell_idx: torch.Tensor,
                     temperature: float = 0.3):
    """Deterministic analogy FLOOR.

    For each test object: cosine-similarity query over the demo objects, softmax-attend, and copy
    the attended demo objects' OUTPUT colours. Scatter the per-object answer back to cells.

    Returns:
      cell_prob [B,L,10]  retrieved output-colour distribution per cell (0 where no slot)
      cell_conf [B,L]     retrieval confidence (peak of the distribution; 0 where no slot)
    """
    B, L = test_cell_idx.shape
    dev = test_feats.device
    # Colour is a HARD gate (only copy from a same-input-colour object -> no invention); geometry
    # (size, centroid) is the SOFT match WITHIN that colour. A plain cosine over the full feature
    # let the 10-d colour one-hot drown the 3-d geometry, so a big and a small object of the same
    # colour looked identical -> the conditional split collapsed. Split them explicitly.
    dcol = demo_feats[..., :N_COLORS].argmax(-1)                             # [B,Md] input colour
    tcol = test_feats[..., :N_COLORS].argmax(-1)                             # [B,Kt]
    dgeo = F.normalize(demo_feats[..., N_COLORS:], dim=-1)                   # [B,Md,9] core-knowledge geom
    tgeo = F.normalize(test_feats[..., N_COLORS:], dim=-1)                   # [B,Kt,9]
    sim = torch.bmm(tgeo, dgeo.transpose(1, 2)) / max(temperature, 1e-4)     # [B,Kt,Md] geometry cosine
    usable = demo_valid & (demo_out_col >= 0)                                 # [B,Md]
    same_col = tcol.unsqueeze(2) == dcol.unsqueeze(1)                        # [B,Kt,Md]
    sim = sim.masked_fill(~(same_col & usable.unsqueeze(1)), -1e9)
    attn = sim.softmax(dim=-1)                                                # [B,Kt,Md]
    out_oh = F.one_hot(demo_out_col.clamp(0, N_COLORS - 1), N_COLORS).float()
    out_oh = out_oh * usable.unsqueeze(-1).float()                            # zero invalid slots
    retr = torch.bmm(attn, out_oh)                                            # [B,Kt,10] prob
    
    # prevent uniform hallucination when no demo object matches the test object's colour
    has_match = (same_col & usable.unsqueeze(1)).any(dim=-1)                 # [B,Kt]
    retr = retr * has_match.unsqueeze(-1).float()
    
    conf = retr.max(dim=-1).values                                           # [B,Kt]

    valid_cell = (test_cell_idx >= 0)
    safe = test_cell_idx.clamp_min(0)
    gathered = retr.gather(1, safe.unsqueeze(-1).expand(-1, -1, N_COLORS))    # [B,L,10]
    cell_prob = torch.where(valid_cell.unsqueeze(-1), gathered,
                            torch.zeros(B, L, N_COLORS, device=dev))
    cell_conf = torch.where(valid_cell, conf.gather(1, safe), torch.zeros(B, L, device=dev))
    return cell_prob, cell_conf


class ObjectRuleBank(nn.Module):
    """Learned lift over the deterministic slot substrate. No-op at init (warm-start safe).

    * encode_rule: relational self-attention over demo slots -> pooled RULE vector for z_H
      injection (the S3 forcing target). rule_proj is zero-init -> rule_vec == 0 at init.
    * recolour: deterministic analogy retrieval, with a ZERO-INIT learned refinement added to the
      slot key features -> at init the retrieval is exactly the cosine floor.
    """

    def __init__(self, rule_dim: int, d_model: int = 128, n_heads: int = 4,
                 slots_per_grid: int = 6, temperature: float = 0.3):
        super().__init__()
        self.K = int(slots_per_grid)
        self.temperature = float(temperature)
        # demo-side embedding sees the OUTPUT colour too (the slot's "answer"); test-side does not.
        self.demo_embed = nn.Linear(SLOT_FEAT + N_COLORS, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        self.rule_proj = nn.Linear(d_model, rule_dim)
        self.key_refine = nn.Linear(SLOT_FEAT, SLOT_FEAT)
        with torch.no_grad():
            self.rule_proj.weight.zero_(); self.rule_proj.bias.zero_()        # rule_vec = 0 at init
            self.key_refine.weight.zero_(); self.key_refine.bias.zero_()      # retrieval = cosine floor at init

    def encode_rule(self, demo) -> torch.Tensor:
        """demo dict -> [B, rule_dim] pooled cross-demo rule vector (zero at init)."""
        out_oh = F.one_hot(demo["out_col"].clamp(0, N_COLORS - 1), N_COLORS).float()
        out_oh = out_oh * (demo["out_col"] >= 0).unsqueeze(-1).float()
        tok = torch.cat([demo["feats"], out_oh], dim=-1)                      # [B, Md, F+10]
        h = self.demo_embed(tok)
        pad = ~demo["valid"]                                                 # [B,Md] True=ignore
        pad = pad.masked_fill(pad.all(dim=1, keepdim=True), False)           # prevent NaN when all demos are masked out
        attn_out, _ = self.attn(h, h, h, key_padding_mask=pad, need_weights=False)
        h = self.attn_norm(h + attn_out)
        w = demo["valid"].float().unsqueeze(-1)
        pooled = (h * w).sum(1) / w.sum(1).clamp_min(1.0)                     # [B, d_model]
        return self.rule_proj(pooled)                                        # [B, rule_dim]

    def recolour(self, demo, target):
        """demo + target dicts -> (cell_prob [B,L,10], cell_conf [B,L]) analogy VALUE."""
        demo_feats = demo["feats"] + self.key_refine(demo["feats"])          # +0 at init
        test_feats = target["feats"] + self.key_refine(target["feats"])
        return analogy_recolour(demo_feats, demo["out_col"], demo["valid"],
                                test_feats, target["cell_idx"], self.temperature)

    def forward(self, context_inputs, context_outputs, context_mask, target_input, side):
        demo = extract_object_slots(context_inputs, context_outputs, context_mask, side, self.K)
        target = extract_target_slots(target_input, side, self.K)
        rule_vec = self.encode_rule(demo)
        cell_prob, cell_conf = self.recolour(demo, target)
        return {"rule_vec": rule_vec, "recolour_prob": cell_prob, "recolour_conf": cell_conf}


# ------------------------------------------------------- S6: RULE-HYPOTHESIS inference (offline prior)
# The narrowing prior for the rule-hypothesis bus. Emits a RANKED list of operation hypotheses from the
# support demos -- NOT a solved grid (proposal-as-rule-hypothesis, not proposal-as-answer). Each
# hypothesis carries a cell-resolvable binding (translate vector / slide direction / sort axis) so a
# later live bus can pair it with the per-cell relmap (slide_right (x) dist_to_right_wall). Coarse family
# routing is trivial (the atlas categories are invariant-defined: shape_change? histogram_preserved?);
# the LOAD-BEARING output is the within-rearrange binding + its cross-demo consistency -- that is what
# makes a rule token actionable rather than a label the TRM could already infer from hist-preservation.
# Pure deterministic, no-grad. Standalone (no model import), self-tested below.
RULE_FAMILIES = ("size_change", "recolor", "rearrange", "identity")


def _compact_colour(grid_flat: torch.Tensor, side: int):
    """flat tokens [L] -> (compact colour grid [H,W] long in 0..9, (H,W)) or (None,None) if empty.

    Bbox = the tight colour extent (PAD=0/EOS=1 stripped). Non-colour cells inside the bbox map to -1."""
    g = grid_flat.long().view(side, side)
    isc = g >= COLOR_OFFSET
    if not bool(isc.any()):
        return None, None
    nz = isc.nonzero(as_tuple=False)
    h = int(nz[:, 0].max()) + 1
    w = int(nz[:, 1].max()) + 1
    sub = g[:h, :w]
    col = (sub - COLOR_OFFSET).clamp(0, N_COLORS - 1)
    col = torch.where(sub >= COLOR_OFFSET, col, torch.full_like(col, -1))
    return col, (h, w)


def _hist10(col: torch.Tensor) -> tuple:
    """compact colour grid -> 10-bin colour histogram (tuple, hashable). -1 cells excluded."""
    flat = col.reshape(-1)
    flat = flat[flat >= 0]
    return tuple(torch.bincount(flat, minlength=N_COLORS).tolist())


def _modal_colour(col: torch.Tensor) -> int:
    h = _hist10(col)
    return int(max(range(N_COLORS), key=lambda c: h[c]))


def _background(col: torch.Tensor) -> int:
    """Background = the colour that is BOTH on the outer frame AND most frequent overall (canvas + frame).
    Robust and CONSISTENT across a task's demos: pure border-modal flips between demos when the frame
    colour changes; pure global-modal flags the foreground in dense grids. The intersection is stable."""
    H, W = col.shape
    border = set()
    for cc in range(W):
        for rr in (0, H - 1):
            v = int(col[rr, cc])
            if v >= 0:
                border.add(v)
    for rr in range(H):
        for cc in (0, W - 1):
            v = int(col[rr, cc])
            if v >= 0:
                border.add(v)
    if not border:
        return _modal_colour(col)
    hist = _hist10(col)
    return max(border, key=lambda c: hist[c] if 0 <= c < N_COLORS else -1)


def _objects(col: torch.Tensor, bg: int, multi: bool = False) -> list:
    """compact colour grid -> objects (4-connected, != bg). multi=False groups SAME-colour cells;
    multi=True groups ANY adjacent non-bg cells so a MULTI-COLOUR stamp is ONE object (many ARC
    'objects' are multi-colour). Each object carries cellcol {(r,c):colour} so renders keep exact
    per-cell colours. 'colour' is the scalar colour (mono) or the colour-set (multi) used as match key."""
    cg = col.tolist()
    H = len(cg); W = len(cg[0]) if H else 0
    seen = [[False] * W for _ in range(H)]
    objs = []
    for r in range(H):
        for c in range(W):
            v = cg[r][c]
            if v < 0 or v == bg or seen[r][c]:
                continue
            stack = [(r, c)]; seen[r][c] = True; cells = []
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < H and 0 <= nx < W and not seen[ny][nx]
                            and cg[ny][nx] >= 0 and cg[ny][nx] != bg
                            and (multi or cg[ny][nx] == v)):
                        seen[ny][nx] = True; stack.append((ny, nx))
            ys = [y for y, _ in cells]; xs = [x for _, x in cells]
            n = len(cells)
            cellcol = {(y, x): cg[y][x] for (y, x) in cells}
            cols = frozenset(cellcol.values())
            key_colour = v if (not multi or len(cols) == 1) else cols
            objs.append({"colour": key_colour, "cols": cols, "cellcol": cellcol,
                         "size": n, "cr": sum(ys) / n, "cc": sum(xs) / n,
                         "rmin": min(ys), "rmax": max(ys), "cmin": min(xs), "cmax": max(xs),
                         "cells": frozenset(cells)})
    return objs


# ------------------------------------------------------- core-knowledge per-object attributes (priors)
def _d4_canon(obj) -> tuple:
    """D4-canonical (rotation/reflection-invariant) signature of an object's cell mask. Same shape ->
    same signature regardless of orientation -> robust object correspondence + a 'shape-type' selector."""
    r0, c0 = obj["rmin"], obj["cmin"]
    H, W = obj["rmax"] - r0, obj["cmax"] - c0
    pts = [(r - r0, c - c0) for (r, c) in obj["cells"]]
    forms = []
    for t in range(8):
        tp = []
        for (r, c) in pts:
            if t == 0: y, x = r, c
            elif t == 1: y, x = c, H - r
            elif t == 2: y, x = H - r, W - c
            elif t == 3: y, x = W - c, r
            elif t == 4: y, x = r, W - c
            elif t == 5: y, x = H - r, c
            elif t == 6: y, x = c, r
            else: y, x = W - c, H - r
            tp.append((y, x))
        my = min(y for y, x in tp); mx = min(x for y, x in tp)
        forms.append(tuple(sorted((y - my, x - mx) for y, x in tp)))
    return min(forms)


def _hole_count(obj) -> int:
    """Number of background regions fully enclosed by the object (topology prior: holes/breaks)."""
    r0, c0, r1, c1 = obj["rmin"], obj["cmin"], obj["rmax"], obj["cmax"]
    H, W = r1 - r0 + 1, c1 - c0 + 1
    occ = [[False] * W for _ in range(H)]
    for (r, c) in obj["cells"]:
        occ[r - r0][c - c0] = True
    seen = [[False] * W for _ in range(H)]
    stack = []
    for r in range(H):
        for c in (0, W - 1):
            if not occ[r][c] and not seen[r][c]:
                seen[r][c] = True; stack.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if not occ[r][c] and not seen[r][c]:
                seen[r][c] = True; stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not occ[nr][nc] and not seen[nr][nc]:
                seen[nr][nc] = True; stack.append((nr, nc))
    holes = 0
    for r in range(H):
        for c in range(W):
            if not occ[r][c] and not seen[r][c]:
                holes += 1
                seen[r][c] = True; st = [(r, c)]
                while st:
                    y, x = st.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dr, x + dc
                        if 0 <= ny < H and 0 <= nx < W and not occ[ny][nx] and not seen[ny][nx]:
                            seen[ny][nx] = True; st.append((ny, nx))
    return holes


def _match_by_shape(oi: list, oo: list) -> list:
    """Pair input->output objects by (colour, size, D4 shape-signature), nearest-centroid tiebreak.
    Robust to MOVEMENT (position-free) -> fixes the colour+size-only matcher's failures."""
    sig = {id(o): _d4_canon(o) for o in oi + oo}
    pairs = []; used = [False] * len(oo)
    for a in oi:
        best = -1; bestd = 1e18
        for j, b in enumerate(oo):
            if used[j] or b["colour"] != a["colour"] or b["size"] != a["size"] or sig[id(b)] != sig[id(a)]:
                continue
            d = abs(a["cr"] - b["cr"]) + abs(a["cc"] - b["cc"])
            if d < bestd:
                bestd = d; best = j
        if best >= 0:
            used[best] = True; pairs.append((a, oo[best]))
    return pairs


def _blocked_in_dir(col_out: torch.Tensor, obj_cells: frozenset, d: str, bg: int) -> bool:
    """Is the OUTPUT object flush against a boundary (grid edge OR another object) in direction d?
    True iff every leading-face cell has no free background cell one step ahead in d."""
    H, W = col_out.shape
    dr, dc = {"down": (1, 0), "up": (-1, 0), "right": (0, 1), "left": (0, -1)}[d]
    cg = col_out
    for (r, c) in obj_cells:
        nr, nc = r + dr, c + dc
        if (nr, nc) in obj_cells:
            continue
        if 0 <= nr < H and 0 <= nc < W and int(cg[nr, nc]) == bg:
            return False
    return True


def boundary_move_eval(col_in: torch.Tensor, col_out: torch.Tensor):
    """Test the MOVE-TO-BOUNDARY hypothesis on one demo. Match objects by shape, and for each moved
    object check it slid along one axis and ends flush to a boundary (edge or another object).

    Returns dict: matched, n_obj, at_boundary (count flush), dirs (per-object direction or None),
    nearest_edge_ok (count whose direction == its input nearest-edge: the 'gravity to nearest wall' selector)."""
    bg = _background(col_in)
    oi = _objects(col_in, bg); oo = _objects(col_out, bg)
    pairs = _match_by_shape(oi, oo)
    H, W = col_in.shape
    at_b = 0; dirs = []; near_ok = 0
    for a, b in pairs:
        dr = int(round(b["cr"] - a["cr"])); dc = int(round(b["cc"] - a["cc"]))
        if dr == 0 and dc == 0:
            dirs.append("stay"); continue
        if dr != 0 and dc != 0:
            dirs.append(None); continue          # diagonal -> not a pure boundary slide
        d = ("down" if dr > 0 else "up") if dc == 0 else ("right" if dc > 0 else "left")
        dirs.append(d)
        if _blocked_in_dir(col_out, b["cells"], d, bg):
            at_b += 1
        # nearest-edge selector: did it move toward its closest grid edge?
        edged = {"up": a["rmin"], "down": H - 1 - a["rmax"], "left": a["cmin"], "right": W - 1 - a["cmax"]}
        if d == min(edged, key=edged.get):
            near_ok += 1
    return {"matched": len(pairs), "n_obj": len(oi), "at_boundary": at_b,
            "dirs": dirs, "nearest_edge_ok": near_ok}


# ------------------------------------------------- analogy_relocate: the position-analogy solver (Lane A)
# Generalises analogy_recolour (copy the analogous demo object's OUTPUT COLOUR) to position: copy the
# analogous demo object's OUTPUT POSITION expressed in a relation-FRAME. A recipe is (frame x key);
# the solver PROPOSES recipes, VERIFIES which reconstructs every demo exactly, and applies the verified
# one to the test input. General by construction -- a small frame library + attribute keys, not 20 rules.
def _compact_to_flat_tokens(col: torch.Tensor, side: int) -> torch.Tensor:
    """compact colour grid [H,W] (0..9) -> flat [side*side] tokens (colour+2, PAD=0), top-left placed."""
    out = torch.zeros(side * side, dtype=torch.long)
    H, W = col.shape
    canvas = torch.zeros(side, side, dtype=torch.long)
    h = min(H, side); w = min(W, side)
    sub = (col[:h, :w].long() + COLOR_OFFSET)
    sub = torch.where(col[:h, :w] >= 0, sub, torch.zeros_like(sub))
    canvas[:h, :w] = sub
    return canvas.reshape(-1)


def _key_val(obj, keyname: str):
    if keyname == "colour": return obj["colour"]
    if keyname == "size": return obj["size"]
    if keyname == "shape": return _d4_canon(obj)
    if keyname == "holes": return _hole_count(obj)
    return None


def _render(objects: list, shape, bg: int, place_cell) -> torch.Tensor:
    """Render objects onto a bg canvas. place_cell(obj, r, c) -> target (r,c). Per-cell colour from
    cellcol, so multi-colour objects move as a rigid body and keep their internal colours."""
    H, W = shape
    canvas = torch.full((H, W), bg, dtype=torch.long)
    for o in objects:
        for (r, c) in o["cells"]:
            tr, tc = place_cell(o, r, c)
            if 0 <= tr < H and 0 <= tc < W:
                canvas[tr, tc] = o["cellcol"][(r, c)]
    return canvas


_DIRS = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}


def _axis_dir(dr: int, dc: int):
    """Displacement -> axis-aligned direction name (N/S/E/W), 'stay', or None (diagonal)."""
    if dr == 0 and dc == 0:
        return "stay"
    if dr == 0:
        return "E" if dc > 0 else "W"
    if dc == 0:
        return "S" if dr > 0 else "N"
    return None


def _slide_shifts(objects: list, shape, dir_vec) -> dict:
    """THE single move-to-boundary/gravity function. Slide every object along dir_vec (DATA, not four
    code paths) until blocked by an edge or an already-placed object. Leading objects placed first so
    the rest pack behind them. Returns {id(obj): (dr_total, dc_total)}."""
    H, W = shape
    dr, dc = dir_vec
    occ = torch.zeros(H, W, dtype=torch.bool)
    shifts = {}
    for o in sorted(objects, key=lambda o: -max(dr * r + dc * c for (r, c) in o["cells"])):
        cells = list(o["cells"]); s = 0
        while True:
            nxt = [(r + dr * (s + 1), c + dc * (s + 1)) for (r, c) in cells]
            if any(not (0 <= r < H and 0 <= c < W) for r, c in nxt):
                break
            if any(bool(occ[r, c]) for r, c in nxt):
                break
            s += 1
        for (r, c) in cells:
            occ[r + dr * s, c + dc * s] = True
        shifts[id(o)] = (dr * s, dc * s)
    return shifts


def _slide_all(objs, shape, dir_of, frozen=None):
    """Unified obstacle-aware slide. Each object moves in dir_of(o) (a per-object (dr,dc), DATA) until
    blocked by the grid edge, a FROZEN cell (a stationary anchor), or an already-placed object. Leading
    objects (furthest along their own direction) are placed first so the rest pack behind them. Returns
    {id(obj): (dr,dc)}. This is the single slide used by gravity / nearest-edge / by-key / anchor / snap."""
    H, W = shape
    occ = set(frozen or ())
    out = {}

    def proj(o):
        dr, dc = dir_of(o)
        return max(dr * r + dc * c for (r, c) in o["cells"]) if (dr, dc) != (0, 0) else -1e18
    for o in sorted(objs, key=lambda o: -proj(o)):
        dr, dc = dir_of(o)
        cells = list(o["cells"]); s = 0
        if (dr, dc) != (0, 0):
            while True:
                nxt = [(r + dr * (s + 1), c + dc * (s + 1)) for (r, c) in cells]
                if any(not (0 <= r < H and 0 <= c < W) for r, c in nxt):
                    break
                if any((r, c) in occ for r, c in nxt):
                    break
                s += 1
        for (r, c) in cells:
            occ.add((r + dr * s, c + dc * s))
        out[id(o)] = (dr * s, dc * s)
    return out


def _frozen_cells(objs, keep_ids):
    """Cells of the NON-mover (anchor) objects -- static obstacles the movers must respect."""
    fr = set()
    for o in objs:
        if id(o) not in keep_ids:
            fr |= set(o["cells"])
    return fr


def _break_count(o) -> int:
    """# of 4-connected SAME-colour sub-components inside the object (1 for a solid mono object)."""
    cells = o["cells"]; cellcol = o["cellcol"]
    seen = set(); n = 0
    for start in cells:
        if start in seen:
            continue
        n += 1; col = cellcol[start]; st = [start]; seen.add(start)
        while st:
            y, x = st.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (y + dy, x + dx)
                if nb in cells and nb not in seen and cellcol[nb] == col:
                    seen.add(nb); st.append(nb)
    return n


def _symmetry(o):
    """(sym_h, sym_v): is the object's coloured bbox mask mirror-equal across its vertical / horizontal axis?"""
    r0, c0 = o["rmin"], o["cmin"]; H = o["rmax"] - r0 + 1; W = o["cmax"] - c0 + 1
    grid = [[-1] * W for _ in range(H)]
    for (r, c), v in o["cellcol"].items():
        grid[r - r0][c - c0] = v
    sym_h = all(grid[r][c] == grid[r][W - 1 - c] for r in range(H) for c in range(W))
    sym_v = all(grid[r][c] == grid[H - 1 - r][c] for r in range(H) for c in range(W))
    return sym_h, sym_v


def _ray_to_object(o, idx: int, occ: dict, shape) -> dict:
    """For each of N/S/E/W: displacement to slide o until its leading face touches ANOTHER object's
    (static) cell -> (dr,dc); None if the ray reaches the grid edge without hitting an object (so a
    direction with no object is not confused with an edge-stop). The 'distance to nearest object' prior."""
    H, W = shape; out = {}
    for name, (dr, dc) in _DIRS.items():
        cells = list(o["cells"]); s = 0; hit = False
        while True:
            nxt = [(r + dr * (s + 1), c + dc * (s + 1)) for (r, c) in cells]
            if any(not (0 <= r < H and 0 <= c < W) for r, c in nxt):
                break                                   # edge, no object this way
            if any(occ.get((r, c), idx) != idx for (r, c) in nxt):
                hit = True; break                       # an object is ahead
            s += 1
        out[name] = (dr * s, dc * s) if hit else None
    return out


def object_kinematics(objects: list, shape) -> list:
    """Attach the GENERALIZED priors to each object ONCE, all directions: wall_vectors (to each edge),
    obj_vectors (ray-cast to nearest other object), holes, breaks, sym_h, sym_v. Shared substrate for
    the Lane-A resolvers and (later) the Lane-B per-cell relmap channels -- one computation, two consumers."""
    H, W = shape
    occ = {}
    for i, o in enumerate(objects):
        for cell in o["cells"]:
            occ[cell] = i
    for i, o in enumerate(objects):
        o["wall_vectors"] = {"N": (-o["rmin"], 0), "S": ((H - 1) - o["rmax"], 0),
                             "W": (0, -o["cmin"]), "E": (0, (W - 1) - o["cmax"])}
        o["obj_vectors"] = _ray_to_object(o, i, occ, shape)
        o["holes"] = _hole_count(o)
        o["breaks"] = _break_count(o)
        o["sym_h"], o["sym_v"] = _symmetry(o)
    return objects


def _grid_objs(col: torch.Tensor, multi: bool):
    bg = _background(col)
    return bg, object_kinematics(_objects(col, bg, multi), col.shape)


def _attr(o, name):
    if name == "holes": return o["holes"]
    if name == "breaks": return o["breaks"]
    if name == "size": return o["size"]
    if name == "colour":
        cv = o["colour"]
        return min(cv) if isinstance(cv, frozenset) else cv
    return 0


def _shift_render(col, multi, shift_of):
    """Common renderer: extract+enrich objects, get a per-object (dr,dc) shift dict, render. None-safe."""
    bg, objs = _grid_objs(col, multi)
    sh = shift_of(objs, col.shape)
    if sh is None:
        return None
    return _render(objs, col.shape, bg, lambda o, r, c: (r + sh[id(o)][0], c + sh[id(o)][1]))


# ---- RESOLVERS (Strategy registry): each is (fit, apply). fit(demos,spec,multi)->params|None ----
def _fit_none(demos, spec, multi):
    return {}


def _fit_translate(demos, spec, multi):
    disps = set()
    for ci, co in demos:
        bg = _background(ci); oi = _objects(ci, bg, multi); oo = _objects(co, bg, multi)
        pairs = _match_by_shape(oi, oo)
        if not oi or len(pairs) < len(oi):
            return None
        for a, b in pairs:
            disps.add((int(round(b["cr"] - a["cr"])), int(round(b["cc"] - a["cc"]))))
    return {"disp": next(iter(disps))} if len(disps) == 1 and disps != {(0, 0)} else None


def _apply_translate(col, spec, params, multi):
    dr, dc = params["disp"]
    return _shift_render(col, multi, lambda objs, shp: {id(o): (dr, dc) for o in objs})


def _fit_displace(demos, spec, multi):
    sel = spec["selector"]
    if sel[0] in ("fixed", "nearest_edge"):
        return {}
    if sel[0] == "by_key":
        mp = {}
        for ci, co in demos:
            bg = _background(ci); oi = _objects(ci, bg, multi); oo = _objects(co, bg, multi)
            pairs = _match_by_shape(oi, oo)
            if not oi or len(pairs) < len(oi):
                return None
            for a, b in pairs:
                d = _axis_dir(int(round(b["cr"] - a["cr"])), int(round(b["cc"] - a["cc"])))
                if d not in _DIRS:
                    return None
                kv = _key_val(a, sel[1])
                if kv in mp and mp[kv] != d:
                    return None
                mp[kv] = d
        return {"map": mp} if mp else None
    return None


def _colour_scalar(o):
    cv = o["colour"]
    return None if isinstance(cv, frozenset) else cv


def _mover_ids(objs, kind):
    """Decouple SELECTOR from ACTION: which objects are the movers (the rest are stationary anchors)."""
    if kind in (None, "all"):
        return {id(o) for o in objs}
    if kind == "singleton":
        return {id(o) for o in objs if o["size"] == 1}
    if kind == "minority_colour":
        import collections as _c
        cnt = _c.Counter(_colour_scalar(o) for o in objs)
        cnt.pop(None, None)
        if not cnt:
            return set()
        mn = min(cnt.values())
        keep = {c for c, n in cnt.items() if n == mn}
        return {id(o) for o in objs if _colour_scalar(o) in keep}
    return {id(o) for o in objs}


def _cell_pseudo_objects(col, bg, colour_ids):
    """One size-1 pseudo-object per cell of the given colours -> CELLULAR (sand) gravity vs rigid-object."""
    H, W = col.shape; objs = []
    for r in range(H):
        for c in range(W):
            v = int(col[r, c])
            if v >= 0 and v != bg and v in colour_ids:
                objs.append({"colour": v, "cols": frozenset([v]), "cellcol": {(r, c): v},
                             "size": 1, "cr": float(r), "cc": float(c),
                             "rmin": r, "rmax": r, "cmin": c, "cmax": c, "cells": frozenset([(r, c)])})
    return objs


def _apply_cellular(col, spec, params, multi):
    sel = spec["selector"]; movers = spec.get("movers", "all")
    bg = _background(col); H, W = col.shape
    allobjs = _objects(col, bg, multi); keep = _mover_ids(allobjs, movers)
    mover_cols = {_colour_scalar(o) for o in allobjs if id(o) in keep} - {None}
    mcells = object_kinematics(_cell_pseudo_objects(col, bg, mover_cols), col.shape)
    frozen = set()
    for r in range(H):
        for c in range(W):
            v = int(col[r, c])
            if v >= 0 and v != bg and v not in mover_cols:
                frozen.add((r, c))

    def dir_of(o):
        if sel[0] == "fixed":
            return _DIRS[sel[1]]
        if sel[0] == "nearest_edge":
            wv = o["wall_vectors"]
            return _DIRS[min(wv, key=lambda k: abs(wv[k][0]) + abs(wv[k][1]))]
        return (0, 0)
    sh = _slide_all(mcells, col.shape, dir_of, frozen)
    out = torch.full((H, W), bg, dtype=torch.long)
    for (r, c) in frozen:
        out[r, c] = int(col[r, c])
    for o in mcells:
        (r, c), = tuple(o["cells"]); dr, dc = sh[id(o)]
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            out[nr, nc] = o["cellcol"][(r, c)]
    return out


def _apply_displace(col, spec, params, multi):
    if spec.get("cellular"):
        return _apply_cellular(col, spec, params, multi)
    sel = spec["selector"]; movers = spec.get("movers", "all")

    def shift_of(objs, shp):
        keep = _mover_ids(objs, movers)
        movers_l = [o for o in objs if id(o) in keep]
        frozen = _frozen_cells(objs, keep)                 # non-movers are static obstacles

        def dir_of(o):
            if sel[0] == "fixed":
                return _DIRS[sel[1]]
            if sel[0] == "nearest_edge":
                wv = o["wall_vectors"]
                return _DIRS[min(wv, key=lambda k: abs(wv[k][0]) + abs(wv[k][1]))]
            if sel[0] == "by_key":
                d = params["map"].get(_key_val(o, sel[1]))
                return _DIRS[d] if d in _DIRS else (0, 0)
            return (0, 0)
        base = _slide_all(movers_l, shp, dir_of, frozen)
        return {id(o): base.get(id(o), (0, 0)) for o in objs}
    return _shift_render(col, multi, shift_of)


def _apply_to_object(col, spec, params, multi):
    def shift_of(objs, shp):
        if not objs:
            return {}
        mx = max(o["size"] for o in objs)
        anchors = [o for o in objs if o["size"] == mx]         # largest objects are the targets
        movers = [o for o in objs if o["size"] < mx]
        if not anchors or not movers:
            return {id(o): (0, 0) for o in objs}
        frozen = set().union(*[set(a["cells"]) for a in anchors])

        def dir_of(o):
            best = None; bd = 1e18
            for (ar, ac) in frozen:
                d = abs(o["cr"] - ar) + abs(o["cc"] - ac)
                if d < bd:
                    bd = d; best = (ar, ac)
            if best is None:
                return (0, 0)
            ddr = best[0] - o["cr"]; ddc = best[1] - o["cc"]
            return (1 if ddr > 0 else -1, 0) if abs(ddr) >= abs(ddc) else (0, 1 if ddc > 0 else -1)
        base = _slide_all(movers, shp, dir_of, frozen)         # snap each mover adjacent to its anchor
        return {id(o): base.get(id(o), (0, 0)) for o in objs}
    return _shift_render(col, multi, shift_of)


def _fit_absolute(demos, spec, multi):
    axis = spec["axis"]; key = spec["key"]; mp = {}
    for ci, co in demos:
        bg = _background(ci); oi = _objects(ci, bg, multi); oo = _objects(co, bg, multi)
        pairs = _match_by_shape(oi, oo)
        if not oi or len(pairs) < len(oi):
            return None
        for a, b in pairs:
            kv = _key_val(a, key); tgt = b["cmin"] if axis == "col" else b["rmin"]
            if kv in mp and mp[kv] != tgt:
                return None
            mp[kv] = tgt
    return {"map": mp} if mp else None


def _apply_absolute(col, spec, params, multi):
    mp = params["map"]; axis = spec["axis"]; key = spec["key"]

    def shift_of(objs, shp):
        out = {}
        for o in objs:
            kv = _key_val(o, key)
            if kv not in mp:
                out[id(o)] = (0, 0)
            elif axis == "col":
                out[id(o)] = (0, mp[kv] - o["cmin"])
            else:
                out[id(o)] = (mp[kv] - o["rmin"], 0)
        return out
    return _shift_render(col, multi, shift_of)


def _apply_reflect(col, spec, params, multi):
    bg, objs = _grid_objs(col, multi); H, W = col.shape; ax = spec["axis"]
    if ax == "h":
        pc = lambda o, r, c: (r, W - 1 - c)
    elif ax == "v":
        pc = lambda o, r, c: (H - 1 - r, c)
    else:
        pc = lambda o, r, c: (H - 1 - r, W - 1 - c)
    return _render(objs, col.shape, bg, pc)


def _fit_sort(demos, spec, multi):
    for ci, co in demos:
        bg = _background(ci)
        if len(_objects(ci, bg, multi)) < 2:
            return None
    return {}


def _apply_sort(col, spec, params, multi):
    axis = spec["axis"]; attr = spec["attr"]; desc = spec.get("desc", False)
    sign = -1 if desc else 1

    def shift_of(objs, shp):
        if len(objs) < 2:
            return None
        # target slots = the objects' own anchor positions along the axis, deterministically ordered
        slots = sorted(objs, key=lambda o: (o["cmin"] if axis == "col" else o["rmin"], o["rmin"], o["cmin"]))
        # rank by the attribute; ties broken by original spatial coordinates so the sort is STABLE/exact
        ranked = sorted(objs, key=lambda o: (sign * _attr(o, attr), o["rmin"], o["cmin"]))
        out = {}
        for slot, o in zip(slots, ranked):
            out[id(o)] = (0, slot["cmin"] - o["cmin"]) if axis == "col" else (slot["rmin"] - o["rmin"], 0)
        return out
    return _shift_render(col, multi, shift_of)


def _fit_anchor(demos, spec, multi):
    """Find a colour A that is the ANCHOR: A-objects stay put while every other object moves toward A.
    The 'boundary' is then an object (a line/bar), and its side may vary per demo -- captured dynamically."""
    common = None
    for ci, co in demos:
        bg = _background(ci); oi = _objects(ci, bg, multi); oo = _objects(co, bg, multi)
        pairs = _match_by_shape(oi, oo)
        if not oi or len(pairs) < len(oi):
            return None
        stat = set(); moved = set()
        for a, b in pairs:
            cc = _colour_scalar(a)
            d = (int(round(b["cr"] - a["cr"])), int(round(b["cc"] - a["cc"])))
            (stat if d == (0, 0) else moved).add(cc)
        anchors = (stat - moved) - {None}
        common = anchors if common is None else (common & anchors)
    return {"anchor": min(common)} if common else None


def _apply_anchor(col, spec, params, multi):
    A = params["anchor"]

    def shift_of(objs, shp):
        H, W = shp
        anchor_cells = set()
        for o in objs:
            if _colour_scalar(o) == A:
                anchor_cells |= set(o["cells"])
        if not anchor_cells:
            return None
        ar = sum(r for r, c in anchor_cells) / len(anchor_cells)
        ac = sum(c for r, c in anchor_cells) / len(anchor_cells)
        movers = [o for o in objs if _colour_scalar(o) != A]

        def dir_of(o):                                              # axis-aligned, toward the anchor bar
            ddr = ar - o["cr"]; ddc = ac - o["cc"]
            return (1 if ddr > 0 else -1, 0) if abs(ddr) >= abs(ddc) else (0, 1 if ddc > 0 else -1)
        base = _slide_all(movers, shp, dir_of, anchor_cells)       # pack toward the bar, anchor frozen
        return {id(o): base.get(id(o), (0, 0)) for o in objs}
    return _shift_render(col, multi, shift_of)


def _fg_components(c: torch.Tensor, bg: int) -> int:
    """# of 4-connected non-bg components in a sub-grid (a band's 'breaks'/segments)."""
    H, W = c.shape; cg = c.tolist(); seen = [[False] * W for _ in range(H)]; n = 0
    for r in range(H):
        for k in range(W):
            if cg[r][k] != bg and not seen[r][k]:
                n += 1; st = [(r, k)]; seen[r][k] = True
                while st:
                    y, x = st.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny][nx] and cg[ny][nx] != bg:
                            seen[ny][nx] = True; st.append((ny, nx))
    return n


def _bands_by_sep(col: torch.Tensor, sep: int, axis: str):
    """Bands between lines made entirely of the SEPARATOR colour `sep` (not the border background --
    e.g. black separator columns inside a red mesh). axis='col' -> vertical bands; 'row' -> horizontal."""
    H, W = col.shape
    n = W if axis == "col" else H
    if axis == "col":
        is_sep = [all(int(col[r, c]) == sep for r in range(H)) for c in range(W)]
    else:
        is_sep = [all(int(col[r, c]) == sep for c in range(W)) for r in range(H)]
    bands = []; s = None
    for i in range(n):
        if not is_sep[i]:
            s = i if s is None else s
        elif s is not None:
            bands.append((s, i - 1)); s = None
    if s is not None:
        bands.append((s, n - 1))
    return bands


def _fit_band_sort(demos, spec, multi):
    """Find a separator colour whose full lines split EVERY demo into >=2 equal-width bands."""
    axis = spec["axis"]; cand = None
    for ci, co in demos:
        ok = set()
        for sep in range(10):
            bands = _bands_by_sep(ci, sep, axis)
            if len(bands) >= 2 and len({e - s for s, e in bands}) == 1:
                ok.add(sep)
        cand = ok if cand is None else (cand & ok)
    return {"sep": min(cand)} if cand else None


def _apply_band_sort(col, spec, params, multi):
    """Reorder equal-width bands by the count of separator-colour cells inside (= holes / breaks)."""
    axis = spec["axis"]; desc = spec.get("desc", False); sep = params["sep"]
    bands = _bands_by_sep(col, sep, axis)
    if len(bands) < 2 or len({e - s for s, e in bands}) != 1:
        return None
    conts = [(col[:, s:e + 1].clone() if axis == "col" else col[s:e + 1, :].clone()) for s, e in bands]
    keys = [int((c == sep).sum()) for c in conts]
    sign = -1 if desc else 1
    order = sorted(range(len(bands)), key=lambda i: (sign * keys[i], i))
    out = col.clone()
    for slot, (s, e) in enumerate(bands):
        src = conts[order[slot]]
        if axis == "col":
            out[:, s:e + 1] = src
        else:
            out[s:e + 1, :] = src
    return out


def _apply_generate(col, spec, params, multi):
    """Constructive frame: count pixels per colour, draw bars bottom-up ordered by colour id (bar chart)."""
    import collections as _c
    bg = _background(col); H, W = col.shape
    cnt = _c.Counter()
    for r in range(H):
        for k in range(W):
            v = int(col[r, k])
            if v != bg and v >= 0:
                cnt[v] += 1
    cols = sorted(cnt)
    out = torch.full((H, W), bg, dtype=torch.long)
    for i, cc in enumerate(cols):
        if i >= W:
            break
        for r in range(min(cnt[cc], H)):
            out[H - 1 - r, i] = cc
    return out


RESOLVERS = {
    "translate": (_fit_translate, _apply_translate),
    "displace": (_fit_displace, _apply_displace),
    "to_object": (_fit_none, _apply_to_object),
    "absolute": (_fit_absolute, _apply_absolute),
    "reflect": (_fit_none, _apply_reflect),
    "sort": (_fit_sort, _apply_sort),
    "anchor": (_fit_anchor, _apply_anchor),
    "band_sort": (_fit_band_sort, _apply_band_sort),
    "generate": (_fit_none, _apply_generate),
}

# The declarative frame space. Adding a frame = ONE row here (+ a resolver only if genuinely novel).
RECIPE_SPACE = [
    {"resolver": "translate"},
    {"resolver": "displace", "selector": ("fixed", "N")},
    {"resolver": "displace", "selector": ("fixed", "S")},
    {"resolver": "displace", "selector": ("fixed", "E")},
    {"resolver": "displace", "selector": ("fixed", "W")},
    {"resolver": "displace", "selector": ("nearest_edge",)},
    {"resolver": "displace", "selector": ("by_key", "colour")},
    {"resolver": "displace", "selector": ("by_key", "shape")},
    # mover-selected variants: apply the action ONLY to the odd-one-out objects (the rest are anchors)
    {"resolver": "displace", "selector": ("nearest_edge",), "movers": "minority_colour"},
    {"resolver": "displace", "selector": ("fixed", "N"), "movers": "minority_colour"},
    {"resolver": "displace", "selector": ("fixed", "S"), "movers": "minority_colour"},
    {"resolver": "displace", "selector": ("fixed", "E"), "movers": "minority_colour"},
    {"resolver": "displace", "selector": ("fixed", "W"), "movers": "minority_colour"},
    {"resolver": "displace", "selector": ("nearest_edge",), "movers": "singleton"},
    # gravity toward a colour anchor (a line/bar whose side varies per demo)
    {"resolver": "anchor"},
    {"resolver": "to_object"},
    {"resolver": "absolute", "axis": "col", "key": "colour"},
    {"resolver": "absolute", "axis": "row", "key": "colour"},
    {"resolver": "absolute", "axis": "col", "key": "shape"},
    {"resolver": "absolute", "axis": "row", "key": "shape"},
    {"resolver": "absolute", "axis": "col", "key": "size"},
    {"resolver": "absolute", "axis": "row", "key": "size"},
    {"resolver": "reflect", "axis": "h"},
    {"resolver": "reflect", "axis": "v"},
    {"resolver": "reflect", "axis": "point"},
    {"resolver": "sort", "axis": "col", "attr": "holes"},
    {"resolver": "sort", "axis": "row", "attr": "holes"},
    {"resolver": "sort", "axis": "col", "attr": "breaks"},
    {"resolver": "sort", "axis": "row", "attr": "breaks"},
    {"resolver": "sort", "axis": "col", "attr": "size"},
    {"resolver": "sort", "axis": "row", "attr": "size"},
    {"resolver": "sort", "axis": "col", "attr": "colour"},
    {"resolver": "sort", "axis": "row", "attr": "colour"},
    {"resolver": "sort", "axis": "col", "attr": "holes", "desc": True},
    {"resolver": "sort", "axis": "row", "attr": "holes", "desc": True},
    {"resolver": "sort", "axis": "col", "attr": "breaks", "desc": True},
    {"resolver": "sort", "axis": "row", "attr": "breaks", "desc": True},
    {"resolver": "sort", "axis": "col", "attr": "size", "desc": True},
    {"resolver": "sort", "axis": "row", "attr": "size", "desc": True},
    # band reorder (grid split into separator-colour lines; reorder bands by separator-cell count)
    {"resolver": "band_sort", "axis": "col"},
    {"resolver": "band_sort", "axis": "col", "desc": True},
    {"resolver": "band_sort", "axis": "row"},
    {"resolver": "band_sort", "axis": "row", "desc": True},
    # cellular gravity: each selected-colour CELL slides independently through the field (per-row/col sand)
    {"resolver": "displace", "selector": ("fixed", "E"), "movers": "minority_colour", "cellular": True},
    {"resolver": "displace", "selector": ("fixed", "W"), "movers": "minority_colour", "cellular": True},
    {"resolver": "displace", "selector": ("fixed", "N"), "movers": "minority_colour", "cellular": True},
    {"resolver": "displace", "selector": ("fixed", "S"), "movers": "minority_colour", "cellular": True},
    {"resolver": "displace", "selector": ("nearest_edge",), "movers": "minority_colour", "cellular": True},
    # constructive: bar chart from per-colour pixel counts
    {"resolver": "generate"},
]


def _meta(spec, multi):
    grp = "multi" if multi else "mono"
    r = spec["resolver"]
    if r == "displace":
        mv = (spec["movers"],) if spec.get("movers", "all") != "all" else ()
        cl = ("cell",) if spec.get("cellular") else ()
        return ("displace",) + tuple(str(x) for x in spec["selector"]) + mv + cl + (grp,)
    if r == "absolute":
        return ("absolute", spec["axis"], spec["key"], grp)
    if r == "sort":
        return ("sort", spec["attr"] + ("_desc" if spec.get("desc") else ""), spec["axis"], grp)
    if r == "band_sort":
        return ("band_sort", spec["axis"] + ("_desc" if spec.get("desc") else ""), grp)
    if r == "reflect":
        return ("reflect", spec["axis"], grp)
    if r == "generate":
        return ("generate", grp)
    return (r, grp)


def _propose_recipes(demos: list) -> list:
    """Generic engine: walk the declarative RECIPE_SPACE x grouping {mono, multi}, fit params via the
    RESOLVERS registry, build the apply. No per-frame branching here -- the frames ARE data. mono is
    walked first so the simpler same-colour explanation is preferred when both verify. Returns
    [(apply_fn, meta)]; apply_fn(col_in)->col_out tensor or None."""
    recipes = []
    for multi in (False, True):
        for spec in RECIPE_SPACE:
            fit, apply = RESOLVERS[spec["resolver"]]
            try:
                params = fit(demos, spec, multi)
            except Exception:
                params = None
            if params is None:
                continue
            apply_fn = (lambda col, apply=apply, spec=spec, params=params, multi=multi:
                        apply(col, spec, params, multi))
            recipes.append((apply_fn, _meta(spec, multi)))
    return recipes


# Frame vocabulary for the Lane-B rule-hypothesis HINT: index 0 = "no verified deterministic frame".
FRAME_VOCAB = ("none", "translate", "displace", "to_object", "absolute", "reflect", "sort",
               "anchor", "band_sort", "generate")
FRAME_TO_IDX = {f: i for i, f in enumerate(FRAME_VOCAB)}


def task_frame_label(support_in: torch.Tensor, support_out: torch.Tensor, side: int) -> int:
    """The verified rearrange-FRAME family of a task (index into FRAME_VOCAB; 0 = none) from its demos.
    This is the narrowing 'rule hypothesis' fed to the TRM (Lane B) -- the operation family the
    deterministic solver verified, NOT the solved grid. Cheap: stops at the first frame reconstructing
    all demos. no-grad."""
    demos = []
    for k in range(int(support_in.shape[0])):
        ci, _ = _compact_colour(support_in[k], side)
        co, _ = _compact_colour(support_out[k], side)
        if ci is None or co is None or ci.shape != co.shape:
            return 0
        demos.append((ci, co))
    if not demos:
        return 0
    for apply_fn, meta in _propose_recipes(demos):
        try:
            ok = True
            for ci, co in demos:
                p = apply_fn(ci)
                if p is None or p.shape != co.shape or not bool(torch.equal(p, co)):
                    ok = False; break
            if ok:
                return FRAME_TO_IDX.get(meta[0], 0)
        except Exception:
            continue
    return 0


def rearrange_candidate(support_in: torch.Tensor, support_out: torch.Tensor, target_in: torch.Tensor,
                        side: int, return_meta: bool = False):
    """The position-analogy solver. Propose frame x key recipes from the demos, keep the FIRST that
    reconstructs EVERY demo exactly, apply it to target_in. Returns flat tokens [side*side] or None.

    support_in/out: [m, L] tokens; target_in: [L] tokens. (return_meta -> (pred, meta))."""
    demos = []
    for k in range(int(support_in.shape[0])):
        ci, _ = _compact_colour(support_in[k], side)
        co, _ = _compact_colour(support_out[k], side)
        if ci is None or co is None or ci.shape != co.shape:
            return (None, None) if return_meta else None
        demos.append((ci, co))
    if not demos:
        return (None, None) if return_meta else None
    tc, _ = _compact_colour(target_in, side)
    if tc is None:
        return (None, None) if return_meta else None
    def _reconstructs(apply_fn):
        for ci, co in demos:
            p = apply_fn(ci)
            if p is None or p.shape != co.shape or not bool(torch.equal(p, co)):
                return False
        return True

    for apply_fn, meta in _propose_recipes(demos):
        try:
            if _reconstructs(apply_fn):
                out = apply_fn(tc)
                if out is None:
                    continue
                pred = _compact_to_flat_tokens(out, side)
                return (pred, meta) if return_meta else pred
        except Exception:
            continue
    return (None, None) if return_meta else None


def rearrange_candidates(support_in: torch.Tensor, support_out: torch.Tensor, target_in: torch.Tensor,
                         side: int, k: int = 2):
    """Up to k DISTINCT verified candidates for target_in (the ARC 2-attempt budget / verifier feed).
    Multiple frames can reconstruct the demos; the first is not always the test-correct one, so we
    expose the top-k distinct predictions. Returns [(pred_tokens[L], meta), ...]."""
    demos = []
    for kk in range(int(support_in.shape[0])):
        ci, _ = _compact_colour(support_in[kk], side)
        co, _ = _compact_colour(support_out[kk], side)
        if ci is None or co is None or ci.shape != co.shape:
            return []
        demos.append((ci, co))
    tc, _ = _compact_colour(target_in, side)
    if not demos or tc is None:
        return []

    def _reconstructs(apply_fn):
        for ci, co in demos:
            p = apply_fn(ci)
            if p is None or p.shape != co.shape or not bool(torch.equal(p, co)):
                return False
        return True

    out_list = []; seen = set()
    for apply_fn, meta in _propose_recipes(demos):
        try:
            if not _reconstructs(apply_fn):
                continue
            out = apply_fn(tc)
            if out is None:
                continue
            pred = _compact_to_flat_tokens(out, side)
            key = tuple(pred.tolist())
            if key in seen:
                continue
            seen.add(key); out_list.append((pred, meta))
            if len(out_list) >= k:
                break
        except Exception:
            continue
    return out_list


def _match_objects(oi: list, oo: list) -> list:
    """pair input->output objects by (colour,size), nearest-centroid within each group. Returns [(a,b)]."""
    pairs = []; used = [False] * len(oo)
    for a in oi:
        best = -1; bestd = 1e18
        for j, b in enumerate(oo):
            if used[j] or b["colour"] != a["colour"] or b["size"] != a["size"]:
                continue
            d = abs(a["cr"] - b["cr"]) + abs(a["cc"] - b["cc"])
            if d < bestd:
                bestd = d; best = j
        if best >= 0:
            used[best] = True; pairs.append((a, oo[best]))
    return pairs


def _rearrange_binding(col_in: torch.Tensor, col_out: torch.Tensor):
    """Characterise the movement rule. Returns a hashable binding tuple (family already == rearrange)."""
    bg = _background(col_in)
    oi = _objects(col_in, bg); oo = _objects(col_out, bg)
    if not oi or len(oo) != len(oi):
        return ("permute_other",)
    pairs = _match_objects(oi, oo)
    if len(pairs) < len(oi):
        return ("permute_other",)
    disp = [(int(round(b["cr"] - a["cr"])), int(round(b["cc"] - a["cc"]))) for a, b in pairs]
    H, W = col_in.shape
    drs = [d[0] for d in disp]; dcs = [d[1] for d in disp]
    # translate: every object shares one non-zero displacement
    if len(set(disp)) == 1 and disp[0] != (0, 0):
        return ("translate", disp[0])
    # slide_to_edge / gravity: single-axis, same-sign, each ends flush to that wall
    if all(c == 0 for c in dcs) and any(r != 0 for r in drs):
        if all(r >= 0 for r in drs) and all(b["rmax"] == H - 1 for _, b in pairs):
            return ("slide_to_edge", "down")
        if all(r <= 0 for r in drs) and all(b["rmin"] == 0 for _, b in pairs):
            return ("slide_to_edge", "up")
    if all(r == 0 for r in drs) and any(c != 0 for c in dcs):
        if all(c >= 0 for c in dcs) and all(b["cmax"] == W - 1 for _, b in pairs):
            return ("slide_to_edge", "right")
        if all(c <= 0 for c in dcs) and all(b["cmin"] == 0 for _, b in pairs):
            return ("slide_to_edge", "left")
    # directional pack/gravity: every object moves along ONE axis with the SAME sign (magnitude free --
    # objects pack against each other or the wall). The magnitude is the job of a clearance relmap channel;
    # the recoverable token is the DIRECTION. This is the canonical gravity primitive.
    if all(r == 0 for r in drs) and any(c != 0 for c in dcs):
        if all(c >= 0 for c in dcs):
            return ("directional", "right")
        if all(c <= 0 for c in dcs):
            return ("directional", "left")
    if all(c == 0 for c in dcs) and any(r != 0 for r in drs):
        if all(r >= 0 for r in drs):
            return ("directional", "down")
        if all(r <= 0 for r in drs):
            return ("directional", "up")
    # sort_pack: objects reordered along an axis by size (output order sorted, input order not)
    for axis, key in (("col", "cc"), ("row", "cr")):
        in_order = [o["size"] for o in sorted(oi, key=lambda o: o[key])]
        out_order = [o["size"] for o in sorted(oo, key=lambda o: o[key])]
        if out_order == sorted(out_order) and in_order != out_order and len(set(in_order)) > 1:
            return ("sort_pack", axis, "size")
    return ("rearrange_move",)


def _binding_direction(binding):
    """Canonical movement DIRECTION of a binding (right/left/up/down) or None. translate and
    slide_to_edge are specializations of a direction, so the gate measures consistency HERE -- the
    direction is the rule token; the magnitude is the relmap clearance channel's job."""
    if binding is None:
        return None
    tag = binding[0]
    if tag in ("directional", "slide_to_edge"):
        return binding[1]
    if tag == "translate":
        dr, dc = binding[1]
        if dr == 0 and dc > 0: return "right"
        if dr == 0 and dc < 0: return "left"
        if dc == 0 and dr > 0: return "down"
        if dc == 0 and dr < 0: return "up"
    return None


def infer_rule_hypotheses(support_in: torch.Tensor, support_out: torch.Tensor, side: int,
                          top_k: int = 2) -> list:
    """[m,L] support pairs -> RANKED list of rule hypotheses (the offline narrowing prior).

    Each hypothesis: {family, score (= demo vote fraction), support_consistency "n/m"}. The rearrange
    hypothesis also carries {binding (hashable tuple), binding_consistency "n/r"}. Ordered by score.
    The first top_k are what a live bus would feed the TRM as evidence tokens."""
    import collections as _c
    m = int(support_in.shape[0])
    fams = []; binds = []
    for k in range(m):
        ci, shi = _compact_colour(support_in[k], side)
        co, sho = _compact_colour(support_out[k], side)
        if ci is None or co is None:
            fams.append("identity"); binds.append(None); continue
        if shi != sho:
            fams.append("size_change"); binds.append(None); continue
        if ci.shape == co.shape and bool(torch.equal(ci, co)):
            fams.append("identity"); binds.append(None)
        elif _hist10(ci) == _hist10(co):
            fams.append("rearrange"); binds.append(_rearrange_binding(ci, co))
        else:
            fams.append("recolor"); binds.append(None)
    cnt = _c.Counter(fams)
    ranked = []
    for fam, c in cnt.most_common():
        h = {"family": fam, "score": c / m, "support_consistency": f"{c}/{m}"}
        if fam == "rearrange":
            r_binds = [b for b, f in zip(binds, fams) if f == "rearrange"]
            dirs = [_binding_direction(b) for b in r_binds]
            known = [d for d in dirs if d is not None]
            if known:
                # primary signal: do the demos agree on a movement DIRECTION (the rule token)?
                dc = _c.Counter(known)
                top_d, dn = dc.most_common(1)[0]
                h["binding"] = ("directional", top_d)
                h["binding_consistency"] = f"{dn}/{c}"          # demos agreeing on the dominant direction
                h["binding_coverage"] = f"{len(known)}/{c}"     # demos that yielded ANY direction
            else:
                bc = _c.Counter(b for b in r_binds if b is not None)
                if bc:
                    top_b, bn = bc.most_common(1)[0]
                    h["binding"] = top_b
                    h["binding_consistency"] = f"{bn}/{c}"
        ranked.append(h)
    return ranked


def _self_test() -> None:
    torch.manual_seed(0)
    S = 8
    L = S * S

    def grid(block_colour, single_colour):
        g = torch.full((S, S), 0 + COLOR_OFFSET, dtype=torch.long)            # colour-0 background
        g[0:3, 0:3] = block_colour + COLOR_OFFSET                             # 9-cell block (large)
        g[6, 6] = single_colour + COLOR_OFFSET                               # singleton
        return g.reshape(L)

    # RULE: a LARGE block of colour 3 -> 7 ; a SINGLETON of colour 3 -> 5.  Same input colour (3),
    # two different outputs by OBJECT SIZE -> a histogram P(out|in=3) cannot decide; analogy can.
    def recolour_grid(flat_in, block_to, single_to):
        g = flat_in.view(S, S).clone()
        g[0:3, 0:3] = block_to + COLOR_OFFSET
        g[6, 6] = single_to + COLOR_OFFSET
        return g.reshape(L)
    d1_in = grid(3, 3); d1_out = recolour_grid(d1_in, 7, 5)
    d2_in = grid(3, 3); d2_out = recolour_grid(d2_in, 7, 5)
    ci = torch.stack([d1_in, d2_in]).unsqueeze(0)                            # [1,2,L]
    co = torch.stack([d1_out, d2_out]).unsqueeze(0)
    cm = torch.ones(1, 2, dtype=torch.bool)

    # TEST input: a block of 3 and a singleton of 3 (same roles, new grid).
    t_in = grid(3, 3).unsqueeze(0)                                           # [1,L]

    demo = extract_object_slots(ci, co, cm, side=S, slots_per_grid=6)
    target = extract_target_slots(t_in, side=S, slots_per_grid=6)

    # the demo slots must carry the conditional split: a colour-3 LARGE slot -> 7 and a colour-3
    # SINGLETON slot -> 5 both exist.
    inc, outc, val = demo["in_col"][0], demo["out_col"][0], demo["valid"][0]
    big3 = ((inc == 3) & (outc == 7) & val).any()
    small3 = ((inc == 3) & (outc == 5) & val).any()
    assert bool(big3) and bool(small3), f"demo slots must split 3->7 (big) and 3->5 (small): {list(zip(inc.tolist(),outc.tolist(),val.tolist()))}"

    cell_prob, cell_conf = analogy_recolour(demo["feats"], demo["out_col"], demo["valid"],
                                            target["feats"], target["cell_idx"])
    pred = cell_prob.argmax(-1).view(S, S)                                   # [S,S]
    # the KILLER assertion: the SAME input colour (3) gets DIFFERENT outputs by object role.
    assert int(pred[0, 0]) == 7, f"block of 3 must retrieve 7, got {int(pred[0,0])}"
    assert int(pred[6, 6]) == 5, f"singleton of 3 must retrieve 5, got {int(pred[6,6])}"
    # a histogram over the demos would tie 3->{7,5} 50/50 and could not produce both. Confirm the
    # cell-only modal map is genuinely ambiguous (so this is a real win, not a trivial case):
    flat_in = ci[0].reshape(-1); flat_out = co[0].reshape(-1)
    mask3 = (flat_in == 3 + COLOR_OFFSET)
    outs3 = (flat_out[mask3] - COLOR_OFFSET).unique()
    assert set(outs3.tolist()) == {7, 5}, "histogram is genuinely ambiguous on colour 3 (7 and 5)"

    # SENSITIVITY (extraction gate): swap in a DIFFERENT task's demos -> retrieval must change.
    d_other_in = grid(4, 4); d_other_out = recolour_grid(d_other_in, 1, 2)   # 4->1 big, 4->2 small
    ci2 = torch.stack([d_other_in, d_other_in]).unsqueeze(0)
    co2 = torch.stack([d_other_out, d_other_out]).unsqueeze(0)
    demo2 = extract_object_slots(ci2, co2, cm, side=S, slots_per_grid=6)
    cp2, _ = analogy_recolour(demo2["feats"], demo2["out_col"], demo2["valid"],
                              target["feats"], target["cell_idx"])
    # with wrong-task demos (no colour-3 object), the colour-3 test objects retrieve from a
    # mismatched key -> the answer is NOT the {7,5} of the real task.
    pred2 = cp2.argmax(-1).view(S, S)
    assert not (int(pred2[0, 0]) == 7 and int(pred2[6, 6]) == 5), "shuffled demos must not reproduce the rule"

    # LEARNED wrapper: no-op at init (rule_vec == 0, retrieval == cosine floor) + grad flows.
    bank = ObjectRuleBank(rule_dim=32, d_model=64, slots_per_grid=6)
    out = bank(ci, co, cm, t_in, side=S)
    assert out["rule_vec"].shape == (1, 32)
    assert float(out["rule_vec"].abs().max()) == 0.0, "rule_vec must be 0 at init (no-op z_H injection)"
    pred_b = out["recolour_prob"].argmax(-1).view(S, S)
    assert int(pred_b[0, 0]) == 7 and int(pred_b[6, 6]) == 5, "wrapper retrieval == deterministic floor at init"
    loss = out["rule_vec"].pow(2).mean() + out["recolour_prob"].pow(2).mean()
    loss.backward()
    assert bank.rule_proj.weight.grad is not None and bank.key_refine.weight.grad is not None

    # ---- S6 rule-hypothesis inference: family routing + binding recovery + consistency ----
    def _flat(gr, side):
        canvas = torch.zeros(side, side, dtype=torch.long)
        for r, row in enumerate(gr):
            for c, v in enumerate(row):
                canvas[r, c] = int(v) + COLOR_OFFSET
        return canvas.reshape(-1)

    sd = 6
    # TRANSLATE: a 2x2 block of colour 4 on a colour-0 field, shifted +2 cols across two demos.
    def block_grid(r0, c0):
        g = [[0] * sd for _ in range(sd)]
        for r in (r0, r0 + 1):
            for c in (c0, c0 + 1):
                g[r][c] = 4
        return g
    tin = torch.stack([_flat(block_grid(0, 0), sd), _flat(block_grid(2, 1), sd)])
    tout = torch.stack([_flat(block_grid(0, 2), sd), _flat(block_grid(2, 3), sd)])
    hyp = infer_rule_hypotheses(tin, tout, sd)
    assert hyp[0]["family"] == "rearrange", f"shifted block must route to rearrange, got {hyp[0]}"
    assert hyp[0].get("binding") == ("directional", "right"), f"binding must canonicalise to directional-right, got {hyp[0].get('binding')}"
    assert hyp[0]["binding_consistency"] == "2/2", f"direction must hold on both demos, got {hyp[0]}"
    # RECOLOR: same positions, colour 4 -> 6 (histogram changes) must NOT route to rearrange.
    def recol(gr, to):
        return [[to if v == 4 else v for v in row] for row in gr]
    rin = torch.stack([_flat(block_grid(0, 0), sd), _flat(block_grid(2, 1), sd)])
    rout = torch.stack([_flat(recol(block_grid(0, 0), 6), sd), _flat(recol(block_grid(2, 1), 6), sd)])
    rhyp = infer_rule_hypotheses(rin, rout, sd)
    assert rhyp[0]["family"] == "recolor", f"in-place colour change must route to recolor, got {rhyp[0]}"
    # SIZE_CHANGE: output bbox differs.
    sin = torch.stack([_flat(block_grid(0, 0), sd)])
    sout = torch.stack([_flat([[4, 4]], sd)])
    shyp = infer_rule_hypotheses(sin, sout, sd)
    assert shyp[0]["family"] == "size_change", f"shape change must route to size_change, got {shyp[0]}"

    # ---- analogy_relocate solver: translate + gravity + absolute-by-key reconstruct exactly ----
    # translate (+2 cols): learned from 2 demos, applied to a held-out test grid.
    p_tr, m_tr = rearrange_candidate(tin, tout, _flat(block_grid(1, 1), sd), sd, return_meta=True)
    assert m_tr is not None and m_tr[0] == "translate", f"translate task must verify translate, got {m_tr}"
    exp_tr = _flat(block_grid(1, 3), sd)
    assert bool(torch.equal(p_tr, exp_tr)), "translate solver must reproduce the shifted block exactly"
    # gravity-down (uniquely): TWO same-colour blocks fall and STACK -> absolute-by-key can't map one
    # colour to two target rows, only gravity stacks them, so this isolates the gravity frame.
    def twoblk(ra, rb):
        g = [[0] * sd for _ in range(sd)]
        for c in (1, 2):
            g[ra][c] = 7; g[rb][c] = 7
        return g
    gin = torch.stack([_flat(twoblk(0, 3), sd), _flat(twoblk(1, 4), sd)])
    gout = torch.stack([_flat(twoblk(sd - 2, sd - 1), sd), _flat(twoblk(sd - 2, sd - 1), sd)])
    p_g, m_g = rearrange_candidate(gin, gout, _flat(twoblk(2, 5), sd), sd, return_meta=True)
    assert m_g is not None and m_g[0] == "displace" and "S" in m_g, f"stacking blocks must verify displace-S, got {m_g}"
    assert bool(torch.equal(p_g, _flat(twoblk(sd - 2, sd - 1), sd))), "displace solver must stack blocks at the floor"
    # absolute-by-key (colour -> column): colour 4 -> col 4, colour 5 -> col 1, row preserved.
    def two(c4row, c5row):
        g = [[0] * sd for _ in range(sd)]
        g[c4row][0] = 4; g[c5row][0] = 5
        return g
    def two_out(c4row, c5row):
        g = [[0] * sd for _ in range(sd)]
        g[c4row][4] = 4; g[c5row][1] = 5
        return g
    ain = torch.stack([_flat(two(0, 3), sd), _flat(two(1, 5), sd)])
    aout = torch.stack([_flat(two_out(0, 3), sd), _flat(two_out(1, 5), sd)])
    p_a, m_a = rearrange_candidate(ain, aout, _flat(two(2, 4), sd), sd, return_meta=True)
    assert m_a is not None and m_a[0] == "absolute", f"colour->column task must verify absolute, got {m_a}"
    assert bool(torch.equal(p_a, _flat(two_out(2, 4), sd))), "absolute-by-key solver must place by colour map"
    # reflect (uniquely): an asymmetric L mirrored horizontally -- no translation reproduces a flip.
    def Lshape():
        g = [[0] * sd for _ in range(sd)]
        g[0][0] = 6; g[0][1] = 6; g[1][0] = 6
        return g
    def Lflip():
        g = [[0] * sd for _ in range(sd)]
        for (r, c) in [(0, 0), (0, 1), (1, 0)]:
            g[r][sd - 1 - c] = 6
        return g
    rin = torch.stack([_flat(Lshape(), sd), _flat(Lshape(), sd)])
    rout = torch.stack([_flat(Lflip(), sd), _flat(Lflip(), sd)])
    p_r, m_r = rearrange_candidate(rin, rout, _flat(Lshape(), sd), sd, return_meta=True)
    assert m_r is not None and m_r[0] == "reflect", f"L mirror must verify reflect, got {m_r}"
    assert bool(torch.equal(p_r, _flat(Lflip(), sd))), "reflect solver must mirror the shape"
    # to_object (uniquely): a lone cell snaps to the larger block; direction differs per demo, so no
    # fixed column/translation explains it -- only 'slide to nearest object' does.
    def blk_lone(lr, lc):
        g = [[0] * sd for _ in range(sd)]
        for r in (4, 5):
            for c in (4, 5):
                g[r][c] = 2
        g[lr][lc] = 3
        return g
    oin = torch.stack([_flat(blk_lone(0, 4), sd), _flat(blk_lone(4, 0), sd)])
    oout = torch.stack([_flat(blk_lone(3, 4), sd), _flat(blk_lone(4, 3), sd)])
    p_o, m_o = rearrange_candidate(oin, oout, _flat(blk_lone(0, 5), sd), sd, return_meta=True)
    # both 'to_object' (move small->nearest object) and 'anchor' (move toward the stationary colour) are
    # valid generalizations of a lone-cell snap; either is a correct verified frame.
    assert m_o is not None and m_o[0] in ("to_object", "anchor"), f"lone-cell snap must verify to_object/anchor, got {m_o}"
    assert bool(torch.equal(p_o, _flat(blk_lone(3, 5), sd))), "snap solver must move the lone cell to touch the block"

    print("object_rule_bank self-test PASS  "
          "(demo slots split 3->7 big / 3->5 small; analogy gives block=7, singleton=5 where the "
          "histogram is 50/50 ambiguous; shuffled demos break it; learned wrapper no-op at init, grad ok; "
          "S6 rule-hypotheses route translate->rearrange(0,+2) 2/2, recolor, size_change)")


if __name__ == "__main__":
    _self_test()
