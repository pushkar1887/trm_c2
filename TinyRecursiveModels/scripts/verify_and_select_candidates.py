"""Dump-once / score-many OFFLINE harness — check the proposed changes WITHOUT changing the system.

Read-only. Treats the demos as a frozen oracle, builds each candidate OUTPUT offline, and scores it
by LODO reconstruction (hold out one demo, predict it from the rest) — the same honest signal the
training uses, but here as an inference-time arbiter. NOTHING in trm_fvr_c2.py / losses_fvr.py /
color_repair_head.py is touched or imported; this only reuses the standalone object_rule_bank module
and (optionally) raw ARC JSON.

Candidates scored (model-FREE, computable from demos alone):
  identity   copy the input            (baseline)
  floor      cond_inout argmax = P(out|in=a) modal recolour   (the deterministic FLOOR)
  analogy    floor + object-level copy-by-relation override    (S2/S5: ObjectRuleBank)

The question this answers, with zero risk and no GPU:
  on REAL conditional tasks, does the analogy candidate's LODO reconstruction BEAT the floor's?
  If yes -> S2/S5 are worth committing. If no -> don't change the system.

The floor-respecting SELECTOR (floor always in the set, tie -> floor) is verified here too:
  selector_exact >= floor_exact on EVERY task, by construction -> the net-negative bug (R13) cannot
  recur once selection is the output path.

Model-dependent candidates (head, oracle-rule_vec) plug in later: dump grid_z/base_logits from a
normal eval forward to an .npz and add a candidate that reads it -- the scoring core is unchanged.

Run:
  trm\\Scripts\\python.exe scripts\\verify_and_select_candidates.py                 # self-test
  trm\\Scripts\\python.exe scripts\\verify_and_select_candidates.py --real          # real concept tasks
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.recursive_reasoning.object_rule_bank import (  # noqa: E402
    extract_object_slots, extract_target_slots, analogy_recolour,
    _compact_colour, _background, _objects, _hole_count, _symmetry, _d4_canon,
)
from models.recursive_reasoning.object_bank import cell_conditioning_signature  # noqa: E402
from lodo_refiner import refine_lodo_recipes
from model_candidate_dump import load_model_dump
from parse import _parse_same
from rule_library import RuleLibrary
from solve import _default_geo_ops, _object_relrecolor_predict, _set_cover_predict

COLOR_OFFSET = 2
N_COLORS = 10


# ---------------------------------------------------------------- candidate predictors (model-free)
def build_cond(support_in: torch.Tensor, support_out: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[m,L] support -> (out_for[10] modal output colour per input colour, had[10] bool, cooc[10,10])."""
    cooc = torch.zeros(N_COLORS, N_COLORS)
    m = support_in.shape[0]
    for k in range(m):
        x, y = support_in[k].long(), support_out[k].long()
        col = (x >= COLOR_OFFSET) & (y >= COLOR_OFFSET)
        xc = (x[col] - COLOR_OFFSET).clamp(0, 9)
        yc = (y[col] - COLOR_OFFSET).clamp(0, 9)
        flat = (xc * N_COLORS + yc)
        cooc.view(-1).scatter_add_(0, flat, torch.ones_like(flat, dtype=cooc.dtype))
    had = cooc.sum(1) > 0
    out_for = cooc.argmax(1)
    out_for = torch.where(had, out_for, torch.arange(N_COLORS))   # no data -> copy a->a
    return out_for, had, cooc


def floor_predict(support_in, support_out, target_in, side):
    """Deterministic FLOOR: each target colour cell -> modal output for its input colour."""
    out_for, had, _ = build_cond(support_in, support_out)
    ti = target_in.long()
    isc = ti >= COLOR_OFFSET
    a = (ti - COLOR_OFFSET).clamp(0, 9)
    mapped = out_for[a] + COLOR_OFFSET
    mapped = torch.where(had[a], mapped, ti)                      # copy where no consensus
    return torch.where(isc, mapped, ti)                          # non-colour cells = structure copy


def identity_predict(support_in, support_out, target_in, side):
    return target_in.long().clone()


def analogy_predict(support_in, support_out, target_in, side, K=6, conf_thresh=0.5):
    """FLOOR, then OBJECT-LEVEL copy-by-relation override (S2/S5) where retrieval is confident."""
    pred = floor_predict(support_in, support_out, target_in, side)
    m = support_in.shape[0]
    demo = extract_object_slots(support_in.unsqueeze(0), support_out.unsqueeze(0),
                                torch.ones(1, m, dtype=torch.bool), side, K)
    tgt = extract_target_slots(target_in.unsqueeze(0), side, K)
    cell_prob, cell_conf = analogy_recolour(demo["feats"], demo["out_col"], demo["valid"],
                                            tgt["feats"], tgt["cell_idx"])
    ana_col = cell_prob.argmax(-1)[0]                            # [L]
    isc = target_in.long() >= COLOR_OFFSET
    has_slot = (tgt["cell_idx"][0] >= 0) & (cell_conf[0] > conf_thresh)
    override = isc & has_slot
    return torch.where(override, ana_col + COLOR_OFFSET, pred)


def neighbour_keys(grid_flat, side):
    """Per cell -> (centre colour, sorted 4-neighbour colours) key; None for non-colour cells.
    The cross-demo 'neighbour-conditioned recolour' key (R17): a cell's output depends on what
    surrounds it -- the cell-level copy-by-relation (VARC) the object lookup cannot express."""
    S = side
    g = grid_flat.view(S, S).long()
    gp = torch.full((S + 2, S + 2), -1, dtype=torch.long)
    gp[1:S + 1, 1:S + 1] = g
    up, down = gp[0:S, 1:S + 1], gp[2:S + 2, 1:S + 1]
    left, right = gp[1:S + 1, 0:S], gp[1:S + 1, 2:S + 2]
    nb, _ = torch.stack([up, down, left, right], -1).sort(-1)                # direction-invariant
    gf, nbf = g.reshape(-1), nb.reshape(-1, 4)
    keys = []
    for i in range(S * S):
        c = int(gf[i])
        keys.append(None if c < COLOR_OFFSET else (c, int(nbf[i, 0]), int(nbf[i, 1]),
                                                   int(nbf[i, 2]), int(nbf[i, 3])))
    return keys


def neighbour_predict(support_in, support_out, target_in, side):
    """Cell-level copy-by-relation: modal output for each (centre, neighbourhood) seen in support."""
    pred = floor_predict(support_in, support_out, target_in, side)
    cond = {}
    for k in range(support_in.shape[0]):
        ink = neighbour_keys(support_in[k], side)
        outt = support_out[k].long()
        for i, key in enumerate(ink):
            if key is None:
                continue
            cond.setdefault(key, {})
            o = int(outt[i]); cond[key][o] = cond[key].get(o, 0) + 1
    out = pred.clone()
    for i, key in enumerate(neighbour_keys(target_in, side)):
        if key is not None and key in cond:
            out[i] = max(cond[key].items(), key=lambda kv: kv[1])[0]
    return out


def analogy_gated_predict(support_in, support_out, target_in, side, K=6, conf_thresh=0.5):
    """THE FIX. Analogy override RESTRICTED to CONDITIONAL input-colours -- those the floor provably
    can't handle (>1 distinct output in the demos). Leaves clean/copy colours to the floor, so it
    can NEVER overwrite a cell the floor already had right. Directly attacks the microscope finding
    (analogy overwrote 55% correct floor cells); makes the deterministic relational route floor-safe."""
    pred = floor_predict(support_in, support_out, target_in, side)
    _, _, cooc = build_cond(support_in, support_out)
    cond_col = (cooc > 0).sum(1) > 1                                       # [10] colour -> >1 output
    m = support_in.shape[0]
    demo = extract_object_slots(support_in.unsqueeze(0), support_out.unsqueeze(0),
                                torch.ones(1, m, dtype=torch.bool), side, K)
    tgt = extract_target_slots(target_in.unsqueeze(0), side, K)
    cp, cc = analogy_recolour(demo["feats"], demo["out_col"], demo["valid"],
                              tgt["feats"], tgt["cell_idx"])
    ana = cp.argmax(-1)[0]
    ti = target_in.long(); a = (ti - COLOR_OFFSET).clamp(0, 9); isc = ti >= COLOR_OFFSET
    has_slot = (tgt["cell_idx"][0] >= 0) & (cc[0] > conf_thresh)
    override = isc & has_slot & cond_col[a]                                # only conditional colours
    return torch.where(override, ana + COLOR_OFFSET, pred)


# ------------------------------------------------------------------ verified SIZE-CHANGE family
# The compose-test measured size_change 0/45 with ZERO offline coverage: every recolor predictor
# assumes same shape. These candidates PROPOSE a (size rule x content generator) from one demo,
# VERIFY exact reconstruction on every support demo, and stay silent otherwise -- the same
# propose->verify->select contract as Lane A, so a wrong generator can never be selected.
def _d4_ops():
    return {"id": lambda g: g,
            "fh": lambda g: torch.flip(g, (0,)),
            "fv": lambda g: torch.flip(g, (1,)),
            "r180": lambda g: torch.rot90(g, 2, (0, 1)),
            "r90": lambda g: torch.rot90(g, 1, (0, 1)),
            "r270": lambda g: torch.rot90(g, 3, (0, 1)),
            "tr": lambda g: g.t(),
            "atr": lambda g: torch.rot90(g.t(), 2, (0, 1))}


def _size_family_predict(support_in, support_out, target_in, side, fit_one, apply_one):
    """Fit params on the first demo, verify EXACT reconstruction on ALL support demos, apply."""
    m = support_in.shape[0]
    gs = [( _flat_to_compact_grid(support_in[k], side),
            _flat_to_compact_grid(support_out[k], side)) for k in range(m)]
    if any(gi is None or go is None for gi, go in gs):
        return None
    params = fit_one(gs[0][0], gs[0][1])
    if params is None:
        return None
    for gi, go in gs:
        pred = apply_one(gi, params)
        if pred is None or pred.shape != go.shape or not bool(torch.equal(pred, go)):
            return None
    tg = _flat_to_compact_grid(target_in, side)
    if tg is None:
        return None
    out = apply_one(tg, params)
    if out is None or out.shape[0] > side or out.shape[1] > side:
        return None
    return _compact_to_flat(out, side)


def upscale_predict(support_in, support_out, target_in, side):
    """Integer pixel upscale: out = each cell repeated (kr x kc)."""
    def fit(gi, go):
        hi, wi = gi.shape; ho, wo = go.shape
        if hi == 0 or wi == 0 or ho % hi or wo % wi:
            return None
        kr, kc = ho // hi, wo // wi
        return (kr, kc) if (kr, kc) != (1, 1) else None
    def apply(g, p):
        return g.repeat_interleave(p[0], 0).repeat_interleave(p[1], 1)
    return _size_family_predict(support_in, support_out, target_in, side, fit, apply)


def downscale_modal_predict(support_in, support_out, target_in, side):
    """Integer block downscale: out cell = modal colour of its (kr x kc) input block."""
    def fit(gi, go):
        hi, wi = gi.shape; ho, wo = go.shape
        if ho == 0 or wo == 0 or hi % ho or wi % wo:
            return None
        kr, kc = hi // ho, wi // wo
        return (kr, kc) if (kr, kc) != (1, 1) else None
    def apply(g, p):
        kr, kc = p
        h, w = g.shape
        if h % kr or w % kc:
            return None
        blocks = g.view(h // kr, kr, w // kc, kc).permute(0, 2, 1, 3).reshape(h // kr, w // kc, -1)
        return blocks.mode(-1).values
    return _size_family_predict(support_in, support_out, target_in, side, fit, apply)


def tile_d4_predict(support_in, support_out, target_in, side):
    """Tiling with a per-position D4 op pattern (covers plain tiling and mirror/rotate tilings)."""
    ops = _d4_ops()
    def fit(gi, go):
        hi, wi = gi.shape; ho, wo = go.shape
        if hi == 0 or wi == 0 or ho % hi or wo % wi:
            return None
        pr, pc = ho // hi, wo // wi
        if (pr, pc) == (1, 1) or pr * pc > 25:
            return None
        pat = []
        for i in range(pr):
            row = []
            for j in range(pc):
                block = go[i * hi:(i + 1) * hi, j * wi:(j + 1) * wi]
                name = next((nm for nm, op in ops.items()
                             if op(gi).shape == block.shape and bool(torch.equal(op(gi), block))), None)
                if name is None:
                    return None
                row.append(name)
            pat.append(row)
        return (pr, pc, pat)
    def apply(g, p):
        pr, pc, pat = p
        try:
            rows = [torch.cat([ops[pat[i][j]](g) for j in range(pc)], 1) for i in range(pr)]
        except RuntimeError:
            return None
        if len({r.shape[1] for r in rows}) != 1:
            return None
        return torch.cat(rows, 0)
    return _size_family_predict(support_in, support_out, target_in, side, fit, apply)


def _bbox_crop(g, mask):
    if not bool(mask.any()):
        return None
    nz = mask.nonzero(as_tuple=False)
    r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
    r1, c1 = int(nz[:, 0].max()), int(nz[:, 1].max())
    return g[r0:r1 + 1, c0:c1 + 1]


def crop_bbox_predict(support_in, support_out, target_in, side):
    """Crop to the bbox of the non-background content (background = modal colour)."""
    def fit(gi, go):
        return ("nonbg",)
    def apply(g, p):
        bg = int(g.reshape(-1).mode().values)
        return _bbox_crop(g, g != bg)
    return _size_family_predict(support_in, support_out, target_in, side, fit, apply)


def crop_colour_predict(support_in, support_out, target_in, side):
    """Crop to the bbox of ONE colour -- the frame ('incl') or its interior ('inner'). The colour
    and variant are fitted on the first demo and must verify on all."""
    def fit(gi, go):
        for tok in sorted(int(t) for t in torch.unique(gi) if int(t) >= COLOR_OFFSET):
            for variant in ("incl", "inner"):
                pred = _apply_crop_colour(gi, (tok, variant))
                if pred is not None and pred.shape == go.shape and bool(torch.equal(pred, go)):
                    return (tok, variant)
        return None
    return _size_family_predict(support_in, support_out, target_in, side, fit, _apply_crop_colour)


def _apply_crop_colour(g, p):
    tok, variant = p
    box = _bbox_crop(g, g == tok)
    if box is None:
        return None
    if variant == "incl":
        return box
    nz = (g == tok).nonzero(as_tuple=False)
    r0, c0 = int(nz[:, 0].min()) + 1, int(nz[:, 1].min()) + 1
    r1, c1 = int(nz[:, 0].max()) - 1, int(nz[:, 1].max()) - 1
    if r1 < r0 or c1 < c0:
        return None
    return g[r0:r1 + 1, c0:c1 + 1]


SIZE_CANDIDATES = {"upscale": upscale_predict, "downscale": downscale_modal_predict,
                   "tile_d4": tile_d4_predict, "crop_bbox": crop_bbox_predict,
                   "crop_colour": crop_colour_predict}


# ============================================== FIX F (2026-07-06): computed-WHERE offline candidates
# Gate result (--value-binding-probe): the flood-fill enclosure + nearest-seed signature keys lift
# multi val_acc +4.6..+7.2 over marginal and DOUBLE multi oExact (encl+seed 5.3 vs 2.3) -- the middle
# band of the pre-registered gate => bank them as OFFLINE candidates (propose -> verify EXACT on ALL
# support -> else None), not a model head. Plus the object-correspondence recolor candidate the
# 2026-07-05 correspondence probe recommended (color+holes / size_rank solve ~2-4 multi tasks).
# All same-shape families (deliberately NOT in NONSHAPE_OK).

_SIG_CACHE: dict = {}


def _sig_seed_encl(grid_flat, side):
    """(encl_colour LONG[L], seed_colour LONG[L]) = cell_conditioning_signature cols 11/12 (FIX H),
    cached by grid bytes (LODO re-scores the same grids many times)."""
    key = (grid_flat.numpy().tobytes(), int(side))
    hit = _SIG_CACHE.get(key)
    if hit is None:
        if len(_SIG_CACHE) > 8192:
            _SIG_CACHE.clear()
        sig, _valid = cell_conditioning_signature(grid_flat.view(1, -1).long(), side)
        hit = (sig[0, :, 11].clone(), sig[0, :, 12].clone())
        _SIG_CACHE[key] = hit
    return hit


def _apply_seed_fill(g, side, enclosed_only):
    """out = g with BACKGROUND cells replaced by their nearest-seed colour (col 12). None when the
    transform has nothing to say (no bg cell with a seed)."""
    col, _hw = _compact_colour(g, side)
    if col is None:
        return None
    bg = _background(col)
    encl, seed = _sig_seed_encl(g, side)
    mask = (g == bg + COLOR_OFFSET) & (seed != 10)
    if enclosed_only:
        mask = mask & (encl != 10)
    if not bool(mask.any()):
        return None
    out = g.clone()
    out[mask] = (seed[mask] + COLOR_OFFSET).to(out.dtype)
    return out


def seed_fill_predictor(enclosed_only):
    """Parameter-free adjacency-fill: bg cells (optionally only flood-fill-ENCLOSED ones) take their
    nearest seed colour. Verified by EXACT reconstruction of every support demo, else None."""
    def _predict(support_in, support_out, target_in, side):
        for i in range(support_in.shape[0]):
            rec = _apply_seed_fill(support_in[i], side, enclosed_only)
            if rec is None or not torch.equal(rec, support_out[i]):
                return None
        return _apply_seed_fill(target_in, side, enclosed_only)
    return _predict


def _fit_encl_fill(support_in, support_out, side):
    """Fit {enclosing colour -> fill colour} over flood-fill-ENCLOSED bg cells. None on any
    inconsistency, or when a demo changes a cell OUTSIDE the enclosed-fill story (the family then
    cannot explain the task alone)."""
    m: dict = {}
    for i in range(support_in.shape[0]):
        gi, go = support_in[i], support_out[i]
        col, _hw = _compact_colour(gi, side)
        if col is None:
            return None
        bg = _background(col)
        encl, _seed = _sig_seed_encl(gi, side)
        fill_mask = (gi == bg + COLOR_OFFSET) & (encl != 10)
        if not torch.equal(gi[~fill_mask], go[~fill_mask]):
            return None
        for j in fill_mask.nonzero(as_tuple=True)[0].tolist():
            e = int(encl[j])
            d = int(go[j]) - COLOR_OFFSET
            if d < 0:
                return None
            if m.get(e, d) != d:
                return None
            m[e] = d
    return m or None


def encl_fill_predict(support_in, support_out, target_in, side):
    """Enclosure-fill: flood-fill-enclosed bg cells -> fitted f(enclosing colour). Propose (fit the
    map) -> verify EXACT on all support -> apply to the target; any miss -> None."""
    m = _fit_encl_fill(support_in, support_out, side)
    if m is None:
        return None

    def _apply(g):
        col, _hw = _compact_colour(g, side)
        if col is None:
            return None
        bg = _background(col)
        encl, _seed = _sig_seed_encl(g, side)
        out = g.clone()
        for j in ((g == bg + COLOR_OFFSET) & (encl != 10)).nonzero(as_tuple=True)[0].tolist():
            e = int(encl[j])
            if e in m:
                out[j] = m[e] + COLOR_OFFSET
        return out

    for i in range(support_in.shape[0]):
        rec = _apply(support_in[i])
        if rec is None or not torch.equal(rec, support_out[i]):
            return None
    return _apply(target_in)


def corr_recolor_predictor(key_fields):
    """Object-correspondence recolor (the correspondence probe's banked recommendation): fit
    key -> output-colour over support INPUT objects (output colour = the object's uniform colour in
    the demo OUTPUT), require global consistency, apply by correspondence to the target objects,
    verify EXACT on every support demo. Uncovered objects/cells copy."""
    def _predict(support_in, support_out, target_in, side):
        table: dict = {}

        def _pairs(gi, go):
            out = []
            for o in _corr_objects(gi, side):
                vals = {int(go[j]) for j in o["_flat"]}
                if len(vals) != 1:
                    return None                          # object not uniformly recoloured
                v = vals.pop() - COLOR_OFFSET
                if v < 0:
                    return None                          # object cells became PAD/EOS
                out.append((tuple(o[f] for f in key_fields), v))
            return out

        for i in range(support_in.shape[0]):
            prs = _pairs(support_in[i], support_out[i])
            if prs is None:
                return None
            for k, v in prs:
                if table.get(k, v) != v:
                    return None                          # same key -> two colours: inconsistent
                table[k] = v
        if not table:
            return None

        def _apply(g):
            out = g.clone()
            for o in _corr_objects(g, side):
                k = tuple(o[f] for f in key_fields)
                if k in table:
                    for j in o["_flat"]:
                        out[j] = table[k] + COLOR_OFFSET
            return out

        for i in range(support_in.shape[0]):
            if not torch.equal(_apply(support_in[i]), support_out[i]):
                return None
        return _apply(target_in)
    return _predict


FIXF_CANDIDATES = {
    "seed_fill_enc": seed_fill_predictor(True),          # enclosed bg -> nearest seed colour
    "seed_fill_bg":  seed_fill_predictor(False),         # ALL bg -> nearest seed colour
    "encl_fill":     encl_fill_predict,                  # enclosed bg -> f(enclosing colour)
    "corr_colholes": corr_recolor_predictor(("colour", "_holes")),
    "corr_szrank":   corr_recolor_predictor(("_grank",)),
}


# ============================================================ FIX 2 (fg half) + FIX 3 (projection/bbox)
# Same propose -> verify-EXACT-on-all-support -> else-None contract as every FIXF family.

def _colour_dists(col, bg):
    """[10, H, W] Manhattan distance to the nearest cell of each colour (INF where colour absent).
    Iterative relaxation on the compact grid (H,W <= 30)."""
    H, W = col.shape
    INF = 10_000
    d = torch.full((10, H, W), INF, dtype=torch.long)
    for c in range(10):
        m = (col == c) & (col != bg) if c != bg else torch.zeros_like(col, dtype=torch.bool)
        d[c][m] = 0
    for _ in range(H + W):
        prev = d.clone()
        d[:, 1:, :] = torch.minimum(d[:, 1:, :], d[:, :-1, :] + 1)
        d[:, :-1, :] = torch.minimum(d[:, :-1, :], d[:, 1:, :] + 1)
        d[:, :, 1:] = torch.minimum(d[:, :, 1:], d[:, :, :-1] + 1)
        d[:, :, :-1] = torch.minimum(d[:, :, :-1], d[:, :, 1:] + 1)
        if torch.equal(prev, d):
            break
    return d


def _apply_fg_seed_recolor(g, side):
    """Every foreground object is recoloured to the colour of the NEAREST other-coloured fg object
    (min Manhattan over the object's cells; tie -> smaller colour). bg untouched. None when fewer
    than two fg colours exist."""
    col, _hw = _compact_colour(g, side)
    if col is None:
        return None
    bg = _background(col)
    objs = _corr_objects(g, side)
    if len(objs) < 2 or len({o["colour"] for o in objs}) < 2:
        return None
    d = _colour_dists(col, bg)
    out = g.clone()
    for o in objs:
        rs = torch.tensor([r for (r, _c2) in o["cells"]])
        cs = torch.tensor([c for (_r2, c) in o["cells"]])
        best_c, best_d = None, None
        for c in range(10):
            if c == o["colour"] or c == bg:
                continue
            dc = int(d[c, rs, cs].min())
            if dc >= 10_000:
                continue
            if best_d is None or dc < best_d or (dc == best_d and c < best_c):
                best_c, best_d = c, dc
        if best_c is None:
            return None
        for j in o["_flat"]:
            out[j] = best_c + COLOR_OFFSET
    return out


def fg_seed_recolor_predict(support_in, support_out, target_in, side):
    """FIX 2 foreground half: recolor each object to its nearest other-coloured seed. Parameter-free;
    verified by EXACT reconstruction of every support demo."""
    for i in range(support_in.shape[0]):
        rec = _apply_fg_seed_recolor(support_in[i], side)
        if rec is None or not torch.equal(rec, support_out[i]):
            return None
    return _apply_fg_seed_recolor(target_in, side)


def _apply_ray_connect(g, side):
    """3a: connect same-colour pairs along rows/columns -- bg cells strictly BETWEEN two cells of the
    same fg colour (with only bg between them) take that colour. None if no cell changes."""
    col, hw = _compact_colour(g, side)
    if col is None:
        return None
    bg = _background(col)
    H, W = col.shape
    out_col = col.clone()
    changed = False
    for lines, is_row in ((range(H), True), (range(W), False)):
        for i in lines:
            line = col[i, :] if is_row else col[:, i]
            fg_pos = [j for j in range(len(line)) if int(line[j]) != bg]
            for a, b in zip(fg_pos, fg_pos[1:]):
                if b - a > 1 and int(line[a]) == int(line[b]):
                    if is_row:
                        out_col[i, a + 1:b] = line[a]
                    else:
                        out_col[a + 1:b, i] = line[a]
                    changed = True
    if not changed:
        return None
    out = g.clone()
    H2, W2 = hw
    for r in range(H2):
        for c in range(W2):
            out[r * side + c] = int(out_col[r, c]) + COLOR_OFFSET
    return out


def ray_connect_predict(support_in, support_out, target_in, side):
    """3a ray/line projection: fill straight bg gaps between same-colour seeds. Parameter-free,
    verify-exact-on-all-support."""
    for i in range(support_in.shape[0]):
        rec = _apply_ray_connect(support_in[i], side)
        if rec is None or not torch.equal(rec, support_out[i]):
            return None
    return _apply_ray_connect(target_in, side)


def _fit_bbox_fill(support_in, support_out, side):
    """3b: fit {object colour -> bbox-interior fill colour} over bg cells inside each fg object's
    bbox. Cells outside every bbox (and fg cells) must be unchanged. None on inconsistency."""
    m: dict = {}
    for i in range(support_in.shape[0]):
        gi, go = support_in[i], support_out[i]
        col, _hw = _compact_colour(gi, side)
        if col is None:
            return None
        bg = _background(col)
        objs = _corr_objects(gi, side)
        if not objs:
            return None
        fill_mask = torch.zeros_like(gi, dtype=torch.bool)
        owner = {}
        for o in objs:
            for r in range(o["rmin"], o["rmax"] + 1):
                for c in range(o["cmin"], o["cmax"] + 1):
                    j = r * side + c
                    if int(gi[j]) == bg + COLOR_OFFSET:
                        fill_mask[j] = True
                        owner[j] = o["colour"]
        if not torch.equal(gi[~fill_mask], go[~fill_mask]):
            return None
        for j in fill_mask.nonzero(as_tuple=True)[0].tolist():
            d = int(go[j]) - COLOR_OFFSET
            if d < 0:
                return None
            k = owner[j]
            if m.get(k, d) != d:
                return None
            m[k] = d
    return m or None


def bbox_fill_predict(support_in, support_out, target_in, side):
    """3b bbox interior fill: bg cells inside an object's bbox -> fitted f(object colour)."""
    m = _fit_bbox_fill(support_in, support_out, side)
    if m is None:
        return None

    def _apply(g):
        col, _hw = _compact_colour(g, side)
        if col is None:
            return None
        bg = _background(col)
        out = g.clone()
        for o in _corr_objects(g, side):
            if o["colour"] not in m:
                continue
            for r in range(o["rmin"], o["rmax"] + 1):
                for c in range(o["cmin"], o["cmax"] + 1):
                    j = r * side + c
                    if int(g[j]) == bg + COLOR_OFFSET:
                        out[j] = m[o["colour"]] + COLOR_OFFSET
        return out

    for i in range(support_in.shape[0]):
        rec = _apply(support_in[i])
        if rec is None or not torch.equal(rec, support_out[i]):
            return None
    return _apply(target_in)


def _apply_periodic_fill(g, side):
    """3c: complete a periodic pattern -- find the smallest (pr, pc) tile on which all NON-bg cells
    agree, then fill bg cells from the tile consensus. None if no proper period or nothing to fill."""
    col, hw = _compact_colour(g, side)
    if col is None:
        return None
    bg = _background(col)
    H, W = col.shape
    if not bool((col == bg).any()):
        return None
    for pr in range(1, H + 1):
        for pc in range(1, W + 1):
            if pr == H and pc == W:
                return None                                   # trivial period = no completion power
            tile = [[-1] * pc for _ in range(pr)]
            ok = True
            for r in range(H):
                for c in range(W):
                    v = int(col[r, c])
                    if v == bg:
                        continue
                    t = tile[r % pr][c % pc]
                    if t == -1:
                        tile[r % pr][c % pc] = v
                    elif t != v:
                        ok = False
                        break
                if not ok:
                    break
            if not ok or any(t == -1 for row in tile for t in row):
                continue
            out = g.clone()
            for r in range(H):
                for c in range(W):
                    if int(col[r, c]) == bg:
                        out[r * side + c] = tile[r % pr][c % pc] + COLOR_OFFSET
            return out
    return None


def periodic_fill_predict(support_in, support_out, target_in, side):
    """3c periodic gap completion: bg cells take the value of the smallest consistent tile."""
    for i in range(support_in.shape[0]):
        rec = _apply_periodic_fill(support_in[i], side)
        if rec is None or not torch.equal(rec, support_out[i]):
            return None
    return _apply_periodic_fill(target_in, side)


PROJECTION_CANDIDATES = {
    "fg_seed_recolor": fg_seed_recolor_predict,          # FIX 2 fg half
    "ray_connect":     ray_connect_predict,              # FIX 3a
    "bbox_fill":       bbox_fill_predict,                # FIX 3b
    "periodic_fill":   periodic_fill_predict,            # FIX 3c
}

# --projection-ceiling verdict (2026-07-06, 400 eval tasks): ray_connect 1, bbox_fill 0,
# periodic_fill 0, fg_seed_recolor 1 -- 3a/3b/3c ALL below the >=10 gate; the "151-task" taxonomy
# family evaporates under the LODO-exact tax. Only the two 1-task winners are wired (floor-safe,
# verify-exact); bbox/periodic stay out of CANDIDATES as reproducible negatives.
WIRED_PROJECTION = {
    "fg_seed_recolor": fg_seed_recolor_predict,
    "ray_connect":     ray_connect_predict,
}


# ============================================================ OBJECT-WHERE x VALUE binding (Phase 0/1)
# The floor (per-src-colour modal map) IS where x value at COLOUR granularity, so a colour-only WHERE
# can never beat it. The only lift is an OBJECT-level predicate that SPLITS cells of the SAME colour
# into different output buckets -- and whose membership transfers by object INVARIANTS (core priors:
# size rank / border / holes / symmetry), NOT by cell-context signatures (cond_split measured those at
# 26.5% transfer). Same propose->verify->silent contract as Lane A and the size family.
OBJECT_PREDICATES = (
    "largest", "smallest", "singleton", "touches_border", "not_border",
    "holes>0", "holes==0", "sym_both",
)


def _object_predicate_masks(grid_flat, side, multi):
    """flat tokens -> {pred: [L] bool} marking cells whose INPUT object satisfies pred. Background /
    pad / non-colour cells are False in every mask. None if the grid has no colour content."""
    col, hw = _compact_colour(grid_flat, side)
    if col is None:
        return None
    bg = _background(col)
    objs = _objects(col, bg, multi=multi)
    L = side * side
    masks = {p: torch.zeros(L, dtype=torch.bool) for p in OBJECT_PREDICATES}
    if not objs:
        return masks
    H, W = hw
    sizes = [o["size"] for o in objs]
    max_sz, min_sz = max(sizes), min(sizes)
    for o in objs:
        holes = _hole_count(o)
        sh, sv = _symmetry(o)
        touches = (o["rmin"] == 0 or o["cmin"] == 0 or o["rmax"] == H - 1 or o["cmax"] == W - 1)
        flags = {
            "largest": o["size"] == max_sz, "smallest": o["size"] == min_sz,
            "singleton": o["size"] == 1, "touches_border": touches, "not_border": not touches,
            "holes>0": holes > 0, "holes==0": holes == 0, "sym_both": sh and sv,
        }
        idx = torch.tensor([r * side + c for (r, c) in o["cells"]], dtype=torch.long)
        for p, on in flags.items():
            if on:
                masks[p][idx] = True
    return masks


def _fit_partition_table(demos, mask_of, side):
    """Learn (in_W?, src_token) -> dst_token consistently across ALL demos under a cell partition.
    Non-colour cells must copy. Returns the table (dict) or None if any assignment is inconsistent."""
    table = {}
    for (x, y) in demos:
        m = mask_of(x)
        if m is None:
            return None
        xi, yi = x.long(), y.long()
        if xi.shape != yi.shape:
            return None
        colour = xi >= COLOR_OFFSET
        if not bool((xi[~colour] == yi[~colour]).all()):        # structure (pad/eos) must copy
            return None
        for j in colour.nonzero(as_tuple=True)[0].tolist():
            key = (bool(m[j]), int(xi[j]))
            v = int(yi[j])
            if table.setdefault(key, v) != v:
                return None
    return table


def _table_beats_floor(table) -> bool:
    """True iff the partition assigns some src colour DIFFERENT dsts inside vs outside W (the split the
    floor cannot express). A table that maps every colour identically in/out is floor-equivalent."""
    for (inW, src), dst in table.items():
        if inW and (False, src) in table and table[(False, src)] != dst:
            return True
    return False


def object_where_recolor_predict(support_in, support_out, target_in, side):
    """Phase-1 candidate: first object predicate (fixed order, mono then multi) whose partition table
    verifies on all support demos AND splits a colour better than the floor; apply to target. Silent
    (None) if none verifies or the target has an unseen (partition, colour) key."""
    demos = [(support_in[k], support_out[k]) for k in range(support_in.shape[0])]
    for multi in (False, True):
        for p in OBJECT_PREDICATES:
            def mask_of(x, _p=p, _m=multi):
                mm = _object_predicate_masks(x, side, _m)
                return None if mm is None else mm[_p]
            table = _fit_partition_table(demos, mask_of, side)
            if table is None or not _table_beats_floor(table):
                continue
            m = mask_of(target_in)
            if m is None:
                continue
            ti = target_in.long(); out = ti.clone(); ok = True
            for j in (ti >= COLOR_OFFSET).nonzero(as_tuple=True)[0].tolist():
                key = (bool(m[j]), int(ti[j]))
                if key not in table:
                    ok = False; break
                out[j] = table[key]
            if ok:
                return out
    return None


# ========================================================== REGION-WHERE x VALUE binding (Phase 0)
# Object-WHERE failed on conditional_recolor because many changed cells are background/empty
# regions rather than cells belonging to a foreground object. This probe tests negative-space
# regions with the same fit->verify->silent contract: a region mask must split a source token
# into copy outside W and a unique dst inside W across all support demos.
REGION_PREDICATES = (
    "background",
    "enclosed_background",
    "foreground_bbox_background",
    "row_gap_background",
    "col_gap_background",
)


def _region_predicate_masks(grid_flat, side):
    """flat tokens -> {pred: [L] bool} for background/negative-space regions.

    The masks are computed on the compact colour grid returned by object_rule_bank helpers and then
    placed back into the top-left side*side canvas. PAD/EOS outside the compact region are False.
    """
    col, hw = _compact_colour(grid_flat, side)
    L = side * side
    masks = {p: torch.zeros(L, dtype=torch.bool) for p in REGION_PREDICATES}
    if col is None:
        return masks
    H, W = hw
    bg = _background(col)
    bg_mask = col == bg
    fg_mask = (col >= 0) & ~bg_mask
    valid = col >= 0

    def _scatter(mask2d, name):
        if mask2d is None:
            return
        rr, cc = mask2d.nonzero(as_tuple=True)
        if rr.numel():
            masks[name][rr * side + cc] = True

    _scatter(bg_mask & valid, "background")

    # Enclosed background components: background regions that do not touch the compact grid border.
    seen = torch.zeros_like(bg_mask, dtype=torch.bool)
    enclosed = torch.zeros_like(bg_mask, dtype=torch.bool)
    for r0, c0 in bg_mask.nonzero(as_tuple=False).tolist():
        if bool(seen[r0, c0]):
            continue
        stack = [(r0, c0)]
        cells = []
        touches_edge = False
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            touches_edge = touches_edge or r == 0 or c == 0 or r == H - 1 or c == W - 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and bool(bg_mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        if not touches_edge:
            for r, c in cells:
                enclosed[r, c] = True
    _scatter(enclosed, "enclosed_background")

    if bool(fg_mask.any()):
        rr, cc = fg_mask.nonzero(as_tuple=True)
        r0, r1 = int(rr.min()), int(rr.max())
        c0, c1 = int(cc.min()), int(cc.max())
        bbox_bg = torch.zeros_like(bg_mask, dtype=torch.bool)
        bbox_bg[r0:r1 + 1, c0:c1 + 1] = True
        _scatter(bbox_bg & bg_mask, "foreground_bbox_background")

    # Row/column bounded gaps: background cells with foreground on both sides along an axis.
    row_gap = torch.zeros_like(bg_mask, dtype=torch.bool)
    col_gap = torch.zeros_like(bg_mask, dtype=torch.bool)
    for r in range(H):
        fcols = fg_mask[r].nonzero(as_tuple=True)[0]
        if fcols.numel() >= 2:
            row_gap[r, int(fcols.min()):int(fcols.max()) + 1] = True
    for c in range(W):
        frows = fg_mask[:, c].nonzero(as_tuple=True)[0]
        if frows.numel() >= 2:
            col_gap[int(frows.min()):int(frows.max()) + 1, c] = True
    _scatter(row_gap & bg_mask, "row_gap_background")
    _scatter(col_gap & bg_mask, "col_gap_background")
    return masks


def region_where_fill_predict(support_in, support_out, target_in, side):
    """Phase-0 candidate: first region predicate whose partition table verifies on support demos.

    It is intentionally not registered in CANDIDATES here; --region-value-ceiling measures whether
    this family has enough coverage before runtime or compose integration is justified.
    """
    demos = [(support_in[k], support_out[k]) for k in range(support_in.shape[0])]
    for p in REGION_PREDICATES:
        def mask_of(x, _p=p):
            return _region_predicate_masks(x, side)[_p]
        table = _fit_partition_table(demos, mask_of, side)
        if table is None or not _table_beats_floor(table):
            continue
        m = mask_of(target_in)
        ti = target_in.long()
        out = ti.clone()
        ok = True
        for j in (ti >= COLOR_OFFSET).nonzero(as_tuple=True)[0].tolist():
            key = (bool(m[j]), int(ti[j]))
            if key not in table:
                ok = False
                break
            out[j] = table[key]
        if ok:
            return out
    return None


# ========================================================== CONSTANT region fill ceiling (read-only)
def _fit_constant_fill(demos, mask_of, side):
    """Fit one constant output token for a verified input-defined region.

    Contract: every support demo must be exactly expressible as:
      output[j] = fill_token   if mask(input)[j]
      output[j] = input[j]     otherwise

    This is deliberately stricter than the region-WHERE partition table above. It measures whether
    the support rule is a true constant fill, not a source-colour-conditioned recolour.
    """
    fill_token = None
    wrote_change = False
    for src, dst in demos:
        src = src.long()
        dst = dst.long()
        if src.shape != dst.shape:
            return None
        mask = mask_of(src)
        if mask is None or mask.shape != src.shape:
            return None
        # Constant-fill is colour-only: it may not explain PAD/EOS/shape edits.
        if bool(((mask) & (dst < COLOR_OFFSET)).any()):
            return None
        inside = mask & (src >= COLOR_OFFSET)
        outside = ~mask
        if bool(outside.any()) and not bool((src[outside] == dst[outside]).all()):
            return None
        if not bool(inside.any()):
            return None
        vals = torch.unique(dst[inside])
        if vals.numel() != 1:
            return None
        v = int(vals[0])
        if fill_token is None:
            fill_token = v
        elif fill_token != v:
            return None
        wrote_change = wrote_change or bool((src[inside] != dst[inside]).any())
    if fill_token is None or not wrote_change:
        return None
    return fill_token


def constant_fill_predict(support_in, support_out, target_in, side):
    """Read-only ceiling candidate: verified region mask -> one constant colour.

    Not registered in CANDIDATES. It exists to answer whether simple constant-fill structure has
    support-verified coverage before any live path receives such evidence.
    """
    demos = [(support_in[k], support_out[k]) for k in range(support_in.shape[0])]
    for p in REGION_PREDICATES:
        def mask_of(x, _p=p):
            return _region_predicate_masks(x, side)[_p]
        fill = _fit_constant_fill(demos, mask_of, side)
        if fill is None:
            continue
        out = target_in.long().clone()
        mask = mask_of(target_in)
        if mask is None or mask.shape != out.shape:
            continue
        out[mask & (out >= COLOR_OFFSET)] = fill
        return out
    return None


# --------------------------------------------------------- verified conditional split (WHERE x VALUE)
# Predicate lattice, GENERAL and fixed-order (first verified wins -> deterministic). Structural
# predicates first (src-colour-agnostic), then palette-driven contact predicates. This is the
# R17 neighbour-conditioned recolour done as propose->verify->select: where neighbour_predict's
# exact 5-tuple signature goes silent on unseen neighbourhoods, a VERIFIED predicate still fires.
PREDICATE_ORDER = (
    ["same4>=1", "same4>=2", "same4>=3", "same4>=4", "edge"]
    + [f"touch4:{t}" for t in range(COLOR_OFFSET, COLOR_OFFSET + N_COLORS)]
    + [f"touch8:{t}" for t in range(COLOR_OFFSET, COLOR_OFFSET + N_COLORS)]
)


def _predicate_masks(grid_flat, side):
    """Flat token grid -> {predicate name: [L] bool}. Same machinery for demos and target."""
    S = side
    g = grid_flat.view(S, S).long()
    gp = torch.full((S + 2, S + 2), -1, dtype=torch.long)
    gp[1:S + 1, 1:S + 1] = g
    n4 = torch.stack([gp[0:S, 1:S + 1], gp[2:S + 2, 1:S + 1],
                      gp[1:S + 1, 0:S], gp[1:S + 1, 2:S + 2]], 0)              # [4,S,S]
    d4 = torch.stack([gp[0:S, 0:S], gp[0:S, 2:S + 2],
                      gp[2:S + 2, 0:S], gp[2:S + 2, 2:S + 2]], 0)
    n8 = torch.cat([n4, d4], 0)                                                # [8,S,S]
    masks = {}
    same4 = (n4 == g.unsqueeze(0)).sum(0)
    for k in (1, 2, 3, 4):
        masks[f"same4>={k}"] = (same4 >= k).reshape(-1)
    masks["edge"] = ((n4 < COLOR_OFFSET) & (n4 != g.unsqueeze(0))).any(0).reshape(-1)
    for t in range(COLOR_OFFSET, COLOR_OFFSET + N_COLORS):
        masks[f"touch4:{t}"] = (n4 == t).any(0).reshape(-1)
        masks[f"touch8:{t}"] = (n8 == t).any(0).reshape(-1)
    return masks


def conditional_split_predict(support_in, support_out, target_in, side):
    """Verified WHERE x VALUE binding. For each CONDITIONAL input colour (>1 distinct outputs in
    support), search PREDICATE_ORDER for a binary split that assigns EVERY support cell of that
    colour consistently (pred -> d_true, ~pred -> d_false, across ALL demos). Only verified
    splits fire; everything else stays on the floor -- a wrong rule cannot be selected."""
    pred = floor_predict(support_in, support_out, target_in, side)
    _, _, cooc = build_cond(support_in, support_out)
    cond_cols = ((cooc > 0).sum(1) > 1).nonzero(as_tuple=True)[0]
    if cond_cols.numel() == 0:
        return pred
    m = support_in.shape[0]
    sup_masks = [_predicate_masks(support_in[k], side) for k in range(m)]
    tgt_masks = _predicate_masks(target_in, side)
    ti = target_in.long()
    out = pred.clone()
    for a in cond_cols.tolist():
        tok = a + COLOR_OFFSET
        for name in PREDICATE_ORDER:
            ok = True
            d_true = d_false = None
            for k in range(m):
                x, y = support_in[k].long(), support_out[k].long()
                cells = x == tok
                if not bool(cells.any()):
                    continue
                mk = sup_masks[k][name]
                for is_true, sel in ((True, cells & mk), (False, cells & ~mk)):
                    if not bool(sel.any()):
                        continue
                    vals = torch.unique(y[sel])
                    if vals.numel() != 1:
                        ok = False
                        break
                    v = int(vals[0])
                    if is_true:
                        if d_true is None:
                            d_true = v
                        elif d_true != v:
                            ok = False
                            break
                    else:
                        if d_false is None:
                            d_false = v
                        elif d_false != v:
                            ok = False
                            break
                if not ok:
                    break
            if ok and d_true is not None and d_false is not None and d_true != d_false:
                out[(ti == tok) & tgt_masks[name]] = d_true
                out[(ti == tok) & ~tgt_masks[name]] = d_false
                break
    return out


def committed_predict_with_trace(support_in, support_out, target_in, side, rule_lib: RuleLibrary | None = None):
    """The full deterministic DSL PROPOSE stack (Stage 2) + ABSTRACT Rule Library (Stage 5)."""
    gin, gout = {}, {}
    m = support_in.shape[0]
    for i in range(m):
        gin[i] = _flat_to_compact_grid(support_in[i], side)
        gout[i] = _flat_to_compact_grid(support_out[i], side)
    h = m
    gin[h] = _flat_to_compact_grid(target_in, side)
    sup = list(range(m))

    from solve import _committed_solve, _default_geo_ops
    from parse import _parse_same
    
    geo_ops = _default_geo_ops()
    labels = {k: _parse_same(v) for k, v in gin.items()}
    if rule_lib is None:
        rule_lib = RuleLibrary()
        
    a1, a2, trace = _committed_solve(
        gin, gout, sup, h, geo_ops, labels, rule_lib=rule_lib, return_trace=True
    )
    if a1 is not None:
        return _compact_to_flat(a1, side), trace
    return target_in.long().clone(), trace


def committed_predict(support_in, support_out, target_in, side):
    pred, _trace = committed_predict_with_trace(support_in, support_out, target_in, side)
    return pred


def _solver_dicts(support_in, support_out, target_in, side):
    gin, gout = {}, {}
    m = support_in.shape[0]
    for i in range(m):
        gin[i] = _flat_to_compact_grid(support_in[i], side)
        gout[i] = _flat_to_compact_grid(support_out[i], side)
    h = m
    gin[h] = _flat_to_compact_grid(target_in, side)
    labels = {k: _parse_same(v) for k, v in gin.items()}
    return gin, gout, list(range(m)), h, labels


def set_cover_predict(support_in, support_out, target_in, side):
    """Split U6 candidate: mixed safe clauses, scored directly by the selector."""
    gin, gout, sup, h, labels = _solver_dicts(support_in, support_out, target_in, side)
    pred = _set_cover_predict(gin, gout, sup, h, labels)
    return None if pred is None else _compact_to_flat(pred, side)


def relation_predictor(relation):
    def _predict(support_in, support_out, target_in, side):
        gin, gout, sup, h, labels = _solver_dicts(support_in, support_out, target_in, side)
        pred = _object_relrecolor_predict(gin, gout, sup, h, labels, relation)
        return None if pred is None else _compact_to_flat(pred, side)

    _predict.__name__ = f"rel_{relation}_predict"
    return _predict


RELATION_CANDIDATES = {
    f"rel_{rel}": relation_predictor(rel)
    for rel in ("nearest", "container", "contained", "aligned", "between", "largest", "smallest")
}


CANDIDATES = {"identity": identity_predict, "floor": floor_predict,
              "analogy": analogy_predict, "neighbour": neighbour_predict,
              "cond_split": conditional_split_predict,
              "analogy_gated": analogy_gated_predict, "set_cover": set_cover_predict,
              **SIZE_CANDIDATES,
              **FIXF_CANDIDATES,
              **WIRED_PROJECTION,
              **RELATION_CANDIDATES, "committed": committed_predict}

# candidates that remain meaningful when input and output shapes differ
NONSHAPE_OK = frozenset({"identity"}) | frozenset(SIZE_CANDIDATES)


# ---------------------------------------------------------------------------- LODO scoring + select
def score_lodo(demos, predict_fn, side):
    """demos = list of (in_flat[L], out_flat[L]). Hold out each, predict from the rest. -> (exact, sim)."""
    M = len(demos)
    if M < 2:
        return None
    ex, sm = [], []
    for i in range(M):
        sup = [d for j, d in enumerate(demos) if j != i]
        sup_in = torch.stack([d[0] for d in sup])
        sup_out = torch.stack([d[1] for d in sup])
        ti, to = demos[i]
        pred = predict_fn(sup_in, sup_out, ti, side)
        if pred is None or pred.shape != to.shape:
            ex.append(0.0)
            sm.append(0.0)
            continue
        eq = (pred == to.long())
        ex.append(float(eq.all()))
        sm.append(float(eq.float().mean()))
    return sum(ex) / M, sum(sm) / M


def score_committed_lodo(demos, side, rule_lib: RuleLibrary | None = None):
    """LODO score for the committed candidate, retaining the internal attempt1 source per fold."""
    M = len(demos)
    if M < 2:
        return None, None
    ex, sm = [], []
    sources, exact_by_source = Counter(), Counter()
    folds = []
    for i in range(M):
        sup = [d for j, d in enumerate(demos) if j != i]
        sup_in = torch.stack([d[0] for d in sup])
        sup_out = torch.stack([d[1] for d in sup])
        ti, to = demos[i]
        pred, trace = committed_predict_with_trace(sup_in, sup_out, ti, side, rule_lib=rule_lib)
        eq = (pred == to.long())
        exact = float(eq.all())
        sim = float(eq.float().mean())
        src = str(trace.get("attempt1_source", "unknown"))
        sources[src] += 1
        exact_by_source[src] += int(exact)
        folds.append({"holdout": i, "source": src, "exact": exact, "sim": sim})
        ex.append(exact)
        sm.append(sim)
    top_source = sources.most_common(1)[0][0] if sources else "unknown"
    return (sum(ex) / M, sum(sm) / M), {
        "sources": dict(sources),
        "exact_by_source": dict(exact_by_source),
        "top_source": top_source,
        "folds": folds,
    }


def fire_stats(demos, predict_fn, side):
    """How often does a candidate OVERRIDE the floor, and is it right when it does? The exact-failure
    microscope: fire~0 => it never acts (retrieval/threshold dead); fire high + precision low => it
    acts wrongly; fire ok + precision ok but exact still 0 => the exactness tax (fixes some, not all)."""
    M = len(demos)
    fire = right = floor_was_right = 0
    for i in range(M):
        sup = [d for j, d in enumerate(demos) if j != i]
        sin = torch.stack([d[0] for d in sup]); sout = torch.stack([d[1] for d in sup])
        ti, to = demos[i]
        fl = floor_predict(sin, sout, ti, side); pr = predict_fn(sin, sout, ti, side)
        isc = ti.long() >= COLOR_OFFSET
        diff = isc & (pr != fl)                                   # cells the candidate changed
        fire += int(diff.sum())
        right += int((diff & (pr == to.long())).sum())           # ... and got correct
        floor_was_right += int((diff & (fl == to.long())).sum()) # ... that the floor already had right
    return fire, right, floor_was_right


def oracle_lodo(demos, side):
    """Upper bound: per fold, per CELL, is ANY candidate correct? all cells covered -> fold exact.
    Tells us whether the answer is even RECOVERABLE from the candidate set (before betting a run).
      oracle high -> the signal is THERE; a per-cell composer / learned selector would capture it.
      oracle low  -> the candidate set does not contain the answer; need a richer mechanism."""
    M = len(demos)
    if M < 2:
        return None
    ex, cov = [], []
    for i in range(M):
        sup = [d for j, d in enumerate(demos) if j != i]
        sin = torch.stack([d[0] for d in sup]); sout = torch.stack([d[1] for d in sup])
        ti, to = demos[i]
        pred_list = []
        for fn in CANDIDATES.values():
            pred = fn(sin, sout, ti, side)
            if pred is not None and pred.shape == to.shape:
                pred_list.append(pred)
        if not pred_list:
            ex.append(0.0); cov.append(0.0); continue
        preds = torch.stack(pred_list)                                                 # [C,L]
        anyc = (preds == to.long()).any(0)                                              # [L]
        ex.append(float(anyc.all())); cov.append(float(anyc.float().mean()))
    return sum(ex) / M, sum(cov) / M


def score_model_dump_lodo(demos, side, task_id, model_dump):
    """LODO score for fold-indexed model candidates from model_candidate_dump.

    Expected keys are (task_id, heldout_demo_index). Candidate 0 is the floor;
    attempt1 is the first distinct non-floor candidate if present, attempt2 is
    the floor. This mirrors the eval bridge without peeking at the held-out
    output during selection.
    """
    if task_id is None or model_dump is None:
        return None
    M = len(demos)
    if M < 2:
        return None
    ex, sm = [], []
    for i in range(M):
        rec = model_dump.get((str(task_id), int(i)))
        if rec is None:
            ex.append(0.0)
            sm.append(0.0)
            continue
        cand = torch.as_tensor(rec["candidates"], dtype=torch.long)
        if cand.ndim != 2 or cand.shape[1] != side * side:
            raise ValueError(f"model_dump candidate for {(task_id, i)} must be [K,{side * side}], got {tuple(cand.shape)}")
        floor = cand[0]
        a1 = floor
        for row in cand[1:]:
            if not bool((row == floor).all()):
                a1 = row
                break
        _ti, to = demos[i]
        target = to.long()
        eq1 = a1 == target
        eq2 = floor == target
        exact = float(bool(eq1.all()) or bool(eq2.all()))
        sim = max(float(eq1.float().mean()), float(eq2.float().mean()))
        ex.append(exact)
        sm.append(sim)
    return sum(ex) / M, sum(sm) / M


def relocate_k_predictor(k_index: int):
    """Return a predictor for the k-th verified rearrangement candidate.

    Each k must be LODO-scored independently. Reusing k=0's score for k=1
    overstates the second frame and can waste the ARC second attempt.
    """
    def _predict(support_in, support_out, target_in, side):
        from models.recursive_reasoning.object_rule_bank import rearrange_candidates

        cands = rearrange_candidates(support_in, support_out, target_in, side, k=k_index + 1)
        return cands[k_index][0] if len(cands) > k_index else None

    _predict.__name__ = f"relocate_at_{k_index + 1}_predict"
    return _predict


def score_relocate_k_lodo(demos, side, k_index: int):
    return score_lodo(demos, relocate_k_predictor(k_index), side)


def model_dump_predictions_for_test(model_dump, task_id, test_index, side, include_floor: bool = True):
    """Return flat token predictions from a target-test model dump record.

    Candidate row 0 is the immutable model floor. The first prediction is the
    highest-ranked distinct non-floor hypothesis if present; the floor is then
    appended as the conservative fallback. No labels or verifier scores are
    read here.
    """
    if model_dump is None or task_id is None:
        return []
    rec = model_dump.get((str(task_id), int(test_index)))
    if rec is None:
        return []
    cand = torch.as_tensor(rec["candidates"], dtype=torch.long)
    if cand.ndim != 2 or cand.shape[1] != side * side:
        raise ValueError(
            f"model_dump test candidate for {(task_id, test_index)} must be [K,{side * side}], "
            f"got {tuple(cand.shape)}"
        )
    floor = cand[0]
    out = []
    for row in cand[1:]:
        if not bool((row == floor).all()):
            out.append(row.clone())
            break
    if include_floor:
        out.append(floor.clone())
    return out


def _flat_to_compact_grid(seq: torch.Tensor, side: int) -> torch.Tensor | None:
    g = seq.long().reshape(side, side)
    valid = g >= COLOR_OFFSET
    if not bool(valid.any()):
        return None
    nz = valid.nonzero(as_tuple=False)
    h = int(nz[:, 0].max().item()) + 1
    w = int(nz[:, 1].max().item()) + 1
    return g[:h, :w].clone()


def _compact_to_flat(g: torch.Tensor, side: int) -> torch.Tensor:
    out = torch.zeros(side, side, dtype=torch.long)
    h = min(int(g.shape[0]), side)
    w = min(int(g.shape[1]), side)
    out[:h, :w] = g[:h, :w].long()
    return out.reshape(-1)


def _grid_dicts_from_demos(demos, side):
    gin, gout = {}, {}
    for i, (x, y) in enumerate(demos):
        gin[i] = _flat_to_compact_grid(x, side)
        gout[i] = _flat_to_compact_grid(y, side)
    return gin, gout, list(range(len(demos)))


def select_winner(scores: dict) -> str:
    """Pick the winning candidate name from {name: (exact, sim)}.

    Order: EXACT primary, SIM secondary, then FLOOR on a full (exact, sim) tie (conservative). Because
    exact is primary and `floor` is always in the set, the winner's exact == max exact >= floor's exact
    -> `selector_exact >= floor_exact` BY CONSTRUCTION (net-negative is impossible). The floor tie-bonus
    is conservatism only; it does NOT create the guarantee. Extracted + adversarially tested in _self_test
    so a future reorder of the key (e.g. sim-first) cannot silently break the guarantee.
    """
    def key(name):
        ex, sm = scores[name]
        return (ex, sm, 1.0 if name == "floor" else 0.0)
    return max(scores, key=key)


def evaluate_task(
    demos,
    side,
    task_id=None,
    model_dump=None,
    rule_lib: RuleLibrary | None = None,
    family: str | None = None,
    refine_close_miss: bool = False,
):
    """Score every candidate by LODO; select best (floor in set, tie -> floor)."""
    scores = {}
    committed_trace = None
    for name, fn in CANDIDATES.items():
        if name == "committed":
            s, committed_trace = score_committed_lodo(demos, side, rule_lib=rule_lib)
        else:
            s = score_lodo(demos, fn, side)
        if s is not None:
            scores[name] = s
    md = score_model_dump_lodo(demos, side, task_id=task_id, model_dump=model_dump)
    if md is not None:
        scores["model_dump"] = md
    refinement = None
    if refine_close_miss:
        gin, gout, valid = _grid_dicts_from_demos(demos, side)
        refinement = refine_lodo_recipes(
            gin,
            gout,
            valid,
            _default_geo_ops(),
            rule_lib=rule_lib,
            family=family,
        )
        if refinement.get("stable_recipe") is not None and refinement.get("n_folds", 0) > 0:
            exact = float(refinement["exact_folds"]) / max(float(refinement["n_folds"]), 1.0)
            scores["refined_recipe"] = (exact, exact)
    if not scores:
        return None
    winner = select_winner(scores)
    sel_exact = scores[winner][0]
    winner_detail = winner
    if winner == "committed" and committed_trace is not None:
        winner_detail = f"committed:{committed_trace['top_source']}"
    return {"scores": scores, "winner": winner, "winner_detail": winner_detail,
            "committed_trace": committed_trace, "refinement": refinement,
            "selector_exact": sel_exact, "floor_exact": scores.get("floor", (0.0, 0.0))[0]}


# --------------------------------------------------------------------------------- real-data driver
def embed(grid, side=30):
    """HxW int grid (colours 0..9) -> flat [side*side] tokens (colour+2, pad=0, top-left placed)."""
    canvas = torch.zeros(side, side, dtype=torch.long)
    h = min(len(grid), side)
    for r in range(h):
        row = grid[r]
        w = min(len(row), side)
        for c in range(w):
            canvas[r, c] = int(row[c]) + COLOR_OFFSET
    return canvas.reshape(-1)


def same_shape(task):
    return all(len(p["input"]) == len(p["output"]) and
               len(p["input"][0]) == len(p["output"][0]) for p in task["train"])


def format_candidate_row(name: str, exact_sum: float, sim_sum: float, count: int, n_tasks: int) -> str:
    if count <= 0:
        return f"{name:12} {'--':>10} {'--':>9} {0:>8}/{max(n_tasks, 1):<8}"
    return (
        f"{name:12} {exact_sum / count * 100:>10.1f}% "
        f"{sim_sum / count * 100:>9.1f}% {count:>8}/{max(n_tasks, 1):<8}"
    )


def run_real(json_path, side=30, max_tasks=0, model_dump_path=None, rule_lib_path=None, refine_close_miss=False):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    model_dump = load_model_dump(model_dump_path, side=side, kind="fold") if model_dump_path else None
    rule_lib = RuleLibrary().load(rule_lib_path) if rule_lib_path else None
    candidate_names = list(CANDIDATES) + (["model_dump"] if model_dump is not None else [])
    if refine_close_miss:
        candidate_names.append("refined_recipe")
    agg = {n: [0.0, 0.0, 0] for n in candidate_names}    # exact_sum, sim_sum, n
    sel_sum = floor_sum = orac_sum = orac_cov = 0.0
    beat = ge = n_tasks = 0
    fire = {n: [0, 0, 0] for n in ("analogy", "neighbour", "analogy_gated")}   # fire, right, floor_was_right
    committed_sources = Counter()
    committed_exact_sources = Counter()
    for tid, task in data.items():
        if not same_shape(task):
            continue                                 # recolour candidates assume same shape
        demos = [(embed(p["input"], side), embed(p["output"], side)) for p in task["train"]]
        res = evaluate_task(
            demos,
            side,
            task_id=tid,
            model_dump=model_dump,
            rule_lib=rule_lib,
            family=None,
            refine_close_miss=refine_close_miss,
        )
        if res is None:
            continue
        n_tasks += 1
        for n in candidate_names:
            if n in res["scores"]:
                agg[n][0] += res["scores"][n][0]; agg[n][1] += res["scores"][n][1]; agg[n][2] += 1
        sel_sum += res["selector_exact"]; floor_sum += res["floor_exact"]
        orac = oracle_lodo(demos, side)
        if orac is not None:
            orac_sum += orac[0]; orac_cov += orac[1]
        for nm in fire:
            f, r, fw = fire_stats(demos, CANDIDATES[nm], side)
            fire[nm][0] += f; fire[nm][1] += r; fire[nm][2] += fw
        if res["scores"].get("analogy", (0,))[0] > res["floor_exact"]:
            beat += 1
        if res["selector_exact"] >= res["floor_exact"] - 1e-9:
            ge += 1
        tr = res.get("committed_trace") or {}
        committed_sources.update(tr.get("sources", {}))
        committed_exact_sources.update(tr.get("exact_by_source", {}))
        if max_tasks and n_tasks >= max_tasks:
            break

    if rule_lib_path and rule_lib is not None:
        rule_lib.save(rule_lib_path)

    print(f"\n=== OFFLINE CANDIDATE SCORE  (real same-shape tasks: {n_tasks}) ===")
    print(f"{'candidate':12} {'LODO exact':>11} {'cell sim':>10} {'tasks':>17}")
    for n in candidate_names:
        e, s, c = agg[n]
        if c:
            print(format_candidate_row(n, e, s, c, n_tasks))
    print(f"\nselector LODO exact : {sel_sum/max(n_tasks,1)*100:5.1f}%   (floor {floor_sum/max(n_tasks,1)*100:.1f}%)")
    print(f"ORACLE  LODO exact  : {orac_sum/max(n_tasks,1)*100:5.1f}%   (per-cell coverage {orac_cov/max(n_tasks,1)*100:.1f}%)")
    print(f"   -> oracle HIGH = answer is recoverable from the candidates (build a composer / learned selector)")
    print(f"   -> oracle LOW  = candidate set lacks the answer (need a richer mechanism, e.g. cell-level copy)")
    print(f"analogy BEATS floor : {beat}/{n_tasks} tasks ({beat/max(n_tasks,1)*100:.1f}%)")
    print(f"selector >= floor   : {ge}/{n_tasks} tasks ({ge/max(n_tasks,1)*100:.1f}%)  <- must be 100% (the guarantee)")
    if committed_sources:
        print("\ncommitted internal sources (attempt1 folds):")
        for src, cnt in committed_sources.most_common(10):
            ex = committed_exact_sources.get(src, 0)
            print(f"  {src:28s} folds={cnt:5d} exact={ex:5d} ({ex / max(cnt, 1) * 100:4.1f}%)")
    if rule_lib is not None:
        print("\n[RuleLibrary] " + rule_lib.summary())
    print(f"\n--- EXACT-FAILURE microscope (override of the floor, on held-out colour cells) ---")
    for nm in fire:
        f, r, fw = fire[nm]
        prec = r / max(f, 1) * 100
        harm = fw / max(f, 1) * 100
        print(f"{nm:10}: fires on {f:6d} cells | {prec:4.1f}% correct when it fires | "
              f"{harm:4.1f}% of fires OVERWROTE a cell the floor already had right")


# ----------------------------------------------------------------- RULE-HYPOTHESIS recall probe (gate)
def run_rule_probe(side=30, challenges=None, categories_csv=None):
    """KILL-GATE for the rule-hypothesis bus. For each atlas-labelled task, run infer_rule_hypotheses
    on the support demos and measure:
      (1) coarse family recall vs the atlas category   -- expected ~100% (categories are invariant-defined;
          identifying 'this is a rearrangement' is trivial -> NOT the real gate),
      (2) WITHIN-rearrangement binding recovery + cross-demo consistency  -- the LOAD-BEARING number:
          a rule token is only actionable if it carries a consistent binding (slide/translate/sort), not
          just the label. Low binding recall => the live bus would feed a label the TRM already infers
          from histogram-preservation, so DON'T wire it.
    Read-only; no model, no GPU, no training."""
    import csv as _csv
    import collections as _c
    from models.recursive_reasoning.object_rule_bank import infer_rule_hypotheses
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    if not chal_path.exists() or not csv_path.exists():
        print(f"[skip --rule-probe] missing {chal_path if not chal_path.exists() else csv_path}")
        return
    chal = json.loads(chal_path.read_text(encoding="utf-8"))
    cat = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            cat[row["task_id"]] = row["category"]
    fam2cat = {"rearrange": "rearrangement", "recolor": "conditional_recolor",
               "size_change": "size_change", "identity": "identity"}
    n = _c.Counter(); top1 = _c.Counter(); top2 = _c.Counter()
    confusion = _c.defaultdict(_c.Counter)
    rearr_binding = _c.Counter(); rearr_total = 0
    for tid, label in cat.items():
        if tid not in chal:
            continue
        demos = [(embed(p["input"], side), embed(p["output"], side)) for p in chal[tid]["train"]]
        si = torch.stack([d[0] for d in demos]); so = torch.stack([d[1] for d in demos])
        ranked = infer_rule_hypotheses(si, so, side)
        cats = [fam2cat.get(h["family"], h["family"]) for h in ranked]
        n[label] += 1
        if cats and cats[0] == label:
            top1[label] += 1
        if label in cats[:2]:
            top2[label] += 1
        confusion[label][cats[0] if cats else "none"] += 1
        if label == "rearrangement":
            rearr_total += 1
            rh = next((h for h in ranked if h["family"] == "rearrange"), None)
            if rh is not None and "binding" in rh:
                bn, bd = (int(x) for x in rh["binding_consistency"].split("/"))
                tag = rh["binding"][0]
                if bn == bd and tag in ("translate", "slide_to_edge", "directional", "sort_pack"):
                    rearr_binding[("PARAMETRIC consistent", tag)] += 1
                elif bn == bd:
                    rearr_binding[("moved (generic) consistent", tag)] += 1
                else:
                    rearr_binding[("mixed across demos", tag)] += 1
            else:
                rearr_binding[("no binding", "-")] += 1

    print(f"\n=== RULE-HYPOTHESIS PROBE  (atlas-labelled eval tasks: {sum(n.values())}) ===")
    print(f"{'category':22} {'n':>4} {'top1':>7} {'top2':>7}")
    for label in ("rearrangement", "conditional_recolor", "size_change"):
        if n[label]:
            print(f"{label:22} {n[label]:>4} {top1[label]/n[label]*100:>6.1f}% {top2[label]/n[label]*100:>6.1f}%")
    print("\nconfusion (true -> predicted top-1):")
    for label in ("rearrangement", "conditional_recolor", "size_change"):
        if n[label]:
            row = ", ".join(f"{k}:{v}" for k, v in confusion[label].most_common())
            print(f"  {label:22} -> {row}")
    print(f"\n--- THE REAL GATE: within-rearrangement binding ({rearr_total} tasks) ---")
    for (kind, tag), c in rearr_binding.most_common():
        print(f"  {c:3d}  {kind:28} {tag}")
    param = sum(c for (k, _t), c in rearr_binding.items() if k.startswith("PARAMETRIC"))
    print(f"\n  PARAMETRIC+consistent binding recall: {param}/{rearr_total} "
          f"({param / max(rearr_total, 1) * 100:.1f}%)  <- go/no-go for the live rule-bus")
    print("  (>= ~80% => a rule token carries an actionable, cross-demo-consistent binding -> wire the bus;")
    print("   low => the bank emits a label the TRM already gets from hist-preservation -> fix the parse first)")


def run_boundary_probe(side=30, challenges=None, categories_csv=None):
    """Test the MOVE-TO-BOUNDARY generalization on the 20 rearrangement tasks. For each task, match
    objects by shape and measure what fraction of moved objects slide to a boundary (edge or another
    object), and whether the 'nearest-edge' selector explains the directions. This is the evidence
    for/against the central primitive before building features around it."""
    import csv as _csv
    from models.recursive_reasoning.object_rule_bank import boundary_move_eval, _compact_colour
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    if not chal_path.exists() or not csv_path.exists():
        print(f"[skip --boundary-probe] missing inputs"); return
    chal = json.loads(chal_path.read_text(encoding="utf-8"))
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rid = [r["task_id"] for r in _csv.DictReader(f)
               if r["category"] == "rearrangement"]
    print(f"\n=== MOVE-TO-BOUNDARY PROBE  (rearrangement tasks: {len(rid)}) ===")
    print(f"{'task':10} {'matched/obj':>12} {'at-boundary':>12} {'nearest-edge':>13}  verdict")
    full_b = full_match = near_sel = 0
    for tid in rid:
        if tid not in chal:
            continue
        demos = chal[tid]["train"]
        tot_obj = tot_match = tot_b = tot_near = 0
        for p in demos:
            ci, _ = _compact_colour(embed(p["input"], side), side)
            co, _ = _compact_colour(embed(p["output"], side), side)
            r = boundary_move_eval(ci, co)
            tot_obj += r["n_obj"]; tot_match += r["matched"]
            tot_b += r["at_boundary"]; tot_near += r["nearest_edge_ok"]
        moved = max(tot_match, 1)
        bfrac = tot_b / moved; nfrac = tot_near / moved; mfrac = tot_match / max(tot_obj, 1)
        verdict = ("BOUNDARY" if bfrac >= 0.95 else "partial" if bfrac >= 0.6 else "-")
        if bfrac >= 0.95: full_b += 1
        if mfrac >= 0.95: full_match += 1
        if nfrac >= 0.95: near_sel += 1
        print(f"{tid:10} {tot_match:4d}/{tot_obj:<6d} {bfrac*100:>10.0f}% {nfrac*100:>11.0f}%   {verdict}")
    n = len([t for t in rid if t in chal])
    print(f"\n  objects matched by shape (>=95%):     {full_match}/{n}")
    print(f"  ALL moves end at a boundary (>=95%):  {full_b}/{n}   <- move-to-boundary is ONE frame, not universal")
    print(f"  nearest-edge selector explains it:    {near_sel}/{n}")
    print("  (low => the target is NOT 'flush to a wall' for most tasks; the relation is attribute-keyed")
    print("   assignment -- e.g. f45f5ca7 is colour->column, not gravity. Generalize analogy_recolour to")
    print("   analogy_relocate: copy the analogous demo object's OUTPUT POSITION-RELATION, frame chosen by verify.)")


def run_relocate_probe(side=30, challenges=None, categories_csv=None):
    """Exact-solve count for the analogy_relocate solver (Lane A) on the 20 rearrangement tasks.
    LODO: hold out each demo, learn the recipe from the rest, reconstruct the held-out output. A task
    counts as SOLVED if every fold reconstructs exactly. Reports the winning frame per task."""
    import csv as _csv
    import collections as _c
    from models.recursive_reasoning.object_rule_bank import rearrange_candidate
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    if not chal_path.exists() or not csv_path.exists():
        print("[skip --relocate-probe] missing inputs"); return
    chal = json.loads(chal_path.read_text(encoding="utf-8"))
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rid = [r["task_id"] for r in _csv.DictReader(f)
               if r["category"] == "rearrangement"]
    print(f"\n=== ANALOGY_RELOCATE SOLVER  (rearrangement tasks: {len(rid)}) ===")
    print(f"{'task':10} {'LODO exact':>11} {'cell sim':>9}  frame (verified on full demo set)")
    solved = 0; n = 0; frames = _c.Counter()
    for tid in rid:
        if tid not in chal:
            continue
        n += 1
        demos = [(embed(p["input"], side), embed(p["output"], side)) for p in chal[tid]["train"]]
        sc = score_lodo(demos, rearrange_candidate, side)
        ex, sm = (sc if sc is not None else (0.0, 0.0))
        si = torch.stack([d[0] for d in demos]); so = torch.stack([d[1] for d in demos])
        _pred, meta = rearrange_candidate(si, so, demos[0][0], side, return_meta=True)
        fr = "-" if meta is None else ":".join(str(x) for x in meta)
        if meta is not None:
            frames[meta[0]] += 1
        if ex >= 1.0 - 1e-9:
            solved += 1
        print(f"{tid:10} {ex*100:>10.0f}% {sm*100:>8.0f}%  {fr}")
    print(f"\n  EXACT (all folds reconstruct): {solved}/{n}  ({solved/max(n,1)*100:.0f}%)  <- Lane A banked solves")
    print(f"  verified frames: {dict(frames)}")
    print("  (these are guaranteed-exact deterministic candidates for the verifier; the rest need")
    print("   richer frames (sort-by-attribute / snap-to-anchor) or the neural Lane B binding.)")


def run_relocate_test(side=30, challenges=None, solutions=None, categories_csv=None):
    """EXACT-ON-TEST: fit the recipe from ALL train demos (the real setting -- the verifier guarantees the
    recipe reconstructs every demo), apply it to the task's held-out TEST input, and compare to the
    solution. This is the 'exact in the test output' metric. It is the right one -- LODO undercounts
    because it hides a demo (a key/colour unique to that demo then cannot be learned)."""
    import csv as _csv
    import collections as _c
    from models.recursive_reasoning.object_rule_bank import rearrange_candidate
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    sol_path = Path(solutions) if solutions else root / "kaggle/combined/arc-agi_evaluation_solutions.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    if not (chal_path.exists() and sol_path.exists() and csv_path.exists()):
        print("[skip --relocate-test] missing inputs"); return
    from models.recursive_reasoning.object_rule_bank import rearrange_candidates
    chal = json.loads(chal_path.read_text(encoding="utf-8"))
    sol = json.loads(sol_path.read_text(encoding="utf-8"))
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rid = [r["task_id"] for r in _csv.DictReader(f)
               if r["category"] == "rearrangement"]
    print(f"\n=== ANALOGY_RELOCATE EXACT-ON-TEST  (rearrangement tasks: {len(rid)}) ===")
    print(f"{'task':10} {'@1':>4} {'@2':>4} {'cell sim':>9}  frame (attempt-1 / winning)")
    solved1 = solved2 = 0; n = 0; frames = _c.Counter()
    for tid in rid:
        if tid not in chal or tid not in sol:
            continue
        n += 1
        si = torch.stack([embed(p["input"], side) for p in chal[tid]["train"]])
        so = torch.stack([embed(p["output"], side) for p in chal[tid]["train"]])
        ok1_all = True; ok2_all = True; sims = []; meta1 = None; winmeta = "-"
        for ti, tin in enumerate(chal[tid]["test"]):
            cands = rearrange_candidates(si, so, embed(tin["input"], side), side, k=2)
            truth = embed(sol[tid][ti], side)
            if not cands:
                ok1_all = ok2_all = False; sims.append(0.0); continue
            if ti == 0:
                meta1 = cands[0][1]
            sims.append(max((p == truth).float().mean().item() for p, _ in cands))
            hit = [m for p, m in cands if bool(torch.equal(p, truth))]
            if not bool(torch.equal(cands[0][0], truth)):
                ok1_all = False
            if not hit:
                ok2_all = False
            elif winmeta == "-":
                winmeta = ":".join(str(x) for x in hit[0])
        if meta1 is not None:
            frames[meta1[0]] += 1
        if ok1_all:
            solved1 += 1
        if ok2_all:
            solved2 += 1
        fr1 = "-" if meta1 is None else ":".join(str(x) for x in meta1)
        print(f"{tid:10} {('Y' if ok1_all else '.'):>4} {('Y' if ok2_all else '.'):>4} {sum(sims)/max(len(sims),1)*100:>8.0f}%  {fr1} / {winmeta}")
    print(f"\n  EXACT @1 (first candidate): {solved1}/{n}  ({solved1/max(n,1)*100:.0f}%)")
    print(f"  EXACT @2 (ARC 2-attempt):   {solved2}/{n}  ({solved2/max(n,1)*100:.0f}%)")
    print(f"  attempt-1 frames: {dict(frames)}")


# ------------------------------------------------------ FIX 1: conditioned VALUE offline probe (learned lens)
# The decisive read the marginal-only runs (Codex V2 aux, Q-lane, C') could never produce: does a
# CONDITIONED value P(out | src, signature) with backoff beat the MARGINAL P(out | src) on the 52/75
# MULTI-TARGET tasks? Uses cell_conditioning_signature (Fix 2) as the context key. Pure VALUE isolation:
# oracle WHERE (we score only cells that truly changed), so WHERE error cannot contaminate the number.
VALUE_SIG_CONFIGS = {
    "marginal":     [],                 # P(out|src) -- the current transition_hint / V2 evidence
    "nbr4":         [1, 2, 3, 4],        # + sorted 4-neighbour colours
    "size_rank":    [5],                 # + enclosing-object size rank
    "holes":        [6],
    "shape_d4":     [7],
    "local":        [8, 9],
    "obj_color":    [10],               # + own/enclosing object colour (the background-fill key)
    "nbr4+objcol":  [1, 2, 3, 4, 10],
    "nbr4+rank":    [1, 2, 3, 4, 5],
    "nbr4+rank+objcol": [1, 2, 3, 4, 5, 10],
    # FIX H computed-WHERE keys (object_bank cols 11/12): TRUE flood-fill enclosure colour and
    # nearest-seed colour identity -- the two algorithmic keys the taxonomy plan says the enclosure
    # (25t) and adjacency (66t) families need. Gate: MULTI val_acc >= +8 over marginal -> wire the
    # model side; +3-8 -> offline candidates only; < +3 -> negative (joins the five converging kills).
    "encl_ff":      [11],
    "seed_col":     [12],
    "nbr4+encl":    [1, 2, 3, 4, 11],
    "nbr4+seed":    [1, 2, 3, 4, 12],
    "encl+seed":    [11, 12],
    "all":          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
}


def _value_task_is_multi(demos):
    """True if some CHANGED source colour maps to >1 output across the demos (needs conditioning)."""
    from collections import defaultdict
    s2d = defaultdict(set)
    for x, y in demos:
        ch = (x >= COLOR_OFFSET) & (y >= COLOR_OFFSET) & (x != y)
        for j in ch.nonzero(as_tuple=True)[0].tolist():
            s2d[int(x[j])].add(int(y[j]))
    return any(len(d) > 1 for d in s2d.values())


def run_value_binding_probe(side=30, challenges=None, categories_csv=None, min_bucket=1, max_tasks=0,
                            category="conditional_recolor"):
    """Fix-1 offline VALUE probe. For each task in `category` (default conditional_recolor;
    "all-colour" = conditional_recolor OR other -- the enclosure/adjacency families partly hide
    under 'other' in the atlas), LODO: learn conditioned + marginal value tables from support
    CHANGED cells, predict held-out CHANGED cells (oracle WHERE), measure value-accuracy and
    per-fold all-correct exactness, SPLIT by single- vs multi-target. The lift of a conditioned
    config over `marginal` on the MULTI subset is the go/no-go for wiring the head."""
    import csv as _csv
    import collections as _c
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    if not (chal_path.exists() and csv_path.exists()):
        print("[skip --value-binding-probe] missing inputs"); return
    chal = json.loads(chal_path.read_text())
    cats = {r["task_id"]: r["category"] for r in _csv.DictReader(open(csv_path, encoding="utf-8-sig"))}
    _accept = ({"conditional_recolor", "other"} if category == "all-colour" else {category})
    # 'other' is not a CSV label -- the atlas names only ~140 tasks; everything unlabeled IS 'other'
    # (same cats.get(tid, "other") convention as --compose-test). Iterate chal, not the CSV.
    cr = [t for t in chal if cats.get(t, "other") in _accept]

    # per config -> subset -> lists of (value_acc, oracle_exact, bucket_coverage) averaged per task
    agg = {cfg: {"single": [[], [], []], "multi": [[], [], []]} for cfg in VALUE_SIG_CONFIGS}
    n_single = n_multi = 0
    for ti, tid in enumerate(cr):
        demos = [(embed(p["input"], side), embed(p["output"], side)) for p in chal[tid]["train"]]
        M = len(demos)
        if M < 2:
            continue
        subset = "multi" if _value_task_is_multi(demos) else "single"
        if subset == "multi": n_multi += 1
        else: n_single += 1
        sig, _valid = cell_conditioning_signature(torch.stack([d[0] for d in demos]), side)   # [M,L,10]
        sig = sig.tolist()
        # per-config fold accumulators for THIS task
        task_acc = {cfg: [[], [], []] for cfg in VALUE_SIG_CONFIGS}
        for h in range(M):
            # build marginal + conditioned tables from SUPPORT changed cells (one pass, all configs)
            marg = _c.defaultdict(_c.Counter)                                     # src -> Counter(dst)
            cond = {cfg: _c.defaultdict(_c.Counter) for cfg in VALUE_SIG_CONFIGS}  # cfg -> (src,key) -> Counter
            for i in range(M):
                if i == h:
                    continue
                xi, yi = demos[i][0], demos[i][1]
                ch = (xi >= COLOR_OFFSET) & (yi >= COLOR_OFFSET) & (xi != yi)
                srow = sig[i]
                for j in ch.nonzero(as_tuple=True)[0].tolist():
                    src = int(xi[j]) - COLOR_OFFSET
                    dst = int(yi[j]) - COLOR_OFFSET
                    marg[src][dst] += 1
                    row = srow[j]
                    for cfg, cols in VALUE_SIG_CONFIGS.items():
                        key = tuple(row[c] for c in cols)
                        cond[cfg][(src, key)][dst] += 1
            # predict held-out CHANGED cells (oracle WHERE)
            xh, yh = demos[h][0], demos[h][1]
            chh = (xh >= COLOR_OFFSET) & (yh >= COLOR_OFFSET) & (xh != yh)
            idx = chh.nonzero(as_tuple=True)[0].tolist()
            srow_h = sig[h]
            for cfg, cols in VALUE_SIG_CONFIGS.items():
                correct = total = covered = 0
                for j in idx:
                    src = int(xh[j]) - COLOR_OFFSET
                    truth = int(yh[j]) - COLOR_OFFSET
                    key = tuple(srow_h[j][c] for c in cols)
                    bucket = cond[cfg].get((src, key))
                    if bucket is not None and sum(bucket.values()) >= min_bucket:
                        pred = max(bucket, key=bucket.get); covered += 1
                    elif marg.get(src):
                        pred = max(marg[src], key=marg[src].get)
                    else:
                        pred = src
                    total += 1
                    correct += int(pred == truth)
                vacc = correct / max(total, 1)
                task_acc[cfg][0].append(vacc)
                task_acc[cfg][1].append(1.0 if (total > 0 and correct == total) else 0.0)
                task_acc[cfg][2].append(covered / max(total, 1))
        for cfg in VALUE_SIG_CONFIGS:
            for k in range(3):
                agg[cfg][subset][k].append(sum(task_acc[cfg][k]) / max(len(task_acc[cfg][k]), 1))
        if max_tasks and (ti + 1) >= max_tasks:
            break

    def _m(v):
        return sum(v) / len(v) * 100 if v else 0.0
    print(f"\n=== CONDITIONED VALUE PROBE  (conditional_recolor: single={n_single} multi={n_multi}, "
          f"oracle-WHERE, LODO, min_bucket={min_bucket}) ===")
    print(f"{'config':12} | {'S val_acc':>9} {'M val_acc':>9} | {'S oExact':>8} {'M oExact':>8} | {'M cover':>7}")
    base_m = _m(agg['marginal']['multi'][0])
    for cfg in VALUE_SIG_CONFIGS:
        sv, mv = _m(agg[cfg]['single'][0]), _m(agg[cfg]['multi'][0])
        se, me = _m(agg[cfg]['single'][1]), _m(agg[cfg]['multi'][1])
        mc = _m(agg[cfg]['multi'][2])
        lift = f"  (+{mv - base_m:.1f} vs marginal)" if cfg != "marginal" else ""
        print(f"{cfg:12} | {sv:9.1f} {mv:9.1f} | {se:8.1f} {me:8.1f} | {mc:7.1f}{lift}")
    print(f"\nGO/NO-GO: a conditioned config must lift MULTI val_acc meaningfully over marginal "
          f"({base_m:.1f}) AND raise M oExact above the marginal floor. If none lifts, the SIGNATURE is")
    print(f"wrong (add kinematics / widen context), not 'TRM cannot' -- an informative negative.")


def run_analogy_value_probe(side=30, challenges=None, categories_csv=None, max_tasks=0):
    """Cross-demo analogy read: SOFT nearest-neighbour value transfer. For each held-out CHANGED cell
    (oracle WHERE), among support changed cells of the SAME source colour, weight each by how many
    signature columns MATCH the target cell, and vote its output. This is the soft/generalising version
    of the exact-bucket probe -- it resolves the caveat 'a learned head could soft-match beyond exact
    buckets'. If soft-NN >> exact-bucket (34), wiring a learned head is justified; if soft-NN ~= 34,
    the info is not in the features and no head helps."""
    import csv as _csv
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    if not (chal_path.exists() and csv_path.exists()):
        print("[skip --analogy-value-probe] missing inputs"); return
    chal = json.loads(chal_path.read_text())
    cats = {r["task_id"]: r["category"] for r in _csv.DictReader(open(csv_path, encoding="utf-8-sig"))}
    cr = [t for t, c in cats.items() if c == "conditional_recolor" and t in chal]
    sig_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                                # context columns (exclude src=0)

    import collections as _c
    acc = {"single": [[], []], "multi": [[], []]}                            # subset -> [val_acc, oExact]
    ns = nm = 0
    for ti, tid in enumerate(cr):
        demos = [(embed(p["input"], side), embed(p["output"], side)) for p in chal[tid]["train"]]
        M = len(demos)
        if M < 2:
            continue
        subset = "multi" if _value_task_is_multi(demos) else "single"
        if subset == "multi": nm += 1
        else: ns += 1
        sig_t, _valid = cell_conditioning_signature(torch.stack([d[0] for d in demos]), side)
        sig = sig_t.tolist()
        task_va, task_ex = [], []
        for h in range(M):
            # support changed cells: (src, [ctx cols], dst)
            sup = []
            marg = _c.defaultdict(_c.Counter)
            for i in range(M):
                if i == h:
                    continue
                xi, yi = demos[i][0], demos[i][1]
                ch = (xi >= COLOR_OFFSET) & (yi >= COLOR_OFFSET) & (xi != yi)
                srow = sig[i]
                for j in ch.nonzero(as_tuple=True)[0].tolist():
                    s = int(xi[j]) - COLOR_OFFSET
                    d = int(yi[j]) - COLOR_OFFSET
                    sup.append((s, [srow[j][c] for c in sig_cols], d))
                    marg[s][d] += 1
            xh, yh = demos[h][0], demos[h][1]
            chh = (xh >= COLOR_OFFSET) & (yh >= COLOR_OFFSET) & (xh != yh)
            srow_h = sig[h]
            correct = total = 0
            for j in chh.nonzero(as_tuple=True)[0].tolist():
                s = int(xh[j]) - COLOR_OFFSET
                truth = int(yh[j]) - COLOR_OFFSET
                tvec = [srow_h[j][c] for c in sig_cols]
                votes = _c.Counter()
                for (ss, svec, dd) in sup:
                    if ss != s:
                        continue
                    w = sum(1 for a, b in zip(tvec, svec) if a == b)          # matching columns (soft similarity)
                    if w > 0:
                        votes[dd] += w
                if votes:
                    pred = max(votes, key=votes.get)
                elif marg.get(s):
                    pred = max(marg[s], key=marg[s].get)
                else:
                    pred = s
                total += 1
                correct += int(pred == truth)
            task_va.append(correct / max(total, 1))
            task_ex.append(1.0 if (total > 0 and correct == total) else 0.0)
        acc[subset][0].append(sum(task_va) / max(len(task_va), 1))
        acc[subset][1].append(sum(task_ex) / max(len(task_ex), 1))
        if max_tasks and (ti + 1) >= max_tasks:
            break

    def _m(v):
        return sum(v) / len(v) * 100 if v else 0.0
    print(f"\n=== CROSS-DEMO ANALOGY (soft-NN) VALUE PROBE  (single={ns} multi={nm}, oracle-WHERE, LODO) ===")
    print(f"{'subset':8} {'val_acc':>9} {'oExact':>8}")
    for sub in ("single", "multi"):
        print(f"{sub:8} {_m(acc[sub][0]):9.1f} {_m(acc[sub][1]):8.1f}")
    print(f"\nCompare MULTI val_acc to: marginal 29.2, best exact-bucket 34.4. soft-NN >> 34 -> a learned")
    print(f"head can soft-match -> wiring justified; soft-NN ~= 34 -> info not in features, no head helps.")


# ==================================================== CORRESPONDENCE-TO-DEMO ceiling (open research)
# Every prior VALUE probe conditioned PER-CELL and read a GLOBAL table -> for multi-target recolor it
# capped at ~31-34 val_acc (marginal 29). The untested hypothesis: the disambiguator is not a per-cell
# feature but a PER-OBJECT correspondence to the demonstrations -- specifically a WITHIN-COLOUR ordinal
# ("colour a's LARGEST blob -> X, its 2nd -> Y"). That is a global-count relation no per-cell signature
# can express. This probe measures the ceiling of transferring an object's OUTPUT colour through such a
# correspondence key, LODO, before any C2 wiring.
CORR_KEY_CONFIGS = {
    "color":            ("colour",),               # per-object marginal == the failing baseline (~29)
    "shape":            ("_shape",),               # D4-canonical shape identity (translation+D4 inv.)
    "size_rank":        ("_grank",),               # GLOBAL size order (colour-agnostic)
    "color+szrank_ic":  ("colour", "_sr_ic"),      # <- key hypothesis: within-colour size order
    "color+rowrank_ic": ("colour", "_rr_ic"),      # within-colour top-edge order
    "color+colrank_ic": ("colour", "_cr_ic"),      # within-colour left-edge order
    "color+read_ic":    ("colour", "_read_ic"),    # within-colour reading order (row, then col)
    "color+shape":      ("colour", "_shape"),      # colour x shape
    "color+holes":      ("colour", "_holes"),      # colour x topology (hole count)
    "color+adj_col":    ("colour", "_adj_col"),    # RELATIONAL: colour of 4-adjacent neighbour object
    "color+cont_col":   ("colour", "_cont_col"),   # RELATIONAL: colour of enclosing (container) object
}


def _corr_objects(grid_flat, side, multi=False):
    """Mono-colour connected components with correspondence attributes attached. Cells map back to the
    side*side canvas via r*side+c (top-left aligned, same convention as _object_predicate_masks)."""
    import collections as _c
    col, _hw = _compact_colour(grid_flat, side)
    if col is None:
        return []
    bg = _background(col)
    objs = _objects(col, bg, multi=multi)
    if not objs:
        return objs
    sizes_desc = sorted({o["size"] for o in objs}, reverse=True)     # dense global rank, 0 = largest
    grank = {s: i for i, s in enumerate(sizes_desc)}
    by_col = _c.defaultdict(list)
    for o in objs:
        o["_flat"] = [r * side + c for (r, c) in o["cells"]]
        o["_shape"] = _d4_canon(o)
        o["_holes"] = _hole_count(o)
        o["_grank"] = grank[o["size"]]
        by_col[o["colour"]].append(o)
    for _c_key, lst in by_col.items():                              # within-colour dense ranks
        szr = {s: i for i, s in enumerate(sorted({o["size"] for o in lst}, reverse=True))}
        rowr = {v: i for i, v in enumerate(sorted({o["rmin"] for o in lst}))}
        colr = {v: i for i, v in enumerate(sorted({o["cmin"] for o in lst}))}
        for i, o in enumerate(sorted(lst, key=lambda z: (z["rmin"], z["cmin"]))):
            o["_read_ic"] = i
        for o in lst:
            o["_sr_ic"] = szr[o["size"]]
            o["_rr_ic"] = rowr[o["rmin"]]
            o["_cr_ic"] = colr[o["cmin"]]
    # relational attributes: adjacency (4-neighbour object colour) and containment (enclosing colour)
    cell2obj = {}
    for oi, o in enumerate(objs):
        for (r, c) in o["cells"]:
            cell2obj[(r, c)] = oi
    for oi, o in enumerate(objs):
        adj = _c.Counter()
        for (r, c) in o["cells"]:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = cell2obj.get((r + dr, c + dc))
                if nb is not None and nb != oi:
                    adj[objs[nb]["colour"]] += 1
        o["_adj_col"] = adj.most_common(1)[0][0] if adj else -1
        cont = None                                                # smallest strictly-enclosing object
        for p in objs:
            if p is o or p["size"] <= o["size"]:
                continue
            if (p["rmin"] <= o["rmin"] and p["rmax"] >= o["rmax"]
                    and p["cmin"] <= o["cmin"] and p["cmax"] >= o["cmax"]):
                if cont is None or p["size"] < cont["size"]:
                    cont = p
        o["_cont_col"] = cont["colour"] if cont is not None else -1
    return objs


def run_correspondence_probe(side=30, challenges=None, categories_csv=None, max_tasks=0):
    """OPEN RESEARCH ceiling (read-only): can multi-target conditional_recolor be recovered by mapping
    each test object to the demonstrations via a per-object correspondence key and transferring the
    corresponding OUTPUT colour? LODO, split single/multi. For each config report:
      val_acc : accuracy on oracle-WHERE changed cells (directly comparable to marginal 29 / soft-NN 34)
      oExact  : all changed cells correct in a fold (comparable to prior oracle-WHERE exact)
      gExact  : FULL-GRID exact match (honest solve metric; penalises over-painting unchanged cells)
    fg_cover reports the fraction of changed cells that lie inside a foreground object at all -- the hard
    ceiling of what object-correspondence can EVER touch (background-fill changed cells are uncoverable).
    GO if a within-colour-ordinal config lifts MULTI val_acc well past 34 AND raises gExact off the floor.
    """
    import csv as _csv
    import collections as _c
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    if not (chal_path.exists() and csv_path.exists()):
        print("[skip --correspondence-probe] missing inputs"); return
    chal = json.loads(chal_path.read_text())
    cats = {r["task_id"]: r["category"] for r in _csv.DictReader(open(csv_path, encoding="utf-8-sig"))}
    cr = [t for t, c in cats.items() if c == "conditional_recolor" and t in chal]

    agg = {cfg: {"single": [[], [], [], []], "multi": [[], [], [], []]} for cfg in CORR_KEY_CONFIGS}
    cover = {"single": [], "multi": []}
    ns = nm = 0
    for ti, tid in enumerate(cr):
        demos = [(embed(p["input"], side), embed(p["output"], side)) for p in chal[tid]["train"]]
        M = len(demos)
        if M < 2:
            continue
        subset = "multi" if _value_task_is_multi(demos) else "single"
        if subset == "multi": nm += 1
        else: ns += 1
        objs_per = [_corr_objects(d[0], side, multi=False) for d in demos]

        cov_task = []
        for (x, y), objs in zip(demos, objs_per):
            ch = (x >= COLOR_OFFSET) & (y >= COLOR_OFFSET) & (x != y)
            idx = set(ch.nonzero(as_tuple=True)[0].tolist())
            if not idx:
                continue
            fg = set().union(*[set(o["_flat"]) for o in objs]) if objs else set()
            cov_task.append(len(idx & fg) / len(idx))
        if cov_task:
            cover[subset].append(sum(cov_task) / len(cov_task))

        task_acc = {cfg: [[], [], [], []] for cfg in CORR_KEY_CONFIGS}
        for h in range(M):
            tables = {cfg: _c.defaultdict(_c.Counter) for cfg in CORR_KEY_CONFIGS}
            marg = _c.defaultdict(_c.Counter)                      # colour -> Counter(output token)
            for i in range(M):
                if i == h:
                    continue
                yi = demos[i][1]
                for o in objs_per[i]:
                    out_tok = _c.Counter(int(yi[f]) for f in o["_flat"]).most_common(1)[0][0]
                    marg[o["colour"]][out_tok] += 1
                    for cfg, attrs in CORR_KEY_CONFIGS.items():
                        tables[cfg][tuple(o[a] for a in attrs)][out_tok] += 1
            xh, yh = demos[h]
            objs_h = objs_per[h]
            fg_h = set().union(*[set(o["_flat"]) for o in objs_h]) if objs_h else set()
            chh = (xh >= COLOR_OFFSET) & (yh >= COLOR_OFFSET) & (xh != yh)
            changed = chh.nonzero(as_tuple=True)[0].tolist()
            changed_fg = [j for j in changed if j in fg_h]        # cells correspondence CAN address
            for cfg, attrs in CORR_KEY_CONFIGS.items():
                out_pred = xh.clone()
                for o in objs_h:
                    bucket = tables[cfg].get(tuple(o[a] for a in attrs))
                    if bucket:
                        tok = bucket.most_common(1)[0][0]
                    elif marg.get(o["colour"]):
                        tok = marg[o["colour"]].most_common(1)[0][0]
                    else:
                        tok = None
                    if tok is not None:
                        for f in o["_flat"]:
                            out_pred[f] = tok
                if changed:
                    correct = sum(int(out_pred[j] == yh[j]) for j in changed)
                    task_acc[cfg][0].append(correct / len(changed))
                    task_acc[cfg][1].append(1.0 if correct == len(changed) else 0.0)
                task_acc[cfg][2].append(1.0 if bool((out_pred == yh).all()) else 0.0)
                if changed_fg:                                    # foreground-restricted: is the KEY right?
                    fgc = sum(int(out_pred[j] == yh[j]) for j in changed_fg)
                    task_acc[cfg][3].append(fgc / len(changed_fg))
        for cfg in CORR_KEY_CONFIGS:
            for k in range(4):
                if task_acc[cfg][k]:
                    agg[cfg][subset][k].append(sum(task_acc[cfg][k]) / len(task_acc[cfg][k]))
        if max_tasks and (ti + 1) >= max_tasks:
            break

    def _m(v):
        return sum(v) / len(v) * 100 if v else 0.0
    print(f"\n=== CORRESPONDENCE-TO-DEMO CEILING  (conditional_recolor: single={ns} multi={nm}, "
          f"oracle-WHERE + full-grid, LODO) ===")
    print(f"fg_cover (changed cells inside a foreground object): single={_m(cover['single']):.1f}  "
          f"multi={_m(cover['multi']):.1f}   <- object-correspondence ceiling; rest is background-fill")
    print(f"fg_val = val_acc on changed cells INSIDE a foreground object (isolates: is the KEY right where")
    print(f"it applies?). M fg_val is the decisive column -- compare configs to color baseline on it.")
    print(f"{'config':18} | {'S fgV':>6} {'M fgV':>6} | {'M val':>6} {'M oEx':>6} {'M gEx':>6}")
    base_fg = _m(agg['color']['multi'][3])
    for cfg in CORR_KEY_CONFIGS:
        sfg, mfg = _m(agg[cfg]['single'][3]), _m(agg[cfg]['multi'][3])
        mv, mo, mg = _m(agg[cfg]['multi'][0]), _m(agg[cfg]['multi'][1]), _m(agg[cfg]['multi'][2])
        lift = f"  (fgV +{mfg - base_fg:.1f})" if cfg != "color" else "  (baseline)"
        print(f"{cfg:18} | {sfg:6.1f} {mfg:6.1f} | {mv:6.1f} {mo:6.1f} {mg:6.1f}{lift}")
    print(f"\nGO/NO-GO: baseline color(per-object) M fg_val={base_fg:.1f}. A within-colour-ordinal key")
    print(f"(szrank_ic/read_ic/holes) must lift M fg_val WELL above baseline AND raise M gExact off ~0 to")
    print(f"justify a C2 correspondence mechanism. If ordinal ties color on fg_val, the disambiguator is")
    print(f"relational (neighbour-of-X), not ordinal -- a sharper characterised bound than the per-cell one.")


def run_where_value_ceiling(side=30, challenges=None, solutions=None, categories_csv=None, max_tasks=0):
    """PHASE 0 (read-only): how many same-shape tasks are EXACTLY expressible as
    (object-WHERE ^ src-consensus-VALUE) -- and how many of those the FLOOR cannot already do.
    This is the ceiling of the whole where x value binding lane; measure before wiring the model.
      floor_ok      : per-src-colour modal map reconstructs every demo (the floor already solves it)
      needs_object  : an object-predicate partition verifies AND the floor does NOT  <- THE CEILING
      lodo_stable   : the fit-on-support predictor reconstructs held-out demos (transfers, not just fits)
    """
    import csv as _csv
    import collections as _c
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    if not chal_path.exists():
        print("[skip --where-value-ceiling] missing challenges"); return
    chal = json.loads(chal_path.read_text(encoding="utf-8"))
    cat = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                cat[row["task_id"]] = row["category"]

    L = side * side
    tally = _c.defaultdict(lambda: [0, 0, 0, 0])          # cat -> [n, needs_object, lodo_stable, floor_ok]
    winners = _c.Counter(); win_rows = []
    n_tasks = 0
    for tid, task in chal.items():
        if not same_shape(task):
            continue
        n_tasks += 1
        label = cat.get(tid, "other")
        demos = [(embed(p["input"], side), embed(p["output"], side)) for p in task["train"]]
        floor_ok = _fit_partition_table(demos, lambda x: torch.zeros(L, dtype=torch.bool), side) is not None
        obj_ok = False; win = None
        for multi in (False, True):
            for p in OBJECT_PREDICATES:
                def mask_of(x, _p=p, _m=multi):
                    mm = _object_predicate_masks(x, side, _m)
                    return None if mm is None else mm[_p]
                table = _fit_partition_table(demos, mask_of, side)
                if table is not None and _table_beats_floor(table):
                    obj_ok = True; win = f"{p}{'/multi' if multi else ''}"; break
            if obj_ok:
                break
        needs = obj_ok and not floor_ok
        lodo = score_lodo(demos, object_where_recolor_predict, side)
        lodo_stable = lodo is not None and lodo[0] >= 1.0 - 1e-9
        t = tally[label]
        t[0] += 1; t[1] += int(needs); t[2] += int(lodo_stable and not floor_ok); t[3] += int(floor_ok)
        if needs:
            winners[win] += 1
            win_rows.append((tid, label, win, "LODO" if (lodo_stable and not floor_ok) else "fit-only"))
        if max_tasks and n_tasks >= max_tasks:
            break

    print(f"\n=== WHERE x VALUE CEILING  (same-shape eval tasks: {n_tasks}) ===")
    print(f"{'category':22} {'n':>5} {'needs_object':>13} {'lodo_stable':>12} {'floor_ok':>9}")
    tot = [0, 0, 0, 0]
    for label in sorted(tally, key=lambda L2: -tally[L2][0]):
        n, need, lodo, fl = tally[label]
        for i, v in enumerate((n, need, lodo, fl)):
            tot[i] += v
        print(f"{label:22} {n:>5} {need:>13} {lodo:>12} {fl:>9}")
    print(f"{'TOTAL':22} {tot[0]:>5} {tot[1]:>13} {tot[2]:>12} {tot[3]:>9}")
    print(f"\nneeds_object = object split verifies AND floor fails  <- THE CEILING for a new candidate")
    print(f"winning predicate on needs_object tasks: {dict(winners)}")
    for tid, label, win, kind in win_rows:
        print(f"  {tid:10} {label:22} {win:18} {kind}")
    cr = tally.get("conditional_recolor", [0, 0, 0, 0])
    print(f"\nGATE: conditional_recolor needs_object = {cr[1]}/{cr[0]}  "
          f"(>= ~8% -> build Phase 1; < ~3% -> KILL the lane, see PLAN section 3)")


def run_region_value_ceiling(side=30, challenges=None, solutions=None, categories_csv=None, max_tasks=0):
    """PHASE 0B (read-only): how many same-shape tasks are EXACTLY expressible as
    (region/negative-space-WHERE ^ src-consensus-VALUE) -- and how many of those the FLOOR cannot do.

    This is the proper follow-up to the object-WHERE kill: changed background cells are not objects,
    so the ceiling must be measured on background/region masks before any TRM wiring.
    """
    import csv as _csv
    import collections as _c
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    if not chal_path.exists():
        print("[skip --region-value-ceiling] missing challenges")
        return
    chal = json.loads(chal_path.read_text(encoding="utf-8"))
    cat = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                cat[row["task_id"]] = row["category"]

    L = side * side
    tally = _c.defaultdict(lambda: [0, 0, 0, 0])          # cat -> [n, needs_region, lodo_stable, floor_ok]
    winners = _c.Counter()
    win_rows = []
    n_tasks = 0
    for tid, task in chal.items():
        if not same_shape(task):
            continue
        n_tasks += 1
        label = cat.get(tid, "other")
        demos = [(embed(p["input"], side), embed(p["output"], side)) for p in task["train"]]
        floor_ok = _fit_partition_table(demos, lambda x: torch.zeros(L, dtype=torch.bool), side) is not None
        region_ok = False
        win = None
        for p in REGION_PREDICATES:
            def mask_of(x, _p=p):
                return _region_predicate_masks(x, side)[_p]
            table = _fit_partition_table(demos, mask_of, side)
            if table is not None and _table_beats_floor(table):
                region_ok = True
                win = p
                break
        needs = region_ok and not floor_ok
        lodo = score_lodo(demos, region_where_fill_predict, side)
        lodo_stable = lodo is not None and lodo[0] >= 1.0 - 1e-9
        t = tally[label]
        t[0] += 1
        t[1] += int(needs)
        t[2] += int(lodo_stable and not floor_ok)
        t[3] += int(floor_ok)
        if needs:
            winners[win] += 1
            win_rows.append((tid, label, win, "LODO" if (lodo_stable and not floor_ok) else "fit-only"))
        if max_tasks and n_tasks >= max_tasks:
            break

    print(f"\n=== REGION x VALUE CEILING  (same-shape eval tasks: {n_tasks}) ===")
    print(f"{'category':22} {'n':>5} {'needs_region':>13} {'lodo_stable':>12} {'floor_ok':>9}")
    tot = [0, 0, 0, 0]
    for label in sorted(tally, key=lambda L2: -tally[L2][0]):
        n, need, lodo, fl = tally[label]
        for i, v in enumerate((n, need, lodo, fl)):
            tot[i] += v
        print(f"{label:22} {n:>5} {need:>13} {lodo:>12} {fl:>9}")
    print(f"{'TOTAL':22} {tot[0]:>5} {tot[1]:>13} {tot[2]:>12} {tot[3]:>9}")
    print("\nneeds_region = region split verifies AND floor fails  <- THE CEILING for a region-fill candidate")
    print(f"winning predicate on needs_region tasks: {dict(winners)}")
    for tid, label, win, kind in win_rows:
        print(f"  {tid:10} {label:22} {win:28} {kind}")
    cr = tally.get("conditional_recolor", [0, 0, 0, 0])
    print(f"\nGATE: conditional_recolor needs_region = {cr[1]}/{cr[0]}  "
          f"(>= 6 tasks -> build offline candidate; 1..5 -> keep offline-only; 0 -> park region-fill)")


def run_constant_fill_ceiling(side=30, challenges=None, categories_csv=None, max_tasks=0):
    """Read-only ceiling: copy outside an input-defined region, fill that region with one constant.

    Unlike --where-value-ceiling and --region-value-ceiling, this intentionally scans ALL eval tasks.
    Shape-changing tasks are allowed into the denominator; the strict fit will reject them unless a
    top-left flat copy/fill can reconstruct PAD/EOS too, which should normally be impossible.
    """
    import csv as _csv
    import collections as _c
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    if not chal_path.exists():
        print("[skip --constant-fill-ceiling] missing challenges")
        return
    chal = json.loads(chal_path.read_text(encoding="utf-8"))
    cat = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                cat[row["task_id"]] = row["category"]

    tally = _c.defaultdict(lambda: [0, 0, 0, 0, 0])  # cat -> [n, fit, lodo, floor_lodo, same_shape]
    winners = _c.Counter()
    win_rows = []
    n_tasks = 0
    for tid, task in chal.items():
        n_tasks += 1
        label = cat.get(tid, "other")
        demos = [(embed(p["input"], side), embed(p["output"], side)) for p in task["train"]]
        is_same = same_shape(task)

        fit_ok = False
        win = None
        for p in REGION_PREDICATES:
            def mask_of(x, _p=p):
                return _region_predicate_masks(x, side)[_p]
            fill = _fit_constant_fill(demos, mask_of, side)
            if fill is not None:
                fit_ok = True
                win = p
                break

        lodo = score_lodo(demos, constant_fill_predict, side)
        lodo_stable = lodo is not None and lodo[0] >= 1.0 - 1e-9
        floor = score_lodo(demos, floor_predict, side)
        floor_ok = floor is not None and floor[0] >= 1.0 - 1e-9

        t = tally[label]
        t[0] += 1
        t[1] += int(fit_ok and not floor_ok)
        t[2] += int(lodo_stable and not floor_ok)
        t[3] += int(floor_ok)
        t[4] += int(is_same)
        if fit_ok and not floor_ok:
            winners[win] += 1
            win_rows.append((tid, label, win, "LODO" if (lodo_stable and not floor_ok) else "fit-only"))

        if max_tasks and n_tasks >= max_tasks:
            break

    print(f"\n=== CONSTANT-FILL CEILING  (all eval tasks scanned: {n_tasks}) ===")
    print(f"{'category':22} {'n':>5} {'same_shape':>11} {'needs_const':>12} {'lodo_stable':>12} {'floor_lodo':>11}")
    tot = [0, 0, 0, 0, 0]
    for label in sorted(tally, key=lambda L2: -tally[L2][0]):
        n, need, lodo, fl, ss = tally[label]
        for i, v in enumerate((n, need, lodo, fl, ss)):
            tot[i] += v
        print(f"{label:22} {n:>5} {ss:>11} {need:>12} {lodo:>12} {fl:>11}")
    print(f"{'TOTAL':22} {tot[0]:>5} {tot[4]:>11} {tot[1]:>12} {tot[2]:>12} {tot[3]:>11}")
    print("\nneeds_const = constant region fill verifies on all support demos AND floor LODO fails")
    print(f"winning predicate on needs_const tasks: {dict(winners)}")
    for tid, label, win, kind in win_rows:
        print(f"  {tid:10} {label:22} {win:28} {kind}")


def run_projection_ceiling(side=30, challenges=None, categories_csv=None, max_tasks=0):
    """FIX 3 Phase A (read-only): size the projection/bbox/periodic sub-buckets (3a/3b/3c) plus the
    FIX 2 foreground nearest-seed recolor, BEFORE wiring anything. For each eval task and each family
    predictor: LODO-stable (exact on every held-out demo) AND floor-LODO fails = net-new headroom."""
    import csv as _csv
    import collections as _c
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    if not chal_path.exists():
        print("[skip --projection-ceiling] missing challenges")
        return
    chal = json.loads(chal_path.read_text(encoding="utf-8"))
    cat = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                cat[row["task_id"]] = row["category"]

    fams = list(PROJECTION_CANDIDATES)
    per_fam = {f: [] for f in fams}                       # f -> [(tid, label)]
    tally = _c.defaultdict(lambda: [0, 0])                # cat -> [n, any_fam_lodo]
    n_tasks = 0
    for tid, task in chal.items():
        n_tasks += 1
        label = cat.get(tid, "other")
        demos = [(embed(p["input"], side), embed(p["output"], side)) for p in task["train"]]
        floor = score_lodo(demos, floor_predict, side)
        floor_ok = floor is not None and floor[0] >= 1.0 - 1e-9
        hit_any = False
        for f in fams:
            if floor_ok:
                continue
            lodo = score_lodo(demos, PROJECTION_CANDIDATES[f], side)
            if lodo is not None and lodo[0] >= 1.0 - 1e-9:
                per_fam[f].append((tid, label))
                hit_any = True
        t = tally[label]
        t[0] += 1
        t[1] += int(hit_any)
        if max_tasks and n_tasks >= max_tasks:
            break

    print(f"\n=== PROJECTION/BBOX CEILING  (FIX 3 Phase A + FIX 2 fg, eval tasks scanned: {n_tasks}) ===")
    print("family counts = LODO-stable AND floor-LODO fails (net-new headroom):")
    for f in fams:
        print(f"  {f:18} {len(per_fam[f]):>4}")
        for tid, label in per_fam[f]:
            print(f"      {tid:10} {label}")
    print(f"\n{'category':22} {'n':>5} {'any_family':>11}")
    tot = [0, 0]
    for label in sorted(tally, key=lambda L2: -tally[L2][0]):
        n, hit = tally[label]
        tot[0] += n
        tot[1] += hit
        print(f"{label:22} {n:>5} {hit:>11}")
    print(f"{'TOTAL':22} {tot[0]:>5} {tot[1]:>11}")
    print("\nGATES (plan): 3a ray >= ~10, 3b bbox >= ~10 -> wire that family into CANDIDATES; "
          "3c periodic expect park; fg_seed_recolor any net-new -> wire (floor-safe).")


def _flat_role_masks(grid_flat, side):
    """Per-cell role masks for microscope reporting only."""
    L = side * side
    out = {
        "src_background": torch.zeros(L, dtype=torch.bool),
        "src_foreground": torch.zeros(L, dtype=torch.bool),
    }
    col, hw = _compact_colour(grid_flat, side)
    if col is None:
        return out
    bg = _background(col)
    valid = col >= 0
    masks2 = {
        "src_background": (col == bg) & valid,
        "src_foreground": (col >= 0) & (col != bg),
    }
    for name, mask2d in masks2.items():
        rr, cc = mask2d.nonzero(as_tuple=True)
        if rr.numel():
            out[name][rr * side + cc] = True
    return out


def _frac(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _conditional_recolor_microscope_row(task_id: str, label: str, demos, side: int,
                                        shape_same: bool = True) -> dict:
    """Summarize what changed cells look like in one task's support pairs.

    This does not fit or execute a candidate. It only asks which existing mask families cover the
    observed changed cells, so the next ceiling probe is chosen from evidence instead of intuition.
    """
    total_changed = 0
    total_added = 0
    total_removed = 0
    hit = Counter()
    src_colours = Counter()
    dst_colours = Counter()
    src_to_dst = {}
    for x, y in demos:
        xi, yi = x.long(), y.long()
        changed = (xi >= COLOR_OFFSET) & (yi >= COLOR_OFFSET) & (xi != yi)
        added = (xi < COLOR_OFFSET) & (yi >= COLOR_OFFSET)
        removed = (xi >= COLOR_OFFSET) & (yi < COLOR_OFFSET)
        total_added += int(added.sum().item())
        total_removed += int(removed.sum().item())
        n = int(changed.sum().item())
        if n == 0:
            continue
        total_changed += n
        roles = _flat_role_masks(xi, side)
        regions = _region_predicate_masks(xi, side)
        for name, mask in roles.items():
            hit[name] += int((changed & mask).sum().item())
        for name, mask in regions.items():
            hit[name] += int((changed & mask).sum().item())
        for src in sorted(int(t) for t in torch.unique(xi[changed]) if int(t) >= COLOR_OFFSET):
            raw_src = src - COLOR_OFFSET
            m = changed & (xi == src)
            src_colours[raw_src] += int(m.sum().item())
            dsts = sorted(int(t) - COLOR_OFFSET for t in torch.unique(yi[m]) if int(t) >= COLOR_OFFSET)
            src_to_dst.setdefault(raw_src, set()).update(dsts)
        for dst in sorted(int(t) for t in torch.unique(yi[changed]) if int(t) >= COLOR_OFFSET):
            dst_colours[dst - COLOR_OFFSET] += int((changed & (yi == dst)).sum().item())

    priority = [
        "enclosed_background",
        "foreground_bbox_background",
        "row_gap_background",
        "col_gap_background",
        "background",
        "foreground",
    ]
    best = max(priority, key=lambda name: (hit[name], -priority.index(name))) if total_changed else "none"
    if total_changed == 0 or hit[best] == 0:
        best = "none"
    multi_dst = {k: sorted(v) for k, v in sorted(src_to_dst.items()) if len(v) >= 2}
    dst_set = sorted(dst_colours)
    src_set = sorted(src_colours)
    if total_changed == 0:
        value_source = "none"
    elif len(dst_set) == 1 and hit["background"] == total_changed:
        value_source = "background_to_constant"
    elif len(dst_set) == 1:
        value_source = "constant"
    else:
        value_source = "mixed"
    return {
        "task_id": task_id,
        "category": label,
        "shape_same": int(bool(shape_same)),
        "n_demos": len(demos),
        "n_changed": total_changed,
        "n_added_colour": total_added,
        "n_removed_colour": total_removed,
        "changed_src_colours": " ".join(str(c) for c in src_set),
        "changed_dst_colours": " ".join(str(c) for c in dst_set),
        "src_multi_dst_colours": " ".join(f"{k}:{','.join(str(x) for x in v)}" for k, v in multi_dst.items()),
        "changed_background_frac": _frac(hit["src_background"], total_changed),
        "changed_foreground_frac": _frac(hit["src_foreground"], total_changed),
        "changed_enclosed_frac": _frac(hit["enclosed_background"], total_changed),
        "changed_inside_bbox_frac": _frac(hit["foreground_bbox_background"], total_changed),
        "changed_row_gap_frac": _frac(hit["row_gap_background"], total_changed),
        "changed_col_gap_frac": _frac(hit["col_gap_background"], total_changed),
        "dominant_output_value_source": value_source,
        "best_explaining_family": best,
        "best_family_frac": _frac(hit[best], total_changed) if best != "none" else 0.0,
    }


def run_conditional_recolor_microscope(side=30, challenges=None, categories_csv=None,
                                       out_dir=None, max_tasks=0, include_all: bool = False):
    """Read-only microscope for atlas-labelled tasks.

    By default this preserves the original conditional_recolor-only report. include_all=True emits
    every eval task and adds shape/add/remove counters so non-same-shape tasks are not misread as
    ordinary in-place recolour examples.
    """
    import csv as _csv
    import collections as _c
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    report_dir = Path(out_dir) if out_dir else root / "reports/conditional_recolor_microscope"
    if not chal_path.exists():
        print("[skip --conditional-recolor-microscope] missing challenges")
        return
    cat = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                cat[row["task_id"]] = row["category"]
    chal = json.loads(chal_path.read_text(encoding="utf-8"))
    rows = []
    for tid, task in chal.items():
        label = cat.get(tid, "other")
        ss = same_shape(task)
        if (not include_all) and (label != "conditional_recolor" or not ss):
            continue
        demos = [(embed(p["input"], side), embed(p["output"], side)) for p in task["train"]]
        rows.append(_conditional_recolor_microscope_row(tid, label, demos, side, shape_same=ss))
        if max_tasks and len(rows) >= max_tasks:
            break

    report_dir.mkdir(parents=True, exist_ok=True)
    out_csv = report_dir / "per_task.csv"
    fieldnames = [
        "task_id", "category", "shape_same", "n_demos", "n_changed",
        "n_added_colour", "n_removed_colour",
        "changed_src_colours", "changed_dst_colours", "src_multi_dst_colours",
        "changed_background_frac", "changed_foreground_frac", "changed_enclosed_frac",
        "changed_inside_bbox_frac", "changed_row_gap_frac", "changed_col_gap_frac",
        "dominant_output_value_source", "best_explaining_family", "best_family_frac",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    total_changed = sum(int(r["n_changed"]) for r in rows)
    total_added = sum(int(r["n_added_colour"]) for r in rows)
    total_removed = sum(int(r["n_removed_colour"]) for r in rows)
    best_counts = _c.Counter(r["best_explaining_family"] for r in rows)
    value_counts = _c.Counter(r["dominant_output_value_source"] for r in rows)
    cat_counts = _c.Counter(r["category"] for r in rows)
    cat_changed = _c.defaultdict(int)
    cat_added = _c.defaultdict(int)
    cat_removed = _c.defaultdict(int)
    for r in rows:
        cat_changed[r["category"]] += int(r["n_changed"])
        cat_added[r["category"]] += int(r["n_added_colour"])
        cat_removed[r["category"]] += int(r["n_removed_colour"])
    weighted = {}
    for key in (
        "changed_background_frac", "changed_foreground_frac", "changed_enclosed_frac",
        "changed_inside_bbox_frac", "changed_row_gap_frac", "changed_col_gap_frac",
    ):
        weighted[key] = (
            sum(float(r[key]) * int(r["n_changed"]) for r in rows) / max(total_changed, 1)
        )
    title = "ALL-TASK MICROSCOPE" if include_all else "CONDITIONAL_RECOLOR MICROSCOPE"
    print(f"\n=== {title} ({len(rows)} tasks) ===")
    print(f"report: {out_csv}")
    print(f"total_changed_cells: {total_changed}")
    print(f"total_added_colour_cells: {total_added}")
    print(f"total_removed_colour_cells: {total_removed}")
    for k, v in weighted.items():
        print(f"{k}: {v:.3f}")
    print(f"best_explaining_family counts: {dict(best_counts)}")
    print(f"dominant_output_value_source counts: {dict(value_counts)}")
    print("per_category token-change counts:")
    for label in sorted(cat_counts, key=lambda k: (-cat_counts[k], k)):
        print(f"  {label}: tasks={cat_counts[label]} changed={cat_changed[label]} "
              f"added={cat_added[label]} removed={cat_removed[label]}")


def run_compose_test(side=30, challenges=None, solutions=None, categories_csv=None,
                     include_committed=True, max_tasks=0, model_dump_path=None):
    """THE COMPOSED NUMBER. Every verified candidate family -- recolor predictors (floor/D1,
    neighbour signatures, cond_split predicates, analogy, set_cover, relations, committed DSL)
    PLUS Lane A's demo-verified rearrange candidates -- ranked per task by LODO on the train
    demos (floor always in the set, exact primary, floor wins ties), the top-2 DISTINCT
    predictions applied to the REAL test input, ARC 2-attempt scoring against the solutions.
    This is what the banked component wins look like as ONE honest held-out metric."""
    import csv as _csv
    import collections as _c
    root = Path(__file__).resolve().parents[1]
    chal_path = Path(challenges) if challenges else root / "kaggle/combined/arc-agi_evaluation_challenges.json"
    sol_path = Path(solutions) if solutions else root / "kaggle/combined/arc-agi_evaluation_solutions.json"
    csv_path = Path(categories_csv) if categories_csv else root / "reports/arc_task_atlas/selected_task_categories.csv"
    if not (chal_path.exists() and sol_path.exists()):
        print("[skip --compose-test] missing challenges/solutions"); return
    chal = json.loads(chal_path.read_text(encoding="utf-8"))
    sol = json.loads(sol_path.read_text(encoding="utf-8"))
    model_fold_dump = None
    model_test_dump = None
    if model_dump_path:
        model_fold_dump = load_model_dump(model_dump_path, side=side, kind="fold", require_kind=True)
        model_test_dump = load_model_dump(model_dump_path, side=side, kind="test", require_kind=True)
    cat = {}
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in _csv.DictReader(f):
                cat[row["task_id"]] = row["category"]

    names = [n for n in CANDIDATES if include_committed or n != "committed"]
    per_cat = _c.defaultdict(lambda: [0, 0, 0])                    # category -> [n, solved@1, solved@2]
    win_family = _c.Counter()
    solved_rows = []
    n_tasks = 0
    for tid, task in chal.items():
        if tid not in sol:
            continue
        n_tasks += 1
        label = cat.get(tid, "other")
        demos = [(embed(p["input"], side), embed(p["output"], side)) for p in task["train"]]
        si = torch.stack([d[0] for d in demos]); so = torch.stack([d[1] for d in demos])
        shape_ok = same_shape(task)
        # rank ONCE per task on the train demos (fit each fold inside score_lodo -- no test peeking)
        ranked = []                                                # (key, name, predict_fn)
        for name in names:
            if not shape_ok and name not in NONSHAPE_OK:
                continue                                           # recolor predictors assume same shape
            s = score_lodo(demos, CANDIDATES[name], side)
            if s is None:
                continue
            ranked.append(((s[0], s[1], 1.0 if name == "floor" else 0.0), name, CANDIDATES[name]))
        md = score_model_dump_lodo(demos, side, task_id=tid, model_dump=model_fold_dump)
        if md is not None:
            ranked.append(((md[0], md[1], 0.0), "model_dump", None))
        rl1 = score_relocate_k_lodo(demos, side, k_index=0)
        rl2 = score_relocate_k_lodo(demos, side, k_index=1)
        if rl1 is not None:
            ranked.append(((rl1[0], rl1[1], 0.0), "relocate@1",
                           relocate_k_predictor(0)))
        if rl2 is not None:
            ranked.append(((rl2[0], rl2[1], 0.0), "relocate@2",
                           relocate_k_predictor(1)))
        ranked.sort(key=lambda r: r[0], reverse=True)
        # top-2 DISTINCT predictions on the real test inputs, fit on ALL demos
        ok1_all = True; ok2_all = True; win = None
        for ti_idx, tin in enumerate(task["test"]):
            tgt_in = embed(tin["input"], side)
            truth = embed(sol[tid][ti_idx], side)
            deterministic_floor = floor_predict(si, so, tgt_in, side)
            attempts = []; attempt_names = []
            for _key, name, fn in ranked:
                if name == "floor":
                    continue
                preds = (
                    model_dump_predictions_for_test(model_test_dump, tid, ti_idx, side, include_floor=False)
                    if name == "model_dump"
                    else [fn(si, so, tgt_in, side)]
                )
                for p in preds:
                    if p is None:
                        continue
                    if deterministic_floor is not None and bool(torch.equal(p, deterministic_floor)):
                        continue
                    if not any(bool(torch.equal(p, q)) for q in attempts):
                        attempts.append(p); attempt_names.append(name)
                    if len(attempts) == 1:
                        break
                if len(attempts) == 1:
                    break
            if deterministic_floor is not None and not any(bool(torch.equal(deterministic_floor, q)) for q in attempts):
                attempts.append(deterministic_floor)
                attempt_names.append("floor")
            if not attempts and deterministic_floor is not None:
                attempts.append(deterministic_floor)
                attempt_names.append("floor")
            hit = [attempt_names[j] for j, p in enumerate(attempts) if bool(torch.equal(p, truth))]
            if not (attempts and bool(torch.equal(attempts[0], truth))):
                ok1_all = False
            if not hit:
                ok2_all = False
            elif win is None:
                win = hit[0]
        per_cat[label][0] += 1
        per_cat[label][1] += int(ok1_all)
        per_cat[label][2] += int(ok2_all)
        if ok2_all:
            win_family[win] += 1
            solved_rows.append((tid, label, "@1" if ok1_all else "@2", win))
        if max_tasks and n_tasks >= max_tasks:
            break

    print(f"\n=== COMPOSED EXACT-ON-TEST  (ARC 2-attempt, held-out eval: {n_tasks} tasks) ===")
    print(f"{'category':22} {'n':>5} {'@1':>6} {'@2':>6}")
    t_n = t_1 = t_2 = 0
    for label in sorted(per_cat, key=lambda L: -per_cat[L][0]):
        n, s1, s2 = per_cat[label]
        t_n += n; t_1 += s1; t_2 += s2
        print(f"{label:22} {n:>5} {s1:>6} {s2:>6}")
    print(f"{'TOTAL':22} {t_n:>5} {t_1:>6} {t_2:>6}   "
          f"(@1 {t_1 / max(t_n, 1) * 100:.1f}%  @2 {t_2 / max(t_n, 1) * 100:.1f}%)")
    print(f"\nwinning family on solved tasks: {dict(win_family)}")
    for tid, label, at, win in solved_rows:
        print(f"  {tid:10} {label:22} {at:3} {win}")
    return {
        "tasks": t_n,
        "total_solved_at1": t_1,
        "total_solved_at2": t_2,
        "per_category": dict(per_cat),
        "winning_family": dict(win_family),
    }


# ----------------------------------------------------------------------------------------- self-test
def _self_test():
    side = 10

    def grid(block_col, single_col, br, bc, sr, sc):
        g = torch.zeros(side, side, dtype=torch.long)            # token 0 = PAD (outside object)
        g[:] = 0 + COLOR_OFFSET                                  # colour-0 background
        g[br:br + 3, bc:bc + 3] = block_col + COLOR_OFFSET
        g[sr, sc] = single_col + COLOR_OFFSET
        return g.reshape(-1)

    # CONDITIONAL: big blue(1) -> red(2); small blue(1) -> green(3). Same input colour, two outputs.
    cond_demos = []
    for (br, bc, sr, sc) in [(0, 0, 7, 7), (1, 5, 8, 1), (5, 5, 0, 8)]:
        i = grid(1, 1, br, bc, sr, sc)
        o = grid(2, 3, br, bc, sr, sc)
        cond_demos.append((i, o))
    rc = evaluate_task(cond_demos, side)
    fa = rc["scores"]["floor"][0]; an = rc["scores"]["analogy"][0]; nb = rc["scores"]["neighbour"][0]
    assert max(an, nb) > fa, f"a relational candidate must beat floor (analogy {an}, neighbour {nb} vs floor {fa})"
    cs = rc["scores"]["cond_split"][0]
    assert cs >= 1.0 - 1e-9, (
        f"cond_split must solve the conditional task EXACTLY via a verified same4>=1 split, got {cs}")
    # and the verified split must NOT fire on unverifiable colours: on the clean task below it
    # must degrade to the floor, never below it (checked via the selector guarantee assert).
    assert rc["winner"] in ("analogy", "neighbour", "cond_split", "committed"), \
        f"selector must pick a relational candidate, got {rc['winner']}"
    assert "winner_detail" in rc and rc["winner_detail"] != "committed", \
        f"committed winner must expose its internal source, got {rc.get('winner_detail')}"
    if rc["winner"] == "committed":
        assert rc["winner_detail"].startswith("committed:object_recolor:"), \
            f"conditional committed route should expose object-recolor source, got {rc['winner_detail']}"
    assert rc["selector_exact"] >= rc["floor_exact"], "selector must be >= floor"

    # CLEAN recolour: all blue(1) -> red(2). The floor is exact; selector ties to floor.
    clean_demos = []
    for (br, bc, sr, sc) in [(0, 0, 7, 7), (1, 5, 8, 1), (5, 5, 0, 8)]:
        i = grid(1, 1, br, bc, sr, sc)
        o = grid(2, 2, br, bc, sr, sc)                          # singleton also -> red (no condition)
        clean_demos.append((i, o))
    rcl = evaluate_task(clean_demos, side)
    assert rcl["scores"]["floor"][0] == 1.0, f"floor must solve clean recolour ({rcl['scores']['floor']})"
    assert rcl["selector_exact"] >= rcl["floor_exact"], "selector >= floor on clean too"

    # ADVERSARIAL guarantee guard: a rival with HIGHER sim but LOWER exact than floor must NOT win.
    # This is what actually protects `selector_exact >= floor_exact` against a key reorder (e.g. sim-first);
    # the two happy-path tasks above would pass even if sim were primary, so they do not guard it.
    adv = {"floor": (0.5, 0.10), "rival_high_sim": (0.0, 0.99)}
    assert select_winner(adv) == "floor", "EXACT must dominate SIM: high-sim/low-exact rival must not win"
    tie = {"floor": (0.5, 0.5), "rival_tie": (0.5, 0.5)}
    assert select_winner(tie) == "floor", "on a full (exact,sim) tie the floor must win (conservative)"
    beat = {"floor": (0.3, 0.9), "rival_beats": (0.6, 0.1)}
    assert select_winner(beat) == "rival_beats", "a strictly higher-EXACT rival must win over floor"

    # SIZE-CHANGE family: propose->verify->apply must solve a 3x upscale and a colour-frame crop.
    up_demos = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        gi = torch.randint(0, 4, (2, 2)) + COLOR_OFFSET
        go = gi.repeat_interleave(3, 0).repeat_interleave(3, 1)
        up_demos.append((_compact_to_flat(gi, side), _compact_to_flat(go, side)))
    up = score_lodo(up_demos, upscale_predict, side)
    assert up is not None and up[0] >= 1.0 - 1e-9, f"upscale must verify+solve the 3x task, got {up}"
    cr_demos = []
    for seed in (3, 4, 5):
        torch.manual_seed(seed)
        g = torch.randint(0, 3, (8, 8)) + COLOR_OFFSET               # colours 0..2 body
        g[2:7, 2:7] = 5 + COLOR_OFFSET                               # colour-5 frame
        inner = torch.randint(0, 3, (3, 3)) + COLOR_OFFSET
        g[3:6, 3:6] = inner
        cr_demos.append((_compact_to_flat(g, side), _compact_to_flat(inner, side)))
    cr = score_lodo(cr_demos, crop_colour_predict, side)
    assert cr is not None and cr[0] >= 1.0 - 1e-9, f"crop_colour(inner) must verify+solve, got {cr}"
    # and on the SAME-shape clean task the size family must stay SILENT (verify fails -> None),
    # so it can never displace the floor there.
    assert upscale_predict(torch.stack([d[0] for d in clean_demos[:2]]),
                           torch.stack([d[1] for d in clean_demos[:2]]),
                           clean_demos[2][0], side) is None, "upscale must not fire on same-shape"

    # OBJECT-WHERE x VALUE: "recolor the LARGEST blue(1) object to red(2), copy the small blue(1)".
    # Same input colour -> two outputs by OBJECT SIZE: the floor is stuck (blue has 2 dsts), the object
    # split must solve it and it must transfer LODO. This is the exact case the lane targets.
    ow_demos = []
    for (br, bc, sr, sc) in [(0, 0, 7, 7), (1, 5, 8, 1), (5, 5, 0, 8), (2, 2, 9, 9)]:
        i = grid(1, 1, br, bc, sr, sc)                      # big blue block + small blue singleton
        o = grid(2, 1, br, bc, sr, sc)                      # big -> red(2); singleton stays blue(1)
        ow_demos.append((i, o))
    fl = score_lodo(ow_demos, floor_predict, side)
    ow = score_lodo(ow_demos, object_where_recolor_predict, side)
    assert fl is not None and fl[0] < 1.0, f"floor must FAIL the object-split task (got {fl})"
    assert ow is not None and ow[0] >= 1.0 - 1e-9, \
        f"object_where_recolor must verify+transfer the largest-object recolour (got {ow})"
    # and it must be SILENT on the clean recolour (no colour is split by object) -> floor-safe.
    assert object_where_recolor_predict(torch.stack([d[0] for d in clean_demos[:2]]),
                                        torch.stack([d[1] for d in clean_demos[:2]]),
                                        clean_demos[2][0], side) is None, \
        "object_where_recolor must stay silent when no object split beats the floor"

    # GUARANTEE: selector_exact >= floor_exact on both tasks (the net-negative-proof property)
    print("verify_and_select_candidates self-test PASS  "
          f"(conditional: floor={fa:.2f} analogy={an:.2f} -> selector picks {rc['winner_detail']}; "
          f"clean: floor=1.00 -> selector ties floor; selector >= floor on every task)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="score real concept tasks (same-shape subset)")
    ap.add_argument("--rule-probe", action="store_true",
                    help="KILL-GATE: measure rule-hypothesis family recall + within-rearrangement binding "
                         "consistency on the atlas-labelled eval tasks (gates the live rule-bus). Read-only.")
    ap.add_argument("--boundary-probe", action="store_true",
                    help="test the move-to-boundary generalization on the 20 rearrangement tasks (read-only).")
    ap.add_argument("--relocate-probe", action="store_true",
                    help="exact-solve count for the analogy_relocate solver (Lane A) on the 20 tasks (read-only).")
    ap.add_argument("--relocate-test", action="store_true",
                    help="EXACT-ON-TEST: fit from all demos, apply to the real test input, check the solution.")
    ap.add_argument("--compose-test", action="store_true",
                    help="THE COMPOSED NUMBER: all candidate families + Lane A relocate under one "
                         "floor-safe LODO ranking, ARC 2-attempt EXACT-ON-TEST on the full held-out eval.")
    ap.add_argument("--where-value-ceiling", action="store_true",
                    help="PHASE 0: count tasks exactly expressible as object-WHERE ^ src-VALUE that the "
                         "floor CANNOT do (the ceiling of the where x value binding lane). Read-only.")
    ap.add_argument("--value-binding-probe", action="store_true",
                    help="FIX 1: does a CONDITIONED value P(out|src,signature) beat the MARGINAL on the "
                         "multi-target conditional_recolor tasks? oracle-WHERE, LODO, split single/multi.")
    ap.add_argument("--value-min-bucket", type=int, default=1,
                    help="min support count for a conditioned bucket to fire before backoff to marginal.")
    ap.add_argument("--value-probe-category", type=str, default="conditional_recolor",
                    help="FIX H: task category for --value-binding-probe. A single atlas category name, or "
                         "'all-colour' = conditional_recolor OR other (enclosure/adjacency families partly "
                         "hide under 'other').")
    ap.add_argument("--correspondence-probe", action="store_true",
                    help="OPEN RESEARCH: per-object correspondence-to-demo ceiling for multi-target recolor")
    ap.add_argument("--analogy-value-probe", action="store_true",
                    help="cross-demo SOFT nearest-neighbour value transfer -- does soft matching beat the "
                         "34 exact-bucket ceiling (i.e. would a learned head help)?")
    ap.add_argument("--region-value-ceiling", action="store_true",
                    help="PHASE 0B: count tasks exactly expressible as background/region-WHERE ^ "
                         "src-VALUE that the floor CANNOT do. Read-only.")
    ap.add_argument("--projection-ceiling", action="store_true",
                    help="FIX 3 Phase A: size 3a ray / 3b bbox / 3c periodic + FIX 2 fg nearest-seed "
                         "recolor (read-only LODO ceiling; gates what gets wired into CANDIDATES).")
    ap.add_argument("--constant-fill-ceiling", action="store_true",
                    help="PHASE 0C: scan all eval tasks for verified input-region -> one constant "
                         "fill rules that the floor cannot do. Read-only.")
    ap.add_argument("--conditional-recolor-microscope", action="store_true",
                    help="Report changed-cell family statistics for atlas conditional_recolor tasks. "
                         "Read-only; writes reports/conditional_recolor_microscope/per_task.csv.")
    ap.add_argument("--microscope-all", action="store_true",
                    help="With --conditional-recolor-microscope, include all eval tasks/categories instead "
                         "of only same-shape conditional_recolor tasks.")
    ap.add_argument("--no-committed", action="store_true",
                    help="skip the committed DSL candidate in --compose-test (much faster).")
    ap.add_argument("--challenges", default=None, help="override eval challenges JSON for --rule-probe")
    ap.add_argument("--categories", default=None, help="override atlas categories CSV for --rule-probe")
    ap.add_argument("--microscope-out-dir", default=None,
                    help="output directory for --conditional-recolor-microscope")
    ap.add_argument("--json", default="kaggle/combined/arc-agi_concept_challenges.json")
    ap.add_argument("--side", type=int, default=30)
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument(
        "--model-dump",
        default=None,
        help="Optional top-K model dump. --real reads kind=fold; --compose-test requires kind=fold and kind=test.",
    )
    ap.add_argument("--rule-lib", default=None, help="Optional JSON RuleLibrary path for exact recipe storage/reuse.")
    ap.add_argument(
        "--refine-close-miss",
        action="store_true",
        help="Run bounded LODO close-miss refinement and store stable exact recipes in --rule-lib.",
    )
    args = ap.parse_args()
    _self_test()
    if args.rule_probe:
        run_rule_probe(side=args.side, challenges=args.challenges, categories_csv=args.categories)
    if args.boundary_probe:
        run_boundary_probe(side=args.side, challenges=args.challenges, categories_csv=args.categories)
    if args.relocate_probe:
        run_relocate_probe(side=args.side, challenges=args.challenges, categories_csv=args.categories)
    if args.relocate_test:
        run_relocate_test(side=args.side, challenges=args.challenges, categories_csv=args.categories)
    if args.where_value_ceiling:
        run_where_value_ceiling(side=args.side, challenges=args.challenges,
                                categories_csv=args.categories, max_tasks=args.max_tasks)
    if args.value_binding_probe:
        run_value_binding_probe(side=args.side, challenges=args.challenges,
                                categories_csv=args.categories, min_bucket=args.value_min_bucket,
                                max_tasks=args.max_tasks, category=args.value_probe_category)
    if args.correspondence_probe:
        run_correspondence_probe(side=args.side, challenges=args.challenges,
                                 categories_csv=args.categories, max_tasks=args.max_tasks)
    if args.analogy_value_probe:
        run_analogy_value_probe(side=args.side, challenges=args.challenges,
                                categories_csv=args.categories, max_tasks=args.max_tasks)
    if args.region_value_ceiling:
        run_region_value_ceiling(side=args.side, challenges=args.challenges,
                                 categories_csv=args.categories, max_tasks=args.max_tasks)
    if args.projection_ceiling:
        run_projection_ceiling(side=args.side, challenges=args.challenges,
                               categories_csv=args.categories, max_tasks=args.max_tasks)
    if args.constant_fill_ceiling:
        run_constant_fill_ceiling(side=args.side, challenges=args.challenges,
                                  categories_csv=args.categories, max_tasks=args.max_tasks)
    if args.conditional_recolor_microscope:
        run_conditional_recolor_microscope(side=args.side, challenges=args.challenges,
                                           categories_csv=args.categories,
                                           out_dir=args.microscope_out_dir,
                                           max_tasks=args.max_tasks,
                                           include_all=args.microscope_all)
    if args.compose_test:
        run_compose_test(side=args.side, challenges=args.challenges, categories_csv=args.categories,
                         include_committed=not args.no_committed, max_tasks=args.max_tasks,
                         model_dump_path=args.model_dump)
    if args.real:
        p = Path(__file__).resolve().parents[1] / args.json
        if not p.exists():
            print(f"[skip --real] not found: {p}")
        else:
            run_real(
                p,
                side=args.side,
                max_tasks=args.max_tasks,
                model_dump_path=args.model_dump,
                rule_lib_path=args.rule_lib,
                refine_close_miss=args.refine_close_miss,
            )


if __name__ == "__main__":
    main()
