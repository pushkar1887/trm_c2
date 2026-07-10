"""0.2 - the CROSS-DEMO EXTRACTION CHECK: the gate every rule extractor must pass.

An extractor that produces a "rule" which does NOT depend on which demos it sees (or how their
inputs pair with their outputs) is useless. That is exactly what C2 collapsed to: gate ~0,
real_vs_shuffle ~0 -- a "rule" invariant to the task. This check makes that impossible to miss.

Before any extractor is allowed to feed the Rule Bus it must pass these tests on synthetic tasks
with a KNOWN injected rule (recolour src -> dst, consistent across a task's demos):

  1. DEMO-SENSITIVITY  corrupt ONE demo's rule -> the descriptor must MOVE  (proves it reads
                       every demo; an extractor that ignores demos won't react)
  2. AGREEMENT         the per-demo signals are consistent on a real task   (diagnostic)
  3. REAL-vs-SHUFFLE   real demos vs pairing-broken demos -> descriptor must DIFFER  (key test)
  4. KNOWN-RULE        on injected src->dst tasks, recover dst from src      (correctness)
  5. COVERAGE          fraction of changed cells the rule explains (dpcc-like)

VERDICT = PASS iff (sensitivity) AND (real_vs_shuffle) AND (known_rule).

Self-validation: run on TWO extractors; the check MUST tell them apart --
  * ColorTransitionBank.cond_inout/cond_changed  -> PASS (task-specific by construction)
  * ShuffleInvariantPooler (input/output colour MARGINALS, ignores in->out pairing) -> FAIL
    (reproduces C2's collapse). The synthetic makes the rule visible ONLY through pairing
    (a dominant NOISE colour hides dst from the output marginal), so the pooler cannot recover it.
  If the check can't separate these, the check is broken.

Run:  trm\\Scripts\\python.exe scripts\\check_extraction.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from models.recursive_reasoning.color_transition_bank import ColorTransitionBank
from models.recursive_reasoning.object_bank import is_singleton_object, size_bucket

COLOR_OFFSET = 2
N_COLORS = 10
GRID_SIDE = 30
GRID_LEN = GRID_SIDE * GRID_SIDE
NOISE = 0                  # dominant background colour (hides dst from the output marginal)

# Real-data loader constants (mirror run_stage1_local.py so the ranker reads the SAME dataset the
# training panel scores). Used only by --real.
CONFIG_PATH = "checkpoints/TRM-FVR-Experiments/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/all_config.yaml"
CKPT_PATH = r"D:\trm_c2\step_518071"
DATASET_PATH = r"D:\trm_c2\arc1concept-aug-1000"

# Selectable diagnostic probe datasets (each value is the ROOT dir containing train/; the loader
# appends the split, see puzzle_dataset._lazy_load_dataset). aug0 is the UN-AUGMENTED 960-task seed
# of aug-1000: identical intra-puzzle LODO structure (context = the OTHER examples of the SAME
# puzzle, _context_from_flat_examples), but canonical palette, 960 unique tasks, loads in ~7MB.
# It is NOT held out (its tasks overlap aug-1000 training) -- a clean MEASURING stick, not a
# generalization test. aug1000 = the actual training distribution (~913x augmented).
PROBE_DATASETS = {
    "aug1000": DATASET_PATH,
    "aug0": r"D:\trm_c2\TinyRecursiveModels\data\arc1concept-aug-0",
}


# ---------------------------------------------------------------------------
# Synthetic tasks. Task b recolours src_b -> dst_b in every demo. Grids are mostly NOISE so the
# rule is recoverable ONLY from the input->output PAIRING, never from the output marginal.
# ---------------------------------------------------------------------------
from parse import _components_2d, _components_2d_adj, _parse_same, _neighbour_key, _position_key, _shape_key, _shape_class_key, _rank_key, _touch_key, _topology_key, _count_key, _combined_key, _extract_grid, _mode, _d4, _shift, _grid_bg, _enclosed_bg, _objgroups, _d4_canon_hash, _object_keymap
def _fit_combined_recolor_map(support_pairs, side):
    """Recolor keyed on (input_colour, combined neighbour+shape_cls+position). Returns dict or None on a
    within-support collision (the joint key STILL maps one key to two outputs -> beyond these per-cell feats)."""
    m = {}
    for ti, to in support_pairs:
        pin, pout = ti >= COLOR_OFFSET, to >= COLOR_OFFSET
        if not bool((pin == pout).all()):
            return None
        ck = _combined_key(ti.unsqueeze(0), side)[0]
        for i in torch.nonzero(pin).flatten().tolist():
            k = (int(ti[i]) - COLOR_OFFSET, int(ck[i]))
            bcol = int(to[i]) - COLOR_OFFSET
            if m.get(k, bcol) != bcol:
                return None
            m[k] = bcol
    return m


def _apply_combined_recolor(ti, m, side):
    out = ti.clone()
    ck = _combined_key(ti.unsqueeze(0), side)[0]
    for i in torch.nonzero(ti >= COLOR_OFFSET).flatten().tolist():
        k = (int(ti[i]) - COLOR_OFFSET, int(ck[i]))
        if k in m:
            out[i] = m[k] + COLOR_OFFSET
    return out


class _DihedralOp:
    name = "dihedral"
    def candidates(self, gi, go):
        if gi is None or go is None:
            return set()
        return {nm for nm, T in _d4(gi).items() if T.shape == go.shape and bool((T == go).all())}
    def apply(self, gi, tag):
        return _d4(gi).get(tag)


class _TranslateOp:
    name = "translate"
    R = 6
    def candidates(self, gi, go):
        if gi is None or go is None or gi.shape != go.shape:
            return set()
        bg = _mode(gi)
        res = set()
        for dr in range(-self.R, self.R + 1):
            for dc in range(-self.R, self.R + 1):
                if dr == 0 and dc == 0:
                    continue
                if bool((_shift(gi, dr, dc, bg) == go).all()):
                    res.add((dr, dc))
        return res
    def apply(self, gi, tag):
        return _shift(gi, tag[0], tag[1], _mode(gi))


class _TileOp:
    name = "tile"
    def candidates(self, gi, go):
        if gi is None or go is None:
            return set()
        H, W = gi.shape
        Ho, Wo = go.shape
        if H == 0 or W == 0 or Ho % H or Wo % W:
            return set()
        k, l = Ho // H, Wo // W
        if k * l <= 1:
            return set()
        return {(k, l)} if bool((gi.repeat(k, l) == go).all()) else set()
    def apply(self, gi, tag):
        return gi.repeat(tag[0], tag[1])


class _CropOp:
    """Size-change primitive: crop to the non-background content bounding box (the common 'extract the
    drawn region' / size_change family). bg = most-common token. Refuses if there's no shape change."""
    name = "crop"
    @staticmethod
    def _crop(gi):
        bg = _mode(gi)
        nz = (gi != bg).nonzero()
        if nz.numel() == 0:
            return None
        r0, c0 = int(nz[:, 0].min()), int(nz[:, 1].min())
        r1, c1 = int(nz[:, 0].max()), int(nz[:, 1].max())
        return gi[r0:r1 + 1, c0:c1 + 1]
    def candidates(self, gi, go):
        if gi is None or go is None:
            return set()
        c = self._crop(gi)
        if c is None or c.shape == gi.shape:                  # no content / no shape change -> not a crop
            return set()
        return {"content"} if c.shape == go.shape and bool((c == go).all()) else set()
    def apply(self, gi, tag):
        return self._crop(gi)


class _ScaleOp:
    """Size-change primitive: integer UP-scale (each cell -> kxk block) or DOWN-scale (kxk block -> 1
    cell). Covers the 'zoom' size_change family that tile/crop miss."""
    name = "scale"
    MAXK = 5
    def candidates(self, gi, go):
        if gi is None or go is None:
            return set()
        H, W = gi.shape
        Ho, Wo = go.shape
        res = set()
        if H and W and Ho % H == 0 and Wo % W == 0:           # UP
            k = Ho // H
            if k == Wo // W and 2 <= k <= self.MAXK:
                up = gi.repeat_interleave(k, 0).repeat_interleave(k, 1)
                if up.shape == go.shape and bool((up == go).all()):
                    res.add(("up", k))
        if Ho and Wo and H % Ho == 0 and W % Wo == 0:         # DOWN
            k = H // Ho
            if k == W // Wo and 2 <= k <= self.MAXK:
                down = gi[::k, ::k]
                if down.shape == go.shape and bool((down == go).all()):
                    res.add(("down", k))
        return res
    def apply(self, gi, tag):
        kind, k = tag
        if kind == "up":
            return gi.repeat_interleave(k, 0).repeat_interleave(k, 1)
        return gi[::k, ::k]


class _SymmetrizeOp:
    """GEOMETRY prior: complete a symmetry / remove an occlusion. The grid is symmetric under
    S∈{flipH,flipV,rot180,transpose}; an occlusion colour k breaks it; restore each occluded cell from its
    symmetric partner. param = (S, k). In-place (same shape). Targets the large 'fill the symmetric original'
    family currently at 0 ROUTED."""
    name = "symmetrize"
    @staticmethod
    def _S(gi, s):
        if s == "flipH": return torch.flip(gi, [1])
        if s == "flipV": return torch.flip(gi, [0])
        if s == "rot180": return torch.flip(gi, [0, 1])
        if s == "transpose": return gi.t().contiguous() if gi.shape[0] == gi.shape[1] else None
        return None
    def candidates(self, gi, go):
        if gi is None or go is None or gi.shape != go.shape:
            return set()
        res = set()
        for s in ("flipH", "flipV", "rot180", "transpose"):
            sg = self._S(gi, s)
            if sg is None or sg.shape != gi.shape:
                continue
            fill = (gi != go)
            if not bool(fill.any()) or not bool((go[fill] == sg[fill]).all()):
                continue
            ks = gi[fill].unique()
            if ks.numel() == 1:
                res.add((s, int(ks[0])))                 # (symmetry, occlusion colour) -- both demo-agreed
        return res
    def apply(self, gi, tag):
        s, k = tag
        sg = self._S(gi, s)
        if sg is None:
            return None
        out = gi.clone(); mask = (gi == k); out[mask] = sg[mask]
        return out


class _FillOp:
    """TOPOLOGY prior: flood-fill enclosed background regions (bg cells not connected to the border) with a
    colour. param = fill_colour. In-place. Targets the large 'fill the inside of the shape' family at 0 ROUTED."""
    name = "fill"
    def candidates(self, gi, go):
        if gi is None or go is None or gi.shape != go.shape:
            return set()
        enc = _enclosed_bg(gi)
        if not bool(enc.any()) or not bool(((gi != go) == enc).all()):
            return set()
        ks = go[enc].unique()
        return {int(ks[0])} if ks.numel() == 1 else set()
    def apply(self, gi, tag):
        out = gi.clone(); out[_enclosed_bg(gi)] = tag
        return out


class _ExtractObjectOp:
    """OBJECTNESS + size-change: output = ONE selected object cropped to its bbox. Selection criterion is
    demo-agreed (largest/smallest/unique_colour/unique_shape). Targets size_change (output is a sub-object)."""
    name = "extract_object"
    CRITS = ("largest", "smallest", "unique_colour", "unique_shape")
    @staticmethod
    def _objects(gi):
        bg = int(_mode(gi))
        lf = _components_2d_adj(gi, bg=bg).reshape(-1)
        objs = {}
        for i in range(lf.numel()):
            l = int(lf[i])
            if l >= 0:
                objs.setdefault(l, []).append(i)
        return objs, gi.reshape(-1), bg
    @staticmethod
    def _sig(cells, W):
        rs = [c // W for c in cells]; cs = [c % W for c in cells]
        r0, c0 = min(rs), min(cs)
        return tuple(sorted((r - r0, c - c0) for r, c in zip(rs, cs)))
    def _select(self, gi, crit):
        objs, gf, bg = self._objects(gi)
        if not objs:
            return None
        W = gi.shape[1]; items = list(objs.items())
        if crit in ("largest", "smallest"):
            target = (max if crit == "largest" else min)(len(c) for _, c in items)
            chosen = [l for l, c in items if len(c) == target]
        elif crit == "unique_colour":
            from collections import Counter as _C
            cols = {l: tuple(sorted(int(gf[x]) for x in c)) for l, c in items}
            freq = _C(cols.values()); chosen = [l for l, cv in cols.items() if freq[cv] == 1]
        else:  # unique_shape
            from collections import Counter as _C
            sigs = {l: self._sig(c, W) for l, c in items}
            freq = _C(sigs.values()); chosen = [l for l, sv in sigs.items() if freq[sv] == 1]
        if len(chosen) != 1:
            return None                                          # ambiguous selection -> refuse (floor-safe)
        cells = objs[chosen[0]]
        rs = [c // W for c in cells]; cs = [c % W for c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        out = torch.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=gi.dtype)
        for c in cells:
            out[c // W - r0, c % W - c0] = gf[c]
        return out
    def candidates(self, gi, go):
        if gi is None or go is None:
            return set()
        res = set()
        for crit in self.CRITS:
            ex = self._select(gi, crit)
            if ex is not None and ex.shape == go.shape and bool((ex == go).all()):
                res.add(crit)
        return res
    def apply(self, gi, crit):
        return self._select(gi, crit)


class _PanelOp:
    """CROSS-PANEL (topology/space): a full constant row/col separator splits the grid into two equal panels;
    output = a boolean combine (and/or/xor/diff) of the panels' foreground, painted one colour. param=(op, colour).
    A non-background separator is preferred. Targets the two-panel logic family."""
    name = "panel"
    @staticmethod
    def _split(gi):
        H, W = gi.shape; bg = int(_mode(gi)); fallback = None
        for c in range(W):
            col = gi[:, c]
            if col.unique().numel() == 1:
                left, right = gi[:, :c], gi[:, c + 1:]
                if left.numel() and left.shape == right.shape:
                    if int(col[0]) != bg:
                        return left, right
                    fallback = fallback or (left, right)
        for r in range(H):
            row = gi[r, :]
            if row.unique().numel() == 1:
                top, bot = gi[:r, :], gi[r + 1:, :]
                if top.numel() and top.shape == bot.shape:
                    if int(row[0]) != bg:
                        return top, bot
                    fallback = fallback or (top, bot)
        return fallback
    @staticmethod
    def _combine(opn, m1, m2):
        if opn == "and": return m1 & m2
        if opn == "or": return m1 | m2
        if opn == "xor": return m1 ^ m2
        return m1 & (~m2)                                        # diff
    def candidates(self, gi, go):
        if gi is None or go is None:
            return set()
        split = self._split(gi)
        if split is None:
            return set()
        p1, p2 = split
        if p1.shape != go.shape:
            return set()
        bg = int(_mode(gi)); m1 = p1 != bg; m2 = p2 != bg; res = set()
        for opn in ("and", "or", "xor", "diff"):
            comb = self._combine(opn, m1, m2)
            if not bool(comb.any()):
                continue
            gov = go[comb]
            if gov.unique().numel() != 1:
                continue
            if bool((go[~comb] == bg).all()):
                res.add((opn, int(gov.reshape(-1)[0])))
        return res
    def apply(self, gi, tag):
        split = self._split(gi)
        if split is None:
            return None
        opn, oc = tag; p1, p2 = split; bg = int(_mode(gi))
        comb = self._combine(opn, p1 != bg, p2 != bg)
        out = torch.full(p1.shape, bg, dtype=gi.dtype)
        out[comb] = oc
        return out


class _FractalOp:
    """GEOMETRY/size-change (B2): self-tiling fractal. Output is (H*H, W*W); block (i,j) = a COPY of the input
    where input cell (i,j) 'fires', else background. param = 'nonbg' (fire on non-bg cells) or 'inv' (fire on bg
    cells). Targets the classic fractal family (e.g. 007bbfb7). The output SHAPE is induced by the (H*H,W*W) check."""
    name = "fractal"
    @staticmethod
    def _build(gi, mode):
        H, W = gi.shape; bg = int(_mode(gi))
        out = torch.full((H * H, W * W), bg, dtype=gi.dtype)
        for i in range(H):
            for j in range(W):
                fire = (int(gi[i, j]) != bg) if mode == "nonbg" else (int(gi[i, j]) == bg)
                if fire:
                    out[i * H:(i + 1) * H, j * W:(j + 1) * W] = gi
        return out
    def candidates(self, gi, go):
        if gi is None or go is None:
            return set()
        H, W = gi.shape
        if H * H > GRID_SIDE or W * W > GRID_SIDE or go.shape != (H * H, W * W):
            return set()
        return {m for m in ("nonbg", "inv") if bool((self._build(gi, m) == go).all())}
    def apply(self, gi, mode):
        H, W = gi.shape
        if H * H > GRID_SIDE or W * W > GRID_SIDE:
            return None
        return self._build(gi, mode)


class _MirrorConcatOp:
    """GEOMETRY/size-change (B2): append a mirror image -> a doubled, mirrored grid. param = (axis, side):
    h -> [in|flipH] (side=right) / [flipH|in] (left), shape H x 2W; v -> stacked, 2H x W. Targets grid-level
    symmetry-completion where the output is the input plus its reflection."""
    name = "mirror_concat"
    @staticmethod
    def _build(gi, tag):
        axis, side = tag
        mir = torch.flip(gi, [1] if axis == "h" else [0])
        dim = 1 if axis == "h" else 0
        return torch.cat([gi, mir], dim=dim) if side == "right" else torch.cat([mir, gi], dim=dim)
    def candidates(self, gi, go):
        if gi is None or go is None:
            return set()
        H, W = gi.shape; res = set()
        for axis in ("h", "v"):
            if go.shape != ((H, 2 * W) if axis == "h" else (2 * H, W)):
                continue
            for side in ("right", "left"):
                if bool((self._build(gi, (axis, side)) == go).all()):
                    res.add((axis, side))
        return res
    def apply(self, gi, tag):
        return self._build(gi, tag)


def _default_geo_ops():
    """The canonical geometric/size DSL op list -- SINGLE SOURCE for s2_selector_probe + the ARC-AGI eval harness
    (eval_arc_agi1.py) so they never drift."""
    return [_DihedralOp(), _TranslateOp(), _TileOp(), _CropOp(), _ScaleOp(), _SymmetrizeOp(), _FillOp(),
            _ExtractObjectOp(), _PanelOp(), _FractalOp(), _MirrorConcatOp()]


def _op1_options(op, gi):
    """All (param -> output grid) op1 can produce from gi, for 2-op composition (op1 THEN recolor).
    Unlike op.candidates (exact-match), this ENUMERATES op1's outputs so a following recolor finishes the
    colour. translate is skipped (shape-preserving -> recolor alone already covers it)."""
    if gi is None:
        return {}
    n = op.name
    if n == "dihedral":
        return _d4(gi)
    if n == "scale":
        H, W = gi.shape
        out = {("up", k): gi.repeat_interleave(k, 0).repeat_interleave(k, 1) for k in range(2, op.MAXK + 1)}
        for k in range(2, op.MAXK + 1):
            if H % k == 0 and W % k == 0:
                out[("down", k)] = gi[::k, ::k]
        return out
    if n == "crop":
        c = _CropOp._crop(gi)
        return {"content": c} if (c is not None and c.shape != gi.shape) else {}
    if n == "tile":
        return {(k, l): gi.repeat(k, l) for k in range(1, 4) for l in range(1, 4) if k * l > 1}
    return {}


def _fit_recolor_2d(pairs):
    """Flat colour map a->b over equal-shape 2D grids; None on conflict (within OR across pairs)."""
    m = {}
    for gi, go in pairs:
        if gi is None or go is None or gi.shape != go.shape:
            return None
        ai, bo = gi.reshape(-1), go.reshape(-1)
        for a in ai.unique().tolist():
            outs = bo[ai == a].unique()
            if outs.numel() != 1:
                return None
            b = int(outs[0])
            if m.get(a, b) != b:
                return None
            m[a] = b
    return m


def _apply_recolor_2d(gi, m):
    if not m:
        return gi
    hi = max([int(gi.max())] + list(m.keys()) + list(m.values()))
    lut = torch.arange(hi + 1)
    for k, v in m.items():
        lut[k] = v
    return lut[gi]


def _compose2_predict(gin, gout, sup, h, geo_ops):
    """2-op program: op1 (geometric/size) THEN a flat recolor, params AGREED across support.
    Returns (recipe, pred) if an agreed rule is found, else None. Non-peeking."""
    gih = gin.get(h)
    if gih is None:
        return None
    for op in geo_ops:
        if op.name == "translate":
            continue
        opts = {m: _op1_options(op, gin[m]) for m in sup}
        opts_h = _op1_options(op, gih)
        for p in opts[sup[0]]:
            t_pairs, ok = [], True
            for m in sup:
                t = opts[m].get(p)
                if t is None or t.shape != gout[m].shape:
                    ok = False
                    break
                t_pairs.append((t, gout[m]))
            if not ok:
                continue
            rm = _fit_recolor_2d(t_pairs)
            if rm is None:
                continue
            th = opts_h.get(p)
            if th is None:
                continue
            pred = _apply_recolor_2d(th, rm)
            return ([op.name, "recolor"], pred)
    return None


def _compose2_solve(gin, gout, sup, h, geo_ops):
    """2-op program: op1 (geometric/size) THEN a flat recolor, params AGREED across support, reconstructs
    the held-out EXACTLY -> recipe list; else None. The Stage-2 composition the 1-op DSL cannot express
    (scale->recolor, crop->recolor, dihedral->recolor, ...). sup=[h] gives the compose2 oracle."""
    c2 = _compose2_predict(gin, gout, sup, h, geo_ops)
    if c2 is not None:
        recipe, pred = c2
        goh = gout.get(h)
        if goh is not None and pred.shape == goh.shape and bool((pred == goh).all()):
            return recipe
    return None


def _execute_recipe_predict(gin, gout, sup, h, recipe, geo_ops):
    """A2 (rule-memory REUSE): EXECUTE a stored macro = a list of op NAMES, params RE-FIT per task (the params are
    NEVER stored -- the cross-demo pillar). Returns predicted grid if an agreed rule is found, else None. Non-peeking."""
    if len(recipe) == 2 and recipe[0] == "object_relrecolor":
        ids = list(sup) + [h]
        labels = {m: _parse_same(gin[m]) for m in ids if gin.get(m) is not None}
        return _object_relrecolor_predict(gin, gout, sup, h, labels, recipe[1])

    if recipe == ["set_cover", "clause_union"]:
        ids = list(sup) + [h]
        labels = {m: _parse_same(gin[m]) for m in ids if gin.get(m) is not None}
        return _set_cover_predict(gin, gout, sup, h, labels)

    SHAPE = {"dihedral", "scale", "crop", "tile", "translate", "symmetrize", "fill", "extract_object", "panel"}
    if len(recipe) != 2 or recipe[0] not in SHAPE or recipe[1] != "recolor":
        return None                                              # not an executable [SHAPE, recolor] recipe (typed)
    op = next((o for o in geo_ops if o.name == recipe[0]), None)
    if op is None:
        return None
    gih = gin.get(h)
    if gih is None:
        return None
    opts = {m: _op1_options(op, gin[m]) for m in sup}
    opts_h = _op1_options(op, gih)
    for p in opts[sup[0]]:                                       # re-fit op1's param by cross-demo agreement
        t_pairs, ok = [], True
        for m in sup:
            t = opts[m].get(p)
            if t is None or t.shape != gout[m].shape:
                ok = False; break
            t_pairs.append((t, gout[m]))
        if not ok:
            continue
        rm = _fit_recolor_2d(t_pairs)                            # re-fit the recolor map on the transformed inputs
        if rm is None:
            continue
        th = opts_h.get(p)
        if th is None:
            continue
        return _apply_recolor_2d(th, rm)
    return None


def _execute_recipe_solve(gin, gout, sup, h, recipe, geo_ops):
    """A2 (rule-memory REUSE): EXECUTE a stored macro = a list of op NAMES, params RE-FIT per task (the params are
    NEVER stored -- the cross-demo pillar), LODO-verified on the held-out. TYPED pruning: a SHAPE_OP must not
    follow a COLOR_OP (reshape-after-recolor is redundant). v1 executes the [SHAPE_OP, 'recolor'] recipes the
    depth-2 search stores (re-fit op1's param by agreement, then a flat recolor); longer/other recipes -> None so
    the caller falls through to the live search. Floor-safe by the final exact check. Returns True/False/None."""
    if len(recipe) == 2 and recipe[0] == "object_relrecolor":
        pred = _execute_recipe_predict(gin, gout, sup, h, recipe, geo_ops)
        goh = gout.get(h)
        if pred is not None and goh is not None and pred.shape == goh.shape and bool((pred == goh).all()):
            return True
        return False

    if recipe == ["set_cover", "clause_union"]:
        pred = _execute_recipe_predict(gin, gout, sup, h, recipe, geo_ops)
        goh = gout.get(h)
        if pred is not None and goh is not None and pred.shape == goh.shape and bool((pred == goh).all()):
            return True
        return False

    SHAPE = {"dihedral", "scale", "crop", "tile", "translate", "symmetrize", "fill", "extract_object", "panel"}
    if len(recipe) != 2 or recipe[0] not in SHAPE or recipe[1] != "recolor":
        return None
    pred = _execute_recipe_predict(gin, gout, sup, h, recipe, geo_ops)
    if pred is not None:
        goh = gout.get(h)
        if goh is not None and pred.shape == goh.shape and bool((pred == goh).all()):
            return True
    return False


def _object_recolor_solve(gin, gout, sup, h, labels, key):
    """OBJECT-LEVEL recolor candidate (Stage-1): parse each grid into objects (precomputed `labels`), key each
    object (rank=size-rank / shape=translation-invariant pixel hash / colourset), fit key->out_colour across
    SUPPORT by cross-demo AGREEMENT (REFUSE on non-monochrome object output or key->two-colour conflict), apply
    to the held-out. Returns True iff it reconstructs gout[h] EXACTLY. In-place only (shape must match).
    Expresses 'the largest object -> red, the line -> blue' which per-cell / neighbour maps cannot."""
    if any(gin.get(m) is None or labels.get(m) is None for m in list(sup) + [h]):
        return False
    def objs(m):
        g2d, lab = gin[m], labels[m]
        bg = int(_mode(g2d))
        gf, lf = g2d.reshape(-1), lab.reshape(-1)
        groups = {}
        for i in range(gf.numel()):
            if int(gf[i]) == bg:
                continue
            l = int(lf[i])
            if l < 0:
                continue
            groups.setdefault(l, []).append(i)
        return groups
    def keymap(g2d, groups):
        H, W = g2d.shape
        if key == "rank":                                        # size-RANK among foreground objects
            ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
            return {l: r for r, (l, _) in enumerate(ordered)}
        gf = g2d.reshape(-1)
        if key == "count_same":                                  # how many objects share this object's colour
            from collections import Counter as _Counter
            col = {l: int(gf[cells[0]]) for l, cells in groups.items()}
            cnt = _Counter(col.values())
            return {l: min(cnt[col[l]], 4) for l in groups}
        if key in ("inside", "between", "aligned", "nearest_colour"):   # RELATIONAL keys (U5, harvested)
            geom = {}                                            # label -> (r0,r1,c0,c1, mean_r, mean_c, colour)
            for l, cells in groups.items():
                rs = [c // W for c in cells]; cs = [c % W for c in cells]
                geom[l] = (min(rs), max(rs), min(cs), max(cs),
                           sum(rs) / len(rs), sum(cs) / len(cs), int(gf[cells[0]]))
            labels = list(groups); km = {}
            for l in labels:
                r0, r1, c0, c1, mr, mc, _ = geom[l]
                others = [o for o in labels if o != l]
                if key == "inside":                              # topological role: 2=inside / 1=container / 0=free
                    ins = any(geom[o][0] < r0 and geom[o][1] > r1 and geom[o][2] < c0 and geom[o][3] > c1
                              for o in others)
                    con = any(geom[o][0] > r0 and geom[o][1] < r1 and geom[o][2] > c0 and geom[o][3] < c1
                              for o in others)
                    km[l] = 2 if ins else (1 if con else 0)
                elif key == "aligned":                           # shares row/col centroid with another (0..3)
                    row = any(abs(geom[o][4] - mr) < 0.5 for o in others)
                    col = any(abs(geom[o][5] - mc) < 0.5 for o in others)
                    km[l] = (1 if row else 0) + (2 if col else 0)
                elif key == "between":                           # collinearly between two other objects
                    btw = 0
                    for i in range(len(others)):
                        a = geom[others[i]]
                        for j in range(i + 1, len(others)):
                            b = geom[others[j]]
                            if abs(a[4] - mr) < 0.5 and abs(b[4] - mr) < 0.5 and min(a[5], b[5]) < mc < max(a[5], b[5]):
                                btw = 1
                            if abs(a[5] - mc) < 0.5 and abs(b[5] - mc) < 0.5 and min(a[4], b[4]) < mr < max(a[4], b[4]):
                                btw = 1
                        if btw:
                            break
                    km[l] = btw
                else:                                            # nearest_colour: colour of the nearest other object
                    if others:
                        nn = min(others, key=lambda o: abs(geom[o][4] - mr) + abs(geom[o][5] - mc))
                        km[l] = geom[nn][6]
                    else:
                        km[l] = -1
            return km
        km = {}
        for l, cells in groups.items():
            rs = [c // W for c in cells]; cs = [c % W for c in cells]
            if key == "shape":                                   # translation-invariant exact pixel hash
                r0, c0 = min(rs), min(cs)
                km[l] = hash(tuple(sorted((r - r0, c - c0) for r, c in zip(rs, cs)))) & 0x7fffffff
            elif key == "shape_d4":                              # D4-canonical (rotation/reflection-invariant) hash
                r0, c0 = min(rs), min(cs)
                km[l] = _d4_canon_hash([(r - r0, c - c0) for r, c in zip(rs, cs)])
            elif key == "position":                              # centroid band -> 9 buckets (3x3 thirds)
                mr = sum(rs) / len(rs); mc = sum(cs) / len(cs)
                rb = min(int(mr * 3 // max(H, 1)), 2); cb = min(int(mc * 3 // max(W, 1)), 2)
                km[l] = rb * 3 + cb
            elif key == "touch_border":                          # does the object touch the grid edge
                km[l] = int(any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in zip(rs, cs)))
            elif key == "colour":                                # the object's own (first-cell) colour
                km[l] = int(gf[cells[0]])
            else:  # colourset                                   # sorted colour multiset
                km[l] = tuple(sorted(int(gf[c]) for c in cells))
        return km
    m_map = {}
    for m in sup:
        if gin[m].shape != gout[m].shape:
            return False
        groups = objs(m); km = keymap(gin[m], groups); gof = gout[m].reshape(-1)
        for l, cells in groups.items():
            outs = {int(gof[c]) for c in cells}
            if len(outs) != 1:
                return False                                   # object output not monochrome -> not object-recolor
            ocol = outs.pop()
            if m_map.get(km[l], ocol) != ocol:
                return False                                   # cross-demo / cross-object key conflict
            m_map[km[l]] = ocol
    if not m_map or gin[h].shape != gout[h].shape:
        return False
    groups = objs(h); km = keymap(gin[h], groups)
    pred = gin[h].clone(); pf = pred.reshape(-1)
    for l, cells in groups.items():
        if km[l] in m_map:
            for c in cells:
                pf[c] = m_map[km[l]]
    return bool((pred == gout[h]).all())


def _fit_recolor_map(support_pairs):
    """support_pairs = [(ti,to) full-token [L]]. Returns dict a->b over input colours IF every support
    demo is recolour-in-place (colour POSITIONS preserved) AND the per-input-colour map is conflict-free
    across demos; else None (recolor cannot express it -> conditional / structural)."""
    m = {}
    for ti, to in support_pairs:
        pin, pout = ti >= COLOR_OFFSET, to >= COLOR_OFFSET
        if not bool((pin == pout).all()):
            return None                                  # positions move -> not a recolour
        for i in torch.nonzero(pin).flatten().tolist():
            a, bcol = int(ti[i]) - COLOR_OFFSET, int(to[i]) - COLOR_OFFSET
            if m.get(a, bcol) != bcol:
                return None                              # same input colour -> 2 outputs: conditional
            m[a] = bcol
    return m


def _apply_recolor(ti, m):
    out = ti.clone()
    for i in torch.nonzero(ti >= COLOR_OFFSET).flatten().tolist():
        a = int(ti[i]) - COLOR_OFFSET
        if a in m:
            out[i] = m[a] + COLOR_OFFSET
    return out


def _fit_neighbour_recolor_map(support_pairs, side):
    """NEIGHBOUR-CONDITIONED recolor: like _fit_recolor_map but keyed on (input_colour, differing-4-
    neighbour-colour) instead of input_colour alone -- the conditional extension the flat map REFUSES.
    Returns dict (a, nb)->b IF every support demo is recolour-in-place AND no (colour,neighbour) key maps
    to two outputs; else None (rule is conditional BEYOND the 1-hop neighbour, or not in-place). The
    deterministic version of the missing primitive (R17): nb from _neighbour_key (most-common adjacent
    colour that differs; N_COLORS = 'no differing neighbour')."""
    m = {}
    for ti, to in support_pairs:
        pin, pout = ti >= COLOR_OFFSET, to >= COLOR_OFFSET
        if not bool((pin == pout).all()):
            return None                                  # positions move -> not a recolour
        nb = _neighbour_key(ti.unsqueeze(0), side)[0]    # [L] in 0..N_COLORS
        for i in torch.nonzero(pin).flatten().tolist():
            k = (int(ti[i]) - COLOR_OFFSET, int(nb[i]))
            bcol = int(to[i]) - COLOR_OFFSET
            if m.get(k, bcol) != bcol:
                return None                              # same (colour,neighbour) -> 2 outputs: deeper-conditional
            m[k] = bcol
    return m


def _apply_neighbour_recolor(ti, m, side):
    """Apply a (colour, differing-neighbour)->colour map per cell. Unseen key -> identity (leave as-is),
    so an EXACT full-grid reconstruction REQUIRES every held-out changed cell's key to have recurred in
    support -- the honest exactness-tax test (per-cell coverage != full-grid exact)."""
    out = ti.clone()
    nb = _neighbour_key(ti.unsqueeze(0), side)[0]
    for i in torch.nonzero(ti >= COLOR_OFFSET).flatten().tolist():
        k = (int(ti[i]) - COLOR_OFFSET, int(nb[i]))
        if k in m:
            out[i] = m[k] + COLOR_OFFSET
    return out


def _object_relrecolor_predict(gin, gout, sup, h, labels, relation):
    """COPY-BY-RELATION (flavor b, U11): recolor each object to the COLOUR OF A RELATED OBJECT
    (relation ∈ {nearest, container, contained, largest, smallest}) -- NOT a fixed per-key colour. The only thing
    fit cross-demo is the RELATION TYPE; the output colours are read dynamically from the related objects. Reads
    INPUT colours (simultaneous semantics), writes output; objects with no related object stay unchanged. Fit the
    relation on support (require EXACT support reconstruction + non-vacuous), then APPLY to the held-out INPUT ->
    prediction grid (NON-PEEKING; gout[h] never read), or None. `_object_relrecolor_solve` wraps with the check."""
    if any(gin.get(m) is None or labels.get(m) is None for m in list(sup) + [h]):
        return None
    def objects(m):
        g2d, lab = gin[m], labels[m]
        bg = int(_mode(g2d)); gf, lf = g2d.reshape(-1), lab.reshape(-1); W = g2d.shape[1]
        objs = {}
        for i in range(gf.numel()):
            if int(gf[i]) == bg:
                continue
            l = int(lf[i])
            if l < 0:
                continue
            objs.setdefault(l, []).append(i)
        info = {}
        for l, cells in objs.items():
            rs = [c // W for c in cells]; cs = [c % W for c in cells]
            info[l] = {"colour": int(gf[cells[0]]), "bbox": (min(rs), max(rs), min(cs), max(cs)),
                       "cen": (sum(rs) / len(rs), sum(cs) / len(cs)), "size": len(cells)}
        return objs, info
    def related(l, info):
        I = info[l]; others = [o for o in info if o != l]
        if not others:
            return None
        mr, mc = I["cen"]
        if relation == "nearest":
            dists = [(abs(info[o]["cen"][0] - mr) + abs(info[o]["cen"][1] - mc), o) for o in others]
            dmin = min(d for d, _ in dists); tied = [o for d, o in dists if d == dmin]
            return tied[0] if len({info[o]["colour"] for o in tied}) == 1 else None   # tie of DIFFERING colours -> refuse
        if relation == "container":
            r0, r1, c0, c1 = I["bbox"]
            cont = [o for o in others if info[o]["bbox"][0] < r0 and info[o]["bbox"][1] > r1
                    and info[o]["bbox"][2] < c0 and info[o]["bbox"][3] > c1]
            return cont[0] if len(cont) == 1 else None
        if relation == "contained":
            r0, r1, c0, c1 = I["bbox"]
            inn = [o for o in others if info[o]["bbox"][0] > r0 and info[o]["bbox"][1] < r1
                   and info[o]["bbox"][2] > c0 and info[o]["bbox"][3] < c1]
            return inn[0] if len(inn) == 1 else None
        if relation == "aligned":                                # the object sharing this one's row/col centroid
            al = [o for o in others if abs(info[o]["cen"][0] - mr) < 0.5 or abs(info[o]["cen"][1] - mc) < 0.5]
            if not al:
                return None
            return al[0] if len({info[o]["colour"] for o in al}) == 1 else None        # single aligned colour else refuse
        if relation == "between":                                # this object collinearly between two others
            endpoints = set()
            for i in range(len(others)):
                a = info[others[i]]["cen"]
                for j in range(i + 1, len(others)):
                    b = info[others[j]]["cen"]
                    row = abs(a[0] - mr) < 0.5 and abs(b[0] - mr) < 0.5 and min(a[1], b[1]) < mc < max(a[1], b[1])
                    col = abs(a[1] - mc) < 0.5 and abs(b[1] - mc) < 0.5 and min(a[0], b[0]) < mr < max(a[0], b[0])
                    if row or col:
                        endpoints.add(others[i]); endpoints.add(others[j])
            if not endpoints:
                return None
            return next(iter(endpoints)) if len({info[o]["colour"] for o in endpoints}) == 1 else None
        if relation in ("largest", "smallest"):
            ext = (max if relation == "largest" else min)(info[o]["size"] for o in info)
            cand = [o for o in info if info[o]["size"] == ext]
            return cand[0] if (len(cand) == 1 and cand[0] != l) else None      # the anchor itself stays
        return None
    def apply_to(m):
        objs, info = objects(m)
        pred = gin[m].clone(); pf = pred.reshape(-1); changed = False
        for l, cells in objs.items():
            rel = related(l, info)
            if rel is not None:
                col = info[rel]["colour"]
                if col != info[l]["colour"]:
                    changed = True
                for c in cells:
                    pf[c] = col
        return pred, changed
    changed_any = False
    for m in sup:
        if gin[m].shape != gout[m].shape:
            return None
        pred, ch = apply_to(m)
        if not bool((pred == gout[m]).all()):
            return None
        changed_any = changed_any or ch
    if not changed_any:
        return None                                              # vacuous (identity) -> not a real relational solve
    pred_h, _ = apply_to(h)
    return pred_h                                                # held-out prediction (NON-PEEKING; no gout[h] read)


def _object_relrecolor_solve(gin, gout, sup, h, labels, relation):
    """Ladder wrapper: predict, then require the held-out EXACT reconstruction (verified solve)."""
    pred = _object_relrecolor_predict(gin, gout, sup, h, labels, relation)
    return pred is not None and gout.get(h) is not None and pred.shape == gout[h].shape and bool((pred == gout[h]).all())


def _set_cover_predict(gin, gout, sup, h, labels):
    """SET-COVER recolor (U6, B1): explain the changed cells with a UNION of SAFE per-(key,value) clauses --
    'objects with key K == v -> colour c' -- where DIFFERENT objects may be explained by DIFFERENT keys
    ('largest -> red AND border-touching -> blue' in ONE program). A clause is SAFE iff across ALL support demos
    every object with K==v is monochrome and maps to the SAME colour c, and >= 1 such object actually changes.
    Safe clauses provably cannot conflict on a shared object, so the UNION is correct AND maximal (no greedy). Fit
    on support (require EXACT support reconstruction + non-vacuous), then APPLY to the held-out INPUT -> prediction
    grid (NON-PEEKING; gout[h] never read), or None. `_set_cover_solve` wraps with the held-out exact check."""
    if any(gin.get(m) is None or labels.get(m) is None for m in list(sup) + [h]):
        return None
    KEYS = ("rank", "shape", "colourset", "inside", "nearest_colour", "touch_border", "count_same")
    for m in sup:
        if gin[m].shape != gout[m].shape:
            return None
    def objs(m):
        g2d, lab = gin[m], labels[m]
        bg = int(_mode(g2d)); gf, lf = g2d.reshape(-1), lab.reshape(-1); groups = {}
        for i in range(gf.numel()):
            if int(gf[i]) == bg:
                continue
            l = int(lf[i])
            if l < 0:
                continue
            groups.setdefault(l, []).append(i)
        return groups
    sup_groups = {m: objs(m) for m in sup}
    # ---- harvest SAFE + CHANGING clauses: (key, value) -> colour
    clauses = {}
    for key in KEYS:
        val_cols, val_change, bad = {}, set(), set()
        for m in sup:
            groups = sup_groups[m]; km = _object_keymap(gin[m], groups, key)
            gf, gof = gin[m].reshape(-1), gout[m].reshape(-1)
            for l, cells in groups.items():
                v = km[l]; outs = {int(gof[c]) for c in cells}
                if len(outs) != 1:
                    bad.add(v); continue                          # object not monochrome under this key-value
                oc = outs.pop(); val_cols.setdefault(v, set()).add(oc)
                if any(int(gf[c]) != oc for c in cells):
                    val_change.add(v)
        for v, cols in val_cols.items():
            if v not in bad and len(cols) == 1 and v in val_change:
                clauses[(key, v)] = next(iter(cols))              # SAFE + changing
    if not clauses:
        return None
    def apply_clauses(m):
        groups = objs(m); pred = gin[m].clone(); pf = pred.reshape(-1)
        kms = {key: _object_keymap(gin[m], groups, key) for key in KEYS}
        for l, cells in groups.items():
            target = None
            for key in KEYS:
                c = clauses.get((key, kms[key].get(l)))
                if c is None:
                    continue
                if target is None:
                    target = c
                elif target != c:
                    return None                                   # cross-key disagreement -> invalid composite
            if target is not None:
                for cell in cells:
                    pf[cell] = target
        return pred
    changed = False
    for m in sup:
        pred = apply_clauses(m)
        if pred is None or not bool((pred == gout[m]).all()):
            return None
        if bool((pred != gin[m]).any()):
            changed = True
    if not changed:
        return None
    return apply_clauses(h)                                       # held-out prediction (NON-PEEKING; no gout[h] read)


def _set_cover_solve(gin, gout, sup, h, labels):
    """Ladder wrapper: predict, then require the held-out EXACT reconstruction (verified solve)."""
    pred = _set_cover_predict(gin, gout, sup, h, labels)
    return pred is not None and gout.get(h) is not None and pred.shape == gout[h].shape and bool((pred == gout[h]).all())


def _distance_recolor_solve(gin, gout, sup, h):
    """Generalised primitive: Distance-based Recolor (e.g. concentric rings).
    Maps Chebyshev/Manhattan distance from a 'source' colour to an output colour.
    Supports modulo cycles (e.g. distance % 2) to generalise from small to large demos."""
    if gin[h] is None or gout[h] is None or gin[h].shape != gout[h].shape:
        return False
    for c_t in range(COLOR_OFFSET, COLOR_OFFSET + N_COLORS):
        for dist_type in ("manhattan", "chebyshev"):
            for P in [0, 1, 2, 3, 4, 5]:
                sup_map = {}
                possible_P = True
                for m in sup:
                    g_in, g_out = gin[m], gout[m]
                    if g_in is None or g_out is None or g_in.shape != g_out.shape:
                        possible_P = False; break
                    coords_t = (g_in == c_t).nonzero(as_tuple=False)
                    if len(coords_t) == 0:
                        possible_P = False; break
                    H, W = g_in.shape
                    grid_r = torch.arange(H).view(-1, 1).expand(H, W)
                    grid_c = torch.arange(W).view(1, -1).expand(H, W)
                    dists = torch.full((H, W), 9999, dtype=torch.long)
                    for r, c in coords_t:
                        if dist_type == "manhattan":
                            d = torch.abs(grid_r - r) + torch.abs(grid_c - c)
                        else:
                            d = torch.maximum(torch.abs(grid_r - r), torch.abs(grid_c - c))
                        dists = torch.minimum(dists, d)
                    demo_map = {}
                    for r in range(H):
                        for c in range(W):
                            # skip if the pixel itself is c_t and doesn't change?
                            # no, we map everything including distance 0
                            d = int(dists[r, c])
                            key = d if P == 0 else d % P
                            out_c = int(g_out[r, c])
                            if key in demo_map and demo_map[key] != out_c:
                                possible_P = False; break
                            demo_map[key] = out_c
                        if not possible_P: break
                    if not possible_P: break
                    for k, v in demo_map.items():
                        if k in sup_map and sup_map[k] != v:
                            possible_P = False; break
                        sup_map[k] = v
                if not possible_P or not sup_map: continue
                g_in_h = gin[h]
                coords_t_h = (g_in_h == c_t).nonzero(as_tuple=False)
                if len(coords_t_h) == 0: continue
                H, W = g_in_h.shape
                grid_r = torch.arange(H).view(-1, 1).expand(H, W)
                grid_c = torch.arange(W).view(1, -1).expand(H, W)
                dists_h = torch.full((H, W), 9999, dtype=torch.long)
                for r, c in coords_t_h:
                    if dist_type == "manhattan":
                        d = torch.abs(grid_r - r) + torch.abs(grid_c - c)
                    else:
                        d = torch.maximum(torch.abs(grid_r - r), torch.abs(grid_c - c))
                    dists_h = torch.minimum(dists_h, d)
                pred = g_in_h.clone()
                success = True
                for r in range(H):
                    for c in range(W):
                        d = int(dists_h[r, c])
                        key = d if P == 0 else d % P
                        if key not in sup_map:
                            success = False; break
                        pred[r, c] = sup_map[key]
                if success and bool((pred == gout[h]).all()):
                    return True
    return False


def _hole_filler_recolor_solve(gin, gout, sup, h):
    """Disabled leakage-prone solver.

    The removed implementation wrote into the held-out output dictionary after
    deriving a prediction for the held-out example. Keeping this function as a
    hard no-op preserves call-site compatibility while making it impossible to
    re-enable by deleting one guard.
    """
    return False


def _legend_match_solve(gin, gout, sup, h):
    """Generalised primitive: Legend/Neighbor-Shape Match.
    Finds a consistent mapping from the exact shape of an adjacent 'legend' object -> Output Colour.
    Solves tasks where objects are recoloured based on the shape of their attached markers."""
    if gin[h] is None or gout[h] is None or gin[h].shape != gout[h].shape:
        return False
        
    def get_objects_of_color(g, c):
        mask = (g == c).long()
        labels = _components_2d_adj(mask, bg=0)
        objs = []
        for u in labels.unique():
            if u == -1: continue
            obj_mask = (labels == u)
            coords = obj_mask.nonzero(as_tuple=False)
            if len(coords) == 0: continue
            r_min, r_max = int(coords[:, 0].min()), int(coords[:, 0].max())
            c_min, c_max = int(coords[:, 1].min()), int(coords[:, 1].max())
            bbox = obj_mask[r_min:r_max+1, c_min:c_max+1]
            shape_hash = str(bbox.int().tolist())
            objs.append({"mask": obj_mask, "color": c, "shape_exact": shape_hash})
        return objs

    for c_target in range(COLOR_OFFSET, COLOR_OFFSET + N_COLORS):
        for c_legend in range(COLOR_OFFSET, COLOR_OFFSET + N_COLORS):
            if c_target == c_legend: continue
            sup_map = {}
            possible = True
            for m in sup:
                g_in, g_out = gin[m], gout[m]
                if g_in is None or g_out is None or g_in.shape != g_out.shape:
                    possible = False; break
                target_objs = get_objects_of_color(g_in, c_target)
                legend_objs = get_objects_of_color(g_in, c_legend)
                if not target_objs or not legend_objs:
                    possible = False; break
                demo_map = {}
                for t in target_objs:
                    t_mask = t["mask"]
                    H, W = t_mask.shape
                    dilated = torch.zeros_like(t_mask)
                    dilated[1:] |= t_mask[:-1]; dilated[:-1] |= t_mask[1:]
                    dilated[:, 1:] |= t_mask[:, :-1]; dilated[:, :-1] |= t_mask[:, 1:]
                    dilated[1:, 1:] |= t_mask[:-1, :-1]; dilated[:-1, :-1] |= t_mask[1:, 1:]
                    dilated[1:, :-1] |= t_mask[:-1, 1:]; dilated[:-1, 1:] |= t_mask[1:, :-1]
                    dilated |= t_mask
                    touching_legends = [l for l in legend_objs if (dilated & l["mask"]).any()]
                    if len(touching_legends) == 1:
                        leg = touching_legends[0]
                    elif len(legend_objs) == 1:
                        leg = legend_objs[0]
                    else:
                        possible = False; break
                    coords = t["mask"].nonzero(as_tuple=False)
                    out_colors = g_out[coords[:, 0], coords[:, 1]].unique()
                    if len(out_colors) != 1:
                        possible = False; break
                    out_c = int(out_colors[0])
                    key = leg["shape_exact"]
                    if key in demo_map and demo_map[key] != out_c:
                        possible = False; break
                    demo_map[key] = out_c
                if not possible: break
                for k, v in demo_map.items():
                    if k in sup_map and sup_map[k] != v:
                        possible = False; break
                    sup_map[k] = v
            if not possible or not sup_map: continue
            g_in_h, g_out_h = gin[h], gout[h]
            target_objs_h = get_objects_of_color(g_in_h, c_target)
            legend_objs_h = get_objects_of_color(g_in_h, c_legend)
            if not target_objs_h: continue
            pred = g_in_h.clone()
            bg = int(_mode(g_in_h))
            for leg in legend_objs_h:
                pred[leg["mask"]] = bg
            success = True
            for t in target_objs_h:
                t_mask = t["mask"]
                H, W = t_mask.shape
                dilated = torch.zeros_like(t_mask)
                dilated[1:] |= t_mask[:-1]; dilated[:-1] |= t_mask[1:]
                dilated[:, 1:] |= t_mask[:, :-1]; dilated[:, :-1] |= t_mask[:, 1:]
                dilated[1:, 1:] |= t_mask[:-1, :-1]; dilated[:-1, :-1] |= t_mask[1:, 1:]
                dilated[1:, :-1] |= t_mask[:-1, 1:]; dilated[:-1, 1:] |= t_mask[1:, :-1]
                dilated |= t_mask
                touching = [l for l in legend_objs_h if (dilated & l["mask"]).any()]
                if len(touching) == 1:
                    leg = touching[0]
                elif len(legend_objs_h) == 1:
                    leg = legend_objs_h[0]
                else:
                    success = False; break
                key = leg["shape_exact"]
                if key not in sup_map:
                    success = False; break
                pred[t_mask] = sup_map[key]
            if success and bool((pred == g_out_h).all()):
                return True
    return False


def _geo_routed(gin, gout, sup, h, geo_ops):
    """Geometric op + param AGREED by all support demos, applied to held-out input, reconstructs the
    held-out output EXACTLY -> op name, else None. Mirrors geometric_detector_diagnostic's inner loop."""
    if gin[h] is None or gout[h] is None:
        return None
    for op in geo_ops:
        inter, ok = None, True
        for m in sup:
            c = op.candidates(gin[m], gout[m])
            inter = c if inter is None else (inter & c)
            if not inter:
                ok = False
                break
        if ok and inter:
            pred = op.apply(gin[h], sorted(inter)[0])
            if pred is not None and pred.shape == gout[h].shape and bool((pred == gout[h]).all()):
                return op.name
    return None


def _committed_solve(gin, gout, sup, h, geo_ops, labels, rule_lib=None, return_trace=False):
    """A1 (CLOSE THE ROUTE): produce a FLOOR-SAFE 2-attempt committed answer for held-out h's INPUT, selected
    using ONLY the support demos (NO peek at gout[h] -- gout[h] is used only for scoring upstream). attempt2 is
    ALWAYS the floor, so committed-exact (either attempt) is >= floor-exact BY CONSTRUCTION (the 2-attempt ARC
    guarantee, and the R13 net-negative kill). attempt1 is the first SUPPORT-VERIFIED op prediction (object-recolor
    fit by cross-demo agreement on the support OUTPUTS then applied to the held-out INPUT, else a cross-support
    agreed geometric op). Returns (attempt1, attempt2). If return_trace=True, returns
    (attempt1, attempt2, trace) with the internal source of attempt1 for selector diagnostics."""
    def _out(a1, a2, source, recipe=None, floor_source="floor_identity"):
        if not return_trace:
            return a1, a2
        return a1, a2, {
            "attempt1_source": source,
            "attempt1_recipe": list(recipe) if recipe is not None else None,
            "attempt1_is_floor": source == "floor",
            "floor_source": floor_source,
        }

    if gin.get(h) is None:
        return _out(None, None, "invalid")
    same_shape = all(gin.get(m) is not None and gout.get(m) is not None and gin[m].shape == gout[m].shape for m in sup)
    # ---- attempt2 = FLOOR: content recolor map fit on support (else identity copy) ----
    floor = gin[h].clone()
    floor_source = "floor_identity"
    if same_shape:
        rmap = _fit_recolor_2d([(gin[m], gout[m]) for m in sup])
        if rmap is not None:
            floor = _apply_recolor_2d(gin[h], rmap)
            floor_source = "floor_recolor"
    # ---- attempt1 = first SUPPORT-VERIFIED op prediction on gin[h] (non-peeking) ----
    a1 = None
    source = None
    source_recipe = None
    if same_shape and labels.get(h) is not None:
        for key in ("rank", "shape", "colourset"):
            m_map, ok = {}, True
            for m in sup:
                if labels.get(m) is None:
                    ok = False; break
                groups = _objgroups(gin[m], labels[m]); km = _object_keymap(gin[m], groups, key)
                gof = gout[m].reshape(-1)
                for l, cells in groups.items():
                    outs = {int(gof[c]) for c in cells}
                    if len(outs) != 1:
                        ok = False; break
                    oc = outs.pop()
                    if m_map.get(km[l], oc) != oc:
                        ok = False; break
                    m_map[km[l]] = oc
                if not ok:
                    break
            if ok and m_map:
                groups_h = _objgroups(gin[h], labels[h]); km_h = _object_keymap(gin[h], groups_h, key)
                pred = gin[h].clone(); pf = pred.reshape(-1)
                for l, cells in groups_h.items():
                    if km_h[l] in m_map:
                        for c in cells:
                            pf[c] = m_map[km_h[l]]
                a1 = pred; source = f"object_recolor:{key}"; source_recipe = ["object_recolor", key]; break
    # A2: REUSE stored macros (params re-fit) FIRST
    if a1 is None and rule_lib is not None and len(rule_lib) > 0:
        for _recipe in rule_lib.recipes():
            pred = _execute_recipe_predict(gin, gout, sup, h, _recipe, geo_ops)
            if pred is not None:
                a1 = pred; source = "macro:" + ">".join(_recipe); source_recipe = list(_recipe); break
    if a1 is None:                                               # geometric op agreed across support (RELIABLE first)
        for op in geo_ops:
            inter = None
            for m in sup:
                c = op.candidates(gin[m], gout[m])
                inter = c if inter is None else (inter & c)
                if not inter:
                    break
            if inter:
                pred = op.apply(gin[h], sorted(inter)[0])
                if pred is not None:
                    a1 = pred; source = f"geo:{op.name}"; source_recipe = [op.name]; break
    # Stage-2 composition (op1->recolor)
    if a1 is None:
        c2 = _compose2_predict(gin, gout, sup, h, geo_ops)
        if c2 is not None:
            recipe, pred = c2
            a1 = pred
            source = "compose2:" + ">".join(recipe)
            source_recipe = list(recipe)
            # Do not abstract here. _committed_solve is a non-peeking predictor, so it cannot prove the
            # recipe is LODO-exact. RuleLibrary writes belong in the verifier after exact fold checks.
    # SPECULATIVE recolor maps LAST: ordered by generalization reliability so these only fire where the rigid ops
    # don't -> purely ADDITIVE (right => +1, wrong => floor fallback), never DISPLACING a reliable prediction.
    if a1 is None:                                               # copy-by-relation (U11) predict
        for _rel in ("nearest", "container", "contained", "aligned", "between", "largest", "smallest"):
            _p = _object_relrecolor_predict(gin, gout, sup, h, labels, _rel)
            if _p is not None:
                a1 = _p; source = f"object_relrecolor:{_rel}"; source_recipe = ["object_relrecolor", _rel]; break
    if a1 is None:                                               # set-cover (U6) predict
        _p = _set_cover_predict(gin, gout, sup, h, labels)
        if _p is not None:
            a1 = _p; source = "set_cover"; source_recipe = ["set_cover"]
    if a1 is None:
        a1 = floor
        source = "floor"
        source_recipe = None
    return _out(a1, floor, source, source_recipe, floor_source=floor_source)


