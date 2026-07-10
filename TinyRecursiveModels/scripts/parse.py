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
def _neighbour_key(grid, side):
    """[N,L] tokens -> [N,L] long: most common 4-neighbour COLOUR (0..9) that differs from the cell's
    own colour; N_COLORS(=10) means 'no real differing neighbour'. Tests adjacency-driven recolours."""
    N, L = grid.shape
    S = side
    g = grid.view(N, S, S)
    real = g >= COLOR_OFFSET
    col = (g - COLOR_OFFSET).clamp(0, 9)
    counts = torch.zeros(N, S, S, N_COLORS)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nreal = torch.roll(real, shifts=(dr, dc), dims=(1, 2))
        ncol = torch.roll(col, shifts=(dr, dc), dims=(1, 2))
        if dr == -1: nreal[:, -1, :] = False                 # mask wrapped (toroidal) edges
        if dr == 1:  nreal[:, 0, :] = False
        if dc == -1: nreal[:, :, -1] = False
        if dc == 1:  nreal[:, :, 0] = False
        valid = nreal & real & (ncol != col)
        counts.scatter_add_(3, ncol.unsqueeze(-1), valid.unsqueeze(-1).float())
    nb = counts.argmax(-1)
    key = torch.where(counts.sum(-1) > 0, nb, torch.full_like(nb, N_COLORS))
    return key.view(N, L)


def _position_key(grid, side, bands=3):
    """[N,L] -> [N,L] long in 0..bands^2-1: which (row-band, col-band) cell the position sits in."""
    N, L = grid.shape
    idx = torch.arange(L)
    rb = (idx // side * bands // side).clamp(0, bands - 1)
    cb = (idx % side * bands // side).clamp(0, bands - 1)
    return (rb * bands + cb).view(1, L).expand(N, L).contiguous()


# Each key_fn: [N,L] input tokens, side -> [N,L] long CONTEXT id (combined with input colour below).
KEYS = {
    "cell":      lambda g, S: torch.zeros_like(g),               # baseline: input colour only (= dpcc)
    "size":      lambda g, S: size_bucket(g, S),                 # the FAILED Phase 2c key (sanity)
    "singleton": lambda g, S: is_singleton_object(g, S).long(),  # scattered pixel vs solid shape
    "position":  lambda g, S: _position_key(g, S),               # where in the grid (3x3 bands)
    "neighbour": lambda g, S: _neighbour_key(g, S),              # adjacency-driven recolour
}


# --- object-level keys for the CONDITIONAL-recolour bucket (the ~26% the cheap keys missed) ---
# These use connected components, so they are heavier; run them on the conditional SUBSET, not globally.
def _components_2d(g2d):
    """[N,S,S] tokens -> [N,S,S] component labels (min flat-index per 4-connected same-token region)."""
    from models.recursive_reasoning.object_bank import connected_components
    return connected_components(g2d)


def _shape_key(grid, side):
    """[N,L] -> [N,L] long: a TRANSLATION-INVARIANT hash of each cell's component SHAPE (the sorted
    cell offsets within its bbox). Same shape -> same id across demos. Tests recolour-by-shape
    ('squares -> red, lines -> blue'). Non-colour cells = 0."""
    N, L = grid.shape
    S = side
    g = grid.view(N, S, S)
    labels = _components_2d(g)
    out = torch.zeros(N, L, dtype=torch.long)
    for n in range(N):
        lab, gg = labels[n].view(-1), g[n].view(-1)
        comp = {}
        for i in torch.nonzero(gg >= COLOR_OFFSET).flatten().tolist():
            comp.setdefault(int(lab[i]), []).append(i)
        row = out[n]
        for cells in comp.values():
            rs = [c // S for c in cells]; cs = [c % S for c in cells]
            r0, c0 = min(rs), min(cs)
            sig = hash(tuple(sorted((r - r0, c - c0) for r, c in zip(rs, cs)))) & 0x7fffffff
            for c in cells:
                row[c] = sig
    return out


def _shape_class_key(grid, side):
    """[N,L] -> [N,L] long: an ABSTRACT, scale + rotation/reflection-INVARIANT shape CLASS of each
    cell's component -- 0 singleton, 1 line (1xk any orientation), 2 solid rect/square (bbox full),
    3 hollow box (perimeter only), 4 blob/other. Unlike _shape_key (exact pixels), a 2x2 and a 4x4
    square BOTH map to class 2, so the class RECURS across a task's differently-sized demos. Tests
    recolour-by-shape-category, the level ARC shape rules actually use."""
    N, L = grid.shape
    S = side
    g = grid.view(N, S, S)
    labels = _components_2d(g)
    out = torch.zeros(N, L, dtype=torch.long)
    for n in range(N):
        lab, gg = labels[n].view(-1), g[n].view(-1)
        comp = {}
        for i in torch.nonzero(gg >= COLOR_OFFSET).flatten().tolist():
            comp.setdefault(int(lab[i]), []).append(i)
        row = out[n]
        for cells in comp.values():
            rs = [c // S for c in cells]; cs = [c % S for c in cells]
            h, w, nc = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1, len(cells)
            if nc == 1:
                cls = 0
            elif min(h, w) == 1:
                cls = 1                                       # straight bar (any orientation)
            elif nc == h * w:
                cls = 2                                       # solid rectangle / square (any size)
            elif h >= 3 and w >= 3 and nc == 2 * (h + w) - 4:
                cls = 3                                       # hollow box (perimeter only)
            else:
                cls = 4                                       # blob / other
            for c in cells:
                row[c] = cls
    return out


def _rank_key(grid, side):
    """[N,L] -> [N,L] long: SIZE-RANK of each cell's component among NON-background components
    (0=largest foreground object, 1=2nd, ... capped 5; background/none=6). Tests recolour-by-ranking
    ('the largest object -> X'). Background = the most common colour token in the grid."""
    N, L = grid.shape
    S = side
    g = grid.view(N, S, S)
    labels = _components_2d(g)
    out = torch.full((N, L), 6, dtype=torch.long)
    for n in range(N):
        lab, gg = labels[n].view(-1), g[n].view(-1)
        idxs = torch.nonzero(gg >= COLOR_OFFSET).flatten().tolist()
        if not idxs:
            continue
        comp, ccol = {}, {}
        for i in idxs:
            l = int(lab[i]); comp.setdefault(l, []).append(i); ccol[l] = int(gg[i])
        bg = Counter(int(gg[i]) for i in idxs).most_common(1)[0][0]
        fg = sorted(((l, len(cs)) for l, cs in comp.items() if ccol[l] != bg), key=lambda x: -x[1])
        rankmap = {l: min(r, 5) for r, (l, _) in enumerate(fg)}
        row = out[n]
        for l, cells in comp.items():
            rk = rankmap.get(l, 6)
            for c in cells:
                row[c] = rk
    return out


def _touch_key(grid, side):
    """[N,L] -> [N,L] long: the dominant colour (0..9) of OTHER-colour components adjacent to this
    cell's component (object-level adjacency; 10 = isolated). Tests RELATIONAL recolour ('the object
    touching red -> X'). Distinct from the per-cell neighbour key (this aggregates over the object)."""
    N, L = grid.shape
    S = side
    g = grid.view(N, S, S)
    labels = _components_2d(g)
    out = torch.full((N, L), 10, dtype=torch.long)
    for n in range(N):
        lab, gg = labels[n].view(-1), g[n].view(-1)
        adj = {}
        for i in torch.nonzero(gg >= COLOR_OFFSET).flatten().tolist():
            r, c, li = i // S, i % S, int(lab[i])
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < S and 0 <= cc < S:
                    j = rr * S + cc
                    if int(gg[j]) >= COLOR_OFFSET and int(lab[j]) != li:
                        adj.setdefault(li, Counter())[int(gg[j]) - COLOR_OFFSET] += 1
        row = out[n]
        for i in torch.nonzero(gg >= COLOR_OFFSET).flatten().tolist():
            li = int(lab[i])
            if li in adj:
                row[i] = adj[li].most_common(1)[0][0]
    return out


KEYS_COND = dict(KEYS, shape=_shape_key, shape_cls=_shape_class_key, rank=_rank_key, touch=_touch_key)


def _combined_key(grid, side):
    """[N,L] -> [N,L] long: mixed-radix FUSION of (neighbour, shape_cls, position) per cell -- the MULTI-KEY
    context. neighbour 0..10, shape_cls 0..4, position 0..8 (bands=3) -> id 0..494. Tests whether FUSING the
    strong single keys (R21: each ~0% grid-exact alone) disambiguates conditional where each one cannot."""
    nb = _neighbour_key(grid, side)                       # 0..N_COLORS (=10)
    sc = _shape_class_key(grid, side)                     # 0..4
    ps = _position_key(grid, side)                        # 0..8
    return nb * 45 + sc * 9 + ps


def _count_key(grid, side):
    """[N,L] -> [N,L] long: COUNTING -- number of 4-connected components sharing this cell's colour
    (capped 6). Distinct from _rank_key (size-rank of the cell's OWN object) and shape_cls (its shape):
    tests recolour-by-object-count ('the colour that appears as 3 objects -> X'). Core-knowledge: counting."""
    from collections import Counter
    N, L = grid.shape
    S = side
    g = grid.view(N, S, S)
    labels = _components_2d(g)
    out = torch.zeros(N, L, dtype=torch.long)
    for n in range(N):
        lab, gg = labels[n].view(-1), g[n].view(-1)
        idxs = torch.nonzero(gg >= COLOR_OFFSET).flatten().tolist()
        col_of_comp = {int(lab[i]): int(gg[i]) for i in idxs}          # one colour per component
        ncomp = Counter(col_of_comp.values())                          # colour-token -> #components
        row = out[n]
        for i in idxs:
            row[i] = min(ncomp[int(gg[i])], 6)
    return out


def _topology_key(grid, side):
    """[N,L] -> [N,L] long: TOPOLOGY -- 0 if the cell's 4-connected same-colour region reaches the grid
    boundary (touches a PAD cell or the canvas edge = OUTSIDE), 1 if fully enclosed by other colours (a
    hole / inside region). Tests fill-enclosed + inside/outside recolour. Core-knowledge: topology."""
    N, L = grid.shape
    S = side
    g = grid.view(N, S, S)
    labels = _components_2d(g)
    out = torch.zeros(N, L, dtype=torch.long)
    for n in range(N):
        gg2d, lab, gg = g[n], labels[n].view(-1), g[n].view(-1)
        idxs = torch.nonzero(gg >= COLOR_OFFSET).flatten().tolist()
        outside = set()
        for i in idxs:
            r, c = i // S, i % S
            touch = (r == 0 or r == S - 1 or c == 0 or c == S - 1)
            if not touch:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if int(gg2d[r + dr, c + dc]) < COLOR_OFFSET:       # PAD neighbour = boundary contact
                        touch = True
                        break
            if touch:
                outside.add(int(lab[i]))
        row = out[n]
        for i in idxs:
            row[i] = 0 if int(lab[i]) in outside else 1
    return out


def _extract_grid(tok, side):
    """[L] canvas tokens -> cropped [h,w] TOKEN grid (the colour-bbox), or None if no colour cell.
    Normalises the augmentation offset so geometric transforms are comparable."""
    g = tok.view(side, side)
    mask = g >= COLOR_OFFSET
    if not mask.any():
        return None
    rows = torch.where(mask.any(1))[0]
    cols = torch.where(mask.any(0))[0]
    return g[rows.min():rows.max() + 1, cols.min():cols.max() + 1].clone()


def _mode(g):
    v, c = torch.unique(g, return_counts=True)
    return int(v[c.argmax()])


def _d4(G):
    """The 8 dihedral transforms of a 2D grid (square or not)."""
    return {
        "id": G, "rot90": torch.rot90(G, 1, (0, 1)), "rot180": torch.rot90(G, 2, (0, 1)),
        "rot270": torch.rot90(G, 3, (0, 1)), "flipH": torch.flip(G, (1,)),
        "flipV": torch.flip(G, (0,)), "transpose": G.transpose(0, 1).contiguous(),
        "antitranspose": torch.flip(G.transpose(0, 1).contiguous(), (0, 1)),
    }


def _shift(g, dr, dc, bg):
    """Translate by (dr,dc) with background fill (NO wrap)."""
    H, W = g.shape
    out = torch.full((H, W), bg, dtype=g.dtype)
    si0, si1, di0, di1 = max(0, -dr), min(H, H - dr), max(0, dr), min(H, H + dr)
    sj0, sj1, dj0, dj1 = max(0, -dc), min(W, W - dc), max(0, dc), min(W, W + dc)
    if si1 > si0 and sj1 > sj0:
        out[di0:di1, dj0:dj1] = g[si0:si1, sj0:sj1]
    return out


def _d4_canon_hash(offs):
    """D4-canonical shape hash (debate R31-C): min hash over the 8 dihedral transforms of a zero-centred
    point cloud -> rotation/reflection-INVARIANT object-shape key. `offs` = list of (r,c) ints."""
    def tf(p, k):
        r, c = p
        return [(r, c), (-r, c), (r, -c), (-r, -c), (c, r), (-c, r), (c, -r), (-c, -r)][k]
    best = None
    for k in range(8):
        t = [tf(p, k) for p in offs]
        r0 = min(p[0] for p in t); c0 = min(p[1] for p in t)
        h = hash(tuple(sorted((p[0] - r0, p[1] - c0) for p in t))) & 0x7fffffff
        best = h if best is None else min(best, h)
    return best


def _enclosed_bg(gi):
    """[H,W] -> bool mask of BACKGROUND cells NOT 4-connected to the border (the enclosed regions / holes).
    Flood from the border through bg cells; bg cells unreached are enclosed. bg = most-common token."""
    H, W = gi.shape
    bg = int(_mode(gi))
    reached = torch.zeros(H, W, dtype=torch.bool)
    stack = []
    for r in range(H):
        for c in (0, W - 1):
            if int(gi[r, c]) == bg and not bool(reached[r, c]):
                reached[r, c] = True; stack.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if int(gi[r, c]) == bg and not bool(reached[r, c]):
                reached[r, c] = True; stack.append((r, c))
    while stack:
        rr, cc = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W and int(gi[nr, nc]) == bg and not bool(reached[nr, nc]):
                reached[nr, nc] = True; stack.append((nr, nc))
    return (gi == bg) & (~reached)


def _components_2d_adj(g2d, bg=None):
    """STAGE-1 #1 FIX: 4-connected components over NON-background cells IGNORING colour (a checkerboard /
    striped square stays ONE object, where same-colour CC shatters it). bg defaults to the most-common token.
    Returns [H,W] long labels; bg cells = -1; each non-bg adjacency-connected region a distinct id. Iterative
    stack (no recursion -- grids are <=30x30)."""
    H, W = g2d.shape
    bgv = int(_mode(g2d)) if bg is None else int(bg)
    lab = torch.full((H, W), -1, dtype=torch.long)
    nxt = 0
    for r in range(H):
        for c in range(W):
            if int(g2d[r, c]) == bgv or int(lab[r, c]) != -1:
                continue
            stack = [(r, c)]; lab[r, c] = nxt
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and int(lab[nr, nc]) == -1 and int(g2d[nr, nc]) != bgv:
                        lab[nr, nc] = nxt; stack.append((nr, nc))
            nxt += 1
    return lab


def _parse_same(g2d):
    """Same-colour 4-connected components as [H,W] labels (bg=-1). Own BFS: works on ANY [H,W] content grid
    (object_bank.connected_components assumes a square 30x30 canvas and breaks on cropped non-square grids)."""
    H, W = g2d.shape
    bg = int(_mode(g2d))
    lab = torch.full((H, W), -1, dtype=torch.long)
    nxt = 0
    for r in range(H):
        for c in range(W):
            if int(g2d[r, c]) == bg or int(lab[r, c]) != -1:
                continue
            col = int(g2d[r, c]); stack = [(r, c)]; lab[r, c] = nxt
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and int(lab[nr, nc]) == -1 and int(g2d[nr, nc]) == col:
                        lab[nr, nc] = nxt; stack.append((nr, nc))
            nxt += 1
    return lab


def _object_keymap(g2d, groups, key):
    """Key each object (`groups` = label -> flat-cell-indices) under `key`. Module-level so `_set_cover_solve` and
    `_object_recolor_solve` share ONE key vocabulary (KEEP THE TWO IN SYNC). Keys: rank / count_same / inside /
    between / aligned / nearest_colour / shape / shape_d4 / position / touch_border / colour / colourset."""
    H, W = g2d.shape
    if key == "rank":
        ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        return {l: r for r, (l, _) in enumerate(ordered)}
    gf = g2d.reshape(-1)
    if key == "count_same":
        from collections import Counter as _Counter
        col = {l: int(gf[cells[0]]) for l, cells in groups.items()}
        cnt = _Counter(col.values())
        return {l: min(cnt[col[l]], 4) for l in groups}
    if key in ("inside", "between", "aligned", "nearest_colour"):
        geom = {}
        for l, cells in groups.items():
            rs = [c // W for c in cells]; cs = [c % W for c in cells]
            geom[l] = (min(rs), max(rs), min(cs), max(cs), sum(rs) / len(rs), sum(cs) / len(cs), int(gf[cells[0]]))
        labels = list(groups); km = {}
        for l in labels:
            r0, r1, c0, c1, mr, mc, _ = geom[l]
            others = [o for o in labels if o != l]
            if key == "inside":
                ins = any(geom[o][0] < r0 and geom[o][1] > r1 and geom[o][2] < c0 and geom[o][3] > c1 for o in others)
                con = any(geom[o][0] > r0 and geom[o][1] < r1 and geom[o][2] > c0 and geom[o][3] < c1 for o in others)
                km[l] = 2 if ins else (1 if con else 0)
            elif key == "aligned":
                row = any(abs(geom[o][4] - mr) < 0.5 for o in others)
                col = any(abs(geom[o][5] - mc) < 0.5 for o in others)
                km[l] = (1 if row else 0) + (2 if col else 0)
            elif key == "between":
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
            else:
                if others:
                    nn = min(others, key=lambda o: abs(geom[o][4] - mr) + abs(geom[o][5] - mc))
                    km[l] = geom[nn][6]
                else:
                    km[l] = -1
        return km
    km = {}
    for l, cells in groups.items():
        rs = [c // W for c in cells]; cs = [c % W for c in cells]
        if key == "shape":
            r0, c0 = min(rs), min(cs)
            km[l] = hash(tuple(sorted((r - r0, c - c0) for r, c in zip(rs, cs)))) & 0x7fffffff
        elif key == "shape_d4":
            r0, c0 = min(rs), min(cs)
            km[l] = _d4_canon_hash([(r - r0, c - c0) for r, c in zip(rs, cs)])
        elif key == "position":
            mr = sum(rs) / len(rs); mc = sum(cs) / len(cs)
            rb = min(int(mr * 3 // max(H, 1)), 2); cb = min(int(mc * 3 // max(W, 1)), 2)
            km[l] = rb * 3 + cb
        elif key == "touch_border":
            km[l] = int(any(r == 0 or r == H - 1 or c == 0 or c == W - 1 for r, c in zip(rs, cs)))
        elif key == "colour":
            km[l] = int(gf[cells[0]])
        else:  # colourset
            km[l] = tuple(sorted(int(gf[c]) for c in cells))
    return km


def _objgroups(g2d, lab):
    """label -> list of flat foreground-cell indices (skip background + unlabeled). Shared object grouping."""
    bg = int(_mode(g2d)); gf, lf = g2d.reshape(-1), lab.reshape(-1); groups = {}
    for i in range(gf.numel()):
        if int(gf[i]) == bg:
            continue
        l = int(lf[i])
        if l < 0:
            continue
        groups.setdefault(l, []).append(i)
    return groups


def _grid_bg(tok):
    real = tok[tok >= COLOR_OFFSET] - COLOR_OFFSET
    if real.numel() == 0:
        return -1
    v, c = torch.unique(real, return_counts=True)
    return int(v[c.argmax()])


