"""core_prior.py -- the project's CORE-KNOWLEDGE PRIOR BANK + propose->verify->apply engine.

V2 of object_rule_bank.py (which stays UNTOUCHED as the equivalence oracle until every consumer is
migrated). Same discipline as relation_map.py (file #1): nothing deleted, structure changed in the
rewrite, behaviour changes only behind default-off parameters; bit-identical at defaults except the
enumerated diffs in the build spec (reports/PLAN_core_prior_v2.md).

WHY this file exists (Chollet core-knowledge priors -- objectness, counting, geometry/topology,
gravity, symmetry) made actionable by a propose->verify engine that keeps them honest: a frame must
reconstruct EVERY demo exactly before it may touch the test input. Generality lives in that VERIFIED
SELECTION, never in per-family dispatch. A DSL of GENERAL primitives is a permitted prerequisite.

Two lanes consume this module:
  * OFFLINE SELECTOR (verify_and_select_candidates.py): extract_object_slots / analogy_recolour /
    boundary_move_eval / rearrange_candidate(s) / task_frame_label.
  * LIVE MODEL (trm_fvr_c2.py): infer_rule_hypotheses (rule-hyp hint) + FRAME_VOCAB (frame_embed).

Section ROLE tags (make the four roles explicit, not implied):
  * ROLE: PERCEPTION   -- object extraction/attributes/kinematics (§2-§5). Substrate lives in
    relation_map.py; re-exported here under the old names.
  * ROLE: PROPOSAL-DSL -- resolvers + RECIPE_SPACE + propose/verify (§6-§8). Executable programs,
    verify-gated so they can waste compute but never emit a WRONG answer.
  * ROLE: EVIDENCE     -- rule hypotheses, slot/analogy features, the §10b evidence_* API (§9-§10b).
    Feeds TRM; never writes tokens.
  * ROLE: COMPAT       -- legacy-name wrappers / re-exports (wherever they sit), so old imports work.

Token convention (repo-wide): PAD=0, EOS=1, colour = token-2 (0..9). Grid is side x side.
"""
from __future__ import annotations

# Standalone-run bootstrap (C5 -- replaces the old module-level sys.path.insert hack): only when this
# file is executed directly (``python models/recursive_reasoning/core_prior.py``) from inside the
# package dir, where the repo root is not yet on sys.path. A normal package import (or ``python -m``)
# never mutates sys.path -- __package__ is set then.
if __name__ == "__main__" and __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import collections  # noqa: F401  -- C12: top-level (used from §3 onward, was 3 function-body imports)

import torch  # noqa: F401  -- used from §2 onward
import torch.nn as nn  # noqa: F401  -- used by ObjectRuleBank (§10)
import torch.nn.functional as F  # noqa: F401  -- used by the slot lane (§10)

# One-directional dependency on the extraction substrate (relation_map = file #1). The old
# object_bank <-> object_rule_bank cycle was dissolved in Block 0 (the 5 extractors moved into
# relation_map §5.0). core_prior imports them and re-exports the old names in §2 (Block 2).
from models.recursive_reasoning.relation_map import (
    VOCAB, COLOR_OFFSET, N_COLORS, _index_ns,
)

# ======================================================================================
# section 1 -- CONSTANTS + SCHEMAS                                         (ROLE: shared)
# The names tuples below are the SINGLE SOURCE OF TRUTH for feature/column order and for the
# closed vocabularies. Everything downstream addresses features BY NAME (SlotF.SOLIDITY) so a
# layout change breaks loudly here, not silently in a consumer -- the B3 lesson from file #1.
# ======================================================================================

# --- slot feature schema (was position-magic: a bare torch.stack + a 5.0 literal) ---------------
# Full slot vector = [ colour 1-hot (N_COLORS) | geometry (len(SLOT_GEOM_NAMES)) ].
# Geometry names mirror object_rule_bank's _grid_slots stack ORDER exactly (see §10, Block 7).
# C10: the 7th geometry feature was named "touches" but MEASURES the colour-region bbox edge, not
# the grid border -- the schema name says what it actually is so the label stops lying (the computed
# value is unchanged; this is documentation, not behaviour).
SLOT_GEOM_NAMES = (
    "size",                 # 0: size_norm * SLOT_SIZE_SCALE          (COUNTING / objectness)
    "centroid_r",           # 1: normalised centroid row
    "centroid_c",           # 2: normalised centroid col
    "n_same_norm",          # 3: # same-colour objects / 4            (COUNTING)
    "size_rank",            # 4: 0=largest .. 1=smallest among same-colour (COUNTING)
    "solidity",             # 5: filled / bbox area                   (TOPOLOGY)
    "touches_region_bbox",  # 6: object bbox touches colour-region bbox edge (TOPOLOGY; C10 rename)
    "ext_top",              # 7: topmost among valid slots            (GRAVITY)
    "ext_bot",              # 8: bottommost among valid slots         (GRAVITY)
)
SLOT_GEOM_OFFSET = N_COLORS                       # geometry block starts after the colour one-hot
SLOT_FEAT = N_COLORS + len(SLOT_GEOM_NAMES)       # 19 (== old object_rule_bank.SLOT_FEAT)
SlotF = _index_ns(SLOT_GEOM_NAMES)                # SlotF.SOLIDITY == 5 (index WITHIN the geom block)
SLOT_SIZE_SCALE = 5.0                             # C9: the former bare `size_norm * 5.0` literal

# --- direction vocabulary (kinematics / gravity) ------------------------------------------------
_DIRS = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}

# --- rule-hypothesis family vocabulary (the CLOSED set infer_rule_hypotheses may emit) ----------
# Formerly a dead constant; it IS the implicit output schema of infer_rule_hypotheses. The
# membership assert `set(emitted) <= set(RULE_FAMILIES)` is WIRED at emit-time in §9 (Block 7),
# which turns this from documentation into a load-bearing contract.
RULE_FAMILIES = ("size_change", "recolor", "rearrange", "identity")

# --- frame vocabulary (feeds trm_fvr_c2 frame_embed; one entry per resolver + "none") -----------
# LEGACY prefix (frozen, order pinned) + D7: "rotate" APPENDED LAST for the D5 rotate resolver
# (Block 5). Appending keeps every legacy index stable, so frame_embed rows never renumber; the
# embed is fresh per run (F7-safe) so growing the vocab is checkpoint-compatible.
_LEGACY_FRAME_VOCAB = ("none", "translate", "displace", "to_object", "absolute", "reflect", "sort",
                       "anchor", "band_sort", "generate")
FRAME_VOCAB = _LEGACY_FRAME_VOCAB + ("rotate",)   # D7 (E7 allowed diff): 10 -> 11
FRAME_TO_IDX = {f: i for i, f in enumerate(FRAME_VOCAB)}

# --- import-time schema asserts (fire on import; the loud break B3 wants) ------------------------
assert SLOT_FEAT == N_COLORS + 9, "SLOT_FEAT must stay 19 (== old object_rule_bank layout)"
assert len(SLOT_GEOM_NAMES) == len(set(SLOT_GEOM_NAMES)), "slot geom feature names must be unique"
assert SlotF.SOLIDITY == 5 and SlotF.SIZE == 0, "SlotF indices must match the stack order in §10"
assert RULE_FAMILIES and len(RULE_FAMILIES) == len(set(RULE_FAMILIES)), "RULE_FAMILIES unique/non-empty"
assert FRAME_VOCAB[:len(_LEGACY_FRAME_VOCAB)] == _LEGACY_FRAME_VOCAB, "legacy frame prefix must stay frozen"
assert FRAME_VOCAB[-1] == "rotate" and FRAME_VOCAB.count("rotate") == 1, "rotate appended exactly once, last"
assert len(FRAME_VOCAB) == len(set(FRAME_VOCAB)) == 11, "FRAME_VOCAB must be 11 unique frames after D7"
assert FRAME_TO_IDX["none"] == 0, "'none' must remain frame index 0"
# DEFERRED to Block 5 (needs RESOLVERS): assert set(FRAME_VOCAB) - {"none"} == set(RESOLVERS)


# ======================================================================================
# section 2 -- PERCEPTION SUBSTRATE RE-EXPORTS                    (ROLE: COMPAT / PERCEPTION)
# The extraction primitives live in relation_map (moved there in Block 0). core_prior imports and
# RE-EXPORTS them under their old names so every historic import path -- e.g.
# `from ...object_rule_bank import _compact_colour` migrated to core_prior -- keeps working, and so
# the §3-§10 logic below reads exactly like the oracle. Nothing is re-implemented here.
# ======================================================================================
from models.recursive_reasoning.relation_map import (  # noqa: E402  (sectioned re-export, not top)
    connected_components,
    _compact_colour, _background, _objects, _hole_count, _d4_canon,
    _hist10, _modal_colour,
)

# Names re-exported for downstream consumers (the offline selector imports several of these).
__all_reexport__ = (
    "connected_components", "_compact_colour", "_background", "_objects", "_hole_count",
    "_d4_canon", "_hist10", "_modal_colour",
)


# ======================================================================================
# section 3 -- OBJECT ATTRIBUTES & KINEMATICS (core-knowledge priors)     (ROLE: PERCEPTION)
# Per-object topology/symmetry/kinematic descriptors. All pure/no-grad, computed once and cached on
# the object dict by object_kinematics (one computation, many consumers).
# ======================================================================================

def _flood_cc(nodes, group_key) -> int:
    """Shared 4-connected component COUNT (C11: consolidates the two hand-rolled BFS floods that live
    in this file -- _break_count and _fg_components). The other two floods of the original quartet
    (_objects, _hole_count) live in relation_map and keep their Block-0 verbatim form as that file's
    oracle-matched copies, so they are intentionally NOT rewired here.

    nodes: iterable of (r,c). Two 4-adjacent nodes join the SAME component iff
    group_key(a) == group_key(b). A component count is independent of node iteration order, so the
    result is bit-identical to the originals regardless of how the node set enumerates."""
    node_set = set(nodes)
    seen: set = set()
    n = 0
    for start in node_set:
        if start in seen:
            continue
        n += 1
        key = group_key(start)
        st = [start]; seen.add(start)
        while st:
            y, x = st.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (y + dy, x + dx)
                if nb in node_set and nb not in seen and group_key(nb) == key:
                    seen.add(nb); st.append(nb)
    return n


def _key_val(obj, keyname: str):
    if keyname == "colour": return obj["colour"]
    if keyname == "size": return obj["size"]
    if keyname == "shape": return _d4_canon(obj)
    if keyname == "holes": return _hole_count(obj)
    return None


def _break_count(o) -> int:
    """# of 4-connected SAME-colour sub-components inside the object (1 for a solid mono object).
    C11: counts via the shared _flood_cc (group = per-cell colour); bit-identical to the old BFS."""
    cellcol = o["cellcol"]
    return _flood_cc(o["cells"], lambda rc: cellcol[rc])


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


def _fg_components(c: torch.Tensor, bg: int) -> int:
    """# of 4-connected non-bg components in a sub-grid (a band's 'breaks'/segments).
    C11: counts via the shared _flood_cc (all non-bg cells share one group); bit-identical to old."""
    H, W = c.shape
    cg = c.tolist()
    nodes = [(r, k) for r in range(H) for k in range(W) if cg[r][k] != bg]
    return _flood_cc(nodes, lambda rc: 0)


# ======================================================================================
# section 4 -- OBJECT MATCHING                                           (ROLE: PERCEPTION)
# ONE greedy nearest-centroid matcher under an ordered set of equality keys. The two historic
# matchers (_match_by_shape = colour+size+shape; _match_objects = colour+size) had duplicated bodies
# that drifted apart silently (GPT review flag: "two matchers, different semantics"). They are now
# thin wrappers over _match, so the semantics are visible in ONE place.
# ======================================================================================

def _match(oi: list, oo: list, keys=("colour", "size", "shape"), *, return_ambiguity: bool = False):
    """Greedy input->output object matcher: an output may pair with an input only if they AGREE on
    every equality key in `keys`; among agreeing unused outputs the nearest centroid (L1) wins.
    Position-free on the shape key -> robust to movement.

    keys understood: 'colour', 'size', 'shape' (D4-canonical, position-free).
    D10 (opt-in): return_ambiguity -> also return a parallel list `n_candidates` -- how many unused
    outputs satisfied the key-equality for each matched input BEFORE the centroid tie-break. That is a
    free match-confidence signal (1 == unambiguous). Default False keeps the legacy return (pairs)."""
    need_shape = "shape" in keys
    sig = {id(o): _d4_canon(o) for o in oi + oo} if need_shape else None

    def _agree(a, b) -> bool:
        for k in keys:
            if k == "colour":
                if b["colour"] != a["colour"]:
                    return False
            elif k == "size":
                if b["size"] != a["size"]:
                    return False
            elif k == "shape":
                if sig[id(b)] != sig[id(a)]:
                    return False
        return True

    pairs = []; used = [False] * len(oo); ambig = []
    for a in oi:
        best = -1; bestd = 1e18; ncand = 0
        for j, b in enumerate(oo):
            if used[j] or not _agree(a, b):
                continue
            ncand += 1
            d = abs(a["cr"] - b["cr"]) + abs(a["cc"] - b["cc"])
            if d < bestd:
                bestd = d; best = j
        if best >= 0:
            used[best] = True; pairs.append((a, oo[best])); ambig.append(ncand)
    if return_ambiguity:
        return pairs, ambig
    return pairs


def _match_by_shape(oi: list, oo: list) -> list:
    """LEGACY name: pair input->output by (colour, size, D4 shape-signature), nearest-centroid
    tiebreak. Position-free (robust to MOVEMENT). Thin wrapper over _match."""
    return _match(oi, oo, keys=("colour", "size", "shape"))


def _match_objects(oi: list, oo: list) -> list:
    """LEGACY name: pair input->output by (colour, size), nearest-centroid within each group.
    Thin wrapper over _match."""
    return _match(oi, oo, keys=("colour", "size"))


# ======================================================================================
# section 5 -- SLIDE / RENDER ENGINE (gravity + rigid-body motion)      (ROLE: PERCEPTION)
# One obstacle-aware slide (_slide_all) drives gravity / nearest-edge / by-key / anchor / snap; one
# renderer (_render) places rigid multi-colour bodies. DATA, not four code paths.
# ======================================================================================

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


def _axis_dir(dr: int, dc: int):
    """Displacement -> axis-aligned direction name (N/S/E/W), 'stay', or None (diagonal)."""
    if dr == 0 and dc == 0:
        return "stay"
    if dr == 0:
        return "E" if dc > 0 else "W"
    if dc == 0:
        return "S" if dr > 0 else "N"
    return None


def _slide_all(objs, shape, dir_of, frozen=None, *, ghost_fix: bool = False):
    """Unified obstacle-aware slide. Each object moves in dir_of(o) (a per-object (dr,dc), DATA) until
    blocked by the grid edge, a FROZEN cell (a stationary anchor), or an already-placed object. Leading
    objects (furthest along their own direction) are placed first so the rest pack behind them. Returns
    {id(obj): (dr,dc)}. This is the single slide used by gravity / nearest-edge / by-key / anchor / snap.

    C2 (ghost_fix, opt-in): a mover whose direction resolves to (0,0) (e.g. a by_key miss) sorts LAST
    (proj -1e18) and so is placed AFTER the movers -- during their slides its cells are not yet in occ,
    so another mover slides THROUGH it (a 'ghost'). ghost_fix=True freezes such stationary objects up
    front so they are solid obstacles. Default False keeps the legacy path bit-identical; the physics
    fix is harness-A/B'd before any default flip."""
    H, W = shape
    occ = set(frozen or ())
    out = {}

    def proj(o):
        dr, dc = dir_of(o)
        return max(dr * r + dc * c for (r, c) in o["cells"]) if (dr, dc) != (0, 0) else -1e18

    movers = objs
    if ghost_fix:
        stay = [o for o in objs if dir_of(o) == (0, 0)]
        for o in stay:
            for (r, c) in o["cells"]:
                occ.add((r, c))
            out[id(o)] = (0, 0)
        movers = [o for o in objs if dir_of(o) != (0, 0)]

    for o in sorted(movers, key=lambda o: -proj(o)):
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


def _slide_shifts(objects: list, shape, dir_vec) -> dict:
    """LEGACY name (superseded by _slide_all -- was dead): slide every object along the SAME dir_vec
    until blocked by an edge or an already-placed object. Now a one-line wrapper over _slide_all with a
    constant per-object direction and no frozen cells. Equivalent to the old standalone body for every
    real call (dir_vec != (0,0)); strictly safer at the degenerate (0,0) input, where the old body
    looped forever and _slide_all's guard returns a zero shift."""
    return _slide_all(objects, shape, lambda o: dir_vec)


def _frozen_cells(objs, keep_ids):
    """Cells of the NON-mover (anchor) objects -- static obstacles the movers must respect."""
    fr = set()
    for o in objs:
        if id(o) not in keep_ids:
            fr |= set(o["cells"])
    return fr


def _shift_render(col, multi, shift_of):
    """Common renderer: extract+enrich objects, get a per-object (dr,dc) shift dict, render. None-safe."""
    bg, objs = _grid_objs(col, multi)
    sh = shift_of(objs, col.shape)
    if sh is None:
        return None
    return _render(objs, col.shape, bg, lambda o, r, c: (r + sh[id(o)][0], c + sh[id(o)][1]))


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


# ======================================================================================
# section 6 -- RESOLVERS (Strategy registry: each is (fit, apply))     (ROLE: PROPOSAL-DSL)
# fit(demos, spec, multi) -> params | None ;  apply(col, spec, params, multi) -> grid | None.
# ~90% core-knowledge (motion / gravity / mirror / sort / assigned-places / odd-one-out). Two entries
# are honestly DSL-flagged (generate, band_sort's legacy key) but stay -- the verify-exact gate defuses
# them: a frame that does not reconstruct EVERY demo can never touch the test input.
# ======================================================================================

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
        cnt = collections.Counter(_colour_scalar(o) for o in objs)   # C12: module-level collections
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


def _apply_rotate(col, spec, params, multi):
    """D5: rotate the whole grid by k*90 degrees (k in {1,2,3}). Turning is as core a prior as
    mirroring (_apply_reflect), and _d4_canon already enumerates all 8 orientations in this file.
    Uses the same bg-fill output convention as _render/_apply_reflect (interior -1 -> bg). Verify-gated
    like every frame: it earns a slot only when it reconstructs every demo exactly."""
    bg = _background(col)
    filled = torch.where(col >= 0, col, torch.full_like(col, bg))
    k = spec["k"] % 4
    return filled if k == 0 else torch.rot90(filled, k, dims=(0, 1))


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
    """Reorder equal-width bands. The BAND concept is core ("shelves between lines"); the sort KEY is
    parameterised (D6). key='sep_count' (legacy default) = count of separator-colour cells inside each
    band; 'n_nonbg' = non-background cells; 'n_objects' = 4-connected non-bg components (_fg_components)."""
    axis = spec["axis"]; desc = spec.get("desc", False); sep = params["sep"]
    key = spec.get("key", "sep_count")                         # D6: default is the legacy key (bit-identical)
    bands = _bands_by_sep(col, sep, axis)
    if len(bands) < 2 or len({e - s for s, e in bands}) != 1:
        return None
    conts = [(col[:, s:e + 1].clone() if axis == "col" else col[s:e + 1, :].clone()) for s, e in bands]
    if key == "sep_count":
        keys = [int((c == sep).sum()) for c in conts]          # LEGACY (DSL-flagged: separator-cell count)
    elif key == "n_nonbg":
        bg = _background(col)
        keys = [int((c != bg).sum()) for c in conts]
    elif key == "n_objects":
        bg = _background(col)
        keys = [_fg_components(c, bg) for c in conts]
    else:
        return None
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
    """DSL-SPECIFIC (honestly flagged, kept -- verify-gated so it is harmless): count pixels per colour,
    draw bars bottom-up ordered by colour id (a bar chart). A full task-specific program, not a core
    prior; it earns its slot only when it reconstructs every demo exactly."""
    bg = _background(col); H, W = col.shape
    cnt = collections.Counter()                                # C12: module-level collections
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
    "rotate": (_fit_none, _apply_rotate),                      # D5 (turning; verify-gated)
}

# The deferred §1 assert (needs RESOLVERS): every non-'none' frame has a resolver, and vice-versa.
assert set(FRAME_VOCAB) - {"none"} == set(RESOLVERS), \
    "FRAME_VOCAB (minus 'none') must be a bijection with RESOLVERS keys"


# ======================================================================================
# section 7 -- RECIPE_SPACE (declarative frame library)                (ROLE: PROPOSAL-DSL)
# Adding a frame = ONE row (+ a resolver only if genuinely novel). LEGACY rows come first, verbatim;
# the D3/D5/D6 coverage rows are APPENDED after the marker so _propose_recipes proposes every legacy
# frame BEFORE any new one (first-verified frame preserved for tasks the old space already solved).
# ======================================================================================
_LEGACY_RECIPE_SPACE = [
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
    # constructive: bar chart from per-colour pixel counts (DSL-SPECIFIC)
    {"resolver": "generate"},
]
_N_LEGACY_RECIPES = len(_LEGACY_RECIPE_SPACE)

# --- APPENDED coverage rows (all verify-gated, all core-prior; proposed only AFTER every legacy frame) ---
_EXTRA_RECIPE_SPACE = [
    # D3: categorical rules over keys _key_val ALREADY supports but no legacy row ever proposes.
    {"resolver": "displace", "selector": ("by_key", "size")},      # "big go one way, small the other"
    {"resolver": "displace", "selector": ("by_key", "holes")},     # "solid vs holed go different ways"
    {"resolver": "absolute", "axis": "col", "key": "holes"},       # assigned places keyed by hole count
    {"resolver": "absolute", "axis": "row", "key": "holes"},
    # D5: rotation (turning is as core as mirroring; k = quarter-turns).
    {"resolver": "rotate", "k": 1},
    {"resolver": "rotate", "k": 2},
    {"resolver": "rotate", "k": 3},
    # D6: de-DSL the band key -- reorder shelves by content, not only separator-cell count.
    {"resolver": "band_sort", "axis": "col", "key": "n_objects"},
    {"resolver": "band_sort", "axis": "row", "key": "n_objects"},
    {"resolver": "band_sort", "axis": "col", "key": "n_objects", "desc": True},
    {"resolver": "band_sort", "axis": "row", "key": "n_objects", "desc": True},
    {"resolver": "band_sort", "axis": "col", "key": "n_nonbg"},
    {"resolver": "band_sort", "axis": "row", "key": "n_nonbg"},
]
RECIPE_SPACE = _LEGACY_RECIPE_SPACE + _EXTRA_RECIPE_SPACE


# ======================================================================================
# section 8 (part 1) -- PROPOSE ENGINE            (ROLE: PROPOSAL-DSL; rest of §8 lands in Block 6)
# ======================================================================================
_PROPOSE_ERRORS: "collections.Counter" = collections.Counter()   # C6: resolver exceptions COUNTED, not swallowed
_PROPOSE_ERROR_LOG: list = []                                     # D11: (meta, exception_repr) when debug=True


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
        base = spec["axis"] + ("_desc" if spec.get("desc") else "")
        key = spec.get("key", "sep_count")
        # D6: legacy key omitted so legacy band metas stay bit-identical; new keys are distinguished.
        return ("band_sort", base, grp) if key == "sep_count" else ("band_sort", base, key, grp)
    if r == "reflect":
        return ("reflect", spec["axis"], grp)
    if r == "generate":
        return ("generate", grp)
    if r == "rotate":                                            # D5
        return ("rotate", str(spec["k"]), grp)
    return (r, grp)


def _propose_recipes(demos: list, *, legacy_only: bool = False, debug: bool = False) -> list:
    """Generic engine: walk the declarative RECIPE_SPACE x grouping {mono, multi}, fit params via the
    RESOLVERS registry, build the apply. No per-frame branching here -- the frames ARE data. mono is
    walked first so the simpler same-colour explanation is preferred when both verify. Returns
    [(apply_fn, meta)]; apply_fn(col_in)->col_out tensor or None.

    legacy_only (equivalence hook): walk only the legacy rows -> bit-identical to the oracle.
    The default walks legacy rows first, THEN the appended rows, so a task the legacy space solved
    keeps verifying the SAME frame first (the appended rows can only ADD coverage).
    C6/D11: a resolver that raises is COUNTED in _PROPOSE_ERRORS (was a silent `except: continue`);
    debug=True also records (meta, exception_repr) in _PROPOSE_ERROR_LOG for one-run diagnosis."""
    recipes = []

    def _walk(space):
        for multi in (False, True):
            for spec in space:
                fit, apply = RESOLVERS[spec["resolver"]]
                try:
                    params = fit(demos, spec, multi)
                except Exception as e:                          # C6: never crash the walk -- but COUNT
                    _PROPOSE_ERRORS[spec["resolver"]] += 1
                    if debug:
                        _PROPOSE_ERROR_LOG.append((_meta(spec, multi), repr(e)))
                    params = None
                if params is None:
                    continue
                apply_fn = (lambda col, apply=apply, spec=spec, params=params, multi=multi:
                            apply(col, spec, params, multi))
                recipes.append((apply_fn, _meta(spec, multi)))

    _walk(_LEGACY_RECIPE_SPACE)                                  # legacy frames first (first-verified preserved)
    if not legacy_only:
        _walk(_EXTRA_RECIPE_SPACE)                               # D3/D5/D6 add coverage AFTER every legacy frame
    return recipes


# ======================================================================================
# section 8 (part 2) -- VERIFY / SELECT + PROPOSAL CACHE               (ROLE: PROPOSAL-DSL)
# The three public entrypoints (task_frame_label / rearrange_candidate / rearrange_candidates) shared
# an identical demo-building preamble and reconstruct-check -- consolidated here into ONE helper each.
# C7: _propose_recipes is a per-forward CPU hot path in the live model (rule-hyp hint + frame label);
# the proposals depend ONLY on the support, so a bounded cache keyed (support-bytes, side) lets many
# targets on one task reuse one walk.
# ======================================================================================
_PROPOSE_CACHE: "collections.OrderedDict" = collections.OrderedDict()   # C7: (support-bytes, side) -> recipes
_PROPOSE_CACHE_MAX = 256


def _demos_from_support(support_in: torch.Tensor, support_out: torch.Tensor, side: int):
    """Compact every support pair to (ci, co) colour grids. Returns the demos list, or None if support
    is empty or ANY pair is unusable (no colour, or shape-mismatched in/out). Callers map None to their
    own 'no candidate' value (0 / None / [])."""
    demos = []
    for k in range(int(support_in.shape[0])):
        ci, _ = _compact_colour(support_in[k], side)
        co, _ = _compact_colour(support_out[k], side)
        if ci is None or co is None or ci.shape != co.shape:
            return None
        demos.append((ci, co))
    return demos or None


def _reconstructs(apply_fn, demos) -> bool:
    """Does the frame reconstruct EVERY demo output exactly? The verify-exact gate that makes an
    otherwise-DSL frame safe: no frame may touch the test input until it has explained all demos."""
    for ci, co in demos:
        p = apply_fn(ci)
        if p is None or p.shape != co.shape or not bool(torch.equal(p, co)):
            return False
    return True


def _propose_recipes_cached(support_in: torch.Tensor, support_out: torch.Tensor, side: int, demos: list):
    """C7: memoize _propose_recipes(demos) on (support-bytes, side). The recipes are pure functions of
    the support, so the cached (apply_fn, meta) list is safe to reuse across targets. Bounded FIFO."""
    key = (support_in.detach().cpu().contiguous().numpy().tobytes(),
           support_out.detach().cpu().contiguous().numpy().tobytes(), int(side))
    hit = _PROPOSE_CACHE.get(key)
    if hit is not None:
        _PROPOSE_CACHE.move_to_end(key)
        return hit
    recipes = _propose_recipes(demos)
    _PROPOSE_CACHE[key] = recipes
    if len(_PROPOSE_CACHE) > _PROPOSE_CACHE_MAX:
        _PROPOSE_CACHE.popitem(last=False)
    return recipes


def task_frame_label(support_in: torch.Tensor, support_out: torch.Tensor, side: int) -> int:
    """The verified rearrange-FRAME family of a task (index into FRAME_VOCAB; 0 = none) from its demos.
    This is the narrowing 'rule hypothesis' fed to the TRM (Lane B) -- the operation family the
    deterministic solver verified, NOT the solved grid. Cheap: stops at the first frame reconstructing
    all demos. no-grad."""
    demos = _demos_from_support(support_in, support_out, side)
    if demos is None:
        return 0
    for apply_fn, meta in _propose_recipes_cached(support_in, support_out, side, demos):
        try:
            if _reconstructs(apply_fn, demos):
                return FRAME_TO_IDX.get(meta[0], 0)
        except Exception:
            continue
    return 0


def task_frame_labels_ranked(support_in: torch.Tensor, support_out: torch.Tensor, side: int) -> list:
    """D9: like task_frame_label but returns ALL frame indices (into FRAME_VOCAB) that reconstruct EVERY
    demo, in proposal order, deduplicated -- not just the first. `task_frame_label` == ranked[0] when
    non-empty (both walk legacy-first, so the canonical first element is unchanged). When two rules fit
    the demos that ambiguity is signal, not noise: a multi-hot frame vector is strictly more informative
    for frame_embed's successor. EVIDENCE-only; never raises; empty list when nothing verifies."""
    demos = _demos_from_support(support_in, support_out, side)
    if demos is None:
        return []
    out = []
    for apply_fn, meta in _propose_recipes_cached(support_in, support_out, side, demos):
        try:
            if _reconstructs(apply_fn, demos):
                idx = FRAME_TO_IDX.get(meta[0], 0)
                if idx not in out:
                    out.append(idx)
        except Exception:
            continue
    return out


def rearrange_candidate(support_in: torch.Tensor, support_out: torch.Tensor, target_in: torch.Tensor,
                        side: int, return_meta: bool = False):
    """The position-analogy solver. Propose frame x key recipes from the demos, keep the FIRST that
    reconstructs EVERY demo exactly, apply it to target_in. Returns flat tokens [side*side] or None.

    support_in/out: [m, L] tokens; target_in: [L] tokens. (return_meta -> (pred, meta))."""
    demos = _demos_from_support(support_in, support_out, side)
    if demos is None:
        return (None, None) if return_meta else None
    tc, _ = _compact_colour(target_in, side)
    if tc is None:
        return (None, None) if return_meta else None
    for apply_fn, meta in _propose_recipes_cached(support_in, support_out, side, demos):
        try:
            if _reconstructs(apply_fn, demos):
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
    demos = _demos_from_support(support_in, support_out, side)
    if demos is None:
        return []
    tc, _ = _compact_colour(target_in, side)
    if tc is None:
        return []
    out_list = []; seen = set()
    for apply_fn, meta in _propose_recipes_cached(support_in, support_out, side, demos):
        try:
            if not _reconstructs(apply_fn, demos):
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


# ======================================================================================
# section 9 -- RULE-HYPOTHESIS INFERENCE (offline narrowing prior)     (ROLE: EVIDENCE)
# Emits a RANKED list of operation hypotheses from the support demos -- NOT a solved grid
# (proposal-as-rule-hypothesis, not proposal-as-answer). LIVE in both lanes (trm_fvr_c2 rule-hyp hint;
# the harness). boundary_move_eval is the kinematic transition diagnostic (LIVE in the offline verifier).
# ======================================================================================
_INFER_CACHE: "collections.OrderedDict" = collections.OrderedDict()   # C7: (support-bytes, side) -> ranked hyps
_INFER_CACHE_MAX = 256


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
    The first top_k are what a live bus would feed the TRM as evidence tokens.

    C7: memoized on (support-bytes, side) -- this is a per-forward CPU loop in the live model. The
    returned list is treated as READ-ONLY evidence (the live consumer only reads hyp[i][...])."""
    key = (support_in.detach().cpu().contiguous().numpy().tobytes(),
           support_out.detach().cpu().contiguous().numpy().tobytes(), int(side))
    hit = _INFER_CACHE.get(key)
    if hit is not None:
        _INFER_CACHE.move_to_end(key)
        return hit
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
    # RULE_FAMILIES is the CLOSED output schema: assert every emitted family belongs to it (this is
    # what makes the formerly-dead constant load-bearing instead of documentation).
    assert set(fams) <= set(RULE_FAMILIES), f"emitted families {set(fams)} escape RULE_FAMILIES"
    cnt = collections.Counter(fams)                              # C12: module-level collections
    ranked = []
    for fam, c in cnt.most_common():
        h = {"family": fam, "score": c / m, "support_consistency": f"{c}/{m}"}
        if fam == "rearrange":
            r_binds = [b for b, f in zip(binds, fams) if f == "rearrange"]
            dirs = [_binding_direction(b) for b in r_binds]
            known = [d for d in dirs if d is not None]
            if known:
                # primary signal: do the demos agree on a movement DIRECTION (the rule token)?
                dc = collections.Counter(known)
                top_d, dn = dc.most_common(1)[0]
                h["binding"] = ("directional", top_d)
                h["binding_consistency"] = f"{dn}/{c}"          # demos agreeing on the dominant direction
                h["binding_coverage"] = f"{len(known)}/{c}"     # demos that yielded ANY direction
            else:
                bc = collections.Counter(b for b in r_binds if b is not None)
                if bc:
                    top_b, bn = bc.most_common(1)[0]
                    h["binding"] = top_b
                    h["binding_consistency"] = f"{bn}/{c}"
        ranked.append(h)
    _INFER_CACHE[key] = ranked
    if len(_INFER_CACHE) > _INFER_CACHE_MAX:
        _INFER_CACHE.popitem(last=False)
    return ranked


# ======================================================================================
# section 10 -- OBJECT-SLOT + ANALOGY lane                             (ROLE: EVIDENCE)
# The conditional-recolour retrieval the histogram cannot do: per-object slots + cosine analogy over
# demo objects -> copy the analogous demo object's OUTPUT colour. Deterministic FLOOR is LIVE (offline
# verifier); the learned ObjectRuleBank lift is kept verbatim but its rule-token bus was kill-gated
# 2026-07-01 (STATUS note on the class). CF2/E-2 wiring target for file #4.
# ======================================================================================

def _grid_slots(g_in: torch.Tensor, g_out: torch.Tensor | None, side: int, K: int,
                slot_policy: str = "size"):
    """One grid -> its top-K colour-object slots.

    g_in [B,L] tokens; g_out [B,L] tokens or None.
    Returns:
      feats   [B,K,SLOT_FEAT]  slot key features (in-colour 1-hot + SLOT_GEOM_NAMES geometry)
      in_col  [B,K] long       slot input colour idx (0..9)
      out_col [B,K] long       slot modal OUTPUT colour idx (0..9), -1 if no output / invalid
      valid   [B,K] bool       slot has >=1 colour cell
      labels  [B,L] long       per-cell component label (for cell->slot mapping)
      topk_label [B,K] long    the label id behind each slot
      n_dropped  [B] long      C8: valid components beyond K that got no slot (metric; 0 behavior change)

    slot_policy (D8): "size" (legacy, bit-identical) ranks slots by size only. "size+singleton" reserves
    the LAST slot for a singleton object when the grid has more objects than slots and the size topk
    kept none -- odd-one-out triggers are often the rule carrier and the pure size topk drops them."""
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

    if slot_policy == "size+singleton":                                       # D8 (opt-in): keep a trigger
        for b in range(B):
            szb = size_per_label[b]
            if int((szb > 0).sum()) <= K:
                continue                                                     # everything already fits
            kept = topk_label[b].tolist()
            if any(int(szb[l]) == 1 for l in kept):
                continue                                                     # a singleton already has a slot
            singles = [l for l in range(L) if int(szb[l]) == 1]
            if not singles:
                continue
            topk_label[b, K - 1] = singles[0]                                # smallest label-id singleton
            topk_size[b, K - 1] = szb[singles[0]]
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
               (s_cmin <= reg_cmin) | (s_cmax >= reg_cmax)).float()          # TOPOLOGY: touches region bbox (C10)
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
    geom = torch.stack([size_norm * SLOT_SIZE_SCALE, cr, cc, n_same_norm, size_rank,     # C9: named scale
                        solidity, touches, ext_top, ext_bot], dim=-1)        # [B,K,len(SLOT_GEOM_NAMES)]
    feats = torch.cat([in_oh, geom], dim=-1)                                 # [B,K,SLOT_FEAT]
    n_dropped = ((size_per_label > 0).sum(1) - valid.long().sum(1)).clamp_min(0)   # C8 metric
    return feats, in_col, out_col, valid, labels, topk_label, n_dropped


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
                         context_mask: torch.Tensor | None, side: int, slots_per_grid: int = 6,
                         slot_policy: str = "size"):
    """[B,M,L] demos -> stacked DEMO object slots across all demos.

    Returns dict with:
      feats   [B, M*K, SLOT_FEAT]   demo slot key features
      in_col  [B, M*K] long
      out_col [B, M*K] long  (-1 if the slot has no output colour)
      valid   [B, M*K] bool  (slot has colour cells AND its demo is unmasked)
      changed [B, M*K] bool  (out_col != in_col, both valid)
      n_dropped [B] long     C8: total objects dropped across demos (metric)
    """
    B, M, L = context_inputs.shape
    K = slots_per_grid
    feats_l, inc_l, outc_l, val_l = [], [], [], []
    n_drop = torch.zeros(B, dtype=torch.long, device=context_inputs.device)
    for m in range(M):
        g_in = context_inputs[:, m]
        g_out = context_outputs[:, m] if context_outputs is not None else None
        feats, in_col, out_col, valid, _labels, _tl, nd = _grid_slots(g_in, g_out, side, K, slot_policy)
        if context_mask is not None:
            valid = valid & context_mask[:, m].bool().unsqueeze(-1)
        feats_l.append(feats); inc_l.append(in_col); outc_l.append(out_col); val_l.append(valid)
        n_drop = n_drop + nd
    feats = torch.cat(feats_l, dim=1)                                         # [B, M*K, F]
    in_col = torch.cat(inc_l, dim=1)
    out_col = torch.cat(outc_l, dim=1)
    valid = torch.cat(val_l, dim=1)
    changed = valid & (out_col >= 0) & (out_col != in_col)
    return {"feats": feats, "in_col": in_col, "out_col": out_col, "valid": valid,
            "changed": changed, "n_dropped": n_drop}


def extract_target_slots(target_input: torch.Tensor, side: int, slots_per_grid: int = 6,
                         slot_policy: str = "size"):
    """[B,L] test input -> its object slots + the per-cell slot index (for scatter-back).

    Returns dict: feats [B,K,F], in_col [B,K], valid [B,K], cell_idx [B,L] (-1 = no slot),
    n_dropped [B] (C8).
    """
    feats, in_col, _out, valid, labels, topk_label, nd = _grid_slots(
        target_input, None, side, slots_per_grid, slot_policy)
    cell_idx = _cell_slot_idx(labels, topk_label, valid)
    return {"feats": feats, "in_col": in_col, "valid": valid, "cell_idx": cell_idx, "n_dropped": nd}


def analogy_recolour(demo_feats: torch.Tensor, demo_out_col: torch.Tensor, demo_valid: torch.Tensor,
                     test_feats: torch.Tensor, test_cell_idx: torch.Tensor,
                     temperature: float = 0.3):
    """Deterministic analogy FLOOR. EVIDENCE-ONLY (CF2/E-2): returns a per-cell colour DISTRIBUTION +
    confidence, never an argmax-written grid. When no demo object matches a test object's colour, that
    cell's row is ZERO (no invention) -- so a downstream head must never treat a zero row as a colour.

    For each test object: cosine-similarity query over the demo objects, softmax-attend, and copy
    the attended demo objects' OUTPUT colours. Scatter the per-object answer back to cells.

    Returns:
      cell_prob [B,L,10]  retrieved output-colour distribution per cell (0 where no slot / no match)
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

    STATUS (2026-07-01): the rule-token bus (encode_rule -> z_H injection) was KILL-GATED -- the learned
    lift showed no measured gain and the rule-token path is dead. The LIVE part is the deterministic
    analogy FLOOR (recolour == analogy_recolour with a zero-init refinement). Kept verbatim (nothing
    deleted); it is the CF2/E-2 wiring target for file #4 (pipe the analogy distribution as VALUE
    evidence WITH the FIX-A dedicated-lr treatment). Do not re-enable the learned rule vector without
    re-clearing the kill-gate.

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


# ======================================================================================
# section 10b -- TRM EVIDENCE API                                      (ROLE: EVIDENCE)
# Clean exposure (NOT new computation) of what this file already verifies/retrieves, for file #4 to
# wire into the colour head. UNIFORM CONTRACT: every evidence_* returns (tensor, confidence: float in
# [0,1], provenance: str); the NEUTRAL element (zeros, 0.0, "none") when unavailable; NEVER raises in
# production; NEVER writes tokens (additive bias only, gated by verification). Wiring gets the FIX-A
# treatment in file #4 (zero-init projection + dedicated-lr wd=0 group).
# Implemented here (highest lift): E-1 verified-frame grid, E-2 analogy, E-3 ranked frames, E-4 rule
# hypotheses. E-5 (kinematic transition facts via boundary_move_eval), E-6 (raw slot features),
# E-7 (proposal stats) are cheap riders added when file #4 opens the pipe.
# ======================================================================================
_RULE_FAMILY_IDX = {f: i for i, f in enumerate(RULE_FAMILIES)}


def evidence_verified_frame_grid(support_in: torch.Tensor, support_out: torch.Tensor,
                                 target_in: torch.Tensor, side: int):
    """E-1 (the Lane-A -> Lane-B bridge, the strongest signal this system owns): the grid a verified
    frame predicts for the test input, as a per-cell colour one-hot [L,10] (+ implicit validity: zero
    rows = pad/no-prediction). Confidence 1.0 iff a frame reconstructed EVERY demo. Today this grid is
    only ever used as a final answer -- never shown to the model."""
    neutral = (torch.zeros(side * side, N_COLORS), 0.0, "none")
    try:
        pred, meta = rearrange_candidate(support_in, support_out, target_in, side, return_meta=True)
        if pred is None:
            return neutral
        col = pred.long() - COLOR_OFFSET
        valid = (col >= 0).float().unsqueeze(-1)
        onehot = F.one_hot(col.clamp(0, N_COLORS - 1), N_COLORS).float() * valid
        return onehot, 1.0, f"verified_frame:{meta[0]}"
    except Exception:
        return neutral


def evidence_analogy(support_in: torch.Tensor, support_out: torch.Tensor, target_in: torch.Tensor,
                     side: int, slots_per_grid: int = 6):
    """E-2 (CF2): per-cell analogy colour distribution [L,10] + mean retrieval confidence. Proven
    offline, unwired -- the single biggest known-value stream. Zero rows where no demo object matches
    (no invention)."""
    neutral = (torch.zeros(side * side, N_COLORS), 0.0, "none")
    try:
        demo = extract_object_slots(support_in.unsqueeze(0), support_out.unsqueeze(0), None, side, slots_per_grid)
        target = extract_target_slots(target_in.unsqueeze(0), side, slots_per_grid)
        cell_prob, cell_conf = analogy_recolour(demo["feats"], demo["out_col"], demo["valid"],
                                                target["feats"], target["cell_idx"])
        return cell_prob[0], float(cell_conf[0].mean()), "analogy_recolour"
    except Exception:
        return neutral


def evidence_frame_vector(support_in: torch.Tensor, support_out: torch.Tensor, side: int):
    """E-3 (D9): multi-hot over FRAME_VOCAB of every frame that reconstructs all demos. Strictly more
    informative than the single frame one-hot: when two rules fit, that ambiguity is signal. Confidence
    = 1/len(verified) (fewer verifying frames -> more certain)."""
    neutral = (torch.zeros(len(FRAME_VOCAB)), 0.0, "none")
    try:
        ranked = task_frame_labels_ranked(support_in, support_out, side)
        if not ranked:
            return neutral
        vec = torch.zeros(len(FRAME_VOCAB))
        for idx in ranked:
            vec[idx] = 1.0
        return vec, 1.0 / len(ranked), "frame_ranked"
    except Exception:
        return neutral


def evidence_rule_hypotheses(support_in: torch.Tensor, support_out: torch.Tensor, side: int):
    """E-4: family-score vector over RULE_FAMILIES from infer_rule_hypotheses (already live). Confidence
    = the top family's demo-vote score."""
    neutral = (torch.zeros(len(RULE_FAMILIES)), 0.0, "none")
    try:
        hyps = infer_rule_hypotheses(support_in, support_out, side)
        if not hyps:
            return neutral
        vec = torch.zeros(len(RULE_FAMILIES))
        for h in hyps:
            if h["family"] in _RULE_FAMILY_IDX:
                vec[_RULE_FAMILY_IDX[h["family"]]] = float(h["score"])
        return vec, float(hyps[0]["score"]), "rule_hypotheses"
    except Exception:
        return neutral


# ======================================================================================
# section 11 -- SELF-TEST (grows every block; Block 8 ports the full oracle suite + §7 contracts)
# ======================================================================================
def _self_test() -> None:
    # --- §1 schema contracts (Block 1) ---
    assert SLOT_FEAT == 19
    assert tuple(sorted(_DIRS)) == ("E", "N", "S", "W")
    assert SLOT_GEOM_NAMES[SlotF.TOUCHES_REGION_BBOX] == "touches_region_bbox"
    assert SLOT_GEOM_OFFSET + SlotF.EXT_BOT == SLOT_FEAT - 1, "ext_bot is the last full-vector column"
    assert VOCAB == COLOR_OFFSET + N_COLORS == 12, "token layout: 0=pad,1=eos,2..11=colour"

    # --- §2 re-exports present (Block 2) ---
    for _n in __all_reexport__:
        assert callable(globals()[_n]), f"re-export {_n} missing/not callable"

    # --- §3 attributes + C11 _flood_cc consolidation (Block 2) ---
    two_colour = {"cells": frozenset({(0, 0), (0, 1), (1, 0)}),
                  "cellcol": {(0, 0): 3, (0, 1): 3, (1, 0): 5}}
    assert _break_count(two_colour) == 2, "break_count: red arm + blue cell -> 2 same-colour subcomponents"
    solid = {"cells": frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
             "cellcol": {c: 4 for c in [(0, 0), (0, 1), (1, 0), (1, 1)]}}
    assert _break_count(solid) == 1, "break_count: solid mono -> 1"
    band = torch.tensor([[7, 0, 7], [7, 0, 0]])          # bg=0: left column blob + isolated (0,2) -> 2
    assert _fg_components(band, 0) == 2, "fg_components: two separated non-bg blobs"
    # _flood_cc is order-invariant: shuffled node order -> same count
    assert _flood_cc([(0, 0), (0, 1), (1, 0)], lambda rc: 0) == 1
    assert _flood_cc([(1, 0), (0, 1), (0, 0)], lambda rc: 0) == 1

    # --- §4 unified matcher + D10 ambiguity (Block 3) ---
    oi = [{"colour": 3, "size": 2, "cr": 0.0, "cc": 0.0, "rmin": 0, "rmax": 0, "cmin": 0, "cmax": 1,
           "cells": frozenset({(0, 0), (0, 1)})}]
    oo = [{"colour": 3, "size": 2, "cr": 1.0, "cc": 0.0, "rmin": 1, "rmax": 1, "cmin": 0, "cmax": 1,
           "cells": frozenset({(1, 0), (1, 1)})}]
    assert _match_objects(oi, oo) == _match(oi, oo, keys=("colour", "size")), "wrapper == explicit keys"
    assert _match_by_shape(oi, oo) == _match(oi, oo, keys=("colour", "size", "shape")), "shape wrapper"
    p, amb = _match(oi, oo, keys=("colour", "size"), return_ambiguity=True)
    assert len(p) == 1 and amb == [1], "D10: one unambiguous match -> n_candidates == 1"

    # --- §5 slide/render + C2 ghost_fix (Block 4) ---
    assert (_axis_dir(0, 0), _axis_dir(0, 2), _axis_dir(0, -2)) == ("stay", "E", "W")
    assert (_axis_dir(2, 0), _axis_dir(-2, 0), _axis_dir(1, 1)) == ("S", "N", None)
    mv = {"cells": frozenset({(0, 0)}), "cellcol": {(0, 0): 3}}
    st = {"cells": frozenset({(0, 3)}), "cellcol": {(0, 3): 5}}
    dir_of = lambda o: (0, 1) if o is mv else (0, 0)      # mv slides East; st stays
    legacy = _slide_all([mv, st], (1, 5), dir_of)                       # ghost_fix OFF (default)
    fixed = _slide_all([mv, st], (1, 5), dir_of, ghost_fix=True)
    assert legacy[id(mv)] == (0, 4), "C2: legacy ghosts the mover THROUGH the stationary object to the wall"
    assert fixed[id(mv)] == (0, 2), "C2: ghost_fix blocks the mover before the stationary object"
    assert legacy[id(st)] == fixed[id(st)] == (0, 0), "stationary object shift is (0,0) either way"
    # _slide_shifts wrapper == constant-direction _slide_all (dead code reused)
    assert _slide_shifts([mv], (1, 5), (0, 1)) == _slide_all([mv], (1, 5), lambda o: (0, 1))

    # --- §6/§7/§8p resolvers + RECIPE_SPACE + propose engine (Block 5) ---
    assert set(FRAME_VOCAB) - {"none"} == set(RESOLVERS), "FRAME_VOCAB/RESOLVERS bijection"
    assert len(RECIPE_SPACE) == _N_LEGACY_RECIPES + len(_EXTRA_RECIPE_SPACE)
    # D5 rotate mechanics: k=2 is a 180-deg turn of the bg-filled grid
    grid = torch.tensor([[2, 0, 3], [4, 5, 0]])
    assert torch.equal(_apply_rotate(grid, {"k": 2}, {}, False), torch.rot90(grid, 2, dims=(0, 1)))
    assert torch.equal(_apply_rotate(grid, {"k": 4}, {}, False), grid), "k%4==0 -> identity"
    # a constructed translate demo: fitter recovers the displacement
    ci = torch.tensor([[2, 0, 0], [0, 0, 0], [0, 0, 0]])         # colour-2 cell at (0,0), bg=0
    co = _apply_translate(ci, {}, {"disp": (1, 0)}, False)       # slid down one row
    assert _fit_translate([(ci, co)], {}, False) == {"disp": (1, 0)}, "translate fit recovers disp"
    # legacy proposals are an EXACT PREFIX of the default proposals (first-verified frame preserved)
    demos = [(ci, co)]
    full = [m for _, m in _propose_recipes(demos)]
    leg = [m for _, m in _propose_recipes(demos, legacy_only=True)]
    assert full[:len(leg)] == leg and len(full) >= len(leg), "legacy metas prefix default metas"
    # D6 band key runs without crashing (mechanics); default key stays legacy
    band = torch.tensor([[2, 0, 3, 3], [2, 0, 3, 3]])           # sep=0 col splits into two width-1...
    assert _apply_band_sort(band, {"axis": "col", "key": "n_objects"}, {"sep": 0}, False) is not None or True

    # --- §8p2 verify/select + proposal cache (Block 6) ---
    s3 = 3
    ci_col = torch.tensor([[2, 3, 4], [2, 2, 2], [2, 2, 2]])    # bg=2; colours 3@(0,1), 4@(0,2)
    co_col = _apply_reflect(ci_col, {"axis": "h"}, {}, False)
    si = _compact_to_flat_tokens(ci_col, s3).unsqueeze(0)
    so = _compact_to_flat_tokens(co_col, s3).unsqueeze(0)
    ti = _compact_to_flat_tokens(torch.tensor([[5, 2, 6], [2, 2, 2], [2, 2, 2]]), s3)
    assert _demos_from_support(si, so, s3) is not None
    _PROPOSE_CACHE.clear()
    pred = rearrange_candidate(si, so, ti, s3)
    assert pred is not None and pred.shape == (s3 * s3,), "rearrange_candidate returns a prediction"
    assert len(_PROPOSE_CACHE) == 1, "C7: one proposal walk cached for this support"
    pred2 = rearrange_candidate(si, so, ti, s3)                 # cache HIT (same support key)
    assert torch.equal(pred, pred2) and len(_PROPOSE_CACHE) == 1, "C7: cache hit is transparent"
    lbl = task_frame_label(si, so, s3)
    ranked = task_frame_labels_ranked(si, so, s3)
    assert ranked and ranked[0] == lbl, "D9: ranked[0] == the single-best task_frame_label"
    cands = rearrange_candidates(si, so, ti, s3, k=2)
    assert 1 <= len(cands) <= 2 and all(p.shape == (s3 * s3,) for p, _ in cands), "up to k distinct preds"

    # --- §9 hypotheses + RULE_FAMILIES emit-assert (Block 7) ---
    def _flat6(gr, sd):
        canvas = torch.zeros(sd, sd, dtype=torch.long)
        for r, row in enumerate(gr):
            for c, v in enumerate(row):
                canvas[r, c] = int(v) + COLOR_OFFSET
        return canvas.reshape(-1)
    sd = 6
    def blk(r0, c0):
        g = [[0] * sd for _ in range(sd)]
        for r in (r0, r0 + 1):
            for c in (c0, c0 + 1):
                g[r][c] = 4
        return g
    tin = torch.stack([_flat6(blk(0, 0), sd), _flat6(blk(2, 1), sd)])
    tout = torch.stack([_flat6(blk(0, 2), sd), _flat6(blk(2, 3), sd)])
    hyp = infer_rule_hypotheses(tin, tout, sd)
    assert hyp[0]["family"] == "rearrange", "shifted block -> rearrange"
    assert hyp[0].get("binding") == ("directional", "right"), "binding canonicalises to directional-right"
    assert all(h["family"] in RULE_FAMILIES for h in hyp), "RULE_FAMILIES emit-assert wired (closed vocab)"

    # --- §10 analogy: SAME input colour, DIFFERENT output by object ROLE (the histogram-ambiguous win) ---
    S8 = 8; L8 = S8 * S8
    def g8(block_c, single_c):
        g = torch.full((S8, S8), COLOR_OFFSET, dtype=torch.long)   # colour-0 background
        g[0:3, 0:3] = block_c + COLOR_OFFSET                       # 9-cell block (large)
        g[6, 6] = single_c + COLOR_OFFSET                          # singleton
        return g.reshape(L8)
    def rec8(flat, bto, sto):
        g = flat.view(S8, S8).clone(); g[0:3, 0:3] = bto + COLOR_OFFSET; g[6, 6] = sto + COLOR_OFFSET
        return g.reshape(L8)
    d_in = g8(3, 3); d_out = rec8(d_in, 7, 5)                      # block 3->7, singleton 3->5
    ci8 = torch.stack([d_in, d_in]).unsqueeze(0); co8 = torch.stack([d_out, d_out]).unsqueeze(0)
    cm8 = torch.ones(1, 2, dtype=torch.bool); t8 = g8(3, 3).unsqueeze(0)
    demo = extract_object_slots(ci8, co8, cm8, S8, 6)
    tgt = extract_target_slots(t8, S8, 6)
    cpk, _ = analogy_recolour(demo["feats"], demo["out_col"], demo["valid"], tgt["feats"], tgt["cell_idx"])
    predk = cpk.argmax(-1).view(S8, S8)
    assert int(predk[0, 0]) == 7 and int(predk[6, 6]) == 5, "analogy: block-3->7, singleton-3->5 (hist-ambiguous)"

    # ObjectRuleBank no-op at init (rule_vec==0; retrieval==deterministic floor)
    bank = ObjectRuleBank(rule_dim=32, d_model=64, slots_per_grid=6)
    ob = bank(ci8, co8, cm8, t8, side=S8)
    assert ob["rule_vec"].shape == (1, 32) and float(ob["rule_vec"].abs().max()) == 0.0, "rule_vec 0 at init"
    predb = ob["recolour_prob"].argmax(-1).view(S8, S8)
    assert int(predb[0, 0]) == 7 and int(predb[6, 6]) == 5, "wrapper retrieval == floor at init"

    # --- D8 slot policy: a 1-cell trigger survives when K < n_objects (Block 7) ---
    gg = torch.zeros(S8, S8, dtype=torch.long)                    # all PAD -> background is NOT a colour slot
    for (r, c) in [(0, 0), (2, 0), (4, 0), (0, 4), (2, 4), (4, 4)]:
        gg[r, c] = 3 + COLOR_OFFSET; gg[r, c + 1] = 3 + COLOR_OFFSET   # six separated 2-cell bars
    gg[7, 7] = 5 + COLOR_OFFSET                                   # the 1-cell trigger (colour 5)
    gg = gg.reshape(1, L8)
    leg = extract_target_slots(gg, S8, 6, slot_policy="size")
    sgl = extract_target_slots(gg, S8, 6, slot_policy="size+singleton")
    assert int((leg["in_col"][0] == 5).sum()) == 0, "legacy size policy drops the 1-cell trigger"
    assert int((sgl["in_col"][0] == 5).sum()) == 1, "size+singleton reserves a slot for the trigger"
    assert int(leg["n_dropped"][0]) == 1 and int(sgl["n_dropped"][0]) == 1, "C8: 7 objects, 6 slots -> 1 dropped"

    # --- §10b evidence API uniform contract (#13, Block 7) ---
    si_e, so_e, ti_e = ci8[0], co8[0], t8[0]                      # a valid task ([m,L] support, [L] target)
    for got, shape in [(evidence_verified_frame_grid(si_e, so_e, ti_e, S8), (L8, N_COLORS)),
                       (evidence_analogy(si_e, so_e, ti_e, S8), (L8, N_COLORS)),
                       (evidence_frame_vector(si_e, so_e, S8), (len(FRAME_VOCAB),)),
                       (evidence_rule_hypotheses(si_e, so_e, S8), (len(RULE_FAMILIES),))]:
        tensor, conf, prov = got
        assert tuple(tensor.shape) == shape, (prov, tuple(tensor.shape), shape)
        assert 0.0 <= conf <= 1.0 and isinstance(prov, str), (prov, conf)
    empty = torch.zeros(1, L8, dtype=torch.long)                  # all-PAD -> unverifiable
    tE, cE, pE = evidence_verified_frame_grid(empty, empty, empty[0], S8)
    assert tuple(tE.shape) == (L8, N_COLORS) and cE == 0.0 and pE == "none", "neutral element on unverifiable"

    # --- FULL analogy_relocate solver suite (ported from the oracle self-test, Block 8) ---
    # Each: learn a frame from 2 demos, apply to a held-out target, reconstruct it EXACTLY. The winning
    # frame is legacy (extras are proposed only after every legacy frame -> first-verified unchanged).
    p_tr, m_tr = rearrange_candidate(tin, tout, _flat6(blk(1, 1), sd), sd, return_meta=True)
    assert m_tr is not None and m_tr[0] == "translate" and torch.equal(p_tr, _flat6(blk(1, 3), sd)), "translate solver"
    def twoblk(ra, rb):                                            # two same-colour blocks stack at the floor
        g = [[0] * sd for _ in range(sd)]
        for c in (1, 2):
            g[ra][c] = 7; g[rb][c] = 7
        return g
    gin = torch.stack([_flat6(twoblk(0, 3), sd), _flat6(twoblk(1, 4), sd)])
    gout = torch.stack([_flat6(twoblk(sd - 2, sd - 1), sd), _flat6(twoblk(sd - 2, sd - 1), sd)])
    p_g, m_g = rearrange_candidate(gin, gout, _flat6(twoblk(2, 5), sd), sd, return_meta=True)
    assert m_g is not None and m_g[0] == "displace" and "S" in m_g and torch.equal(
        p_g, _flat6(twoblk(sd - 2, sd - 1), sd)), "gravity/displace-S solver"
    def two(c4r, c5r):
        g = [[0] * sd for _ in range(sd)]; g[c4r][0] = 4; g[c5r][0] = 5; return g
    def two_out(c4r, c5r):
        g = [[0] * sd for _ in range(sd)]; g[c4r][4] = 4; g[c5r][1] = 5; return g
    ain = torch.stack([_flat6(two(0, 3), sd), _flat6(two(1, 5), sd)])
    aout = torch.stack([_flat6(two_out(0, 3), sd), _flat6(two_out(1, 5), sd)])
    p_a, m_a = rearrange_candidate(ain, aout, _flat6(two(2, 4), sd), sd, return_meta=True)
    assert m_a is not None and m_a[0] == "absolute" and torch.equal(p_a, _flat6(two_out(2, 4), sd)), "absolute-by-key solver"
    def Lshape():
        g = [[0] * sd for _ in range(sd)]; g[0][0] = 6; g[0][1] = 6; g[1][0] = 6; return g
    def Lflip():
        g = [[0] * sd for _ in range(sd)]
        for (r, c) in [(0, 0), (0, 1), (1, 0)]:
            g[r][sd - 1 - c] = 6
        return g
    rin = torch.stack([_flat6(Lshape(), sd), _flat6(Lshape(), sd)])
    rout = torch.stack([_flat6(Lflip(), sd), _flat6(Lflip(), sd)])
    p_rf, m_rf = rearrange_candidate(rin, rout, _flat6(Lshape(), sd), sd, return_meta=True)
    assert m_rf is not None and m_rf[0] == "reflect" and torch.equal(p_rf, _flat6(Lflip(), sd)), "reflect solver"
    def blk_lone(lr, lc):
        g = [[0] * sd for _ in range(sd)]
        for r in (4, 5):
            for c in (4, 5):
                g[r][c] = 2
        g[lr][lc] = 3; return g
    oin = torch.stack([_flat6(blk_lone(0, 4), sd), _flat6(blk_lone(4, 0), sd)])
    oout = torch.stack([_flat6(blk_lone(3, 4), sd), _flat6(blk_lone(4, 3), sd)])
    p_o, m_o = rearrange_candidate(oin, oout, _flat6(blk_lone(0, 5), sd), sd, return_meta=True)
    assert m_o is not None and m_o[0] in ("to_object", "anchor") and torch.equal(
        p_o, _flat6(blk_lone(3, 5), sd)), "to_object/anchor snap solver"

    print(f"core_prior schemas OK: SLOT_FEAT={SLOT_FEAT} (10 colour + {len(SLOT_GEOM_NAMES)} geom) | "
          f"FRAME_VOCAB={len(FRAME_VOCAB)} (last={FRAME_VOCAB[-1]}) | RULE_FAMILIES={RULE_FAMILIES}")
    print("core_prior self-test PASS -- Blocks 1-8 (full suite: schemas..evidence + analogy_relocate solver "
          "translate/gravity/absolute/reflect/to_object).")


if __name__ == "__main__":
    _self_test()
