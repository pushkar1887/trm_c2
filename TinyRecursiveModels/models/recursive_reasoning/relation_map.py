"""Per-cell relational substrate for BOTH lanes (V2 of object_bank.py -- see reports/PLAN_relation_map_v2.md).

Consumers:
  MODEL lane   -- trm_fvr_c2: relational_maps (13-ch input/output evidence), relational_where_hint
                  (WHERE evidence), cell_conditioning_signature (conditioned-VALUE context key).
  OFFLINE lane -- verify_and_select_candidates (signature + primitives), probe tools
                  (parse.py / solve.py / oracle_eval.py: is_singleton_object, size_bucket).

Token convention: PAD=0, EOS=1, colour = token-2. Grid is side x side. All functions are
deterministic and run under no_grad -- this module never owns parameters.

Layering (each layer depends only on the ones above it):
  section 1  constants + channel SCHEMAS (RelCh / SigCol / RelP -- the ONLY place layout is defined)
  section 2  layer-0 grid primitives (connected components, distance transform, shared helpers)
  section 3  layer-1 per-grid feature maps (relational_maps + the legacy component quartet)
  section 4  layer-2 WHERE predicates (predicate bank + relational_where_hint)
  section 5  layer-3 conditioning signature (cell_conditioning_signature + helpers)
  section 6  cross-demo consensus (conditioned_transitions engine)
  section 7  self-test

Objects are 4-connected throughout (matches object_rule_bank._objects); 8-connectivity is the
opt-in ``diag=True`` on connected_components. The 5 object extractors (_compact_colour, _background,
_objects, _hole_count, _d4_canon) + their 2 helpers (_hist10, _modal_colour) LIVE HERE now, in
section 5.0 (moved verbatim from object_rule_bank in the file-#2 pass, Block 0). relation_map is the
extraction substrate; core_prior.py (file #2) imports them one-directionally and re-exports the old
names. The old object_bank <-> object_rule_bank cycle is therefore dissolved: this module imports
nothing from object_rule_bank.

Equivalence contract: bit-identical to object_bank.py at default arguments EXCEPT the six
enumerated diffs D1-D6 in the build spec (CC long dtype; signature cols 13-15 appended; signature
entry clamp; opt-in kwargs all defaulting to legacy; early-exit perf; diag_boundary predicate).
"""
from __future__ import annotations

import math
from collections import deque
from types import SimpleNamespace

import torch
import torch.nn.functional as F

# ======================================================================================
# section 1 -- CONSTANTS + CHANNEL SCHEMAS
# The names tuples below are the SINGLE SOURCE OF TRUTH for channel/column order.
# RelCh / SigCol / RelP are auto-derived from them; consumers address channels by NAME
# (rm[..., RelCh.SOLIDITY]) so a layout change breaks loudly here, not silently in a
# predicate (bug B3 / GPT-#7: hard-coded magic indices survived one layout drift already --
# the old 7/8/9 indices silently read clearance channels after the 10->13 widening).
# ======================================================================================

VOCAB = 12
COLOR_OFFSET = 2
N_COLORS = 10          # ARC palette size; VOCAB = COLOR_OFFSET + N_COLORS (used by the §5.0 extractors)
OBJ_DIM = 4            # legacy component_features/object_features channel count (section 3)
N_SIZE_BUCKETS = 3     # legacy size_bucket key count; default n_keys of conditioned_transitions

# Per-cell relational-map channel count (the output dim of relational_maps). The single collapsed
# distance_to_edge channel was split into 4 DIRECTIONAL clearances (top/bottom/left/right within the
# valid bbox, NOT the 30x30 pad -- the B4 bug) -- the per-cell substrate for "slide to wall". Consumers
# reference this constant so the zero-init projections resize together (F7-safe: the new channels are
# zero at init like the rest, so step-0 is byte-identical and the loss must earn the weight).
REL_MAP_CHANNELS = 13
REL_MAP_RELATION_NAMES = (
    "valid_mask", "is_background", "comp_size_log", "is_largest", "is_singleton", "on_boundary",
    "dist_to_top", "dist_to_bottom", "dist_to_left", "dist_to_right",   # 4-way directional clearance
    "solidity", "inside_container", "distance_to_nearest_colour",
)

# Relation predicates for the WHERE hint (section 4). Order defines the predicate bank layout:
# P = 10 colour masks + len(REL_WHERE_RELATION_NAMES) relations + 10*len(...) conjunctions.
# "diag_boundary" is APPENDED LAST (D6): in-grid DIAGONAL neighbour of a different colour -- the
# 45-degree structure the 4-neighbour on_boundary cannot express. Nothing persists weights keyed
# on P (verified in the plan pass); only the logged rel_where_predicate_index renumbers.
REL_WHERE_RELATION_NAMES = (
    "valid",
    "is_background",
    "is_largest",
    "is_singleton",
    "on_boundary",
    "edge_dist_low",
    "edge_dist_high",
    "solid",
    "not_solid",
    "inside_container",
    "nearest_low",
    "nearest_high",
    "comp_small",
    "comp_medium",
    "comp_large",
    "diag_boundary",      # D6 (appended last so legacy relation indices 0..14 are unchanged)
)

# Per-cell CONDITIONING KEY for conditioned value P(out | src, signature). This is a SEPARATE tensor
# from relational_maps -- it does NOT change REL_MAP_CHANNELS or any saved projection, so it carries
# ZERO checkpoint risk (unlike widening the 13-ch relmap). Column 0 is the SOURCE colour (the value
# table's primary key); columns 1..9 are the CONTEXT signature that disambiguates a multi-target
# source (e.g. colour 3 -> red on the largest object, -> green on the singleton). The measured need:
# 52/75 conditional_recolor tasks are multi-target, dominated by background(0) fill, so the signature
# must (a) be universal (neighbour colours work on background cells that own no object) and (b) for a
# background cell, inherit the ENCLOSING object's rank/shape/holes -- the "inside which shape" key.
#
# ADDITIVE-ONLY RULE: columns 0-12 are frozen in place (FIX-C _algo_where_maps and the rich-ctx hash
# address cols 11/12 by index). New columns append at the END. Cols 13-15 are the V2 additions (D2):
#   13 nearest_seed_tie    -- >=2 different seed colours at the SAME minimal distance (col 12 keeps
#                             the deterministic smaller-colour winner; the flag makes the colour-
#                             permutation asymmetry VISIBLE instead of silent -- bug B6 / GPT-#9)
#   14 touch_colour_mode   -- modal non-bg colour 4-adjacent to this cell's object (bg cells inherit
#                             their enclosing object's value, same attribution rule as cols 5-10);
#                             mode tie -> smaller colour (same convention as seeds). Bug B13: the
#                             signature knew "what encloses me"(11) and "what is nearest"(12) but not
#                             "what TOUCHES my object" -- the key for touch-triggered recolour tasks.
#   15 touch_colour_count  -- distinct touching colours, clipped bucket 0,1,2,3(=3+)
CELL_SIG_NAMES = (
    "self_color", "nbr4_a", "nbr4_b", "nbr4_c", "nbr4_d",
    "obj_size_rank", "obj_holes", "obj_shape_d4", "local_row3", "local_col3",
    "obj_color",   # colour of the own/enclosing object -- the fill-majority key (bg cell's container colour)
    "encl_color_ff",       # TRUE flood-fill enclosure colour (bg component not reaching border -> modal boundary colour; else 10)
    "nearest_seed_color",  # colour of the nearest non-bg cell by Manhattan distance (own colour for fg; 10 if no seeds)
    "nearest_seed_tie",    # V2: 1 if >=2 seed colours tie at the minimal distance (col 12 = smaller colour)
    "touch_colour_mode",   # V2: modal non-bg colour 4-adjacent to this cell's object; 10 = none
    "touch_colour_count",  # V2: distinct touching colours bucket 0/1/2/3+; 4 = no object attribution
)
#                -1 = null self colour; 10 = null colour; ranks/holes/shape/thirds use one-past-max.
CELL_SIG_NONE = (-1, 10, 10, 10, 10, 8, 4, 7, 3, 3, 10, 10, 10, 0, 10, 4)   # per-column null/out-of-grid sentinel
CELL_SIG_DIM = len(CELL_SIG_NAMES)


def _index_ns(names: tuple) -> SimpleNamespace:
    """names tuple -> namespace of UPPERCASED name -> index. The tuples stay the single source of
    truth; these are just readable handles (RelCh.SOLIDITY == 10)."""
    ns = SimpleNamespace()
    for i, n in enumerate(names):
        setattr(ns, n.upper(), i)
    return ns


RelCh = _index_ns(REL_MAP_RELATION_NAMES)      # relational_maps channels
SigCol = _index_ns(CELL_SIG_NAMES)             # cell_conditioning_signature columns
RelP = _index_ns(REL_WHERE_RELATION_NAMES)     # WHERE relation order inside the predicate bank


def hierarchical_context_keys(
    signature: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build five collision-free, increasingly specific task-local VALUE keys.

    The old live path compressed a rich tuple with ``% 512``. This mixed unrelated contexts. These
    mixed-radix keys are injective over the declared signature domains; invalid cells receive -1 at
    every level. Callers can aggregate only keys observed in the current task with ``torch.unique``.
    """
    if signature.shape[-1] < CELL_SIG_DIM:
        raise ValueError(
            f"signature has {signature.shape[-1]} columns; expected at least {CELL_SIG_DIM}")
    s = signature.long()
    if valid is None:
        valid = s[..., SigCol.SELF_COLOR] >= 0
    valid = valid.to(device=s.device, dtype=torch.bool)

    def bounded(column: int, high: int) -> torch.Tensor:
        return s[..., column].clamp(0, high)

    src = bounded(SigCol.SELF_COLOR, 9)
    object_colour = bounded(SigCol.OBJ_COLOR, 10)
    # Background is a role, not colour 0. Foreground cells are attributed to an object of their
    # own colour; outside or enclosed background cells have a sentinel/enclosing object colour.
    background_role = (object_colour != src).long()
    enclosure = bounded(SigCol.ENCL_COLOR_FF, 10)
    touch_colour = bounded(SigCol.TOUCH_COLOUR_MODE, 10)
    touch_count = bounded(SigCol.TOUCH_COLOUR_COUNT, 4)
    nearest_seed = bounded(SigCol.NEAREST_SEED_COLOR, 10)
    rank = bounded(SigCol.OBJ_SIZE_RANK, 8)
    holes = bounded(SigCol.OBJ_HOLES, 4)
    local_row = bounded(SigCol.LOCAL_ROW3, 3)
    local_col = bounded(SigCol.LOCAL_COL3, 3)

    k0 = src
    k1 = k0 * 2 + background_role
    k2 = ((k1 * 11 + enclosure) * 11 + touch_colour) * 5 + touch_count
    k3 = k2 * 11 + nearest_seed
    k4 = (((k3 * 9 + rank) * 5 + holes) * 4 + local_row) * 4 + local_col
    keys = torch.stack((k0, k1, k2, k3, k4), dim=-1)
    return torch.where(valid.unsqueeze(-1), keys, torch.full_like(keys, -1))


def hierarchical_value_binding(
    support_signature: torch.Tensor,
    support_dst_colour: torch.Tensor,
    support_valid: torch.Tensor,
    support_changed: torch.Tensor,
    target_signature: torch.Tensor,
    target_valid: torch.Tensor,
    tau: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Task-local collision-free VALUE binding with hierarchical Dirichlet backoff.

    Outcomes are represented jointly as ten changed destination colours plus one COPY event. At each
    key level, observed counts update the previous distribution; absent keys retain the previous
    distribution exactly. This produces the canonical mixture in one pass rather than asking several
    independent classifiers to rediscover copy-vs-change multiplication.

    P1 (outcome-specific backoff): changed and copy SUPPORT are selected independently -- each uses
    the deepest level whose own outcome count is positive, so a deeper copy-only context can no
    longer erase changed support observed at an ancestor (and vice versa). The joint Dirichlet
    posterior itself is unchanged. ``marginal_distribution`` is the posterior snapshotted after K0
    (source colour only), before any K1..K4 context conditioning.
    """
    if tau <= 0:
        raise ValueError(f"hierarchical_value_binding requires tau > 0, got {tau}")
    if support_signature.ndim != 3 or target_signature.ndim != 2:
        raise ValueError("support_signature must be [M,L,C] and target_signature [L,C]")
    if support_signature.shape[-1] < CELL_SIG_DIM or target_signature.shape[-1] < CELL_SIG_DIM:
        raise ValueError("conditioning signatures do not match the declared schema")
    device = target_signature.device
    s_valid = support_valid.to(device=device, dtype=torch.bool).reshape(-1)
    s_changed = support_changed.to(device=device, dtype=torch.bool).reshape(-1)
    s_dst = support_dst_colour.to(device=device, dtype=torch.long).reshape(-1).clamp(0, 9)
    t_valid = target_valid.to(device=device, dtype=torch.bool).reshape(-1)
    s_keys = hierarchical_context_keys(
        support_signature.to(device).reshape(-1, CELL_SIG_DIM), s_valid)
    t_keys = hierarchical_context_keys(target_signature.to(device), t_valid)

    # outcome 0..9 = changed destination colour; outcome 10 = copy
    outcome = torch.where(s_changed, s_dst, torch.full_like(s_dst, 10))
    global_counts = torch.zeros(11, device=device, dtype=torch.float32)
    if bool(s_valid.any()):
        global_counts.scatter_add_(0, outcome[s_valid], torch.ones_like(outcome[s_valid], dtype=torch.float32))
        global_prob = global_counts / global_counts.sum().clamp_min(1.0)
    else:
        global_prob = torch.zeros(11, device=device, dtype=torch.float32)
        global_prob[10] = 1.0
    prob = global_prob.view(1, 11).expand(target_signature.shape[0], -1).clone()
    marginal_prob = prob.clone()   # overwritten by the post-K0 snapshot below when K0 hits
    support_count = torch.zeros(target_signature.shape[0], device=device, dtype=torch.float32)
    changed_support_count = torch.zeros_like(support_count)
    copy_support_count = torch.zeros_like(support_count)
    level_used = torch.full(
        (target_signature.shape[0],), -1, device=device, dtype=torch.long)
    changed_level_used = torch.full_like(level_used, -1)
    copy_level_used = torch.full_like(level_used, -1)

    for level in range(5):
        sk = s_keys[:, level]
        tk = t_keys[:, level]
        observed = s_valid & (sk >= 0)
        if not bool(observed.any()):
            continue
        unique, inverse = torch.unique(sk[observed], sorted=True, return_inverse=True)
        counts = torch.zeros((unique.numel(), 11), device=device, dtype=torch.float32)
        counts.index_put_(
            (inverse, outcome[observed]),
            torch.ones_like(outcome[observed], dtype=torch.float32),
            accumulate=True,
        )
        idx = torch.searchsorted(unique, tk.clamp_min(0))
        safe_idx = idx.clamp_max(max(unique.numel() - 1, 0))
        hit = t_valid & (tk >= 0) & (idx < unique.numel()) & (unique[safe_idx] == tk)
        gathered = counts[safe_idx]
        n = gathered.sum(dim=-1, keepdim=True)
        updated = (gathered + float(tau) * prob) / (n + float(tau))
        prob = torch.where(hit.unsqueeze(-1), updated, prob)
        if level == 0:
            marginal_prob = prob.clone()   # "after K0, before K1..K4"
        support_count = torch.where(hit, n.squeeze(-1), support_count)
        level_used = torch.where(hit, torch.full_like(level_used, level), level_used)
        # OUTCOME-SPECIFIC selection (P1): a level only claims changed/copy support when ITS OWN
        # outcome count is positive -- deeper copy-only keys keep the ancestor's changed support.
        lvl_changed_n = gathered[:, :10].sum(dim=-1)
        lvl_copy_n = gathered[:, 10]
        chg_hit = hit & (lvl_changed_n > 0)
        cpy_hit = hit & (lvl_copy_n > 0)
        changed_support_count = torch.where(chg_hit, lvl_changed_n, changed_support_count)
        changed_level_used = torch.where(chg_hit, torch.full_like(changed_level_used, level), changed_level_used)
        copy_support_count = torch.where(cpy_hit, lvl_copy_n, copy_support_count)
        copy_level_used = torch.where(cpy_hit, torch.full_like(copy_level_used, level), copy_level_used)

    target_src = target_signature[:, SigCol.SELF_COLOR].long().clamp(0, 9)
    copy_hot = F.one_hot(target_src, num_classes=10).to(torch.float32)
    bind = (prob[:, :10] + copy_hot * prob[:, 10:11]) * t_valid.unsqueeze(-1).float()
    marginal = (marginal_prob[:, :10] + copy_hot * marginal_prob[:, 10:11]) * t_valid.unsqueeze(-1).float()
    support_count = support_count * t_valid.float()
    changed_support_count = changed_support_count * t_valid.float()
    copy_support_count = copy_support_count * t_valid.float()
    return {
        "distribution": bind,
        "marginal_distribution": marginal,
        "copy_probability": prob[:, 10] * t_valid.float(),
        "change_probability": prob[:, :10].sum(dim=-1) * t_valid.float(),
        "support_count": support_count,
        "support_reliability": support_count / (support_count + float(tau)),
        "changed_support_count": changed_support_count,
        "copy_support_count": copy_support_count,
        "changed_supported": changed_support_count > 0,
        "copy_supported": copy_support_count > 0,
        "level_used": level_used,
        "changed_level_used": changed_level_used,
        "copy_level_used": copy_level_used,
        "collision_count": torch.zeros((), device=device, dtype=torch.long),
    }

# --- WHERE predicate thresholds (section 4). Formerly magic numbers inside the mask builder. ------
EDGE_NEAR = 0.16       # "near the bbox edge": <= ~1/6 of bbox extent (bbox-normalised clearance)
EDGE_FAR = 0.33        # "away from the edge": >= ~1/3 of bbox extent
NEAREST_NEAR = 0.10    # "close to another colour": <= 10% of grid side (grid-normalised distance)
NEAREST_FAR = 0.25     # "far from other colours": >= 25% of grid side
COMP_SMALL_MAX = 4.0   # component size buckets over INTEGER sizes -- small: 1..4 cells
COMP_MED_MAX = 12.0    # medium: 5..12 cells; large: >= 13 (no gap: sizes are integers)
SOLID_MIN = 0.99       # solidity ~1.0 = component fills its bbox (rectangles); below = concave/sparse

# --- import-time layout guards: a names/constant drift fails HERE, at import, loudly. -------------
assert len(REL_MAP_RELATION_NAMES) == REL_MAP_CHANNELS, (
    f"REL_MAP_RELATION_NAMES has {len(REL_MAP_RELATION_NAMES)} entries but REL_MAP_CHANNELS="
    f"{REL_MAP_CHANNELS}; the names tuple is the single source of truth -- fix the drift here."
)
assert len(CELL_SIG_NAMES) == CELL_SIG_DIM == len(CELL_SIG_NONE), (
    f"CELL_SIG layout drift: names={len(CELL_SIG_NAMES)} dim={CELL_SIG_DIM} none={len(CELL_SIG_NONE)}"
)
# cols 0-12 are FROZEN (FIX-C and the rich-ctx hash address 11/12 by index); appends only.
assert SigCol.ENCL_COLOR_FF == 11 and SigCol.NEAREST_SEED_COLOR == 12, (
    "signature cols 11/12 moved -- FIX-C _algo_where_maps and the rich-ctx hash read them by index; "
    "new columns must APPEND, never reorder."
)
assert REL_WHERE_RELATION_NAMES[-1] == "diag_boundary" and RelP.ON_BOUNDARY == 4, (
    "relation order drifted -- diag_boundary must stay LAST (D6) and legacy indices 0..14 unchanged."
)


# ======================================================================================
# section 2 -- LAYER-0 GRID PRIMITIVES (pure, no internal deps)
# ======================================================================================


def _edge_masked_same(g: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """same-colour neighbour mask via roll, with the WRAPPED edge zeroed (grid is not toroidal)."""
    same = torch.roll(g, shift, dims=dim) == g
    if dim == 1:
        idx = 0 if shift == 1 else g.shape[1] - 1                  # the row that wrapped
        same[:, idx, :] = False
    else:
        idx = 0 if shift == 1 else g.shape[2] - 1
        same[:, :, idx] = False
    return same


def _in_grid_diff(g: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """neighbour is IN-GRID and a DIFFERENT colour (for the object-boundary feature). Out-of-grid
    is NOT counted as different -- otherwise every grid-edge cell would look like a boundary."""
    diff = torch.roll(g, shift, dims=dim) != g
    if dim == 1:
        idx = 0 if shift == 1 else g.shape[1] - 1
        diff[:, idx, :] = False
    else:
        idx = 0 if shift == 1 else g.shape[2] - 1
        diff[:, :, idx] = False
    return diff


def _edge_masked_same_diag(g: torch.Tensor, sh_r: int, sh_c: int) -> torch.Tensor:
    """DIAGONAL sibling of _edge_masked_same: same-colour diagonal neighbour, BOTH wrapped edges
    zeroed. Substrate for connected_components(diag=True) -- 8-connectivity opt-in."""
    same = torch.roll(g, shifts=(sh_r, sh_c), dims=(1, 2)) == g
    same[:, 0 if sh_r == 1 else g.shape[1] - 1, :] = False
    same[:, :, 0 if sh_c == 1 else g.shape[2] - 1] = False
    return same


def _in_grid_diff_diag(g: torch.Tensor, sh_r: int, sh_c: int) -> torch.Tensor:
    """DIAGONAL sibling of _in_grid_diff: in-grid diagonal neighbour of a DIFFERENT colour, both
    wrapped edges zeroed. Substrate for the diag_boundary predicate (D6) and _on_boundary(diag=True)."""
    diff = torch.roll(g, shifts=(sh_r, sh_c), dims=(1, 2)) != g
    diff[:, 0 if sh_r == 1 else g.shape[1] - 1, :] = False
    diff[:, :, 0 if sh_c == 1 else g.shape[2] - 1] = False
    return diff


_DIAG_SHIFTS = ((1, 1), (1, -1), (-1, 1), (-1, -1))                # (row-shift, col-shift) x 4 corners


def connected_components(g: torch.Tensor, n_iter: int = 30, n_jump: int = 6,
                         *, diag: bool = False) -> torch.Tensor:
    """[B,S,S] colour ids -> [B,S,S] LONG component label (the min flat-index in each same-colour
    component; 4-connected by default, 8-connected with diag=True).

    D1: returns LONG (was float -- every consumer immediately re-cast; float labels invite precision
    misuse and pointless casts). D4 opt-in: diag=True adds the 4 diagonal neighbour pairs for ARC
    tasks that treat diagonal-touching cells as ONE object (default False = legacy 4-conn exactly,
    matching object_rule_bank._objects).

    Neighbour min-propagation INTERLEAVED with pointer jumping (label <- label[label], the GPU
    union-find trick): agreement travels through the LABEL graph (log steps), not the pixel grid
    (linear steps). Bounded propagation alone has a GEODESIC horizon -- it split one 30x30
    serpentine component into 230 labels at the old n_iter=48, corrupting every label-derived
    channel (comp_size/is_largest/is_singleton/solidity/inside_container) on snake/spiral/maze
    tasks; even n_iter=460 left fragments. n_iter=30 + n_jump=6 converges the full serpentine
    (asserted in _self_test) and is FASTER than the old wrong 48 rounds. At convergence the result
    is identical to exhaustive propagation: every cell holds its component's min flat index."""
    B, S, _ = g.shape
    dev = g.device
    L = S * S
    labels = torch.arange(L, device=dev).view(1, S, S).expand(B, S, S).clone()
    BIG = L + 1
    nbrs = [(1, 1), (-1, 1), (1, 2), (-1, 2)]                      # (shift, dim): up, down, left, right
    for _ in range(n_iter):
        prev = labels
        cand = [labels]
        for sh, dm in nbrs:
            same = _edge_masked_same(g, sh, dm)
            rolled = torch.roll(labels, sh, dims=dm)
            cand.append(torch.where(same, rolled, torch.full_like(rolled, BIG)))
        if diag:
            for sh_r, sh_c in _DIAG_SHIFTS:
                same = _edge_masked_same_diag(g, sh_r, sh_c)
                rolled = torch.roll(labels, shifts=(sh_r, sh_c), dims=(1, 2))
                cand.append(torch.where(same, rolled, torch.full_like(rolled, BIG)))
        labels = torch.stack(cand, dim=0).min(dim=0).values
        # pointer jumping: labels are flat indices, so label-of-my-label is a gather. min-labels
        # only ever DECREASE, so compression preserves the min-index semantics exactly.
        flat = labels.view(B, L)
        for _ in range(n_jump):
            flat = torch.gather(flat, 1, flat)
        labels = flat.view(B, S, S)
        # Fixpoint check: stop once a full iteration changes nothing (correctness does not rely on the
        # tuned n_iter reaching convergence -- a pathological topology that hasn't converged simply
        # keeps iterating up to n_iter; early exit only fires when already stable). One sync/iter, and
        # convergence is typically well before n_iter so this is usually a net speedup.
        if bool(torch.equal(labels, prev)):
            break
    return labels                                                   # LONG (D1; was .float())


def distance_transform(mask: torch.Tensor, max_dist: int) -> torch.Tensor:
    """[B,H,W] boolean mask -> CLIPPED Manhattan distance to the nearest True cell.

    CONTRACT (B8/GPT-#5): distances SATURATE at max_dist -- this is a clipped transform, not an
    exact one. Max Manhattan distance on an HxW grid is H+W-2, so the standard call max_dist=S
    (relational_maps, square) intentionally reads saturation as 'far'; downstream thresholds and
    the /S normalisation rely on that. Pass max_dist >= H+W-2 if exact distances are needed
    (the fast nearest-seed path does).
    D5: fixpoint early-exit added -- each round only lowers values, so an unchanged round proves
    convergence; values are bit-identical to the fixed-round version.
    V2: generalized to RECTANGULAR grids (the old body hardcoded S for both dims -- a latent
    square-only assumption no caller had tripped until the compact-grid fast seed path)."""
    B, H, W = mask.shape
    dist = torch.full((B, H, W), float(max_dist), device=mask.device)
    dist[mask] = 0.0
    for _ in range(max_dist):
        prev = dist
        up = torch.roll(dist, 1, dims=1); up[:, 0, :] = max_dist
        dn = torch.roll(dist, -1, dims=1); dn[:, -1, :] = max_dist
        lt = torch.roll(dist, 1, dims=2); lt[:, :, 0] = max_dist
        rt = torch.roll(dist, -1, dims=2); rt[:, :, -1] = max_dist
        dist = torch.min(dist, torch.min(torch.min(up, dn), torch.min(lt, rt)) + 1.0)
        if bool(torch.equal(dist, prev)):
            break
    return dist


def _component_sizes(g: torch.Tensor, n_iter: int = 30,
                     *, diag: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """[B,S,S] colour ids -> (labels LONG [B,S,S], comp_size FLOAT [B,S,S]).

    B9: THE single source for the `connected_components -> scatter_add -> gather` block that was
    duplicated 5x across object_bank (relational_maps, object_features, is_largest_object,
    is_singleton_object, component_size). Everything below calls this."""
    B, S, _ = g.shape
    L = S * S
    dev = g.device
    labels = connected_components(g, n_iter, diag=diag)
    labels_flat = labels.view(B, L)
    ones = torch.ones(B, L, device=dev)
    size_per_label = torch.zeros(B, L, device=dev).scatter_add(1, labels_flat, ones)
    comp_size = torch.gather(size_per_label, 1, labels_flat).view(B, S, S)
    return labels, comp_size


def _on_boundary(g: torch.Tensor, *, diag: bool = False) -> torch.Tensor:
    """[B,S,S] colour ids -> BOOL [B,S,S]: any IN-GRID neighbour of a DIFFERENT colour.

    B9: the 4-neighbour OR-loop written once (was duplicated in relational_maps + object_features).
    diag=True also checks the 4 diagonal neighbours (used by the diag_boundary predicate, D6)."""
    B, S, _ = g.shape
    diff = torch.zeros(B, S, S, dtype=torch.bool, device=g.device)
    for sh, dm in [(1, 1), (-1, 1), (1, 2), (-1, 2)]:
        diff = diff | _in_grid_diff(g, sh, dm)
    if diag:
        for sh_r, sh_c in _DIAG_SHIFTS:
            diff = diff | _in_grid_diff_diag(g, sh_r, sh_c)
    return diff


def _largest_mask(comp_size: torch.Tensor, g: torch.Tensor, *, exclude_background: bool,
                  bg_token: int | torch.Tensor = COLOR_OFFSET) -> torch.Tensor:
    """FLOAT [B,S,S] is-largest-component mask -- THE fix for bug B1 (GPT-#6): object_bank carried
    TWO incompatible is_largest semantics under one name (relational_maps EXCLUDED the background
    component; object_features INCLUDED it, so on typical grids it flags the bg blob). One
    implementation, explicit keyword; no caller may compute largest-ness any other way.

      exclude_background=True   relational_maps semantics: only valid non-bg cells compete; the
                                max is over object cells (bg blob cannot win). Returned mask is
                                zero outside valid object cells.
      exclude_background=False  legacy object_features semantics: EVERY component competes
                                (including background and PAD blobs); no valid-mask applied.

    bg_token: token value treated as background for the True branch. int (default COLOR_OFFSET =
    colour-0-as-bg, today's behavior) or a broadcastable tensor ([B]/[B,1,1]) for per-sample modal
    background (the bg_mode='modal' path, B2)."""
    B = g.shape[0]
    if not exclude_background:
        max_size = comp_size.view(B, -1).max(dim=1, keepdim=True).values.clamp_min(1.0).view(B, 1, 1)
        return (comp_size == max_size).float()
    if isinstance(bg_token, torch.Tensor):
        bg = bg_token.view(B, 1, 1)
    else:
        bg = torch.full((1, 1, 1), int(bg_token), device=g.device, dtype=g.dtype)
    valid_bool = (g != 0) & (g != 1)
    valid_obj_mask = valid_bool & (g != bg)
    valid_obj_sizes = comp_size * valid_obj_mask.float()
    max_size = valid_obj_sizes.view(B, -1).max(dim=1, keepdim=True).values.clamp_min(1.0).view(B, 1, 1)
    return ((comp_size == max_size) & valid_obj_mask).float()


# ======================================================================================
# section 3 -- LAYER-1 PER-GRID FEATURE MAPS
# ======================================================================================


def _modal_background(g: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """[B,S,S] tokens + valid float mask -> [B] modal-colour BACKGROUND token per sample.
    Ties -> smaller colour (argmax picks the first maximum). No valid cells -> colour 0.
    The bg_mode='modal' substrate (B2): object_bank hardcoded background = colour-0 everywhere."""
    B = g.shape[0]
    L = g.shape[1] * g.shape[2]
    cnt = torch.zeros(B, 10, device=g.device)
    idx = (g.view(B, L) - COLOR_OFFSET).clamp(0, 9)
    cnt.scatter_add_(1, idx, valid_mask.view(B, L))
    return (cnt.argmax(dim=1) + COLOR_OFFSET).to(g.dtype)


def relational_maps(input_tokens: torch.Tensor, side: int, n_iter: int = 30,
                    *, bg_mode: str = "colour0") -> torch.Tensor:
    """[B,L] -> [B,L,REL_MAP_CHANNELS(=13)] deterministic relational map features (no grad).
    Channels (see REL_MAP_RELATION_NAMES / RelCh): valid_mask, is_background, comp_size_log,
    is_largest_non_bg, is_singleton, on_boundary, dist_to_top, dist_to_bottom, dist_to_left,
    dist_to_right (4-way directional clearance within the valid bbox -- the "slide to wall"
    substrate), solidity, inside_container, distance_to_nearest_colour.

    bg_mode (B2, opt-in): 'colour0' (default, bit-identical to object_bank) treats colour-0 as the
    background for is_background / largest-exclusion / can-contain / nearest-colour attribution;
    'modal' uses each sample's modal valid colour instead -- on non-black-background tasks the
    colour-0 assumption silently mis-labels every bg-conditioned channel. A/B deliberately; never
    switch silently."""
    with torch.no_grad():
        B, L = input_tokens.shape
        S = side
        dev = input_tokens.device
        g = input_tokens.long().clamp(0, VOCAB - 1).view(B, S, S)

        valid_mask = ((g != 0) & (g != 1)).float()
        valid_bool = valid_mask.bool()
        if bg_mode == "colour0":
            bg_tok = torch.full((B,), COLOR_OFFSET, device=dev, dtype=g.dtype)
        elif bg_mode == "modal":
            bg_tok = _modal_background(g, valid_mask)
        else:
            raise ValueError(f"bg_mode must be 'colour0' or 'modal', got {bg_mode!r}")
        bgv = bg_tok.view(B, 1, 1)
        is_bg = (g == bgv).float() * valid_mask

        labels, comp_size = _component_sizes(g, n_iter)                        # B9: shared helper
        labels_flat = labels.view(B, L)
        comp_size_log = torch.log1p(comp_size) * valid_mask

        valid_obj_mask = valid_bool & (g != bgv)
        is_largest = _largest_mask(comp_size, g, exclude_background=True, bg_token=bg_tok)  # B1
        is_singleton = (comp_size <= 1.0).float() * valid_obj_mask.float()
        on_boundary = _on_boundary(g).float() * valid_mask                     # B9: shared helper

        # B4 fix: distance to the VALID bounding-box edge, not the padding boundary.
        # Build a mask of the bbox border of the valid region per sample.
        valid_rows = valid_bool.any(dim=2)  # [B, S]
        valid_cols = valid_bool.any(dim=1)  # [B, S]
        # Find the first/last valid row and column per sample
        row_idx = torch.arange(S, device=dev).view(1, S)
        col_idx = torch.arange(S, device=dev).view(1, S)
        first_row = torch.where(valid_rows, row_idx, torch.full_like(row_idx, S)).min(dim=1, keepdim=True).values  # [B,1]
        last_row = torch.where(valid_rows, row_idx, torch.full_like(row_idx, -1)).max(dim=1, keepdim=True).values
        first_col = torch.where(valid_cols, col_idx, torch.full_like(col_idx, S)).min(dim=1, keepdim=True).values
        last_col = torch.where(valid_cols, col_idx, torch.full_like(col_idx, -1)).max(dim=1, keepdim=True).values
        # Per-cell distance to nearest bbox edge (row-wise and col-wise min)
        r = torch.arange(S, device=dev).view(1, S, 1).expand(B, S, S).float()
        c = torch.arange(S, device=dev).view(1, 1, S).expand(B, S, S).float()
        fr = first_row.view(B, 1, 1).float()
        lr = last_row.view(B, 1, 1).float()
        fc = first_col.view(B, 1, 1).float()
        lc = last_col.view(B, 1, 1).float()
        dist_to_top = (r - fr).clamp_min(0)
        dist_to_bot = (lr - r).clamp_min(0)
        dist_to_left = (c - fc).clamp_min(0)
        dist_to_right = (lc - c).clamp_min(0)
        bbox_extent = torch.max(lr - fr, lc - fc).clamp_min(1.0)  # normalise by bbox size, not grid size
        # 4 DIRECTIONAL clearances within the valid bbox (was a single collapsed min -> "slide to wall"
        # could not tell which wall is which). Each normalised by bbox_extent and masked to valid cells.
        dist_top_n = (dist_to_top / bbox_extent) * valid_mask
        dist_bot_n = (dist_to_bot / bbox_extent) * valid_mask
        dist_left_n = (dist_to_left / bbox_extent) * valid_mask
        dist_right_n = (dist_to_right / bbox_extent) * valid_mask

        # Per-LABEL bboxes via scatter-reduce, O(L) -- replaces a [B,L,L] one_hot blow-up (this runs
        # in the DATALOADER hot path for the target AND every context demo grid, num_workers=0).
        rows = torch.arange(S, device=dev).view(1, S, 1).expand(B, S, S).reshape(B, L)
        cols = torch.arange(S, device=dev).view(1, 1, S).expand(B, S, S).reshape(B, L)
        min_r = torch.full((B, L), S, device=dev, dtype=rows.dtype).scatter_reduce_(
            1, labels_flat, rows, reduce="amin", include_self=False)
        max_r = torch.full((B, L), -1, device=dev, dtype=rows.dtype).scatter_reduce_(
            1, labels_flat, rows, reduce="amax", include_self=False)
        min_c = torch.full((B, L), S, device=dev, dtype=cols.dtype).scatter_reduce_(
            1, labels_flat, cols, reduce="amin", include_self=False)
        max_c = torch.full((B, L), -1, device=dev, dtype=cols.dtype).scatter_reduce_(
            1, labels_flat, cols, reduce="amax", include_self=False)

        area = ((max_r - min_r + 1).clamp_min(1) * (max_c - min_c + 1).clamp_min(1)).float()
        cell_bbox_area = torch.gather(area, 1, labels_flat).view(B, S, S)
        solidity = (comp_size / cell_bbox_area) * valid_mask

        # inside_container on COMPACTED unique labels: ARC grids have ~<=50 components, so the
        # pairwise bbox-containment test is U^2 (<~2.5K) instead of L^2 (810K). A label's colour is
        # just g_flat[label] -- label values ARE flat member-cell indices (the component min).
        g_flat = g.view(B, L)
        inside_per_label = torch.zeros(B, L, device=dev)
        for b in range(B):
            uniq = torch.unique(labels_flat[b])                                   # [U]
            vals = g_flat[b, uniq]
            # real colour, not pad/eos, not background (bg-aware: == `vals > COLOR_OFFSET` for colour0)
            can_contain = (vals >= COLOR_OFFSET) & (vals != bg_tok[b])
            mnr, mxr = min_r[b, uniq], max_r[b, uniq]
            mnc, mxc = min_c[b, uniq], max_c[b, uniq]
            contains = ((mnr.unsqueeze(1) <= mnr.unsqueeze(0)) & (mxr.unsqueeze(1) >= mxr.unsqueeze(0))
                        & (mnc.unsqueeze(1) <= mnc.unsqueeze(0)) & (mxc.unsqueeze(1) >= mxc.unsqueeze(0)))
            contains &= can_contain.unsqueeze(1) & ~torch.eye(uniq.numel(), dtype=torch.bool, device=dev)
            inside_per_label[b, uniq] = contains.any(dim=0).float()
        inside_container = torch.gather(inside_per_label, 1, labels_flat).view(B, S, S) * valid_mask

        # ONE batched distance transform over all 10 colours (was 10 sequential calls = 300 roll-rounds).
        c_masks = torch.stack([(g == (c + COLOR_OFFSET)) & valid_bool for c in range(10)], dim=1)
        dist_to_c = distance_transform(c_masks.view(B * 10, S, S), max_dist=S).view(B, 10, S, S)

        # bg cells: distance to the nearest non-bg object. non-bg colour cells: distance to the
        # nearest OTHER non-bg colour (own channel masked out). pad/eos: saturated, zeroed by the mask.
        # bg-aware formulation: masking the bg CHANNEL to S then min over all 10 == the old
        # `dist_to_c[:, 1:]` slice when bg==colour0 (values are clipped at S, so min is unchanged).
        ch = torch.arange(10, device=dev).view(1, 10, 1, 1)
        bg_ch = (bg_tok - COLOR_OFFSET).view(B, 1, 1, 1)
        dist_nonbg = dist_to_c.masked_fill(ch == bg_ch, float(S))
        dist_to_any_obj = dist_nonbg.min(dim=1).values
        idx_c = (g - COLOR_OFFSET).clamp(0, 9)
        own = ch == idx_c.unsqueeze(1)
        min_others = dist_nonbg.masked_fill(own, float(S)).min(dim=1).values
        dist_nearest = torch.full((B, S, S), float(S), device=dev)
        dist_nearest = torch.where(g == bgv, dist_to_any_obj, dist_nearest)
        dist_nearest = torch.where(valid_bool & (g != bgv), min_others, dist_nearest)

        distance_to_nearest_colour = (dist_nearest / float(S)) * valid_mask

        maps = torch.stack([
            valid_mask, is_bg, comp_size_log, is_largest, is_singleton, on_boundary,
            dist_top_n, dist_bot_n, dist_left_n, dist_right_n,        # 4-way directional clearance
            solidity, inside_container, distance_to_nearest_colour
        ], dim=-1)

        return maps.view(B, L, REL_MAP_CHANNELS)


def component_features(input_tokens: torch.Tensor, side: int, n_iter: int = 30,
                       *, exclude_background: bool = False) -> torch.Tensor:
    """[B,L] tokens -> [B,L,OBJ_DIM] deterministic per-cell object features (no grad).
    V2 name of object_features, rebuilt on the section-2 helpers. Channels: comp_size_norm,
    is_largest, is_singleton, on_boundary.

    exclude_background (B1): False = LEGACY object_features semantics -- EVERY component competes
    for is_largest, so on typical grids the background blob wins (the self-test pins this);
    True = relational_maps semantics (bg cannot win). The two are INCOMPATIBLE under one name --
    that keyword is the fix; pick explicitly when reviving the repair lane."""
    with torch.no_grad():
        B, L = input_tokens.shape
        S = side
        g = input_tokens.long().clamp(0, VOCAB - 1).view(B, S, S)
        _labels, comp_size = _component_sizes(g, n_iter)
        comp_size_norm = comp_size / float(L)
        is_largest = _largest_mask(comp_size, g, exclude_background=exclude_background)
        is_singleton = (comp_size <= 1.0).float()
        on_boundary = _on_boundary(g).float()
        feats = torch.stack([comp_size_norm, is_largest, is_singleton, on_boundary], dim=-1)
        return feats.view(B, L, OBJ_DIM)


def object_features(input_tokens: torch.Tensor, side: int, n_iter: int = 30) -> torch.Tensor:
    """LEGACY name for component_features (Phase 2b; consumer: gated-off color_repair_head).
    CAUTION preserved: is_largest here does NOT exclude the background component -- same name,
    different meaning vs relational_maps' channel. component_features(exclude_background=True)
    is the resolved version; this wrapper keeps the historic behavior bit-exact."""
    return component_features(input_tokens, side, n_iter, exclude_background=False)


def is_largest_object(input_tokens: torch.Tensor, side: int, n_iter: int = 30,
                      *, exclude_background: bool = False) -> torch.Tensor:
    """[B,L] tokens -> [B,L] bool: is this cell in the largest (by size) connected component?
    Used to CONDITION the recolour rule on object context (Phase 2c), not just on cell colour.
    LEGACY default exclude_background=False keeps the historic (bg-dominated) semantics;
    True asks the question the name implies -- largest actual OBJECT (B1)."""
    with torch.no_grad():
        B, L = input_tokens.shape
        g = input_tokens.long().clamp(0, VOCAB - 1).view(B, side, side)
        _labels, comp_size = _component_sizes(g, n_iter)
        return _largest_mask(comp_size, g, exclude_background=exclude_background).bool().view(B, L)


def is_singleton_object(input_tokens: torch.Tensor, side: int, n_iter: int = 30) -> torch.Tensor:
    """[B,L] tokens -> [B,L] bool: is this cell an isolated single-cell component (size 1)?
    Distinguishes scattered pixels from solid shapes -- the right conditioning when a recolour
    depends on object SIZE (e.g. big block -> b, scattered cells -> c). is_largest is dominated by
    the background and won't separate them. LIVE consumers: parse.py / solve.py / oracle_eval.py."""
    with torch.no_grad():
        B, L = input_tokens.shape
        g = input_tokens.long().clamp(0, VOCAB - 1).view(B, side, side)
        _labels, comp_size = _component_sizes(g, n_iter)
        return (comp_size <= 1.0).view(B, L)


def component_size(input_tokens: torch.Tensor, side: int, n_iter: int = 30) -> torch.Tensor:
    """[B,L] tokens -> [B,L] float: size of this cell's connected component.
    Consumer: size_bucket below (and through it the probe tools)."""
    with torch.no_grad():
        B, L = input_tokens.shape
        g = input_tokens.long().clamp(0, VOCAB - 1).view(B, side, side)
        _labels, comp_size = _component_sizes(g, n_iter)
        return comp_size.view(B, L)


def size_bucket(input_tokens: torch.Tensor, side: int, n_iter: int = 30,
                small_max: int = 8) -> torch.Tensor:
    """[B,L] tokens -> [B,L] long bucket: 0=singleton(size 1), 1=small(2..small_max), 2=large(>small_max).
    The object-context key for the conditioned VALUE prior (Phase 2c); default key_fn of
    conditioned_transitions (section 6). LIVE consumers: parse.py / solve.py / oracle_eval.py."""
    cs = component_size(input_tokens, side, n_iter)
    bucket = torch.zeros_like(cs, dtype=torch.long)
    bucket = torch.where(cs > 1.0, torch.ones_like(bucket), bucket)
    bucket = torch.where(cs > float(small_max), torch.full_like(bucket, 2), bucket)
    return bucket


# ======================================================================================
# section 4 -- LAYER-2 WHERE PREDICATES
# ======================================================================================


def _rel_where_relation_masks(tokens: torch.Tensor, rel_maps: torch.Tensor) -> torch.Tensor:
    """Existing relmap facts as boolean predicate masks. Order == REL_WHERE_RELATION_NAMES.

    Shape:
        tokens [..., L]
        rel_maps [..., L, REL_MAP_CHANNELS]
        return [..., L, R]  (R = len(REL_WHERE_RELATION_NAMES))

    All channel reads go through RelCh and the named thresholds (B3/GPT-#7: the old magic indices
    silently survived one 10->13 layout drift -- the 7/8/9 reads landed on clearance channels)."""
    valid = tokens.long() >= COLOR_OFFSET
    rm = rel_maps.float()
    # comp_size_log = log1p(size)*valid -> expm1 inverts it exactly on valid cells (0 stays 0).
    comp_size = torch.expm1(rm[..., RelCh.COMP_SIZE_LOG]).clamp_min(0.0)
    # 13-channel layout: distance_to_edge is 4 DIRECTIONAL clearances; the old single
    # "nearest-edge distance" == their per-cell min.
    dist_edge = torch.minimum(
        torch.minimum(rm[..., RelCh.DIST_TO_TOP], rm[..., RelCh.DIST_TO_BOTTOM]),
        torch.minimum(rm[..., RelCh.DIST_TO_LEFT], rm[..., RelCh.DIST_TO_RIGHT]))
    dist_nearest = rm[..., RelCh.DISTANCE_TO_NEAREST_COLOUR]
    # D6 diag_boundary: computed from the TOKEN grid, not rel_maps -- REL_MAP_CHANNELS stays 13 and
    # no zero-init projection resizes. Same convention as on_boundary: any IN-GRID (here diagonal)
    # neighbour of a different value counts, including pad/eos neighbours at the valid border.
    lead = tokens.shape[:-1]
    L = tokens.shape[-1]
    S = int(math.isqrt(L))
    g = tokens.long().reshape(-1, S, S)
    diagb = torch.zeros(g.shape, dtype=torch.bool, device=g.device)
    for sh_r, sh_c in _DIAG_SHIFTS:
        diagb = diagb | _in_grid_diff_diag(g, sh_r, sh_c)
    diag_boundary = diagb.reshape(*lead, L)
    masks = [
        valid,
        (rm[..., RelCh.IS_BACKGROUND] > 0.5) & valid,
        (rm[..., RelCh.IS_LARGEST] > 0.5) & valid,
        (rm[..., RelCh.IS_SINGLETON] > 0.5) & valid,
        (rm[..., RelCh.ON_BOUNDARY] > 0.5) & valid,
        (dist_edge <= EDGE_NEAR) & valid,
        (dist_edge >= EDGE_FAR) & valid,
        (rm[..., RelCh.SOLIDITY] >= SOLID_MIN) & valid,
        (rm[..., RelCh.SOLIDITY] > 0.0) & (rm[..., RelCh.SOLIDITY] < SOLID_MIN) & valid,
        (rm[..., RelCh.INSIDE_CONTAINER] > 0.5) & valid,
        (dist_nearest <= NEAREST_NEAR) & valid,
        (dist_nearest >= NEAREST_FAR) & valid,
        (comp_size > 0.0) & (comp_size <= COMP_SMALL_MAX) & valid,
        (comp_size >= COMP_SMALL_MAX + 1.0) & (comp_size <= COMP_MED_MAX) & valid,
        (comp_size >= COMP_MED_MAX + 1.0) & valid,
        diag_boundary & valid,                                   # D6, appended LAST
    ]
    return torch.stack(masks, dim=-1)


def _rel_where_candidate_masks(tokens: torch.Tensor, rel_maps: torch.Tensor) -> torch.Tensor:
    """Predicate bank: input-colour, relation, and input-colour+relation conjunctions.
    P = 10 + R + 10*R (R=16 with diag_boundary -> P=186; was 175). Layout: [colour | relation |
    conjunction (colour-major)]. NOTE (D6): legacy conjunction indices shift because R grew --
    only the logged rel_where_predicate_index renumbers; no weights are keyed on P."""
    valid = tokens.long() >= COLOR_OFFSET
    colour_masks = torch.stack(
        [(tokens.long() == (COLOR_OFFSET + c)) & valid for c in range(10)],
        dim=-1,
    )
    rel_masks = _rel_where_relation_masks(tokens, rel_maps)
    conj = (colour_masks.unsqueeze(-1) & rel_masks.unsqueeze(-2)).flatten(-2)
    return torch.cat([colour_masks, rel_masks, conj], dim=-1)


def relational_where_hint(
    target_inputs: torch.Tensor,
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
    context_mask: torch.Tensor,
    *,
    target_rel_maps: torch.Tensor | None = None,
    context_rel_maps: torch.Tensor | None = None,
    side: int = 30,
    topk: int = 1,
    overselect_penalty: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Support-derived WHERE evidence from existing relmap facts.

    This is a hint, not an executor: it ranks simple predicates by how well they
    explain support changed cells, then applies the best predicate to target input.
    It never reads target output and runs under no_grad.

    Zero-support contract (stated, self-tested): no valid changed support cells -> every score is
    -1 -> hint all-zero, predicate index -1, confidence 0. Callers may rely on the zeros.

    overselect_penalty (opt-in, default 0.0 = legacy scores exactly): subtracts
    penalty * log(|selected| / |true changed|) from each predicate's score on rows that HAVE
    changed cells -- same F1, same FPR, prefer the mask whose size is closest to the true changed
    mass. A/B knob; never on silently."""
    with torch.no_grad():
        ti = target_inputs.long()
        ci = context_inputs.long()
        co = context_outputs.long()
        cm = context_mask.to(torch.bool)
        B, L = ti.shape
        M = ci.shape[1]
        if target_rel_maps is None:
            target_rel_maps = relational_maps(ti, side=side)
        if context_rel_maps is None:
            context_rel_maps = relational_maps(ci.reshape(B * M, L), side=side).view(B, M, L, -1)

        target_candidates = _rel_where_candidate_masks(ti, target_rel_maps).to(torch.bool)      # [B,L,P]
        support_candidates = _rel_where_candidate_masks(ci, context_rel_maps).to(torch.bool)    # [B,M,L,P]
        valid = (ci >= COLOR_OFFSET) & (co >= COLOR_OFFSET) & cm.unsqueeze(-1)
        changed = valid & (ci != co)

        # Batched predicate scoring (no per-row Python loop / GPU sync: this runs on the MAIN forward
        # and BOTH aux forwards every training step). Semantics identical to the per-row version.
        cand = support_candidates.reshape(B, M * L, -1)                  # [B, ML, P]
        v = valid.reshape(B, M * L, 1)
        truth = changed.reshape(B, M * L, 1)
        selected = cand & v
        tp = (selected & truth).sum(dim=1).float()                       # [B, P]
        fp = (selected & (~truth) & v).sum(dim=1).float()
        fn = ((~selected) & truth).sum(dim=1).float()
        tn = ((~selected) & (~truth) & v).sum(dim=1).float()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = torch.where(
            (precision + recall) > 0,
            2.0 * precision * recall / (precision + recall).clamp_min(1e-12),
            torch.zeros_like(precision),
        )
        fpr = fp / (fp + tn).clamp_min(1.0)
        score = torch.where((tp + fp) > 0, f1 - 0.25 * fpr, torch.full_like(f1, -1.0))
        if overselect_penalty > 0.0:
            # Opt-in mask-bloat penalty (accepted external suggestion): guard on rows that actually
            # have changed cells so empty-truth rows keep their legacy -1 scores untouched.
            truth_count = truth.sum(dim=1).float()                       # [B, 1]
            selected_count = selected.sum(dim=1).float()                 # [B, P]
            overselect = (selected_count / truth_count.clamp_min(1.0)).clamp_min(1.0)
            score = torch.where(truth_count > 0,
                                score - overselect_penalty * torch.log(overselect), score)
        # FIX D: expose the top-K predicate masks, each scaled by its own score, and let the head
        # combine them. K=1 reproduces the legacy single hard winner exactly (channel 0 semantics
        # unchanged); stats below are always computed from the top-1.
        K = max(1, min(int(topk), score.shape[-1]))
        top_scores, top_idx = score.topk(K, dim=-1)                      # [B, K]
        # topk tie-breaking differs from argmax; pin channel 0 to the argmax winner so K=1 (and
        # channel-0 consumers at any K) reproduce the legacy hint EXACTLY, ties included.
        idx = score.argmax(dim=-1)
        top_idx[:, 0] = idx
        top_scores[:, 0] = score.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        rows = torch.arange(B, device=ti.device)
        picked = score[rows, idx] > 0
        best_index = torch.where(picked, idx, torch.full_like(idx, -1))
        best_f1 = torch.where(picked, f1[rows, idx], torch.zeros(B, device=ti.device))
        best_fpr = torch.where(picked, fpr[rows, idx], torch.zeros(B, device=ti.device))
        confidence = torch.where(picked, score[rows, idx].clamp_min(0.0), torch.zeros(B, device=ti.device))
        masks = target_candidates.gather(
            -1, top_idx.view(B, 1, K).expand(-1, L, -1)).to(target_rel_maps.dtype)         # [B, L, K]
        hint = masks * top_scores.clamp_min(0.0).view(B, 1, K).to(masks.dtype)             # score<=0 -> zeroed

        return hint, {
            "rel_where_f1": best_f1,
            "rel_where_fpr": best_fpr,
            "rel_where_confidence": confidence,
            "rel_where_predicate_index": best_index,
        }


# ======================================================================================
# section 5 -- LAYER-3 CONDITIONING SIGNATURE
# cell_conditioning_signature was a 165-line monster welding four unrelated computations;
# V2 splits it into testable helpers orchestrated by a thin top function (same output).
# ======================================================================================

# --- section 5.0: object extractors (moved verbatim from object_rule_bank, Block 0) -------------
# These 5 extractors + 2 helpers were formerly imported lazily via _rule_bank(); they now live here
# so relation_map imports nothing from object_rule_bank (cycle dissolved). core_prior.py (file #2)
# re-exports them under their old names. Bodies are byte-for-byte the object_rule_bank originals --
# do NOT edit here without an equivalence A/B; the old file remains the oracle.

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


def _third(v: int, lo: int, hi: int) -> int:
    """Which third (0=first,1=mid,2=last) of [lo,hi] does v fall in? Robust to lo==hi."""
    span = hi - lo
    if span <= 0:
        return 1
    return min(2, int((v - lo) * 3 // (span + 1)))


def _sig_neighbour_cols(g: torch.Tensor) -> torch.Tensor:
    """[B,S,S] tokens -> [B,S,S,4] SORTED 4-neighbour colours (0-9; 10 = out-of-grid/pad).
    Universal (works on background cells that own no object); direction-invariant via sort."""
    B, S, _ = g.shape
    colour = g >= COLOR_OFFSET
    gc = torch.where(colour, g - COLOR_OFFSET, torch.full_like(g, 10))     # pad/eos -> 10
    gp = torch.full((B, S + 2, S + 2), 10, dtype=torch.long, device=g.device)  # out-of-grid -> 10
    gp[:, 1:S + 1, 1:S + 1] = gc
    up, dn = gp[:, 0:S, 1:S + 1], gp[:, 2:S + 2, 1:S + 1]
    lt, rt = gp[:, 1:S + 1, 0:S], gp[:, 1:S + 1, 2:S + 2]
    return torch.stack([up, dn, lt, rt], dim=-1).sort(dim=-1).values


def _sig_object_cols(sig: torch.Tensor, b: int, side: int, col, hw, bg: int, objs: list) -> None:
    """Per-grid object columns 5-10 (rank/holes/shape/local-thirds/object-colour) PLUS the V2
    touch columns 14-15 (B13). Foreground cells use their OWN object; background cells use the
    SMALLEST enclosing object (bbox attribution). Writes sig[b] in place.

    Touch (col 14/15) definition, pinned: 4-adjacency on the compact grid; neighbour colour
    != own object colour and != bg; mode counted by adjacency INCIDENTS, tie -> smaller colour
    (same convention as seeds); count = distinct touching colours, bucket 0/1/2/3+."""
    H, W = hw
    order = sorted(range(len(objs)), key=lambda i: -objs[i]["size"])
    rank_of = {i: min(r, 7) for r, i in enumerate(order)}
    canon_bucket: dict = {}
    holes = [min(_hole_count(o), 3) for o in objs]
    shapes = []
    for o in objs:
        ck = _d4_canon(o)
        if ck not in canon_bucket:
            canon_bucket[ck] = min(len(canon_bucket), 6)
        shapes.append(canon_bucket[ck])

    # V2 touch columns (B13): per-object modal touching colour + distinct-count bucket.
    cg = col.tolist()
    touch_mode: list[int] = []
    touch_cnt: list[int] = []
    for o in objs:
        oc = o["colour"]
        oc_int = oc if isinstance(oc, int) else -1        # multi-colour stamp: exclude nothing extra
        hist = [0] * 10
        member = o["cells"] if isinstance(o["cells"], set) else set(o["cells"])
        for (r, c) in o["cells"]:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (0 <= nr < H and 0 <= nc < W and (nr, nc) not in member):
                    v = cg[nr][nc]
                    if v >= 0 and v != bg and v != oc_int:
                        hist[v] += 1
        if sum(hist) > 0:
            touch_mode.append(max(range(10), key=lambda k: (hist[k], -k)))
            touch_cnt.append(min(sum(1 for h in hist if h > 0), 3))
        else:
            touch_mode.append(10)                          # touches nothing -> colour sentinel
            touch_cnt.append(0)                            # attributed, zero touching colours

    def _write(j: int, i: int, r: int, c: int) -> None:
        o = objs[i]
        sig[b, j, 5] = rank_of[i]
        sig[b, j, 6] = holes[i]
        sig[b, j, 7] = shapes[i]
        sig[b, j, 8] = _third(r, o["rmin"], o["rmax"])
        sig[b, j, 9] = _third(c, o["cmin"], o["cmax"])
        oc = o["colour"]
        sig[b, j, 10] = int(oc) if isinstance(oc, int) else 10          # mono colour; multi-stamp -> none
        sig[b, j, 14] = touch_mode[i]                                    # V2 (B13)
        sig[b, j, 15] = touch_cnt[i]

    for i, o in enumerate(objs):                                        # foreground: own object
        for (r, c) in o["cells"]:
            _write(r * side + c, i, r, c)
    for r in range(H):                                                  # background: smallest enclosing object
        for c in range(W):
            if int(col[r, c]) != bg:
                continue
            best, best_area = None, 1 << 30
            for i, o in enumerate(objs):
                if o["rmin"] <= r <= o["rmax"] and o["cmin"] <= c <= o["cmax"]:
                    a = (o["rmax"] - o["rmin"] + 1) * (o["cmax"] - o["cmin"] + 1)
                    if a < best_area:
                        best_area, best = a, i
            if best is not None:
                _write(r * side + c, best, r, c)


def _sig_enclosure_colour(cg: list, bg: int, H: int, W: int) -> list:
    """Col 11: TRUE flood-fill enclosure colour. Connected components of BACKGROUND cells (4-conn);
    a component touching the compact-grid border is OUTSIDE (sentinel 10); an enclosed component's
    value is the modal colour among non-bg cells 4-adjacent to it.

    DELIBERATELY a second, hand-rolled CC -- concavity-aware where the bbox key (col 10) is not:
    on a C-shape the 'interior' is border-reachable through the gap, so col 11 says OUTSIDE while
    col 10 still claims containment. That divergence is the whole point; do NOT de-duplicate this
    into connected_components. Returns encl[H][W] (colour for bg cells; 10 elsewhere/outside)."""
    comp_id = [[-1] * W for _ in range(H)]
    comp_touches_border = []
    comp_cells: list = []
    for rr in range(H):
        for cc in range(W):
            if cg[rr][cc] != bg or comp_id[rr][cc] != -1:
                continue
            cid = len(comp_cells)
            stack = [(rr, cc)]
            comp_id[rr][cc] = cid
            cells = []
            touches = False
            while stack:
                pr, pc = stack.pop()
                cells.append((pr, pc))
                if pr == 0 or pr == H - 1 or pc == 0 or pc == W - 1:
                    touches = True
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = pr + dr, pc + dc
                    if 0 <= nr < H and 0 <= nc < W and cg[nr][nc] == bg and comp_id[nr][nc] == -1:
                        comp_id[nr][nc] = cid
                        stack.append((nr, nc))
            comp_cells.append(cells)
            comp_touches_border.append(touches)
    comp_encl_colour = [10] * len(comp_cells)
    for cid, cells in enumerate(comp_cells):
        if comp_touches_border[cid]:
            continue
        hist = [0] * 10
        for (pr, pc) in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < H and 0 <= nc < W and cg[nr][nc] != bg and cg[nr][nc] >= 0:
                    hist[cg[nr][nc]] += 1
        if sum(hist) > 0:
            comp_encl_colour[cid] = max(range(10), key=lambda k: (hist[k], -k))
    encl = [[10] * W for _ in range(H)]
    for rr in range(H):
        for cc in range(W):
            if cg[rr][cc] == bg:
                encl[rr][cc] = comp_encl_colour[comp_id[rr][cc]]
    return encl


def _sig_nearest_seed_ref(cg: list, bg: int, H: int, W: int):
    """Cols 12(+13) REFERENCE: the original combined multi-source BFS (Manhattan), colour on ties ->
    smaller index, PLUS tie propagation for col 13 (tie = a second distinct colour reaches the cell
    at the same minimal distance, directly or through a tied predecessor). Kept as the self-test
    oracle for the vectorized fast path. Returns (seed[H][W], tie[H][W], has_seed)."""
    seed_colour = [[-1] * W for _ in range(H)]
    dist = [[-1] * W for _ in range(H)]
    tie = [[False] * W for _ in range(H)]
    q = deque()
    for rr in range(H):
        for cc in range(W):
            if cg[rr][cc] != bg and cg[rr][cc] >= 0:
                seed_colour[rr][cc] = cg[rr][cc]
                dist[rr][cc] = 0
                q.append((rr, cc))
    has_seed = len(q) > 0
    while q:
        pr, pc = q.popleft()
        d = dist[pr][pc]
        sc = seed_colour[pr][pc]
        st = tie[pr][pc]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < H and 0 <= nc < W:
                if dist[nr][nc] == -1:
                    dist[nr][nc] = d + 1
                    seed_colour[nr][nc] = sc
                    tie[nr][nc] = st
                    q.append((nr, nc))
                elif dist[nr][nc] == d + 1:
                    if sc != seed_colour[nr][nc]:
                        tie[nr][nc] = True                      # two distinct colours at min distance
                        if sc < seed_colour[nr][nc]:
                            seed_colour[nr][nc] = sc            # tie -> smaller colour index (legacy)
                    else:
                        tie[nr][nc] = tie[nr][nc] or st         # same colour, tied upstream
    return seed_colour, tie, has_seed


def _sig_nearest_seed_fast(cg: list, bg: int, H: int, W: int, device) -> tuple:
    """Cols 12(+13) FAST (B7/GPT-#4): per-colour EXACT distance transform (max_dist=H+W covers the
    H+W-2 diameter, so nothing clips) + argmin over the colour-ordered stack. Colour order makes
    argmin reproduce the smaller-colour-wins tie-break EXACTLY; tie flag = >=2 colours at the min.
    Returns (seed LONG [H,W], tie LONG [H,W], has_seed) -- provably equal to the ref (self-tested)."""
    grid = torch.tensor(cg, dtype=torch.long, device=device)
    masks = torch.stack([(grid == c) if c != bg else torch.zeros_like(grid, dtype=torch.bool)
                         for c in range(10)], dim=0)                    # [10,H,W]; bg never a seed
    has_seed = bool(masks.any())
    if not has_seed:
        z = torch.zeros(H, W, dtype=torch.long, device=device)
        return z - 1, z, False
    md = H + W                                                          # exact: diameter is H+W-2
    dist = distance_transform(masks, md)                                # [10,H,W]
    dmin = dist.min(dim=0).values
    seed = dist.argmin(dim=0)                                           # first (=smallest) colour at min
    tie = ((dist == dmin.unsqueeze(0)).sum(dim=0) > 1).long()
    return seed, tie, True


# Bounded FIFO cache (B7): the same target grid is signed 3x per training step (MAIN + 2 aux
# forwards) and the frozen fixed-eval batches recur every 50 steps. Key = raw token bytes + side +
# device; values are cloned on hit (callers may write into the returned tensors).
_SIG_CACHE: dict = {}
_SIG_CACHE_CAP = 256


def cell_conditioning_signature(input_tokens: torch.Tensor, side: int, *, cache: bool = True):
    """[B,L] tokens -> (sig LONG [B,L,CELL_SIG_DIM], valid BOOL [B,L]). Deterministic, no-grad.

    Columns (see CELL_SIG_NAMES / SigCol): 0 source colour (0-9, -1 pad/eos); 1-4 sorted
    4-neighbour colours (0-9, 10 out-of-grid/pad); 5 enclosing-object size-rank (0=largest.., 8=none);
    6 hole-count (0-3, 4=none); 7 D4 shape-signature bucket (0-6 per grid, 7=none); 8-9 local
    row/col third within the object's bbox (0-2, 3=none); 10 own/enclosing object colour (10=none);
    11 flood-fill enclosure colour (10=outside); 12 nearest non-bg seed colour (10=no seeds);
    13 V2 nearest-seed tie flag; 14-15 V2 object touch colour mode / count bucket (B13).
    Foreground cells use their OWN object; background cells the SMALLEST enclosing object.

    D3: tokens are clamped to [0, VOCAB-1] at entry (relational_maps always clamped; this function
    did not -- a stray token >= 12 leaked garbage into the neighbour/seed colour columns)."""
    with torch.no_grad():
        input_tokens = input_tokens.long().clamp(0, VOCAB - 1)          # D3 entry clamp
        B, L = input_tokens.shape
        S = side
        dev = input_tokens.device
        key = None
        if cache:
            key = (input_tokens.cpu().numpy().tobytes(), side, str(dev))
            hit = _SIG_CACHE.get(key)
            if hit is not None:
                return hit[0].clone(), hit[1].clone()
        g = input_tokens.view(B, S, S)
        colour = g >= COLOR_OFFSET

        # --- vectorized columns 0-4: source colour + sorted 4-neighbour colours (universal) ---
        nbr = _sig_neighbour_cols(g)                                     # [B,S,S,4]
        sig = torch.empty(B, L, CELL_SIG_DIM, dtype=torch.long, device=dev)
        for k in range(CELL_SIG_DIM):
            sig[:, :, k] = CELL_SIG_NONE[k]
        sig[:, :, 0] = torch.where(colour, g - COLOR_OFFSET, torch.full_like(g, -1)).view(B, L)
        sig[:, :, 1:5] = nbr.reshape(B, L, 4)

        # --- per-grid columns 5-15 ---
        for b in range(B):
            col, hw = _compact_colour(input_tokens[b], side)
            if col is None:
                continue
            H, W = hw
            bg = _background(col)
            objs = _objects(col, bg, multi=False)
            if objs:
                _sig_object_cols(sig, b, side, col, hw, bg, objs)        # cols 5-10 + 14-15

            cg = col.tolist()
            encl = _sig_enclosure_colour(cg, bg, H, W)                   # col 11
            for rr in range(H):
                for cc in range(W):
                    if cg[rr][cc] == bg:
                        sig[b, rr * side + cc, 11] = encl[rr][cc]

            seed, tie, has_seed = _sig_nearest_seed_fast(cg, bg, H, W, dev)  # cols 12-13
            if has_seed:
                for rr in range(H):
                    for cc in range(W):
                        if cg[rr][cc] >= 0:
                            j = rr * side + cc
                            sig[b, j, 12] = int(seed[rr, cc])
                            sig[b, j, 13] = int(tie[rr, cc])

        valid = colour.view(B, L)
        if cache and key is not None:
            if len(_SIG_CACHE) >= _SIG_CACHE_CAP:
                _SIG_CACHE.pop(next(iter(_SIG_CACHE)))                   # FIFO evict
            _SIG_CACHE[key] = (sig.clone(), valid.clone())
        return sig, valid


# ======================================================================================
# section 6 -- CROSS-DEMO CONSENSUS
# ======================================================================================


def conditioned_transitions(context_inputs: torch.Tensor, context_outputs: torch.Tensor,
                            context_mask: torch.Tensor, side: int, n_iter: int = 30,
                            *, key_fn=None, n_keys: int = N_SIZE_BUCKETS) -> torch.Tensor:
    """[B,M,L] demos -> [B, n_keys, 10, 10] = per-key P(out | in=a, CHANGED) consensus. No grad.

    THE general engine for "per-key transition table over changed support cells" (V2 reuse of the
    formerly-dead object_conditioned_transitions): key_fn(tokens, side) -> LONG [B,L] per-cell
    bucket in [0, n_keys). key_fn=None -> size_bucket -> BIT-IDENTICAL to the legacy function.
    Pass a cell_conditioning_signature-derived bucket and this becomes the offline
    conditional-recolor proposer's table builder (file #7) -- one engine, not three copies.
    Out-of-range keys clamp into the last bucket (legacy flat-index clamp preserved)."""
    with torch.no_grad():
        B, M, L = context_inputs.shape
        K = int(n_keys)
        dev = context_inputs.device
        x = context_inputs.long()
        y = context_outputs.long()
        cmb = context_mask.to(torch.bool)
        cooc = torch.zeros(B, K * 100, device=dev)
        for m in range(M):
            xin, yout = x[:, m], y[:, m]
            bucket = key_fn(xin, side) if key_fn is not None else size_bucket(xin, side, n_iter)
            real = (xin >= COLOR_OFFSET) & (yout >= COLOR_OFFSET)
            changed = (real & (xin != yout) & cmb[:, m].unsqueeze(-1)).float()  # [B,L]
            xc = (xin - COLOR_OFFSET).clamp(0, 9)
            yc = (yout - COLOR_OFFSET).clamp(0, 9)
            flat = (bucket * 100 + xc * 10 + yc).clamp(0, K * 100 - 1)          # [B,L]
            cooc.scatter_add_(1, flat, changed)
        cooc = cooc.view(B, K, 10, 10)
        return cooc / cooc.sum(dim=-1, keepdim=True).clamp_min(1e-6)            # [B,K,10,10]


def object_conditioned_transitions(context_inputs: torch.Tensor, context_outputs: torch.Tensor,
                                   context_mask: torch.Tensor, side: int,
                                   n_iter: int = 30) -> torch.Tensor:
    """[B,M,L] demos -> [B, K, 10, 10] = per-SIZE-BUCKET P(out | in=a, CHANGED) consensus (K=3).
    Unlike the cell-colour-only cond_changed, this can express "colour a -> b in a big shape,
    a -> c as a scattered pixel" -- the conditional recolours that cap HEAD changed at dpcc.
    LEGACY name: one-line wrapper over the general conditioned_transitions engine above."""
    return conditioned_transitions(context_inputs, context_outputs, context_mask, side, n_iter)


# ======================================================================================
# section 7 -- SELF-TEST
# Part A ports every object_bank.py assertion VERBATIM (the legacy contracts must survive).
# Part B adds the 14 V2 contracts (D1-D6, the shared helpers, the new columns, the cache).
# ======================================================================================


def _self_test() -> None:
    # ---------------------------------------------------------------------------------------
    # Part A -- legacy contracts, ported from object_bank._self_test (semantics unchanged).
    # ---------------------------------------------------------------------------------------
    S = 6
    grid = torch.full((S, S), 0 + COLOR_OFFSET, dtype=torch.long)
    grid[0:2, 0:2] = 3 + COLOR_OFFSET                                          # 2x2 block, size 4
    grid[4, 4] = 5 + COLOR_OFFSET                                              # singleton
    inp = grid.reshape(1, S * S)

    labels = connected_components(grid.unsqueeze(0))
    assert labels.dtype == torch.long, "D1: connected_components must return long labels"
    blk = labels[0, 0:2, 0:2]
    assert (blk == blk[0, 0]).all(), "2x2 block must be one component"
    assert labels[0, 4, 4] != labels[0, 0, 0], "singleton != block"

    feats = component_features(inp, side=S)                                    # [1, 36, OBJ_DIM]
    f = feats[0].view(S, S, OBJ_DIM)
    comp_size_norm, is_largest, is_singleton, on_boundary = (f[..., i] for i in range(OBJ_DIM))
    assert abs(float(comp_size_norm[0, 0]) - 4.0 / 36.0) < 1e-5, float(comp_size_norm[0, 0])
    assert float(is_singleton[4, 4]) == 1.0 and float(is_singleton[0, 0]) == 0.0
    # legacy object_features semantics: background is the largest component (bg NOT excluded)
    assert float(is_largest[5, 5]) == 1.0 and float(is_largest[0, 0]) == 0.0
    assert float(on_boundary[1, 1]) == 1.0, "block cell touching background must be a boundary"
    assert float(on_boundary[5, 0]) == 0.0, "far background cell is not a boundary"

    # object_conditioned_transitions: colour 3 -> 7 in a LARGE block, 3 -> 5 as a SINGLETON.
    S2 = 8
    gin = torch.full((S2, S2), 0 + COLOR_OFFSET, dtype=torch.long)
    gin[0:3, 0:3] = 3 + COLOR_OFFSET                                           # 9-cell block (large)
    gin[6, 6] = 3 + COLOR_OFFSET                                               # singleton
    gout = gin.clone()
    gout[0:3, 0:3] = 7 + COLOR_OFFSET
    gout[6, 6] = 5 + COLOR_OFFSET
    ci2 = gin.reshape(1, 1, S2 * S2)
    co2 = gout.reshape(1, 1, S2 * S2)
    cm2 = torch.ones(1, 1, dtype=torch.bool)
    bk = size_bucket(ci2[:, 0], side=S2)
    assert int(bk[0, 0]) == 2 and int(bk[0, 6 * S2 + 6]) == 0, "block=large(2), singleton=0"
    cond_obj = object_conditioned_transitions(ci2, co2, cm2, side=S2)
    assert int(cond_obj[0, 2, 3].argmax()) == 7, "large-bucket: colour 3 -> 7"
    assert int(cond_obj[0, 0, 3].argmax()) == 5, "singleton-bucket: colour 3 -> 5"

    # SNAKY topology: a 30x30 serpentine is ONE component.
    S3 = 30
    serp = torch.full((S3, S3), 0 + COLOR_OFFSET, dtype=torch.long)
    for r in range(S3):
        if r % 2 == 0:
            serp[r, :] = 3 + COLOR_OFFSET
        else:
            serp[r, S3 - 1 if (r // 2) % 2 == 0 else 0] = 3 + COLOR_OFFSET
    slab = connected_components(serp.unsqueeze(0))
    n_serp = len(set(slab[0][serp == 3 + COLOR_OFFSET].tolist()))
    assert n_serp == 1, f"serpentine must be ONE component, got {n_serp} (propagation horizon too small)"

    # relational_maps CONTRACT on a TRANSLATED canvas: pad-zero, bbox clearances, containment.
    S4 = 8
    canvas = torch.zeros(S4, S4, dtype=torch.long)
    canvas[1:5, 2:5] = 0 + COLOR_OFFSET
    canvas[2, 3] = 4 + COLOR_OFFSET
    canvas[5, 2:5] = 1
    canvas[1:5, 5] = 1
    rm = relational_maps(canvas.reshape(1, S4 * S4), side=S4).view(S4, S4, REL_MAP_CHANNELS)
    assert float(rm[canvas < COLOR_OFFSET].abs().max()) == 0.0, "PAD/EOS cells must be all-zero in every channel"
    cell = rm[2, 3]
    for ch, want in ((RelCh.DIST_TO_TOP, 1 / 3), (RelCh.DIST_TO_BOTTOM, 2 / 3),
                     (RelCh.DIST_TO_LEFT, 1 / 3), (RelCh.DIST_TO_RIGHT, 1 / 3)):
        assert abs(float(cell[ch]) - want) < 1e-5, f"clearance ch{ch}: {float(cell[ch])} != {want}"
    assert float(rm[..., RelCh.INSIDE_CONTAINER].max()) == 0.0, "no enclosing object -> inside_container 0"
    ring = torch.full((S4, S4), 0 + COLOR_OFFSET, dtype=torch.long)
    ring[1:5, 1:5] = 6 + COLOR_OFFSET
    ring[2:4, 2:4] = 0 + COLOR_OFFSET
    ring[2, 2] = 4 + COLOR_OFFSET
    rm2 = relational_maps(ring.reshape(1, S4 * S4), side=S4).view(S4, S4, REL_MAP_CHANNELS)
    assert float(rm2[2, 2, RelCh.INSIDE_CONTAINER]) == 1.0, "object inside a ring's bbox must be inside_container"
    assert float(rm2[0, 0, RelCh.INSIDE_CONTAINER]) == 0.0, "outside cells must not be inside_container"

    # cell_conditioning_signature CONTRACT: the KEY must separate a MULTI-TARGET source colour.
    S5 = 10
    cs = torch.full((S5, S5), 0 + COLOR_OFFSET, dtype=torch.long)
    cs[1:4, 1:4] = 3 + COLOR_OFFSET
    cs[7, 7] = 3 + COLOR_OFFSET
    cs[9, 9] = 0
    sig, valid = cell_conditioning_signature(cs.reshape(1, S5 * S5), side=S5, cache=False)
    sig = sig.view(S5, S5, CELL_SIG_DIM); valid = valid.view(S5, S5)
    big_rank = int(sig[1, 1, SigCol.OBJ_SIZE_RANK]); small_rank = int(sig[7, 7, SigCol.OBJ_SIZE_RANK])
    assert big_rank != small_rank, f"big vs singleton must get DIFFERENT size-rank ({big_rank} vs {small_rank})"
    assert big_rank == 0 and small_rank > 0, "largest object rank 0, singleton higher"
    assert int(sig[1, 1, 0]) == 3 and int(sig[7, 7, 0]) == 3, "source colour column == cell colour (=3)"
    assert int(sig[0, 0, 0]) == 0, "background(colour-0) is a real source colour"
    assert not bool(valid[9, 9]) and int(sig[9, 9, 0]) == -1, "PAD (token 0) -> not valid, self_color null"
    fr = torch.full((S5, S5), 0 + COLOR_OFFSET, dtype=torch.long)
    fr[1:6, 1:6] = 5 + COLOR_OFFSET
    fr[2:5, 2:5] = 0 + COLOR_OFFSET
    sig2, _ = cell_conditioning_signature(fr.reshape(1, S5 * S5), side=S5, cache=False)
    sig2 = sig2.view(S5, S5, CELL_SIG_DIM)
    assert int(sig2[3, 3, SigCol.OBJ_SIZE_RANK]) != CELL_SIG_NONE[SigCol.OBJ_SIZE_RANK], "enclosed bg inherits rank"
    assert int(sig2[0, 0, SigCol.OBJ_SIZE_RANK]) == CELL_SIG_NONE[SigCol.OBJ_SIZE_RANK], "un-enclosed bg -> none"
    assert int(sig2[3, 3, SigCol.OBJ_COLOR]) == 5, "enclosed bg inherits frame COLOUR (=5), the fill key"
    assert int(sig2[0, 0, SigCol.OBJ_COLOR]) == CELL_SIG_NONE[SigCol.OBJ_COLOR], "un-enclosed bg -> no container colour"

    # FIX H columns 11/12 + C-shape divergence (concavity), from object_bank.
    assert int(sig2[3, 3, 11]) == 5, "flood-fill: bg cell inside a CLOSED colour-5 ring -> encl colour 5"
    assert int(sig2[0, 0, 11]) == CELL_SIG_NONE[11], "flood-fill: border-reachable bg -> sentinel"
    cshape = torch.full((S5, S5), 0 + COLOR_OFFSET, dtype=torch.long)
    cshape[1:6, 1:6] = 5 + COLOR_OFFSET
    cshape[2:5, 2:5] = 0 + COLOR_OFFSET
    cshape[3, 5] = 0 + COLOR_OFFSET                                          # the GAP in the right wall
    sig3, _ = cell_conditioning_signature(cshape.reshape(1, S5 * S5), side=S5, cache=False)
    sig3 = sig3.view(S5, S5, CELL_SIG_DIM)
    assert int(sig3[3, 3, 11]) == CELL_SIG_NONE[11], "C-shape interior is border-reachable -> OUTSIDE"
    assert int(sig3[3, 3, 10]) == 5, "bbox key still claims containment on the C-shape"
    assert int(sig3[3, 3, 11]) != int(sig3[3, 3, 10]), "col 11 and col 10 must DIFFER on concave enclosure"
    ns = torch.full((S5, S5), 0 + COLOR_OFFSET, dtype=torch.long)
    ns[4, 0] = 3 + COLOR_OFFSET
    ns[4, 9] = 6 + COLOR_OFFSET
    sig4, _ = cell_conditioning_signature(ns.reshape(1, S5 * S5), side=S5, cache=False)
    sig4 = sig4.view(S5, S5, CELL_SIG_DIM)
    assert int(sig4[4, 1, 12]) == 3, "bg cell next to the colour-3 seed -> nearest seed 3"
    assert int(sig4[4, 8, 12]) == 6, "bg cell next to the colour-6 seed -> nearest seed 6"
    assert int(sig4[4, 0, 12]) == 3, "a seed cell's nearest seed is its own colour"
    assert int(sig[9, 9, 11]) == CELL_SIG_NONE[11] and int(sig[9, 9, 12]) == CELL_SIG_NONE[12], (
        "PAD cell must carry sentinels in the FIX H columns")

    # ---------------------------------------------------------------------------------------
    # Part B -- V2 contracts (D1-D6, shared helpers, new columns, cache).
    # ---------------------------------------------------------------------------------------
    # (1) connected_components dtype + diag opt-in.
    dg = torch.full((1, 8, 8), 0, dtype=torch.long)
    for i in range(8):
        dg[0, i, i] = 7
    n4 = len(set(connected_components(dg)[0][dg[0] == 7].tolist()))
    n8 = len(set(connected_components(dg, diag=True)[0][dg[0] == 7].tolist()))
    assert n4 == 8 and n8 == 1, f"diag chain: 4-conn={n4} (want 8), 8-conn={n8} (want 1)"

    # (2) _largest_mask BOTH semantics pinned on ONE grid (the B1 regression guard).
    _lab, cs6 = _component_sizes(grid.unsqueeze(0))
    incl = _largest_mask(cs6, grid.unsqueeze(0), exclude_background=False)[0]
    excl = _largest_mask(cs6, grid.unsqueeze(0), exclude_background=True)[0]
    assert float(incl[5, 5]) == 1.0, "exclude_background=False: background blob wins (legacy)"
    assert float(excl[0, 0]) == 1.0 and float(excl[5, 5]) == 0.0, "exclude_background=True: the block wins"
    assert not torch.equal(incl, excl), "the two is_largest semantics MUST differ (B1 collision made visible)"

    # (3) component_features(False) == object_features (alias equivalence).
    assert torch.equal(component_features(inp, S, exclude_background=False), object_features(inp, S))

    # (4) relational_maps edge cases: all-PAD and single-valid-cell -> finite, PAD rows zero.
    allpad = relational_maps(torch.zeros(1, 36, dtype=torch.long), side=6)
    assert torch.isfinite(allpad).all() and float(allpad.abs().max()) == 0.0, "all-PAD -> zero, finite"
    one = torch.zeros(1, 36, dtype=torch.long); one[0, 14] = 5 + COLOR_OFFSET
    rm_one = relational_maps(one, side=6)
    assert torch.isfinite(rm_one).all(), "single valid cell -> finite"

    # (5) relational_where_hint zero-support contract.
    ti = torch.randint(0, 12, (2, 100))
    z = torch.zeros(2, 2, 100, dtype=torch.long)
    h, info = relational_where_hint(ti, z, z, torch.zeros(2, 2, dtype=torch.bool), side=10)
    assert float(h.abs().max()) == 0.0 and int(info["rel_where_predicate_index"].max()) == -1, "zero-support -> zero hint"

    # (6) diag_boundary predicate expressible where on_boundary cannot: pure diagonal pattern.
    dgrid = torch.full((1, 36), 0 + COLOR_OFFSET, dtype=torch.long)
    dv = dgrid.view(1, 6, 6); dv[0, 2, 2] = 7 + COLOR_OFFSET; dv[0, 3, 3] = 7 + COLOR_OFFSET
    rmap = relational_maps(dgrid, side=6)
    rel = _rel_where_relation_masks(dgrid, rmap)
    assert bool(rel[0, 2 * 6 + 2, RelP.DIAG_BOUNDARY]), "cell with a diagonal-different neighbour -> diag_boundary"

    # (7) overselect_penalty: 0.0 == legacy scores; >0 never raises confidence.
    ci = torch.randint(0, 12, (2, 3, 400)); co = ci.clone()
    mut = torch.rand(2, 3, 400) < 0.1
    co[mut] = torch.randint(2, 12, (int(mut.sum()),))
    cm = torch.ones(2, 3, dtype=torch.bool)
    _, i0 = relational_where_hint(ti[:, :400] if ti.shape[1] >= 400 else torch.randint(0, 12, (2, 400)),
                                  ci, co, cm, side=20)
    tgt = torch.randint(0, 12, (2, 400))
    _, ia = relational_where_hint(tgt, ci, co, cm, side=20)
    _, ib = relational_where_hint(tgt, ci, co, cm, side=20, overselect_penalty=0.1)
    assert bool((ib["rel_where_confidence"] <= ia["rel_where_confidence"] + 1e-6).all()), "penalty never raises conf"

    # (8) nearest-seed tie flag (col 13): equidistant two-colour seeds.
    ns2 = torch.full((10, 10), 0 + COLOR_OFFSET, dtype=torch.long); ns2[4, 1] = 3 + COLOR_OFFSET; ns2[4, 7] = 6 + COLOR_OFFSET
    st, _ = cell_conditioning_signature(ns2.reshape(1, 100), 10, cache=False); st = st.view(10, 10, CELL_SIG_DIM)
    assert int(st[4, 4, 13]) == 1 and int(st[4, 4, 12]) == 3, "midpoint tie=1, col12=smaller colour"
    assert int(st[4, 2, 13]) == 0, "off-midpoint tie=0"

    # (9) touch colour cols 14/15.
    rg = torch.full((10, 10), 0 + COLOR_OFFSET, dtype=torch.long)
    rg[1:6, 1:6] = 5 + COLOR_OFFSET; rg[2:5, 2:5] = 0 + COLOR_OFFSET; rg[2, 3] = 4 + COLOR_OFFSET
    stt, _ = cell_conditioning_signature(rg.reshape(1, 100), 10, cache=False); stt = stt.view(10, 10, CELL_SIG_DIM)
    assert int(stt[1, 1, 14]) == 4 and int(stt[1, 1, 15]) == 1, "frame touches the colour-4 object: mode 4, count 1"
    assert int(stt[2, 2, 14]) == int(stt[1, 1, 14]), "enclosed bg inherits enclosing object's touch cols"
    dgt = torch.full((10, 10), 0 + COLOR_OFFSET, dtype=torch.long); dgt[4, 4] = 7 + COLOR_OFFSET; dgt[5, 5] = 6 + COLOR_OFFSET
    sdd, _ = cell_conditioning_signature(dgt.reshape(1, 100), 10, cache=False); sdd = sdd.view(10, 10, CELL_SIG_DIM)
    assert int(sdd[4, 4, 14]) == CELL_SIG_NONE[14], "diagonal-only contact does NOT count as touch"

    # (10) fast == ref nearest seed on randomized rectangular grids (cols 12+13).
    for _ in range(20):
        H, W = int(torch.randint(3, 12, (1,))), int(torch.randint(3, 12, (1,)))
        cg = torch.randint(0, 4, (H, W)).tolist()
        sf, tf, hf = _sig_nearest_seed_fast(cg, 0, H, W, torch.device("cpu"))
        sr, tr, hr = _sig_nearest_seed_ref(cg, 0, H, W)
        assert hf == hr
        if hf:
            for r in range(H):
                for c in range(W):
                    if cg[r][c] >= 0:
                        assert int(sf[r, c]) == sr[r][c] and bool(tf[r, c]) == tr[r][c], "fast != ref"

    # (11) entry clamp (D3): stray token >= VOCAB == its clamped value.
    raw = torch.full((1, 36), 0 + COLOR_OFFSET, dtype=torch.long); raw[0, 14] = 13
    clp = raw.clone(); clp[0, 14] = 11
    assert torch.equal(cell_conditioning_signature(raw, 6, cache=False)[0],
                       cell_conditioning_signature(clp, 6, cache=False)[0]), "D3 clamp"

    # (12) conditioned_transitions general engine == legacy wrapper with default key.
    assert torch.equal(conditioned_transitions(ci2, co2, cm2, side=S2),
                       object_conditioned_transitions(ci2, co2, cm2, side=S2)), "engine==legacy at key_fn=None"

    # (13) signature cache: hit equal + mutation-safe + bypass.
    _SIG_CACHE.clear()
    gg = torch.randint(0, 12, (2, 100))
    s1, _ = cell_conditioning_signature(gg, 10)
    s2, _ = cell_conditioning_signature(gg, 10)
    assert torch.equal(s1, s2)
    s2[0, 0, 0] = 99
    assert int(cell_conditioning_signature(gg, 10)[0][0, 0, 0]) != 99, "cache clone-on-hit"

    # (14) distance_transform rectangular support (B14: latent square-only bug).
    m = torch.zeros(1, 4, 7, dtype=torch.bool); m[0, 0, 0] = True
    d = distance_transform(m, 4 + 7)
    assert float(d[0, 3, 6]) == 9.0, f"rectangular Manhattan distance wrong: {float(d[0, 3, 6])}"

    print(
        "relation_map self-test PASS -- Part A (legacy contracts) + Part B (14 V2 contracts): "
        "CC long+diag, is_largest both semantics, relmap pad-zero/clearance/containment, signature "
        "big/singleton + enclosure + C-shape, diag_boundary, seed tie flag, touch cols, fast==ref, "
        "D3 clamp, consensus engine==legacy, cache clone-on-hit, rectangular distance."
    )


if __name__ == "__main__":
    print(
        f"relation_map schemas OK: REL_MAP_CHANNELS={REL_MAP_CHANNELS} "
        f"relations={len(REL_WHERE_RELATION_NAMES)} (last={REL_WHERE_RELATION_NAMES[-1]}) "
        f"CELL_SIG_DIM={CELL_SIG_DIM} (cols 13-15: {CELL_SIG_NAMES[13:]}) | "
        f"predicate bank P={10 + len(REL_WHERE_RELATION_NAMES) * 11}"
    )
    _self_test()
