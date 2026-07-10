"""Phase 2b: real connected-component OBJECT features 

Phase 2a used cheap 5x5 proxies and produced the first exact held-out solves (color_exact 0 -> 9.4%,
marginal). This computes TRUE 4-connected same-colour components via bounded label propagation
(GPU-friendly, deterministic, no grad) and the per-cell object features the gate needs to localise
OBJECT-level recolours that a 5x5 window cannot see:

  comp_size_norm  size of this cell's component / grid area   (distinguishes big vs small objects)
  is_largest      is this the largest component?              ("recolour the biggest shape")
  is_singleton    component size == 1                          (isolated pixels)
  on_boundary     has a 4-neighbour of a DIFFERENT colour      (object edge)

Token convention: PAD=0, EOS=1, colour = token-2. Grid is side x side.
"""
from __future__ import annotations

import torch

VOCAB = 12
COLOR_OFFSET = 2
OBJ_DIM = 4
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
)


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


def connected_components(g: torch.Tensor, n_iter: int = 30, n_jump: int = 6) -> torch.Tensor:
    """[B,S,S] colour ids -> [B,S,S] component label (the min flat-index in each 4-connected
    same-colour component).

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
    return labels.float()


def distance_transform(mask: torch.Tensor, max_dist: int) -> torch.Tensor:
    """[B, S, S] boolean mask -> distance to True cells."""
    B, S, S_ = mask.shape
    dist = torch.full((B, S, S), float(max_dist), device=mask.device)
    dist[mask] = 0.0
    for _ in range(max_dist):
        up = torch.roll(dist, 1, dims=1); up[:, 0, :] = max_dist
        dn = torch.roll(dist, -1, dims=1); dn[:, -1, :] = max_dist
        lt = torch.roll(dist, 1, dims=2); lt[:, :, 0] = max_dist
        rt = torch.roll(dist, -1, dims=2); rt[:, :, -1] = max_dist
        dist = torch.min(dist, torch.min(torch.min(up, dn), torch.min(lt, rt)) + 1.0)
    return dist


def _rel_where_relation_masks(tokens: torch.Tensor, rel_maps: torch.Tensor) -> torch.Tensor:
    """Existing relmap facts as boolean predicate masks.

    Shape:
        tokens [..., L]
        rel_maps [..., L, REL_MAP_CHANNELS]
        return [..., L, R]
    """
    valid = tokens.long() >= COLOR_OFFSET
    rm = rel_maps.float()
    comp_size = torch.expm1(rm[..., 2]).clamp_min(0.0)
    # 13-channel layout (was 10): distance_to_edge is now 4 DIRECTIONAL clearances (idx 6..9); the old
    # single "nearest-edge distance" == their per-cell min. solidity/inside_container/nearest_colour
    # live at 10/11/12 (FIXED 2026-07-01 -- the old 7/8/9 indices silently read clearance channels).
    dist_edge = torch.minimum(torch.minimum(rm[..., 6], rm[..., 7]), torch.minimum(rm[..., 8], rm[..., 9]))
    dist_nearest = rm[..., 12]
    masks = [
        valid,
        (rm[..., 1] > 0.5) & valid,
        (rm[..., 3] > 0.5) & valid,
        (rm[..., 4] > 0.5) & valid,
        (rm[..., 5] > 0.5) & valid,
        (dist_edge <= 0.16) & valid,
        (dist_edge >= 0.33) & valid,
        (rm[..., 10] >= 0.99) & valid,
        (rm[..., 10] > 0.0) & (rm[..., 10] < 0.99) & valid,
        (rm[..., 11] > 0.5) & valid,
        (dist_nearest <= 0.10) & valid,
        (dist_nearest >= 0.25) & valid,
        (comp_size > 0.0) & (comp_size <= 4.0) & valid,
        (comp_size >= 5.0) & (comp_size <= 12.0) & valid,
        (comp_size >= 13.0) & valid,
    ]
    return torch.stack(masks, dim=-1)


def _rel_where_candidate_masks(tokens: torch.Tensor, rel_maps: torch.Tensor) -> torch.Tensor:
    """Predicate bank: input-colour, relation, and input-colour+relation conjunctions."""
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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Support-derived WHERE evidence from existing ObjectBank/relmap facts.

    This is a hint, not an executor: it ranks simple predicates by how well they
    explain support changed cells, then applies the best predicate to target input.
    It never reads target output and runs under no_grad.
    """
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


def relational_maps(input_tokens: torch.Tensor, side: int, n_iter: int = 30) -> torch.Tensor:
    """[B,L] -> [B,L,REL_MAP_CHANNELS(=13)] deterministic relational map features (no grad).
    Channels (see REL_MAP_RELATION_NAMES): valid_mask, is_background, comp_size_log, is_largest_non_bg,
    is_singleton, on_boundary, dist_to_top, dist_to_bottom, dist_to_left, dist_to_right (4-way directional
    clearance within the valid bbox -- the "slide to wall" substrate), solidity, inside_container,
    distance_to_nearest_colour.
    """
    with torch.no_grad():
        B, L = input_tokens.shape
        S = side
        dev = input_tokens.device
        g = input_tokens.long().clamp(0, VOCAB - 1).view(B, S, S)
        
        valid_mask = ((g != 0) & (g != 1)).float()
        valid_bool = valid_mask.bool()
        is_bg = (g == COLOR_OFFSET).float() * valid_mask
        
        labels = connected_components(g, n_iter)
        labels_flat = labels.view(B, L).long()
        
        ones = torch.ones(B, L, device=dev)
        size_per_label = torch.zeros(B, L, device=dev).scatter_add(1, labels_flat, ones)
        comp_size = torch.gather(size_per_label, 1, labels_flat).view(B, S, S)
        comp_size_log = torch.log1p(comp_size) * valid_mask
        
        valid_obj_mask = valid_bool & (g != COLOR_OFFSET)
        valid_obj_sizes = comp_size * valid_obj_mask.float()
        max_size = valid_obj_sizes.view(B, L).max(dim=1, keepdim=True).values.clamp_min(1.0).view(B, 1, 1)
        is_largest = ((comp_size == max_size) & valid_obj_mask).float()
        
        is_singleton = (comp_size <= 1.0).float() * valid_obj_mask.float()
        
        diff = torch.zeros(B, S, S, dtype=torch.bool, device=dev)
        for sh, dm in [(1, 1), (-1, 1), (1, 2), (-1, 2)]:
            diff = diff | _in_grid_diff(g, sh, dm)
        on_boundary = diff.float() * valid_mask
        
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
            can_contain = g_flat[b, uniq] > COLOR_OFFSET                          # real colour, not pad/eos/bg
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

        # bg cells: distance to the nearest non-bg object. colour-c cells (c>=1): distance to the
        # nearest OTHER non-bg colour (own channel masked out). pad/eos: saturated, zeroed by the mask.
        dist_to_any_obj = dist_to_c[:, 1:].min(dim=1).values
        idx_c = (g - COLOR_OFFSET).clamp(0, 9)
        own = torch.arange(1, 10, device=dev).view(1, 9, 1, 1) == idx_c.unsqueeze(1)
        min_others = dist_to_c[:, 1:].masked_fill(own, float(S)).min(dim=1).values
        dist_nearest = torch.full((B, S, S), float(S), device=dev)
        dist_nearest = torch.where(g == COLOR_OFFSET, dist_to_any_obj, dist_nearest)
        dist_nearest = torch.where(g > COLOR_OFFSET, min_others, dist_nearest)

        distance_to_nearest_colour = (dist_nearest / float(S)) * valid_mask
        
        maps = torch.stack([
            valid_mask, is_bg, comp_size_log, is_largest, is_singleton, on_boundary,
            dist_top_n, dist_bot_n, dist_left_n, dist_right_n,        # 4-way directional clearance
            solidity, inside_container, distance_to_nearest_colour
        ], dim=-1)

        return maps.view(B, L, REL_MAP_CHANNELS)


# Per-cell CONDITIONING KEY for conditioned value P(out | src, signature). This is a SEPARATE tensor
# from relational_maps -- it does NOT change REL_MAP_CHANNELS or any saved projection, so it carries
# ZERO checkpoint risk (unlike widening the 13-ch relmap). Column 0 is the SOURCE colour (the value
# table's primary key); columns 1..9 are the CONTEXT signature that disambiguates a multi-target
# source (e.g. colour 3 -> red on the largest object, -> green on the singleton). The measured need:
# 52/75 conditional_recolor tasks are multi-target, dominated by background(0) fill, so the signature
# must (a) be universal (neighbour colours work on background cells that own no object) and (b) for a
# background cell, inherit the ENCLOSING object's rank/shape/holes -- the "inside which shape" key.
CELL_SIG_NAMES = (
    "self_color", "nbr4_a", "nbr4_b", "nbr4_c", "nbr4_d",
    "obj_size_rank", "obj_holes", "obj_shape_d4", "local_row3", "local_col3",
    "obj_color",   # colour of the own/enclosing object -- the fill-majority key (bg cell's container colour)
    "encl_color_ff",       # TRUE flood-fill enclosure colour (bg component not reaching border -> modal boundary colour; else 10)
    "nearest_seed_color",  # colour of the nearest non-bg cell by Manhattan distance (own colour for fg; 10 if no seeds)
)
CELL_SIG_NONE = (-1, 10, 10, 10, 10, 8, 4, 7, 3, 3, 10, 10, 10)   # per-column null/out-of-grid sentinel
CELL_SIG_DIM = len(CELL_SIG_NAMES)


def _third(v: int, lo: int, hi: int) -> int:
    """Which third (0=first,1=mid,2=last) of [lo,hi] does v fall in? Robust to lo==hi."""
    span = hi - lo
    if span <= 0:
        return 1
    return min(2, int((v - lo) * 3 // (span + 1)))


def cell_conditioning_signature(input_tokens: torch.Tensor, side: int):
    """[B,L] tokens -> (sig LONG [B,L,CELL_SIG_DIM], valid BOOL [B,L]). Deterministic, no-grad.

    Columns (see CELL_SIG_NAMES): 0 source colour (0-9, -1 pad/eos); 1-4 sorted 4-neighbour colours
    (0-9, 10 out-of-grid/pad); 5 enclosing-object size-rank (0=largest.., 8=none); 6 hole-count
    (0-3, 4=none); 7 D4 shape-signature bucket (0-6 per grid, 7=none); 8-9 local row/col third within
    the object's bbox (0-2, 3=none). Foreground cells use their OWN object; background cells use the
    SMALLEST enclosing object. Reuses object_rule_bank's trusted deterministic extractors."""
    try:  # lazy: avoid the object_bank<->object_rule_bank import cycle (safe once object_bank is loaded)
        from models.recursive_reasoning.object_rule_bank import (
            _compact_colour, _background, _objects, _hole_count, _d4_canon,
        )
    except ModuleNotFoundError:  # standalone (__main__) run: put the repo root on the path, then retry
        import sys as _sys
        from pathlib import Path as _Path
        _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
        from models.recursive_reasoning.object_rule_bank import (
            _compact_colour, _background, _objects, _hole_count, _d4_canon,
        )
    with torch.no_grad():
        B, L = input_tokens.shape
        S = side
        dev = input_tokens.device
        g = input_tokens.long().view(B, S, S)
        colour = g >= COLOR_OFFSET

        # --- vectorized columns 0-4: source colour + sorted 4-neighbour colours (universal, incl bg) ---
        gc = torch.where(colour, g - COLOR_OFFSET, torch.full_like(g, 10))     # pad/eos -> 10
        gp = torch.full((B, S + 2, S + 2), 10, dtype=torch.long, device=dev)   # out-of-grid -> 10
        gp[:, 1:S + 1, 1:S + 1] = gc
        up, dn = gp[:, 0:S, 1:S + 1], gp[:, 2:S + 2, 1:S + 1]
        lt, rt = gp[:, 1:S + 1, 0:S], gp[:, 1:S + 1, 2:S + 2]
        nbr = torch.stack([up, dn, lt, rt], dim=-1).sort(dim=-1).values         # [B,S,S,4] direction-invariant
        sig = torch.empty(B, L, CELL_SIG_DIM, dtype=torch.long, device=dev)
        for k in range(CELL_SIG_DIM):
            sig[:, :, k] = CELL_SIG_NONE[k]
        sig[:, :, 0] = torch.where(colour, g - COLOR_OFFSET, torch.full_like(g, -1)).view(B, L)
        sig[:, :, 1:5] = nbr.reshape(B, L, 4)

        # --- per-grid columns 5-9: object attributes (own object for fg, enclosing object for bg) ---
        for b in range(B):
            col, hw = _compact_colour(input_tokens[b], side)
            if col is None:
                continue
            H, W = hw
            bg = _background(col)
            objs = _objects(col, bg, multi=False)
            if not objs:
                continue
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

            def _write(j: int, i: int, r: int, c: int) -> None:
                o = objs[i]
                sig[b, j, 5] = rank_of[i]
                sig[b, j, 6] = holes[i]
                sig[b, j, 7] = shapes[i]
                sig[b, j, 8] = _third(r, o["rmin"], o["rmax"])
                sig[b, j, 9] = _third(c, o["cmin"], o["cmax"])
                oc = o["colour"]
                sig[b, j, 10] = int(oc) if isinstance(oc, int) else 10          # mono colour; multi-stamp -> none

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

            # --- col 11: TRUE flood-fill enclosure colour ---------------------------------------
            # Connected components of BACKGROUND cells (4-conn). Any bg component touching the
            # compact-grid border is OUTSIDE (sentinel). A non-border-touching bg component is
            # ENCLOSED -- its value is the modal colour among non-bg cells 4-adjacent to it.
            cg = col.tolist()
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
            for rr in range(H):
                for cc in range(W):
                    if cg[rr][cc] == bg:
                        cid = comp_id[rr][cc]
                        sig[b, rr * side + cc, 11] = comp_encl_colour[cid]

            # --- col 12: nearest non-background seed colour (multi-source BFS, Manhattan) -------
            seed_colour = [[-1] * W for _ in range(H)]
            dist = [[-1] * W for _ in range(H)]
            from collections import deque as _deque
            q = _deque()
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = pr + dr, pc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        if dist[nr][nc] == -1:
                            dist[nr][nc] = d + 1
                            seed_colour[nr][nc] = sc
                            q.append((nr, nc))
                        elif dist[nr][nc] == d + 1 and sc < seed_colour[nr][nc]:
                            seed_colour[nr][nc] = sc  # tie -> smaller colour index
            if has_seed:
                for rr in range(H):
                    for cc in range(W):
                        if cg[rr][cc] >= 0:
                            sig[b, rr * side + cc, 12] = seed_colour[rr][cc]

        valid = colour.view(B, L)
        return sig, valid


def object_features(input_tokens: torch.Tensor, side: int, n_iter: int = 30) -> torch.Tensor:
    """[B,L] tokens -> [B,L,OBJ_DIM] deterministic per-cell object features (no grad).

    LEGACY (Phase 2b, consumer: gated-off color_repair_head). CAUTION -- semantic divergence from
    relational_maps: is_largest here does NOT exclude the background component, so on typical grids
    it flags the bg blob (the self-test pins this); relational_maps' is_largest channel EXCLUDES bg.
    Same name, different meaning -- do not swap one for the other when reviving the repair lane."""
    with torch.no_grad():
        B, L = input_tokens.shape
        S = side
        dev = input_tokens.device
        g = input_tokens.long().clamp(0, VOCAB - 1).view(B, S, S)
        labels = connected_components(g, n_iter).view(B, L).long()             # [B,L]
        # component size: scatter-count cells per label, gather back to each cell
        ones = torch.ones(B, L, device=dev)
        size_per_label = torch.zeros(B, L, device=dev).scatter_add(1, labels, ones)
        comp_size = torch.gather(size_per_label, 1, labels)                    # [B,L]
        comp_size_norm = (comp_size / float(L)).view(B, S, S)
        max_size = comp_size.max(dim=1, keepdim=True).values.clamp_min(1.0)
        is_largest = (comp_size == max_size).float().view(B, S, S)
        is_singleton = (comp_size <= 1.0).float().view(B, S, S)
        # on object boundary: any 4-neighbour that is IN-GRID and a DIFFERENT colour
        diff = torch.zeros(B, S, S, dtype=torch.bool, device=dev)
        for sh, dm in [(1, 1), (-1, 1), (1, 2), (-1, 2)]:
            diff = diff | _in_grid_diff(g, sh, dm)
        on_boundary = diff.float()
        feats = torch.stack([comp_size_norm, is_largest, is_singleton, on_boundary], dim=-1)
        return feats.view(B, L, OBJ_DIM)


def is_largest_object(input_tokens: torch.Tensor, side: int, n_iter: int = 30) -> torch.Tensor:
    """[B,L] tokens -> [B,L] bool: is this cell in the largest (by size) connected component?
    Used to CONDITION the recolour rule on object context (Phase 2c), not just on cell colour."""
    with torch.no_grad():
        B, L = input_tokens.shape
        dev = input_tokens.device
        g = input_tokens.long().clamp(0, VOCAB - 1).view(B, side, side)
        labels = connected_components(g, n_iter).view(B, L).long()
        ones = torch.ones(B, L, device=dev)
        size_per_label = torch.zeros(B, L, device=dev).scatter_add(1, labels, ones)
        comp_size = torch.gather(size_per_label, 1, labels)
        max_size = comp_size.max(dim=1, keepdim=True).values.clamp_min(1.0)
        return comp_size == max_size


def is_singleton_object(input_tokens: torch.Tensor, side: int, n_iter: int = 30) -> torch.Tensor:
    """[B,L] tokens -> [B,L] bool: is this cell an isolated single-cell component (size 1)?
    Distinguishes scattered pixels from solid shapes -- the right conditioning when a recolour
    depends on object SIZE (e.g. big block -> b, scattered cells -> c). is_largest is dominated by
    the background and won't separate them."""
    with torch.no_grad():
        B, L = input_tokens.shape
        dev = input_tokens.device
        g = input_tokens.long().clamp(0, VOCAB - 1).view(B, side, side)
        labels = connected_components(g, n_iter).view(B, L).long()
        ones = torch.ones(B, L, device=dev)
        size_per_label = torch.zeros(B, L, device=dev).scatter_add(1, labels, ones)
        comp_size = torch.gather(size_per_label, 1, labels)
        return comp_size <= 1.0


N_SIZE_BUCKETS = 3


def component_size(input_tokens: torch.Tensor, side: int, n_iter: int = 30) -> torch.Tensor:
    """[B,L] tokens -> [B,L] float: size of this cell's connected component.
    LEGACY: no live consumers (kept per the gate-off rule; size_bucket inlines the same math)."""
    with torch.no_grad():
        B, L = input_tokens.shape
        dev = input_tokens.device
        g = input_tokens.long().clamp(0, VOCAB - 1).view(B, side, side)
        labels = connected_components(g, n_iter).view(B, L).long()
        ones = torch.ones(B, L, device=dev)
        size_per_label = torch.zeros(B, L, device=dev).scatter_add(1, labels, ones)
        return torch.gather(size_per_label, 1, labels)


def size_bucket(input_tokens: torch.Tensor, side: int, n_iter: int = 30,
                small_max: int = 8) -> torch.Tensor:
    """[B,L] tokens -> [B,L] long bucket: 0=singleton(size 1), 1=small(2..small_max), 2=large(>small_max).
    The object-context key for the conditioned VALUE prior (Phase 2c)."""
    cs = component_size(input_tokens, side, n_iter)
    bucket = torch.zeros_like(cs, dtype=torch.long)
    bucket = torch.where(cs > 1.0, torch.ones_like(bucket), bucket)
    bucket = torch.where(cs > float(small_max), torch.full_like(bucket, 2), bucket)
    return bucket


def object_conditioned_transitions(context_inputs: torch.Tensor, context_outputs: torch.Tensor,
                                   context_mask: torch.Tensor, side: int,
                                   n_iter: int = 30) -> torch.Tensor:
    """[B,M,L] demos -> [B, K, 10, 10] = per-size-bucket P(out | in=a, CHANGED) consensus (K=3).
    Unlike the cell-colour-only cond_changed, this can express "colour a -> b in a big shape,
    a -> c as a scattered pixel" -- the conditional recolours that cap HEAD changed at dpcc. No grad.
    LEGACY: no live consumers (kept per the gate-off rule; superseded by the relmap-fed color_head)."""
    with torch.no_grad():
        B, M, L = context_inputs.shape
        K = N_SIZE_BUCKETS
        dev = context_inputs.device
        x = context_inputs.long()
        y = context_outputs.long()
        cmb = context_mask.to(torch.bool)
        cooc = torch.zeros(B, K * 100, device=dev)
        for m in range(M):
            xin, yout = x[:, m], y[:, m]
            bucket = size_bucket(xin, side, n_iter)                            # [B,L]
            real = (xin >= COLOR_OFFSET) & (yout >= COLOR_OFFSET)
            changed = (real & (xin != yout) & cmb[:, m].unsqueeze(-1)).float()  # [B,L]
            xc = (xin - COLOR_OFFSET).clamp(0, 9)
            yc = (yout - COLOR_OFFSET).clamp(0, 9)
            flat = (bucket * 100 + xc * 10 + yc).clamp(0, K * 100 - 1)         # [B,L]
            cooc.scatter_add_(1, flat, changed)
        cooc = cooc.view(B, K, 10, 10)
        return cooc / cooc.sum(dim=-1, keepdim=True).clamp_min(1e-6)           # [B,K,10,10]


def _self_test() -> None:
    S = 6
    COLOR_OFFSET = 2
    # grid: a 2x2 block of colour 3 (top-left), a single colour 5 cell, rest background colour 0.
    g = torch.zeros(1, S, S, dtype=torch.long)                                 # all colour 0 (token 2)
    grid = torch.full((S, S), 0 + COLOR_OFFSET, dtype=torch.long)
    grid[0:2, 0:2] = 3 + COLOR_OFFSET                                          # 2x2 block, size 4
    grid[4, 4] = 5 + COLOR_OFFSET                                              # singleton
    inp = grid.reshape(1, S * S)

    labels = connected_components(grid.unsqueeze(0))
    # the 2x2 block must share ONE label; the singleton its own; background one big component
    blk = labels[0, 0:2, 0:2]
    assert (blk == blk[0, 0]).all(), "2x2 block must be one component"
    assert labels[0, 4, 4] != labels[0, 0, 0], "singleton != block"

    feats = object_features(inp, side=S)                                       # [1, 36, OBJ_DIM]
    f = feats[0].view(S, S, OBJ_DIM)
    comp_size_norm, is_largest, is_singleton, on_boundary = (f[..., i] for i in range(OBJ_DIM))
    # block cells: size 4 -> norm 4/36
    assert abs(float(comp_size_norm[0, 0]) - 4.0 / 36.0) < 1e-5, float(comp_size_norm[0, 0])
    # singleton flagged
    assert float(is_singleton[4, 4]) == 1.0 and float(is_singleton[0, 0]) == 0.0
    # background (colour 0) is the largest component
    assert float(is_largest[5, 5]) == 1.0 and float(is_largest[0, 0]) == 0.0
    # a block cell adjacent to background is a boundary; a background cell far from objects is not
    assert float(on_boundary[1, 1]) == 1.0, "block cell touching background must be a boundary"
    assert float(on_boundary[5, 0]) == 0.0, "far background cell is not a boundary"

    # object_conditioned_transitions: colour 3 -> 7 in a LARGE block, 3 -> 5 as a SINGLETON.
    S2 = 8
    gin = torch.full((S2, S2), 0 + COLOR_OFFSET, dtype=torch.long)
    gin[0:3, 0:3] = 3 + COLOR_OFFSET                                           # 9-cell block (large)
    gin[6, 6] = 3 + COLOR_OFFSET                                               # singleton
    gout = gin.clone()
    gout[0:3, 0:3] = 7 + COLOR_OFFSET                                          # block -> 7
    gout[6, 6] = 5 + COLOR_OFFSET                                              # singleton -> 5
    ci2 = gin.reshape(1, 1, S2 * S2)
    co2 = gout.reshape(1, 1, S2 * S2)
    cm2 = torch.ones(1, 1, dtype=torch.bool)
    bk = size_bucket(ci2[:, 0], side=S2)
    assert int(bk[0, 0]) == 2 and int(bk[0, 6 * S2 + 6]) == 0, "block=large(2), singleton=0"
    cond_obj = object_conditioned_transitions(ci2, co2, cm2, side=S2)          # [1,3,10,10]
    assert int(cond_obj[0, 2, 3].argmax()) == 7, "large-bucket: colour 3 -> 7"
    assert int(cond_obj[0, 0, 3].argmax()) == 5, "singleton-bucket: colour 3 -> 5"

    # SNAKY topology: a 30x30 serpentine is ONE component. Bounded propagation without pointer
    # jumping split it into 230 labels (the measured geodesic-horizon bug) -- this assertion
    # protects the fix, and with it every label-derived channel.
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

    # relational_maps CONTRACT on a TRANSLATED canvas: (a) every channel is exactly 0 on PAD/EOS
    # cells (the valid-mask rule -- 'the single most important correctness fix'); (b) directional
    # clearances are bbox-relative; (c) bbox containment fires only for a real enclosing object.
    S4 = 8
    canvas = torch.zeros(S4, S4, dtype=torch.long)                             # PAD
    canvas[1:5, 2:5] = 0 + COLOR_OFFSET                                       # 4x3 valid box at (1,2), bg
    canvas[2, 3] = 4 + COLOR_OFFSET                                           # one object cell
    canvas[5, 2:5] = 1                                                         # EOS row (thin-L)
    canvas[1:5, 5] = 1                                                         # EOS col
    rm = relational_maps(canvas.reshape(1, S4 * S4), side=S4).view(S4, S4, REL_MAP_CHANNELS)
    assert float(rm[canvas < COLOR_OFFSET].abs().max()) == 0.0, "PAD/EOS cells must be all-zero in every channel"
    cell = rm[2, 3]                                                            # bbox rows 1..4 cols 2..4, extent 3
    for ch, want in ((6, 1 / 3), (7, 2 / 3), (8, 1 / 3), (9, 1 / 3)):          # top/bottom/left/right
        assert abs(float(cell[ch]) - want) < 1e-5, f"clearance ch{ch}: {float(cell[ch])} != {want}"
    assert float(rm[..., 11].max()) == 0.0, "no enclosing object -> inside_container must be 0 everywhere"
    ring = torch.full((S4, S4), 0 + COLOR_OFFSET, dtype=torch.long)
    ring[1:5, 1:5] = 6 + COLOR_OFFSET
    ring[2:4, 2:4] = 0 + COLOR_OFFSET                                          # bg hole inside the ring
    ring[2, 2] = 4 + COLOR_OFFSET                                              # object inside the ring
    rm2 = relational_maps(ring.reshape(1, S4 * S4), side=S4).view(S4, S4, REL_MAP_CHANNELS)
    assert float(rm2[2, 2, 11]) == 1.0, "object inside a ring's bbox must be inside_container"
    assert float(rm2[0, 0, 11]) == 0.0, "outside cells must not be inside_container"

    # cell_conditioning_signature CONTRACT: the KEY must separate a MULTI-TARGET source colour that
    # the marginal P(out|src) cannot -- this is the whole point (52/75 conditional_recolor need it).
    S5 = 10
    cs = torch.zeros(S5, S5, dtype=torch.long)                                # colour-0 background
    cs[:] = 0 + COLOR_OFFSET
    cs[1:4, 1:4] = 3 + COLOR_OFFSET                                           # BIG colour-3 object (size 9)
    cs[7, 7] = 3 + COLOR_OFFSET                                               # SMALL colour-3 singleton (size 1)
    cs[9, 9] = 0                                                              # a real PAD cell (token 0, not colour)
    sig, valid = cell_conditioning_signature(cs.reshape(1, S5 * S5), side=S5)
    sig = sig.view(S5, S5, CELL_SIG_DIM); valid = valid.view(S5, S5)
    big_rank = int(sig[1, 1, 5]); small_rank = int(sig[7, 7, 5])
    assert big_rank != small_rank, f"same-colour big vs singleton must get DIFFERENT size-rank ({big_rank} vs {small_rank})"
    assert big_rank == 0 and small_rank > 0, "largest object must be rank 0, singleton a higher rank"
    assert int(sig[1, 1, 0]) == 3 and int(sig[7, 7, 0]) == 3, "source colour column must equal the cell colour (=3)"
    # a value table keyed on (src=3, size_rank) CAN now route big->X, small->Y; marginal on src=3 cannot.
    assert int(sig[0, 0, 0]) == 0, "background(colour-0) is a real source colour -> source column 0"
    assert not bool(valid[9, 9]) and int(sig[9, 9, 0]) == -1, "PAD (token 0) -> not valid, self_color null"
    # background-FILL attribution: a bg cell ENCLOSED by an object inherits that object's rank (the
    # 'inside which shape' key the fill majority needs).
    fr = torch.full((S5, S5), 0 + COLOR_OFFSET, dtype=torch.long)
    fr[1:6, 1:6] = 5 + COLOR_OFFSET                                           # colour-5 frame
    fr[2:5, 2:5] = 0 + COLOR_OFFSET                                           # bg hole inside the frame
    sig2, _ = cell_conditioning_signature(fr.reshape(1, S5 * S5), side=S5)
    sig2 = sig2.view(S5, S5, CELL_SIG_DIM)
    assert int(sig2[3, 3, 5]) != CELL_SIG_NONE[5], "an enclosed background cell must inherit the frame's size-rank"
    assert int(sig2[0, 0, 5]) == CELL_SIG_NONE[5], "an un-enclosed background cell has no object -> none"
    assert int(sig2[3, 3, 10]) == 5, "an enclosed background cell must inherit the frame's COLOUR (=5), the fill key"
    assert int(sig2[0, 0, 10]) == CELL_SIG_NONE[10], "an un-enclosed background cell has no container colour"

    # --- FIX H columns: col 11 (flood-fill enclosure colour) + col 12 (nearest seed colour) ------
    # (a) CLOSED ring: interior bg cell -> col 11 == ring colour; outside bg -> sentinel.
    assert int(sig2[3, 3, 11]) == 5, "flood-fill: bg cell inside a CLOSED colour-5 ring -> encl colour 5"
    assert int(sig2[0, 0, 11]) == CELL_SIG_NONE[11], "flood-fill: border-reachable bg -> sentinel (outside)"
    # (b) C-SHAPE (ring with a gap): the 'interior' is border-reachable through the gap. Flood-fill
    #     (col 11) must say OUTSIDE while the bbox key (col 10) still claims containment -- they must
    #     DIFFER on that cell; this divergence on concave shapes is the whole point of col 11.
    cshape = torch.full((S5, S5), 0 + COLOR_OFFSET, dtype=torch.long)
    cshape[1:6, 1:6] = 5 + COLOR_OFFSET
    cshape[2:5, 2:5] = 0 + COLOR_OFFSET
    cshape[3, 5] = 0 + COLOR_OFFSET                                          # the GAP in the right wall
    sig3, _ = cell_conditioning_signature(cshape.reshape(1, S5 * S5), side=S5)
    sig3 = sig3.view(S5, S5, CELL_SIG_DIM)
    assert int(sig3[3, 3, 11]) == CELL_SIG_NONE[11], "flood-fill: C-shape interior is border-reachable -> OUTSIDE"
    assert int(sig3[3, 3, 10]) == 5, "bbox key still claims containment on the C-shape (the divergence col 11 fixes)"
    assert int(sig3[3, 3, 11]) != int(sig3[3, 3, 10]), "col 11 and col 10 must DIFFER on concave enclosure"
    # (c) nearest-seed colour: two seeds, colour 3 left / colour 6 right; bg cells inherit the nearer.
    ns = torch.full((S5, S5), 0 + COLOR_OFFSET, dtype=torch.long)
    ns[4, 0] = 3 + COLOR_OFFSET
    ns[4, 9] = 6 + COLOR_OFFSET
    sig4, _ = cell_conditioning_signature(ns.reshape(1, S5 * S5), side=S5)
    sig4 = sig4.view(S5, S5, CELL_SIG_DIM)
    assert int(sig4[4, 1, 12]) == 3, "bg cell next to the colour-3 seed -> nearest seed 3"
    assert int(sig4[4, 8, 12]) == 6, "bg cell next to the colour-6 seed -> nearest seed 6"
    assert int(sig4[4, 0, 12]) == 3, "a seed cell's nearest seed is its own colour"
    # (d) PAD cell -> both new cols sentinel, not valid.
    assert int(sig[9, 9, 11]) == CELL_SIG_NONE[11] and int(sig[9, 9, 12]) == CELL_SIG_NONE[12], (
        "PAD cell must carry sentinels in the FIX H columns")

    print(f"object_bank self-test PASS  (2x2 block one comp size {float(comp_size_norm[0,0])*36:.0f}, "
          f"singleton flagged, largest=background, boundary ok, object-conditioned 3->7(large)/3->5(single), "
          f"serpentine=1 component, relmap pad-zero/clearance/containment ok, "
          f"cell-signature separates big/singleton same-colour + attributes enclosing object to bg fill)")


if __name__ == "__main__":
    _self_test()
