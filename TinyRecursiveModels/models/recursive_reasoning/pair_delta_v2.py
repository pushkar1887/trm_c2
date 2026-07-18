"""pair_delta_v2 -- File #5 of the V2 senior-dev rewrite of pair_delta_encoder.py.

LOCKED rules (same as relation_map/core_prior/trm_fvr_v2):
  * NOTHING deleted -- every public symbol of the old file is ported verbatim (byte-identical
    semantics); dead lanes (RuleConditionedDecoder, the unused intent-dict outputs) are KEPT and
    reused, never dropped.
  * The old file `pair_delta_encoder.py` stays UNTOUCHED as the equivalence oracle
    (gate: scripts/pd_v2_gate.py).
  * All NEW behaviour is default-off (new function args / new model flags), appended LAST.

Block map (built block-by-block, each gated):
  Block 0  scripts/pd_v2_gate.py -- oracle harness (positive + negative control).
  Block 1  THIS skeleton: SS1 schema + SS3 verbatim deterministic port (demo_delta_features,
           pairdelta_intent_features). Gate: byte-equal vs oracle on random + ARC-like + edge batches.
  Block 2  SS4 verbatim learned port (PairDeltaEncoder, RuleConditionedDecoder; same param names ->
           checkpoint-compatible) + SS2 scatter kernel (fast path, allclose-gated, default-off).
  Block 3  SS5 pd_color_evidence  [B,L,PD_COLOR_DIM] (demo-agreement VALUE/WHERE + positional prior).
  Block 4  SS6 pd_structure_evidence [B,L,PD_STRUCT_DIM] (preserve/transpose/bbox extent masks).
  Block 5  consumer wiring in trm_fvr_v2 (import flip + EVIDENCE_COLS append + structure proj).
  Block 6  driver flags + offline probes.

WHY a rewrite (the measured diagnosis, see plan in session log):
  The old file computes a rich per-demo dict and exports ONE scalar (conditional_recolor_score) as
  evidence; it is strictly uni-directional (x->y only) and position-blind (bag-of-cells: histograms
  and rates over the flattened grid -- it cannot see H vs W, so transpose is invisible, and it cannot
  say WHERE changes happen). The V2 blocks add the missing axes (2D position, H/W kinematics,
  cross-demo agreement) as OUTPUT-side evidence columns; the verbatim lanes stay byte-identical.

Token convention (unchanged): PAD=0, EOS=1, color = token-2 (colors 0..9). Grid is 30x30=900 tokens.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================================
# SS1 SCHEMA -- constants verbatim from the oracle + NAMED layouts (the M4 pattern).
#     The named layouts are documentation-grade constants only: they pin the meaning of
#     every column so consumers stop hand-counting offsets. No behaviour.
# =====================================================================================
PAD_TOKEN = 0
EOS_TOKEN = 1
COLOR_OFFSET = 2
N_COLORS = 10
N_TRANSITIONS = N_COLORS * N_COLORS  # 100
GRID_SIDE = 30
GRID_LEN = GRID_SIDE * GRID_SIDE     # 900
VOCAB = 12                            # PAD, EOS, 10 colors

# demo_delta_features layout [B, M, 126] (was implicit in the oracle's cat order):
DDF_TRANS = 0                         # [0:100)   changed-cell transition histogram (row-major src*10+dst)
DDF_IN_HIST = 100                     # [100:110) input colour histogram
DDF_OUT_HIST = 110                    # [110:120) output colour histogram
DDF_SCALARS = 120                     # [120:126) changed_rate, area_ratio/5, in_nonbg*, out_nonbg*,
                                      #           add_rate, del_rate
                                      #   (*) NOTE (M-d): "nonbg" is a historical misnomer -- it is the
                                      #   coloured-cell rate INCLUDING colour 0 (ARC black background).
                                      #   Kept verbatim; background-aware variants are a SS7 extension.
FEATURE_DIM = N_TRANSITIONS + N_COLORS + N_COLORS + 6  # 126 (defined early; oracle defines it late)

# Optional spatial-delta branch. These columns do NOT alter FEATURE_DIM or the legacy pair_mlp
# checkpoint contract; PairDeltaEncoder fuses them through a separate zero-init residual only when
# explicitly enabled.
PDS_DY = 0
PDS_DX = 1
PDS_DIRECTION_CONSISTENCY = 2
PDS_BBOX_DH = 3
PDS_BBOX_DW = 4
PDS_SAME_SHAPE_TRANSPORT = 5
PDS_CREATION_CONFIDENCE = 6
PDS_DELETION_CONFIDENCE = 7
SPATIAL_DELTA_DIM = 8
SPATIAL_DELTA_NAMES = (
    "dominant_dy", "dominant_dx", "direction_consistency", "bbox_dh", "bbox_dw",
    "same_shape_transport", "creation_confidence", "deletion_confidence",
)

# pairdelta_intent_features dict keys, split by CURRENT consumer status (Block 5 wires the dead ones):
INTENT_KEYS_LIVE = ("feature",)                                   # the 1 evidence scalar
INTENT_KEYS_METRICS = ("conditional_recolor_score", "global_recolor_score",
                       "shape_preserved", "changed_rate")         # logged means only
INTENT_KEYS_DEAD = ("dominant_source_color", "dominant_target_color",
                    "changed_cells", "valid_cells")               # computed, never consumed (M-b)


# =====================================================================================
# SS2 KERNELS -- the ONE scatter-based transition-count engine (M-c).
#     Same statistic as the oracle's F.one_hot(pair,100)*mask, WITHOUT the O(B*M*L*100)
#     intermediate. Float accumulation order differs from the one_hot sum, so this is
#     allclose-equal, NOT byte-equal -- which is why every verbatim path keeps the one_hot
#     math as its DEFAULT and takes this kernel only under an explicit fast=True.
#     (trm_fvr_v2's M3 `_changed_transition_counts` is the pooled [B,100] sibling; this is
#     the grouped form -- per-demo [B,M,100] or pooled -- so SS5/SS6 build on one engine.)
# =====================================================================================
def transition_counts(
    mask: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, per_demo: bool = True,
) -> torch.Tensor:
    """Counts of src->dst over cells where `mask` is True.

    mask/src/dst: [B, M, L] (src/dst are 0..9 colour ids). Returns [B, M, 100] when
    per_demo else [B, 100] (row-major src*10+dst, same layout as DDF_TRANS).
    """
    pair = (src * N_COLORS + dst).clamp(0, N_TRANSITIONS - 1)
    if per_demo:
        B, M, _ = pair.shape
        flat = pair.reshape(B * M, -1)
        counts = torch.zeros((B * M, N_TRANSITIONS), device=pair.device, dtype=torch.float32)
        counts.scatter_add_(1, flat, mask.reshape(B * M, -1).to(torch.float32))
        return counts.view(B, M, N_TRANSITIONS)
    B = pair.shape[0]
    flat = pair.reshape(B, -1)
    counts = torch.zeros((B, N_TRANSITIONS), device=pair.device, dtype=torch.float32)
    counts.scatter_add_(1, flat, mask.reshape(B, -1).to(torch.float32))
    return counts


# =====================================================================================
# SS3 VERBATIM DETERMINISTIC PORT -- byte-identical to pair_delta_encoder.py.
#     Do not "improve" anything here; improvements live in SS5/SS6/SS7 behind new names.
#     (`fast=True` is the ONLY additive knob: it swaps the histogram inner loop for the
#     SS2 kernel -- allclose-equal, default OFF, oracle path untouched.)
# =====================================================================================
def _color_grid(tokens: torch.Tensor) -> torch.Tensor:
    """token grid -> color id (0..9) where colored, else -1 (pad/eos)."""
    is_color = tokens >= COLOR_OFFSET
    return torch.where(is_color, (tokens - COLOR_OFFSET).clamp(0, 9),
                       torch.full_like(tokens, -1))


def demo_delta_features(context_inputs: torch.Tensor, context_outputs: torch.Tensor,
                        context_mask: torch.Tensor, fast: bool = False,
                        include_identity: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-demo explicit transformation features.

    Returns:
        feats [B, M, F]   (F = 100 transition + 10 in-hist + 10 out-hist + 6 scalars = 126)
        valid [B, M]      bool: demo has at least one changed colored cell

    VERBATIM oracle math. Known limitation kept as-is: a no-change (identity) demo is marked
    INVALID (valid=False) -- the encoder cannot represent "the rule is copy" (SS7 extension).
    The F.one_hot(pair,100)*changed intermediate is O(B*M*L*100) (M-c); the scatter fast path
    lands in Block 2 as a default-off arg because float accumulation order differs (allclose,
    not byte-equal).
    """
    x = context_inputs.long()
    y = context_outputs.long()
    cm = context_mask.to(torch.bool)
    xs = x >= COLOR_OFFSET
    ys = y >= COLOR_OFFSET
    real = xs & ys
    changed = real & (x != y) & cm.unsqueeze(-1)              # [B,M,L]
    xc = (x - COLOR_OFFSET).clamp(0, 9)
    yc = (y - COLOR_OFFSET).clamp(0, 9)

    # 1) transition histogram (changed cells)
    if fast:
        trans = transition_counts(changed, xc, yc, per_demo=True)          # SS2 kernel (M-c)
    else:
        pair = (xc * N_COLORS + yc).clamp(0, N_TRANSITIONS - 1)
        trans = (F.one_hot(pair, N_TRANSITIONS).float() * changed.unsqueeze(-1).float()).sum(2)
    trans = trans / (trans.sum(-1, keepdim=True) + 1e-6)      # [B,M,100]

    # 2) input/output color histograms (over colored cells, masked)
    in_oh = F.one_hot(xc, N_COLORS).float() * (xs & cm.unsqueeze(-1)).unsqueeze(-1).float()
    out_oh = F.one_hot(yc, N_COLORS).float() * (ys & cm.unsqueeze(-1)).unsqueeze(-1).float()
    in_hist = in_oh.sum(2); in_hist = in_hist / (in_hist.sum(-1, keepdim=True) + 1e-6)
    out_hist = out_oh.sum(2); out_hist = out_hist / (out_hist.sum(-1, keepdim=True) + 1e-6)

    # 3) scalar deltas
    in_area = (xs & cm.unsqueeze(-1)).float().sum(-1)         # [B,M]
    out_area = (ys & cm.unsqueeze(-1)).float().sum(-1)
    changed_rate = changed.float().sum(-1) / (real & cm.unsqueeze(-1)).float().sum(-1).clamp_min(1)
    area_ratio = out_area / in_area.clamp_min(1)
    in_nonbg = in_area / GRID_LEN
    out_nonbg = out_area / GRID_LEN
    # crude bbox-width proxy via per-row/col presence is expensive; use area-derived shape
    # delta surrogate: sqrt(area) difference (cheap, differentiable-free)
    dshape = (out_area.sqrt() - in_area.sqrt()).abs() / GRID_SIDE
    add_rate = ((~xs) & ys & cm.unsqueeze(-1)).float().sum(-1) / GRID_LEN   # cells that became colored
    del_rate = (xs & (~ys) & cm.unsqueeze(-1)).float().sum(-1) / GRID_LEN   # cells that became blank
    scalars = torch.stack([changed_rate, area_ratio.clamp(0, 5) / 5.0,
                           in_nonbg, out_nonbg, add_rate, del_rate], dim=-1)  # [B,M,6]

    feats = torch.cat([trans, in_hist, out_hist, scalars], dim=-1)           # [B,M,126]
    valid = ((real.any(dim=-1) if include_identity else changed.any(dim=-1)) & cm)  # [B,M]
    return feats, valid


def pairdelta_intent_features(
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
    context_mask: torch.Tensor,
    fast: bool = False,
) -> Dict[str, torch.Tensor]:
    """Cheap PairDelta intent diagnostics, not a solver.

    Returns per-task scalars used as evidence/router hints. They are computed
    directly from support input->output deltas and do not write logits.

    VERBATIM oracle math. Known issues documented (NOT fixed here -- old columns must stay
    byte-identical; the strict versions are SS5/SS7 additions):
      * M-a: `shape_preserved` averages structural equality over ALL 900 tokens, so it is
        pad-diluted (a 3x3->5x5 size change still scores ~0.97 "shape same").
      * M-b: only `feature` (== conditional_recolor_score) is consumed as evidence; the
        global/dominant outputs are dead until Block 5 wires them.
      * `dshape` is unused here; `del_rate`/`add_rate` say how much, never WHERE or WHICH colour.
    """
    with torch.no_grad():
        x = context_inputs.long()
        y = context_outputs.long()
        cm = context_mask.to(torch.bool)
        xs = x >= COLOR_OFFSET
        ys = y >= COLOR_OFFSET
        valid = xs & ys & cm.unsqueeze(-1)
        changed = valid & (x != y)
        active = cm.float().sum(dim=1).clamp_min(1.0)
        valid_counts = valid.float().sum(dim=(1, 2)).clamp_min(1.0)
        changed_counts = changed.float().sum(dim=(1, 2))
        changed_rate = changed_counts / valid_counts

        x_struct = torch.where(xs, torch.full_like(x, 2), torch.where(x == EOS_TOKEN, torch.ones_like(x), torch.zeros_like(x)))
        y_struct = torch.where(ys, torch.full_like(y, 2), torch.where(y == EOS_TOKEN, torch.ones_like(y), torch.zeros_like(y)))
        shape_same_per_demo = ((x_struct == y_struct).float().mean(dim=-1) * cm.float()).sum(dim=1) / active
        non_empty = (changed_counts > 0).float()
        sparse = ((changed_rate > 0.0) & (changed_rate < 0.5)).float()

        B = x.shape[0]
        xc = (x - COLOR_OFFSET).clamp(0, 9)
        yc = (y - COLOR_OFFSET).clamp(0, 9)
        if fast:
            counts = transition_counts(changed, xc, yc, per_demo=False).view(B, N_COLORS, N_COLORS)
        else:
            # (verbatim oracle math, incl. the dead zeros-init it overwrote -- M-d class, dropped here
            #  as it was provably unread; the one_hot path itself stays byte-identical)
            pair = (xc * N_COLORS + yc).clamp(0, N_TRANSITIONS - 1)
            pair_oh = F.one_hot(pair, N_TRANSITIONS).float() * changed.unsqueeze(-1).float()
            counts = pair_oh.sum(dim=(1, 2)).view(B, N_COLORS, N_COLORS)
        total = counts.sum(dim=(1, 2))
        row_peaks = counts.max(dim=2).values.sum(dim=1)
        global_consistency = row_peaks / total.clamp_min(1.0)
        src_mass = counts.sum(dim=2)
        dominant_source = src_mass.argmax(dim=1)
        dominant_target = counts[torch.arange(B, device=x.device), dominant_source].argmax(dim=1)

        conditional_score = shape_same_per_demo * non_empty * sparse * (1.0 - changed_rate.clamp(0.0, 1.0))
        global_score = shape_same_per_demo * non_empty * global_consistency
        feature = conditional_score.unsqueeze(-1)
        return {
            "feature": feature,
            "shape_preserved": shape_same_per_demo,
            "changed_rate": changed_rate,
            "dominant_source_color": dominant_source.to(torch.float32),
            "dominant_target_color": dominant_target.to(torch.float32),
            "conditional_recolor_score": conditional_score,
            "global_recolor_score": global_score,
            "changed_cells": changed_counts,
            "valid_cells": valid_counts,
        }


def spatial_delta_features(
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
    context_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-demo object transport evidence for the existing PairDelta encoder.

    Returns normalized ``[B,M,8]`` features and a ``[B,M]`` extraction-valid mask. Object
    correspondence is colour-agnostic, so a slide-plus-recolour demo still yields movement. This
    function supplies evidence only; it never renders or proposes an output grid.
    """
    if context_inputs.ndim != 3 or context_outputs.shape != context_inputs.shape:
        raise ValueError("context_inputs/context_outputs must have shape [B,M,L]")
    B, M, L = context_inputs.shape
    side = int(round(L ** 0.5))
    if side * side != L:
        raise ValueError(f"PairDelta spatial features require a square token canvas, got L={L}")
    device = context_inputs.device
    out = torch.zeros((B, M, SPATIAL_DELTA_DIM), device=device, dtype=torch.float32)
    extracted = torch.zeros((B, M), device=device, dtype=torch.bool)
    cm = context_mask.to(device=device, dtype=torch.bool)
    ci = context_inputs.detach().to(device="cpu", dtype=torch.long)
    co = context_outputs.detach().to(device="cpu", dtype=torch.long)
    scale = float(max(side - 1, 1))

    # core_prior owns object matching. Keeping this import local avoids a module cycle and ensures
    # PairDelta and the rule-factor diagnostics use one correspondence definition.
    from models.recursive_reasoning.core_prior import evidence_rule_factors

    with torch.no_grad():
        for b in range(B):
            for m in cm[b].nonzero(as_tuple=True)[0].tolist():
                try:
                    factors = evidence_rule_factors(ci[b, m:m + 1], co[b, m:m + 1], side)
                except (AssertionError, RuntimeError, TypeError, ValueError):
                    continue
                out[b, m] = torch.tensor((
                    float(factors["dominant_dy"]) / scale,
                    float(factors["dominant_dx"]) / scale,
                    float(factors["direction_consistency"]),
                    float(factors["bbox_dh"]) / scale,
                    float(factors["bbox_dw"]) / scale,
                    float(factors["same_shape_transport"]),
                    float(factors["creation_confidence"]),
                    float(factors["deletion_confidence"]),
                ), device=device, dtype=torch.float32)
                extracted[b, m] = True
    return out, extracted


# =====================================================================================
# SS4 VERBATIM LEARNED PORT -- parameter names IDENTICAL to the oracle's modules, so an
#     oracle state_dict loads strict=True (checkpoint-compatible both ways). Forward math
#     verbatim. Additive default-off knobs only:
#       * PairDeltaEncoder(fast=...): threads the SS2 kernel into demo_delta_features.
#       * RuleConditionedDecoder(grid_len=...): the oracle hard-codes pos_embed at
#         GRID_LEN=900 and ASSERTS on any other length; grid_len generalizes the probe to
#         other grid sizes (default 900 => byte-identical params + behaviour). This is the
#         reuse lane: the decoder becomes the OFFLINE evidence-probe head (Blocks 3/6).
# =====================================================================================
class PairDeltaEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, n_slots: int = 8, n_heads: int = 4,
                 fast: bool = False, include_identity: bool = False,
                 include_spatial: bool = False):
        super().__init__()
        self.D = hidden_dim
        self.K = n_slots
        self.fast = bool(fast)                     # SS2 fast path (allclose, default off)
        self.include_identity = bool(include_identity)
        self.include_spatial = bool(include_spatial)
        self.pair_mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        if self.include_spatial:
            self.spatial_mlp = nn.Sequential(
                nn.Linear(SPATIAL_DELTA_DIM, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
            )
            # Exact step-0 no-op: old checkpoints and the 126-column colour branch retain their
            # behaviour while the dedicated spatial parameters can learn at the PairDelta LR.
            nn.init.zeros_(self.spatial_mlp[-1].weight)
        self.slot_queries = nn.Parameter(torch.randn(n_slots, hidden_dim) * 0.02)
        self.slot_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.confidence = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                        nn.Linear(hidden_dim, 1))

    def forward(self, context_inputs, context_outputs, context_mask) -> Dict[str, torch.Tensor]:
        feats, valid = demo_delta_features(context_inputs, context_outputs, context_mask,
                                           fast=self.fast, include_identity=self.include_identity)
        B, M, _ = feats.shape
        d = self.pair_mlp(feats)                                  # [B,M,D]
        spatial_norm = torch.zeros((), device=d.device, dtype=torch.float32)
        spatial_valid = torch.zeros_like(valid)
        if self.include_spatial:
            spatial, spatial_valid = spatial_delta_features(
                context_inputs, context_outputs, context_mask)
            d = d + self.spatial_mlp(spatial.to(d.dtype)) * spatial_valid.unsqueeze(-1).to(d.dtype)
            denom = spatial_valid.float().sum().clamp_min(1.0)
            spatial_norm = (
                spatial.float().norm(dim=-1) * spatial_valid.float()).sum() / denom
        # masked slot attention: slots (query) attend over per-demo features (key/value)
        q = self.slot_queries.unsqueeze(0).expand(B, -1, -1)      # [B,K,D]
        kpm = ~valid                                             # True = mask out
        all_masked = kpm.all(dim=-1, keepdim=True)
        kpm = torch.where(all_masked, torch.zeros_like(kpm), kpm)
        slots, _ = self.slot_attn(q, d, d, key_padding_mask=kpm)  # [B,K,D]
        slots = self.slot_norm(slots)
        # rule vector = mean slot; confidence from it
        rule_vec = slots.mean(dim=1)                              # [B,D]
        conf = torch.sigmoid(self.confidence(rule_vec)).squeeze(-1)  # [B]
        # The kpm all-masked flip above only avoids the attention NaN; the slots it produces are over
        # a fully-masked (garbage) key set. Zero the rule + confidence for those rows so a consumer
        # cannot silently read a confident-looking rule for an empty task (emptiness is now explicit
        # in BOTH rule_confidence==0 AND demo_valid, not demo_valid alone).
        empty = all_masked.squeeze(-1)                           # [B]
        slots = torch.where(empty.view(B, 1, 1), torch.zeros_like(slots), slots)
        rule_vec = torch.where(empty.unsqueeze(-1), torch.zeros_like(rule_vec), rule_vec)
        conf = torch.where(empty, torch.zeros_like(conf), conf)
        return {"rule_slots": slots, "rule_vec": rule_vec,
                "rule_confidence": conf, "demo_valid": valid,
                "spatial_valid": spatial_valid,
                "spatial_feature_norm": spatial_norm.detach()}


class RuleConditionedDecoder(nn.Module):
    """Phase-A probe decoder: predict y_j from x_j conditioned on rule_slots.

    Dead in the live model (no consumer) -- KEPT as the offline evidence-probe head: train it
    briefly on frozen LODO batches with/without a new evidence block to measure separation
    BEFORE anything touches the real model (File #5 reuse lane #3).
    """

    def __init__(self, hidden_dim: int = 256, n_heads: int = 4, n_layers: int = 2,
                 grid_len: int = GRID_LEN):
        super().__init__()
        self.tok_embed = nn.Embedding(VOCAB, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, int(grid_len), hidden_dim) * 0.02)
        self.cross = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        enc = nn.TransformerEncoderLayer(hidden_dim, n_heads, dim_feedforward=hidden_dim * 2,
                                         batch_first=True, activation="gelu")
        self.body = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(hidden_dim, VOCAB)

    def forward(self, query_input: torch.Tensor, rule_slots: Optional[torch.Tensor]) -> torch.Tensor:
        assert query_input.shape[-1] == self.pos_embed.shape[1], (
            f"RuleConditionedDecoder.pos_embed is fixed at length {self.pos_embed.shape[1]}; "
            f"got query_input length {query_input.shape[-1]}. Build with grid_len= for other sizes.")
        h = self.tok_embed(query_input.long()) + self.pos_embed          # [B,grid_len,D]
        if rule_slots is not None:
            crossed, _ = self.cross(h, rule_slots, rule_slots)
            h = self.cross_norm(h + crossed)
        h = self.body(h)
        return self.head(h)                                              # [B,grid_len,VOCAB]


# =====================================================================================
# SS5 PD-COLOR EVIDENCE (Block 3, NEW) -- per-cell colour WHERE/VALUE from the axes the
#     pooled tables cannot see: CROSS-DEMO AGREEMENT (every existing statistic pools
#     counts across demos, so one big demo outvotes three small ones) and 2D POSITION
#     (the old file is a bag-of-cells). Deterministic, no params, no_grad, evidence-only.
#     LODO safety is the CALLER's job: pass the _active_context_* tensors.
#
#     Layout [B, L, PD_COLOR_DIM=14] (PDC_* offsets):
#       [0:10) consensus  P(dst | src=cell colour) counting each transition ONCE PER DEMO
#              (presence, not cell count), normalized by #demos whose INPUT contains src.
#              "4->7 in every demo that has a 4" => 1.0 at dst 7.
#       [10]   min_change_rate  min over demos-containing-src of that demo's
#              P(change | src) -- "this colour changes in EVERY demo at least this much".
#       [11]   src_support  #demos containing src / #valid demos (the confidence for the
#              two agreement channels; single-demo tasks are trivially 'consistent').
#       [12]   row_prior    positional change prior for the cell's row band
#       [13]   col_prior    ... col band. Per-demo P(change | band within the demo's own
#              extent), cross-demo masked mean x consistency (1 - max-min spread). Target
#              cells are banded within the TARGET's extent (extent-relative, so a 5x5 demo
#              teaches "top row changes" to a 20x20 target).
#     All 14 columns are zeroed on PAD/EOS target cells.
# =====================================================================================
PD_COLOR_DIM = 14
PDC_CONSENSUS = 0
PDC_MIN_CHANGE = 10
PDC_SUPPORT = 11
PDC_ROW_PRIOR = 12
PDC_COL_PRIOR = 13
PD_POS_BANDS = 6


def _grid_extents(tokens: torch.Tensor, side: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """[..., L] token grids -> (h, w) [...]: extent of the COLOUR region (canonical ARC layout
    puts the grid top-left). 0 when the grid has no colour cells."""
    cmask = tokens >= COLOR_OFFSET
    rows = (torch.arange(tokens.shape[-1], device=tokens.device) // side) + 1
    cols = (torch.arange(tokens.shape[-1], device=tokens.device) % side) + 1
    h = (rows * cmask.long()).amax(dim=-1)
    w = (cols * cmask.long()).amax(dim=-1)
    return h, w


def _content_bbox(tokens: torch.Tensor, side: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """(h, w) of the NON-BACKGROUND content bbox (colour != 0, i.e. token > 2), anchored at
    the origin like the canonical layout. 0 when there is no non-background cell."""
    nb = tokens > COLOR_OFFSET
    rows = (torch.arange(tokens.shape[-1], device=tokens.device) // side) + 1
    cols = (torch.arange(tokens.shape[-1], device=tokens.device) % side) + 1
    h = (rows * nb.long()).amax(dim=-1)
    w = (cols * nb.long()).amax(dim=-1)
    return h, w


def pd_color_evidence(
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
    context_mask: torch.Tensor,
    target_inputs: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """ci/co [B,M,L], cm [B,M], target [B,L] -> (features [B,L,14], stats dict of scalars)."""
    B, M, L = context_inputs.shape
    device = target_inputs.device
    side = int(L ** 0.5)
    feats = torch.zeros((B, L, PD_COLOR_DIM), device=device, dtype=torch.float32)
    stats = {
        "pd_color_consensus_mass": torch.zeros((), device=device),
        "pd_color_min_change": torch.zeros((), device=device),
        "pd_color_pos_prior": torch.zeros((), device=device),
    }
    if side * side != L or M == 0:
        return feats, stats
    with torch.no_grad():
        x = context_inputs.long().to(device)
        y = context_outputs.long().to(device)
        t = target_inputs.long().to(device)
        demo_ok = context_mask.to(device=device, dtype=torch.bool)
        xs = x >= COLOR_OFFSET
        valid = xs & (y >= COLOR_OFFSET) & demo_ok.unsqueeze(-1)          # [B,M,L]
        changed = valid & (x != y)
        src = (x - COLOR_OFFSET).clamp(0, 9)
        dst = (y - COLOR_OFFSET).clamp(0, 9)

        # ---- cross-demo agreement (VALUE + WHERE-by-colour) --------------------------------
        per_demo = transition_counts(changed, src, dst, per_demo=True).view(B, M, 10, 10)
        presence = (per_demo > 0).float()                                  # [B,M,10,10]
        src_flat = src.reshape(B * M, L)
        src_valid = torch.zeros((B * M, 10), device=device, dtype=torch.float32)
        src_valid.scatter_add_(1, src_flat, valid.reshape(B * M, L).float())
        src_changed = torch.zeros_like(src_valid)
        src_changed.scatter_add_(1, src_flat, changed.reshape(B * M, L).float())
        src_valid = src_valid.view(B, M, 10)
        src_changed = src_changed.view(B, M, 10)
        demo_has_src = src_valid > 0                                       # [B,M,10]
        n_demo_src = demo_has_src.float().sum(dim=1)                       # [B,10]
        n_valid_demos = demo_ok.float().sum(dim=1).clamp_min(1.0)          # [B]

        consensus = presence.sum(dim=1) / n_demo_src.clamp_min(1.0).unsqueeze(-1)   # [B,10,10]
        consensus = consensus * (n_demo_src > 0).float().unsqueeze(-1)

        p_change_demo = src_changed / src_valid.clamp_min(1.0)             # [B,M,10]
        p_masked = torch.where(demo_has_src, p_change_demo, torch.full_like(p_change_demo, 2.0))
        min_rate = p_masked.amin(dim=1).clamp(0.0, 1.0)                    # [B,10]
        min_rate = min_rate * (n_demo_src > 0).float()
        support_frac = n_demo_src / n_valid_demos.unsqueeze(-1)            # [B,10]

        # ---- positional prior (WHERE-by-position, extent-relative bands) -------------------
        idx = torch.arange(L, device=device)
        row_i = (idx // side).view(1, 1, L)
        col_i = (idx % side).view(1, 1, L)
        dh, dw = _grid_extents(x, side)                                    # [B,M] demo INPUT extents
        row_band = (row_i * PD_POS_BANDS // dh.clamp_min(1).unsqueeze(-1)).clamp(0, PD_POS_BANDS - 1)
        col_band = (col_i * PD_POS_BANDS // dw.clamp_min(1).unsqueeze(-1)).clamp(0, PD_POS_BANDS - 1)

        def band_prior(band: torch.Tensor) -> torch.Tensor:
            """[B,M,L] band ids -> [B, PD_POS_BANDS] masked-mean x consistency prior."""
            bt = torch.zeros((B * M, PD_POS_BANDS), device=device, dtype=torch.float32)
            bc = torch.zeros_like(bt)
            bflat = band.expand(B, M, L).reshape(B * M, L)
            bt.scatter_add_(1, bflat, valid.reshape(B * M, L).float())
            bc.scatter_add_(1, bflat, changed.reshape(B * M, L).float())
            bt = bt.view(B, M, PD_POS_BANDS)
            bc = bc.view(B, M, PD_POS_BANDS)
            has = bt > 0                                                   # demo contributes to band
            p = bc / bt.clamp_min(1.0)
            n = has.float().sum(dim=1)                                     # [B,NB]
            mean = (p * has.float()).sum(dim=1) / n.clamp_min(1.0)
            pmax = torch.where(has, p, torch.full_like(p, -1.0)).amax(dim=1)
            pmin = torch.where(has, p, torch.full_like(p, 2.0)).amin(dim=1)
            consistency = (1.0 - (pmax - pmin)).clamp(0.0, 1.0)
            return mean * consistency * (n > 0).float()                    # [B,NB]

        row_prior = band_prior(row_band)
        col_prior = band_prior(col_band)

        th, tw = _grid_extents(t, side)                                    # [B]
        t_row_band = ((idx // side).view(1, L) * PD_POS_BANDS // th.clamp_min(1).unsqueeze(-1)).clamp(0, PD_POS_BANDS - 1)
        t_col_band = ((idx % side).view(1, L) * PD_POS_BANDS // tw.clamp_min(1).unsqueeze(-1)).clamp(0, PD_POS_BANDS - 1)

        # ---- gather to target cells ---------------------------------------------------------
        tc = (t - COLOR_OFFSET).clamp(0, 9)                                # [B,L]
        tvalid = (t >= COLOR_OFFSET).float().unsqueeze(-1)                 # [B,L,1]
        feats[..., PDC_CONSENSUS:PDC_CONSENSUS + 10] = consensus.gather(
            1, tc.unsqueeze(-1).expand(-1, -1, 10))
        feats[..., PDC_MIN_CHANGE] = min_rate.gather(1, tc)
        feats[..., PDC_SUPPORT] = support_frac.gather(1, tc)
        feats[..., PDC_ROW_PRIOR] = row_prior.gather(1, t_row_band)
        feats[..., PDC_COL_PRIOR] = col_prior.gather(1, t_col_band)
        feats = feats * tvalid

        nv = tvalid.sum().clamp_min(1.0)
        stats["pd_color_consensus_mass"] = feats[..., PDC_CONSENSUS:PDC_CONSENSUS + 10].sum() / nv
        stats["pd_color_min_change"] = feats[..., PDC_MIN_CHANGE].sum() / nv.squeeze()
        stats["pd_color_pos_prior"] = (
            feats[..., PDC_ROW_PRIOR].sum() + feats[..., PDC_COL_PRIOR].sum()) / (2.0 * nv.squeeze())
    return feats, stats


# =====================================================================================
# SS6 PD-STRUCT EVIDENCE (Block 4, NEW) -- per-cell STRUCTURE evidence from demo shape
#     kinematics: the model's `_predicted_extent` engine only knows {identity, constant,
#     ratio}; TRANSPOSE (rot90: H<->W swap) and CROP-TO-BBOX output shapes are
#     unrepresentable there. Same ordered-verify ethos: a family fires ONLY if it exactly
#     reconstructs the output extent of EVERY valid demo (and is non-degenerate: at least
#     one demo distinguishes it from plain preserve).
#
#     Layout [B, L, PD_STRUCT_DIM=6] (PDS_* offsets); each family contributes its predicted
#     VALID-region mask and its predicted thin-L EOS ring, x verified-confidence {0,1}:
#       [0] preserve valid   [1] preserve eos     (out extent == in extent, all demos)
#       [2] transpose valid  [3] transpose eos    (out h,w == in w,h; >=1 non-square demo)
#       [4] bbox valid       [5] bbox eos         (out extent == non-bg content bbox of the
#                                                  input; >=1 demo where bbox != extent)
#     This is EVIDENCE for the structure logits (zero-init [6->3] proj in the model), not a
#     writer -- unlike the warm-init extent levers it cannot move anything at step 0.
# =====================================================================================
PD_STRUCT_DIM = 6
PDS_PRESERVE = 0
PDS_TRANSPOSE = 2
PDS_BBOX = 4


def pd_structure_evidence(
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
    context_mask: torch.Tensor,
    target_inputs: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """ci/co [B,M,L], cm [B,M], target [B,L] -> (features [B,L,6], stats dict of scalars)."""
    B, M, L = context_inputs.shape
    device = target_inputs.device
    side = int(L ** 0.5)
    feats = torch.zeros((B, L, PD_STRUCT_DIM), device=device, dtype=torch.float32)
    stats = {"pd_struct_conf": torch.zeros((), device=device)}
    if side * side != L or M == 0:
        return feats, stats
    with torch.no_grad():
        x = context_inputs.long().to(device)
        y = context_outputs.long().to(device)
        t = target_inputs.long().to(device)
        demo_ok = context_mask.to(device=device, dtype=torch.bool)

        in_h, in_w = _grid_extents(x, side)                                # [B,M]
        out_h, out_w = _grid_extents(y, side)
        bb_h, bb_w = _content_bbox(x, side)
        valid_demo = demo_ok & (in_h > 0) & (out_h > 0)
        n_valid = valid_demo.float().sum(dim=1)                            # [B]

        def verify(ph: torch.Tensor, pw: torch.Tensor,
                   nondegen: torch.Tensor | None = None) -> torch.Tensor:
            ok = ((ph == out_h) & (pw == out_w)) | ~valid_demo
            v = ok.all(dim=1) & (n_valid >= 1)
            if nondegen is not None:
                v = v & nondegen
            return v

        pres_v = verify(in_h, in_w)
        trans_v = verify(in_w, in_h, nondegen=((in_h != in_w) & valid_demo).any(dim=1))
        bbox_v = verify(bb_h, bb_w,
                        nondegen=(((bb_h != in_h) | (bb_w != in_w)) & valid_demo).any(dim=1))

        th, tw = _grid_extents(t, side)                                    # [B] target INPUT extent
        tbh, tbw = _content_bbox(t, side)
        idx = torch.arange(L, device=device)
        row_i = (idx // side).view(1, L)
        col_i = (idx % side).view(1, L)

        def masks(ph: torch.Tensor, pw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """predicted extent (h,w) [B] -> (valid_mask, eos_thin_L_mask) each [B,L] float."""
            ph_ = ph.view(-1, 1)
            pw_ = pw.view(-1, 1)
            vmask = (row_i < ph_) & (col_i < pw_)
            eos_row = (row_i == ph_) & (col_i <= pw_) & (ph_ < side)
            eos_col = (col_i == pw_) & (row_i <= ph_) & (pw_ < side)
            return vmask.float(), (eos_row | eos_col).float()

        for offset, v, ph, pw in ((PDS_PRESERVE, pres_v, th, tw),
                                  (PDS_TRANSPOSE, trans_v, tw, th),
                                  (PDS_BBOX, bbox_v, tbh, tbw)):
            conf = v.float().view(-1, 1)
            vm, em = masks(ph, pw)
            feats[..., offset] = vm * conf
            feats[..., offset + 1] = em * conf

        stats["pd_struct_conf"] = torch.stack(
            (pres_v.float(), trans_v.float(), bbox_v.float())).amax(dim=0).mean()
    return feats, stats


# =====================================================================================
# SS7 BI-DIRECTIONAL EVIDENCE (D10, NEW) -- the reverse question the old file never asks.
#     Everything upstream is x->y only; the y->x view adds:
#       * INVERTIBILITY: if the changed-transition matrix is (near-)bijective in BOTH
#         directions, the task is near-proof a global recolor (trust the mapping); heavy
#         many-to-one collapse means fill/erase (distrust per-src VALUE tables).
#       * DELETION localization: del_rate said "5% of cells vanished" but never WHICH
#         colour; per-src deletion rate (colour -> pad/eos) says "colour 4 always leaves".
#       * DST-MASS: is this cell's colour a rule OUTPUT (mass arrives at it)? Output
#         colours are what changed cells become -- a keep/target prior the src-side
#         tables cannot see.
#
#     Layout [B, L, PD_BIDI_DIM=4] (PDB_* offsets), gathered per-cell by INPUT colour,
#     zeroed on PAD/EOS cells:
#       [0] invertibility  fwd_consistency x bwd_consistency of pooled changed transitions
#                          (broadcast, per task)
#       [1] del_rate       pooled P(this colour's cell leaves the grid) over demos
#       [2] del_min        min over demos-containing-src (agreement form of [1])
#       [3] dst_mass       fraction of changed mass ARRIVING at this colour
# =====================================================================================
PD_BIDI_DIM = 4
PDB_INVERT = 0
PDB_DEL_RATE = 1
PDB_DEL_MIN = 2
PDB_DST_MASS = 3


def pd_bidi_evidence(
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
    context_mask: torch.Tensor,
    target_inputs: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """ci/co [B,M,L], cm [B,M], target [B,L] -> (features [B,L,4], stats dict of scalars)."""
    B, M, L = context_inputs.shape
    device = target_inputs.device
    side = int(L ** 0.5)
    feats = torch.zeros((B, L, PD_BIDI_DIM), device=device, dtype=torch.float32)
    stats = {
        "pd_bidi_invertibility": torch.zeros((), device=device),
        "pd_bidi_del_rate": torch.zeros((), device=device),
    }
    if side * side != L or M == 0:
        return feats, stats
    with torch.no_grad():
        x = context_inputs.long().to(device)
        y = context_outputs.long().to(device)
        t = target_inputs.long().to(device)
        demo_ok = context_mask.to(device=device, dtype=torch.bool)
        xs = x >= COLOR_OFFSET
        ys = y >= COLOR_OFFSET
        valid = xs & ys & demo_ok.unsqueeze(-1)
        changed = valid & (x != y)
        src = (x - COLOR_OFFSET).clamp(0, 9)
        dst = (y - COLOR_OFFSET).clamp(0, 9)

        # invertibility: row-peak consistency (fwd) x col-peak consistency (bwd)
        counts = transition_counts(changed, src, dst, per_demo=False).view(B, 10, 10)
        total = counts.sum(dim=(1, 2))
        fwd = counts.amax(dim=2).sum(dim=1) / total.clamp_min(1.0)
        bwd = counts.amax(dim=1).sum(dim=1) / total.clamp_min(1.0)
        invert = fwd * bwd * (total > 0).float()                           # [B]

        # per-src deletion (colour cell -> pad/eos), pooled + cross-demo min
        deleted = xs & (~ys) & demo_ok.unsqueeze(-1)                       # [B,M,L]
        present = xs & demo_ok.unsqueeze(-1)
        src_flat = src.reshape(B * M, L)
        n_pres = torch.zeros((B * M, 10), device=device, dtype=torch.float32)
        n_del = torch.zeros_like(n_pres)
        n_pres.scatter_add_(1, src_flat, present.reshape(B * M, L).float())
        n_del.scatter_add_(1, src_flat, deleted.reshape(B * M, L).float())
        n_pres = n_pres.view(B, M, 10)
        n_del = n_del.view(B, M, 10)
        has_src = n_pres > 0
        del_pooled = n_del.sum(dim=1) / n_pres.sum(dim=1).clamp_min(1.0)   # [B,10]
        del_demo = n_del / n_pres.clamp_min(1.0)
        del_masked = torch.where(has_src, del_demo, torch.full_like(del_demo, 2.0))
        del_min = del_masked.amin(dim=1).clamp(0.0, 1.0)
        del_min = del_min * (has_src.any(dim=1)).float()

        # dst-mass: fraction of changed mass arriving at each colour
        dst_mass = counts.sum(dim=1) / total.clamp_min(1.0).unsqueeze(-1)  # [B,10]

        tc = (t - COLOR_OFFSET).clamp(0, 9)
        tvalid = (t >= COLOR_OFFSET).float()
        feats[..., PDB_INVERT] = invert.unsqueeze(-1) * tvalid
        feats[..., PDB_DEL_RATE] = del_pooled.gather(1, tc) * tvalid
        feats[..., PDB_DEL_MIN] = del_min.gather(1, tc) * tvalid
        feats[..., PDB_DST_MASS] = dst_mass.gather(1, tc) * tvalid

        stats["pd_bidi_invertibility"] = invert.mean()
        nv = tvalid.sum().clamp_min(1.0)
        stats["pd_bidi_del_rate"] = feats[..., PDB_DEL_RATE].sum() / nv
    return feats, stats


def _self_test() -> None:
    """Block-1 self-test: schema constants + a tiny sanity forward of the ported functions.
    (The REAL gate is scripts/pd_v2_gate.py -- byte-equality vs the oracle.)"""
    assert FEATURE_DIM == 126 and DDF_SCALARS + 6 == FEATURE_DIM
    B, M, L = 2, 3, GRID_LEN
    ci = torch.full((B, M, L), PAD_TOKEN); co = torch.full((B, M, L), PAD_TOKEN)
    ci[:, :, :40] = 4; co[:, :, :40] = 7
    cm = torch.ones(B, M, dtype=torch.bool)
    feats, valid = demo_delta_features(ci, co, cm)
    assert feats.shape == (B, M, FEATURE_DIM) and bool(valid.all())
    out = pairdelta_intent_features(ci, co, cm)
    expected = {"feature", "shape_preserved", "changed_rate", "dominant_source_color",
                "dominant_target_color", "conditional_recolor_score", "global_recolor_score",
                "changed_cells", "valid_cells"}
    assert set(out) == expected, set(out) ^ expected
    print("pair_delta_v2 Block-1 self-test PASS")


if __name__ == "__main__":
    _self_test()
