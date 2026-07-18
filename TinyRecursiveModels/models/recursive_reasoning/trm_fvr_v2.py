"""trm_fvr_v2.py -- V2 of trm_fvr_c2.py (the LIVE C2/V3 model). File #4 of the senior-dev rewrite.

Same discipline as files #1/#2 (relation_map, core_prior): NOTHING deleted, structure changed in the
rewrite, behaviour changes only behind default-off flags. The oracle here is the LIVE MODEL, so the
master gate is STEP-0 BYTE-IDENTITY: same config + same WEIGHTS + same batch => identical
logits / q_logits / every extras tensor, old-file vs V2 (scripts/step0_identity_harness.py). That is
stronger than files #1/#2's function equality -- the whole forward graph (warm-started heads included)
must agree tensor-for-tensor. Old file `trm_fvr_c2.py` stays UNTOUCHED as that oracle until every
consumer (pretrain / run_stage1_local / losses) is migrated.

V3-CLEAN THESIS (unchanged): the factored structure_head ⟂ color_head is the SOLE output writer;
relational maps + demo evidence enter as INPUT features and are READ by color_head at OUTPUT. Evidence
in, LEARNED selection, NEVER a writer. New evidence goes OUTPUT-side (color_evidence_proj / quarantine)
on a dedicated wd=0 lr group -- the measured lesson that input-side broadcast hints break MAIN once
their norm grows, and that zero-init evidence at the core lr is inert.

Section ROLE tags:
  * §1  SCHEMA        -- constants + the EVIDENCE COLUMN SCHEMA (EvidenceSchema / QuarCol): ONE source
                         of truth for the evidence layout that was hand-synced across 4 sites (M4).
  * §2  CONFIG        -- FVR_C2_Config, fields VERBATIM (order/name/default preserved for pydantic +
                         checkpoint config compat), grouped by ROLE; measured-dead flags flagged.
  * §3  GEOMETRY      -- extent box / pad / eos masks (PERCEPTION).
  * §4  ENCODERS      -- TokenGridEncoder / CrossAttention / TestConditionedC2 (INPUT).
  * §5  EVIDENCE      -- palette / transition / value_v2 / algo_where builders + core_prior E-1/E-2.
  * §6  INPUT HINTS   -- broadcast-hint injectors + _condition_grid_features.
  * §7  MODEL INIT    -- Inner.__init__, heads BUILT FROM THE SCHEMA.
  * §8  LODO          -- leave-one-demo aux + shuffle contrast.
  * §9  OUTPUT        -- _output_logits decomposed into testable helpers.
  * §10 ACT WRAPPER   -- outer halting model.

Token convention (repo-wide): PAD=0, EOS=1, colour = token-2 (0..9). Grids are square.
"""
from typing import Dict, Tuple

from contextlib import nullcontext
import hashlib
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from models.common import trunc_normal_init_
from models.layers import CastedEmbedding, CastedLinear, RotaryEmbedding, rms_norm
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Block,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
    TinyRecursiveReasoningModel_ACTV1ReasoningModule,
)
from models.recursive_reasoning.relation_map import REL_MAP_CHANNELS   # M11: was object_bank (bit-identical)
from models.sparse_embedding import CastedSparseEmbedding
try:
    from models.c2_siglip_rule_adapter import C2SigLIPRuleAdapter
except ModuleNotFoundError:
    C2SigLIPRuleAdapter = None
# M13: the visual_arc_renderer SOURCE was removed (dead lane). Old file hard-imports ARCTokenSpec ->
# any import of the module outside the training env NameError/ModuleNotFoundErrors. Guard it like the
# adapter above: ARCTokenSpec is used ONLY inside the c2_visual_rule_adapter construction (default off,
# and that path already raises via the C2SigLIPRuleAdapter-is-None check).
try:
    from models.visual_arc_renderer import ARCTokenSpec
except ModuleNotFoundError:
    ARCTokenSpec = None


# ======================================================================================
# section 1 -- CONSTANTS + EVIDENCE SCHEMA                                 (ROLE: SCHEMA)
# The evidence column layout was maintained BY HAND in FOUR places that had to agree silently
# (M4, the B3 bug class): the __init__ extra_cols arithmetic, the _output_logits evidence_parts
# append order + v2_col_offset, the _quarantine_features concat, and the quarantine warm-init column
# INTEGERS. One drift = silently garbage evidence, no crash. These schemas are the single source of
# truth; every consumer addresses columns by NAME.
# ======================================================================================
VALUE_EVIDENCE_V2_DIM = 36
# FIX C: [enclosed(1) | flood-fill enclosing-colour one-hot(10) | nearest-seed-colour one-hot(10)]
ALGO_WHERE_MAP_DIM = 21
# Context-bucket capacity (transient no-grad tensor, not a saved param -> checkpoint-safe to change).
VALUE_EVIDENCE_V2_CONTEXT_BUCKETS = 512

# Operation-family vocab for the in-model rule-hypothesis hint (index 0 = "none"). The other four are
# exactly the families object_rule_bank / core_prior.infer_rule_hypotheses can return.
RULE_FAMILY_VOCAB = ("none", "identity", "recolor", "rearrange", "size_change")
RULE_FAMILY_INDEX = {name: i for i, name in enumerate(RULE_FAMILY_VOCAB)}

# M5: RULE_FAMILY_VOCAB hand-mirrors the solver's family set; a NEW family would silently map to "none"
# via RULE_FAMILY_INDEX.get(..., 0). Assert the two vocabularies stay in sync at import (fail loud).
# M11: the assert target is now core_prior.RULE_FAMILIES (== object_rule_bank.RULE_FAMILIES), tracking
# the infer_rule_hypotheses import flip below.
from models.recursive_reasoning.core_prior import RULE_FAMILIES as _RULE_FAMILIES  # noqa: E402
assert set(RULE_FAMILY_VOCAB[1:]) == set(_RULE_FAMILIES), (
    f"RULE_FAMILY_VOCAB {set(RULE_FAMILY_VOCAB[1:])} drifted from solver families {set(_RULE_FAMILIES)}")


# File #5: the pair-delta V2 evidence widths (schema-owned by pair_delta_v2, not hand-synced here).
from models.recursive_reasoning.pair_delta_v2 import PD_BIDI_DIM, PD_COLOR_DIM, PD_STRUCT_DIM  # noqa: E402
# E-5 (audit A3): kinematic evidence width (schema-owned by core_prior).
from models.recursive_reasoning.core_prior import (  # noqa: E402
    KINEMATIC_DIM,
    RULE_FACTOR_NAMES,
    RULE_FACTOR_SEMVER,
)
RULE_FACTOR_DIM = len(RULE_FACTOR_NAMES)

# --- OUTPUT-side evidence columns (fed to color_evidence_proj). ORDER IS THE CONTRACT: it is the
#     append order in _output_logits and the column layout of the zero-init projection. Legacy entries
#     first (frozen); D7/D1/D2/D8 appended LAST so a checkpoint warm-start copies the min(cols) prefix
#     column-aligned (the FIX-C rule). width is an int or a callable(cfg)->int.
EVIDENCE_COLS = (
    ("relmap",         REL_MAP_CHANNELS,                                  "c2_relmap"),
    ("palette",        10,                                                "c2_task_palette_feature"),
    ("where_hint",     lambda c: max(1, int(getattr(c, "c2_rel_where_topk", 1))), "c2_rel_where_hint"),
    ("intent",         1,                                                 "c2_pairdelta_intent_hint"),
    ("transition",     10,                                                "c2_transition_hint"),
    ("value_v2",       VALUE_EVIDENCE_V2_DIM,                             "c2_value_evidence_v2"),
    ("algo_where",     ALGO_WHERE_MAP_DIM,                                "c2_algo_where_maps"),
    # --- APPENDED LAST (checkpoint prefix stability); all NEW flags default off -> width 0 at legacy config:
    ("value_ctx_gate", 2,                                                 "c2_value_ctx_gate"),           # D7 (user)
    ("verified_frame", 11,                                                "c2_verified_frame_evidence"),  # D1 (E-1)
    ("analogy",        11,                                                "c2_analogy_evidence"),         # D2 (E-2)
    ("pd_color",       PD_COLOR_DIM,                                      "c2_pairdelta_color_evidence"), # D8 (File #5)
    ("pd_bidi",        PD_BIDI_DIM,                                       "c2_pairdelta_bidi_evidence"),  # D10 (SS7)
    ("value_ctx_bind", 20,                                                "c2_value_ctx_bind"),           # D11 (codex)
    ("algo_touch",     14,                                                "c2_algo_where_touch"),         # D6 (B13)
    ("kinematic",      KINEMATIC_DIM,                                     "c2_kinematic_evidence"),       # E-5 (A3)
    ("canonical_bind", 10,                                                "c2_canonical_value_binder"),
)
# v3: canonical hierarchical keys define background by object attribution rather than ARC colour 0.
# This is a semantic change at unchanged width, so interim v2 checkpoints must not load silently.
# v4 (P1): canonical bind evidence is the RAW normalized distribution -- the same-position route no
# longer multiplies it (route is confidence, not probability mass) and changed/copy support flags
# are outcome-specific. Same ten-column width, different semantics: v3 checkpoints must not load
# silently into the canonical consumer.
EVIDENCE_SCHEMA_SEMVER = 4

_CANONICAL_SUPPRESSED_EVIDENCE = frozenset({
    "palette", "where_hint", "intent", "transition", "value_v2", "algo_where",
    "value_ctx_gate", "verified_frame", "analogy", "pd_color", "pd_bidi",
    "value_ctx_bind", "algo_touch", "kinematic",
})


def _col_width(width, cfg) -> int:
    return int(width(cfg)) if callable(width) else int(width)


def evidence_layout(cfg) -> Tuple[list, int]:
    """Active evidence columns for a config -> ([(name, width, start_offset)], total_width). Replaces
    the hand `extra_cols` arithmetic AND the `evidence_parts`/`v2_col_offset` bookkeeping."""
    layout = []
    off = 0
    canonical = bool(getattr(cfg, "c2_canonical_value_binder", False))
    for name, width, flag in EVIDENCE_COLS:
        if canonical and name in _CANONICAL_SUPPRESSED_EVIDENCE:
            continue
        if bool(getattr(cfg, flag, False)):
            w = _col_width(width, cfg)
            layout.append((name, w, off))
            off += w
    return layout, off


def evidence_total(cfg) -> int:
    return evidence_layout(cfg)[1]


def evidence_slice(cfg, name: str):
    """(start, width) of one active evidence block by name, or None if its flag is off."""
    for n, w, s in evidence_layout(cfg)[0]:
        if n == name:
            return s, w
    return None


def evidence_schema_fingerprint(cfg) -> torch.Tensor:
    """Stable semantic fingerprint for the active ordered evidence layout.

    Width equality is insufficient for checkpoint compatibility: a ten-column palette block and a
    ten-column transition block have different meanings. The digest therefore includes semantic
    version, ordered names, widths, and offsets. It is represented as a persistent uint8 tensor so
    normal checkpoint state-dict machinery can carry it without custom serialization.
    """
    active, total = evidence_layout(cfg)
    semantic_inputs = []
    if any(bool(getattr(cfg, flag, False)) for flag in (
        "c2_rule_factor_hint",
        "c2_object_pair_tokens",
        "c2_extent_conditioned_structure",
        "c2_canonical_value_binder",
    )):
        semantic_inputs.append(("rule_factors", RULE_FACTOR_SEMVER))
    if bool(getattr(cfg, "c2_pairdelta_spatial", False)):
        semantic_inputs.append(("pairdelta_spatial", 1))
    payload = repr((
        EVIDENCE_SCHEMA_SEMVER,
        tuple(active),
        total,
        tuple(semantic_inputs),
    )).encode("utf-8")
    return torch.tensor(list(hashlib.sha256(payload).digest()), dtype=torch.uint8)


def extent_conditioned_structure(
    floor_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    same_extent_probability: torch.Tensor,
) -> torch.Tensor:
    """Route structure residuals by the support-derived same-extent probability.

    ``p=1`` preserves the floor exactly; ``p=0`` permits the full candidate. Intermediate values
    interpolate logits and keep gradients available to the existing structure head.
    """
    p = same_extent_probability.to(device=floor_logits.device, dtype=floor_logits.dtype)
    while p.ndim < floor_logits.ndim:
        p = p.unsqueeze(-1)
    p = p.clamp(0.0, 1.0)
    mixed = floor_logits + (1.0 - p) * (candidate_logits - floor_logits)
    # Preserve the endpoint contracts byte-for-byte; the arithmetic interpolation can otherwise
    # differ by one ULP from ``candidate_logits`` when p==0.
    return torch.where(p >= 1.0, floor_logits, torch.where(p <= 0.0, candidate_logits, mixed))


def bounded_residual(
    base: torch.Tensor,
    residual: torch.Tensor,
    rho: float,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a per-example residual norm budget and report raw/applied norm ratios."""
    if base.shape != residual.shape:
        raise ValueError(f"base/residual shape mismatch: {tuple(base.shape)} != {tuple(residual.shape)}")
    dims = tuple(range(1, base.ndim))
    base_norm = torch.linalg.vector_norm(base.float(), dim=dims).clamp_min(eps)
    residual_norm = torch.linalg.vector_norm(residual.float(), dim=dims)
    scale = torch.minimum(
        torch.ones_like(base_norm),
        float(rho) * base_norm / residual_norm.clamp_min(eps),
    )
    view_shape = (base.shape[0],) + (1,) * (base.ndim - 1)
    applied = residual * scale.to(residual.dtype).view(view_shape)
    return (
        base + applied,
        (residual_norm / base_norm).mean(),
        ((residual_norm * scale) / base_norm).mean(),
    )


# --- PID-QUARANTINED candidate feature layout (fixed; independent of flags). Named offsets replace the
#     bare warm-init integers (old: weight[c, 2+c] / [c, 108+c] / [c, 130+c]).
QUARANTINE_COLS = (
    ("input_onehot", 12),        # target-input one-hot (PAD/EOS/10 colours)
    ("neigh_onehot", 96),        # 8-neighbour one-hots (8 * 12)
    ("transition",   10),        # MARGINAL P(out | src)
    ("where",        1),         # rel-where best-predicate gate
    ("palette",      10),        # task palette
    ("intent",       1),         # pairdelta intent
    ("conditioned",  10),        # FIX B: CONDITIONED P(out | src, ctx) with backoff
    ("relmap",       REL_MAP_CHANNELS),
)


def _cumulative_ns(cols) -> SimpleNamespace:
    """names+widths -> namespace of UPPERCASED name -> START offset (+ TOTAL). QuarCol.CONDITIONED==130."""
    ns = SimpleNamespace()
    off = 0
    for name, w in cols:
        setattr(ns, name.upper(), off)
        off += w
    ns.TOTAL = off
    return ns


QuarCol = _cumulative_ns(QUARANTINE_COLS)
assert QuarCol.TOTAL == 140 + REL_MAP_CHANNELS, f"quarantine layout total drifted: {QuarCol.TOTAL}"
assert (QuarCol.INPUT_ONEHOT, QuarCol.TRANSITION, QuarCol.CONDITIONED, QuarCol.RELMAP) == (0, 108, 130, 140), \
    "quarantine named offsets must match the legacy integer layout (2+c / 108+c / 130+c warm-init)"


# ======================================================================================
# section 2 -- CONFIG  (ROLE: CONFIG). Fields are VERBATIM from trm_fvr_c2 (order/name/default
# preserved for pydantic + checkpoint config compatibility); only ROLE-group comments + the appended
# default-off V2 flags are new. DO NOT reorder or rename -- that breaks checkpoint config load.
# ======================================================================================
class FVR_C2_Config(TinyRecursiveReasoningModel_ACTV1Config):
    # --- core C2 ---
    c2_enabled: bool = True
    c2_num_context: int = 3
    c2_mode: str = "test_conditioned"
    c2_heads: int = 4
    c2_gate_init: float = 0.0
    c2_use_cross_demo: bool = True
    c2_pid_dropout: float = 0.0
    c2_leave_one_demo_weight: float = 0.0
    c2_lodo_force_build: bool = False
    c2_lodo_force_shuffle: bool = False
    c2_lodo_contrast_weight: float = 0.0
    c2_lodo_contrast_margin: float = 0.05
    c2_lodo_max_samples: int = 4
    c2_use_change_features: bool = True
    c2_lodo_blank_pid: bool = False
    c2_modulate_pid: bool = False
    c2_per_token_gate: bool = False
    c2_token_gate_where: bool = False
    c2_positive_where_gate: bool = False
    c2_gate_selector_detach: bool = False
    # --- P3A support-conditioned WHERE (Blocks 1-3; default-off, flag-off path bitwise unchanged) ---
    c2_isolated_relmap_query: bool = False
    c2_support_interaction_gate: bool = False
    c2_lodo_zero_support: bool = False
    c2_ordered_evidence_flow: bool = False
    c2_bounded_evidence_fusion: bool = False
    c2_target_relmap_rho: float = 0.10
    c2_post_hint_rho: float = 0.10
    c2_update_rho: float = 0.15
    c2_allow_legacy_evidence_schema: bool = False
    c2_gate_dropout: float = 0.0
    c2_gate_l2_weight: float = 0.0
    # --- V3 factored output head + floor/candidate split ---
    c2_dual_output_head: bool = True
    c2_floor_candidate_split: bool = False
    c2_candidate_floor_structure: bool = False
    c2_relmap: bool = True
    # --- Lane B: rule-hypothesis / frame hints (input-side, zero-init, F7-safe) ---
    c2_frame_hint: bool = False
    # NOTE the rule-hypothesis TOKEN was measured DOA (--rule-probe ~3/20 actionable; C' 4th neg);
    # provided default-OFF for A/B, NOT expected to convert to exact solves.
    c2_rule_hypothesis_hint: bool = False
    # --- V3 colour palette prior ---
    c2_task_palette_feature: bool = False
    c2_task_palette_bias: bool = False
    c2_task_palette_strength: float = 4.0
    c2_task_palette_hard: bool = False
    # --- relational-colour probe signals (color_head input features only, never writers) ---
    c2_rel_where_hint: bool = False
    c2_rel_where_topk: int = 1
    c2_algo_where_maps: bool = False
    c2_pairdelta_intent_hint: bool = False
    c2_transition_hint: bool = False
    c2_value_evidence_v2: bool = False
    c2_canonical_value_binder: bool = False
    c2_value_backoff_tau: float = 3.0
    c2_value_v2_rich_ctx: bool = False
    c2_color_head_mlp_dim: int = 0
    # --- PID-quarantined candidate ---
    c2_quarantine_candidate: bool = False
    c2_quarantine_hidden: int = 256
    # --- §15.2 cross-demo input-side upgrades ---
    c2_relmap_demos: bool = False
    c2_pairdelta_input_feature: bool = False
    c2_pairdelta_include_identity: bool = False
    c2_pairdelta_spatial: bool = False
    c2_rule_factor_hint: bool = False
    c2_object_pair_tokens: bool = False
    # --- geometry / structure heads + extent levers ---
    c2_geometry_aux_head: bool = True
    c2_relmap_structure: bool = True
    c2_structure_from_lmhead: bool = False
    c2_relmap_outside_grid: bool = False
    c2_structure_outside_warm_init: bool = False
    c2_structure_outside_warm_init_value: float = 1000.0
    c2_relmap_eos_grid: bool = False
    c2_structure_eos_warm_init: bool = False
    c2_structure_eos_warm_init_value: float = 1000.0
    c2_extent_use_shape_head: bool = False
    c2_extent_shape_head_tau: float = 0.5
    c2_extent_shape_head_conf: float = 0.5
    c2_shape_head: bool = False
    c2_shape_pool: str = "zH_puzzle_gridmean"
    c2_structure_fusion_alpha: float = 0.0
    c2_extent_conditioned_structure: bool = False
    # --- PairDelta encoder dims (the KEPT --pairdelta-input hint reuses these) ---
    c2_delta_rule_encoder_dim: int = 256
    c2_delta_rule_slots: int = 8
    c2_delta_expose_base_logits: bool = False
    # --- DEPRECATED / MEASURED-DEAD (kept for config-load compat; lanes are inert or raise) ---
    # visual encoder lane REMOVED (plan §12.2): c2_visual_encoder=True raises at build.
    c2_visual_encoder: bool = False
    c2_visual_cache_path: str | None = None
    c2_visual_model_name: str | None = None
    c2_visual_gate_init: float = 0.0
    c2_visual_feature_dim: int = 1024
    c2_visual_project_dim: int = 512
    c2_visual_cache_level: str = "dino_patch_16x16"
    c2_visual_rule_adapter: bool = False
    c2_visual_encoder_name: str = "google/siglip-base-patch16-224"
    c2_visual_mode: str = "pooled_demo_delta_symbolic"
    c2_visual_rule_dim: int = 256
    c2_visual_use_query_output: bool = False
    # --- V2 NEW flags (all default-off => step-0 byte-identical to trm_fvr_c2 at legacy configs) ---
    c2_value_ctx_gate: bool = False           # D7 (user): context-conditioned copy/change gate (+2 EvCol)
    c2_verified_frame_evidence: bool = False  # D1 (E-1): verified-frame applied grid -> 11 EvCol columns
    c2_analogy_evidence: bool = False         # D2 (E-2): analogy per-cell colour dist -> 11 EvCol columns
    c2_value_v2_backoff: bool = False         # D4 (=CF1/M1): collision-free ordered backoff context key
    c2_frame_hint_ranked: bool = False        # D5 (E-3): multi-hot ranked-frame hint
    c2_algo_where_touch: bool = False         # D6 (WIRED): touch-colour sig cols 14/15 -> 14 EvCol (10 mode + 4 count)
    c2_pairdelta_color_evidence: bool = False # D8 (File #5): cross-demo agreement + positional prior -> 14 EvCol
    c2_pairdelta_structure_evidence: bool = False  # D9 (File #5): verified extent-family masks -> zero-init [6->3] structure proj
    c2_pairdelta_bidi_evidence: bool = False  # D10 (SS7): y->x view (invertibility/deletion/dst-mass) -> 4 EvCol
    c2_pairdelta_input_conf_gate: bool = False  # SS7 reuse: gate the input rule_vec broadcast by rule_confidence
    c2_value_ctx_bind: bool = False           # D11 (codex): explicit gate x value product columns -> 20 EvCol
    c2_kinematic_evidence: bool = False       # E-5 (A3): per-cell mover/(dr,dc)/blocked bits -> 7 EvCol


def _select_heads(hidden_size: int, requested_heads: int, max_heads: int) -> int:
    heads = max(1, min(requested_heads, max_heads))
    while hidden_size % heads != 0:
        heads -= 1
    return heads


# ======================================================================================
# section 3 -- EXTENT GEOMETRY (predicted output box -> PAD/EOS masks)   (ROLE: PERCEPTION)
# The predicted output box offset is read from the INPUT content bbox (the tokenizer pads input+output
# with ONE (pad_r,pad_c)). Support-safe; verified IoU=100%/eos-leak=0 on the 518K aux.
# ======================================================================================
def _extent_box_geometry(inputs: torch.Tensor, h: torch.Tensor, w: torch.Tensor, side: int):
    """Shared geometry for extent_pad_mask / extent_eos_mask.

    The predicted output box is ``[off_r:off_r+h, off_c:off_c+w]`` on the ``side x side`` row-major
    canvas. Its top-left offset ``(off_r, off_c)`` is read from the INPUT content bbox -- provably the
    output offset too, because the tokenizer pads input and output with ONE ``(pad_r, pad_c)``
    (build_arc_dataset.np_grid_to_seq_translational_augment:54) -- support-safe: the test INPUT is known,
    only the test OUTPUT is hidden. EOS mirrors the tokenizer's thin-L: one row directly below the box
    spanning the content columns + one column directly right spanning the content rows; the bottom-right
    corner belongs to neither (it is PAD). Verified IoU=100%/eos-leak=0 on the 518K aux
    (scripts/verify_outside_grid_lever.py).

    M10: when the predicted box reaches the canvas edge (r1 == side or c1 == side), the corresponding
    EOS row/col is simply absent (`ar == r1` is never true) -- correct for full-canvas-content tasks,
    documented here so it is not mistaken for a bug.

    Args:
        inputs: [B, L] token grid, L == side*side (PAD 0 / EOS 1 / colours >=2).
        h, w:   [B] predicted output height/width in CELLS.
    Returns:
        (in_box, eos_row, eos_col, has): [B,side,side] bools + [B] non-empty flag.
    """
    assert inputs.ndim == 2 and inputs.shape[1] == side * side, (
        f"extent mask expects [B, {side * side}], got {tuple(inputs.shape)}")
    b = inputs.shape[0]
    dev = inputs.device
    content = (inputs.reshape(b, side, side) >= 2)          # [B, side, side] colour cells
    has = content.any(dim=(1, 2))                           # [B] non-empty grid
    off_r = torch.argmax(content.any(dim=2).int(), dim=1)   # first content row (0 if empty)
    off_c = torch.argmax(content.any(dim=1).int(), dim=1)   # first content col
    hh = h.round().clamp(0, side).long()
    ww = w.round().clamp(0, side).long()
    ar = torch.arange(side, device=dev).view(1, side, 1)    # row index
    ac = torch.arange(side, device=dev).view(1, 1, side)    # col index
    r0, c0 = off_r.view(b, 1, 1), off_c.view(b, 1, 1)
    r1 = (off_r + hh).view(b, 1, 1)                         # one past the box (== eos row)
    c1 = (off_c + ww).view(b, 1, 1)                         # one past the box (== eos col)
    in_box = (ar >= r0) & (ar < r1) & (ac >= c0) & (ac < c1)
    eos_row = (ar == r1) & (ac >= c0) & (ac < c1)           # thin-L: row below box, content cols
    eos_col = (ac == c1) & (ar >= r0) & (ar < r1)           # thin-L: col right of box, content rows
    return in_box, eos_row, eos_col, has


def extent_pad_mask(inputs: torch.Tensor, h: torch.Tensor, w: torch.Tensor, side: int) -> torch.Tensor:
    """[B,L] float, 1.0 on predicted-PAD cells: outside the output box AND off the thin-L EOS border
    (eos-clean -> the near-hard override never converts EOS->PAD). Geometry: _extent_box_geometry."""
    in_box, eos_row, eos_col, has = _extent_box_geometry(inputs, h, w, side)
    pad = (~in_box) & (~eos_row) & (~eos_col) & has.view(-1, 1, 1)
    return pad.reshape(inputs.shape[0], side * side).to(torch.float32)


def extent_eos_mask(inputs: torch.Tensor, h: torch.Tensor, w: torch.Tensor, side: int) -> torch.Tensor:
    """[B,L] float, 1.0 on predicted thin-L EOS cells (row below + column right of the output box; the
    bottom-right corner stays PAD). Geometry: _extent_box_geometry."""
    in_box, eos_row, eos_col, has = _extent_box_geometry(inputs, h, w, side)
    eos = (eos_row | eos_col) & has.view(-1, 1, 1)
    return eos.reshape(inputs.shape[0], side * side).to(torch.float32)


# ======================================================================================
# section 4 -- ENCODERS (target/demo feature fields + cross-demo rule bank)   (ROLE: INPUT)
# ======================================================================================
class TokenGridEncoder(nn.Module):
    """Token-grid feature boundary for C2.

    C2 consumes [B, S, D] feature fields. Today those fields come from TRM's token embedding table; a
    future visual encoder can replace this class while keeping TestConditionedC2 unchanged.
    """

    def __init__(self, embed_tokens: CastedEmbedding, embed_scale: float):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_scale = embed_scale

    def forward(
        self,
        tokens: torch.Tensor,
        cached_patch_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed_scale * self.embed_tokens(tokens.to(torch.int32))


class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.q_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.k_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.v_proj = CastedLinear(hidden_size, hidden_size, bias=False)
        self.o_proj = CastedLinear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        # If a row has no valid demo tokens, use a zero-valued all-valid bank to
        # avoid NaNs while preserving a no-op output through bias-free projections.
        batch_size = query.shape[0]
        key_mask = key_mask.to(torch.bool)
        has_key = key_mask.any(dim=-1, keepdim=True)
        safe_mask = torch.where(has_key, key_mask, torch.ones_like(key_mask))
        key_value = torch.where(key_mask.unsqueeze(-1), key_value, torch.zeros_like(key_value))

        query = self._split_heads(self.q_proj(query))
        key = self._split_heads(self.k_proj(key_value))
        value = self._split_heads(self.v_proj(key_value))

        attn_mask = safe_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return self.o_proj(attended)


class TestConditionedC2(nn.Module):
    def __init__(self, config: FVR_C2_Config):
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps
        heads = _select_heads(config.hidden_size, config.c2_heads, config.num_heads)

        self.demo_proj = CastedLinear(config.hidden_size * 3, config.hidden_size, bias=False)
        self.demo_scalar_proj = CastedLinear(4, config.hidden_size, bias=False)
        self.demo_mix = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.pair_proj = CastedLinear(config.hidden_size * 3, config.hidden_size, bias=False)
        self.pair_mix = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.cross_attn = CrossAttention(config.hidden_size, heads)
        self.patch_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.global_proj = CastedLinear(config.hidden_size, config.hidden_size, bias=False)
        self.gate_patch = nn.Parameter(torch.tensor(float(config.c2_gate_init)))
        self.gate_global = nn.Parameter(torch.tensor(float(config.c2_gate_init)))
        if self.config.c2_token_gate_where and not self.config.c2_per_token_gate:
            raise ValueError("c2_token_gate_where requires c2_per_token_gate=True")
        if self.config.c2_positive_where_gate and not self.config.c2_token_gate_where:
            raise ValueError("c2_positive_where_gate requires c2_token_gate_where=True")
        if self.config.c2_gate_selector_detach and not self.config.c2_positive_where_gate:
            raise ValueError("c2_gate_selector_detach requires c2_positive_where_gate=True")
        if self.config.c2_positive_where_gate and not self.config.c2_ordered_evidence_flow:
            raise ValueError("c2_positive_where_gate requires c2_ordered_evidence_flow=True")
        if self.config.c2_positive_where_gate and not self.config.c2_rel_where_hint:
            raise ValueError("c2_positive_where_gate requires c2_rel_where_hint=True")
        if self.config.c2_positive_where_gate and abs(float(self.config.c2_gate_init)) > 1e-12:
            raise ValueError("c2_positive_where_gate requires c2_gate_init=0 for step-0 identity")
        if self.config.c2_ordered_evidence_flow and not self.config.c2_relmap:
            raise ValueError("c2_ordered_evidence_flow requires c2_relmap=True")
        # P3A Block 1/2 validity: x_query only exists on the ordered flow, and the interaction
        # gate reads x_query and needs the sigmoid selector for q in [0,1].
        if (getattr(self.config, "c2_isolated_relmap_query", False)
                and not self.config.c2_ordered_evidence_flow):
            raise ValueError("c2_isolated_relmap_query requires c2_ordered_evidence_flow=True")
        if getattr(self.config, "c2_support_interaction_gate", False):
            if not getattr(self.config, "c2_isolated_relmap_query", False):
                raise ValueError("c2_support_interaction_gate requires c2_isolated_relmap_query=True")
            if not self.config.c2_positive_where_gate:
                raise ValueError("c2_support_interaction_gate requires c2_positive_where_gate=True")
            if not self.config.c2_gate_selector_detach:
                raise ValueError(
                    "c2_support_interaction_gate requires c2_gate_selector_detach=True: WHERE "
                    "supervision owns the selector; transport losses must not rewrite it (P3A)")
        if self.config.c2_per_token_gate:
            # The WHERE arm adds a zero-initialized module. Preserve the outer RNG stream so
            # flag-on/off arms keep identical initialization for every module constructed later.
            _fork_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            _rng_ctx = (
                torch.random.fork_rng(devices=_fork_devices)
                if self.config.c2_token_gate_where
                else nullcontext()
            )
            with _rng_ctx:
                self.gate_patch_token = CastedLinear(config.hidden_size, 1, bias=True)
            with torch.no_grad():
                if self.config.c2_positive_where_gate:
                    self.gate_patch_token.weight.normal_(mean=0.0, std=1e-3)
                    self.gate_patch_token.bias.zero_()
                else:
                    self.gate_patch_token.weight.zero_()
                    self.gate_patch_token.bias.fill_(float(config.c2_gate_init))
        if self.config.c2_positive_where_gate:
            self.where_gate_weights = nn.Parameter(torch.zeros(
                max(1, int(getattr(config, "c2_rel_where_topk", 1))), dtype=torch.float32))

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(x.dtype).unsqueeze(-1)
        return (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1)

    def _position_features(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        width = math.isqrt(seq_len)
        assert width * width == seq_len, f"ARC grids are square; got seq_len={seq_len}"
        height = width
        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        rows = positions // width
        cols = positions % width
        row_norm = rows.to(dtype) / max(height - 1, 1)
        col_norm = cols.to(dtype) / max(width - 1, 1)
        return torch.stack((row_norm, col_norm), dim=-1).view(1, 1, seq_len, 2)

    def _canvas_extent_stats(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return normalized height, width, and area for tokenized ARC grids."""
        assert tokens.ndim == 3, f"Expected tokens [B, M, S], got {tuple(tokens.shape)}"

        batch_size, num_demos, seq_len = tokens.shape
        side = math.isqrt(seq_len)
        assert side * side == seq_len, f"Expected square ARC token grid, got sequence length={seq_len}"

        valid_canvas = (tokens >= 2).reshape(batch_size, num_demos, side, side)
        row_has_content = valid_canvas.any(dim=-1)
        col_has_content = valid_canvas.any(dim=-2)

        height = row_has_content.sum(dim=-1).to(torch.float32) / float(side)
        width = col_has_content.sum(dim=-1).to(torch.float32) / float(side)
        area = valid_canvas.to(torch.float32).sum(dim=(-1, -2)) / float(side * side)

        stats = torch.stack((height, width, area), dim=-1)
        assert stats.shape == (batch_size, num_demos, 3), f"Bad canvas stats shape: {tuple(stats.shape)}"
        assert torch.isfinite(stats).all(), "Non-finite canvas extent statistics."
        return stats

    def _demo_tokens(
        self,
        context_inputs: torch.Tensor,
        context_outputs: torch.Tensor,
        context_input_features: torch.Tensor,
        context_output_features: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        demo_mask = context_mask.to(torch.bool)
        token_mask = ((context_inputs >= 2) | (context_outputs >= 2)) & demo_mask[:, :, None]
        changed_mask = (context_inputs != context_outputs) & token_mask
        positions = torch.arange(context_inputs.shape[-1], device=context_inputs.device, dtype=torch.long)
        token_keys = (
            (context_inputs.long() * self.config.vocab_size + context_outputs.long())
            * context_inputs.shape[-1]
            + positions.view(1, 1, -1)
        )

        transform = torch.cat(
            (
                rms_norm(context_input_features, variance_epsilon=self.norm_eps),
                rms_norm(context_output_features, variance_epsilon=self.norm_eps),
                rms_norm(context_output_features - context_input_features, variance_epsilon=self.norm_eps),
            ),
            dim=-1,
        )
        demo_base = self.demo_proj(transform)
        if self.config.c2_use_change_features:
            position_features = self._position_features(
                seq_len=context_inputs.shape[-1],
                device=context_inputs.device,
                dtype=context_input_features.dtype,
            )
            scalar_features = torch.cat(
                (
                    changed_mask.to(context_input_features.dtype).unsqueeze(-1),
                    token_mask.to(context_input_features.dtype).unsqueeze(-1),
                    position_features.expand(context_inputs.shape[0], context_inputs.shape[1], -1, -1),
                ),
                dim=-1,
            )
            demo_base = demo_base + self.demo_scalar_proj(scalar_features)
        demo_tokens = self.demo_mix(F.silu(demo_base))
        demo_tokens = demo_tokens * token_mask.to(demo_tokens.dtype).unsqueeze(-1)
        return demo_tokens, token_mask, token_keys, changed_mask

    def _rule_bank(
        self,
        demo_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        token_keys: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_demos, seq_len, hidden_size = demo_tokens.shape
        rule_parts = [demo_tokens.reshape(batch_size, num_demos * seq_len, hidden_size)]
        mask_parts = [token_mask.reshape(batch_size, num_demos * seq_len)]
        key_parts = [token_keys.reshape(batch_size, num_demos * seq_len)]
        key_base = self.config.vocab_size * self.config.vocab_size * seq_len + seq_len

        if self.config.c2_use_cross_demo and num_demos > 1:
            for i in range(num_demos):
                for j in range(i + 1, num_demos):
                    pair_mask = token_mask[:, i] & token_mask[:, j]
                    key_i = token_keys[:, i]
                    key_j = token_keys[:, j]
                    pair_keys = key_base + torch.minimum(key_i, key_j) * key_base + torch.maximum(key_i, key_j)
                    pair_features = torch.cat(
                        (
                            demo_tokens[:, i] + demo_tokens[:, j],
                            torch.abs(demo_tokens[:, i] - demo_tokens[:, j]),
                            demo_tokens[:, i] * demo_tokens[:, j],
                        ),
                        dim=-1,
                    )
                    pair_tokens = self.pair_mix(F.silu(self.pair_proj(pair_features)))
                    pair_tokens = pair_tokens * pair_mask.to(pair_tokens.dtype).unsqueeze(-1)
                    rule_parts.append(pair_tokens)
                    mask_parts.append(pair_mask)
                    key_parts.append(pair_keys)

        rule_bank = torch.cat(rule_parts, dim=1)
        rule_mask = torch.cat(mask_parts, dim=1)
        rule_keys = torch.cat(key_parts, dim=1)
        safe_keys = torch.where(
            rule_mask,
            rule_keys,
            torch.full_like(rule_keys, torch.iinfo(torch.long).max),
        )
        order = torch.argsort(safe_keys, dim=1, stable=True)
        rule_bank = rule_bank.gather(1, order.unsqueeze(-1).expand(-1, -1, hidden_size))
        rule_mask = rule_mask.gather(1, order)
        return rule_bank, rule_mask

    def _object_pair_memory(
        self,
        context_inputs: torch.Tensor,
        context_outputs: torch.Tensor,
        context_input_features: torch.Tensor,
        context_output_features: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None, Dict[str, torch.Tensor]]:
        """Pool colour-agnostic matched objects into the existing C2 pair-token space."""
        from models.recursive_reasoning.core_prior import object_correspondences
        B, M, L = context_inputs.shape
        side = int(math.isqrt(L))
        rows: list[list[torch.Tensor]] = [[] for _ in range(B)]
        coverages = []
        precisions = []

        def pool(features: torch.Tensor, cells: tuple) -> torch.Tensor:
            if not cells:
                return features.new_zeros(features.shape[-1])
            idx = torch.tensor(
                [int(r) * side + int(c) for r, c in cells],
                device=features.device, dtype=torch.long)
            return features.index_select(0, idx).mean(dim=0)

        for b in range(B):
            for m in range(M):
                if not bool(context_mask[b, m]):
                    continue
                corr = object_correspondences(context_inputs[b, m], context_outputs[b, m], side)
                coverages.append(float(corr["coverage"]))
                precisions.append(float(corr["precision_proxy"]))
                for item in corr["matched"]:
                    inp = pool(context_input_features[b, m], item["input_cells"])
                    out = pool(context_output_features[b, m], item["output_cells"])
                    rows[b].append(torch.cat((inp, out, out - inp), dim=-1))
                for cells in corr["created"]:
                    out = pool(context_output_features[b, m], cells)
                    rows[b].append(torch.cat((torch.zeros_like(out), out, out), dim=-1))
                for cells in corr["deleted"]:
                    inp = pool(context_input_features[b, m], cells)
                    rows[b].append(torch.cat((inp, torch.zeros_like(inp), -inp), dim=-1))

        max_tokens = max((len(row) for row in rows), default=0)
        stats = {
            "c2_object_pair_count": torch.tensor(
                sum(len(row) for row in rows) / max(B, 1), device=context_inputs.device),
            "c2_object_match_coverage": torch.tensor(
                sum(coverages) / max(len(coverages), 1), device=context_inputs.device),
            "c2_object_match_precision_proxy": torch.tensor(
                sum(precisions) / max(len(precisions), 1), device=context_inputs.device),
        }
        if max_tokens == 0:
            return None, None, stats
        raw = context_input_features.new_zeros((B, max_tokens, self.config.hidden_size * 3))
        mask = torch.zeros((B, max_tokens), device=context_inputs.device, dtype=torch.bool)
        for b, row in enumerate(rows):
            if row:
                raw[b, :len(row)] = torch.stack(row)
                mask[b, :len(row)] = True
        tokens = self.pair_mix(F.silu(self.pair_proj(raw)))
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, mask, stats

    def expose_demo_encoding(
        self,
        context_inputs: torch.Tensor,
        context_outputs: torch.Tensor,
        context_input_features: torch.Tensor,
        context_output_features: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose C2's demo encoding for the Demo-Consistency Verifier to reuse.

        Returns:
            rule_bank:  [B, R, D]  per-task rule tokens (demo + pair features, sorted by key)
            rule_mask:  [B, R]     valid-token mask for rule_bank
            struct_features: [B, M, 10] zeros -- kept for verifier API compatibility (the learned
                struct branch that once filled it was removed; consumers treat it as a placeholder).
        """
        demo_tokens, token_mask, token_keys, changed_mask = self._demo_tokens(
            context_inputs=context_inputs,
            context_outputs=context_outputs,
            context_input_features=context_input_features,
            context_output_features=context_output_features,
            context_mask=context_mask,
        )
        rule_bank, rule_mask = self._rule_bank(demo_tokens, token_mask, token_keys)
        batch_size, num_demos, _ = context_inputs.shape
        struct_features = torch.zeros(
            (batch_size, num_demos, 10),
            device=context_inputs.device,
            dtype=context_input_features.dtype,
        )
        return rule_bank, rule_mask, struct_features

    def forward(
        self,
        target_features: torch.Tensor,
        context_inputs: torch.Tensor,
        context_outputs: torch.Tensor,
        context_input_features: torch.Tensor,
        context_output_features: torch.Tensor,
        context_mask: torch.Tensor,
        target_where_hint: torch.Tensor | None = None,
        rule_factors: torch.Tensor | None = None,
        target_query_features: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor | None]:
        # P3A Block 1: under c2_isolated_relmap_query, `target_features` is x_base (the recurrence
        # input, relmap-free) and `target_query_features` is x_query (x_base + bounded relmap
        # residual). x_query feeds ONLY the cross-attention query and the WHERE gate; every update
        # is added to x_base and x_query is never returned. Flag-off: query IS target (same
        # tensor object), so the legacy path is bitwise unchanged.
        _isolated = bool(getattr(self.config, "c2_isolated_relmap_query", False))
        if _isolated and target_query_features is None:
            raise ValueError("c2_isolated_relmap_query=True requires target_query_features (x_query)")
        if not _isolated and target_query_features is not None:
            raise ValueError("target_query_features passed without c2_isolated_relmap_query=True")
        query_features = target_query_features if _isolated else target_features
        demo_tokens, token_mask, token_keys, changed_mask = self._demo_tokens(
            context_inputs=context_inputs,
            context_outputs=context_outputs,
            context_input_features=context_input_features,
            context_output_features=context_output_features,
            context_mask=context_mask,
        )
        if self.config.c2_object_pair_tokens and rule_factors is not None:
            from models.recursive_reasoning.core_prior import RULE_FACTOR_INDEX
            same_position = (
                rule_factors[:, RULE_FACTOR_INDEX["extent_same"]]
                * rule_factors[:, RULE_FACTOR_INDEX["move_none"]]
            ) >= 0.5
            token_mask = token_mask & same_position[:, None, None]
            demo_tokens = demo_tokens * token_mask.unsqueeze(-1).to(demo_tokens.dtype)
        rule_bank, rule_mask = self._rule_bank(demo_tokens, token_mask, token_keys)
        object_stats: Dict[str, torch.Tensor] = {}
        if self.config.c2_object_pair_tokens:
            object_tokens, object_mask, object_stats = self._object_pair_memory(
                context_inputs, context_outputs, context_input_features,
                context_output_features, context_mask)
            if object_tokens is not None and object_mask is not None:
                rule_bank = torch.cat((rule_bank, object_tokens), dim=1)
                rule_mask = torch.cat((rule_mask, object_mask), dim=1)

        patch_context = self.cross_attn(
            query=rms_norm(query_features, variance_epsilon=self.norm_eps),
            key_value=rms_norm(rule_bank, variance_epsilon=self.norm_eps),
            key_mask=rule_mask,
        )
        patch_context = self.patch_proj(patch_context)

        global_context = self._masked_mean(rule_bank, rule_mask)
        global_context = self.global_proj(global_context).unsqueeze(1)

        gate_global = torch.tanh(self.gate_global).to(target_features.dtype)
        if self.config.c2_per_token_gate:
            normed_target = rms_norm(query_features, variance_epsilon=self.norm_eps)
            if getattr(self.config, "c2_support_interaction_gate", False):
                # P3A Block 2: multiplicative target x support interaction. The additive gate input
                # let a support-blind bias dominate (M1: gate FPR 99.4% -- all-on); a product cannot
                # be faked by either factor alone, and zero/foreign patch_context reshapes it
                # directly. WHERE gradients still reach cross-attention through patch_context.
                gate_input = normed_target * rms_norm(patch_context, variance_epsilon=self.norm_eps)
            else:
                gate_input = normed_target
                if self.config.c2_token_gate_where:
                    gate_input = gate_input + rms_norm(patch_context, variance_epsilon=self.norm_eps)
            gate_logits = self.gate_patch_token(gate_input).squeeze(-1)
            if self.config.c2_positive_where_gate:
                if target_where_hint is not None:
                    if target_where_hint.ndim != 3 or target_where_hint.shape[:2] != gate_logits.shape:
                        raise ValueError(
                            "target_where_hint must have shape [B,L,K], got "
                            f"{tuple(target_where_hint.shape)} for gate {tuple(gate_logits.shape)}")
                    if target_where_hint.shape[-1] != self.where_gate_weights.numel():
                        raise ValueError(
                            f"target_where_hint K={target_where_hint.shape[-1]} does not match "
                            f"configured K={self.where_gate_weights.numel()}")
                    gate_logits = gate_logits + (
                        target_where_hint.to(gate_logits.dtype)
                        * self.where_gate_weights.to(gate_logits.dtype).view(1, 1, -1)
                    ).sum(dim=-1)
                gate_patch_per_token = torch.sigmoid(gate_logits).to(target_features.dtype)
            else:
                gate_patch_per_token = torch.tanh(gate_logits).to(target_features.dtype)
            if self.training and self.config.c2_gate_dropout > 0:
                keep = torch.rand_like(gate_patch_per_token.float()) > float(self.config.c2_gate_dropout)
                gate_patch_per_token = gate_patch_per_token * keep.to(gate_patch_per_token.dtype)
            gate_patch_field = gate_patch_per_token.unsqueeze(-1)
            # Repair A: WHERE supervision owns the selector. Candidate colour/transport losses may
            # still train the C2 update strengths and content projections, but must not rewrite the
            # changed-cell selector through this multiplication path.
            gate_patch_transport_field = (
                gate_patch_field.detach()
                if self.config.c2_gate_selector_detach
                else gate_patch_field
            )
            gate_patch_scalar_metric = gate_patch_per_token.float().mean().detach()
            gate_patch_abs_metric = gate_patch_per_token.float().abs().mean().detach()
            gate_patch_std_metric = gate_patch_per_token.float().std().detach()
            gate_patch_l2 = gate_patch_per_token.float().square().mean()
        else:
            gate_patch_scalar = torch.tanh(self.gate_patch).to(target_features.dtype)
            gate_patch_field = gate_patch_scalar
            gate_patch_transport_field = gate_patch_field
            gate_patch_scalar_metric = torch.tanh(self.gate_patch.float()).detach()
            gate_patch_abs_metric = torch.tanh(self.gate_patch.float()).abs().detach()
            gate_patch_std_metric = torch.zeros((), device=target_features.device).detach()
            gate_patch_l2 = torch.zeros((), device=target_features.device, dtype=torch.float32)

        if self.config.c2_positive_where_gate:
            gate_patch_strength = torch.tanh(self.gate_patch).to(target_features.dtype)
            patch_update = gate_patch_transport_field * gate_patch_strength * rms_norm(
                patch_context, variance_epsilon=self.norm_eps)
            global_update = gate_patch_transport_field * gate_global * rms_norm(
                global_context, variance_epsilon=self.norm_eps)
            update = patch_update + global_update
        else:
            patch_update = gate_patch_field * rms_norm(patch_context, variance_epsilon=self.norm_eps)
            global_update = gate_global * rms_norm(global_context, variance_epsilon=self.norm_eps)
            update = patch_update + global_update
        # DIAGNOSTIC-ONLY forced-signal amplifier (run_stage1_local --zh-amp): default 1.0 == no-op.
        _amp = float(getattr(self, "_demo_injection_scale", 1.0))
        if _amp != 1.0:
            update = update * _amp
        target_norm = target_features.float().norm(dim=-1).mean().clamp_min(1e-6)
        patch_update_norm = patch_update.float().norm(dim=-1).mean()
        global_update_norm = global_update.float().norm(dim=-1).mean()
        update_norm = update.float().norm(dim=-1).mean()
        if self.config.c2_bounded_evidence_fusion:
            target_features, raw_update_ratio, applied_update_ratio = bounded_residual(
                target_features, update, float(self.config.c2_update_rho))
        else:
            target_features = target_features + update
            raw_update_ratio = update_norm / target_norm
            applied_update_ratio = raw_update_ratio
        valid_possible = context_mask.float().sum().clamp_min(1) * context_inputs.shape[-1]
        valid_count = token_mask.float().sum().clamp_min(1)

        pid_task_vec: torch.Tensor | None = None
        if self.config.c2_modulate_pid:
            pid_task_vec = self._masked_mean(rule_bank, rule_mask)

        metrics = {
            "c2_gate_patch": gate_patch_scalar_metric,
            "c2_gate_global": torch.tanh(self.gate_global.float()).detach(),
            "c2_gate_patch_abs": gate_patch_abs_metric,
            "c2_gate_patch_std": gate_patch_std_metric,
            "c2_gate_patch_l2": gate_patch_l2,
            "c2_gate_global_abs": torch.tanh(self.gate_global.float()).abs().detach(),
            "c2_context_count": context_mask.float().sum(dim=-1).mean().detach(),
            "c2_update_norm_ratio": applied_update_ratio.detach(),
            "c2_update_raw_norm_ratio": raw_update_ratio.detach(),
            "c2_update_applied_norm_ratio": applied_update_ratio.detach(),
            "c2_patch_update_norm_ratio": (patch_update_norm / target_norm).detach(),
            "c2_global_update_norm_ratio": (global_update_norm / target_norm).detach(),
            "c2_rule_bank_token_count": rule_mask.float().sum(dim=-1).mean().detach(),
            "c2_changed_token_frac": (changed_mask.float().sum() / valid_count).detach(),
            "c2_valid_token_frac": (token_mask.float().sum() / valid_possible).detach(),
        }
        metrics.update({k: v.detach() for k, v in object_stats.items()})
        if self.config.c2_token_gate_where:
            # Kept attached for the LODO WHERE loss. Scalar panel aggregators ignore this [B,L]
            # tensor; _run_aux_logits explicitly threads it to correct/shuffled aux outputs.
            metrics["c2_gate_where_values"] = gate_patch_per_token
        return target_features, metrics, pid_task_vec


# ======================================================================================
# section 5 -- EVIDENCE BUILDERS (output-side, read by color_head; NEVER writers)   (ROLE: EVIDENCE)
# All builders are no_grad, support+target-derived (PID-free), and LODO-safe: they read the
# `_active_context_*` overrides so the held-out demo never leaks counts into an aux forward.
#
# These live in a MIXIN so §7's Inner inherits them unchanged (method resolution identical to the old
# single-class layout) while staying testable in isolation (Block 3 gate builds the mixin directly).
#
# M3 (consolidation): the marginal CHANGED-transition count P(out|src) was scatter-added in TWO places
# -- `_transition_hint` and `_value_evidence_v2`. A clamp/mask edit to one silently diverged from the
# other. Both now call ONE `_changed_transition_counts` engine over shared `_support_transition_masks`.
# ======================================================================================
class C2EvidenceMixin:
    """Output-side evidence builders for the C2 colour head. Requires `self.config`."""

    # --- M3: shared support-cell masks + the single changed-transition counting engine --------------
    @staticmethod
    def _support_transition_masks(
        x: torch.Tensor, y: torch.Tensor, demo_ok: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(valid, changed, copied, src, dst) over support cells (x=in, y=out tokens, demo_ok mask).

        valid = both endpoints are colours in a valid demo; changed/copied split by (x!=y)/(x==y).
        src/dst are 0..9 colour indices (clamped). Device follows x -- callers keep their own x/y
        construction, so this stays byte-identical whether or not the caller moved tensors to `device`.
        """
        valid = (x >= 2) & (y >= 2) & demo_ok.unsqueeze(-1)
        changed = valid & (x != y)
        copied = valid & (x == y)
        src = (x - 2).clamp(0, 9)
        dst = (y - 2).clamp(0, 9)
        return valid, changed, copied, src, dst

    @staticmethod
    def _changed_transition_counts(
        changed: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
        batch_size: int, device: torch.device,
    ) -> torch.Tensor:
        """[B,10,10] count of CHANGED support transitions src->dst. The ONE engine (M3) behind both
        `_transition_hint`'s consensus rows and `_value_evidence_v2`'s marginal changed-distribution."""
        pair = (src * 10 + dst).reshape(batch_size, -1)
        counts = torch.zeros((batch_size, 100), device=device, dtype=torch.float32)
        counts.scatter_add_(1, pair, changed.reshape(batch_size, -1).to(torch.float32))
        return counts.view(batch_size, 10, 10)

    def _task_palette_mask(
        self,
        batch: Dict[str, torch.Tensor] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Colours observed in support inputs/outputs plus the target input.

        Returns [B, 10] over raw ARC colours 0..9. PAD/EOS tokens are ignored.
        If evidence is missing for a row, fall back to all colours rather than
        creating an impossible hard constraint.
        """
        palette = torch.zeros((batch_size, 10), device=device, dtype=torch.bool)
        if batch is None:
            return torch.ones_like(palette)

        def _add(tokens: torch.Tensor | None, demo_mask: torch.Tensor | None = None) -> None:
            nonlocal palette
            if tokens is None:
                return
            t = tokens.to(device=device, dtype=torch.long)
            is_colour = (t >= 2) & (t < 12)
            idx = (t - 2).clamp(0, 9)
            oh = F.one_hot(idx, num_classes=10).to(torch.bool) & is_colour.unsqueeze(-1)
            if t.ndim == 2:
                palette = palette | oh.any(dim=1)
            elif t.ndim == 3:
                if demo_mask is not None:
                    dm = demo_mask.to(device=device, dtype=torch.bool)
                    oh = oh & dm[:, :, None, None]
                palette = palette | oh.any(dim=(1, 2))

        _add(batch.get("inputs"))
        context_inputs = batch.get("_active_context_inputs", batch.get("context_inputs"))
        context_outputs = batch.get("_active_context_outputs", batch.get("context_outputs"))
        context_mask = batch.get("_active_context_mask", batch.get("context_mask"))
        _add(context_inputs, context_mask)
        _add(context_outputs, context_mask)

        empty = ~palette.any(dim=-1)
        if empty.any():
            palette = palette.clone()
            palette[empty] = True
        return palette

    def _transition_hint(
        self,
        batch: Dict[str, torch.Tensor] | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-cell VALUE evidence [B, L, 10]: the demo-consensus P(out_colour | in_colour).

        Counts CHANGED support cells only (input colour a -> output colour b, a != b), row-normalizes
        per input colour, then gathers each target cell's row by its INPUT colour. Cells whose colour
        was never observed changing get an all-zero row (no evidence != identity claim); PAD/EOS cells
        are zeroed. LODO-safe via _active_context_* (the held-out demo never contributes counts).

        M3: the changed-transition counting is now the shared `_changed_transition_counts` engine.
        """
        hint = torch.zeros((batch_size, seq_len, 10), device=device, dtype=torch.float32)
        if batch is None:
            return hint
        ci = batch.get("_active_context_inputs", batch.get("context_inputs"))
        co = batch.get("_active_context_outputs", batch.get("context_outputs"))
        ti = batch.get("inputs")
        if ci is None or co is None or ti is None or ti.ndim != 2:
            return hint
        with torch.no_grad():
            x = ci.long()
            y = co.long()
            cm = batch.get("_active_context_mask", batch.get("context_mask"))
            demo_ok = cm.to(torch.bool) if cm is not None else torch.ones(x.shape[:2], dtype=torch.bool, device=device)
            _valid, changed, _copied, src, dst = self._support_transition_masks(x, y, demo_ok)
            cond = self._changed_transition_counts(changed, src, dst, batch_size, device)
            cond = cond / cond.sum(dim=-1, keepdim=True).clamp_min(1.0)              # zero rows stay zero
            tin = (ti.long() - 2).clamp(0, 9)
            hint = cond.gather(1, tin.unsqueeze(-1).expand(-1, -1, 10))              # [B, L, 10]
            hint = hint * (ti >= 2).unsqueeze(-1).to(hint.dtype)
        return hint

    def _algo_where_maps(
        self,
        batch: Dict[str, torch.Tensor] | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """FIX C: [B,L,21] algorithmic WHERE/VALUE maps from cell_conditioning_signature cols 11/12.

        ch0 = enclosed (flood-fill: bg component not touching the border);
        ch1..10 = enclosing-colour one-hot; ch11..20 = nearest-seed-colour one-hot.
        Target INPUT only (never output), no_grad, evidence columns only -- never a writer.
        Sentinel 10 (no enclosure / no seed) maps to all-zero one-hots = natural confidence gate.
        """
        zeros = torch.zeros((batch_size, seq_len, ALGO_WHERE_MAP_DIM), device=device, dtype=torch.float32)
        ti = batch.get("inputs") if batch is not None else None
        if ti is None or ti.shape[0] != batch_size or ti.shape[-1] != seq_len:
            return zeros
        side = int(math.isqrt(int(seq_len)))
        if side * side != int(seq_len):
            return zeros
        from models.recursive_reasoning.relation_map import cell_conditioning_signature   # M11
        with torch.no_grad():
            csig, _valid = cell_conditioning_signature(ti.reshape(-1, seq_len).to("cpu"), side)
            if csig.shape[-1] < 13:
                return zeros
            encl = csig[..., 11].to(device)                                  # 0..9 colour, 10 sentinel
            seed = csig[..., 12].to(device)
            maps = zeros.clone()
            maps[..., 0] = (encl != 10).float()
            encl_oh = F.one_hot(encl.clamp(0, 10), num_classes=11)[..., :10].float()
            seed_oh = F.one_hot(seed.clamp(0, 10), num_classes=11)[..., :10].float()
            maps[..., 1:11] = encl_oh
            maps[..., 11:21] = seed_oh
            return maps

    def _evidence_algo_touch(
        self,
        batch: Dict[str, torch.Tensor] | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """D6 (B13 close-out): [B,L,14] touch-colour evidence from cell_conditioning_signature cols 14/15.

        ch0..9 = touch_colour_mode one-hot (modal non-bg colour 4-adjacent to this cell's object;
        bg cells inherit their enclosing object's value; sentinel 10 = no touch -> all-zero);
        ch10..13 = touch_colour_count bucket one-hot (0/1/2/3+; sentinel 4 = no attribution ->
        all-zero). The substrate of the neighbour-conditioned recolor primitive: the signature has
        known "what TOUCHES my object" since File #1 (B13) -- nothing read it until this column.
        Target INPUT only (never any output => LODO/holdout-safe by construction), no_grad,
        evidence columns only -- never a writer. Same delivery pattern as _algo_where_maps.
        """
        zeros = torch.zeros((batch_size, seq_len, 14), device=device, dtype=torch.float32)
        ti = batch.get("inputs") if batch is not None else None
        if ti is None or ti.shape[0] != batch_size or ti.shape[-1] != seq_len:
            return zeros
        side = int(math.isqrt(int(seq_len)))
        if side * side != int(seq_len):
            return zeros
        from models.recursive_reasoning.relation_map import cell_conditioning_signature   # M11
        with torch.no_grad():
            csig, _valid = cell_conditioning_signature(ti.reshape(-1, seq_len).to("cpu"), side)
            if csig.shape[-1] < 16:                       # signature predates cols 14/15 -> no evidence
                return zeros
            touch = csig[..., 14].to(device)              # 0..9 modal touching colour, 10 = none
            cnt = csig[..., 15].to(device)                # 0..3 distinct-count bucket, 4 = none
            maps = zeros.clone()
            maps[..., 0:10] = F.one_hot(touch.clamp(0, 10), num_classes=11)[..., :10].float()
            maps[..., 10:14] = F.one_hot(cnt.clamp(0, 4), num_classes=5)[..., :4].float()
            return maps

    def _evidence_kinematic(
        self, batch: Dict[str, torch.Tensor] | None, batch_size: int, seq_len: int, device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """E-5 / A3: [B, L, KINEMATIC_DIM] per-cell kinematic facts -- mover mask, signed (dr,dc),
        blocked-direction bits (see core_prior.evidence_kinematics). Movement cols live only under a
        cross-demo-consistent verified binding; blocked bits are target-input geometry. LODO-safe:
        reads the `_active_context_*` support. Per-row CPU under no_grad (same lane as D1/D2)."""
        out = torch.zeros((batch_size, seq_len, KINEMATIC_DIM), device=device, dtype=torch.float32)
        stats: Dict[str, torch.Tensor] = {}
        ci, co, cm = self._active_context(batch)
        ti = batch.get("inputs") if batch is not None else None
        if ci is None or co is None or ti is None or ti.ndim != 2:
            return out, stats
        from models.recursive_reasoning.core_prior import evidence_kinematics
        side = int(math.isqrt(seq_len))
        conf_sum = 0.0
        with torch.no_grad():
            ci_c = ci.detach().to("cpu", torch.long); co_c = co.detach().to("cpu", torch.long)
            ti_c = ti.detach().to("cpu", torch.long)
            cm_c = (cm.detach().to("cpu").bool() if cm is not None
                    else torch.ones(ci.shape[:2], dtype=torch.bool))
            for b in range(batch_size):
                keep = cm_c[b].nonzero(as_tuple=True)[0]
                if keep.numel() == 0:
                    continue
                grid, conf, _prov = evidence_kinematics(ci_c[b][keep], co_c[b][keep], ti_c[b], side)
                if conf > 0:
                    out[b] = grid.to(device)
                conf_sum += float(conf)
        stats["kin_mover_mass"] = out[..., 0].mean().detach()
        stats["kin_conf"] = torch.tensor(conf_sum / max(1, batch_size))
        return out, stats

    def _value_context_signature(
        self,
        tokens: torch.Tensor,
        rel_maps: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Compact per-cell context bucket for VALUE binding.

        Default: derived from existing ObjectBank relational maps (background-role, inside-container,
        boundary, component-size bucket, edge-nearness). Missing relmaps -> bucket 0 (backs off to
        source-colour marginals). If c2_value_v2_rich_ctx: use `cell_conditioning_signature` (Fix 2)
        -- sorted 4-nbr colours + enclosing-object size-rank/holes/D4-shape + container colour --
        hashed into the bucket range. (Fix-1 probe: this key caps multi-target at ~34%; banked, not a
        solve.) Handles both [B,L] (target) and [B,M,L] (support) shapes.
        """
        if bool(getattr(self.config, "c2_value_v2_backoff", False)):
            # M1/D4: COLLISION-FREE bounded context key (NO modulo). The rich-ctx path below folds a
            # large mixed-radix hash into 512 buckets via `% 512`, silently MERGING unrelated contexts
            # (measured corruption of the conditioned table). This key is a bounded product of the two
            # value-relevant signature cols -- enclosure colour (0..10) x nearest-seed colour (0..10) x
            # background-role bit -- so max bucket = (10*11+10)*2+1 = 241 < 512 and DISTINCT contexts map
            # to DISTINCT buckets by construction. It composes with the existing ctx->src-marginal alpha
            # backoff in _value_evidence_v2 (the "full -> src marginal" chain). Same ~34% ceiling caveat.
            from models.recursive_reasoning.relation_map import cell_conditioning_signature   # M11
            with torch.no_grad():
                orig = tokens.shape
                L = orig[-1]
                side = int(math.isqrt(int(L)))
                csig, _valid = cell_conditioning_signature(tokens.reshape(-1, L).to("cpu"), side)
                csig = csig.to(device)                                       # [N, L, C] (N = prod(orig[:-1]))
                zero = torch.zeros_like(csig[..., 0])                        # [N, L]
                encl = csig[..., 11].clamp(0, 10) if csig.shape[-1] >= 13 else zero
                seed = csig[..., 12].clamp(0, 10) if csig.shape[-1] >= 13 else zero
                bg = zero
                if (rel_maps is not None and rel_maps.shape[-1] >= REL_MAP_CHANNELS
                        and rel_maps.shape[:-1] == tokens.shape):
                    rm_flat = rel_maps.to(device=device, dtype=torch.float32).reshape(-1, L, rel_maps.shape[-1])
                    bg = (rm_flat[..., 1] > 0.5).long()                      # [N, L], same flat frame as csig
                bucket = (encl * 11 + seed) * 2 + bg                         # <= 241 < 512, injective
                bucket = torch.where(csig[..., 0] >= 0, bucket, zero)        # no-colour -> bucket 0 (marginal)
                return bucket.reshape(orig)
        if bool(getattr(self.config, "c2_value_v2_rich_ctx", False)):
            from models.recursive_reasoning.relation_map import cell_conditioning_signature   # M11
            with torch.no_grad():
                orig = tokens.shape
                L = orig[-1]
                side = int(math.isqrt(int(L)))
                flat = tokens.reshape(-1, L).to("cpu")                      # signature fn is CPU/Python
                csig, _valid = cell_conditioning_signature(flat, side)      # [N, L, CELL_SIG_DIM] long
                csig = csig.to(device)
                # hash informative columns (4-nbr colours, size-rank, holes, container colour) into the
                # bucket range via mixed radix + modulo. nbr in 0..10, rank 0..8, holes 0..4, col 0..10.
                nb = ((csig[..., 1] * 11 + csig[..., 2]) * 11 + csig[..., 3]) * 11 + csig[..., 4]
                h = ((nb * 9 + csig[..., 5].clamp(0, 8)) * 5 + csig[..., 6].clamp(0, 4)) * 11 + csig[..., 10]
                if csig.shape[-1] >= 13:
                    # FIX H cols: flood-fill enclosure colour + nearest-seed colour (0..10 each)
                    h = (h * 11 + csig[..., 11]) * 11 + csig[..., 12]
                bucket = (h % VALUE_EVIDENCE_V2_CONTEXT_BUCKETS).long()
                # cells with no colour (self_color == -1) collapse to bucket 0 -> marginal backoff
                bucket = torch.where(csig[..., 0] >= 0, bucket, torch.zeros_like(bucket))
                return bucket.reshape(orig)
        sig = torch.zeros(tokens.shape, device=device, dtype=torch.long)
        if rel_maps is None or rel_maps.shape[-1] < REL_MAP_CHANNELS:
            return sig
        rm = rel_maps.to(device=device, dtype=torch.float32)
        if rm.shape[:-1] != tokens.shape:
            return sig
        bg = (rm[..., 1] > 0.5).long()
        size_raw = rm[..., 2]
        size_bucket = torch.where(
            size_raw < 0.08,
            torch.zeros_like(sig),
            torch.where(size_raw < 0.20, torch.ones_like(sig), torch.full_like(sig, 2)),
        )
        boundary = (rm[..., 5] > 0.5).long()
        edge = (rm[..., 6:10].amin(dim=-1) <= 0.05).long()
        inside = (rm[..., 11] > 0.5).long()
        sig = (((bg * 2 + inside) * 2 + boundary) * 3 + size_bucket) * 2 + edge
        return sig.clamp_(0, VALUE_EVIDENCE_V2_CONTEXT_BUCKETS - 1)

    def _value_evidence_v2(
        self,
        batch: Dict[str, torch.Tensor] | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        rel_maps: torch.Tensor | None = None,
        rel_where_hint: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
        """Copy-vs-change and context-conditioned VALUE evidence [B, L, 36].

        Layout:
          [0:10]   copy_dist = one_hot(input_colour) * P(copy | source_colour)
          [10:20]  conditioned_dist = P(changed output_colour | source_colour, context), backed off
                   to P(changed output_colour | source_colour) when the context bucket is sparse
          [20:26]  scalars: change_rate, copy_rate, support_conf, changed_support_conf,
                   entropy_conf, top_margin
          [26:36]  rel_where_hint * conditioned_dist

        This is evidence only. It reads active LODO support when present and never reads target output
        to build features.

        Returns (features, stats, ctx_gate, ctx_bind). `ctx_gate` (D7, `c2_value_ctx_gate`) is a
        separate [B,L,2] block -- P(change|src,ctx), P(copy|src,ctx) with source-marginal backoff --
        or None when its flag is off. `ctx_bind` (D11/codex, `c2_value_ctx_bind`) is a separate
        [B,L,20] block of EXPLICIT gate x value products (change_value[10] | copy_value[10]): the
        linear evidence proj can weight columns but never MULTIPLY them, so the finished
        recommendation must arrive as a column. Both ride this method's EXISTING context pass (no
        duplicate signature compute, M3 spirit) and are deliberately NOT folded into `features`, so
        value_v2's 36 columns stay byte-identical whichever flags are on (each is its own EvCol block).
        """
        features = torch.zeros((batch_size, seq_len, VALUE_EVIDENCE_V2_DIM), device=device, dtype=torch.float32)
        zero_stats = {
            "c2_value_v2_change_rate_on_changed": torch.zeros((), device=device),
            "c2_value_v2_change_rate_on_copy": torch.zeros((), device=device),
            "c2_value_v2_copy_rate_on_copy": torch.zeros((), device=device),
            "c2_value_v2_support_coverage": torch.zeros((), device=device),
            "c2_value_v2_entropy_conf": torch.zeros((), device=device),
            "c2_value_v2_margin": torch.zeros((), device=device),
            "c2_value_v2_where_mass": torch.zeros((), device=device),
        }
        if batch is None:
            return features, zero_stats, None, None
        ci = batch.get("_active_context_inputs", batch.get("context_inputs"))
        co = batch.get("_active_context_outputs", batch.get("context_outputs"))
        ti = batch.get("inputs")
        if ci is None or co is None or ti is None or ti.ndim != 2:
            return features, zero_stats, None, None
        with torch.no_grad():
            x = ci.long().to(device)
            y = co.long().to(device)
            target = ti.long().to(device)
            cm = batch.get("_active_context_mask", batch.get("context_mask"))
            demo_ok = cm.to(device=device, dtype=torch.bool) if cm is not None else torch.ones(
                x.shape[:2], dtype=torch.bool, device=device)
            valid, changed, copied, src, dst = self._support_transition_masks(x, y, demo_ok)

            # Marginal support by source colour.
            src_flat = src.reshape(batch_size, -1)
            total_by_src = torch.zeros((batch_size, 10), device=device, dtype=torch.float32)
            changed_by_src = torch.zeros_like(total_by_src)
            copy_by_src = torch.zeros_like(total_by_src)
            total_by_src.scatter_add_(1, src_flat, valid.reshape(batch_size, -1).float())
            changed_by_src.scatter_add_(1, src_flat, changed.reshape(batch_size, -1).float())
            copy_by_src.scatter_add_(1, src_flat, copied.reshape(batch_size, -1).float())

            marginal_dist = self._changed_transition_counts(changed, src, dst, batch_size, device)  # M3
            marginal_dist = marginal_dist / marginal_dist.sum(dim=-1, keepdim=True).clamp_min(1.0)

            # Context-conditioned changed counts. Missing relmaps intentionally collapse to ctx=0.
            support_rel_maps = batch.get("_active_context_rel_maps", batch.get("context_rel_maps"))
            if support_rel_maps is not None:
                support_rel_maps = support_rel_maps.to(device)
            support_ctx = self._value_context_signature(x, support_rel_maps, device)
            ctx_key = (support_ctx * 100 + src * 10 + dst).reshape(batch_size, -1)
            ctx_counts = torch.zeros(
                (batch_size, VALUE_EVIDENCE_V2_CONTEXT_BUCKETS * 100),
                device=device,
                dtype=torch.float32,
            )
            ctx_counts.scatter_add_(1, ctx_key, changed.reshape(batch_size, -1).float())
            ctx_dist = ctx_counts.view(batch_size, VALUE_EVIDENCE_V2_CONTEXT_BUCKETS, 10, 10)
            ctx_support = ctx_dist.sum(dim=-1, keepdim=True)
            ctx_dist = ctx_dist / ctx_support.clamp_min(1.0)

            target_src = (target - 2).clamp(0, 9)
            target_ctx = self._value_context_signature(target, rel_maps, device)
            row = (target_ctx * 10 + target_src).clamp(0, VALUE_EVIDENCE_V2_CONTEXT_BUCKETS * 10 - 1)
            conditioned = ctx_dist.view(batch_size, VALUE_EVIDENCE_V2_CONTEXT_BUCKETS * 10, 10).gather(
                1, row.unsqueeze(-1).expand(-1, -1, 10))
            cond_n = ctx_support.view(batch_size, VALUE_EVIDENCE_V2_CONTEXT_BUCKETS * 10, 1).gather(
                1, row.unsqueeze(-1))
            marginal = marginal_dist.gather(1, target_src.unsqueeze(-1).expand(-1, -1, 10))
            alpha = (cond_n / 3.0).clamp(0.0, 1.0)
            conditioned = alpha * conditioned + (1.0 - alpha) * marginal

            gathered_total = total_by_src.gather(1, target_src)
            gathered_changed = changed_by_src.gather(1, target_src)
            gathered_copy = copy_by_src.gather(1, target_src)
            change_rate = gathered_changed / gathered_total.clamp_min(1.0)
            copy_rate = gathered_copy / gathered_total.clamp_min(1.0)
            denom = math.log1p(float(max(1, x.shape[1] * x.shape[2])))
            support_conf = torch.log1p(gathered_total) / denom
            changed_support_conf = torch.log1p(gathered_changed) / denom

            p = conditioned.clamp_min(1e-8)
            entropy_conf = 1.0 + (p * p.log()).sum(dim=-1) / math.log(10.0)
            top2 = torch.topk(conditioned, k=2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]
            copy_dist = F.one_hot(target_src, num_classes=10).to(torch.float32) * copy_rate.unsqueeze(-1)
            where = torch.zeros((batch_size, seq_len, 1), device=device, dtype=torch.float32)
            if rel_where_hint is not None:
                where = rel_where_hint.to(device=device, dtype=torch.float32)
                if where.ndim == 2:
                    where = where.unsqueeze(-1)
                # FIX D: topk>1 widens the hint; the v2 WHERE gate stays the best predicate (ch 0).
                if where.shape[-1] > 1:
                    where = where[..., :1]
            valid_target = (target >= 2).unsqueeze(-1).float()

            features[..., 0:10] = copy_dist
            features[..., 10:20] = conditioned
            features[..., 20:21] = change_rate.unsqueeze(-1)
            features[..., 21:22] = copy_rate.unsqueeze(-1)
            features[..., 22:23] = support_conf.unsqueeze(-1)
            features[..., 23:24] = changed_support_conf.unsqueeze(-1)
            features[..., 24:25] = entropy_conf.unsqueeze(-1)
            features[..., 25:26] = margin.unsqueeze(-1)
            features[..., 26:36] = where * conditioned
            features = features * valid_target

            # --- D7 (user): context-conditioned copy/change GATE, own EvCol block (default off) ------
            # Closes value_v2's asymmetry: the changed-colour DISTRIBUTION [10:20] is context-aware but
            # the copy/change RATES [20:22] are source-only. Whether a cell CHANGES AT ALL depends on
            # context (red-inside-box changes; red-in-bg copies). Reuses support_ctx/target_ctx +
            # valid/copied from THIS pass; only +2 scatter-adds (total + copy by a (ctx,src) key).
            # D11 (codex): the BIND block rides the same gate math -- computed when EITHER flag is on,
            # but each output block is emitted ONLY under its own flag (step-0 decoupling, D7 pattern).
            ctx_gate: torch.Tensor | None = None
            ctx_bind: torch.Tensor | None = None
            _want_gate = bool(getattr(self.config, "c2_value_ctx_gate", False))
            _want_bind = bool(getattr(self.config, "c2_value_ctx_bind", False))
            if _want_gate or _want_bind:
                csrc_key = (support_ctx * 10 + src).reshape(batch_size, -1)
                gate_total = torch.zeros(
                    (batch_size, VALUE_EVIDENCE_V2_CONTEXT_BUCKETS * 10), device=device, dtype=torch.float32)
                gate_copy = torch.zeros_like(gate_total)
                gate_total.scatter_add_(1, csrc_key, valid.reshape(batch_size, -1).float())
                gate_copy.scatter_add_(1, csrc_key, copied.reshape(batch_size, -1).float())
                grow = (target_ctx * 10 + target_src).clamp(0, VALUE_EVIDENCE_V2_CONTEXT_BUCKETS * 10 - 1)
                tot = gate_total.gather(1, grow)                                   # [B, L] per-(ctx,src) support
                cop = gate_copy.gather(1, grow)
                g_alpha = (tot / 3.0).clamp(0.0, 1.0)                              # same sparse-backoff as `alpha`
                p_change_ctx = (tot - cop) / tot.clamp_min(1.0)                    # changed = total - copied
                p_copy_ctx = cop / tot.clamp_min(1.0)
                p_change = g_alpha * p_change_ctx + (1.0 - g_alpha) * change_rate  # backoff to source-marginal rate
                p_copy = g_alpha * p_copy_ctx + (1.0 - g_alpha) * copy_rate
                if _want_gate:
                    ctx_gate = torch.stack((p_change, p_copy), dim=-1) * valid_target  # [B, L, 2]
                if _want_bind:
                    # D11 (codex): EXPLICIT copy/change VALUE binding. color_evidence_proj is LINEAR --
                    # it can weight the gate (2 cols) and the distribution (10 cols) but can never
                    # MULTIPLY them; the finished recommendation must arrive as a column:
                    #   change_value[10] = P(change|src,ctx) * P(dst|src,ctx)   (what it becomes, gated)
                    #   copy_value[10]   = P(copy|src,ctx)   * one_hot(src)     (keep-as-is, gated)
                    change_value = (p_change.unsqueeze(-1) * conditioned) * valid_target
                    copy_value = (p_copy.unsqueeze(-1)
                                  * F.one_hot(target_src, num_classes=10).to(torch.float32)) * valid_target
                    ctx_bind = torch.cat((change_value, copy_value), dim=-1)       # [B, L, 20]

            stats = dict(zero_stats)
            labels = batch.get("labels")
            if labels is not None and labels.ndim == 2 and labels.shape == target.shape:
                lab = labels.long().to(device)
                colour = (lab >= 2) & (target >= 2)
                true_changed = colour & (lab != target)
                true_copy = colour & (lab == target)

                def _mean_on(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
                    return v[m].mean() if bool(m.any()) else torch.zeros((), device=device)

                stats["c2_value_v2_change_rate_on_changed"] = _mean_on(change_rate, true_changed)
                stats["c2_value_v2_change_rate_on_copy"] = _mean_on(change_rate, true_copy)
                stats["c2_value_v2_copy_rate_on_copy"] = _mean_on(copy_rate, true_copy)
                stats["c2_value_v2_support_coverage"] = _mean_on((gathered_total > 0).float(), colour)
            stats["c2_value_v2_entropy_conf"] = entropy_conf[target >= 2].mean() if bool((target >= 2).any()) else torch.zeros((), device=device)
            stats["c2_value_v2_margin"] = margin[target >= 2].mean() if bool((target >= 2).any()) else torch.zeros((), device=device)
            stats["c2_value_v2_where_mass"] = features[..., 26:36].sum(dim=-1)[target >= 2].mean() if bool((target >= 2).any()) else torch.zeros((), device=device)
        return features, stats, ctx_gate, ctx_bind

    def _canonical_value_binding(
        self,
        batch: Dict[str, torch.Tensor] | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """One support-derived colour authority with exact task-local context keys.

        The binder is deliberately same-position only. Independent operation factors route it off
        for movement or extent-changing tasks, preventing a recolour table from writing values into
        cells whose source object came from another position.
        """
        distribution = torch.zeros((batch_size, seq_len, 10), device=device, dtype=torch.float32)
        supported = torch.zeros((batch_size, seq_len), device=device, dtype=torch.bool)
        changed_supported = torch.zeros_like(supported)
        copy_supported = torch.zeros_like(supported)
        # P1: distribution / route / reliability are carried SEPARATELY -- the route no longer
        # multiplies the distribution (confidence, not probability mass), so P_bind stays a
        # normalized simplex and downstream consumers decide applicability themselves.
        marginal = torch.zeros_like(distribution)
        route_map = torch.zeros((batch_size, seq_len), device=device, dtype=torch.float32)
        reliability = torch.zeros_like(route_map)
        per_cell = {
            "marginal": marginal,
            "route": route_map,
            "reliability": reliability,
        }
        stats = {
            "c2_canonical_bind_same_position_route": torch.zeros((), device=device),
            "c2_canonical_bind_same_extent": torch.zeros((), device=device),
            "c2_canonical_bind_histogram_change": torch.zeros((), device=device),
            "c2_canonical_bind_movement": torch.zeros((), device=device),
            "c2_canonical_bind_support_coverage": torch.zeros((), device=device),
            "c2_canonical_bind_changed_support_coverage": torch.zeros((), device=device),
            "c2_canonical_bind_copy_support_coverage": torch.zeros((), device=device),
            "c2_canonical_bind_key_collisions": torch.zeros((), device=device),
        }
        ci, co, cm = self._active_context(batch)
        ti = batch.get("inputs") if batch is not None else None
        if ci is None or co is None or ti is None or ti.ndim != 2:
            return distribution, stats, supported, changed_supported, copy_supported, per_cell

        from models.recursive_reasoning.relation_map import (
            cell_conditioning_signature,
            hierarchical_value_binding,
        )
        from models.recursive_reasoning.core_prior import (
            RULE_FACTOR_INDEX,
            evidence_rule_factors,
        )
        side = int(math.isqrt(seq_len))
        with torch.no_grad():
            ci_cpu = ci.detach().to("cpu", torch.long)
            co_cpu = co.detach().to("cpu", torch.long)
            ti_cpu = ti.detach().to("cpu", torch.long)
            cm_cpu = (cm.detach().to("cpu", torch.bool)
                      if cm is not None else torch.ones(ci.shape[:2], dtype=torch.bool))
            support_sig, support_sig_valid = cell_conditioning_signature(
                ci_cpu.reshape(-1, seq_len), side)
            support_sig = support_sig.view(batch_size, ci.shape[1], seq_len, -1)
            support_sig_valid = support_sig_valid.view(batch_size, ci.shape[1], seq_len)
            target_sig, target_valid = cell_conditioning_signature(ti_cpu, side)

            routes = []
            same_extents = []
            hist_changes = []
            movements = []
            collision_total = 0
            tau = float(getattr(self.config, "c2_value_backoff_tau", 3.0))
            for b in range(batch_size):
                demo_mask = cm_cpu[b]
                support_valid = (
                    support_sig_valid[b]
                    & (co_cpu[b] >= 2)
                    & demo_mask.unsqueeze(-1)
                )
                support_changed = support_valid & (ci_cpu[b] != co_cpu[b])
                bind = hierarchical_value_binding(
                    support_sig[b],
                    (co_cpu[b] - 2).clamp(0, 9),
                    support_valid,
                    support_changed,
                    target_sig[b],
                    target_valid[b],
                    tau=tau,
                )
                keep = demo_mask.nonzero(as_tuple=True)[0]
                if keep.numel() > 0:
                    factors = evidence_rule_factors(
                        ci_cpu[b][keep], co_cpu[b][keep], side)
                    scores = factors["scores"].float()
                    p_same_extent = float(scores[RULE_FACTOR_INDEX["extent_same"]])
                    p_hist_change = float(scores[RULE_FACTOR_INDEX["colour_recolour"]])
                    p_movement = float(sum(
                        scores[RULE_FACTOR_INDEX[name]]
                        for name in ("move_up", "move_down", "move_left", "move_right")
                    ).clamp(0.0, 1.0))
                else:
                    p_same_extent = p_hist_change = p_movement = 0.0
                route = max(0.0, min(1.0, p_same_extent * p_hist_change * (1.0 - p_movement)))
                # P1: raw normalized distribution -- NO route multiply, NO route>0 support gating.
                # Route rides separately as task-level applicability confidence.
                distribution[b] = bind["distribution"].to(device)
                marginal[b] = bind["marginal_distribution"].to(device)
                route_map[b] = route
                reliability[b] = bind["support_reliability"].to(device)
                target_is_valid = target_valid[b].to(device)
                changed_supported[b] = bind["changed_supported"].to(device) & target_is_valid
                copy_supported[b] = bind["copy_supported"].to(device) & target_is_valid
                supported[b] = changed_supported[b] | copy_supported[b]
                routes.append(route)
                same_extents.append(p_same_extent)
                hist_changes.append(p_hist_change)
                movements.append(p_movement)
                collision_total += int(bind["collision_count"])

            def mean_scalar(values: list[float]) -> torch.Tensor:
                return torch.tensor(sum(values) / max(len(values), 1), device=device)

            stats["c2_canonical_bind_same_position_route"] = mean_scalar(routes)
            stats["c2_canonical_bind_same_extent"] = mean_scalar(same_extents)
            stats["c2_canonical_bind_histogram_change"] = mean_scalar(hist_changes)
            stats["c2_canonical_bind_movement"] = mean_scalar(movements)
            valid_target = ti.to(device) >= 2
            stats["c2_canonical_bind_support_coverage"] = (
                supported.float().sum() / valid_target.float().sum().clamp_min(1.0))
            stats["c2_canonical_bind_changed_support_coverage"] = (
                changed_supported.float().sum() / valid_target.float().sum().clamp_min(1.0))
            stats["c2_canonical_bind_copy_support_coverage"] = (
                copy_supported.float().sum() / valid_target.float().sum().clamp_min(1.0))
            stats["c2_canonical_bind_key_collisions"] = torch.tensor(
                float(collision_total), device=device)
        return distribution, stats, supported, changed_supported, copy_supported, per_cell

    @staticmethod
    def _rule_factor_batch(
        context_inputs: torch.Tensor,
        context_outputs: torch.Tensor,
        context_mask: torch.Tensor,
        side: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Batch the core-prior independent factor API for neural routing."""
        from models.recursive_reasoning.core_prior import (
            RULE_FACTOR_INDEX,
            evidence_rule_factors,
        )
        B = context_inputs.shape[0]
        scores = torch.zeros((B, RULE_FACTOR_DIM), device=device, dtype=torch.float32)
        # Missing support defaults to the conservative identity route.
        for name in ("extent_same", "colour_copy", "move_none", "count_preserved"):
            scores[:, RULE_FACTOR_INDEX[name]] = 1.0
        spatial = torch.zeros((B, 8), device=device, dtype=torch.float32)
        ci = context_inputs.detach().to("cpu", torch.long)
        co = context_outputs.detach().to("cpu", torch.long)
        cm = context_mask.detach().to("cpu", torch.bool)
        with torch.no_grad():
            for b in range(B):
                keep = cm[b].nonzero(as_tuple=True)[0]
                if keep.numel() == 0:
                    continue
                result = evidence_rule_factors(ci[b][keep], co[b][keep], side)
                scores[b] = result["scores"].to(device)
                spatial[b] = torch.stack((
                    result["dominant_dy"], result["dominant_dx"],
                    result["direction_consistency"], result["bbox_dh"], result["bbox_dw"],
                    result["same_shape_transport"], result["creation_confidence"],
                    result["deletion_confidence"],
                )).to(device)
        stats = {
            "c2_rule_factor_same_extent": scores[:, RULE_FACTOR_INDEX["extent_same"]].mean().detach(),
            "c2_rule_factor_recolour": scores[:, RULE_FACTOR_INDEX["colour_recolour"]].mean().detach(),
            "c2_rule_factor_move": scores[:, [
                RULE_FACTOR_INDEX["move_up"], RULE_FACTOR_INDEX["move_down"],
                RULE_FACTOR_INDEX["move_left"], RULE_FACTOR_INDEX["move_right"],
            ]].sum(dim=-1).clamp(0.0, 1.0).mean().detach(),
            "c2_rule_factor_match_coverage": spatial[:, 5].mean().detach(),
            "c2_rule_factor_direction_consistency": spatial[:, 2].mean().detach(),
        }
        return scores, stats


# ======================================================================================
# section 7 -- MODEL INIT (heads built FROM THE SCHEMA)   (ROLE: MODEL INIT)
# section 6 -- INPUT HINTS (broadcast injectors + _condition_grid_features)   (ROLE: INPUT)
# section 8 -- LODO machinery   (ROLE: LODO)
# section 9 -- OUTPUT (_output_logits + forward)   (ROLE: OUTPUT)
# These are all methods of the ONE Inner class (single-class layout preserved for checkpoint/module
# compatibility); the section tags below mark where each role lives inside it. Inner INHERITS the §5
# evidence builders from C2EvidenceMixin -- identical method resolution to the old monolithic class.
#
# Block 4+5 scope: heads-from-schema (M4: extra_cols == evidence_total; quarantine warm-init by QuarCol
# name), input consolidation (M8 `_add_broadcast_hint`, M9 `_active_context`). Everything else is a
# VERBATIM port. Deferred by design: the §9 decomposition + M2 (P_off self-stash -> extras) land in
# Block 6; the D-block evidence appends (D7 value_ctx_gate / D1 verified_frame / D2 analogy) land in
# Block 8. Until then those flags stay default-off and the width assert guards against a half-wired ON.
# ======================================================================================
class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module, C2EvidenceMixin):
    def __init__(self, config: FVR_C2_Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        if self.config.c2_visual_encoder:
            # The cached-patch visual lane (HybridGridEncoder/CachedPatchGridEncoder) was REMOVED
            # (plan §12.2: visual = throw). Raise loudly instead of NameError-ing on a stale config.
            raise ValueError(
                "c2_visual_encoder=True but the cached-patch visual encoder lane was removed; "
                "unset the flag (token embeddings are the only grid encoder).")
        self.grid_encoder = TokenGridEncoder(self.embed_tokens, self.embed_scale)
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if self.config.c2_dual_output_head:
            # FIX A (2026-07-05): evidence columns live in a SEPARATE zero-init color_evidence_proj (own
            # optimizer group, --evidence-lr wd=0). Welded inside color_head they were pinned to the
            # core lr and measured inert. Function class identical: cat-linear == sum of two linears.
            # M4: the evidence width is now the SCHEMA (EVIDENCE_COLS) via evidence_total -- ONE source
            # of truth for the layout hand-synced across 4 sites. At legacy (new-flags-off) configs this
            # equals the old extra_cols arithmetic exactly (asserted, Block 1 gate).
            extra_cols = evidence_total(self.config)
            self.color_evidence_dim = extra_cols
            self.color_head = CastedLinear(self.config.hidden_size, 10, bias=False)
            if extra_cols:
                self.color_evidence_proj = CastedLinear(extra_cols, 10, bias=False)
                with torch.no_grad():
                    self.color_evidence_proj.weight.zero_()
            # Interaction MLP residual (c2_color_head_mlp_dim > 0): reads the FULL concat
            # [grid_z | evidence], zero-init output layer -> step-0 == the warm-started linear head
            # exactly; capacity for input-colour x evidence products the linear map cannot represent.
            _mlp_d = int(getattr(self.config, "c2_color_head_mlp_dim", 0))
            if _mlp_d > 0:
                self.color_head_mlp_in = CastedLinear(self.config.hidden_size + extra_cols, _mlp_d, bias=True)
                self.color_head_mlp_out = CastedLinear(_mlp_d, 10, bias=False)
                with torch.no_grad():
                    self.color_head_mlp_out.weight.zero_()
        # PID-QUARANTINED candidate head. Feature layout MUST match _quarantine_features / QuarCol.
        # Warm-init = copy-unless-consensus: logit(c) gets +4 from "input IS colour c", +8 from "demo
        # marginal consensus says c", +9 from "conditioned consensus says c" (backoff makes the
        # conditioned column equal the marginal when the context bucket is sparse). M4: the warm-init
        # columns are addressed BY NAME (QuarCol.*) not the old bare integers 2+c / 108+c / 130+c.
        if getattr(self.config, "c2_quarantine_candidate", False):
            if not getattr(self.config, "c2_floor_candidate_split", False):
                import warnings
                warnings.warn(
                    "c2_quarantine_candidate=True but c2_floor_candidate_split=False: the quarantined "
                    "head only feeds the CANDIDATE lane; without the split it is INERT. Enable "
                    "--floor-candidate-split.", RuntimeWarning, stacklevel=2)
            _q_in = self._quarantine_total()          # QuarCol.TOTAL (+11 verified-frame block under D3)
            _q_h = int(getattr(self.config, "c2_quarantine_hidden", 256))
            self.quarantine_lin = CastedLinear(_q_in, 10, bias=False)
            self.quarantine_mlp_in = CastedLinear(_q_in, _q_h, bias=True)
            self.quarantine_mlp_out = CastedLinear(_q_h, 10, bias=False)
            with torch.no_grad():
                self.quarantine_lin.weight.zero_()
                self.quarantine_mlp_out.weight.zero_()
                for _c in range(10):
                    self.quarantine_lin.weight[_c, QuarCol.INPUT_ONEHOT + 2 + _c] = 4.0  # copy: input colour c -> logit c
                    self.quarantine_lin.weight[_c, QuarCol.TRANSITION + _c] = 8.0        # marginal consensus P(out=c|in)
                    self.quarantine_lin.weight[_c, QuarCol.CONDITIONED + _c] = 9.0       # conditioned consensus P(out=c|in,ctx)
                # D3: verified-frame block warm-init +10 > conditioned +9 > marginal +8 > copy +4 -- a
                # demo-EXACT frame beats every statistical column at step 0. NOT step-0 inert (deliberate),
                # but only fires where a frame verifies (the block is zero elsewhere -> no change there).
                if getattr(self.config, "c2_verified_frame_evidence", False):
                    for _c in range(10):
                        self.quarantine_lin.weight[_c, QuarCol.TOTAL + _c] = 10.0
        if getattr(self.config, "c2_relmap", False):
            self.relmap_proj = CastedLinear(REL_MAP_CHANNELS, self.config.hidden_size, bias=False)
            with torch.no_grad():
                self.relmap_proj.weight.zero_()
        # Lane B: zero-init embedding of the deterministic FRAME family (the rule-hypothesis hint). Added
        # to grid_features input-side -> the TRM combines the narrowed operation family with relmaps/C2.
        # Zero-init => step-0 byte-identical (F7-safe); the loss earns the binding.
        if getattr(self.config, "c2_frame_hint", False):
            # M11: FRAME_VOCAB from core_prior (len 11: adds "rotate") vs object_rule_bank (len 10).
            # Checkpoint-safe: frame_embed is fresh per run (init_std=0, zeroed); the extra index is
            # APPENDED, so a warm-start copies the len-10 prefix column-aligned and index 10 stays zero.
            from models.recursive_reasoning.core_prior import FRAME_VOCAB
            self.frame_embed = CastedEmbedding(len(FRAME_VOCAB), self.config.hidden_size,
                                               init_std=0.0, cast_to=self.forward_dtype)
            with torch.no_grad():
                self.frame_embed.embedding_weight.zero_()
        # In-model rule-hypothesis hint: zero-init embedding of the top operation-family inferred live
        # from the support pairs. Zero-init => step-0 byte-identical (F7-safe); the loss earns any signal.
        if getattr(self.config, "c2_rule_hypothesis_hint", False):
            self.rule_hyp_embed = CastedEmbedding(len(RULE_FAMILY_VOCAB), self.config.hidden_size,
                                                  init_std=0.0, cast_to=self.forward_dtype)
            with torch.no_grad():
                self.rule_hyp_embed.embedding_weight.zero_()
        # §15.2-A: separate zero-init projection for SUPPORT-side maps fed into C2's demo features.
        # Zero-init => the demo enrichment contributes nothing at step 0 (F7-safe, baseline unchanged).
        if ((getattr(self.config, "c2_relmap_demos", False)
             or getattr(self.config, "c2_ordered_evidence_flow", False))
                and getattr(self.config, "c2_relmap", False)):
            self.c2_demo_relmap_proj = CastedLinear(REL_MAP_CHANNELS, self.config.hidden_size, bias=False)
            with torch.no_grad():
                self.c2_demo_relmap_proj.weight.zero_()
        # §15.2-B: PairDelta as an INPUT-ONLY zero-init hint (own encoder; independent of the delta branch
        # so it survives V3-clean). delta_rule_input_proj zero-init => no-op at step 0.
        if getattr(self.config, "c2_pairdelta_input_feature", False):
            # File #5 import flip: pair_delta_v2's port is byte-identical + state_dict-compatible
            # (pd_v2_gate.py shared-weight identity), so step-0 is unchanged.
            from models.recursive_reasoning.pair_delta_v2 import PairDeltaEncoder as _PDE
            _enc_dim = int(getattr(self.config, "c2_delta_rule_encoder_dim", 256))
            self.pairdelta_input_encoder = _PDE(
                hidden_dim=_enc_dim, n_slots=int(getattr(self.config, "c2_delta_rule_slots", 8)),
                include_identity=bool(getattr(self.config, "c2_pairdelta_include_identity", False)),
                include_spatial=bool(getattr(self.config, "c2_pairdelta_spatial", False)))
            self.delta_rule_input_proj = CastedLinear(_enc_dim, self.config.hidden_size, bias=False)
            with torch.no_grad():
                self.delta_rule_input_proj.weight.zero_()
        if getattr(self.config, "c2_rule_factor_hint", False):
            self.rule_factor_proj = CastedLinear(RULE_FACTOR_DIM, self.config.hidden_size, bias=False)
            with torch.no_grad():
                self.rule_factor_proj.weight.zero_()
        if self.config.c2_dual_output_head or self.config.c2_geometry_aux_head:
            self.structure_head = CastedLinear(self.config.hidden_size, 3, bias=False)
        # §15.6: let structure_head SEE the relmap (zero-init [REL_MAP_CHANNELS->3] proj added to structure
        # logits). §15.9 boundary lever (bias=True): pad_logit = b - k*valid_mask copies the input boundary
        # instead of over-predicting pad. Zero-init weight+bias => step-0 == floor (F7-safe).
        if (getattr(self.config, "c2_relmap", False)
                and getattr(self.config, "c2_relmap_structure", False)
                and (self.config.c2_dual_output_head or self.config.c2_geometry_aux_head)):
            self.structure_relmap_proj = CastedLinear(REL_MAP_CHANNELS, 3, bias=True)
            with torch.no_grad():
                self.structure_relmap_proj.weight.zero_()
                self.structure_relmap_proj.bias.zero_()
        # §15.9.1 extent PAD override: dedicated [1->3] proj reading extent_pad_mask. Zero-init (F7-safe)
        # UNLESS warm-init (a near-HARD override: pad +V, eos+valid -V; the frozen lm_head's colour-over-pad
        # gap on padding is too large for a small additive lever to flip). Mask is eos-clean.
        if (getattr(self.config, "c2_relmap_outside_grid", False)
                and (self.config.c2_dual_output_head or self.config.c2_geometry_aux_head)):
            self.structure_outside_proj = CastedLinear(1, 3, bias=False)
            with torch.no_grad():
                self.structure_outside_proj.weight.zero_()
                if getattr(self.config, "c2_structure_outside_warm_init", False):
                    _v = float(getattr(self.config, "c2_structure_outside_warm_init_value", 1000.0))
                    self.structure_outside_proj.weight[0, 0] = _v     # PAD up
                    self.structure_outside_proj.weight[1, 0] = -_v    # EOS down
                    self.structure_outside_proj.weight[2, 0] = -_v    # VALID down
        # EOS uses the thin-L boundary geometry (different from PAD): its own [1->3] projection.
        if (getattr(self.config, "c2_relmap_eos_grid", False)
                and (self.config.c2_dual_output_head or self.config.c2_geometry_aux_head)):
            self.structure_eos_proj = CastedLinear(1, 3, bias=False)
            with torch.no_grad():
                self.structure_eos_proj.weight.zero_()
                if getattr(self.config, "c2_structure_eos_warm_init", False):
                    _v = float(getattr(self.config, "c2_structure_eos_warm_init_value", 1000.0))
                    self.structure_eos_proj.weight[0, 0] = -_v     # PAD down
                    self.structure_eos_proj.weight[1, 0] = _v      # EOS up
                    self.structure_eos_proj.weight[2, 0] = -_v     # VALID down
        # D9 (File #5): pair-delta STRUCTURE evidence. The extent engine above only knows
        # {identity, constant, ratio}; pd_structure_evidence adds the verified {preserve,
        # transpose, bbox} family masks as a zero-init [PD_STRUCT_DIM->3] ADDITIVE lever
        # (F7-safe: cannot move anything at step 0; the LODO boundary CE earns it).
        if (getattr(self.config, "c2_pairdelta_structure_evidence", False)
                and (self.config.c2_dual_output_head or self.config.c2_geometry_aux_head)):
            self.structure_pairdelta_proj = CastedLinear(PD_STRUCT_DIM, 3, bias=False)
            with torch.no_grad():
                self.structure_pairdelta_proj.weight.zero_()
        if self.config.c2_shape_head:
            # zH_rowcol adds per-row + per-col occupancy profiles so the head can SEPARATE height from
            # width (a plain grid MEAN pool is permutation-invariant -> dimension-blind -> collapses).
            _shape_extra = 2 * int(math.isqrt(self.config.seq_len)) if self.config.c2_shape_pool == "zH_rowcol" else 0
            self.shape_h_head = CastedLinear(2 * self.config.hidden_size + _shape_extra, 30, bias=True)
            self.shape_w_head = CastedLinear(2 * self.config.hidden_size + _shape_extra, 30, bias=True)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # --- D11 guard (codex): backoff SILENTLY wins over rich-ctx in _value_context_signature
        #     (checked first, returns early). Both on = rich-ctx dead without a trace -- the V3-1
        #     silent-default disease. Warn loudly here; run_stage1_local refuses outright.
        if (getattr(self.config, "c2_value_v2_backoff", False)
                and getattr(self.config, "c2_value_v2_rich_ctx", False)):
            import warnings
            warnings.warn(
                "c2_value_v2_backoff AND c2_value_v2_rich_ctx are both on: backoff takes precedence "
                "and rich-ctx is silently DEAD. Pick one (this is the A/B variable).",
                RuntimeWarning, stacklevel=2)

        if getattr(self.config, "c2_canonical_value_binder", False):
            if int(getattr(self.config, "c2_color_head_mlp_dim", 0)) > 0:
                raise ValueError(
                    "c2_canonical_value_binder requires c2_color_head_mlp_dim=0; the MLP would be a "
                    "second uncalibrated VALUE authority.")
            conflicting = [
                name for name, _width, flag in EVIDENCE_COLS
                if name in _CANONICAL_SUPPRESSED_EVIDENCE
                and name != "where_hint"
                and bool(getattr(self.config, flag, False))
            ]
            if conflicting:
                raise ValueError(
                    "canonical VALUE binding suppresses independent full-colour evidence blocks; "
                    f"disable these conflicting flags: {', '.join(conflicting)}")
        if (getattr(self.config, "c2_extent_conditioned_structure", False)
                and getattr(self.config, "c2_candidate_floor_structure", False)):
            raise ValueError(
                "c2_extent_conditioned_structure replaces global candidate-floor-structure; "
                "the two routes are mutually exclusive.")

        # --- D5 NOT-YET-WIRED GUARD (audit A2): the field parses but NOTHING consumes it -- no
        #     evidence width, so even the width-drift assert is blind. A parseable no-op flag is
        #     the V3-1 silent-default disease; refuse outright until the wiring block ships.
        #     Delete the line the day its consumer lands (D5 -> ranked-frame hint, AFTER the
        #     File-#7 dataloader flip). D6 c2_algo_where_touch was REMOVED from this guard when
        #     its consumer (_evidence_algo_touch) landed.
        for _k in ("c2_frame_hint_ranked",):
            if getattr(self.config, _k, False):
                raise ValueError(
                    f"config.{_k}=True, but this flag is NOT YET WIRED in trm_fvr_v2 (no consumer): "
                    f"the run would silently test nothing. Unset it, or implement its block first.")

        # --- §15.0 V3-CLEAN INVARIANT GUARD (no behaviour change; surfaces config drift) ----------
        self._v3_clean_invariant_check()

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        self.c2 = (
            TestConditionedC2(self.config)
            if self.config.c2_enabled and self.config.c2_mode == "test_conditioned" and self.config.c2_num_context > 0
            else None
        )
        if self.config.c2_visual_rule_adapter and C2SigLIPRuleAdapter is None:
            raise ModuleNotFoundError(
                "c2_visual_rule_adapter=True but models.c2_siglip_rule_adapter is not available"
            )
        self.visual_rule_adapter = (
            C2SigLIPRuleAdapter(
                token_spec=ARCTokenSpec(
                    pad_id=0,
                    eos_id=1,
                    color_token_ids=tuple(range(2, 12)),
                    vocab_size=self.config.vocab_size,
                    grid_size=30,
                    image_size=224,
                ),
                hidden_size=self.config.hidden_size,
                rule_dim=self.config.c2_visual_rule_dim,
                gate_init=self.config.c2_visual_gate_init,
                model_name=self.config.c2_visual_encoder_name,
            )
            if self.config.c2_visual_rule_adapter
            else None
        )
        if self.visual_rule_adapter is not None:
            assert self.config.c2_visual_mode == "pooled_demo_delta_symbolic", self.config.c2_visual_mode
            assert not self.config.c2_visual_use_query_output, (
                "c2_visual_use_query_output must remain false; hidden query outputs "
                "cannot be used as visual inputs."
            )
        if self.c2 is not None and self.config.c2_modulate_pid:
            # Demo-derived task vector additively modulates the raw puzzle embedding. Scalar gate is the
            # zero-init lever (tanh(0)=0 => no-op at init); the modulator weight uses default init so
            # gradient can flow to the gate at step 1 (zero-init on both would deadlock).
            self.pid_task_modulator = CastedLinear(
                self.config.hidden_size,
                self.config.puzzle_emb_ndim,
                bias=False,
            )
            self.pid_task_gate = nn.Parameter(torch.tensor(0.0))
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _ in range(self.config.L_layers)]
        )

        self.register_buffer(
            "evidence_schema_fingerprint",
            evidence_schema_fingerprint(self.config),
            persistent=True,
        )

        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    # Output-side writer flags (plan §15.3). In V3-clean (c2_dual_output_head=True) NONE execute --
    # they all live under `if not c2_dual_output_head` -- so enabling one is silently inert.
    def _v3_clean_invariant_check(self) -> None:
        """Warn (never raise) on the two ways a config silently defeats V3-clean. See plan §15.0/§15.3."""
        import warnings
        relmap = bool(getattr(self.config, "c2_relmap", False))
        dual = bool(getattr(self.config, "c2_dual_output_head", False))
        if relmap and not dual:
            warnings.warn(
                "[V3-clean section 15] c2_relmap=True but c2_dual_output_head=False: relational maps are added "
                "to the INPUT (zero-init, inert until trained) but the factored color_head that READS them "
                "at output is NOT built/called -- the run is NOT exercising V3. Set c2_dual_output_head=True "
                "(run_stage1_local.py --v3-clean) to activate the factored structure/color head.",
                RuntimeWarning, stacklevel=2,
            )
        if dual and float(getattr(self.config, "c2_structure_fusion_alpha", 0.0)) != 0.0:
            warnings.warn(
                "[V3-clean section 15] c2_dual_output_head=True (factored head is the sole output writer), but "
                "c2_structure_fusion_alpha != 0 -- that geomaux fusion is inert under V3-clean. Set it to 0.",
                RuntimeWarning, stacklevel=2,
            )

    @staticmethod
    def _active_context(batch: Dict[str, torch.Tensor] | None):
        """(ci, co, cm) preferring the LODO `_active_context_*` overrides over the raw context (M9: was
        the `batch.get('_active_context_*', batch.get('context_*'))` triple repeated across methods)."""
        if batch is None:
            return None, None, None
        ci = batch.get("_active_context_inputs", batch.get("context_inputs"))
        co = batch.get("_active_context_outputs", batch.get("context_outputs"))
        cm = batch.get("_active_context_mask", batch.get("context_mask"))
        return ci, co, cm

    def _add_broadcast_hint(
        self,
        grid_features: torch.Tensor,
        vec: torch.Tensor,
        norm_key: str,
        c2_metrics: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """M8: the shared broadcast-add + norm-metric behind the three input-side hint injectors
        (pairdelta_input / frame / rule-hyp). `vec` is [B, hidden]; it is broadcast over all cells and
        added (scaled by the --zh-amp diagnostic, default 1.0 == no-op). Any hint-specific EXTRA metric
        (e.g. nonzero_frac) is added by the caller. Byte-identical to the three former inline copies."""
        _amp = float(getattr(self, "_demo_injection_scale", 1.0))
        residual = _amp * vec.unsqueeze(1).expand(-1, grid_features.shape[1], -1)
        if getattr(self.config, "c2_bounded_evidence_fusion", False):
            grid_features, raw_ratio, applied_ratio = bounded_residual(
                grid_features, residual, float(getattr(self.config, "c2_post_hint_rho", 0.10)))
        else:
            grid_features = grid_features + residual
            raw_ratio = torch.zeros((), device=grid_features.device)
            applied_ratio = raw_ratio
        with torch.no_grad():
            c2_metrics = dict(c2_metrics)
            c2_metrics[norm_key] = vec.float().norm(dim=-1).mean().detach()
            if getattr(self.config, "c2_bounded_evidence_fusion", False):
                c2_metrics[f"{norm_key}_raw_ratio"] = raw_ratio.detach()
                c2_metrics[f"{norm_key}_applied_ratio"] = applied_ratio.detach()
        return grid_features, c2_metrics

    def _maybe_drop_puzzle_ids(self, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        if self.training and self.config.c2_pid_dropout > 0:
            drop = torch.rand_like(puzzle_identifiers.float()) < self.config.c2_pid_dropout
            return torch.where(drop, torch.zeros_like(puzzle_identifiers), puzzle_identifiers)
        return puzzle_identifiers

    def _puzzle_embedding(self, puzzle_identifiers: torch.Tensor, use_sparse_training_buffer: bool) -> torch.Tensor:
        if self.training and not use_sparse_training_buffer:
            return self.puzzle_emb.weights[puzzle_identifiers].to(self.forward_dtype)
        return self.puzzle_emb(puzzle_identifiers)

    # ---------------------------------------------------------------- §6 INPUT: condition + embeddings
    def _condition_grid_features(
        self,
        target_inputs: torch.Tensor,
        target_visual_features: torch.Tensor | None = None,
        context_inputs: torch.Tensor | None = None,
        context_outputs: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        context_input_visual_features: torch.Tensor | None = None,
        context_output_visual_features: torch.Tensor | None = None,
        rel_maps: torch.Tensor | None = None,
        context_rel_maps: torch.Tensor | None = None,
        context_output_rel_maps: torch.Tensor | None = None,
        frame_label: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor | None, Dict[str, torch.Tensor]]:
        # Per-forward evidence hints are RETURNED (input_hints), never stashed on self: instance-attr
        # mutation inside forward is invisible to torch.compile graph capture. The dict is threaded to
        # _output_logits so main / aux-LODO / shuffle forwards each carry the hint from THEIR context.
        input_hints: Dict[str, torch.Tensor] = {}
        grid_features = self.grid_encoder(target_inputs, target_visual_features)
        c2_metrics: Dict[str, torch.Tensor] = {}
        pid_task_vec: torch.Tensor | None = None
        target_query_features: torch.Tensor | None = None   # P3A Block 1 x_query (isolated lane)
        ordered_flow = bool(getattr(self.config, "c2_ordered_evidence_flow", False))
        injection_scale = float(getattr(self, "_demo_injection_scale", 1.0))

        # Ordered V2 path: target relations and support-fitted WHERE predicates must exist before
        # C2 forms its target queries. The legacy path below remains untouched when the flag is off.
        if ordered_flow:
            from models.recursive_reasoning.relation_map import (
                relational_maps as _ordered_relational_maps,
                relational_where_hint as _ordered_where_hint,
            )
            B, L = target_inputs.shape
            side = int(math.isqrt(L))
            if rel_maps is None:
                rel_maps = _ordered_relational_maps(target_inputs, side=side)
            if context_inputs is not None and context_rel_maps is None:
                BM = context_inputs.shape[0] * context_inputs.shape[1]
                context_rel_maps = _ordered_relational_maps(
                    context_inputs.reshape(BM, L), side=side).view(
                        context_inputs.shape[0], context_inputs.shape[1], L, -1)
            if (context_outputs is not None and context_output_rel_maps is None
                    and getattr(self, "c2_demo_relmap_proj", None) is not None):
                BM = context_outputs.shape[0] * context_outputs.shape[1]
                context_output_rel_maps = _ordered_relational_maps(
                    context_outputs.reshape(BM, L), side=side).view(
                        context_outputs.shape[0], context_outputs.shape[1], L, -1)

            rel_maps = rel_maps.to(grid_features.dtype)
            rel_delta = injection_scale * self.relmap_proj(rel_maps)
            if getattr(self.config, "c2_bounded_evidence_fusion", False):
                _relmap_fused, raw_ratio, applied_ratio = bounded_residual(
                    grid_features, rel_delta, float(getattr(self.config, "c2_target_relmap_rho", 0.10)))
                c2_metrics["c2_target_relmap_raw_norm_ratio"] = raw_ratio.detach()
                c2_metrics["c2_target_relmap_applied_norm_ratio"] = applied_ratio.detach()
            else:
                _relmap_fused = grid_features + rel_delta
            if getattr(self.config, "c2_isolated_relmap_query", False):
                # P3A Block 1: the relmap-enriched features become the C2 QUERY lane only (x_query).
                # The recurrence keeps x_base = grid_encoder output; target relations may shape the
                # C2 query and WHERE gate but can no longer write into the recurrent input directly.
                target_query_features = _relmap_fused
            else:
                grid_features = _relmap_fused

            if (getattr(self.config, "c2_rel_where_hint", False)
                    and context_inputs is not None and context_outputs is not None
                    and context_mask is not None):
                where_hint, where_info = _ordered_where_hint(
                    target_inputs,
                    context_inputs,
                    context_outputs,
                    context_mask,
                    target_rel_maps=rel_maps.float(),
                    context_rel_maps=context_rel_maps.float() if context_rel_maps is not None else None,
                    side=side,
                    topk=max(1, int(getattr(self.config, "c2_rel_where_topk", 1))),
                )
                input_hints["rel_where"] = where_hint.to(grid_features.dtype)
                c2_metrics["c2_rel_where_confidence"] = (
                    where_info["rel_where_confidence"].float().mean().detach())
                c2_metrics["c2_rel_where_f1"] = where_info["rel_where_f1"].float().mean().detach()
                c2_metrics["c2_rel_where_fpr"] = where_info["rel_where_fpr"].float().mean().detach()
                c2_metrics["c2_where_support_fit_f1"] = c2_metrics["c2_rel_where_f1"]
                c2_metrics["c2_where_support_fit_fpr"] = c2_metrics["c2_rel_where_fpr"]

        rule_factor_scores = None
        if (context_inputs is not None and context_outputs is not None and context_mask is not None
                and (getattr(self.config, "c2_rule_factor_hint", False)
                     or getattr(self.config, "c2_object_pair_tokens", False)
                     or getattr(self.config, "c2_extent_conditioned_structure", False))):
            rule_factor_scores, rule_factor_stats = self._rule_factor_batch(
                context_inputs, context_outputs, context_mask,
                int(math.isqrt(target_inputs.shape[-1])), grid_features.device)
            input_hints["rule_factors"] = rule_factor_scores.to(grid_features.dtype)
            c2_metrics.update(rule_factor_stats)

        if self.c2 is not None and context_inputs is not None and context_outputs is not None and context_mask is not None:
            context_input_features = self.grid_encoder(context_inputs, context_input_visual_features)
            context_output_features = self.grid_encoder(context_outputs, context_output_visual_features)
            # §15.2-A: enrich SUPPORT demo features with their relational maps BEFORE C2 attention. Zero-init
            # proj => no-op at step 0. Inline fallback computes the support maps if the dataloader did not.
            if getattr(self, "c2_demo_relmap_proj", None) is not None:
                if context_rel_maps is None or context_output_rel_maps is None:
                    from models.recursive_reasoning.relation_map import relational_maps as _rm   # M11
                    _B, _M, _L = context_inputs.shape
                    _side = int(math.isqrt(_L))
                    if context_rel_maps is None:
                        context_rel_maps = _rm(context_inputs.reshape(_B * _M, _L), side=_side).view(_B, _M, _L, -1)
                    if context_output_rel_maps is None:
                        context_output_rel_maps = _rm(context_outputs.reshape(_B * _M, _L), side=_side).view(_B, _M, _L, -1)
                _support_in_delta = injection_scale * self.c2_demo_relmap_proj(
                    context_rel_maps.to(context_input_features.dtype))
                _support_out_delta = injection_scale * self.c2_demo_relmap_proj(
                    context_output_rel_maps.to(context_output_features.dtype))
                if getattr(self.config, "c2_bounded_evidence_fusion", False):
                    context_input_features, _sir, _sia = bounded_residual(
                        context_input_features, _support_in_delta,
                        float(getattr(self.config, "c2_target_relmap_rho", 0.10)))
                    context_output_features, _sor, _soa = bounded_residual(
                        context_output_features, _support_out_delta,
                        float(getattr(self.config, "c2_target_relmap_rho", 0.10)))
                    c2_metrics["c2_support_input_relmap_raw_norm_ratio"] = _sir.detach()
                    c2_metrics["c2_support_input_relmap_applied_norm_ratio"] = _sia.detach()
                    c2_metrics["c2_support_output_relmap_raw_norm_ratio"] = _sor.detach()
                    c2_metrics["c2_support_output_relmap_applied_norm_ratio"] = _soa.detach()
                else:
                    context_input_features = context_input_features + _support_in_delta
                    context_output_features = context_output_features + _support_out_delta
            grid_features, _c2_runtime_metrics, pid_task_vec = self.c2(
                target_features=grid_features,
                context_inputs=context_inputs,
                context_outputs=context_outputs,
                context_input_features=context_input_features,
                context_output_features=context_output_features,
                context_mask=context_mask,
                target_where_hint=input_hints.get("rel_where") if ordered_flow else None,
                rule_factors=rule_factor_scores,
                target_query_features=target_query_features,
            )
            c2_metrics = {**c2_metrics, **_c2_runtime_metrics}
        if getattr(self.config, "c2_rule_factor_hint", False) and rule_factor_scores is not None:
            rule_hint = self.rule_factor_proj(rule_factor_scores.to(grid_features.dtype))
            grid_features, c2_metrics = self._add_broadcast_hint(
                grid_features, rule_hint, "c2_rule_factor_hint_norm", c2_metrics)
        # §15.2-B: PairDelta as an INPUT-ONLY hint -- broadcast the learned cross-demo rule_vec to all
        # cells (zero-init proj => no-op at step 0). Pure input evidence; the TRM recurrence fuses it.
        if (getattr(self, "pairdelta_input_encoder", None) is not None
                and context_inputs is not None and context_outputs is not None and context_mask is not None):
            _pd = self.pairdelta_input_encoder(context_inputs, context_outputs, context_mask)
            _vec = _pd["rule_vec"]
            if getattr(self.config, "c2_pairdelta_spatial", False):
                with torch.no_grad():
                    c2_metrics = dict(c2_metrics)
                    c2_metrics["c2_pairdelta_spatial_feature_norm"] = (
                        _pd["spatial_feature_norm"].float().detach())
                    _spatial_denom = context_mask.float().sum().clamp_min(1.0)
                    c2_metrics["c2_pairdelta_spatial_valid_rate"] = (
                        _pd["spatial_valid"].float().sum() / _spatial_denom).detach()
            if getattr(self.config, "c2_pairdelta_input_conf_gate", False):
                # SS7 reuse #2: rule_confidence was DEAD in this lane. Gating the broadcast by it makes
                # low-signal tasks shrink the hint instead of feeding the norm growth that breaks
                # MAIN-ON (measured: pairdelta_norm 0.002 -> 3.7 over 500 steps, MAIN-ON dip to 96.9).
                # F7-safe: the proj stays zero-init, so step-0 is byte-identical either way.
                _vec = _vec * _pd["rule_confidence"].unsqueeze(-1).to(_vec.dtype)
                with torch.no_grad():
                    c2_metrics = dict(c2_metrics)
                    c2_metrics["c2_pairdelta_input_conf"] = _pd["rule_confidence"].float().mean().detach()
            _rule = self.delta_rule_input_proj(_vec.to(grid_features.dtype))              # [B, hidden]
            grid_features, c2_metrics = self._add_broadcast_hint(
                grid_features, _rule, "c2_pairdelta_input_norm", c2_metrics)               # M8
        if (getattr(self.config, "c2_pairdelta_intent_hint", False)
                and context_inputs is not None and context_outputs is not None and context_mask is not None):
            from models.recursive_reasoning.pair_delta_v2 import pairdelta_intent_features  # File #5 flip (byte-equal)
            _intent = pairdelta_intent_features(context_inputs, context_outputs, context_mask)
            input_hints["pairdelta_intent"] = _intent["feature"].to(grid_features.dtype)  # [B,1]
            with torch.no_grad():
                c2_metrics = dict(c2_metrics)
                c2_metrics["c2_pairdelta_conditional_score"] = (
                    _intent["conditional_recolor_score"].float().mean().detach())
                c2_metrics["c2_pairdelta_global_score"] = (
                    _intent["global_recolor_score"].float().mean().detach())
                c2_metrics["c2_pairdelta_shape_preserved"] = (
                    _intent["shape_preserved"].float().mean().detach())
                c2_metrics["c2_pairdelta_changed_rate"] = (
                    _intent["changed_rate"].float().mean().detach())
        if (
            self.visual_rule_adapter is not None
            and context_inputs is not None
            and context_outputs is not None
            and context_mask is not None
        ):
            pre_visual_features = grid_features
            adapted_features, visual_rule_metrics = self.visual_rule_adapter(
                base_features=pre_visual_features,
                target_inputs=target_inputs,
                context_inputs=context_inputs,
                context_outputs=context_outputs,
                context_mask=context_mask,
            )
            grid_features = pre_visual_features + injection_scale * (
                adapted_features - pre_visual_features)
            c2_metrics = dict(c2_metrics)
            c2_metrics.update(visual_rule_metrics)
        if (getattr(self.config, "c2_relmap", False) and target_inputs is not None
                and not ordered_flow):
            if rel_maps is None:
                # Fallback: compute inline if not pre-computed in the dataloader.
                if not getattr(self, "_relmap_fallback_warned", False):
                    import warnings
                    warnings.warn(
                        "rel_maps missing from batch; falling back to inline relational_maps compute. "
                        "Training runs should forward arch.c2_relmap into PuzzleDatasetConfig.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._relmap_fallback_warned = True
                from models.recursive_reasoning.relation_map import relational_maps as _compute_relational_maps  # M11
                _side = int(math.isqrt(target_inputs.shape[-1]))
                rel_maps = _compute_relational_maps(target_inputs, side=_side)
            rel_maps = rel_maps.to(grid_features.dtype)
            grid_features = grid_features + injection_scale * self.relmap_proj(rel_maps)
            if (getattr(self.config, "c2_rel_where_hint", False)
                    and context_inputs is not None and context_outputs is not None and context_mask is not None):
                from models.recursive_reasoning.relation_map import relational_where_hint   # M11
                _hint, _where_info = relational_where_hint(
                    target_inputs,
                    context_inputs,
                    context_outputs,
                    context_mask,
                    target_rel_maps=rel_maps.float(),
                    context_rel_maps=context_rel_maps.float() if context_rel_maps is not None else None,
                    side=int(math.isqrt(target_inputs.shape[-1])),
                    topk=max(1, int(getattr(self.config, "c2_rel_where_topk", 1))),
                )
                input_hints["rel_where"] = _hint.to(grid_features.dtype)
                with torch.no_grad():
                    c2_metrics = dict(c2_metrics)
                    c2_metrics["c2_rel_where_confidence"] = (
                        _where_info["rel_where_confidence"].float().mean().detach())
                    c2_metrics["c2_rel_where_f1"] = (_where_info["rel_where_f1"].float().mean().detach())
                    c2_metrics["c2_rel_where_fpr"] = (_where_info["rel_where_fpr"].float().mean().detach())
                    c2_metrics["c2_where_support_fit_f1"] = c2_metrics["c2_rel_where_f1"]
                    c2_metrics["c2_where_support_fit_fpr"] = c2_metrics["c2_rel_where_fpr"]

        # Lane B: broadcast-add the FRAME-family hint embedding (zero at init -> F7-safe). Independent of
        # c2_relmap so the rule-hypothesis bus can be A/B'd on its own.
        if getattr(self.config, "c2_frame_hint", False) and frame_label is not None and hasattr(self, "frame_embed"):
            fe = self.frame_embed(frame_label.to(torch.long)).to(grid_features.dtype)     # [B, hidden]
            fe = fe * (frame_label != 0).to(fe.dtype).unsqueeze(-1)
            grid_features, c2_metrics = self._add_broadcast_hint(
                grid_features, fe, "c2_frame_hint_norm", c2_metrics)                       # M8
            with torch.no_grad():
                c2_metrics = dict(c2_metrics)
                c2_metrics["c2_frame_hint_nonzero_frac"] = (frame_label != 0).float().mean().detach()

        # In-model rule-hypothesis hint: infer the TOP operation-family from the (LODO-correct) support
        # pairs and broadcast-add its zero-init embedding. context_inputs is already the held-out support
        # on the aux path (no target leak). Inference is CPU/python under no_grad; only the embedding learns.
        if (getattr(self.config, "c2_rule_hypothesis_hint", False) and hasattr(self, "rule_hyp_embed")
                and context_inputs is not None and context_outputs is not None):
            from models.recursive_reasoning.core_prior import infer_rule_hypotheses   # M11 (+C7 cache)
            B, M, L = context_inputs.shape
            side = int(math.isqrt(L))
            ci_cpu = context_inputs.detach().to("cpu", torch.long)
            co_cpu = context_outputs.detach().to("cpu", torch.long)
            cm_cpu = (context_mask.detach().to("cpu").bool().view(B, M)
                      if context_mask is not None else torch.ones(B, M, dtype=torch.bool))
            fam_idx = torch.zeros(B, dtype=torch.long)
            with torch.no_grad():
                for b in range(B):
                    keep = cm_cpu[b].nonzero(as_tuple=True)[0]
                    if keep.numel() == 0:
                        continue
                    ranked = infer_rule_hypotheses(ci_cpu[b][keep], co_cpu[b][keep], side)
                    if ranked:
                        fam_idx[b] = RULE_FAMILY_INDEX.get(ranked[0]["family"], 0)
            fam_idx = fam_idx.to(grid_features.device)
            rh = self.rule_hyp_embed(fam_idx).to(grid_features.dtype)                     # [B, hidden]
            rh = rh * (fam_idx != 0).to(rh.dtype).unsqueeze(-1)
            grid_features, c2_metrics = self._add_broadcast_hint(
                grid_features, rh, "c2_rule_hyp_norm", c2_metrics)                         # M8
            with torch.no_grad():
                c2_metrics = dict(c2_metrics)
                c2_metrics["c2_rule_hyp_nonzero_frac"] = (fam_idx != 0).float().mean().detach()

        return grid_features, c2_metrics, pid_task_vec, rel_maps, input_hints

    def _input_embeddings(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor | None, Dict[str, torch.Tensor]]:
        grid_features, c2_metrics, pid_task_vec, rel_maps, input_hints = self._condition_grid_features(
            target_inputs=batch["inputs"],
            target_visual_features=batch.get("input_visual_features"),
            context_inputs=batch.get("context_inputs"),
            context_outputs=batch.get("context_outputs"),
            context_mask=batch.get("context_mask"),
            context_input_visual_features=batch.get("context_input_visual_features"),
            context_output_visual_features=batch.get("context_output_visual_features"),
            rel_maps=batch.get("rel_maps"),
            context_rel_maps=batch.get("context_rel_maps"),
            context_output_rel_maps=batch.get("context_output_rel_maps"),
            frame_label=batch.get("frame_label"),
        )
        # rel_maps is RETURNED, never written into `batch` (the batch dict IS the ACT carry's
        # current_data; mutating it leaks keys into the carry -> next-step key-merge KeyErrors).
        input_embeddings = self._prepend_puzzle_embeddings(
            grid_features,
            batch["puzzle_identifiers"],
            pid_task_vec=pid_task_vec,
        )
        if pid_task_vec is not None and self.config.c2_modulate_pid and hasattr(self, "pid_task_gate"):
            with torch.no_grad():
                c2_metrics = dict(c2_metrics)
                c2_metrics["c2_pid_task_gate"] = torch.tanh(self.pid_task_gate.float()).detach()
                c2_metrics["c2_pid_task_vec_norm"] = pid_task_vec.float().norm(dim=-1).mean().detach()
        return input_embeddings, c2_metrics, rel_maps, input_hints

    def _prepend_puzzle_embeddings(
        self,
        grid_features: torch.Tensor,
        puzzle_identifiers: torch.Tensor,
        use_sparse_training_buffer: bool = True,
        apply_pid_dropout: bool = True,
        pid_task_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = grid_features
        if self.config.puzzle_emb_ndim > 0:
            if apply_pid_dropout:
                puzzle_identifiers = self._maybe_drop_puzzle_ids(puzzle_identifiers)
            puzzle_embedding = self._puzzle_embedding(puzzle_identifiers, use_sparse_training_buffer)
            if (
                pid_task_vec is not None
                and self.config.c2_modulate_pid
                and self.c2 is not None
            ):
                modulation = self.pid_task_modulator(pid_task_vec).to(puzzle_embedding.dtype)
                gate = torch.tanh(self.pid_task_gate).to(puzzle_embedding.dtype)
                injection_scale = float(getattr(self, "_demo_injection_scale", 1.0))
                puzzle_embedding = puzzle_embedding + injection_scale * gate * modulation
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            puzzle_embedding = puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size)
            embedding = torch.cat((self.embed_scale * puzzle_embedding, embedding), dim=-2)

        if self.config.pos_encodings == "learned":
            pos = self.embed_scale * self.embed_pos.embedding_weight.to(self.forward_dtype)
            embedding = 0.707106781 * (embedding + pos)
        return embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def fresh_carry(self, batch_size: int):
        seq_len = self.config.seq_len + self.puzzle_emb_len
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=self.H_init.view(1, 1, -1).expand(batch_size, seq_len, -1).clone(),
            z_L=self.L_init.view(1, 1, -1).expand(batch_size, seq_len, -1).clone(),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def _run_recurrence(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        input_embeddings: torch.Tensor,
        seq_info: Dict[str, torch.Tensor | None],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_H, z_L = carry.z_H, carry.z_L
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        for _ in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
        return z_H, z_L

    # ---------------------------------------------------------------------------- §8 LODO machinery
    def _build_lodo_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor] | None:
        force_marker = batch.get("_force_lodo_eval", False)
        if torch.is_tensor(force_marker):
            expected = batch["inputs"].shape[:1]
            if force_marker.shape != expected or force_marker.dtype != torch.bool:
                raise ValueError("_force_lodo_eval must be a Bool tensor with shape [B]")
            force_lodo_eval = bool(force_marker.all())
        else:
            force_lodo_eval = bool(force_marker)
        if (
            (not self.training and not force_lodo_eval)
            or self.c2 is None
            or (self.config.c2_leave_one_demo_weight <= 0
                and not getattr(self.config, "c2_lodo_force_build", False))
            or not {"context_inputs", "context_outputs", "context_mask"}.issubset(batch.keys())
        ):
            return None

        context_mask = batch["context_mask"].to(torch.bool)
        valid_counts = context_mask.sum(dim=-1)
        aux_valid = valid_counts >= 2
        if not aux_valid.any():
            return None

        contract_valid = batch.get("_lodo_aux_valid")
        if contract_valid is not None:
            if not torch.is_tensor(contract_valid) or contract_valid.shape != aux_valid.shape:
                raise ValueError("_lodo_aux_valid must be a Bool tensor with shape [B]")
            contract_valid = contract_valid.to(device=aux_valid.device, dtype=torch.bool)
            if bool((contract_valid & ~aux_valid).any()):
                raise ValueError("_lodo_aux_valid selected a row with fewer than two valid demonstrations")
            aux_valid = aux_valid & contract_valid
        else:
            max_samples = int(self.config.c2_lodo_max_samples)
            if max_samples > 0:
                valid_indices = torch.nonzero(aux_valid, as_tuple=False).flatten()
                if valid_indices.numel() > max_samples:
                    selected = valid_indices[
                        torch.randperm(valid_indices.numel(), device=valid_indices.device)[:max_samples]]
                    limited_valid = torch.zeros_like(aux_valid)
                    limited_valid[selected] = True
                    aux_valid = limited_valid

        contract_holdout = batch.get("_lodo_holdout_idx")
        if contract_holdout is not None:
            if not torch.is_tensor(contract_holdout) or contract_holdout.shape != valid_counts.shape:
                raise ValueError("_lodo_holdout_idx must be a Long tensor with shape [B]")
            if contract_holdout.dtype != torch.long:
                raise ValueError("_lodo_holdout_idx must be a Long tensor with shape [B]")
            holdout_idx = contract_holdout.to(device=context_mask.device, dtype=torch.long)
            if bool(((holdout_idx < 0) | (holdout_idx >= context_mask.shape[1])).any()):
                raise ValueError("_lodo_holdout_idx contains an out-of-range demonstration index")
            chosen_is_valid = context_mask.gather(1, holdout_idx.view(-1, 1)).squeeze(1)
            if bool((aux_valid & ~chosen_is_valid).any()):
                raise ValueError("_lodo_holdout_idx selected a masked demonstration")
        else:
            random_scores = torch.rand(context_mask.shape, device=context_mask.device)
            random_scores = random_scores.masked_fill(~context_mask, -1)
            holdout_idx = random_scores.argmax(dim=-1)

        gather_index = holdout_idx.view(-1, 1, 1).expand(-1, 1, batch["context_inputs"].shape[-1])
        aux_inputs = batch["context_inputs"].gather(1, gather_index).squeeze(1)
        aux_labels = batch["context_outputs"].gather(1, gather_index).squeeze(1)
        aux_input_visual_features = None
        if "context_input_visual_features" in batch:
            visual_gather_index = holdout_idx.view(-1, 1, 1, 1).expand(
                -1,
                1,
                batch["context_input_visual_features"].shape[-2],
                batch["context_input_visual_features"].shape[-1],
            )
            aux_input_visual_features = batch["context_input_visual_features"].gather(
                1,
                visual_gather_index,
            ).squeeze(1)

        aux_rel_maps = None
        if "context_rel_maps" in batch:
            relmap_gather_index = holdout_idx.view(-1, 1, 1, 1).expand(
                -1,
                1,
                batch["context_rel_maps"].shape[-2],
                batch["context_rel_maps"].shape[-1],
            )
            aux_rel_maps = batch["context_rel_maps"].gather(
                1,
                relmap_gather_index,
            ).squeeze(1)

        aux_context_mask = context_mask.clone()
        aux_context_mask.scatter_(1, holdout_idx.view(-1, 1), False)
        aux_context_mask = aux_context_mask & aux_valid.view(-1, 1)
        row_indices = torch.nonzero(aux_valid, as_tuple=False).flatten()

        aux_puzzle_identifiers = batch["puzzle_identifiers"][row_indices]
        # SHUFFLE (wrong-task) control: ONE source index per row drives EVERY per-demo tensor below
        # (tokens, mask, relmaps, visual) so wrong-task demo tokens can never pair with correct-task
        # side-channels. Rows without a wrong-task candidate keep shuffle_valid=False (masked by loss).
        shuffle_src = row_indices
        shuffle_valid = torch.zeros(row_indices.shape[0], device=aux_valid.device, dtype=torch.bool)
        shuffle_context_mask = aux_context_mask[row_indices].to(batch["context_mask"].dtype)
        if (self.config.c2_lodo_contrast_weight > 0
                or getattr(self.config, "c2_lodo_force_shuffle", False)) and row_indices.numel() > 0:
            source_puzzle_identifiers = batch["puzzle_identifiers"]
            correct_counts = aux_context_mask[row_indices].sum(dim=-1)
            source_counts = context_mask.sum(dim=-1)
            source_has_context = source_counts > 0
            wrong_candidates = (
                source_puzzle_identifiers.unsqueeze(0) != aux_puzzle_identifiers.unsqueeze(1)
            ) & source_has_context.unsqueeze(0)
            if getattr(self.config, "c2_positive_where_gate", False):
                wrong_candidates = wrong_candidates & (
                    source_counts.unsqueeze(0) >= correct_counts.unsqueeze(1))
            shuffle_valid = wrong_candidates.any(dim=-1)
            if shuffle_valid.any():
                wrong_scores = torch.rand(wrong_candidates.shape, device=wrong_candidates.device)
                wrong_scores = wrong_scores.masked_fill(~wrong_candidates, -1)
                shuffle_src = wrong_scores.argmax(dim=-1)
                shuffle_context_mask = batch["context_mask"][shuffle_src]
                if getattr(self.config, "c2_positive_where_gate", False):
                    # Matched-count contrast: select exactly the same number of demos as the correct
                    # LODO fold. Stable first-valid selection keeps the comparison deterministic.
                    source_valid = shuffle_context_mask.to(torch.bool)
                    order = source_valid.long().cumsum(dim=-1)
                    shuffle_context_mask = (
                        source_valid & (order <= correct_counts.unsqueeze(-1))
                    ).to(batch["context_mask"].dtype)

        def _rows(key: str) -> torch.Tensor:      # correct-task per-demo tensor for the LODO rows
            return batch[key][row_indices]

        def _shuf(key: str) -> torch.Tensor:      # the SAME tensor from the shuffle source rows
            return batch[key][shuffle_src]

        out = {
            "inputs": aux_inputs[row_indices],
            "labels": aux_labels[row_indices],
            "lodo_src_index": row_indices,
            "puzzle_identifiers": aux_puzzle_identifiers,
            "context_inputs": _rows("context_inputs"),
            "context_outputs": _rows("context_outputs"),
            "context_mask": aux_context_mask[row_indices].to(batch["context_mask"].dtype),
            "aux_valid": torch.ones(row_indices.shape[0], device=aux_valid.device, dtype=torch.bool),
            "shuffle_context_inputs": _shuf("context_inputs"),
            "shuffle_context_outputs": _shuf("context_outputs"),
            "shuffle_context_mask": shuffle_context_mask,
            "shuffle_valid": shuffle_valid,
        }
        if aux_input_visual_features is not None:
            out.update({
                "input_visual_features": aux_input_visual_features[row_indices],
                "context_input_visual_features": _rows("context_input_visual_features"),
                "context_output_visual_features": _rows("context_output_visual_features"),
                "shuffle_context_input_visual_features": _shuf("context_input_visual_features"),
                "shuffle_context_output_visual_features": _shuf("context_output_visual_features"),
            })
        if aux_rel_maps is not None:
            out.update({
                "rel_maps": aux_rel_maps[row_indices],
                "context_rel_maps": _rows("context_rel_maps"),
                "shuffle_context_rel_maps": _shuf("context_rel_maps"),
            })
        # Demo OUTPUT relmaps + frame label ride along whenever the dataloader emits them (the aux forward
        # once dropped these -> LODO trained a DIFFERENT input contract than MAIN).
        if "context_output_rel_maps" in batch:
            out["context_output_rel_maps"] = _rows("context_output_rel_maps")
            out["shuffle_context_output_rel_maps"] = _shuf("context_output_rel_maps")
        if "frame_label" in batch:
            out["frame_label"] = _rows("frame_label")
            out["shuffle_frame_label"] = _shuf("frame_label")
        return out

    def _run_aux_logits(
        self,
        aux_batch: Dict[str, torch.Tensor],
        seq_info: Dict[str, torch.Tensor | None],
        context_inputs_key: str,
        context_outputs_key: str,
        context_mask_key: str,
        return_extras: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # The aux forward must see the SAME input contract as the main forward. Key-replace derives the
        # LODO ("context_*") and SHUFFLE ("shuffle_context_*") variants from one pattern.
        def _aux_key(base_key: str, feature: str) -> torch.Tensor | None:
            prefix = "context_inputs" if base_key == context_inputs_key else "context_outputs"
            return aux_batch.get(base_key.replace(prefix, feature))

        grid_features, aux_c2_metrics, pid_task_vec, rel_maps, aux_input_hints = self._condition_grid_features(
            target_inputs=aux_batch["inputs"],
            target_visual_features=aux_batch.get("input_visual_features"),
            context_inputs=aux_batch[context_inputs_key],
            context_outputs=aux_batch[context_outputs_key],
            context_mask=aux_batch[context_mask_key],
            context_input_visual_features=_aux_key(context_inputs_key, "context_input_visual_features"),
            context_output_visual_features=_aux_key(context_outputs_key, "context_output_visual_features"),
            rel_maps=aux_batch.get("rel_maps"),
            context_rel_maps=_aux_key(context_inputs_key, "context_rel_maps"),
            context_output_rel_maps=_aux_key(context_outputs_key, "context_output_rel_maps"),
            frame_label=_aux_key(context_inputs_key, "frame_label"),
        )
        input_embeddings = self._prepend_puzzle_embeddings(
            grid_features,
            aux_batch["puzzle_identifiers"],
            use_sparse_training_buffer=False,
            apply_pid_dropout=False,
            pid_task_vec=pid_task_vec,
        )
        aux_carry = self.fresh_carry(aux_batch["inputs"].shape[0])
        z_H, _z_L = self._run_recurrence(aux_carry, input_embeddings, seq_info)
        aux_batch["_active_context_inputs"] = aux_batch[context_inputs_key]
        aux_batch["_active_context_outputs"] = aux_batch[context_outputs_key]
        aux_batch["_active_context_mask"] = aux_batch[context_mask_key]
        active_rel_maps = _aux_key(context_inputs_key, "context_rel_maps")
        if active_rel_maps is not None:
            aux_batch["_active_context_rel_maps"] = active_rel_maps
        logits, extra_outputs = self._output_logits(z_H, aux_batch, rel_maps=rel_maps, input_hints=aux_input_hints)
        if "c2_gate_where_values" in aux_c2_metrics:
            extra_outputs["c2_gate_where_values"] = aux_c2_metrics["c2_gate_where_values"]
        if (
            getattr(self.config, "c2_floor_candidate_split", False)
            and "c2_candidate_logits" in extra_outputs
        ):
            logits = extra_outputs["c2_candidate_logits"]
        if return_extras:
            return logits, extra_outputs
        return logits

    def _quarantine_total(self) -> int:
        """Quarantine feature/head width: QuarCol.TOTAL plus the D3 verified-frame block (11) when
        c2_verified_frame_evidence is on. Single source for __init__ (head + warm-init) and features."""
        return QuarCol.TOTAL + (11 if getattr(self.config, "c2_verified_frame_evidence", False) else 0)

    def _quarantine_features(
        self,
        batch: Dict[str, torch.Tensor] | None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        rel_maps: torch.Tensor | None,
        transition_hint: torch.Tensor | None,
        rel_where_hint: torch.Tensor | None,
        pairdelta_intent_hint: torch.Tensor | None,
        task_palette: torch.Tensor | None,
        conditioned_hint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """[B, L, _quarantine_total()] PID-free per-cell evidence for the quarantined candidate head.
        Layout is QuarCol (input one-hot | 8-neighbour one-hots | marginal transition | rel-where |
        palette | intent | conditioned transition | relmap) + the D3 verified-frame block (11) when on.
        Every part derives from target INPUT or support demos -- PID never enters, so the head's train
        (blank-PID LODO aux) and deploy (PID-ful main) features are identical."""
        _q_dim = self._quarantine_total()
        ti = batch.get("inputs") if batch is not None else None
        if ti is None or ti.ndim != 2 or ti.shape[1] != seq_len:
            return torch.zeros((batch_size, seq_len, _q_dim), device=device, dtype=dtype)
        with torch.no_grad():
            side = int(math.isqrt(seq_len))
            tok = ti.long().clamp(0, 11)
            onehot = F.one_hot(tok, num_classes=12).to(dtype)                     # [B,L,12]
            grid = tok.view(batch_size, side, side)
            padded = F.pad(grid, (1, 1, 1, 1), value=0)                           # PAD border
            shifted = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    shifted.append(
                        padded[:, 1 + dr:1 + dr + side, 1 + dc:1 + dc + side].reshape(batch_size, seq_len))
            neigh = F.one_hot(torch.stack(shifted, dim=-1), num_classes=12).to(dtype)
            neigh = neigh.reshape(batch_size, seq_len, 96)                        # [B,L,8*12]
            if transition_hint is None:
                transition_hint = self._transition_hint(batch, batch_size, seq_len, device)
            if task_palette is None:
                task_palette = self._task_palette_mask(batch, batch_size, device)
            if conditioned_hint is None:
                # FIX B: self-compute the conditioned dist when the caller has no V2 features. LODO-safe
                # (_value_evidence_v2 reads _active_context_* itself). PID never enters.
                _v2, _, _ = self._value_evidence_v2(
                    batch, batch_size, seq_len, device,
                    rel_maps=rel_maps, rel_where_hint=rel_where_hint)
                conditioned_hint = _v2[..., 10:20]
            # FIX D: quarantine layout has ONE rel-where column [118:119]; keep channel 0 (best predicate).
            where = (rel_where_hint.to(dtype)[..., :1] if rel_where_hint is not None
                     else torch.zeros((batch_size, seq_len, 1), device=device, dtype=dtype))
            intent = (pairdelta_intent_hint.to(dtype)[:, None, :].expand(-1, seq_len, -1)
                      if pairdelta_intent_hint is not None
                      else torch.zeros((batch_size, seq_len, 1), device=device, dtype=dtype))
            rmap = (rel_maps.to(dtype) if rel_maps is not None
                    else torch.zeros((batch_size, seq_len, REL_MAP_CHANNELS), device=device, dtype=dtype))
            parts = [
                onehot, neigh, transition_hint.to(dtype), where,
                task_palette[:, None, :].expand(-1, seq_len, -1).to(dtype),
                intent, conditioned_hint.to(dtype), rmap]
            if getattr(self.config, "c2_verified_frame_evidence", False):
                # D3: the same E-1 verified-frame block the colour head sees (11 = 10 one-hot + conf).
                parts.append(self._evidence_verified_frame(batch, batch_size, seq_len, device).to(dtype))
            feats = torch.cat(parts, dim=-1)
        assert feats.shape[-1] == _q_dim, f"quarantine feature layout drifted: {feats.shape[-1]} != {_q_dim}"
        return feats

    def _apply_task_palette_bias(
        self,
        color_logits: torch.Tensor,
        palette_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Soft (or hard) suppression of colours absent from the task palette, applied to COLOUR logits.

        M6 SCOPE NOTE (the silent semantic this docstring exists to kill): this bias is applied ONLY
        inside `_color_logits` -- the MAIN colour lane. The quarantine CANDIDATE lane
        (`_quarantine_logits`) receives the palette purely as a FEATURE column (QUARANTINE_COLS
        "palette"), never through this bias: running `--task-palette-bias` with
        `--quarantine-candidate` does NOT palette-constrain the candidate's colours. If that is ever
        wanted, add an explicit `c2_quarantine_palette_bias` flag rather than widening this one."""
        disallowed = ~palette_mask[:, None, :]
        if getattr(self.config, "c2_task_palette_hard", False):
            return color_logits.masked_fill(disallowed, torch.finfo(color_logits.dtype).min)
        strength = float(getattr(self.config, "c2_task_palette_strength", 4.0))
        if strength <= 0:
            return color_logits
        return color_logits - strength * disallowed.to(color_logits.dtype)

    def _predicted_extent(self, batch: Dict[str, torch.Tensor] | None, shape_logits=None):
        """(h, w, conf) each [B]: predicted OUTPUT extent in CELLS + confidence in [0, 1].

        Fits out_hw = f(in_hw) over the SUPPORT demos from an ORDERED set {identity, constant, ratio},
        VERIFIES f reconstructs EVERY valid demo exactly, then applies the first verified f to the TEST
        INPUT. Support-safe. identity (parameter-free) verifies on 1 demo; constant/ratio (fitted) need
        >= 2. conf: 1.0 demo-verified, c2_extent_shape_head_conf for the learned fallback, else 0.0.
        """
        if batch is None:
            return None
        ci, co, cm = self._active_context(batch)                    # M9
        ti = batch.get("inputs")
        c2 = getattr(self, "c2", None)
        if (ci is None or co is None or ti is None or ti.ndim != 2
                or c2 is None or not hasattr(c2, "_canvas_extent_stats")):
            return None
        side = int(math.isqrt(ti.shape[1]))
        dev = ti.device
        in_hw = (c2._canvas_extent_stats(ci)[..., :2] * side).round()                 # [B,D,2] cells
        out_hw = (c2._canvas_extent_stats(co)[..., :2] * side).round()                # [B,D,2]
        ti_hw = (c2._canvas_extent_stats(ti.unsqueeze(1))[:, 0, :2] * side).round()   # [B,2] test input
        bsz = in_hw.shape[0]
        valid = cm.to(torch.bool) if cm is not None else torch.ones(in_hw.shape[:2], dtype=torch.bool, device=dev)
        vexp = valid.unsqueeze(-1)                                                     # [B,D,1]
        n_valid = valid.sum(dim=1)                                                     # [B]
        idx = torch.arange(bsz, device=dev)
        first_valid = torch.argmax(valid.int(), dim=1)                                # [B]

        h = ti_hw[:, 0].clone()
        w = ti_hw[:, 1].clone()
        conf = torch.zeros(bsz, device=dev)

        def _consider(pred_demo: torch.Tensor, pred_test: torch.Tensor, min_demos: int) -> None:
            ok = ((pred_demo == out_hw) | ~vexp).all(dim=1).all(dim=-1) & (n_valid >= min_demos)
            take = ok & (conf < 0.5)
            h[take] = pred_test[take, 0]
            w[take] = pred_test[take, 1]
            conf[take] = 1.0

        _consider(in_hw, ti_hw, min_demos=1)                                          # (1) identity
        const = out_hw[idx, first_valid]                                              # (2) constant
        _consider(const.unsqueeze(1).expand_as(out_hw), const, min_demos=2)
        safe_in = in_hw.clamp_min(1)                                                  # (3) integer ratio
        kk = (out_hw[idx, first_valid] / safe_in[idx, first_valid]).round()           # [B,2]
        _consider((safe_in * kk.unsqueeze(1)).round(), (ti_hw.clamp_min(1) * kk).round(), min_demos=2)

        if shape_logits is not None and getattr(self.config, "c2_extent_use_shape_head", False):
            hl, wl = shape_logits                                                     # each [B,30]
            margin = torch.minimum(
                F.softmax(hl.float(), dim=-1).amax(dim=-1),
                F.softmax(wl.float(), dim=-1).amax(dim=-1),
            )
            tau = float(getattr(self.config, "c2_extent_shape_head_tau", 0.5))
            take = (conf < 0.5) & (margin > tau)
            h[take] = (hl.argmax(dim=-1)[take] + 1).to(h.dtype)
            w[take] = (wl.argmax(dim=-1)[take] + 1).to(w.dtype)
            conf[take] = float(getattr(self.config, "c2_extent_shape_head_conf", 0.5))

        return h, w, conf

    # ------------------------------------------------------------------- §9 OUTPUT (decomposed)
    # `_output_logits` is the ORCHESTRATOR; each stage is a testable helper with the SAME op order as
    # the old monolith (the master step-0 gate catches any reorder). M2: P_off is threaded through
    # extras["c2_pre_delta_logits"] instead of the `self._last_pre_delta_logits` stash -- the stash's
    # correctness depended on a hand-maintained call ORDER (cleared in forward -> set in aux -> read
    # before the shuffle forward overwrote it); returning it per-forward removes that hazard and is
    # torch.compile-safe. `c2_aux_base_logits` (the loss contract in losses_fvr) is unchanged, now
    # sourced from the aux forward's own extras in `_aux_outputs`.
    def _legacy_output_logits(self, z_H: torch.Tensor, grid_z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Non-dual (legacy lm_head writer) output path."""
        base_logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        extras: Dict[str, torch.Tensor] = {}
        if self.config.c2_geometry_aux_head:
            structure_logits = self.structure_head(grid_z)
            extras["c2_structure_logits"] = structure_logits
            extras["c2_geometry_aux_active"] = torch.as_tensor(1.0, device=z_H.device, dtype=torch.float32)
            alpha = float(self.config.c2_structure_fusion_alpha)
            if alpha != 0.0:
                assert base_logits.shape[:-1] == structure_logits.shape[:-1], (
                    f"Logit spatial mismatch: lm={tuple(base_logits.shape)}, "
                    f"structure={tuple(structure_logits.shape)}")
                assert base_logits.shape[-1] >= 3, (
                    f"Expected PAD/EOS/colour vocabulary, got {base_logits.shape[-1]}")
                assert structure_logits.shape[-1] == 3, (
                    f"Expected PAD/EOS/VALID logits, got {structure_logits.shape[-1]}")
                valid_ref = structure_logits[..., 2:3].to(base_logits.dtype)
                structure_bias = torch.zeros_like(base_logits)
                structure_bias[..., 0:1] = structure_logits[..., 0:1].to(base_logits.dtype) - valid_ref
                structure_bias[..., 1:2] = structure_logits[..., 1:2].to(base_logits.dtype) - valid_ref
                base_logits = base_logits + alpha * structure_bias
                extras["c2_structure_fusion_alpha"] = torch.as_tensor(alpha, device=z_H.device, dtype=torch.float32)
                extras["c2_structure_fusion_bias_abs_mean"] = structure_bias.float().abs().mean().detach()
        if getattr(self.config, "c2_delta_expose_base_logits", False):
            extras["c2_pre_delta_logits"] = base_logits.detach()      # M2: P_off in extras, not on self
        return base_logits, extras

    def _evidence_verified_frame(
        self, batch: Dict[str, torch.Tensor] | None, batch_size: int, seq_len: int, device: torch.device,
    ) -> torch.Tensor:
        """D1 / E-1 (the Lane-A -> Lane-B bridge): [B, L, 11] = per-cell verified-frame colour one-hot
        (10) + broadcast confidence (1). The ONLY signal in the system with an exactness proof; conf 1.0
        iff a frame reconstructs EVERY demo, else the row is zero (no invention). LODO-safe: reads the
        `_active_context_*` support. Per-row CPU inference under no_grad; rides core_prior's propose-cache."""
        out = torch.zeros((batch_size, seq_len, 11), device=device, dtype=torch.float32)
        ci, co, cm = self._active_context(batch)
        ti = batch.get("inputs") if batch is not None else None
        if ci is None or co is None or ti is None or ti.ndim != 2:
            return out
        from models.recursive_reasoning.core_prior import evidence_verified_frame_grid
        side = int(math.isqrt(seq_len))
        with torch.no_grad():
            ci_c = ci.detach().to("cpu", torch.long); co_c = co.detach().to("cpu", torch.long)
            ti_c = ti.detach().to("cpu", torch.long)
            cm_c = (cm.detach().to("cpu").bool() if cm is not None
                    else torch.ones(ci.shape[:2], dtype=torch.bool))
            for b in range(batch_size):
                keep = cm_c[b].nonzero(as_tuple=True)[0]
                if keep.numel() == 0:
                    continue
                grid, conf, _prov = evidence_verified_frame_grid(ci_c[b][keep], co_c[b][keep], ti_c[b], side)
                if conf > 0:
                    out[b, :, :10] = grid.to(device)
                    out[b, :, 10] = float(conf)
        return out

    def _evidence_analogy(
        self, batch: Dict[str, torch.Tensor] | None, batch_size: int, seq_len: int, device: torch.device,
    ) -> torch.Tensor:
        """D2 / E-2 (CF2): [B, L, 11] = per-cell analogy colour distribution (10) + broadcast retrieval
        confidence (1). Zero rows where no demo object matches (no invention). LODO-safe; per-row CPU."""
        out = torch.zeros((batch_size, seq_len, 11), device=device, dtype=torch.float32)
        ci, co, cm = self._active_context(batch)
        ti = batch.get("inputs") if batch is not None else None
        if ci is None or co is None or ti is None or ti.ndim != 2:
            return out
        from models.recursive_reasoning.core_prior import evidence_analogy
        side = int(math.isqrt(seq_len))
        with torch.no_grad():
            ci_c = ci.detach().to("cpu", torch.long); co_c = co.detach().to("cpu", torch.long)
            ti_c = ti.detach().to("cpu", torch.long)
            cm_c = (cm.detach().to("cpu").bool() if cm is not None
                    else torch.ones(ci.shape[:2], dtype=torch.bool))
            for b in range(batch_size):
                keep = cm_c[b].nonzero(as_tuple=True)[0]
                if keep.numel() == 0:
                    continue
                cell_prob, conf, _prov = evidence_analogy(ci_c[b][keep], co_c[b][keep], ti_c[b], side)
                out[b, :, :10] = cell_prob.to(device)
                out[b, :, 10] = float(conf)
        return out

    def _evidence_pd_color(
        self, batch: Dict[str, torch.Tensor] | None, batch_size: int, seq_len: int, device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """D8 (File #5): [B, L, PD_COLOR_DIM] cross-demo agreement + positional WHERE prior
        (pair_delta_v2 SS5). Batched tensor math (no per-row CPU loop). LODO-safe via _active_context."""
        ci, co, cm = self._active_context(batch)                            # M9
        ti = batch.get("inputs") if batch is not None else None
        if ci is None or co is None or ti is None or ti.ndim != 2:
            return (torch.zeros((batch_size, seq_len, PD_COLOR_DIM), device=device, dtype=torch.float32), {})
        from models.recursive_reasoning.pair_delta_v2 import pd_color_evidence
        cm_ = cm if cm is not None else torch.ones(ci.shape[:2], dtype=torch.bool, device=ci.device)
        return pd_color_evidence(ci, co, cm_, ti.to(device))

    def _evidence_pd_structure(
        self, batch: Dict[str, torch.Tensor] | None, batch_size: int, seq_len: int, device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """D9 (File #5): [B, L, PD_STRUCT_DIM] verified {preserve, transpose, bbox} extent-family
        masks (pair_delta_v2 SS6) for the structure lane. LODO-safe via _active_context."""
        ci, co, cm = self._active_context(batch)                            # M9
        ti = batch.get("inputs") if batch is not None else None
        if ci is None or co is None or ti is None or ti.ndim != 2:
            return (torch.zeros((batch_size, seq_len, PD_STRUCT_DIM), device=device, dtype=torch.float32), {})
        from models.recursive_reasoning.pair_delta_v2 import pd_structure_evidence
        cm_ = cm if cm is not None else torch.ones(ci.shape[:2], dtype=torch.bool, device=ci.device)
        return pd_structure_evidence(ci, co, cm_, ti.to(device))

    def _evidence_pd_bidi(
        self, batch: Dict[str, torch.Tensor] | None, batch_size: int, seq_len: int, device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """D10 (SS7): [B, L, PD_BIDI_DIM] reverse-direction (y->x) evidence -- invertibility,
        per-src deletion, dst-mass (pair_delta_v2 SS7). LODO-safe via _active_context."""
        ci, co, cm = self._active_context(batch)                            # M9
        ti = batch.get("inputs") if batch is not None else None
        if ci is None or co is None or ti is None or ti.ndim != 2:
            return (torch.zeros((batch_size, seq_len, PD_BIDI_DIM), device=device, dtype=torch.float32), {})
        from models.recursive_reasoning.pair_delta_v2 import pd_bidi_evidence
        cm_ = cm if cm is not None else torch.ones(ci.shape[:2], dtype=torch.bool, device=ci.device)
        return pd_bidi_evidence(ci, co, cm_, ti.to(device))

    def _collect_evidence(
        self,
        batch: Dict[str, torch.Tensor] | None,
        batch_rel_maps: torch.Tensor | None,
        grid_z: torch.Tensor,
        input_hints: Dict[str, torch.Tensor],
    ) -> Tuple[list, SimpleNamespace]:
        """Assemble the output-side evidence_parts (FIX A) in EVIDENCE_COLS order + the named pieces the
        colour/quarantine stages reuse. `ev` carries task_palette, the three input hints, value_v2 (+its
        stats/logits) and algo_maps. D-blocks (value_ctx_gate/verified_frame/analogy) append here in
        Block 8; until then their flags stay off and evidence_total excludes them."""
        canonical_binder = bool(getattr(self.config, "c2_canonical_value_binder", False))
        use_palette_feature = bool(getattr(self.config, "c2_task_palette_feature", False)) and not canonical_binder
        use_palette_bias = bool(getattr(self.config, "c2_task_palette_bias", False))
        need_rel_where_hint = bool(getattr(self.config, "c2_rel_where_hint", False))
        use_rel_where_hint = need_rel_where_hint and not canonical_binder
        use_pairdelta_intent_hint = (
            bool(getattr(self.config, "c2_pairdelta_intent_hint", False)) and not canonical_binder)
        task_palette = None
        if use_palette_feature or use_palette_bias:
            task_palette = self._task_palette_mask(batch, grid_z.shape[0], grid_z.device)

        evidence_parts: list = []
        if getattr(self.config, "c2_relmap", False):
            if batch_rel_maps is not None:
                evidence_parts.append(batch_rel_maps.to(grid_z.dtype))
            else:
                # M12: unreachable on the wired paths (_input_embeddings/_run_aux_logits thread rel_maps);
                # a debug caller that forgot them gets a zero relmap block -> the colour head loses its
                # X-ray evidence silently. Warn ONCE (same pattern as the input-side relmap fallback).
                if not getattr(self, "_evidence_relmap_fallback_warned", False):
                    import warnings
                    warnings.warn(
                        "c2_relmap on but no rel_maps reached _output_logits; appending a ZERO relmap "
                        "evidence block (colour head sees no relational evidence this forward). Wired "
                        "paths thread rel_maps -- this is a debug/direct-call fallback.",
                        RuntimeWarning, stacklevel=2)
                    self._evidence_relmap_fallback_warned = True
                evidence_parts.append(torch.zeros(
                    (*grid_z.shape[:2], REL_MAP_CHANNELS), device=grid_z.device, dtype=grid_z.dtype))
        if use_palette_feature:
            assert task_palette is not None
            evidence_parts.append(task_palette[:, None, :].expand(-1, grid_z.shape[1], -1).to(grid_z.dtype))
        rel_where_hint = None
        if need_rel_where_hint:
            rel_where_hint = input_hints.get("rel_where")
            if rel_where_hint is None:
                _wk = max(1, int(getattr(self.config, "c2_rel_where_topk", 1)))
                rel_where_hint = torch.zeros((*grid_z.shape[:2], _wk), device=grid_z.device, dtype=grid_z.dtype)
            else:
                rel_where_hint = rel_where_hint.to(grid_z.dtype)
            if use_rel_where_hint:
                evidence_parts.append(rel_where_hint)
        pairdelta_intent_hint = None
        if use_pairdelta_intent_hint:
            pairdelta_intent_hint = input_hints.get("pairdelta_intent")
            if pairdelta_intent_hint is None:
                pairdelta_intent_hint = torch.zeros((grid_z.shape[0], 1), device=grid_z.device, dtype=grid_z.dtype)
            else:
                pairdelta_intent_hint = pairdelta_intent_hint.to(grid_z.dtype)
            evidence_parts.append(pairdelta_intent_hint[:, None, :].expand(-1, grid_z.shape[1], -1))
        transition_hint = None
        if getattr(self.config, "c2_transition_hint", False) and not canonical_binder:
            transition_hint = self._transition_hint(batch, grid_z.shape[0], grid_z.shape[1], grid_z.device)
            evidence_parts.append(transition_hint.to(grid_z.dtype))
        value_v2 = None
        value_v2_stats: Dict[str, torch.Tensor] = {}
        value_v2_logits = None
        value_ctx_gate = None
        value_ctx_bind = None
        value_ctx_bind_logits = None
        value_ctx_bind_support = None
        # D7/D11 need value_v2's ctx pass even if the value_v2 COLUMNS themselves are off, so compute
        # value_v2 when ANY of the flags is on -- but only APPEND each block under its own flag.
        _need_v2 = (getattr(self.config, "c2_value_evidence_v2", False)
                    or getattr(self.config, "c2_value_ctx_gate", False)
                    or getattr(self.config, "c2_value_ctx_bind", False))
        if _need_v2:
            value_v2, value_v2_stats, value_ctx_gate, value_ctx_bind = self._value_evidence_v2(
                batch, grid_z.shape[0], grid_z.shape[1], grid_z.device,
                rel_maps=batch_rel_maps, rel_where_hint=rel_where_hint)
            if getattr(self.config, "c2_value_evidence_v2", False):
                v2_col_offset = sum(int(p.shape[-1]) for p in evidence_parts)
                evidence_parts.append(value_v2.to(grid_z.dtype))
                value_v2_logits = F.linear(
                    value_v2.to(self.color_evidence_proj.weight.dtype),
                    self.color_evidence_proj.weight[:, v2_col_offset:v2_col_offset + VALUE_EVIDENCE_V2_DIM],
                ).to(grid_z.dtype)
        algo_maps = None
        if getattr(self.config, "c2_algo_where_maps", False) and not canonical_binder:
            algo_maps = self._algo_where_maps(batch, grid_z.shape[0], grid_z.shape[1], grid_z.device)
            evidence_parts.append(algo_maps.to(grid_z.dtype))
        # --- APPENDED LAST (EVIDENCE_COLS order): the D-blocks, all zero-init in color_evidence_proj so
        #     step-0 logits are byte-identical even when the evidence is NONZERO (the F7 rule). -------
        if getattr(self.config, "c2_value_ctx_gate", False) and not canonical_binder:            # D7
            assert value_ctx_gate is not None, "c2_value_ctx_gate needs value_v2's ctx pass"
            evidence_parts.append(value_ctx_gate.to(grid_z.dtype))
        verified_frame = None
        if getattr(self.config, "c2_verified_frame_evidence", False) and not canonical_binder:   # D1 / E-1
            verified_frame = self._evidence_verified_frame(batch, grid_z.shape[0], grid_z.shape[1], grid_z.device)
            evidence_parts.append(verified_frame.to(grid_z.dtype))
        analogy = None
        if getattr(self.config, "c2_analogy_evidence", False) and not canonical_binder:          # D2 / E-2
            analogy = self._evidence_analogy(batch, grid_z.shape[0], grid_z.shape[1], grid_z.device)
            evidence_parts.append(analogy.to(grid_z.dtype))
        pd_color = None
        pd_color_stats: Dict[str, torch.Tensor] = {}
        if getattr(self.config, "c2_pairdelta_color_evidence", False) and not canonical_binder:  # D8 (File #5)
            pd_color, pd_color_stats = self._evidence_pd_color(
                batch, grid_z.shape[0], grid_z.shape[1], grid_z.device)
            evidence_parts.append(pd_color.to(grid_z.dtype))
        pd_bidi = None
        pd_bidi_stats: Dict[str, torch.Tensor] = {}
        if getattr(self.config, "c2_pairdelta_bidi_evidence", False) and not canonical_binder:   # D10 (SS7)
            pd_bidi, pd_bidi_stats = self._evidence_pd_bidi(
                batch, grid_z.shape[0], grid_z.shape[1], grid_z.device)
            evidence_parts.append(pd_bidi.to(grid_z.dtype))
        if getattr(self.config, "c2_value_ctx_bind", False) and not canonical_binder:            # D11 (codex)
            assert value_ctx_bind is not None, "c2_value_ctx_bind needs value_v2's ctx pass"
            evidence_parts.append(value_ctx_bind.to(grid_z.dtype))
            _bind_loc = evidence_slice(self.config, "value_ctx_bind")
            assert _bind_loc is not None and int(_bind_loc[1]) == int(value_ctx_bind.shape[-1]), (
                "value_ctx_bind evidence schema is inconsistent with the emitted tensor")
            _bind_off, _bind_width = int(_bind_loc[0]), int(_bind_loc[1])
            value_ctx_bind_logits = F.linear(
                value_ctx_bind.to(self.color_evidence_proj.weight.dtype),
                self.color_evidence_proj.weight[:, _bind_off:_bind_off + _bind_width],
            ).to(grid_z.dtype)
            value_ctx_bind_support = value_ctx_bind.abs().sum(dim=-1) > 0
        algo_touch = None
        if getattr(self.config, "c2_algo_where_touch", False) and not canonical_binder:          # D6 (B13)
            algo_touch = self._evidence_algo_touch(batch, grid_z.shape[0], grid_z.shape[1], grid_z.device)
            evidence_parts.append(algo_touch.to(grid_z.dtype))
        kinematic = None
        kin_stats: Dict[str, torch.Tensor] = {}
        if getattr(self.config, "c2_kinematic_evidence", False) and not canonical_binder:        # E-5 (A3)
            kinematic, kin_stats = self._evidence_kinematic(
                batch, grid_z.shape[0], grid_z.shape[1], grid_z.device)
            evidence_parts.append(kinematic.to(grid_z.dtype))
        canonical_bind = None
        canonical_bind_stats: Dict[str, torch.Tensor] = {}
        canonical_bind_logits = None
        canonical_bind_support = None
        canonical_bind_changed_support = None
        canonical_bind_copy_support = None
        canonical_bind_per_cell: Dict[str, torch.Tensor] = {}
        if canonical_binder:
            (
                canonical_bind,
                canonical_bind_stats,
                canonical_bind_support,
                canonical_bind_changed_support,
                canonical_bind_copy_support,
                canonical_bind_per_cell,
            ) = self._canonical_value_binding(batch, grid_z.shape[0], grid_z.shape[1], grid_z.device)
            evidence_parts.append(canonical_bind.to(grid_z.dtype))
            _canonical_loc = evidence_slice(self.config, "canonical_bind")
            assert _canonical_loc is not None and int(_canonical_loc[1]) == 10
            _canonical_off, _canonical_width = map(int, _canonical_loc)
            canonical_bind_logits = F.linear(
                canonical_bind.to(self.color_evidence_proj.weight.dtype),
                self.color_evidence_proj.weight[
                    :, _canonical_off:_canonical_off + _canonical_width],
            ).to(grid_z.dtype)
        ev = SimpleNamespace(
            task_palette=task_palette, use_palette_bias=use_palette_bias,
            rel_where_hint=rel_where_hint, pairdelta_intent_hint=pairdelta_intent_hint,
            transition_hint=transition_hint, value_v2=value_v2, value_v2_stats=value_v2_stats,
            value_v2_logits=value_v2_logits, algo_maps=algo_maps,
            value_ctx_gate=value_ctx_gate, verified_frame=verified_frame, analogy=analogy,
            pd_color=pd_color, pd_color_stats=pd_color_stats,
            pd_bidi=pd_bidi, pd_bidi_stats=pd_bidi_stats, value_ctx_bind=value_ctx_bind,
            value_ctx_bind_logits=value_ctx_bind_logits,
            value_ctx_bind_support=value_ctx_bind_support,
            algo_touch=algo_touch, kinematic=kinematic, kin_stats=kin_stats,
            canonical_bind=canonical_bind, canonical_bind_stats=canonical_bind_stats,
            canonical_bind_logits=canonical_bind_logits,
            canonical_bind_support=canonical_bind_support,
            canonical_bind_changed_support=canonical_bind_changed_support,
            canonical_bind_copy_support=canonical_bind_copy_support,
            canonical_bind_marginal=canonical_bind_per_cell.get("marginal"),
            canonical_bind_route=canonical_bind_per_cell.get("route"),
            canonical_bind_reliability=canonical_bind_per_cell.get("reliability"))
        return evidence_parts, ev

    def _color_logits(self, grid_z: torch.Tensor, evidence_parts: list, ev: SimpleNamespace) -> torch.Tensor:
        """color_head(grid_z) + zero-init color_evidence_proj(evidence) (+ optional interaction MLP,
        + palette bias). The evidence-width assert guards the schema against a half-wired D-block."""
        color_logits = self.color_head(grid_z)
        if evidence_parts:
            evidence_features = torch.cat(evidence_parts, dim=-1)
            assert evidence_features.shape[-1] == self.color_evidence_dim, (
                f"evidence width drifted: {evidence_features.shape[-1]} != {self.color_evidence_dim} "
                f"(flag set changed after __init__, or a D-block flag turned on before Block 8 wired it?)")
            color_logits = color_logits + self.color_evidence_proj(evidence_features)
        if getattr(self, "color_head_mlp_in", None) is not None:
            color_features = torch.cat([grid_z] + evidence_parts, dim=-1) if evidence_parts else grid_z
            color_logits = color_logits + self.color_head_mlp_out(F.silu(self.color_head_mlp_in(color_features)))
        if ev.use_palette_bias:
            assert ev.task_palette is not None
            color_logits = self._apply_task_palette_bias(color_logits, ev.task_palette)
        return color_logits

    def _structure_logits(
        self,
        z_H: torch.Tensor,
        grid_z: torch.Tensor,
        batch: Dict[str, torch.Tensor] | None,
        batch_rel_maps: torch.Tensor | None,
        floor_logits: torch.Tensor,
        input_hints: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """PAD/EOS/VALID logits: structure-from-lmhead (§15.8) or the fresh structure_head, + zero-init
        relmap proj (§15.6) + D9 pair-delta family masks (File #5) + the extent PAD/EOS near-hard levers
        (§15.9.1). Returns the logits plus the two extent-confidence scalars and the pd-struct verified
        fraction (each None when its lever is off)."""
        if bool(getattr(self.config, "c2_structure_from_lmhead", False)):
            fl = floor_logits.to(torch.float32)
            structure_logits = torch.cat(
                (fl[..., 0:1], fl[..., 1:2], torch.logsumexp(fl[..., 2:12], dim=-1, keepdim=True)),
                dim=-1,
            )
        else:
            structure_logits = self.structure_head(grid_z)
        if getattr(self, "structure_relmap_proj", None) is not None and batch_rel_maps is not None:
            structure_logits = structure_logits + self.structure_relmap_proj(batch_rel_maps.to(structure_logits.dtype))
        # D9 (File #5): pair-delta verified extent-family masks -> zero-init additive lever.
        pd_struct_conf = None
        if getattr(self, "structure_pairdelta_proj", None) is not None:
            pd_struct, pd_struct_stats = self._evidence_pd_structure(
                batch, structure_logits.shape[0], structure_logits.shape[1], structure_logits.device)
            structure_logits = structure_logits + self.structure_pairdelta_proj(
                pd_struct.to(structure_logits.dtype))
            if "pd_struct_conf" in pd_struct_stats:
                pd_struct_conf = pd_struct_stats["pd_struct_conf"].detach()
        outside_conf = None
        eos_conf = None
        _outside_on = (getattr(self.config, "c2_relmap_outside_grid", False)
                       and getattr(self, "structure_outside_proj", None) is not None)
        _eos_on = (getattr(self.config, "c2_relmap_eos_grid", False)
                   and getattr(self, "structure_eos_proj", None) is not None)
        if _outside_on or _eos_on:
            tgt_in = batch.get("inputs") if batch is not None else None
            if tgt_in is not None and tgt_in.ndim == 2 and tgt_in.shape[-1] == structure_logits.shape[1]:
                shape_logits = None
                if getattr(self.config, "c2_extent_use_shape_head", False) and self.config.c2_shape_head:
                    shape_logits = self._shape_logits(z_H, batch)
                ext = self._predicted_extent(batch, shape_logits=shape_logits)
                if ext is not None:
                    h_pred, w_pred, conf = ext
                    side = int(math.isqrt(tgt_in.shape[1]))
                    conf_col = conf.view(-1, 1)
                    if _outside_on:
                        pad_mask = extent_pad_mask(tgt_in, h_pred, w_pred, side)     # [B,L] eos-clean PAD
                        structure_logits = structure_logits + self.structure_outside_proj(
                            (pad_mask * conf_col).to(structure_logits.dtype).unsqueeze(-1))
                        outside_conf = conf.float().mean().detach()
                    if _eos_on:
                        eos_mask = extent_eos_mask(tgt_in, h_pred, w_pred, side)     # [B,L] thin-L EOS
                        structure_logits = structure_logits + self.structure_eos_proj(
                            (eos_mask * conf_col).to(structure_logits.dtype).unsqueeze(-1))
                        eos_conf = conf.float().mean().detach()
        if getattr(self.config, "c2_extent_conditioned_structure", False):
            fl = floor_logits.to(torch.float32)
            floor_structure = torch.cat((
                fl[..., 0:1],
                fl[..., 1:2],
                torch.logsumexp(fl[..., 2:12], dim=-1, keepdim=True),
            ), dim=-1).to(structure_logits.dtype)
            factors = (input_hints or {}).get("rule_factors")
            if factors is None:
                p_same_extent = torch.ones(
                    structure_logits.shape[0], device=structure_logits.device, dtype=structure_logits.dtype)
            else:
                from models.recursive_reasoning.core_prior import RULE_FACTOR_INDEX
                p_same_extent = factors[:, RULE_FACTOR_INDEX["extent_same"]]
            structure_logits = extent_conditioned_structure(
                floor_structure, structure_logits, p_same_extent)
        return structure_logits, outside_conf, eos_conf, pd_struct_conf

    def _quarantine_logits(
        self,
        batch: Dict[str, torch.Tensor] | None,
        grid_z: torch.Tensor,
        batch_rel_maps: torch.Tensor | None,
        ev: SimpleNamespace,
    ) -> torch.Tensor | None:
        """PID-quarantined candidate colour logits (linear warm-init + zero-init MLP residual) over the
        PID-free _quarantine_features. Returns None unless the quarantine head was built."""
        q_feats = self._quarantine_features(
            batch, grid_z.shape[0], grid_z.shape[1], grid_z.device, grid_z.dtype,
            rel_maps=batch_rel_maps, transition_hint=ev.transition_hint,
            rel_where_hint=ev.rel_where_hint, pairdelta_intent_hint=ev.pairdelta_intent_hint,
            task_palette=ev.task_palette,
            conditioned_hint=(ev.value_v2[..., 10:20] if ev.value_v2 is not None else None))
        return self.quarantine_lin(q_feats) + self.quarantine_mlp_out(F.silu(self.quarantine_mlp_in(q_feats)))

    def _assemble_candidate(
        self,
        structure_logp: torch.Tensor,
        cand_color_logp: torch.Tensor,
        structure_logits: torch.Tensor,
        floor_logits: torch.Tensor,
        color_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Factored candidate = [pad_logp | eos_logp | valid_logp + colour_logp]; optionally the
        floor-height-preserving hybrid (c2_candidate_floor_structure). Returns (candidate, factored)."""
        factored_candidate_logits = torch.cat(
            (
                structure_logp[..., 0:1],
                structure_logp[..., 1:2],
                structure_logp[..., 2:3] + cand_color_logp,
            ),
            dim=-1,
        ).to(color_dtype)
        if bool(getattr(self.config, "c2_candidate_floor_structure", False)):
            struct_pad_eos = structure_logits[..., 0:2].to(color_dtype)
            floor_color = floor_logits[..., 2:12].to(color_dtype)
            floor_is_valid = floor_color.amax(dim=-1, keepdim=True) > struct_pad_eos.amax(dim=-1, keepdim=True)
            color_delta = cand_color_logp - cand_color_logp.amax(dim=-1, keepdim=True)
            candidate_color = (floor_color.amax(dim=-1, keepdim=True) + color_delta).to(color_dtype)
            hybrid_color = torch.where(floor_is_valid, candidate_color, floor_color)
            candidate_logits = torch.cat((struct_pad_eos, hybrid_color), dim=-1)
        else:
            candidate_logits = factored_candidate_logits
        return candidate_logits, factored_candidate_logits

    def _output_logits(
        self,
        z_H: torch.Tensor,
        batch: Dict[str, torch.Tensor] | None = None,
        rel_maps: torch.Tensor | None = None,
        input_hints: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        input_hints = input_hints or {}
        grid_z = z_H[:, self.puzzle_emb_len:]
        if not self.config.c2_dual_output_head:
            return self._legacy_output_logits(z_H, grid_z)

        batch_rel_maps = rel_maps if rel_maps is not None else (
            batch.get("rel_maps") if batch is not None else None)
        floor_logits = self._floor_logits(z_H)
        extras: Dict[str, torch.Tensor] = {}
        if getattr(self.config, "c2_delta_expose_base_logits", False):
            extras["c2_pre_delta_logits"] = floor_logits.detach()     # M2: P_off in extras, not on self

        evidence_parts, ev = self._collect_evidence(batch, batch_rel_maps, grid_z, input_hints)
        color_logits = self._color_logits(grid_z, evidence_parts, ev)
        canonical_bind_base_logits = (
            color_logits - ev.canonical_bind_logits
            if ev.canonical_bind_logits is not None else None)
        structure_logits, outside_conf, eos_conf, pd_struct_conf = self._structure_logits(
            z_H, grid_z, batch, batch_rel_maps, floor_logits, input_hints=input_hints)

        structure_logp = F.log_softmax(structure_logits.to(torch.float32), dim=-1)
        color_logp = F.log_softmax(color_logits.to(torch.float32), dim=-1)
        split = bool(getattr(self.config, "c2_floor_candidate_split", False))
        # PID-QUARANTINED candidate colour source (MAIN stays the floor via `split`).
        q_logits = None
        cand_color_logp = color_logp
        if bool(getattr(self.config, "c2_quarantine_candidate", False)) and split:
            # M6: palette reaches this lane as a FEATURE only (QUARANTINE_COLS "palette");
            # --task-palette-bias does NOT bias the candidate colour distribution.
            q_logits = self._quarantine_logits(batch, grid_z, batch_rel_maps, ev)
            cand_color_logp = F.log_softmax(q_logits.to(torch.float32), dim=-1)

        candidate_logits, factored_candidate_logits = self._assemble_candidate(
            structure_logp, cand_color_logp, structure_logits, floor_logits, color_logits.dtype)
        logits = floor_logits.to(color_logits.dtype) if split else candidate_logits

        extras.update({
            "c2_color_logits": color_logits,
            "c2_structure_logits": structure_logits,
            "c2_dual_output_active": torch.as_tensor(1.0, device=z_H.device, dtype=torch.float32),
        })
        if outside_conf is not None:
            extras["c2_outside_grid_extent_conf"] = outside_conf
        if eos_conf is not None:
            extras["c2_eos_grid_extent_conf"] = eos_conf
        if ev.task_palette is not None:
            extras.update({
                "c2_task_palette_mask": ev.task_palette,
                "c2_task_palette_allowed_frac": ev.task_palette.float().mean().detach(),
                "c2_task_palette_allowed_count": ev.task_palette.float().sum(dim=-1).mean().detach(),
            })
        if ev.rel_where_hint is not None:
            extras.update({
                "c2_rel_where_hint": ev.rel_where_hint.detach(),
                "c2_rel_where_hint_mean": ev.rel_where_hint.float().mean().detach(),
            })
        if ev.algo_maps is not None:
            extras["c2_algo_where_maps"] = ev.algo_maps.detach()
        if ev.pairdelta_intent_hint is not None:
            extras.update({
                "c2_pairdelta_intent_hint": ev.pairdelta_intent_hint.detach(),
                "c2_pairdelta_intent_hint_mean": ev.pairdelta_intent_hint.float().mean().detach(),
            })
        if ev.transition_hint is not None:
            _colour_cells = (batch["inputs"] >= 2) if (batch is not None and "inputs" in batch) else None
            _mass = ev.transition_hint.sum(-1)
            extras["c2_transition_hint_coverage"] = (
                (_mass[_colour_cells] > 0).float().mean().detach()
                if _colour_cells is not None and bool(_colour_cells.any()) else _mass.mean().detach())
        if ev.value_v2 is not None:
            extras.update({k: v.detach() for k, v in ev.value_v2_stats.items()})
            if ev.value_v2_logits is not None:
                extras["c2_value_v2_logits"] = ev.value_v2_logits
        if ev.verified_frame is not None:
            # D1 coverage: fraction of ROWS where a frame verified (conf col > 0) -- the E-1 hit rate.
            extras["c2_verified_frame_conf"] = ev.verified_frame[..., 10].amax(dim=-1).float().mean().detach()
            extras["c2_verified_frame_coverage"] = (ev.verified_frame[..., :10].sum(-1) > 0).float().mean().detach()
        if ev.analogy is not None:
            extras["c2_analogy_conf"] = ev.analogy[..., 10].amax(dim=-1).float().mean().detach()
            extras["c2_analogy_coverage"] = (ev.analogy[..., :10].sum(-1) > 0).float().mean().detach()
        if ev.pd_color is not None:                                              # D8 (File #5)
            extras.update({f"c2_{k}": v.detach() for k, v in ev.pd_color_stats.items()})
        if ev.pd_bidi is not None:                                               # D10 (SS7)
            extras.update({f"c2_{k}": v.detach() for k, v in ev.pd_bidi_stats.items()})
        if ev.value_ctx_bind is not None:                                        # D11 (codex)
            extras["c2_value_ctx_bind_mass"] = ev.value_ctx_bind.abs().sum(-1).mean().detach()
            extras["c2_value_ctx_bind_logits"] = ev.value_ctx_bind_logits
            extras["c2_value_ctx_bind_support"] = ev.value_ctx_bind_support
        if ev.algo_touch is not None:                                            # D6 (B13)
            # coverage: fraction of cells whose object HAS a touching colour (mode one-hot mass)
            extras["c2_algo_touch_mass"] = ev.algo_touch[..., 0:10].sum(-1).mean().detach()
        if ev.kinematic is not None:                                             # E-5 (A3)
            extras.update({f"c2_{k}": v.detach() for k, v in ev.kin_stats.items()})
        if ev.canonical_bind is not None:
            extras.update({k: v.detach() for k, v in ev.canonical_bind_stats.items()})
            extras["c2_canonical_bind_logits"] = ev.canonical_bind_logits
            extras["c2_canonical_bind_support"] = ev.canonical_bind_support
            extras["c2_canonical_bind_changed_support"] = ev.canonical_bind_changed_support
            extras["c2_canonical_bind_copy_support"] = ev.canonical_bind_copy_support
            extras["c2_canonical_bind_base_logits"] = canonical_bind_base_logits
            # P1: raw distribution, K0 marginal, and per-cell route/reliability -- carried
            # SEPARATELY (route is applicability confidence, never probability mass). The
            # per-cell route key is distinct from the scalar panel stat of the same concept.
            extras["c2_canonical_bind_distribution"] = ev.canonical_bind
            extras["c2_canonical_bind_marginal"] = ev.canonical_bind_marginal
            extras["c2_canonical_bind_route"] = ev.canonical_bind_route
            extras["c2_canonical_bind_reliability"] = ev.canonical_bind_reliability
        if pd_struct_conf is not None:                                           # D9 (File #5)
            extras["c2_pd_struct_conf"] = pd_struct_conf
        if q_logits is not None:
            extras["c2_quarantine_logits"] = q_logits
        if split:
            extras.update({
                "c2_candidate_logits": candidate_logits,
                "c2_factored_candidate_logits": factored_candidate_logits,
                "c2_floor_logits": floor_logits.detach(),
                "c2_main_uses_floor": torch.as_tensor(1.0, device=z_H.device, dtype=torch.float32),
                "c2_candidate_floor_structure": torch.as_tensor(
                    1.0 if getattr(self.config, "c2_candidate_floor_structure", False) else 0.0,
                    device=z_H.device, dtype=torch.float32),
            })
        return logits, extras

    def _floor_logits(self, z_H: torch.Tensor) -> torch.Tensor:
        """The 518K-trained lm_head floor (grid cells only)."""
        return self.lm_head(z_H)[:, self.puzzle_emb_len:]

    def _shape_logits(self, z_H: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_z = z_H[:, self.puzzle_emb_len:]
        puzzle_z = z_H[:, 0]
        if self.config.c2_shape_pool == "zH_puzzle_inputvalid_gridmean":
            valid_mask = (batch["inputs"] >= 2).unsqueeze(-1).to(grid_z.dtype)
            grid_pool = (grid_z * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
            shape_state = torch.cat((puzzle_z, grid_pool), dim=-1)
        elif self.config.c2_shape_pool == "zH_rowcol":
            side = int(math.isqrt(self.config.seq_len))
            mag = grid_z.float().norm(dim=-1).view(-1, side, side)               # [B,side,side]
            row_prof = mag.mean(dim=2).to(grid_z.dtype)                          # [B,side] vertical (height)
            col_prof = mag.mean(dim=1).to(grid_z.dtype)                          # [B,side] horizontal (width)
            grid_pool = grid_z.mean(dim=1)
            shape_state = torch.cat((puzzle_z, grid_pool, row_prof, col_prof), dim=-1)
        elif self.config.c2_shape_pool == "zH_puzzle_gridmean":
            grid_pool = grid_z.mean(dim=1)
            shape_state = torch.cat((puzzle_z, grid_pool), dim=-1)
        else:
            raise ValueError(f"Unsupported c2_shape_pool={self.config.c2_shape_pool!r}")
        return self.shape_h_head(shape_state), self.shape_w_head(shape_state)

    def _aux_outputs(
        self,
        batch: Dict[str, torch.Tensor],
        seq_info: Dict[str, torch.Tensor | None],
    ) -> Dict[str, torch.Tensor]:
        aux_batch = self._build_lodo_batch(batch)
        if aux_batch is None:
            return {}
        if self.config.c2_lodo_blank_pid:
            aux_batch["puzzle_identifiers"] = torch.zeros_like(aux_batch["puzzle_identifiers"])
        else:
            aux_batch["puzzle_identifiers"] = self._maybe_drop_puzzle_ids(aux_batch["puzzle_identifiers"])

        aux_logits, aux_extra_outputs = self._run_aux_logits(
            aux_batch,
            seq_info,
            context_inputs_key="context_inputs",
            context_outputs_key="context_outputs",
            context_mask_key="context_mask",
            return_extras=True,
        )
        # M2: P_off comes from the LODO aux forward's OWN extras (threaded return), not a self-stash --
        # so the later shuffle forward cannot overwrite it and there is no call-order hazard.
        base_logits_lodo = aux_extra_outputs.get("c2_pre_delta_logits")
        outputs = {
            "c2_aux_logits": aux_logits,
            "c2_aux_labels": aux_batch["labels"],
            "c2_aux_inputs": aux_batch["inputs"],
            "c2_aux_valid": aux_batch["aux_valid"],
            "c2_aux_weight": torch.as_tensor(
                float(self.config.c2_leave_one_demo_weight),
                device=aux_batch["inputs"].device,
                dtype=torch.float32,
            ),
            "c2_lodo_blank_pid_active": torch.as_tensor(
                1.0 if self.config.c2_lodo_blank_pid else 0.0,
                device=aux_batch["inputs"].device,
                dtype=torch.float32,
            ),
            "c2_aux_positive_where_gate": torch.as_tensor(
                1.0 if getattr(self.config, "c2_positive_where_gate", False) else 0.0,
                device=aux_batch["inputs"].device,
                dtype=torch.float32,
            ),
        }
        if base_logits_lodo is not None:
            outputs["c2_aux_base_logits"] = base_logits_lodo   # P_off for the Stage-2 KL
        if "c2_value_v2_logits" in aux_extra_outputs:
            outputs["c2_aux_value_v2_logits"] = aux_extra_outputs["c2_value_v2_logits"]
        if "c2_value_ctx_bind_logits" in aux_extra_outputs:
            outputs["c2_aux_value_ctx_bind_logits"] = aux_extra_outputs["c2_value_ctx_bind_logits"]
            outputs["c2_aux_value_ctx_bind_support"] = aux_extra_outputs["c2_value_ctx_bind_support"]
        if "c2_canonical_bind_logits" in aux_extra_outputs:
            outputs["c2_aux_canonical_bind_logits"] = aux_extra_outputs["c2_canonical_bind_logits"]
            outputs["c2_aux_canonical_bind_support"] = aux_extra_outputs["c2_canonical_bind_support"]
            outputs["c2_aux_canonical_bind_changed_support"] = (
                aux_extra_outputs["c2_canonical_bind_changed_support"])
            outputs["c2_aux_canonical_bind_copy_support"] = (
                aux_extra_outputs["c2_canonical_bind_copy_support"])
            outputs["c2_aux_canonical_bind_base_logits"] = (
                aux_extra_outputs["c2_canonical_bind_base_logits"])
            # P1 extraction inputs (LODO pass): raw distribution, K0 marginal, route, reliability.
            outputs["c2_aux_canonical_bind_distribution"] = (
                aux_extra_outputs["c2_canonical_bind_distribution"])
            outputs["c2_aux_canonical_bind_marginal"] = (
                aux_extra_outputs["c2_canonical_bind_marginal"])
            outputs["c2_aux_canonical_bind_route"] = (
                aux_extra_outputs["c2_canonical_bind_route"])
            outputs["c2_aux_canonical_bind_reliability"] = (
                aux_extra_outputs["c2_canonical_bind_reliability"])
        if "c2_gate_where_values" in aux_extra_outputs:
            outputs["c2_aux_gate_where_values"] = aux_extra_outputs["c2_gate_where_values"]
        if self.config.c2_lodo_contrast_weight > 0 or getattr(self.config, "c2_lodo_force_shuffle", False):
            _need_shuffle_extras = bool(self.config.c2_token_gate_where)
            _shuffle_result = self._run_aux_logits(
                aux_batch,
                seq_info,
                context_inputs_key="shuffle_context_inputs",
                context_outputs_key="shuffle_context_outputs",
                context_mask_key="shuffle_context_mask",
                return_extras=_need_shuffle_extras,
            )
            if _need_shuffle_extras:
                shuffle_logits, shuffle_extra_outputs = _shuffle_result
            else:
                shuffle_logits = _shuffle_result
                shuffle_extra_outputs = {}
            outputs.update({
                "c2_lodo_shuffle_logits": shuffle_logits,
                "c2_lodo_shuffle_labels": aux_batch["labels"],
                "c2_lodo_shuffle_valid": aux_batch["shuffle_valid"],
                "c2_lodo_contrast_weight": torch.as_tensor(
                    float(self.config.c2_lodo_contrast_weight),
                    device=aux_batch["inputs"].device,
                    dtype=torch.float32,
                ),
                "c2_lodo_contrast_margin": torch.as_tensor(
                    float(self.config.c2_lodo_contrast_margin),
                    device=aux_batch["inputs"].device,
                    dtype=torch.float32,
                ),
            })
            if "c2_gate_where_values" in shuffle_extra_outputs:
                outputs["c2_shuffle_gate_where_values"] = shuffle_extra_outputs["c2_gate_where_values"]
        if getattr(self.config, "c2_lodo_zero_support", False) and self.config.c2_token_gate_where:
            # P3A Block 3: zero-support counterfactual. Same held-out targets, same context TENSORS,
            # all-false context mask -- the third arm of the WHERE ladder (correct / matched-count
            # shuffled / zero). A support-conditioned selector must collapse here; a support-blind
            # bias scores the same as with correct support. CrossAttention handles the all-false
            # memory mask (zero-valued all-valid bank through bias-free projections).
            aux_batch["zero_context_mask"] = torch.zeros_like(aux_batch["context_mask"])
            _zero_logits, zero_extra_outputs = self._run_aux_logits(
                aux_batch,
                seq_info,
                context_inputs_key="context_inputs",
                context_outputs_key="context_outputs",
                context_mask_key="zero_context_mask",
                return_extras=True,
            )
            if "c2_gate_where_values" in zero_extra_outputs:
                _zero_gate = zero_extra_outputs["c2_gate_where_values"]
                outputs["c2_zero_gate_where_values"] = _zero_gate
                outputs["c2_zero_support_valid"] = (
                    aux_batch["aux_valid"].to(torch.bool)
                    & torch.isfinite(_zero_gate).all(dim=-1)
                )
        return outputs

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)
        # M2: no `self._last_pre_delta_logits = None` clear -- P_off is threaded per-forward via extras.
        input_embeddings, c2_metrics, rel_maps, input_hints = self._input_embeddings(batch)

        z_H, z_L = self._run_recurrence(carry, input_embeddings, seq_info)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output, output_extras = self._output_logits(z_H, batch, rel_maps=rel_maps, input_hints=input_hints)
        if self.config.c2_shape_head:
            h_logits, w_logits = self._shape_logits(z_H, batch)
            output_extras.update({
                "c2_shape_h_logits": h_logits,
                "c2_shape_w_logits": w_logits,
            })
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        aux_outputs = self._aux_outputs(batch, seq_info)
        merged = {**c2_metrics, **output_extras, **aux_outputs}
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), merged


# ======================================================================================
# section 10 -- ACT WRAPPER (outer halting model)   (ROLE: ACT WRAPPER)  [verbatim]
# ======================================================================================
class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = FVR_C2_Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]):
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), c2_metrics = self.inner(new_inner_carry, new_current_data)
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            **c2_metrics,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step
            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
