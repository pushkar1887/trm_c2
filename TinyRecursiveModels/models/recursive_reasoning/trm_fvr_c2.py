from typing import Dict, Tuple

import math
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
from models.recursive_reasoning.object_bank import REL_MAP_CHANNELS
from models.sparse_embedding import CastedSparseEmbedding
try:
    from models.c2_siglip_rule_adapter import C2SigLIPRuleAdapter
except ModuleNotFoundError:
    C2SigLIPRuleAdapter = None
# M13 (backported from trm_fvr_v2): visual_arc_renderer's SOURCE was removed (dead lane). The hard
# import made this whole module unimportable, crashing anything that touches it (e.g. the value-v2-aux
# panel diagnostic). ARCTokenSpec is only used under c2_visual_rule_adapter=True (default off; that
# path already raises via the C2SigLIPRuleAdapter-is-None check), so guard it like the adapter above.
try:
    from models.visual_arc_renderer import ARCTokenSpec
except ModuleNotFoundError:
    ARCTokenSpec = None


VALUE_EVIDENCE_V2_DIM = 36
# FIX C: [enclosed(1) | flood-fill enclosing-colour one-hot(10) | nearest-seed-colour one-hot(10)]
ALGO_WHERE_MAP_DIM = 21
# Context-bucket capacity. The default relmap bucket uses 0..47; the Fix-2 rich key hashes into the
# full range. 512 covers both (ctx_counts is a transient no-grad tensor, not a saved param -> changing
# this is checkpoint-safe).
VALUE_EVIDENCE_V2_CONTEXT_BUCKETS = 512

# Operation-family vocab for the in-model rule-hypothesis hint (c2_rule_hypothesis_hint). Index 0 =
# "none" (no valid support / empty). The other four are exactly the families object_rule_bank.
# infer_rule_hypotheses can return; the hint embeds the TOP-ranked family per task.
RULE_FAMILY_VOCAB = ("none", "identity", "recolor", "rearrange", "size_change")
RULE_FAMILY_INDEX = {name: i for i, name in enumerate(RULE_FAMILY_VOCAB)}


class FVR_C2_Config(TinyRecursiveReasoningModel_ACTV1Config):
    c2_enabled: bool = True
    c2_num_context: int = 3
    c2_mode: str = "test_conditioned"
    c2_heads: int = 4
    c2_gate_init: float = 0.0
    c2_use_cross_demo: bool = True
    c2_pid_dropout: float = 0.0
    c2_leave_one_demo_weight: float = 0.0
    # Force the blank-pid LODO aux batch to be built even when c2_leave_one_demo_weight==0.
    # Needed when the cross-demo signal comes from the Phase-B delta-LODO loss (in the loss
    # head) instead of the weak built-in c2_aux CE: the model must still emit c2_aux_* outputs.
    c2_lodo_force_build: bool = False
    # Force the SHUFFLE (wrong-task) aux to be built even when c2_lodo_contrast_weight==0,
    # so the Phase-B two-region CONTRAST loss (real vs shuffle on changed cells) can run.
    c2_lodo_force_shuffle: bool = False
    # NOTE two DIFFERENT contrast knobs exist: these MODEL-side c2_lodo_contrast_* drive the legacy
    # gated-output row-CE contrast (and gate the shuffle build); the LOSS-side c2_delta_contrast_*
    # (losses_fvr / run script, margin 0.5) drive the Phase-B changed-cell hinge. Tune the loss-side one.
    c2_lodo_contrast_weight: float = 0.0
    c2_lodo_contrast_margin: float = 0.05
    c2_lodo_max_samples: int = 4
    c2_use_change_features: bool = True
    c2_lodo_blank_pid: bool = False
    # Stage 1: demo-derived task vector additively modulates the puzzle prefix
    # at the same scale as PID. Zero-init projector + scalar gate so this is
    # a no-op until training learns to use it.
    c2_modulate_pid: bool = False
    # Stage 1: per-token gate replaces the single scalar gate_patch so C2 can
    # apply different update magnitudes at different target positions.
    c2_per_token_gate: bool = False
    c2_gate_dropout: float = 0.0
    c2_gate_l2_weight: float = 0.0
    # Geometry recovery: split output into color and structure heads while
    # preserving the standard [B, S, vocab] logits contract.
    c2_dual_output_head: bool = True
    # V3 floor/candidate split: MAIN emits the old lm_head floor, while the
    # factored V3 head is exposed as a candidate and remains the LODO training
    # target. This is a safety split, not a new solver.
    c2_floor_candidate_split: bool = False
    # Candidate variant for the split: keep PAD/EOS/VALID behavior from the
    # floor and use the V3 head only for colour values on floor-valid cells.
    c2_candidate_floor_structure: bool = False
    c2_relmap: bool = True
    # Lane B: feed the deterministic solver's verified rearrange-FRAME family (an index into
    # object_rule_bank.FRAME_VOCAB; 0=none) as an INPUT-side rule-hypothesis hint. A zero-init embedding
    # broadcast-added to grid_features (F7-safe: 0 at step-0; the loss earns the binding). The dataloader
    # precomputes batch["frame_label"]. This is the rule-hypothesis BUS: the narrowing operation-family
    # signal the TRM combines with relmaps -- NOT the solved grid.
    c2_frame_hint: bool = False
    # Sibling of c2_frame_hint, derived LIVE in the forward: run object_rule_bank.infer_rule_hypotheses
    # on the (LODO-correct) support pairs, take the TOP operation-family, and broadcast-add a zero-init
    # embedding to grid_features (F7-safe: 0 at step-0; default path unchanged). Unlike c2_frame_hint
    # (dataloader-precomputed frame_label), this imports and calls the hypothesis inference inside the
    # model. NOTE: the rule-hypothesis TOKEN was measured DOA (--rule-probe ~3/20 actionable; C' 4th
    # neg) -- provided default-OFF for A/B, NOT expected to convert to exact solves. See plan Appendix Q.
    c2_rule_hypothesis_hint: bool = False
    # V3 colour safety/value prior: build a per-task palette from support inputs,
    # support outputs, and the target input. This is NOT CTBank and NOT a direct
    # recolour rule; it only tells the colour head which colours are available in
    # the current task. Feature = expose the palette to color_head. Bias = reduce
    # probability mass on absent colours. Hard mask stays off until coverage proves
    # it is safe.
    c2_task_palette_feature: bool = False
    c2_task_palette_bias: bool = False
    c2_task_palette_strength: float = 4.0
    c2_task_palette_hard: bool = False
    # Passed relational-colour probe signals, folded into existing components only:
    # rel_where_hint = ObjectBank/relmap WHERE evidence; pairdelta_intent_hint = cheap
    # PairDelta routing evidence. They are color_head input features only, never writers.
    c2_rel_where_hint: bool = False
    # FIX D: how many top-scoring rel-where predicate masks to expose as evidence columns (each scaled
    # by its own score). 1 = legacy single winner. Only the evidence path widens; the WHERE gate used
    # by value-v2 / quarantine stays channel-0 (the best predicate), so their semantics are unchanged.
    c2_rel_where_topk: int = 1
    # FIX C: algorithmic WHERE maps carrier -- 21 zero-init evidence columns from cell_conditioning_
    # signature cols 11/12: [enclosed(1) | flood-fill enclosing-colour one-hot(10) | nearest-seed-colour
    # one-hot(10)]. Separate columns in color_evidence_proj (appended LAST so earlier column layout is
    # checkpoint-stable), NOT relmap widening. nearest-seed one-hot is directly the VALUE for
    # adjacency-recolor tasks. CPU signature cost per forward; gated OFF by default.
    c2_algo_where_maps: bool = False
    c2_pairdelta_intent_hint: bool = False
    # VALUE-binding hint (the missing half of the WHERE hint): per-cell demo-consensus transition
    # distribution P(out_colour | in_colour) over CHANGED support cells, gathered at each target cell
    # by its INPUT colour -> 10 zero-init color_head columns. This routes the D1-proven task-specific
    # transition evidence DIRECTLY to the colour decision (one linear layer), bypassing the frozen
    # recurrence that attenuates input-side demo injections. Rows with no observed transition are
    # all-zero (natural confidence); the head LEARNS when to trust the consensus vs z_H/relmaps --
    # evidence in, learned selection, never a writer. LODO-safe: reads _active_context_* (holdout
    # excluded on the aux path). F7-safe: zero-init columns => step-0 byte-identical.
    c2_transition_hint: bool = False
    # VALUE evidence V2: copy-vs-change reliability plus context-conditioned changed-colour
    # distribution. This keeps the old marginal c2_transition_hint stable while adding the missing
    # question the colour head needs to answer: should this cell copy, or should it bind a changed
    # value under this local/object context? Evidence only, never a writer; LODO-safe via
    # _active_context_*; zero-init columns preserve step-0 behaviour.
    c2_value_evidence_v2: bool = False
    # Fix 2 (2026-07-04): use the RICHER object-aware key `cell_conditioning_signature` (sorted 4-nbr
    # colours + enclosing-object size-RANK/holes/D4-shape + container COLOUR) as the V2 context bucket,
    # instead of the coarse relmap bucket. Default OFF. HONEST CAVEAT: the offline Fix-1 probe measured
    # this key caps multi-target conditional_recolor VALUE acc at ~34% (exact-bucket) and soft-NN at
    # ~31% -- i.e. the multi-target value is NOT recoverable from per-cell features. This flag is banked
    # infrastructure, not a solve; forward-time compute is slower (dataloader precompute is the prod path).
    c2_value_v2_rich_ctx: bool = False
    # Interaction capacity for the colour decision: a LINEAR color_head cannot express
    # "IF input colour a THEN colour b" (a PRODUCT of input identity and rule evidence). This adds a
    # small MLP RESIDUAL on the same feature concat -- color_logits = linear(f) + W2·silu(W1·f) with
    # W2 zero-init -- so the lm_head-warm-started linear path is untouched and step-0 is byte-identical
    # (F7-safe). 0 = off; 128 is the suggested dim. Molds the existing head, adds no new writer.
    c2_color_head_mlp_dim: int = 0
    # PID-QUARANTINED CANDIDATE (the cheat-proof colour lane). A small MLP head whose input contains
    # NO PID-conditioned feature: target-input one-hot + 3x3 neighbourhood + transition hint + rel-where
    # + palette + intent + relmap -- every part is a demo/target-derived aggregate, z_H is never read.
    # Closes the three measured cheats BY CONSTRUCTION instead of by counter-loss:
    #   (1) cannot memorise -- no PID in its input path. (Also fixes a silent train/deploy contract
    #       mismatch of the z_H candidate: the LODO aux path trains on BLANK-PID z_H
    #       (c2_lodo_blank_pid) while the main forward scores the candidate on PID-ful z_H;
    #       quarantine features are IDENTICAL on both paths.)
    #   (2) with --color-perm, marginal colour statistics are worthless while the transition table
    #       permutes consistently with the target -> the only CE-reducing direction is reading the table;
    #   (3) trained head-only (run_stage1_local --train-scope quarantine), z_H is not in the gradient
    #       path -> contrast/drift cannot pay for anything and MAIN risk is structurally zero.
    # The candidate colour CHOICE comes from this head (anchored at the floor colour height, the same
    # scale mechanics as c2_candidate_floor_structure); the canvas stays the solved floor structure.
    # Only meaningful under c2_floor_candidate_split (without the split there is no candidate lane).
    c2_quarantine_candidate: bool = False
    c2_quarantine_hidden: int = 256
    # §15.2 C2 UPGRADE (cross-demo, input-side, F7-safe): feed the SUPPORT-side relational maps into
    # TestConditionedC2's demo features BEFORE the cross-attention, so the demo->target matching can use
    # object/inside/distance facts instead of token embeddings alone. Uses a SEPARATE zero-init projection
    # (c2_demo_relmap_proj) -> step-0 forward is byte-identical to map-only baseline. Needs c2_relmap on
    # (the dataloader then also supplies context_rel_maps + context_output_rel_maps). Default OFF: this is
    # an A/B upgrade (map-only vs map+C2), not part of the V3-clean floor.
    c2_relmap_demos: bool = False
    # §15.2 PairDelta DEMOTION: the learned cross-demo rule_vec as an INPUT-ONLY zero-init hint (broadcast
    # add to grid_features), NOT an output writer. Builds its OWN PairDeltaEncoder + zero-init proj,
    # independent of the (gated-off) c2_delta_rule_branch, so it is usable under V3-clean. Default OFF.
    c2_pairdelta_input_feature: bool = False
    # Geometry recovery: train an auxiliary PAD/EOS/VALID structure classifier
    # while leaving the standard lm_head decoder unchanged.
    c2_geometry_aux_head: bool = True
    # §15.6 STRUCTURE-READS-MAP fix (the factored-head pad/shape weakness): the factored color_head reads
    # the relmap but structure_head was BLIND to it, so PAD-vs-VALID had to be inferred from grid_z alone
    # (=> shape degrades under backbone drift). The relmap's valid_mask/distance/boundary channels decide
    # the canvas directly. Implemented as a SEPARATE zero-init additive logit bias (structure_relmap_proj)
    # so the legacy structure_head(grid_z) aux call is untouched and step-0 is byte-identical (F7-safe).
    # NOTE: relmap is computed from the INPUT grid -> a strong output-boundary signal for shape-PRESERVING
    # tasks; for size-change the learned zero-init proj must earn its weight (structure CE teaches restraint).
    c2_relmap_structure: bool = True
    # §15.8 STRUCTURE-FROM-LM_HEAD: in the factored dual head, derive PAD/EOS/VALID from the trained
    # lm_head (validity logit = logsumexp over the 10 colour logits) instead of a fresh structure_head.
    # log_softmax([pad, eos, logsumexp(colour)]) reproduces the floor's structure partition EXACTLY
    # (logsumexp([a,b,logsumexp([c...])]) == logsumexp([a,b,c...])), so the factored head inherits
    # lm_head's good pad/eos/shape BY CONSTRUCTION. The fresh structure_head only had ~300 probe steps ->
    # that is the LODO pad/shape regression. color_head still owns the colour CHOICE inside VALID cells:
    # "lm_head structure / relmap colour". Warm-start-independent and floor-safe.
    c2_structure_from_lmhead: bool = False
    # §15.9.1 EXTENT-BASED PAD LEVER. The relmap is all-zero on true-PAD cells (every channel *
    # valid_mask), so the structure head has no per-cell landmark to PLACE pad. extent_pad_mask supplies
    # it: PAD == cells outside the PREDICTED output box (_predicted_extent: verified demo size-rule
    # {identity, constant, integer-ratio}, support-safe; the box offset comes from the input bbox, which
    # the tokenizer shares between input and output). Same-shape is just the identity rule; size-change
    # is constant/ratio -> ONE mechanism, no per-regime gate. Fed to a dedicated structure_outside_proj
    # [1->3], scaled by conf (1 = demo-verified, 0 = floor untouched).
    c2_relmap_outside_grid: bool = False
    # Warm-init the outside_grid PAD row so pad is asserted on the padding from step 0 instead of climbing
    # from zero over ~1500 steps (breaks step-0==floor deliberately; frozen-core is MAIN-safe).
    c2_structure_outside_warm_init: bool = False
    # Half the pad-vs-valid swing (applied as pad +V, eos/valid -V => swing 2V). MEASURED colour-over-pad
    # gap on the 518K floor: mean~177, p90~300, max~620 (scripts/verify_outside_grid_lever.py), so the
    # default V=1000 (swing 2000) dominates the measured max with >3x margin. Safe: the mask is eos-clean
    # + conf-scaled, and the verifier asserts 0 target-pad cells with gap > 2V.
    c2_structure_outside_warm_init_value: float = 1000.0
    # Thin-L EOS analogue of the outside-grid PAD lever. EOS is not the outside region; it is the row just
    # below the predicted output box plus the column just right of it. Keep this as a separate projection so
    # PAD and EOS do not share contradictory weights.
    c2_relmap_eos_grid: bool = False
    c2_structure_eos_warm_init: bool = False
    c2_structure_eos_warm_init_value: float = 1000.0
    # Optional LEARNED extent fallback: the shape head supplies (h,w) for rows no closed-form size rule
    # verifies, but only where its softmax margin clears tau on BOTH axes -- and at a CAPPED confidence
    # (conf below, < 1), because a learned argmax is not a demo-verified proof and the override is
    # near-hard. Both extent levers share _predicted_extent, so this governs PAD and EOS together.
    c2_extent_use_shape_head: bool = False
    c2_extent_shape_head_tau: float = 0.5
    c2_extent_shape_head_conf: float = 0.5
    c2_shape_head: bool = False
    c2_shape_pool: str = "zH_puzzle_gridmean"
    # Inference-only diagnostic: residual fusion from trained structure_head
    # into preserved lm_head logits. Zero reproduces clean C0 exactly.
    c2_structure_fusion_alpha: float = 0.0
    # PairDelta encoder dims (the KEPT --pairdelta-input hint reuses these; the old delta-rule branch
    # that also used them was DELETED 2026-07-01).
    c2_delta_rule_encoder_dim: int = 256
    c2_delta_rule_slots: int = 8
    # Surface the pre-branch base logits (P_off, the lm_head floor) so the loss head can compute the
    # preservation KL and the selector can score floor-vs-candidate. PURE EXPOSURE (no-op when unused).
    # (The rule-vec exposure + NCE/cons lane and the ColorTransitionBank were REMOVED 2026-07-02:
    # their producers died in the delta-branch deletion, leaving readers that could never fire.)
    c2_delta_expose_base_logits: bool = False
    # Frozen visual cache branch. Default off. When enabled, the dataset
    # supplies cached 16x16 patch features [B, 256, D] for each token grid;
    # the model upsamples them to 30x30 and adds a zero-gated residual to the
    # preserved token embedding path.
    c2_visual_encoder: bool = False
    c2_visual_cache_path: str | None = None
    c2_visual_model_name: str | None = None
    c2_visual_gate_init: float = 0.0
    c2_visual_feature_dim: int = 1024
    c2_visual_project_dim: int = 512
    c2_visual_cache_level: str = "dino_patch_16x16"
    # Frozen pooled SigLIP-B demo-delta rule adapter. This is separate from the
    # cached patch visual branch above and defaults off.
    c2_visual_rule_adapter: bool = False
    c2_visual_encoder_name: str = "google/siglip-base-patch16-224"
    c2_visual_mode: str = "pooled_demo_delta_symbolic"
    c2_visual_rule_dim: int = 256
    c2_visual_use_query_output: bool = False


def _select_heads(hidden_size: int, requested_heads: int, max_heads: int) -> int:
    heads = max(1, min(requested_heads, max_heads))
    while hidden_size % heads != 0:
        heads -= 1
    return heads


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


class TokenGridEncoder(nn.Module):
    """Token-grid feature boundary for C2.

    C2 consumes [B, S, D] feature fields. Today those fields come from TRM's
    token embedding table; a future visual encoder can replace this class while
    keeping TestConditionedC2 unchanged.
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
        if self.config.c2_per_token_gate:
            self.gate_patch_token = CastedLinear(config.hidden_size, 1, bias=True)
            with torch.no_grad():
                self.gate_patch_token.weight.zero_()
                self.gate_patch_token.bias.fill_(float(config.c2_gate_init))

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor | None]:
        demo_tokens, token_mask, token_keys, changed_mask = self._demo_tokens(
            context_inputs=context_inputs,
            context_outputs=context_outputs,
            context_input_features=context_input_features,
            context_output_features=context_output_features,
            context_mask=context_mask,
        )
        rule_bank, rule_mask = self._rule_bank(demo_tokens, token_mask, token_keys)

        patch_context = self.cross_attn(
            query=rms_norm(target_features, variance_epsilon=self.norm_eps),
            key_value=rms_norm(rule_bank, variance_epsilon=self.norm_eps),
            key_mask=rule_mask,
        )
        patch_context = self.patch_proj(patch_context)

        global_context = self._masked_mean(rule_bank, rule_mask)
        global_context = self.global_proj(global_context).unsqueeze(1)

        gate_global = torch.tanh(self.gate_global).to(target_features.dtype)
        if self.config.c2_per_token_gate:
            # Per-token gate: tanh of a learned linear projection of each target
            # token's features. Weight zero-init + bias=c2_gate_init keeps the
            # initial gate uniform and matches the scalar baseline.
            normed_target = rms_norm(target_features, variance_epsilon=self.norm_eps)
            gate_logits = self.gate_patch_token(normed_target).squeeze(-1)
            gate_patch_per_token = torch.tanh(gate_logits).to(target_features.dtype)
            if self.training and self.config.c2_gate_dropout > 0:
                keep = torch.rand_like(gate_patch_per_token.float()) > float(self.config.c2_gate_dropout)
                gate_patch_per_token = gate_patch_per_token * keep.to(gate_patch_per_token.dtype)
            gate_patch_field = gate_patch_per_token.unsqueeze(-1)
            gate_patch_scalar_metric = gate_patch_per_token.float().mean().detach()
            gate_patch_abs_metric = gate_patch_per_token.float().abs().mean().detach()
            gate_patch_std_metric = gate_patch_per_token.float().std().detach()
            gate_patch_l2 = gate_patch_per_token.float().square().mean()
        else:
            gate_patch_scalar = torch.tanh(self.gate_patch).to(target_features.dtype)
            gate_patch_field = gate_patch_scalar
            gate_patch_scalar_metric = torch.tanh(self.gate_patch.float()).detach()
            gate_patch_abs_metric = torch.tanh(self.gate_patch.float()).abs().detach()
            gate_patch_std_metric = torch.zeros((), device=target_features.device).detach()
            gate_patch_l2 = torch.zeros((), device=target_features.device, dtype=torch.float32)

        patch_update = gate_patch_field * rms_norm(patch_context, variance_epsilon=self.norm_eps)
        global_update = gate_global * rms_norm(global_context, variance_epsilon=self.norm_eps)
        update = patch_update + global_update
        # DIAGNOSTIC-ONLY forced-signal amplifier (run_stage1_local --zh-amp): scales the demo->target
        # update to answer "is the path too weak or disconnected?". Default 1.0 == exact no-op; no
        # training code sets it. If amplified z_H moves but trained z_H does not, the path works and
        # scale/optimization is the blocker; if even 50x moves nothing, the path is dead.
        _amp = float(getattr(self, "_demo_injection_scale", 1.0))
        if _amp != 1.0:
            update = update * _amp
        target_norm = target_features.float().norm(dim=-1).mean().clamp_min(1e-6)
        patch_update_norm = patch_update.float().norm(dim=-1).mean()
        global_update_norm = global_update.float().norm(dim=-1).mean()
        update_norm = update.float().norm(dim=-1).mean()
        target_features = target_features + update
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
            "c2_update_norm_ratio": (update_norm / target_norm).detach(),
            "c2_patch_update_norm_ratio": (patch_update_norm / target_norm).detach(),
            "c2_global_update_norm_ratio": (global_update_norm / target_norm).detach(),
            "c2_rule_bank_token_count": rule_mask.float().sum(dim=-1).mean().detach(),
            "c2_changed_token_frac": (changed_mask.float().sum() / valid_count).detach(),
            "c2_valid_token_frac": (token_mask.float().sum() / valid_possible).detach(),
        }
        return target_features, metrics, pid_task_vec


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
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
            # FIX A (2026-07-05): evidence columns moved OUT of color_head into a SEPARATE zero-init
            # color_evidence_proj. Welded inside color_head.weight they were pinned to the core-safe
            # lr (param groups split per-TENSOR): measured V2TAIL w_norm 6e-6->6.7e-3 over 300 steps
            # at lr 1e-5, logit_abs ~4e-5 vs warm-started logits of 5-20 -> every evidence column was
            # inert in every run. A separate tensor can join a dedicated optimizer group
            # (--evidence-lr, wd=0). Function class identical: cat-linear == sum of two linears.
            # Evidence order (must match the evidence_parts concat in _output_logits):
            # relmap, palette, where-hint(topk), intent-hint, transition-hint, value-evidence-v2,
            # algo-where-maps (FIX C, appended LAST for checkpoint column stability).
            _where_k = max(1, int(getattr(self.config, "c2_rel_where_topk", 1)))
            extra_cols = (
                (REL_MAP_CHANNELS if getattr(self.config, "c2_relmap", False) else 0)
                + (10 if getattr(self.config, "c2_task_palette_feature", False) else 0)
                + (_where_k if getattr(self.config, "c2_rel_where_hint", False) else 0)
                + (1 if getattr(self.config, "c2_pairdelta_intent_hint", False) else 0)
                + (10 if getattr(self.config, "c2_transition_hint", False) else 0)
                + (VALUE_EVIDENCE_V2_DIM if getattr(self.config, "c2_value_evidence_v2", False) else 0)
                + (ALGO_WHERE_MAP_DIM if getattr(self.config, "c2_algo_where_maps", False) else 0)
            )
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
        # PID-QUARANTINED candidate head (config note above). Feature layout MUST match
        # _quarantine_features: [0:12] input one-hot | [12:108] 8-neighbour one-hots |
        # [108:118] transition hint | [118:119] rel-where | [119:129] palette | [129:130] intent |
        # [130:130+REL_MAP_CHANNELS] relmap. Warm-init = copy-unless-consensus: logit(c) gets +4 from
        # "input IS colour c" and +8 from "demo consensus says c" (so consensus P>0.5 beats copy),
        # making the step-0 candidate the D1 deterministic baseline; the MLP residual (zero-init out)
        # learns corrections and input-colour x evidence interactions on top.
        if getattr(self.config, "c2_quarantine_candidate", False):
            if not getattr(self.config, "c2_floor_candidate_split", False):
                import warnings
                warnings.warn(
                    "c2_quarantine_candidate=True but c2_floor_candidate_split=False: the quarantined "
                    "head only feeds the CANDIDATE lane; without the split it is INERT. Enable "
                    "--floor-candidate-split.", RuntimeWarning, stacklevel=2)
            # FIX B (2026-07-05): +10 CONDITIONED columns [130:140] = P(out | src, context) with Katz
            # backoff (the _value_evidence_v2[...,10:20] block). The marginal [108:118] is identical for
            # every cell of a source colour -> structurally cannot express multi-target recolor (52/75
            # of conditional_recolor); the conditioned table measured ~34 vs ~29 val_acc on multi
            # (--value-binding-probe). Warm-init +9 > marginal +8 > copy +4: conditioned consensus wins
            # when present, backoff makes it equal the marginal when the context bucket is sparse.
            _q_in = 140 + REL_MAP_CHANNELS
            _q_h = int(getattr(self.config, "c2_quarantine_hidden", 256))
            self.quarantine_lin = CastedLinear(_q_in, 10, bias=False)
            self.quarantine_mlp_in = CastedLinear(_q_in, _q_h, bias=True)
            self.quarantine_mlp_out = CastedLinear(_q_h, 10, bias=False)
            with torch.no_grad():
                self.quarantine_lin.weight.zero_()
                self.quarantine_mlp_out.weight.zero_()
                for _c in range(10):
                    self.quarantine_lin.weight[_c, 2 + _c] = 4.0     # copy: input colour c -> logit c
                    self.quarantine_lin.weight[_c, 108 + _c] = 8.0   # marginal consensus P(out=c|in)
                    self.quarantine_lin.weight[_c, 130 + _c] = 9.0   # conditioned consensus P(out=c|in,ctx)
        if getattr(self.config, "c2_relmap", False):
            self.relmap_proj = CastedLinear(REL_MAP_CHANNELS, self.config.hidden_size, bias=False)
            with torch.no_grad():
                self.relmap_proj.weight.zero_()
        # Lane B: zero-init embedding of the deterministic FRAME family (the rule-hypothesis hint). Added
        # to grid_features input-side -> the TRM combines the narrowed operation family with relmaps/C2.
        # Zero-init => step-0 byte-identical (F7-safe); the loss earns the binding.
        if getattr(self.config, "c2_frame_hint", False):
            from models.recursive_reasoning.object_rule_bank import FRAME_VOCAB
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
        if getattr(self.config, "c2_relmap_demos", False) and getattr(self.config, "c2_relmap", False):
            self.c2_demo_relmap_proj = CastedLinear(REL_MAP_CHANNELS, self.config.hidden_size, bias=False)
            with torch.no_grad():
                self.c2_demo_relmap_proj.weight.zero_()
        # §15.2-B: PairDelta as an INPUT-ONLY zero-init hint (own encoder; independent of the delta branch
        # so it survives V3-clean). delta_rule_input_proj zero-init => no-op at step 0.
        if getattr(self.config, "c2_pairdelta_input_feature", False):
            from models.recursive_reasoning.pair_delta_encoder import PairDeltaEncoder as _PDE
            _enc_dim = int(getattr(self.config, "c2_delta_rule_encoder_dim", 256))
            self.pairdelta_input_encoder = _PDE(
                hidden_dim=_enc_dim, n_slots=int(getattr(self.config, "c2_delta_rule_slots", 8)))
            self.delta_rule_input_proj = CastedLinear(_enc_dim, self.config.hidden_size, bias=False)
            with torch.no_grad():
                self.delta_rule_input_proj.weight.zero_()
        if self.config.c2_dual_output_head or self.config.c2_geometry_aux_head:
            self.structure_head = CastedLinear(self.config.hidden_size, 3, bias=False)
        # §15.6: let structure_head SEE the relmap. Separate zero-init projection [REL_MAP_CHANNELS->3] added to the
        # structure logits in the dual recombine -> PAD/EOS/VALID can read valid_mask/distance/boundary
        # directly. Zero-init => step-0 unchanged (F7-safe); the structure CE earns the weight.
        # §15.9 BOUNDARY LEVER (bias=True): a weight-only proj is FEATURELESS on true-PAD cells (every relmap
        # channel ~0 outside the grid) -> it could lift VALID where valid_mask=1 but never PLACE pad, so the
        # boundary had to be learned through the shared core (which damaged the floor colour under --unified).
        # bias b + a learned negative weight on valid_mask gives  pad_logit = b - k*valid_mask  (high on pad,
        # low on valid) == an implicit (1 - valid_mask) channel: the head COPIES the input boundary instead of
        # over-predicting pad. CastedLinear zero-inits the bias, weight is zeroed below -> step-0 == floor (F7-safe).
        if (getattr(self.config, "c2_relmap", False)
                and getattr(self.config, "c2_relmap_structure", False)
                and (self.config.c2_dual_output_head or self.config.c2_geometry_aux_head)):
            self.structure_relmap_proj = CastedLinear(REL_MAP_CHANNELS, 3, bias=True)
            with torch.no_grad():
                self.structure_relmap_proj.weight.zero_()
                self.structure_relmap_proj.bias.zero_()
        # §15.9.1 extent PAD override: a dedicated [1->3] proj reading extent_pad_mask (PAD == outside the
        # PREDICTED output box, conf-scaled; see _predicted_extent). Zero-init (F7-safe) UNLESS warm-init.
        # MEASURED (scripts/verify_outside_grid_lever.py, 518K floor): the frozen lm_head's colour-over-pad
        # gap on padding is mean~177 / max~620 (stablemax saturates logits), so a small additive lever
        # CANNOT flip pad. The warm-init is therefore a near-HARD override: pad row +V, eos+valid rows -V
        # (swing 2V, dominates the gap; the verifier asserts 0 cells with gap > 2V). The mask is eos-clean,
        # so EOS cells are untouched.
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
        # EOS uses a different geometry than PAD: the thin-L boundary of the predicted output box. It needs
        # its own [1->3] projection; sharing the PAD projection would encode contradictory targets.
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
        if self.config.c2_shape_head:
            # zH_rowcol adds per-row + per-col occupancy profiles ([side]+[side]) so the head can
            # SEPARATE height from width. A plain grid MEAN pool is permutation-invariant over the
            # 900 cells -> dimension-blind -> the head collapses (probed: h_pred stuck near height 2).
            _shape_extra = 2 * int(math.isqrt(self.config.seq_len)) if self.config.c2_shape_pool == "zH_rowcol" else 0
            self.shape_h_head = CastedLinear(2 * self.config.hidden_size + _shape_extra, 30, bias=True)
            self.shape_w_head = CastedLinear(2 * self.config.hidden_size + _shape_extra, 30, bias=True)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        # --- §15.0 V3-CLEAN INVARIANT GUARD (no behaviour change; surfaces config drift) ----------
        # The anti-fighting invariant: in V3-clean the factored structure_head ⟂ color_head is the
        # SOLE output writer; relational maps enter as input evidence AND are READ by color_head at
        # output (1677+). Two ways a config silently violates V3 -- both warned here, not raised, so a
        # run is never broken, only flagged. See plan §15.0/§15.3.
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
            # Demo-derived task vector additively modulates the raw puzzle
            # embedding (puzzle_emb_ndim dims), so it lives in the same
            # position as the real PID after the pad/reshape that follows.
            # The scalar gate is the zero-init lever (tanh(0)=0 => no-op at
            # init); the modulator weight uses default trunc-normal so gradient
            # can flow to the gate at step 1. Zero-init on BOTH would deadlock.
            self.pid_task_modulator = CastedLinear(
                self.config.hidden_size,
                self.config.puzzle_emb_ndim,
                bias=False,
            )
            self.pid_task_gate = nn.Parameter(torch.tensor(0.0))
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _ in range(self.config.L_layers)]
        )

        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _maybe_drop_puzzle_ids(self, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        if self.training and self.config.c2_pid_dropout > 0:
            drop = torch.rand_like(puzzle_identifiers.float()) < self.config.c2_pid_dropout
            return torch.where(drop, torch.zeros_like(puzzle_identifiers), puzzle_identifiers)
        return puzzle_identifiers

    def _puzzle_embedding(self, puzzle_identifiers: torch.Tensor, use_sparse_training_buffer: bool) -> torch.Tensor:
        if self.training and not use_sparse_training_buffer:
            return self.puzzle_emb.weights[puzzle_identifiers].to(self.forward_dtype)
        return self.puzzle_emb(puzzle_identifiers)

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
                puzzle_embedding = puzzle_embedding + gate * modulation
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            puzzle_embedding = puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size)
            embedding = torch.cat((self.embed_scale * puzzle_embedding, embedding), dim=-2)

        if self.config.pos_encodings == "learned":
            pos = self.embed_scale * self.embed_pos.embedding_weight.to(self.forward_dtype)
            embedding = 0.707106781 * (embedding + pos)
        return embedding

    # Output-side writer flags (plan §15.3). In V3-clean (c2_dual_output_head=True) NONE of these
    # execute -- they all live under `if not c2_dual_output_head` -- so enabling one is silently
    # inert. Listing them here is the single source of truth for the invariant guard.
    def _v3_clean_invariant_check(self) -> None:
        """Warn (never raise) on the two ways a config silently defeats V3-clean. See plan §15.0/§15.3."""
        import warnings
        relmap = bool(getattr(self.config, "c2_relmap", False))
        dual = bool(getattr(self.config, "c2_dual_output_head", False))

        # (1) THE WIRING GAP: relmap injected as input evidence but the factored color_head that READS
        # it at output (trm_fvr_c2 ~1677) only runs when dual=True. With dual=False the X-ray enters as
        # a zero-init input residual and is NEVER read at output => the V3 colour-lookup thesis (§12.4)
        # is inactive; the run is effectively legacy lm_head + an inert relmap residual.
        if relmap and not dual:
            warnings.warn(
                "[V3-clean section 15] c2_relmap=True but c2_dual_output_head=False: relational maps are added "
                "to the INPUT (zero-init, inert until trained) but the factored color_head that READS them "
                "at output is NOT built/called -- the run is NOT exercising V3. Set c2_dual_output_head=True "
                "(run_stage1_local.py --v3-clean) to activate the factored structure/color head.",
                RuntimeWarning, stacklevel=2,
            )

        # (2) THE INVARIANT: in dual mode the factored head is the SOLE writer. The 11 competing output
        # writers were physically DELETED (2026-07-01); the only remaining legacy knob is the geomaux
        # structure-fusion, which is inert under V3-clean (alpha forced 0). Flag it if a stale config sets it.
        if dual and float(getattr(self.config, "c2_structure_fusion_alpha", 0.0)) != 0.0:
            warnings.warn(
                "[V3-clean section 15] c2_dual_output_head=True (factored head is the sole output writer), but "
                "c2_structure_fusion_alpha != 0 -- that geomaux fusion is inert under V3-clean. Set it to 0.",
                RuntimeWarning, stacklevel=2,
            )

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
        # _output_logits, so main / aux-LODO / shuffle forwards each carry the hint computed from THEIR
        # own context without cross-call state.
        input_hints: Dict[str, torch.Tensor] = {}
        grid_features = self.grid_encoder(target_inputs, target_visual_features)
        c2_metrics: Dict[str, torch.Tensor] = {}
        pid_task_vec: torch.Tensor | None = None
        if self.c2 is not None and context_inputs is not None and context_outputs is not None and context_mask is not None:
            context_input_features = self.grid_encoder(context_inputs, context_input_visual_features)
            context_output_features = self.grid_encoder(context_outputs, context_output_visual_features)
            # §15.2-A: enrich the SUPPORT demo features with their relational maps BEFORE C2 attention,
            # so demo->target matching can use object/inside/distance facts. Zero-init proj => no-op at
            # step 0. Inline fallback computes the support maps if the dataloader did not supply them.
            if getattr(self, "c2_demo_relmap_proj", None) is not None:
                if context_rel_maps is None or context_output_rel_maps is None:
                    from models.recursive_reasoning.object_bank import relational_maps as _rm
                    _B, _M, _L = context_inputs.shape
                    _side = int(math.isqrt(_L))
                    if context_rel_maps is None:
                        context_rel_maps = _rm(context_inputs.reshape(_B * _M, _L), side=_side).view(_B, _M, _L, -1)
                    if context_output_rel_maps is None:
                        context_output_rel_maps = _rm(context_outputs.reshape(_B * _M, _L), side=_side).view(_B, _M, _L, -1)
                context_input_features = context_input_features + self.c2_demo_relmap_proj(
                    context_rel_maps.to(context_input_features.dtype))
                context_output_features = context_output_features + self.c2_demo_relmap_proj(
                    context_output_rel_maps.to(context_output_features.dtype))
            grid_features, c2_metrics, pid_task_vec = self.c2(
                target_features=grid_features,
                context_inputs=context_inputs,
                context_outputs=context_outputs,
                context_input_features=context_input_features,
                context_output_features=context_output_features,
                context_mask=context_mask,
            )
        # §15.2-B: PairDelta as an INPUT-ONLY hint -- broadcast the learned cross-demo rule_vec to all
        # cells and add (zero-init proj => no-op at step 0). Pure input evidence; the TRM recurrence
        # fuses it. No output writer, no logit residual (the demoted, V3-safe role).
        if (getattr(self, "pairdelta_input_encoder", None) is not None
                and context_inputs is not None and context_outputs is not None and context_mask is not None):
            _pd = self.pairdelta_input_encoder(context_inputs, context_outputs, context_mask)
            _rule = self.delta_rule_input_proj(_pd["rule_vec"].to(grid_features.dtype))   # [B, hidden]
            _amp = float(getattr(self, "_demo_injection_scale", 1.0))                     # --zh-amp diagnostic
            grid_features = grid_features + _amp * _rule.unsqueeze(1)                     # broadcast [B,1,hidden]
            with torch.no_grad():
                c2_metrics = dict(c2_metrics)
                c2_metrics["c2_pairdelta_input_norm"] = _rule.float().norm(dim=-1).mean().detach()
        if (getattr(self.config, "c2_pairdelta_intent_hint", False)
                and context_inputs is not None and context_outputs is not None and context_mask is not None):
            from models.recursive_reasoning.pair_delta_encoder import pairdelta_intent_features
            _intent = pairdelta_intent_features(context_inputs, context_outputs, context_mask)
            input_hints["pairdelta_intent"] = _intent["feature"].to(grid_features.dtype)  # [B,1]
            with torch.no_grad():
                c2_metrics = dict(c2_metrics)
                c2_metrics["c2_pairdelta_conditional_score"] = (
                    _intent["conditional_recolor_score"].float().mean().detach()
                )
                c2_metrics["c2_pairdelta_global_score"] = (
                    _intent["global_recolor_score"].float().mean().detach()
                )
                c2_metrics["c2_pairdelta_shape_preserved"] = (
                    _intent["shape_preserved"].float().mean().detach()
                )
                c2_metrics["c2_pairdelta_changed_rate"] = (
                    _intent["changed_rate"].float().mean().detach()
                )
        if (
            self.visual_rule_adapter is not None
            and context_inputs is not None
            and context_outputs is not None
            and context_mask is not None
        ):
            grid_features, visual_rule_metrics = self.visual_rule_adapter(
                base_features=grid_features,
                target_inputs=target_inputs,
                context_inputs=context_inputs,
                context_outputs=context_outputs,
                context_mask=context_mask,
            )
            c2_metrics = dict(c2_metrics)
            c2_metrics.update(visual_rule_metrics)
        if getattr(self.config, "c2_relmap", False) and target_inputs is not None:
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
                from models.recursive_reasoning.object_bank import relational_maps as _compute_relational_maps
                _side = int(math.isqrt(target_inputs.shape[-1]))
                rel_maps = _compute_relational_maps(target_inputs, side=_side)
            rel_maps = rel_maps.to(grid_features.dtype)
            grid_features = grid_features + self.relmap_proj(rel_maps)
            if (getattr(self.config, "c2_rel_where_hint", False)
                    and context_inputs is not None and context_outputs is not None and context_mask is not None):
                from models.recursive_reasoning.object_bank import relational_where_hint
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
                        _where_info["rel_where_confidence"].float().mean().detach()
                    )
                    c2_metrics["c2_rel_where_f1"] = (
                        _where_info["rel_where_f1"].float().mean().detach()
                    )
                    c2_metrics["c2_rel_where_fpr"] = (
                        _where_info["rel_where_fpr"].float().mean().detach()
                    )

        # Lane B: broadcast-add the FRAME-family hint embedding (zero at init -> F7-safe). Independent of
        # c2_relmap so the rule-hypothesis bus can be A/B'd on its own.
        if getattr(self.config, "c2_frame_hint", False) and frame_label is not None and hasattr(self, "frame_embed"):
            fe = self.frame_embed(frame_label.to(torch.long)).to(grid_features.dtype)     # [B, hidden]
            _amp = float(getattr(self, "_demo_injection_scale", 1.0))                     # --zh-amp diagnostic
            grid_features = grid_features + _amp * fe.unsqueeze(1)                        # broadcast over cells
            with torch.no_grad():
                c2_metrics = dict(c2_metrics)
                c2_metrics["c2_frame_hint_norm"] = fe.float().norm(dim=-1).mean().detach()
                c2_metrics["c2_frame_hint_nonzero_frac"] = (frame_label != 0).float().mean().detach()

        # In-model rule-hypothesis hint: infer the TOP operation-family from the (LODO-correct) support
        # pairs and broadcast-add its zero-init embedding. context_inputs is [B, M, L] and is already the
        # held-out support on the aux path (the caller passes the LODO variant), so there is no target
        # leak. Inference is CPU/python (non-differentiable) under no_grad; only the embedding is learned.
        if (getattr(self.config, "c2_rule_hypothesis_hint", False) and hasattr(self, "rule_hyp_embed")
                and context_inputs is not None and context_outputs is not None):
            from models.recursive_reasoning.object_rule_bank import infer_rule_hypotheses
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
            _amp = float(getattr(self, "_demo_injection_scale", 1.0))                     # --zh-amp diagnostic
            grid_features = grid_features + _amp * rh.unsqueeze(1)                        # broadcast over cells
            with torch.no_grad():
                c2_metrics = dict(c2_metrics)
                c2_metrics["c2_rule_hyp_norm"] = rh.float().norm(dim=-1).mean().detach()
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
        # rel_maps is RETURNED, never written into `batch`: the batch dict IS the ACT carry's
        # current_data, so mutating it leaks keys into the carry -- on the inline-fallback path
        # (dataloader emitted no rel_maps) the next step's key-merge then KeyErrors.
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

    def _build_lodo_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor] | None:
        if (
            not self.training
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

        max_samples = int(self.config.c2_lodo_max_samples)
        if max_samples > 0:
            valid_indices = torch.nonzero(aux_valid, as_tuple=False).flatten()
            if valid_indices.numel() > max_samples:
                selected = valid_indices[torch.randperm(valid_indices.numel(), device=valid_indices.device)[:max_samples]]
                limited_valid = torch.zeros_like(aux_valid)
                limited_valid[selected] = True
                aux_valid = limited_valid

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
        # (tokens, mask, relmaps, visual). A single source makes it IMPOSSIBLE to pair wrong-task demo
        # tokens with correct-task side-channels (the old per-tensor ternaries keyed the relmap/visual
        # gathers on c2_lodo_contrast_weight while the build keyed on force_shuffle too -> mismatch).
        # Rows without a wrong-task candidate keep shuffle_valid=False and are masked by the loss.
        shuffle_src = row_indices
        shuffle_valid = torch.zeros(row_indices.shape[0], device=aux_valid.device, dtype=torch.bool)
        shuffle_context_mask = aux_context_mask[row_indices].to(batch["context_mask"].dtype)
        if (self.config.c2_lodo_contrast_weight > 0
                or getattr(self.config, "c2_lodo_force_shuffle", False)) and row_indices.numel() > 0:
            source_puzzle_identifiers = batch["puzzle_identifiers"]
            source_has_context = context_mask.any(dim=-1)
            wrong_candidates = (
                source_puzzle_identifiers.unsqueeze(0) != aux_puzzle_identifiers.unsqueeze(1)
            ) & source_has_context.unsqueeze(0)
            shuffle_valid = wrong_candidates.any(dim=-1)
            if shuffle_valid.any():
                wrong_scores = torch.rand(wrong_candidates.shape, device=wrong_candidates.device)
                wrong_scores = wrong_scores.masked_fill(~wrong_candidates, -1)
                shuffle_src = wrong_scores.argmax(dim=-1)
                shuffle_context_mask = batch["context_mask"][shuffle_src]

        def _rows(key: str) -> torch.Tensor:      # correct-task per-demo tensor for the LODO rows
            return batch[key][row_indices]

        def _shuf(key: str) -> torch.Tensor:      # the SAME tensor from the shuffle source rows
            return batch[key][shuffle_src]

        out = {
            "inputs": aux_inputs[row_indices],
            "labels": aux_labels[row_indices],
            # main-batch indices kept for LODO (examples with >=2 demos). Lets the loss head
            # align r_full (main, all examples) to r_loo (this subset) for the NCE/cons losses.
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
        # Demo OUTPUT relmaps + frame label ride along whenever the dataloader emits them. The aux
        # forward previously dropped these -> LODO trained a DIFFERENT input contract than MAIN
        # (c2_relmap_demos / c2_frame_hint were silent no-ops on the cross-demo trainer).
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
        # The aux forward must see the SAME input contract as the main forward (_input_embeddings) or
        # LODO trains a different input distribution than it deploys. Key-replace derives the LODO
        # ("context_*") and SHUFFLE ("shuffle_context_*") variants from one pattern instead of a second
        # hand-maintained kwarg list (the bug class: context_rel_maps/frame_label used to be dropped).
        def _aux_key(base_key: str, feature: str) -> torch.Tensor | None:
            prefix = "context_inputs" if base_key == context_inputs_key else "context_outputs"
            return aux_batch.get(base_key.replace(prefix, feature))

        grid_features, _, pid_task_vec, rel_maps, aux_input_hints = self._condition_grid_features(
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
        if (
            getattr(self.config, "c2_floor_candidate_split", False)
            and "c2_candidate_logits" in extra_outputs
        ):
            logits = extra_outputs["c2_candidate_logits"]
        if return_extras:
            return logits, extra_outputs
        return logits

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
            changed = (x >= 2) & (y >= 2) & (x != y) & demo_ok.unsqueeze(-1)
            pair = ((x - 2).clamp(0, 9) * 10 + (y - 2).clamp(0, 9)).reshape(batch_size, -1)
            counts = torch.zeros((batch_size, 100), device=device, dtype=torch.float32)
            counts.scatter_add_(1, pair, changed.reshape(batch_size, -1).to(torch.float32))
            cond = counts.view(batch_size, 10, 10)
            cond = cond / cond.sum(dim=-1, keepdim=True).clamp_min(1.0)          # zero rows stay zero
            tin = (ti.long() - 2).clamp(0, 9)
            hint = cond.gather(1, tin.unsqueeze(-1).expand(-1, -1, 10))          # [B, L, 10]
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
        from models.recursive_reasoning.object_bank import cell_conditioning_signature
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
        if bool(getattr(self.config, "c2_value_v2_rich_ctx", False)):
            from models.recursive_reasoning.object_bank import cell_conditioning_signature
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
            return features, zero_stats
        ci = batch.get("_active_context_inputs", batch.get("context_inputs"))
        co = batch.get("_active_context_outputs", batch.get("context_outputs"))
        ti = batch.get("inputs")
        if ci is None or co is None or ti is None or ti.ndim != 2:
            return features, zero_stats
        with torch.no_grad():
            x = ci.long().to(device)
            y = co.long().to(device)
            target = ti.long().to(device)
            cm = batch.get("_active_context_mask", batch.get("context_mask"))
            demo_ok = cm.to(device=device, dtype=torch.bool) if cm is not None else torch.ones(
                x.shape[:2], dtype=torch.bool, device=device)
            valid = (x >= 2) & (y >= 2) & demo_ok.unsqueeze(-1)
            changed = valid & (x != y)
            copied = valid & (x == y)
            src = (x - 2).clamp(0, 9)
            dst = (y - 2).clamp(0, 9)

            # Marginal support by source colour.
            src_flat = src.reshape(batch_size, -1)
            total_by_src = torch.zeros((batch_size, 10), device=device, dtype=torch.float32)
            changed_by_src = torch.zeros_like(total_by_src)
            copy_by_src = torch.zeros_like(total_by_src)
            total_by_src.scatter_add_(1, src_flat, valid.reshape(batch_size, -1).float())
            changed_by_src.scatter_add_(1, src_flat, changed.reshape(batch_size, -1).float())
            copy_by_src.scatter_add_(1, src_flat, copied.reshape(batch_size, -1).float())

            pair_key = (src * 10 + dst).reshape(batch_size, -1)
            changed_counts = torch.zeros((batch_size, 100), device=device, dtype=torch.float32)
            changed_counts.scatter_add_(1, pair_key, changed.reshape(batch_size, -1).float())
            marginal_dist = changed_counts.view(batch_size, 10, 10)
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
        return features, stats

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
        """[B, L, 140+REL_MAP_CHANNELS] PID-free per-cell evidence for the quarantined candidate head.

        Layout (must match the warm-init column indices in __init__): input one-hot [0:12],
        8-neighbour one-hots [12:108], transition hint (MARGINAL P(out|src)) [108:118],
        rel-where [118:119], palette [119:129], intent [129:130],
        CONDITIONED transition (FIX B: P(out|src,ctx) w/ backoff, _value_evidence_v2[...,10:20])
        [130:140], relmap [140:]. Every part derives from the target INPUT tokens or the support
        demos -- puzzle_identifiers never enter this tensor, so the head cannot memorise, and its
        train (blank-PID LODO aux) and deploy (PID-ful main) features are identical. Missing
        evidence (flag off / no demos) degrades to zeros, never to a different layout.
        """
        _q_dim = 140 + REL_MAP_CHANNELS
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
                # FIX B: self-compute the conditioned dist when the caller has no V2 features
                # (quarantine can run without c2_value_evidence_v2). LODO-safe: _value_evidence_v2
                # reads _active_context_* itself. PID never enters -> quarantine property preserved.
                _v2, _ = self._value_evidence_v2(
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
            feats = torch.cat((
                onehot, neigh, transition_hint.to(dtype), where,
                task_palette[:, None, :].expand(-1, seq_len, -1).to(dtype),
                intent, conditioned_hint.to(dtype), rmap), dim=-1)
        assert feats.shape[-1] == _q_dim, f"quarantine feature layout drifted: {feats.shape[-1]} != {_q_dim}"
        return feats

    def _apply_task_palette_bias(
        self,
        color_logits: torch.Tensor,
        palette_mask: torch.Tensor,
    ) -> torch.Tensor:
        disallowed = ~palette_mask[:, None, :]
        if getattr(self.config, "c2_task_palette_hard", False):
            return color_logits.masked_fill(disallowed, torch.finfo(color_logits.dtype).min)
        strength = float(getattr(self.config, "c2_task_palette_strength", 4.0))
        if strength <= 0:
            return color_logits
        return color_logits - strength * disallowed.to(color_logits.dtype)

    def _predicted_extent(self, batch: Dict[str, torch.Tensor] | None, shape_logits=None):
        """(h, w, conf) each [B]: predicted OUTPUT extent in CELLS + confidence in [0, 1].

        Supersedes the old same-shape gate (identity is now just one of the rules). Fits out_hw = f(in_hw)
        over the SUPPORT demos from an ORDERED set of general size maps {identity, constant, integer-ratio},
        VERIFIES that f reconstructs EVERY valid demo's out_hw exactly, then applies the first verified
        f to the TEST INPUT extent. Support-safe: fits on demo (in, out) pairs plus the KNOWN test INPUT;
        the test OUTPUT is never read.

        NON-VACUOUS verification: identity carries no free parameter (input-derived) -> 1 demo suffices;
        constant/ratio FIT their parameter from the demos, so a single demo reconstructs itself by
        construction -- "verified" on 1 demo is vacuous and asserted the WRONG box on 2-demo tasks under
        LODO holdout (measured; hides in v2pad, not pad). They require >= 2 valid demos.

        conf: 1.0 for a demo-verified rule; c2_extent_shape_head_conf (< 1, default 0.5) for the optional
        LEARNED shape-head fallback (softmax margin > tau on BOTH axes) -- a learned guess is not a proof,
        so the near-hard override fires at reduced strength; 0.0 otherwise (lever stays at floor).
        """
        if batch is None:
            return None
        ci = batch.get("_active_context_inputs", batch.get("context_inputs"))
        co = batch.get("_active_context_outputs", batch.get("context_outputs"))
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
        cm = batch.get("_active_context_mask", batch.get("context_mask"))
        valid = cm.to(torch.bool) if cm is not None else torch.ones(in_hw.shape[:2], dtype=torch.bool, device=dev)
        vexp = valid.unsqueeze(-1)                                                     # [B,D,1]
        n_valid = valid.sum(dim=1)                                                     # [B]
        idx = torch.arange(bsz, device=dev)
        first_valid = torch.argmax(valid.int(), dim=1)                                # [B]

        h = ti_hw[:, 0].clone()
        w = ti_hw[:, 1].clone()
        conf = torch.zeros(bsz, device=dev)

        def _consider(pred_demo: torch.Tensor, pred_test: torch.Tensor, min_demos: int) -> None:
            # verified iff f reconstructs EVERY valid demo (both axes) AND enough demos to be non-vacuous
            ok = ((pred_demo == out_hw) | ~vexp).all(dim=1).all(dim=-1) & (n_valid >= min_demos)
            take = ok & (conf < 0.5)
            h[take] = pred_test[take, 0]
            w[take] = pred_test[take, 1]
            conf[take] = 1.0

        # (1) identity: out == in (parameter-free -> safe from a single demo)
        _consider(in_hw, ti_hw, min_demos=1)
        # (2) constant: all valid demos share one out_hw (fitted -> needs >= 2 demos)
        const = out_hw[idx, first_valid]                                              # [B,2]
        _consider(const.unsqueeze(1).expand_as(out_hw), const, min_demos=2)
        # (3) integer ratio: out == k * in per axis (fitted -> needs >= 2 demos; guards in==0)
        safe_in = in_hw.clamp_min(1)
        kk = (out_hw[idx, first_valid] / safe_in[idx, first_valid]).round()           # [B,2]
        _consider((safe_in * kk.unsqueeze(1)).round(), (ti_hw.clamp_min(1) * kk).round(), min_demos=2)

        # (4) optional learned shape-head fallback for the still-unverified rows (default off)
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

    def _output_logits(
        self,
        z_H: torch.Tensor,
        batch: Dict[str, torch.Tensor] | None = None,
        rel_maps: torch.Tensor | None = None,
        input_hints: Dict[str, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # input_hints (rel_where / pairdelta_intent) are THREADED from _condition_grid_features so this
        # forward's colour head reads the hint computed from ITS context (main vs aux-LODO vs shuffle),
        # with no instance-attr stash -> torch.compile-safe.
        input_hints = input_hints or {}
        grid_z = z_H[:, self.puzzle_emb_len:]
        if not self.config.c2_dual_output_head:
            base_logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
            extras: Dict[str, torch.Tensor] = {}
            if self.config.c2_geometry_aux_head:
                structure_logits = self.structure_head(grid_z)
                extras["c2_structure_logits"] = structure_logits
                extras["c2_geometry_aux_active"] = torch.as_tensor(
                    1.0,
                    device=z_H.device,
                    dtype=torch.float32,
                )
                alpha = float(self.config.c2_structure_fusion_alpha)
                if alpha != 0.0:
                    assert base_logits.shape[:-1] == structure_logits.shape[:-1], (
                        f"Logit spatial mismatch: lm={tuple(base_logits.shape)}, "
                        f"structure={tuple(structure_logits.shape)}"
                    )
                    assert base_logits.shape[-1] >= 3, (
                        f"Expected PAD/EOS/colour vocabulary, got {base_logits.shape[-1]}"
                    )
                    assert structure_logits.shape[-1] == 3, (
                        f"Expected PAD/EOS/VALID logits, got {structure_logits.shape[-1]}"
                    )

                    valid_ref = structure_logits[..., 2:3].to(base_logits.dtype)
                    structure_bias = torch.zeros_like(base_logits)
                    structure_bias[..., 0:1] = structure_logits[..., 0:1].to(base_logits.dtype) - valid_ref
                    structure_bias[..., 1:2] = structure_logits[..., 1:2].to(base_logits.dtype) - valid_ref
                    base_logits = base_logits + alpha * structure_bias

                    extras["c2_structure_fusion_alpha"] = torch.as_tensor(
                        alpha,
                        device=z_H.device,
                        dtype=torch.float32,
                    )
                    extras["c2_structure_fusion_bias_abs_mean"] = (
                        structure_bias.float().abs().mean().detach()
                    )
            if getattr(self.config, "c2_delta_expose_base_logits", False):
                # P_off = logits BEFORE the (removed) writers. Stop-grad target for the Stage-2 KL.
                self._last_pre_delta_logits = base_logits.detach()
            return base_logits, extras

        # rel_maps comes from the caller (forward / _run_aux_logits thread it through); the batch key
        # is only a fallback for external debug callers whose batch came straight from the dataloader.
        batch_rel_maps = rel_maps if rel_maps is not None else (
            batch.get("rel_maps") if batch is not None else None)
        floor_logits = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        if getattr(self.config, "c2_delta_expose_base_logits", False):
            # V3 dual-output mode replaces lm_head with factored structure/color logits,
            # but Stage-2 preservation and selector scoring still need a stable P_off target.
            # Use the lm_head floor as the stop-grad reference for the LODO aux path.
            self._last_pre_delta_logits = floor_logits.detach()
        use_palette_feature = bool(getattr(self.config, "c2_task_palette_feature", False))
        use_palette_bias = bool(getattr(self.config, "c2_task_palette_bias", False))
        use_rel_where_hint = bool(getattr(self.config, "c2_rel_where_hint", False))
        use_pairdelta_intent_hint = bool(getattr(self.config, "c2_pairdelta_intent_hint", False))
        task_palette = None
        if use_palette_feature or use_palette_bias:
            task_palette = self._task_palette_mask(batch, grid_z.shape[0], grid_z.device)

        # FIX A: evidence features are collected SEPARATELY from grid_z and read by the dedicated
        # zero-init color_evidence_proj (own optimizer group), not by widened color_head columns
        # (those were welded to the core lr and provably inert -- see the __init__ comment).
        evidence_parts = []
        if getattr(self.config, "c2_relmap", False):
            if batch_rel_maps is not None:
                evidence_parts.append(batch_rel_maps.to(grid_z.dtype))
            else:
                # Unreachable in the wired paths (_input_embeddings/_run_aux_logits both stash
                # batch["rel_maps"]); width MUST track REL_MAP_CHANNELS or the concat breaks the proj.
                evidence_parts.append(torch.zeros(
                    (*grid_z.shape[:2], REL_MAP_CHANNELS), device=grid_z.device, dtype=grid_z.dtype))
        if use_palette_feature:
            assert task_palette is not None
            evidence_parts.append(
                task_palette[:, None, :].expand(-1, grid_z.shape[1], -1).to(grid_z.dtype)
            )
        rel_where_hint = None
        if use_rel_where_hint:
            rel_where_hint = input_hints.get("rel_where")
            if rel_where_hint is None:
                _wk = max(1, int(getattr(self.config, "c2_rel_where_topk", 1)))
                rel_where_hint = torch.zeros((*grid_z.shape[:2], _wk), device=grid_z.device, dtype=grid_z.dtype)
            else:
                rel_where_hint = rel_where_hint.to(grid_z.dtype)
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
        if getattr(self.config, "c2_transition_hint", False):
            transition_hint = self._transition_hint(batch, grid_z.shape[0], grid_z.shape[1], grid_z.device)
            evidence_parts.append(transition_hint.to(grid_z.dtype))
        value_v2 = None
        value_v2_stats: Dict[str, torch.Tensor] = {}
        value_v2_logits = None
        if getattr(self.config, "c2_value_evidence_v2", False):
            value_v2, value_v2_stats = self._value_evidence_v2(
                batch,
                grid_z.shape[0],
                grid_z.shape[1],
                grid_z.device,
                rel_maps=batch_rel_maps,
                rel_where_hint=rel_where_hint,
            )
            # value_v2's OWN evidence columns start at this offset within color_evidence_proj (= width
            # of all evidence_parts before it). Slicing by explicit offset (not -VALUE_EVIDENCE_V2_DIM:)
            # makes the aux CE correct regardless of append order.
            v2_col_offset = sum(int(p.shape[-1]) for p in evidence_parts)
            evidence_parts.append(value_v2.to(grid_z.dtype))
            value_v2_logits = F.linear(
                value_v2.to(self.color_evidence_proj.weight.dtype),
                self.color_evidence_proj.weight[:, v2_col_offset:v2_col_offset + VALUE_EVIDENCE_V2_DIM],
            ).to(grid_z.dtype)
        algo_maps = None
        if getattr(self.config, "c2_algo_where_maps", False):
            # FIX C: appended LAST so all earlier color_evidence_proj columns keep their positions
            # when the flag is toggled (checkpoint warm-start copies min(cols) prefix-aligned).
            algo_maps = self._algo_where_maps(batch, grid_z.shape[0], grid_z.shape[1], grid_z.device)
            evidence_parts.append(algo_maps.to(grid_z.dtype))
        color_logits = self.color_head(grid_z)
        if evidence_parts:
            evidence_features = torch.cat(evidence_parts, dim=-1)
            assert evidence_features.shape[-1] == self.color_evidence_dim, (
                f"evidence width drifted: {evidence_features.shape[-1]} != {self.color_evidence_dim} "
                f"(flag set changed after __init__?)")
            color_logits = color_logits + self.color_evidence_proj(evidence_features)
        if getattr(self, "color_head_mlp_in", None) is not None:
            # zero-init residual: adds input-colour x evidence interaction capacity, step-0 no-op.
            # Reads the FULL concat [grid_z | evidence] (matches color_head_mlp_in width).
            color_features = torch.cat([grid_z] + evidence_parts, dim=-1) if evidence_parts else grid_z
            color_logits = color_logits + self.color_head_mlp_out(F.silu(self.color_head_mlp_in(color_features)))
        if use_palette_bias:
            assert task_palette is not None
            color_logits = self._apply_task_palette_bias(color_logits, task_palette)
        if bool(getattr(self.config, "c2_structure_from_lmhead", False)):
            # §15.8 STRUCTURE-FROM-LM_HEAD (the factored pad/shape fix). Derive PAD/EOS/VALID from the
            # 518K-trained lm_head itself: the validity logit is the logsumexp over the 10 colour logits,
            # so log_softmax([pad, eos, logsumexp(colour)]) reproduces the floor's PAD/EOS/VALID partition
            # EXACTLY (logsumexp([a,b,logsumexp([c...])]) == logsumexp([a,b,c...])). The factored head then
            # inherits lm_head's pad/eos/shape by construction instead of relearning structure in a fresh
            # 3-way head. The colour CHOICE inside VALID is still the relmap-reading color_head.
            fl = floor_logits.to(torch.float32)
            structure_logits = torch.cat(
                (fl[..., 0:1], fl[..., 1:2], torch.logsumexp(fl[..., 2:12], dim=-1, keepdim=True)),
                dim=-1,
            )
        else:
            structure_logits = self.structure_head(grid_z)
        # §15.6: structure logits read the relmap too (zero-init proj => no-op at init). valid_mask/distance
        # /boundary lock PAD-vs-VALID, robust to the colour head's confidence -> fixes the factored pad/shape.
        if getattr(self, "structure_relmap_proj", None) is not None and batch_rel_maps is not None:
            structure_logits = structure_logits + self.structure_relmap_proj(batch_rel_maps.to(structure_logits.dtype))
        # §15.9.1 GENERALIZED extent levers. PAD == cells outside the predicted OUTPUT box, EOS == its
        # thin-L border; both masks come from ONE _predicted_extent call (verified demo size-rule,
        # support-safe) + the shared _extent_box_geometry (offset off the INPUT bbox -- the tokenizer pads
        # input+output with one (pad_r,pad_c)). Same-shape is the identity rule (mask == (input==PAD)
        # exactly); size-change is the constant/ratio rule -> ONE mechanism, no per-regime gate. conf
        # scales the near-hard override (1 = demo-verified; <1 = learned fallback; 0 = floor untouched).
        # Verified IoU=100%/eos-leak=0 on the 518K aux by scripts/verify_outside_grid_lever.py.
        _outside_ssc_mean = None
        _eos_grid_conf_mean = None
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
                        _outside_ssc_mean = conf.float().mean().detach()
                    if _eos_on:
                        eos_mask = extent_eos_mask(tgt_in, h_pred, w_pred, side)     # [B,L] thin-L EOS
                        structure_logits = structure_logits + self.structure_eos_proj(
                            (eos_mask * conf_col).to(structure_logits.dtype).unsqueeze(-1))
                        _eos_grid_conf_mean = conf.float().mean().detach()
        structure_logp = F.log_softmax(structure_logits.to(torch.float32), dim=-1)
        color_logp = F.log_softmax(color_logits.to(torch.float32), dim=-1)
        split = bool(getattr(self.config, "c2_floor_candidate_split", False))
        # PID-QUARANTINED candidate colour source: substitute the z_H-reading color_head with the
        # quarantine head for the CANDIDATE lane only (MAIN stays the floor via `split`). The head
        # reads exclusively PID-free evidence (_quarantine_features), so the candidate's colour
        # CHOICE cannot memorise and is identical between the blank-PID LODO train path and the
        # PID-ful main scoring path. (The floor-height anchor below still backprops into lm_head,
        # but under --train-scope quarantine only quarantine_* params are stepped; the head itself
        # never reads z_H, so its LEARNED function is PID-free by construction, not by optimizer.)
        q_logits = None
        cand_color_logp = color_logp
        if bool(getattr(self.config, "c2_quarantine_candidate", False)) and split:
            q_feats = self._quarantine_features(
                batch, grid_z.shape[0], grid_z.shape[1], grid_z.device, grid_z.dtype,
                rel_maps=batch_rel_maps, transition_hint=transition_hint,
                rel_where_hint=rel_where_hint, pairdelta_intent_hint=pairdelta_intent_hint,
                task_palette=task_palette,
                # FIX B: reuse the V2 conditioned dist when already computed this forward
                conditioned_hint=(value_v2[..., 10:20] if value_v2 is not None else None))
            q_logits = self.quarantine_lin(q_feats) + self.quarantine_mlp_out(
                F.silu(self.quarantine_mlp_in(q_feats)))
            cand_color_logp = F.log_softmax(q_logits.to(torch.float32), dim=-1)
        factored_candidate_logits = torch.cat(
            (
                structure_logp[..., 0:1],
                structure_logp[..., 1:2],
                structure_logp[..., 2:3] + cand_color_logp,
            ),
            dim=-1,
        ).to(color_logits.dtype)
        if bool(getattr(self.config, "c2_candidate_floor_structure", False)):
            # Structure channels come from the LEVER-CORRECTED structure_logits, NOT the raw floor.
            # Under structure-from-lmhead these ARE the floor's pad/eos logits (same scale) PLUS the
            # relmap proj and the +-1000*conf extent PAD/EOS overrides. Reading floor_logits[...,0:2]
            # here bypassed the levers -> on the blank-pid LODO forward the raw lm_head colours over
            # the padding (measured gap mean~177) and LODO pad collapsed 97.5% -> 1% (run A' step 0).
            # NOTE: assumes structure_from_lmhead (floor logit scale); with --fresh-structure-head the
            # scales are incoherent and this hybrid should not be used.
            struct_pad_eos = structure_logits[..., 0:2].to(color_logits.dtype)
            floor_color = floor_logits[..., 2:12].to(color_logits.dtype)
            floor_is_valid = floor_color.amax(dim=-1, keepdim=True) > struct_pad_eos.amax(dim=-1, keepdim=True)
            # Preserve the floor's colour-channel HEIGHT on valid cells: the V3 colour distribution
            # only redistributes mass among the 10 colours (the head learns choice, not scale).
            # cand_color_logp = color_head, or the quarantine head when c2_quarantine_candidate.
            color_delta = cand_color_logp - cand_color_logp.amax(dim=-1, keepdim=True)
            candidate_color = (floor_color.amax(dim=-1, keepdim=True) + color_delta).to(color_logits.dtype)
            hybrid_color = torch.where(floor_is_valid, candidate_color, floor_color)
            candidate_logits = torch.cat((struct_pad_eos, hybrid_color), dim=-1)
        else:
            candidate_logits = factored_candidate_logits
        logits = floor_logits.to(color_logits.dtype) if split else candidate_logits
        extras = {
            "c2_color_logits": color_logits,
            "c2_structure_logits": structure_logits,
            "c2_dual_output_active": torch.as_tensor(
                1.0,
                device=z_H.device,
                dtype=torch.float32,
            ),
        }
        if _outside_ssc_mean is not None:
            extras["c2_outside_grid_extent_conf"] = _outside_ssc_mean
        if _eos_grid_conf_mean is not None:
            extras["c2_eos_grid_extent_conf"] = _eos_grid_conf_mean
        if task_palette is not None:
            extras.update(
                {
                    "c2_task_palette_mask": task_palette,
                    "c2_task_palette_allowed_frac": task_palette.float().mean().detach(),
                    "c2_task_palette_allowed_count": task_palette.float().sum(dim=-1).mean().detach(),
                }
            )
        if rel_where_hint is not None:
            extras.update(
                {
                    "c2_rel_where_hint": rel_where_hint.detach(),
                    "c2_rel_where_hint_mean": rel_where_hint.float().mean().detach(),
                }
            )
        if algo_maps is not None:
            # FIX 5: expose the computed masks so the loss can report coverage on changed vs copy
            # cells (attributes a null result: dead key vs uncovered cells).
            extras["c2_algo_where_maps"] = algo_maps.detach()
        if pairdelta_intent_hint is not None:
            extras.update(
                {
                    "c2_pairdelta_intent_hint": pairdelta_intent_hint.detach(),
                    "c2_pairdelta_intent_hint_mean": pairdelta_intent_hint.float().mean().detach(),
                }
            )
        if transition_hint is not None:
            # coverage = fraction of colour cells with ANY transition evidence; a low value on a
            # recolor family means the consensus is not reaching the head (extraction problem).
            _colour_cells = (batch["inputs"] >= 2) if (batch is not None and "inputs" in batch) else None
            _mass = transition_hint.sum(-1)
            extras["c2_transition_hint_coverage"] = (
                (_mass[_colour_cells] > 0).float().mean().detach()
                if _colour_cells is not None and bool(_colour_cells.any()) else _mass.mean().detach()
            )
        if value_v2 is not None:
            extras.update({k: v.detach() for k, v in value_v2_stats.items()})
            if value_v2_logits is not None:
                extras["c2_value_v2_logits"] = value_v2_logits
        if q_logits is not None:
            # The quarantine head's raw colour logits: bit-identical under any PID perturbation
            # (the regression test's quarantine property) and between the LODO and main forwards.
            extras["c2_quarantine_logits"] = q_logits
        if split:
            extras.update(
                {
                    "c2_candidate_logits": candidate_logits,
                    "c2_factored_candidate_logits": factored_candidate_logits,
                    "c2_floor_logits": floor_logits.detach(),
                    "c2_main_uses_floor": torch.as_tensor(
                        1.0,
                        device=z_H.device,
                        dtype=torch.float32,
                    ),
                    "c2_candidate_floor_structure": torch.as_tensor(
                        1.0 if getattr(self.config, "c2_candidate_floor_structure", False) else 0.0,
                        device=z_H.device,
                        dtype=torch.float32,
                    ),
                }
            )
        return logits, extras

    def _shape_logits(self, z_H: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_z = z_H[:, self.puzzle_emb_len:]
        puzzle_z = z_H[:, 0]
        if self.config.c2_shape_pool == "zH_puzzle_inputvalid_gridmean":
            valid_mask = (batch["inputs"] >= 2).unsqueeze(-1).to(grid_z.dtype)
            grid_pool = (grid_z * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1)
            shape_state = torch.cat((puzzle_z, grid_pool), dim=-1)
        elif self.config.c2_shape_pool == "zH_rowcol":
            # DIRECTIONAL pool (H/W-separable). per-cell activation magnitude -> per-row / per-col
            # occupancy profiles, so the head can read WHERE content starts/ends along each axis.
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
        # Stage-0 exposure: capture P_off NOW, before the shuffle forward (below) overwrites the
        # per-forward stash.
        base_logits_lodo = getattr(self, "_last_pre_delta_logits", None)
        outputs = {
            "c2_aux_logits": aux_logits,
            "c2_aux_labels": aux_batch["labels"],
            # held-out demo INPUT — lets the delta-LODO loss mark CHANGED cells (input!=target)
            # for the two-region (inside/outside) balanced CE that trains the factored branch.
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
        }
        if base_logits_lodo is not None:
            outputs["c2_aux_base_logits"] = base_logits_lodo   # P_off for the Stage-2 KL
        if "c2_value_v2_logits" in aux_extra_outputs:
            outputs["c2_aux_value_v2_logits"] = aux_extra_outputs["c2_value_v2_logits"]
        if self.config.c2_lodo_contrast_weight > 0 or getattr(self.config, "c2_lodo_force_shuffle", False):
            shuffle_logits = self._run_aux_logits(
                aux_batch,
                seq_info,
                context_inputs_key="shuffle_context_inputs",
                context_outputs_key="shuffle_context_outputs",
                context_mask_key="shuffle_context_mask",
            )
            outputs.update(
                {
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
                }
            )
        return outputs

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_ACTV1InnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)
        # Stage-0 exposure: clear the stale stash so we never capture a previous batch's floor.
        if getattr(self.config, "c2_delta_expose_base_logits", False):
            self._last_pre_delta_logits = None
        input_embeddings, c2_metrics, rel_maps, input_hints = self._input_embeddings(batch)

        z_H, z_L = self._run_recurrence(carry, input_embeddings, seq_info)

        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output, output_extras = self._output_logits(z_H, batch, rel_maps=rel_maps, input_hints=input_hints)
        if self.config.c2_shape_head:
            h_logits, w_logits = self._shape_logits(z_H, batch)
            output_extras.update(
                {
                    "c2_shape_h_logits": h_logits,
                    "c2_shape_w_logits": w_logits,
                }
            )
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        aux_outputs = self._aux_outputs(batch, seq_info)
        merged = {**c2_metrics, **output_extras, **aux_outputs}
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), merged


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
