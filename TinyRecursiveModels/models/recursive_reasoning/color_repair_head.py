"""The single COLOUR brain: a coherent extract -> value -> where -> output flow.

Whole-session diagnosis (why this shape):
  * The colour RULE is extractable and reliable as a deterministic consensus map:
    ColorTransitionBank.cond_inout[a] = P(out|in=a) over the OTHER demos, with src_agree[a]
    = cross-demo agreement. This is the trustworthy "WHAT colour does a become".
  * A colour-conditioned prior cannot decide WHICH `a`-cells change on a *conditional* recolour
    -- that is positional. The earlier head tried to relearn the colour from scratch (zero-init)
    and FAILED (changed-colour worse than the frozen base, repair loss flat at random).

So this head STOPS relearning the colour and instead COMPOSES the pieces, in order:

  EXTRACT  cond_inout[a], src_agree[a]                      (given; deterministic, no-grad)
  VALUE    repair = softplus(scale)*log cond_inout[a] + residual    <- WHAT colour (prior + small learned fix)
  WHERE    det = src_agree[a]*peak  on CONFIDENT RECOLOUR maps (argmax(cond)!=a)
           learned = sigmoid(local 3x3 gate)               <- conditional recolours
           gg = is_color * max(det, learned)
  PRESERVE copy cells (argmax(cond)==a) get det=0 -> left to the base (KL guards them) -> no seesaw
  OUTPUT   new_color = (1-gg)*base + gg*repair             on colour channels [2:12] only

Consequences:
  * Full recolours + copies are solved at INIT with NO training (det fires / leaves base alone).
  * The learned gate + residual only have to handle conditional recolours.
  * PAD/EOS are never produced here (colour channels only); is_color gates non-colour input cells off.

Token convention (repo-wide): PAD=0, EOS=1, colour = token-2 (0..9), grid 30x30=900.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from models.recursive_reasoning.object_bank import OBJ_DIM, object_features, size_bucket  # noqa: E402

VOCAB = 12          # PAD, EOS, 10 colours
COLOR_OFFSET = 2
N_COLORS = 10
LOCAL_DIM = VOCAB * 9 + 2   # 3x3 one-hots (108) + same-colour-count + edge flag
# Phase 2a: deterministic positional / object-proxy features for the WHERE-gate. The cell-local 3x3
# gate cannot tell a changed cell from a copy cell (run A: GATE changed approx GATE copy) because a
# 3x3 patch is blind to object/positional structure. These give it: norm row/col, this-colour global
# frequency, is-this-the-most-common-non-bg-colour, same-colour blob size (5x5 proxy), 3x3 colour
# heterogeneity. Cheap (a couple of grouped convs); fed to the gate so it can localise the recolour.
POS_DIM = 6
# COPY-SAFE softening: blend 0.9*identity + 0.1*changed-map instead of a HARD one-hot. A hard one-hot
# makes the VALUE degenerate {0,1}; scale*log(.) then drives off-class logits into the -30 clamp floor
# (P~e^-60), and that razor-sharp distribution produced the non-finite gradient the moment the gate
# opened on a mis-protected cell (copy-safe run NaN @ step 156). The soft floor keeps argmax==a (copy
# preserved) while leaving bounded off-class mass -> tame gradients.
SOFT_COPY_ALPHA = 0.9


def demo_recolour_consistency(context_inputs: torch.Tensor, context_outputs: torch.Tensor,
                              context_mask: torch.Tensor | None = None) -> torch.Tensor:
    """[B,M,L] demos -> [B] router score = self-consistency of the pooled cell-colour CHANGED map:
    fraction of all changed cells the single best (modal) out-colour per in-colour explains. High =>
    a plain per-cell recolour fits the task (CLEAN, head fires); low => conditional/relational/
    structural (head should shut). The validated demo-only router signal. 0 if no changed cells."""
    B, _M, _L = context_inputs.shape
    x, y = context_inputs.long(), context_outputs.long()
    valid = (x >= COLOR_OFFSET) & (y >= COLOR_OFFSET) & (x != y)
    if context_mask is not None:
        valid = valid & context_mask.bool().unsqueeze(-1)
    a = (x - COLOR_OFFSET).clamp(0, 9)
    d = (y - COLOR_OFFSET).clamp(0, 9)
    flat = (a * N_COLORS + d).clamp(0, N_COLORS * N_COLORS - 1)                 # [B,M,L]
    cooc = torch.zeros(B, N_COLORS * N_COLORS, device=context_inputs.device)
    cooc.scatter_add_(1, flat.reshape(B, -1), valid.reshape(B, -1).float())
    cooc = cooc.view(B, N_COLORS, N_COLORS)                                     # cooc[b,a,d]
    total = cooc.sum(dim=(1, 2))                                               # all changed cells
    explained = cooc.max(dim=-1).values.sum(dim=-1)                            # sum_a max_d cooc[b,a,d]
    return explained / total.clamp_min(1.0)                                    # [B] in [0,1]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE pairing on the last dim: (x0,x1,x2,x3,...) -> (-x1,x0,-x3,x2,...)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


def _build_2d_rope(side: int, head_dim: int, theta: float = 10000.0):
    """2D axial RoPE (VARC / EVA-02 VisionRotaryEmbeddingFast, ADAPTED to a standalone 30x30 head with
    NO prefix token). cos/sin [side*side, head_dim]: the first head_dim/2 dims encode ROW position, the
    last head_dim/2 encode COL -> attention becomes 2D-RELATIVE (a cell attends 'up'/'left'/'to the
    touching object' position-generally, transferring across grid sizes). This is the ONE perception
    piece we take from VARC; our TRM trunk's 1D rope over a row-major flatten cannot express it."""
    assert head_dim % 4 == 0, f"head_dim must be divisible by 4 for 2D rope (got {head_dim})"
    dh = head_dim // 2                                                  # dims per axis (row, col)
    freqs = 1.0 / (theta ** (torch.arange(0, dh, 2).float() / dh))      # [dh/2]
    pos = torch.arange(side).float()                                   # [side]
    ang = torch.outer(pos, freqs).repeat_interleave(2, dim=-1)         # [side, dh] (pair-shared angle)
    cos_axis, sin_axis = ang.cos(), ang.sin()                          # [side, dh]
    idx = torch.arange(side * side)
    rows, cols = idx // side, idx % side
    cos = torch.cat([cos_axis[rows], cos_axis[cols]], dim=-1)          # [L, head_dim]
    sin = torch.cat([sin_axis[rows], sin_axis[cols]], dim=-1)
    return cos, sin


class CopyByRelationHead(nn.Module):
    """VARC-inspired COPY-BY-RELATION value source -- the relational organ the 10x10 lookup lacks
    (R23: deterministic relational candidates are NET-NEGATIVE at 32% precision; only a LEARNED,
    position-general head that GENERALISES unseen relations can beat that). Each TARGET cell attends
    (2D-rope) over the INPUT colour cells and copies the attended colour. The attention VALUE is the
    input colour ONE-HOT, so the output is a distribution over colours PRESENT in the grid -- PALETTE-
    SAFE by construction (cannot invent; matches the 97.4% no-invention finding). Returns out_scale *
    log p_copy with out_scale ZERO-INIT -> EXACT no-op at init (warm-start safe). Trains through the
    existing repair/LODO CE on repair_color, so it fine-tunes alongside the rest of TRM."""

    def __init__(self, hidden_dim: int, grid_side: int = 30, attn_dim: int = 64, n_heads: int = 2):
        super().__init__()
        self.side = grid_side
        self.n_heads = n_heads
        self.head_dim = attn_dim // n_heads
        assert self.head_dim % 4 == 0, "attn_dim // n_heads must be divisible by 4 (2D rope)"
        self.q_proj = nn.Linear(hidden_dim, attn_dim)
        self.color_embed = nn.Embedding(VOCAB, attn_dim)
        self.k_proj = nn.Linear(attn_dim, attn_dim)
        self.out_scale = nn.Parameter(torch.zeros(1))                  # ZERO-INIT -> no-op at init
        cos, sin = _build_2d_rope(grid_side, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, grid_z: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        """grid_z [B,L,H] (query) + input_tokens [B,L] (keys & values) -> [B,L,10] relational VALUE."""
        B, L, _ = grid_z.shape
        dt = self.q_proj.weight.dtype
        nh, hd = self.n_heads, self.head_dim
        toks = input_tokens.long().clamp(0, VOCAB - 1)
        q = self.q_proj(grid_z.to(dt)).view(B, L, nh, hd).transpose(1, 2)            # [B,nh,L,hd]
        k = self.k_proj(self.color_embed(toks).to(dt)).view(B, L, nh, hd).transpose(1, 2)
        cos = self.rope_cos.to(dt).view(1, 1, L, hd)
        sin = self.rope_sin.to(dt).view(1, 1, L, hd)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        attn = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)                    # [B,nh,L,L]
        is_color = (input_tokens.long() >= COLOR_OFFSET).view(B, 1, 1, L)            # valid keys only
        attn = attn.masked_fill(~is_color, torch.finfo(attn.dtype).min)             # finite min -> all-masked = uniform, no NaN
        attn = attn.softmax(dim=-1)
        vcol = (toks - COLOR_OFFSET).clamp(0, N_COLORS - 1)                          # [B,L]
        v = F.one_hot(vcol, N_COLORS).to(dt).unsqueeze(1).expand(B, nh, L, N_COLORS)
        pcopy = torch.matmul(attn, v).mean(dim=1)                                    # [B,L,10] prob dist over present colours
        return self.out_scale.to(dt) * torch.log(pcopy.clamp_min(1e-4))             # log p_copy, 0 at init


class ColorRepairHead(nn.Module):
    def __init__(self, hidden_dim: int, grid_side: int = 30, mlp_dim: int = 256,
                 prior_scale_init: float = 4.0, rule_vec_dim: int | None = None,
                 gate_positional: bool = False, gate_object: bool = False,
                 router_gate: bool = False, palette_constrain: bool = False,
                 router_threshold: float = 0.9, router_band: float = 0.2,
                 palette_strength: float = 4.0, value_copy_safe: bool = False,
                 demote_lookup: bool = False, copy_relation: bool = False):
        super().__init__()
        self.side = grid_side
        self.gate_positional = bool(gate_positional)
        self.gate_object = bool(gate_object)
        # Phase 3 ROUTER: gate the WHOLE head by a per-task score r in [0,1] = self-consistency of the
        # demos' cell-colour map. Clean recolours (high r) -> head FIRES; relational/structural tasks
        # (low r, where a per-cell map provably fails) -> head SHUT, output = base. A clamped linear
        # ramp (1 at/above router_threshold, 0 at/below threshold-band) so CLEAN tasks are NOT damped.
        self.router_gate = bool(router_gate)
        self.router_threshold = float(router_threshold)
        self.router_band = float(router_band)
        # Phase 3 PALETTE: penalise output colours NOT present in the input grid (the demos show 97%
        # "no invention" on relational tasks). Strength = (1 - router_mult) so it ONLY bites on
        # relational tasks (clean recolours INVENT new colours, 20% in-palette -> must stay free).
        self.palette_constrain = bool(palette_constrain)
        # GENTLE bias (NOT a hard mask): a hard -30 drove out-of-palette probs to ~0, and the
        # preserve-KL vs base then hit log(0) -> NaN the moment the gate opened (1500-run aborted
        # ~step 1260). A small penalty nudges argmax toward in-palette where the VALUE is uncertain
        # (relational: prior suppressed, residual ~0) without ever zeroing a probability.
        self.palette_strength = float(palette_strength)
        # COPY-SAFE VALUE: --repair-changed-value makes the VALUE "what colour WHEN it changes", which
        # is WRONG on copy cells -> a gate that leaks open on a copy DESTROYS it (the unchg cap). When
        # on, cells the identity-aware prior calls a COPY (argmax P(out|in=a)==a) get an identity VALUE,
        # so a leaked-open gate keeps the colour. Changed cells still use the changed-only target.
        self.value_copy_safe = bool(value_copy_safe)
        # S5 DEMOTE LOOKUP: weight scale*log(cond) by the lookup CONFIDENCE (peak of the value map)
        # so AMBIGUOUS/conditional maps are suppressed and the z_H relational residual decides --
        # the only way the VALUE can exceed the per-cell-colour ceiling (dpcc) on conditional tasks.
        self.demote_lookup = bool(demote_lookup)
        # feat = grid_z | base(10) | cond_cell(10) | src_agree(1) | cond_peak(1) | local(110) [| pos(6)] [| obj(4)]
        feat_in = (hidden_dim + N_COLORS + N_COLORS + 1 + 1 + LOCAL_DIM
                   + (POS_DIM if gate_positional else 0)
                   + (OBJ_DIM if gate_object else 0))
        self.mlp = nn.Sequential(
            nn.Linear(feat_in, mlp_dim), nn.GELU(),
            nn.Linear(mlp_dim, mlp_dim), nn.GELU(),
        )
        self.gate_head = nn.Linear(mlp_dim, 1)
        self.color_head = nn.Linear(mlp_dim, N_COLORS)   # LEARNED RESIDUAL on top of the prior
        # B1 (Rule Bus): OPTIONAL learned cross-demo rule (PairDeltaEncoder.rule_vec) -> hidden,
        # ADDED into the shared feature h so it lifts BOTH the WHERE-gate and the VALUE-residual.
        # zero-init => true no-op at init (so turning it on never disturbs a warm-start); learns to
        # carry the positional/conditional recolours the deterministic prior caps out on (~37%).
        # None => head is exactly the prior-only head (the alternative path is preserved).
        self.rule_proj = nn.Linear(rule_vec_dim, mlp_dim) if rule_vec_dim else None
        # COPY-BY-RELATION (VARC 2D-rope): a relational VALUE source -- each cell attends over the
        # input cells and copies a related cell's colour (palette-safe). out_scale=0 -> no-op at init.
        self.copy_relation = CopyByRelationHead(hidden_dim, grid_side) if copy_relation else None
        # softplus(prior_scale) multiplies log P(out|in=a): sharpens a peaked map into a near-hard
        # choice while staying differentiable. Learnable so KL/CE can temper it.
        self.prior_scale = nn.Parameter(torch.tensor(float(prior_scale_init)))
        with torch.no_grad():
            self.gate_head.bias.fill_(-6.0)     # gate starts ~shut (sigmoid~0.0025) -> init is a
                                                # TRUE no-op on MAIN; opens only as it learns.
            self.color_head.weight.zero_()      # residual = 0 at init -> VALUE is the pure prior
            self.color_head.bias.zero_()
            if self.rule_proj is not None:
                self.rule_proj.weight.zero_()   # rule contribution = 0 at init (no-op warm-start)
                self.rule_proj.bias.zero_()

    def local_features(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """[B, L] integer tokens -> [B, L, LOCAL_DIM] 3x3 context features."""
        B, L = input_tokens.shape
        g = input_tokens.long().clamp(0, VOCAB - 1).view(B, self.side, self.side)
        oh = F.one_hot(g, VOCAB).permute(0, 3, 1, 2).float()        # [B, VOCAB, S, S]
        patches = F.unfold(oh, kernel_size=3, padding=1)            # [B, VOCAB*9, L]
        patches = patches.transpose(1, 2).contiguous()             # [B, L, VOCAB*9]
        pv = patches.view(B, L, VOCAB, 9)                          # channel-major, kernel inner
        center_oh = pv[:, :, :, 4]                                 # [B, L, VOCAB] (centre = the cell)
        neigh = torch.cat([pv[..., :4], pv[..., 5:]], dim=-1)      # [B, L, VOCAB, 8] (8 neighbours)
        same = (neigh * center_oh.unsqueeze(-1)).sum(dim=(2, 3)).unsqueeze(-1)   # same-colour count [B,L,1]
        pad_eos = neigh[:, :, 0:2, :].sum(dim=(2, 3)).unsqueeze(-1)              # PAD/EOS neighbours
        edge = (pad_eos > 0).float()                              # touches boundary [B,L,1]
        return torch.cat([patches, same, edge], dim=-1)           # [B, L, LOCAL_DIM]

    def positional_features(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """[B,L] tokens -> [B,L,POS_DIM] deterministic positional / object-proxy features (Phase 2a).
        Gives the WHERE-gate the structure a 3x3 patch can't see: position, colour rarity, blob size."""
        B, L = input_tokens.shape
        S = self.side
        dev = input_tokens.device
        g = input_tokens.long().clamp(0, VOCAB - 1).view(B, S, S)
        ohc = F.one_hot(g, VOCAB).permute(0, 3, 1, 2).float()          # [B,VOCAB,S,S]
        # 1-2: normalised row / col
        rr = (torch.arange(S, device=dev).float() / max(S - 1, 1)).view(1, S, 1).expand(B, S, S)
        cc = (torch.arange(S, device=dev).float() / max(S - 1, 1)).view(1, 1, S).expand(B, S, S)
        # 3: this cell's colour global frequency (fraction of grid with the same colour)
        color_count = ohc.sum(dim=(2, 3))                             # [B,VOCAB]
        color_freq = (ohc * color_count.view(B, VOCAB, 1, 1)).sum(1) / float(L)   # [B,S,S]
        # 4: is this the MOST COMMON non-background (non PAD/EOS) colour?
        cc_counts = color_count.clone()
        cc_counts[:, 0:2] = -1.0
        most_common = cc_counts.argmax(-1)                            # [B]
        is_most_common = (g == most_common.view(B, 1, 1)).float()      # [B,S,S]
        # 5: same-colour blob size proxy = same-colour count in a 5x5 window (normalised)
        w5 = torch.ones(VOCAB, 1, 5, 5, device=dev)
        same5 = F.conv2d(ohc, w5, padding=2, groups=VOCAB)            # [B,VOCAB,S,S]
        blob5 = (same5 * ohc).sum(1) / 25.0                          # [B,S,S]
        # 6: 3x3 colour heterogeneity = fraction of distinct colours present in the 3x3 neighbourhood
        w3 = torch.ones(VOCAB, 1, 3, 3, device=dev)
        present3 = (F.conv2d((ohc > 0).float(), w3, padding=1, groups=VOCAB) > 0).float().sum(1) / VOCAB
        feats = torch.stack([rr, cc, color_freq, is_most_common, blob5, present3], dim=-1)  # [B,S,S,6]
        return feats.view(B, L, POS_DIM)

    def forward(self, grid_z: torch.Tensor, base_color_logits: torch.Tensor,
                input_tokens: torch.Tensor, cond_inout: torch.Tensor,
                src_agree: torch.Tensor | None, rule_vec: torch.Tensor | None = None,
                cond_value: torch.Tensor | None = None,
                cond_value_obj: torch.Tensor | None = None,
                router_score: torch.Tensor | None = None):
        """grid_z [B,L,H]; base_color_logits [B,L,10]; input_tokens [B,L];
        cond_inout [B,10,10] (P(out|in=a), identity-aware -> gate FEATURES); src_agree [B,10] or None;
        rule_vec [B,D] optional learned cross-demo rule (B1); ignored unless rule_proj built;
        cond_value [B,10,10] optional CHANGED-ONLY map (0.3) -> the VALUE prior; None => use cond_inout.

        Returns:
          new_color   [B,L,10]  blended colour logits, ready to slot into channels [2:12]
          gate_logits [B,L]     learned gate (for the sparse repair-gate BCE)
          repair_color[B,L,10]  the VALUE logits (prior+residual) BEFORE gating (for repair-colour CE)
          applied     [B,L]     gg, the per-cell application strength (diagnostics)
        """
        B, L, _H = grid_z.shape
        pdt = self.mlp[0].weight.dtype
        a = (input_tokens.long() - COLOR_OFFSET).clamp(0, 9)                    # [B,L] input colour idx
        is_color = (input_tokens.long() >= COLOR_OFFSET).to(pdt)               # [B,L] colour-input cells

        # Phase 3 ROUTER multiplier [B,1,1]: clamped linear ramp of the per-task self-consistency score.
        # 1 (head fires) at/above router_threshold; 0 (head shut) at/below threshold-band. Off-flag or
        # no score -> 1.0 (identical to before). At init the gate is shut anyway, so this stays a no-op.
        if self.router_gate and router_score is not None:
            lo = self.router_threshold - self.router_band
            router_mult = ((router_score.to(pdt) - lo) / max(self.router_band, 1e-6)).clamp(0.0, 1.0)
            router_mult = router_mult.view(B, 1, 1)
        else:
            router_mult = torch.ones(B, 1, 1, device=grid_z.device, dtype=pdt)

        cond = cond_inout.float()                                              # [B,10,10] identity-aware
        cond_cell = cond.gather(1, a.unsqueeze(-1).expand(-1, -1, N_COLORS))   # [B,L,10] P(out|in=a)
        cond_peak = cond_cell.max(dim=-1).values                              # [B,L] map confidence (feature)
        # 0.3: the VALUE (what colour a becomes WHEN it changes) uses the CHANGED-ONLY map when
        # given; identity-aware cond_inout biases changed cells toward copy (HEAD changed 27 < dpcc
        # 52). The gate FEATURES below keep identity-aware cond_cell (it helps decide change-vs-copy).
        if cond_value_obj is not None:
            # Phase 2c: OBJECT-CONDITIONED VALUE -- look up P(out | in=a, size-bucket(cell)). This is
            # the only path that lets the VALUE exceed the cell-colour ceiling (dpcc): it can say
            # "a -> b in a big shape, a -> c as a scattered pixel". bucket from the target grid.
            bucket = size_bucket(input_tokens, self.side)                       # [B,L] in {0,1,2}
            K = cond_value_obj.shape[1]
            obj_flat = cond_value_obj.float().reshape(B, K * N_COLORS, N_COLORS)  # [B,K*10,10]
            vidx = (bucket * N_COLORS + a).clamp(0, K * N_COLORS - 1)           # [B,L]
            value_cond_cell = torch.gather(obj_flat, 1, vidx.unsqueeze(-1).expand(-1, -1, N_COLORS))
        else:
            value_cond = cond_value.float() if cond_value is not None else cond
            value_cond_cell = value_cond.gather(1, a.unsqueeze(-1).expand(-1, -1, N_COLORS))  # [B,L,10]
        if self.value_copy_safe:
            # Only protect CONFIDENT copies: argmax P(out|in=a)==a AND P(a->a) high. A blunt override
            # (argmax only) also forced identity on CONDITIONAL cells (colours that usually copy but
            # sometimes change) -> HEAD changed fell 57->43. The confidence gate leaves those to the
            # gate + changed-only VALUE, protecting only colours that almost never change.
            copy_conf = cond_cell.gather(-1, a.unsqueeze(-1)).squeeze(-1)       # P(a->a) per cell
            prior_copy = (cond_cell.argmax(dim=-1) == a) & (copy_conf > 0.7)    # [B,L] confident copy
            # SOFT identity (NaN-safety, see SOFT_COPY_ALPHA): mass on a, but a bounded floor on the
            # other classes so the downstream log/scale never produces a degenerate -30 cliff.
            identity = F.one_hot(a, N_COLORS).to(value_cond_cell.dtype)        # [B,L,10]
            soft_identity = SOFT_COPY_ALPHA * identity + (1.0 - SOFT_COPY_ALPHA) * value_cond_cell
            value_cond_cell = torch.where(prior_copy.unsqueeze(-1), soft_identity, value_cond_cell)
        if src_agree is not None:
            agree_cell = src_agree.float().gather(1, a)                        # [B,L]
        else:
            agree_cell = torch.ones(B, L, device=grid_z.device)

        # ---- WHERE (learned positional gate over LOCAL context) ----
        local = self.local_features(input_tokens).to(pdt)                      # [B,L,LOCAL_DIM]
        feat_parts = [
            grid_z.to(pdt),
            base_color_logits.to(pdt),
            cond_cell.to(pdt),
            agree_cell.unsqueeze(-1).to(pdt),
            cond_peak.unsqueeze(-1).to(pdt),
            local,
        ]
        if self.gate_positional:
            # Phase 2a: positional / object-proxy features so the gate can localise WHERE (the cell
            # cannot be told changed-vs-copy from a 3x3 patch alone).
            feat_parts.append(self.positional_features(input_tokens).to(pdt))   # [B,L,POS_DIM]
        if self.gate_object:
            # Phase 2b: TRUE connected-component object features (real size / is-largest / singleton
            # / boundary) -- the structure the 5x5 proxy only approximated.
            feat_parts.append(object_features(input_tokens, self.side).to(pdt))  # [B,L,OBJ_DIM]
        feat = torch.cat(feat_parts, dim=-1)
        h = self.mlp(feat)
        if self.rule_proj is not None and rule_vec is not None:
            # B1: add the learned cross-demo rule into the shared feature (broadcast over cells).
            # zero-init rule_proj => no-op at init; feeds BOTH gate (WHERE) and residual (VALUE).
            h = h + self.rule_proj(rule_vec.to(pdt)).unsqueeze(1)              # [B,1,mlp]->[B,L,mlp]
        gate_logits = self.gate_head(h).squeeze(-1)                            # [B,L]
        residual = self.color_head(h)                                          # [B,L,10] (zero-init)

        # ---- VALUE (deterministic consensus prior + learned residual) ----
        # 0.1 NUMERICAL BOUNDS: clamp prior strength (softplus<=8) + raise cond floor 1e-6->1e-3 so
        # |scale*log_cond| stays <~ 8*6.9, and a final clamp(-30,30) caps the output. Unbounded
        # scale*log(1e-6) blew to -inf -> bf16 NaN the moment the gate opened.
        # 0.3: log_cond uses the CHANGED-ONLY value_cond_cell (the right "what colour" prior).
        scale = F.softplus(self.prior_scale).clamp(max=8.0).to(pdt)
        log_cond = torch.log(value_cond_cell.clamp_min(1e-3)).to(pdt)          # [B,L,10]
        if self.demote_lookup:
            # S5: down-weight the lookup by its CONFIDENCE (peak of P(out|in=a)). A deterministic
            # map (peak~1) keeps full strength; an ambiguous/conditional map (peak low) is suppressed
            # so the z_H relational residual decides. zero-init residual => at init this only RESCALES
            # an already-confident prior (safe no-op on clean recolours; opens room on conditional).
            value_peak = value_cond_cell.max(dim=-1, keepdim=True).values.to(pdt)   # [B,L,1] in [0,1]
            repair_color = scale * value_peak * log_cond + residual           # WHAT colour, pre-gate
        else:
            repair_color = scale * log_cond + residual                        # WHAT colour, pre-gate
        if self.copy_relation is not None:
            # R23/VARC: ADD the relational copy-by-relation VALUE (2D-rope attention over input cells).
            # out_scale zero-init => exact no-op at init; the LODO/repair CE grows it where copying a
            # related cell's colour beats the lookup. Palette-safe (value = input colour one-hot).
            repair_color = repair_color + self.copy_relation(grid_z, input_tokens).to(pdt)
        if self.palette_constrain:
            # Phase 3: penalise out-of-INPUT-palette colours (no-invention prior) with strength
            # (1 - router_mult) -> active only on relational tasks; CLEAN tasks (router_mult~1) free.
            oh = F.one_hot(a, N_COLORS).to(pdt)                                # [B,L,10] input colour 1-hot
            present = ((oh * is_color.unsqueeze(-1)).sum(dim=1) > 0).to(pdt)    # [B,10] colour in input grid
            strength = (1.0 - router_mult)                                     # [B,1,1] strong on relational
            repair_color = repair_color - strength * self.palette_strength * (1.0 - present).unsqueeze(1)
        repair_color = repair_color.clamp(-30.0, 30.0)

        # ---- WHERE: the LEARNED gate decides, period. ----
        # A deterministic override (apply the consensus map wherever it looks confident) was tried
        # and REMOVED: even peak>0.9 maps don't transfer safely to conditional tasks, so it
        # corrupted already-correct MAIN cells (the colour seesaw). The gate starts ~shut (bias -6),
        # so init is a true no-op on MAIN; the LODO repair losses teach it WHERE to open.
        gg = (torch.sigmoid(gate_logits) * is_color).unsqueeze(-1)            # [B,L,1]
        gg = gg * router_mult                                                  # Phase 3: task-level router shut

        base = base_color_logits.to(pdt)
        new_color = (1 - gg) * base + gg * repair_color                        # [B,L,10]
        return new_color, gate_logits, repair_color, gg.squeeze(-1)


def _self_test() -> None:
    torch.manual_seed(0)
    B, S, H = 2, 30, 64
    L = S * S
    head = ColorRepairHead(hidden_dim=H, grid_side=S)
    inp = torch.full((B, L), 0, dtype=torch.long)
    inp[:, :200] = 3 + COLOR_OFFSET          # a block of colour-3 cells
    inp[:, 200:400] = 5 + COLOR_OFFSET        # a block of colour-5 cells
    grid_z = torch.randn(B, L, H)
    base = torch.randn(B, L, N_COLORS)
    # consensus map: 3->7 ALWAYS (full recolour), 5->5 (copy/identity).
    cond = torch.zeros(B, 10, 10)
    cond[:, 3, 7] = 1.0                        # P(3->7)=1   (recolour)
    cond[:, 5, 5] = 1.0                        # P(5->5)=1   (copy)
    agree = torch.ones(B, 10)

    new_color, gate_logits, repair_color, applied = head(grid_z, base, inp, cond, agree)
    assert new_color.shape == (B, L, N_COLORS), new_color.shape
    assert gate_logits.shape == (B, L), gate_logits.shape
    assert repair_color.shape == (B, L, N_COLORS), repair_color.shape
    base_pred = base.argmax(-1)

    # (1) WARM-START no-op: gate ~shut at init -> head barely touches anything (MAIN preserved).
    assert float(applied.mean()) < 0.01, "init must be a near no-op (warm-start safe)"
    assert (new_color.argmax(-1)[:, 400:] == base_pred[:, 400:]).all(), "non-colour cells keep base"
    assert float(applied[:, 400:].abs().max()) == 0.0, "non-colour cells must get zero gate"

    # (2) VALUE is correct from the prior (independent of the gate): the consensus map says 3->7,
    #     so repair_color (= scale*log cond + residual) argmaxes to 7 on the colour-3 block.
    assert (repair_color.argmax(-1)[:, :200] == 7).all(), "VALUE must point to consensus colour 7"
    assert (repair_color.argmax(-1)[:, 200:400] == 5).all(), "VALUE must keep identity colour 5"

    # (3) when the gate is OPEN, the head applies the VALUE: recolour block -> 7, copy block -> 5.
    with torch.no_grad():
        head.gate_head.bias.fill_(6.0)
    nc_open, _, _, applied_open = head(grid_z, base, inp, cond, agree)
    assert (nc_open.argmax(-1)[:, :200] == 7).all(), "open gate must recolour the 3-block to 7"
    assert (nc_open.argmax(-1)[:, 200:400] == 5).all(), "open gate must keep the 5-block at 5"
    assert float(applied_open[:, :400].mean()) > 0.9, "open gate must fully apply on colour cells"
    with torch.no_grad():
        head.gate_head.bias.fill_(-6.0)    # restore

    # (4) gradient flows to gate + residual + scale.
    loss = new_color.pow(2).mean() + gate_logits.pow(2).mean() + repair_color.pow(2).mean()
    loss.backward()
    assert head.gate_head.weight.grad is not None
    assert head.color_head.weight.grad is not None
    assert head.prior_scale.grad is not None

    # (5) B1 Rule Bus: a head built with rule_vec_dim must STILL be a no-op at init (zero-init
    #     rule_proj) and must route gradient into rule_proj when a rule_vec is supplied.
    head_rv = ColorRepairHead(hidden_dim=H, grid_side=S, rule_vec_dim=32)
    rv = torch.randn(B, 32)
    nc_rv, gl_rv, rc_rv, applied_rv = head_rv(grid_z, base, inp, cond, agree, rule_vec=rv)
    assert float(applied_rv.mean()) < 0.01, "rule_vec head must still be a no-op at init"
    assert (nc_rv.argmax(-1)[:, 400:] == base.argmax(-1)[:, 400:]).all(), "non-colour cells keep base"
    (nc_rv.pow(2).mean() + gl_rv.pow(2).mean() + rc_rv.pow(2).mean()).backward()
    assert head_rv.rule_proj.weight.grad is not None, "gradient must reach rule_proj"

    # (6) Phase 2a: a gate_positional head builds with +POS_DIM features, is STILL a no-op at init
    #     (gate biased shut + residual zero), and routes gradient into the (larger) gate MLP.
    head_pos = ColorRepairHead(hidden_dim=H, grid_side=S, gate_positional=True)
    assert head_pos.mlp[0].in_features == head.mlp[0].in_features + POS_DIM, "positional adds POS_DIM"
    nc_p, gl_p, _, applied_p = head_pos(grid_z, base, inp, cond, agree)
    assert float(applied_p.mean()) < 0.01, "gate_positional head must still be a no-op at init"
    pf = head_pos.positional_features(inp)
    assert pf.shape == (B, L, POS_DIM), pf.shape
    (nc_p.pow(2).mean() + gl_p.pow(2).mean()).backward()
    assert head_pos.mlp[0].weight.grad is not None, "gradient must reach the positional gate MLP"

    # (7) Phase 2b: a gate_object head builds with +OBJ_DIM real connected-component features and is
    #     STILL a no-op at init.
    head_obj = ColorRepairHead(hidden_dim=H, grid_side=S, gate_object=True)
    assert head_obj.mlp[0].in_features == head.mlp[0].in_features + OBJ_DIM, "object adds OBJ_DIM"
    _, _, _, applied_o = head_obj(grid_z, base, inp, cond, agree)
    assert float(applied_o.mean()) < 0.01, "gate_object head must still be a no-op at init"

    # (8) Phase 2c: object-conditioned VALUE -- cond_value_obj [B,K,10,10] keyed on (size-bucket, in).
    #     no-op at init (gate shut); the VALUE follows the per-bucket map; runs end-to-end.
    cond_obj = torch.zeros(B, 3, 10, 10)
    cond_obj[:, 2, 3, 7] = 1.0                                   # large bucket: colour 3 -> 7
    nc_oc, _, rc_oc, applied_oc = head(grid_z, base, inp, cond, agree, cond_value_obj=cond_obj)
    assert float(applied_oc.mean()) < 0.01, "object-conditioned VALUE head must be a no-op at init"
    assert rc_oc.shape == (B, L, N_COLORS)

    # (9) Phase 3 ROUTER: a router_gate head is a no-op at init; with the gate forced OPEN, a LOW
    #     router score SHUTS the head (applied~0) while a HIGH score lets it fire (applied>0.9).
    head_r = ColorRepairHead(hidden_dim=H, grid_side=S, router_gate=True, router_threshold=0.9)
    _, _, _, ap_init = head_r(grid_z, base, inp, cond, agree, router_score=torch.tensor([0.3, 1.0]))
    assert float(ap_init.mean()) < 0.01, "router head must be a no-op at init"
    with torch.no_grad():
        head_r.gate_head.bias.fill_(6.0)
    _, _, _, ap_lo = head_r(grid_z, base, inp, cond, agree, router_score=torch.tensor([0.3, 0.3]))
    _, _, _, ap_hi = head_r(grid_z, base, inp, cond, agree, router_score=torch.tensor([1.0, 1.0]))
    assert float(ap_lo[:, :400].mean()) < 0.01, "low router score must SHUT the head (open gate)"
    assert float(ap_hi[:, :400].mean()) > 0.9, "high router score must let the head FIRE (open gate)"

    # (10) Phase 3 PALETTE (gentle): with palette_constrain + a LOW router score (relational), an
    #      UNCERTAIN VALUE is nudged to the INPUT palette. cond weakly prefers 3->7 (0.6) but 7 is NOT
    #      in the input {3,5}; the gentle penalty tips repair_color to in-palette 3 on the 3-block.
    #      With a HIGH router score (clean) the penalty is OFF, so the invented colour 7 survives.
    cond_w = torch.zeros(B, 10, 10)
    cond_w[:, 3, 7] = 0.6; cond_w[:, 3, 3] = 0.4; cond_w[:, 5, 5] = 1.0       # weak 3->7 prior
    head_pal = ColorRepairHead(hidden_dim=H, grid_side=S, router_gate=True, palette_constrain=True)
    with torch.no_grad():
        head_pal.gate_head.bias.fill_(6.0)
    _, _, rc_lo, _ = head_pal(grid_z, base, inp, cond_w, agree, router_score=torch.tensor([0.0, 0.0]))
    _, _, rc_hi, _ = head_pal(grid_z, base, inp, cond_w, agree, router_score=torch.tensor([1.0, 1.0]))
    assert (rc_lo.argmax(-1)[:, :200] != 7).all(), "palette (relational) must nudge off out-of-palette 7"
    assert (rc_hi.argmax(-1)[:, :200] == 7).all(), "clean task (high router) keeps the invented colour 7"

    # (11) router score helper: a fully-consistent recolour task scores 1.0; a 50/50 conditional ~0.5.
    ci = torch.full((2, 3, L), COLOR_OFFSET, dtype=torch.long)
    co = ci.clone()
    ci[0, :, :100] = 3 + COLOR_OFFSET; co[0, :, :100] = 7 + COLOR_OFFSET           # 3->7 always (clean)
    ci[1, :, :100] = 3 + COLOR_OFFSET                                              # task 1: 3 -> half 7 half 8
    co[1, :, :50] = 7 + COLOR_OFFSET; co[1, :, 50:100] = 8 + COLOR_OFFSET
    rs = demo_recolour_consistency(ci, co)
    assert abs(float(rs[0]) - 1.0) < 1e-4 and abs(float(rs[1]) - 0.5) < 0.05, f"router score {rs.tolist()}"

    # (12) COPY-SAFE VALUE: a changed-only VALUE that would recolour a COPY cell (5->7) is overridden
    #      to identity where the prior says copy (cond_inout[5,5]=1) -> repair keeps 5, not 7.
    cond_changed = torch.zeros(B, 10, 10)
    cond_changed[:, 5, 7] = 1.0; cond_changed[:, 3, 7] = 1.0                  # changed-only: 5->7, 3->7
    head_cs = ColorRepairHead(hidden_dim=H, grid_side=S, value_copy_safe=True)
    with torch.no_grad():
        head_cs.gate_head.bias.fill_(6.0)
    _, _, rc_cs, _ = head_cs(grid_z, base, inp, cond, agree, cond_value=cond_changed)
    head_unsafe = ColorRepairHead(hidden_dim=H, grid_side=S, value_copy_safe=False)
    with torch.no_grad():
        head_unsafe.gate_head.bias.fill_(6.0)
    _, _, rc_un, _ = head_unsafe(grid_z, base, inp, cond, agree, cond_value=cond_changed)
    assert (rc_cs.argmax(-1)[:, 200:400] == 5).all(), "copy-safe must KEEP the copy colour 5"
    assert (rc_un.argmax(-1)[:, 200:400] == 7).all(), "without copy-safe the changed-only VALUE destroys it (5->7)"
    assert (rc_cs.argmax(-1)[:, :200] == 7).all(), "copy-safe must still recolour the changed 3-block to 7"

    # (12b) SOFTENING (NaN-safety): at a HIGH prior_scale the protected copy cell must NOT collapse to
    #       a degenerate -30 cliff. The changed-only class 7 keeps a BOUNDED (non-floor) logit -- this
    #       is what avoids the non-finite gradient when the gate opens on a mis-protected cell. A HARD
    #       one-hot would drive class 7 to the -30 clamp floor (8*log(1e-3) = -55 -> clamp -30).
    with torch.no_grad():
        head_cs.prior_scale.fill_(8.0)
    _, _, rc_cs_hi, _ = head_cs(grid_z, base, inp, cond, agree, cond_value=cond_changed)
    assert (rc_cs_hi.argmax(-1)[:, 200:400] == 5).all(), "copy still preserved at high scale"
    seven_logit = rc_cs_hi[:, 200:400, 7]
    assert float(seven_logit.min()) > -29.0, \
        f"soft identity must keep the changed class off the -30 cliff (got {float(seven_logit.min()):.1f})"

    # (13) COPY-BY-RELATION (VARC 2D-rope): EXACT no-op at init (out_scale=0); with out_scale forced,
    #      the relational VALUE is PALETTE-SAFE (argmax only in colours PRESENT in the input) and
    #      gradient reaches the copy-relation head. The relational organ the 10x10 lookup lacks (R23).
    head_cr = ColorRepairHead(hidden_dim=H, grid_side=S, copy_relation=True)
    cr_noop = float(head_cr.copy_relation(grid_z, inp).abs().max())
    assert cr_noop == 0.0, "copy_relation term must be EXACTLY 0 at init (out_scale=0)"
    _, _, _, ap_cr = head_cr(grid_z, base, inp, cond, agree)
    assert float(ap_cr.mean()) < 0.01, "copy_relation head must be a no-op at init (gate shut)"
    with torch.no_grad():
        head_cr.copy_relation.out_scale.fill_(1.0)
    cr_term = head_cr.copy_relation(grid_z, inp)                                # [B,L,10]
    present = set(((inp[:, :400].reshape(-1) - COLOR_OFFSET).clamp(0, 9)).unique().tolist())
    cr_arg = set(cr_term.argmax(-1)[:, :400].reshape(-1).unique().tolist())
    assert cr_arg.issubset(present), f"copy-relation VALUE must stay in input palette ({cr_arg} vs {present})"
    cr_term.pow(2).mean().backward()
    assert head_cr.copy_relation.q_proj.weight.grad is not None, "gradient must reach copy_relation"

    print(f"color_repair_head self-test PASS  (feat_in={head.mlp[0].in_features}, "
          f"init no-op applied={float(applied.mean()):.4f}, VALUE->7 ok, open-gate recolour ok, "
          f"rule_vec no-op={float(applied_rv.mean()):.4f}, pos no-op={float(applied_p.mean()):.4f}, "
          f"obj no-op={float(applied_o.mean()):.4f}, router shut={float(ap_lo[:, :400].mean()):.3f}/"
          f"fire={float(ap_hi[:, :400].mean()):.3f}, palette+router ok, copy-relation no-op={cr_noop:.1f} "
          f"palette-safe ok, score={rs.tolist()})")


if __name__ == "__main__":
    _self_test()
