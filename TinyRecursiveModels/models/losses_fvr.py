from typing import Any, Dict, Optional, Sequence, Tuple

import math
import torch
import torch.nn.functional as F
from torch import nn

from models.losses import (
    IGNORE_LABEL_ID,
    softmax_cross_entropy,
    stablemax_cross_entropy,
)


def _loss_per_token(loss_type: str, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if loss_type == "stablemax_cross_entropy":
        return stablemax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask)
    if loss_type == "softmax_cross_entropy":
        return softmax_cross_entropy(logits, labels, ignore_index=IGNORE_LABEL_ID)
    raise ValueError(f"Unknown loss_type={loss_type!r}")


def _context_labels_to_loss_labels(labels: torch.Tensor) -> torch.Tensor:
    return torch.where(labels == 0, torch.full_like(labels, IGNORE_LABEL_ID), labels)


def _ce_canvas_regions(logits, target, inputs, changed_w, color_w, pad_w, eos_w):
    """Phase-B TWO-REGION balanced CE from the legacy delta-rule prototype.

    PAD(0)/EOS(1) are SUPERVISED as real classes (only a -100 sentinel is ignored), so the
    model learns the canvas BOUNDARY (not just the coloured sub-region). The output's two
    jobs are graded SEPARATELY as count-normalized MEANS so PAD's cell-count cannot drown the
    few transform cells:
      OUTSIDE = PAD + EOS   (frame / shape)
      INSIDE  = colour>=2   (content; changed cells = the transform, up-weighted)
    Returns (total, L_inside, L_outside) for logging."""
    V = logits.shape[-1]
    t = target.reshape(-1).long()
    keep = t >= 0
    t = t.clamp_min(0)
    ce = F.cross_entropy(logits.reshape(-1, V).float(), t, reduction="none")
    inp = inputs.reshape(-1).long()
    pad = (t == 0) & keep
    eos = (t == 1) & keep
    color = (t >= 2) & keep
    changed = color & (inp != t)
    unchanged = color & (inp == t)

    def cm(m):
        m = m.float()
        return (ce * m).sum() / m.sum().clamp_min(1.0)

    l_out = pad_w * cm(pad) + eos_w * cm(eos)
    l_in = changed_w * cm(changed) + color_w * cm(unchanged)
    return l_in + l_out, l_in, l_out


def _changed_cell_ce(logits, target, inputs, per_row: bool = False):
    """Mean CE over ONLY the transform cells (colour cells where input != target).
    This is the task-specific part of the output — used by the two-region CONTRAST so the
    discriminative pressure (real demos vs shuffled demos) targets exactly the changed-colour
    bottleneck, not the task-invariant PAD/boundary/copy cells.

    per_row=True returns ([B] per-row means, [B] has-changed-cells mask) instead of the batch
    scalar, so the caller can hinge each task separately and MASK rows whose shuffle control is
    invalid (the batch-scalar form dilutes the gap with rows where shuffle == real)."""
    B, V = logits.shape[0], logits.shape[-1]
    t = target.reshape(-1).long()
    keep = t >= 0
    t = t.clamp_min(0)
    ce = F.cross_entropy(logits.reshape(-1, V).float(), t, reduction="none")
    inp = inputs.reshape(-1).long()
    changed = ((t >= 2) & keep & (inp != t)).float()
    if per_row:
        ce = ce.view(B, -1)
        changed = changed.view(B, -1)
        n = changed.sum(-1)
        return (ce * changed).sum(-1) / n.clamp_min(1.0), n > 0
    return (ce * changed).sum() / changed.sum().clamp_min(1.0)


def _value_v2_aux_ce(logits: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor,
                     changed_w: float, copy_w: float):
    """CE on the VALUE-V2-only colour contribution.

    The normal candidate CE can ignore the zero-init V2 tail columns by leaning on the old colour
    logits. This auxiliary term directly trains the V2 evidence columns to map copy cells to the
    input colour and changed cells to the held-out output colour. It is still LODO-only and colour-only:
    PAD/EOS/structure are untouched here.
    """
    t = target.long()
    keep = t >= 2
    inp = inputs.long()
    changed = keep & (inp != t)
    copied = keep & (inp == t)
    color_target = (t - 2).clamp(0, 9)
    ce = F.cross_entropy(logits.reshape(-1, 10).float(), color_target.reshape(-1), reduction="none").view_as(t)

    def cm(mask_: torch.Tensor) -> torch.Tensor:
        return (ce * mask_.float()).sum() / mask_.float().sum().clamp_min(1.0)

    loss_changed = cm(changed)
    loss_copy = cm(copied)
    with torch.no_grad():
        pred = logits.argmax(dim=-1) + 2
        changed_acc = ((pred == t) & changed).float().sum() / changed.float().sum().clamp_min(1.0)
        copy_acc = ((pred == t) & copied).float().sum() / copied.float().sum().clamp_min(1.0)
    return changed_w * loss_changed + copy_w * loss_copy, loss_changed, loss_copy, changed_acc, copy_acc


# (The _repair_gate_bce / _repair_color_ce / _rule_nce_cons helpers were REMOVED 2026-07-02: their
# model-side producers (delta repair branch, rule-vec exposure) died in the delta-branch deletion,
# so the losses could never fire -- the CLI flags were silent no-ops printing NaN metrics.)


def _preserve_kl(off_logits: torch.Tensor, on_logits: torch.Tensor,
                 labels: torch.Tensor, inputs: torch.Tensor):
    """Preservation KL(sg(P_off) || P_on) from the legacy delta-rule prototype.

    P_off = logits BEFORE the delta branch; P_on = final logits. The KL is applied ONLY on
    cells that are UNCHANGED (input==label, the copy region) AND already correct under P_off.
    This stops the rule branch from overwriting what the frozen TRM already solves -> kills the
    shape/content fight WITHOUT handcuffing the rule on the CHANGED cells (those are governed by
    the delta-LODO loss). Returns (L_keep, preserved_correct_frac)."""
    t = labels.long()
    keep = t >= 0
    off = off_logits.float()
    on = on_logits.float()
    off_logp = F.log_softmax(off, dim=-1)
    # NaN-safety: floor on_logp so a near-zero P_on (a very confident-wrong cell the instant the gate
    # opens) cannot make (off_logp - on_logp) blow up -> a finite, bounded KL even against a degenerate
    # distribution. Bites only where P_on < ~1e-13 (the pathological tail we never want gradients from).
    on_logp = F.log_softmax(on, dim=-1).clamp_min(-30.0)
    kl_cell = (off_logp.exp() * (off_logp - on_logp)).sum(-1)        # [B, L] KL per cell
    base_correct = (off.argmax(dim=-1) == t) & keep
    unchanged = (inputs.long() == t) & keep
    preserve_set = base_correct & unchanged
    m = preserve_set.float()
    L_keep = (kl_cell * m).sum() / m.sum().clamp_min(1.0)
    with torch.no_grad():
        on_correct = (on.argmax(dim=-1) == t) & preserve_set
        preserved = on_correct.sum().float() / preserve_set.sum().clamp_min(1).float()
    return L_keep, preserved


def _struct_class(x: torch.Tensor) -> torch.Tensor:
    """token -> structure class: PAD(0) / EOS(1) / VALID(2)."""
    return torch.where(x >= 2, torch.full_like(x, 2),
                       torch.where(x == 1, torch.ones_like(x), torch.zeros_like(x)))


def _health_metrics(labels, main_preds, main_inputs,
                    aux_logits=None, aux_floor_logits=None, aux_labels=None, aux_inputs=None, ct_hist=None,
                    ct_src_agree=None, ct_src_support=None,
                    tpeak_thresh: float = 0.5):
    """Structured training-log panel. Three views + denominators:
      MAIN   - normal model output on this batch (is the base/final output healthy?)
      LODO   - reconstruct a held-out demo from the OTHERS (the real cross-demo rule test)
      CTBANK - explicit colour-rule extraction, SPLIT by transition-peak (clean-recolor rows)
      COUNTS - the denominator behind every percentage (a % over 2 cells != a % over 200)

    The decisive diagnostic is
        ct_application_gap_pct = ct_high_tpeak_dpcc_pct - ct_high_tpeak_lodo_changed_color_pct
      gap HIGH  -> extraction works, APPLICATION broken (rule known, model won't apply it)
      dpcc LOW  -> EXTRACTION broken (consensus rule itself wrong)
      both HIGH -> good.
    All values raw (percent 0..100, counts, entropy); the caller scales by count for logging."""
    out = {}

    def pct(num_mask, den_mask):
        return (num_mask.float().sum() / den_mask.float().sum().clamp_min(1.0)) * 100.0

    # ===================== MAIN (test-pair prediction on this batch) =====================
    t = labels.long()
    keep = t != IGNORE_LABEL_ID
    inp = main_inputs.long()
    cp = main_preds
    corr = cp == t
    color = (t >= 2) & keep
    pad = (t == 0) & keep
    eos = (t == 1) & keep
    changed = color & (inp != t)
    struct_row_ok = ((_struct_class(cp) == _struct_class(t)) | ~keep).all(-1)
    out["main_color_acc_pct"] = pct(corr & color, color)
    out["main_changed_color_acc_pct"] = pct(corr & changed, changed)
    out["main_pad_acc_pct"] = pct(corr & pad, pad)
    out["main_eos_acc_pct"] = pct(corr & eos, eos)
    out["main_shape_exact_pct"] = struct_row_ok.float().mean() * 100.0     # boundary all-right per row
    out["main_strict_exact_pct"] = (corr | ~keep).all(-1).float().mean() * 100.0
    out["n_color_cells"] = color.float().sum()
    out["n_changed_color_cells"] = changed.float().sum()
    out["n_pad_cells"] = pad.float().sum()
    out["n_eos_cells"] = eos.float().sum()

    # ===================== LODO (reconstruct held-out demo from the others) =============
    if aux_logits is not None and aux_labels is not None and aux_inputs is not None:
        ap = aux_logits.argmax(-1)
        at = aux_labels.long()
        ak = at != IGNORE_LABEL_ID
        ain = aux_inputs.long()
        acorr = ap == at
        acolor = (at >= 2) & ak
        apad = (at == 0) & ak
        aeos = (at == 1) & ak
        achanged = acolor & (ain != at)
        aunchanged = acolor & (ain == at)                                # copy cells (input==output)
        astruct_ok = ((_struct_class(ap) == _struct_class(at)) | ~ak).all(-1)
        acontent_ok = (acorr | ~acolor).all(-1)
        arow_strict = (acorr | ~ak).all(-1)
        # MAIN labels ignore PAD. Keep the same task-level exact view for LODO:
        # compare colours and EOS, but do not let untrained 30x30 PAD cells define exactness.
        atask = ak & (at != 0)
        atask_shape_ok = ((_struct_class(ap) == _struct_class(at)) | ~atask).all(-1)
        atask_strict = (acorr | ~atask).all(-1)
        out["lodo_color_acc_pct"] = pct(acorr & acolor, acolor)
        out["lodo_changed_color_acc_pct"] = pct(acorr & achanged, achanged)   # the TRANSFORM cells
        out["lodo_unchanged_color_acc_pct"] = pct(acorr & aunchanged, aunchanged)  # the COPY cells
        out["lodo_pad_acc_pct"] = pct(acorr & apad, apad)
        out["lodo_eos_acc_pct"] = pct(acorr & aeos, aeos)
        out["lodo_color_exact_pct"] = (acorr | ~acolor).all(-1).float().mean() * 100.0
        out["lodo_shape_exact_pct"] = astruct_ok.float().mean() * 100.0
        out["lodo_task_shape_exact_pct"] = atask_shape_ok.float().mean() * 100.0
        out["lodo_task_strict_exact_pct"] = atask_strict.float().mean() * 100.0
        # valid->pad FALSE POSITIVE: of cells that should be VALID colour, fraction predicted as PAD.
        # This is the over-prediction that pad-recall HIDES: a high `pad` with a high `v2pad` means the
        # boundary CE is winning by eating valid cells -> shape_task collapses. Watch v2pad, not pad alone.
        out["lodo_valid_to_pad_pct"] = pct((_struct_class(ap) == 0) & acolor, acolor)
        # valid->eos FALSE POSITIVE: the OTHER way shape_task can be wrong while pad looks fine -- the head
        # marks a valid colour cell as EOS. When pad is low, eos true-cell acc is high, but shape_task is
        # still bad, EOS over-prediction on valid cells is the hidden culprit. v2pad + v2eos = full struct FP.
        out["lodo_valid_to_eos_pct"] = pct((_struct_class(ap) == 1) & acolor, acolor)
        out["lodo_strict_exact_pct"] = arow_strict.float().mean() * 100.0
        # WHERE vs VALUE split (colour failure attribution): WHERE = does the model change the RIGHT
        # cells (F1 of its own predicted-change mask vs the true change mask, colour cells only);
        # VALUE = of the cells it DID change, how often is the new colour correct. Read together with
        # lodo_changed_color_acc_pct (VALUE+WHERE combined on true-changed): WHERE high + VALUE low
        # => binding failure (knows where, wrong colour); WHERE low => selection failure.
        apredchg = (ap != ain) & acolor
        _tp = (apredchg & achanged).float().sum()
        _prec = _tp / apredchg.float().sum().clamp_min(1.0)
        _rec = _tp / achanged.float().sum().clamp_min(1.0)
        out["lodo_where_f1_pct"] = (2.0 * _prec * _rec / (_prec + _rec).clamp_min(1e-12)) * 100.0
        out["lodo_value_on_pred_changed_pct"] = pct(acorr & apredchg, apredchg)
        # COLOUR-CHOICE CE (anchor-free graded trend). The candidate's colour values ride a
        # floor-height anchor (candidate_floor_structure) and fall back to RAW saturated floor
        # colours outside floor_valid, so a full-vocab CE is dominated by that FROZEN floor
        # component (measured: panel chgCE ~43/gap +144 flat while per-family aux CE is 3-12 --
        # the trainable head's <=10-nat contribution is invisible). log_softmax over channels
        # 2:12 cancels any per-cell anchor EXACTLY: these two numbers move iff the colour CHOICE
        # moves -- the graded read that shows progress long before argmax metrics flip. Read the
        # PAIR: changed_ce falling with copy_ce flat = binding; copy_ce rising = the CE-calibration
        # pull eroding the copy basin (the changed:copy weight trade).
        _clogp = aux_logits[..., 2:12].float().log_softmax(-1)
        _cce = -_clogp.gather(-1, (at - 2).clamp(0, 9).unsqueeze(-1)).squeeze(-1)
        out["lodo_changed_color_ce"] = (_cce * achanged.float()).sum() / achanged.float().sum().clamp_min(1.0)
        out["lodo_copy_color_ce"] = (_cce * aunchanged.float()).sum() / aunchanged.float().sum().clamp_min(1.0)
        # PER-EXAMPLE views for the VERIFY-AND-SELECT eval (floor gg=0 vs head gg=on candidates scored
        # per task). [B] tensors -> ignored by the scalar-only panel aggregators (numel!=1), read only
        # by eval_selector. cellsim = graded fraction of colour cells correct (the LODO tiebreak).
        out["lodo_color_exact_per_ex"] = (acorr | ~acolor).all(-1).float()
        out["lodo_strict_exact_per_ex"] = arow_strict.float()
        out["lodo_color_cellsim_per_ex"] = (acorr & acolor).float().sum(-1) / acolor.float().sum(-1).clamp_min(1)
        out["lodo_close_pct"] = ((acorr & ak).sum(-1).float() / ak.sum(-1).clamp_min(1)).mean() * 100.0
        if aux_floor_logits is not None:
            fp = aux_floor_logits.argmax(-1)
            fcorr = fp == at
            frow_strict = (fcorr | ~ak).all(-1).float()
            fcolor_exact = (fcorr | ~acolor).all(-1).float()
            fcolor_sim = (fcorr & acolor).float().sum(-1) / acolor.float().sum(-1).clamp_min(1)
            ccolor_exact = out["lodo_color_exact_per_ex"]
            ccolor_sim = out["lodo_color_cellsim_per_ex"]
            cstrict = out["lodo_strict_exact_per_ex"]
            choose_candidate = (ccolor_exact > fcolor_exact) | (
                (ccolor_exact == fcolor_exact) & (ccolor_sim > fcolor_sim + 1e-6)
            )
            sel_color_exact = torch.where(choose_candidate, ccolor_exact, fcolor_exact)
            sel_color_sim = torch.where(choose_candidate, ccolor_sim, fcolor_sim)
            sel_strict = torch.where(choose_candidate, cstrict, frow_strict)

            out["lodo_floor_color_exact_pct"] = fcolor_exact.mean() * 100.0
            out["lodo_floor_color_cellsim_pct"] = fcolor_sim.mean() * 100.0
            out["lodo_floor_strict_exact_pct"] = frow_strict.mean() * 100.0
            out["lodo_select_color_exact_pct"] = sel_color_exact.mean() * 100.0
            out["lodo_select_color_cellsim_pct"] = sel_color_sim.mean() * 100.0
            out["lodo_select_strict_exact_pct"] = sel_strict.mean() * 100.0
            out["lodo_select_ge_floor_pct"] = (sel_color_exact >= fcolor_exact - 1e-9).float().mean() * 100.0
            out["lodo_candidate_chosen_pct"] = choose_candidate.float().mean() * 100.0
            # PER-EXAMPLE floor views: with these, eval_selector reads floor AND candidate from ONE
            # forward (the dead _force_delta_off two-pass compare returned identical logits twice).
            out["lodo_floor_color_exact_per_ex"] = fcolor_exact
            out["lodo_floor_color_cellsim_per_ex"] = fcolor_sim
            out["lodo_floor_strict_exact_per_ex"] = frow_strict
        # exact-solve BLOCKER attribution (of failing rows): shape-only / colour-only / both
        afail = ~arow_strict
        adn = afail.float().sum().clamp_min(1.0)
        apad_ok = (acorr | ~apad).all(-1)
        aeos_ok = (acorr | ~aeos).all(-1)
        out["lodo_block_shape_pct"] = (afail & ~astruct_ok).float().sum() / adn * 100.0
        out["lodo_block_color_pct"] = (afail & ~acontent_ok).float().sum() / adn * 100.0
        out["lodo_block_pad_pct"] = (afail & ~apad_ok).float().sum() / adn * 100.0
        out["lodo_block_eos_pct"] = (afail & ~aeos_ok).float().sum() / adn * 100.0
        out["lodo_block_shape_only_pct"] = (afail & acontent_ok & ~astruct_ok).float().sum() / adn * 100.0
        out["lodo_block_color_only_pct"] = (afail & astruct_ok & ~acontent_ok).float().sum() / adn * 100.0
        out["lodo_block_both_pct"] = (afail & ~astruct_ok & ~acontent_ok).float().sum() / adn * 100.0
        out["n_lodo_color_cells"] = acolor.float().sum()
        out["n_lodo_changed_cells"] = achanged.float().sum()
        out["n_lodo_unchanged_cells"] = aunchanged.float().sum()
        out["n_lodo_pad_cells"] = apad.float().sum()
        out["n_lodo_eos_cells"] = aeos.float().sum()
        out["n_lodo_rows"] = torch.as_tensor(float(at.shape[0]), device=at.device)

        # ===================== CTBANK (explicit colour rule, tpeak-split) ===============
        if ct_hist is not None:
            B = ct_hist.shape[0]
            h = ct_hist.float()
            tpeak_row = h.max(-1).values                              # [B] per-task transition peak
            out["ct_transition_peak_pct"] = tpeak_row.mean() * 100.0
            out["ct_transition_entropy"] = -(h * (h + 1e-12).log()).sum(-1).mean()
            cond = h.view(B, 10, 10)
            cond = cond / cond.sum(-1, keepdim=True).clamp_min(1e-6)  # P(out|in=a)
            pred_out = cond.argmax(-1)                                # [B,10]
            a = (ain - 2).clamp(0, 9)
            prior_tok = pred_out.gather(1, a) + 2                     # consensus token per cell
            dpcc_cell = (prior_tok == at) & achanged                 # consensus correct on changed
            out["ct_dpcc_pct"] = pct(dpcc_cell, achanged)
            hi = (tpeak_row > tpeak_thresh).unsqueeze(-1)            # [B,1] clean-recolor rows
            hi_chg = achanged & hi
            lo_chg = achanged & ~hi
            out["ct_high_tpeak_dpcc_pct"] = pct(dpcc_cell & hi, hi_chg)
            out["ct_low_tpeak_dpcc_pct"] = pct(dpcc_cell & ~hi, lo_chg)
            out["ct_high_tpeak_lodo_changed_color_pct"] = pct(acorr & hi_chg, hi_chg)
            out["ct_low_tpeak_lodo_changed_color_pct"] = pct(acorr & lo_chg, lo_chg)
            out["ct_application_gap_pct"] = (
                out["ct_high_tpeak_dpcc_pct"] - out["ct_high_tpeak_lodo_changed_color_pct"]
            )
            out["n_high_tpeak_rows"] = (tpeak_row > tpeak_thresh).float().sum()
            if ct_src_agree is not None and ct_src_support is not None:
                agree = ct_src_agree.float()
                support = ct_src_support.float()
                supported = support > 0
                out["ct_src_agree_pct"] = (
                    agree[supported].mean() * 100.0 if supported.any()
                    else torch.zeros((), device=at.device)
                )
                out["ct_src_support_pct"] = support.mean() * 100.0
    return out


def _changed_valid_mask(inputs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert inputs.shape == labels.shape, (
        f"Input/label shape mismatch: inputs={tuple(inputs.shape)}, labels={tuple(labels.shape)}"
    )
    return mask & (labels >= 2) & (inputs != labels)


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float() - x.float().mean(dim=0, keepdim=True)
    y = y.float() - y.float().mean(dim=0, keepdim=True)
    xtx = (x.T @ x).norm()
    yty = (y.T @ y).norm()
    xty = (x.T @ y).norm()
    return (xty * xty) / (xtx * yty + 1e-9)


def _median_bandwidth(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if x.shape[0] <= 1:
            return torch.tensor(1.0, device=x.device)
        d = torch.pdist(x.float())
        if d.numel() == 0:
            return torch.tensor(1.0, device=x.device)
        return torch.median(d).clamp(min=1e-3)


def _hsic_pair(x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    if n <= 1:
        return torch.zeros((), device=x.device)
    h = torch.eye(n, device=x.device, dtype=torch.float32) - (
        torch.ones(n, n, device=x.device, dtype=torch.float32) / n
    )
    kx = torch.exp(-(torch.cdist(x.float(), x.float()) ** 2) / (2 * sigma ** 2))
    ky = torch.exp(-(torch.cdist(y.float(), y.float()) ** 2) / (2 * sigma ** 2))
    return torch.trace(h @ kx @ h @ ky) / ((n - 1) ** 2)


def specialization_loss(
    z_summary: torch.Tensor,
    lam_hsic: float,
    lam_cov: float,
    lam_var: float,
    gamma: float,
) -> torch.Tensor:
    # z_summary: [K, B, D]
    if z_summary.ndim != 3 or z_summary.shape[0] < 2 or z_summary.shape[1] < 2:
        return torch.zeros((), device=z_summary.device)

    k_count, batch_size, _ = z_summary.shape
    n_pairs = k_count * (k_count - 1) // 2
    hsic = torch.zeros((), device=z_summary.device)
    cov = torch.zeros((), device=z_summary.device)

    for i in range(k_count):
        for j in range(i + 1, k_count):
            zi = z_summary[i].float()
            zj = z_summary[j].float()
            sigma = (_median_bandwidth(zi) + _median_bandwidth(zj)) / 2
            hsic = hsic + _hsic_pair(zi, zj, sigma)

            zi = zi - zi.mean(dim=0, keepdim=True)
            zj = zj - zj.mean(dim=0, keepdim=True)
            cov_ij = (zi.T @ zj) / max(batch_size - 1, 1)
            cov = cov + cov_ij.square().mean()

    stds = z_summary.float().std(dim=1, unbiased=False)
    var = torch.relu(gamma - stds).mean()
    return lam_hsic * hsic / n_pairs + lam_cov * cov / n_pairs + lam_var * var


class FVRACTLossHead(nn.Module):
    """ACT loss head with FVR diagnostics.

    This preserves the vanilla ACTLossHead contract used by pretrain.py, while
    adding optional K-stream auxiliary loss, stream specialization loss, and
    collapse diagnostics when the wrapped model returns those tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_type: str = "stablemax_cross_entropy",
        lambda_aux: float = 0.0,
        lam_hsic: float = 0.0,
        lam_cov: float = 0.0,
        lam_var: float = 0.0,
        gamma: float = 1.0,
        c2_shape_loss_weight: float = 0.0,
        # S3 colour-forcing keystone: un-detached CE on a pure-z_H colour head over the LODO
        # held-out demo (pid blanked -> cannot memorise -> forces z_H to carry the cross-demo
        # colour rule). The colour analogue of c2_shape_loss_weight. 0 = off.
        c2_color_force_weight: float = 0.0,
        c2_pad_loss_weight: float = 0.0,
        c2_boundary_pad_weight: float = 3.0,
        c2_valid_mask_loss_weight: float = 0.0,
        c2_eos_loss_weight: float = 0.0,
        c2_changed_valid_loss_weight: float = 0.0,
        # --- Phase-B delta-rule branch: two-region (inside/outside) balanced LODO loss ---
        # Computed on the model's blank-pid held-out-demo reconstruction (c2_aux_logits),
        # which flows through the factored struct+colour delta head. Replaces the weak
        # c2_leave_one_demo aux as the cross-demo trainer. 0 = off.
        c2_delta_lodo_weight: float = 0.0,
        c2_delta_changed_weight: float = 5.0,
        c2_delta_color_weight: float = 1.0,
        c2_delta_pad_weight: float = 1.0,
        c2_delta_eos_weight: float = 3.0,
        # Two-region CONTRAST: hinge forcing real-demo reconstruction of the CHANGED cells to
        # beat shuffled (wrong-task) demos -> makes the rule TASK-SPECIFIC. 0 = off.
        c2_delta_contrast_weight: float = 0.0,
        c2_delta_contrast_margin: float = 0.5,
        # PER-ROW contrast (run_stage1_local --contrast-per-row): hinge each LODO task separately and
        # mask rows without a valid wrong-task shuffle. The batch-scalar hinge (default, kept for
        # provenance) lets rows whose shuffle fell back to the CORRECT demos dilute the gap AND leak
        # gradient that pushes the model to be worse on its own real demos through the shuffle forward.
        c2_delta_contrast_per_row: bool = False,
        # --- Stage 2: preservation KL (keep base-correct UNCHANGED cells; fixes the
        # shape/content fight). Needs c2_delta_expose_base_logits=True on the model. 0 = off.
        c2_delta_preserve_weight: float = 0.0,
        # VALUE-V2 explicit use loss: trains the V2-only contribution columns on the blank-pid LODO
        # held-out demo so the main colour head cannot silently ignore the evidence block.
        c2_value_v2_aux_weight: float = 0.0,
        c2_value_v2_aux_changed_weight: float = 1.0,
        c2_value_v2_aux_copy_weight: float = 1.0,
        # --- Stage 0: no-grad diagnostic panel (free when off; needs the expose flag on) ---
        # (Stage-1 NCE/cons and the repair gate/colour kwargs were REMOVED 2026-07-02:
        # their model-side producers died in the delta-branch deletion.)
        c2_delta_diag: bool = False,
    ):
        super().__init__()
        if loss_type not in ("stablemax_cross_entropy", "softmax_cross_entropy"):
            raise ValueError("loss_type must be stablemax_cross_entropy or softmax_cross_entropy")
        self.model = model
        self.loss_type = loss_type
        self.lambda_aux = float(lambda_aux)
        self.lam_hsic = float(lam_hsic)
        self.lam_cov = float(lam_cov)
        self.lam_var = float(lam_var)
        self.gamma = float(gamma)
        self.c2_shape_loss_weight = float(c2_shape_loss_weight)
        self.c2_color_force_weight = float(c2_color_force_weight)
        self.c2_pad_loss_weight = float(c2_pad_loss_weight)
        self.c2_boundary_pad_weight = float(c2_boundary_pad_weight)
        self.c2_valid_mask_loss_weight = float(c2_valid_mask_loss_weight)
        self.c2_eos_loss_weight = float(c2_eos_loss_weight)
        self.c2_changed_valid_loss_weight = float(c2_changed_valid_loss_weight)
        self.c2_delta_lodo_weight = float(c2_delta_lodo_weight)
        self.c2_delta_changed_weight = float(c2_delta_changed_weight)
        self.c2_delta_color_weight = float(c2_delta_color_weight)
        self.c2_delta_pad_weight = float(c2_delta_pad_weight)
        self.c2_delta_eos_weight = float(c2_delta_eos_weight)
        self.c2_delta_contrast_weight = float(c2_delta_contrast_weight)
        self.c2_delta_contrast_margin = float(c2_delta_contrast_margin)
        self.c2_delta_contrast_per_row = bool(c2_delta_contrast_per_row)
        self.c2_delta_preserve_weight = float(c2_delta_preserve_weight)
        self.c2_value_v2_aux_weight = float(c2_value_v2_aux_weight)
        self.c2_value_v2_aux_changed_weight = float(c2_value_v2_aux_changed_weight)
        self.c2_value_v2_aux_copy_weight = float(c2_value_v2_aux_copy_weight)
        self.c2_delta_diag = bool(c2_delta_diag)

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]
        logits = outputs["logits"]

        with torch.no_grad():
            outputs["preds"] = torch.argmax(logits, dim=-1)
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
            is_correct = mask & (outputs["preds"] == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            valid_metrics = new_carry.halted & (loss_counts > 0)

            valid_pred_values = outputs["preds"][mask]
            unique_preds = (
                torch.unique(valid_pred_values).numel()
                if valid_pred_values.numel() > 0
                else 0
            )
            valid_labels = labels[mask]
            if valid_labels.numel() > 0:
                majority = torch.bincount(valid_labels.long()).max().float() / valid_labels.numel()
            else:
                majority = torch.zeros((), device=logits.device)

            accuracy_sum = torch.where(
                valid_metrics,
                (is_correct.to(torch.float32) / loss_divisor).sum(-1),
                torch.zeros_like(loss_counts, dtype=torch.float32),
            ).sum()
            exact_sum = (valid_metrics & seq_is_correct).sum()

            metric_count = valid_metrics.sum().clamp_min(1)
            metrics = {
                "count": valid_metrics.sum(),
                "accuracy": accuracy_sum,
                "content_accuracy": accuracy_sum,
                "exact_accuracy": exact_sum,
                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps": torch.where(valid_metrics, new_carry.steps, 0).sum(),
                "majority_floor": majority.detach() * metric_count,
                "pred_unique_classes": torch.as_tensor(float(unique_preds), device=logits.device) * metric_count,
            }

            if "alpha" in outputs:
                alpha = outputs["alpha"].detach()
                metrics["alpha_max"] = alpha.max() * metric_count
                metrics["alpha_min"] = alpha.min() * metric_count
                for idx in range(alpha.numel()):
                    metrics[f"alpha_{idx}"] = alpha[idx] * metric_count

            for c2_key, c2_value in outputs.items():
                if (
                    c2_key.startswith("c2_")
                    and torch.is_tensor(c2_value)
                    and c2_value.ndim == 0
                    and not c2_key.endswith("_weight")
                ):
                    metrics[c2_key] = c2_value.detach() * metric_count

            if "c2_rel_where_hint" in outputs:
                rel_hint = outputs["c2_rel_where_hint"].detach().float()
                if rel_hint.ndim == 3:
                    # FIX D: topk>1 widens the hint to [B,L,K]; the panel stat stays channel 0
                    # (the best predicate) so the RELHINT line is comparable across K.
                    rel_hint = rel_hint[..., 0]
                input_tokens = new_carry.current_data["inputs"].long()
                color_labels = labels >= 2
                changed_color = color_labels & (input_tokens != labels)
                unchanged_color = color_labels & (input_tokens == labels)

                def _hint_mean(mask_: torch.Tensor) -> torch.Tensor:
                    if not mask_.any():
                        return torch.zeros((), device=logits.device)
                    return rel_hint[mask_].mean()

                rel_chg = _hint_mean(changed_color)
                rel_unchg = _hint_mean(unchanged_color)
                metrics["c2_rel_where_hint_on_changed"] = rel_chg * metric_count
                metrics["c2_rel_where_hint_on_unchanged"] = rel_unchg * metric_count
                metrics["c2_rel_where_gap"] = (rel_chg - rel_unchg) * metric_count

            if "c2_algo_where_maps" in outputs:
                # FIX 5: computed-mask coverage split by changed vs copy cells. Layout matches
                # _algo_where_maps: ch0 = flood-fill enclosed, ch11:21 = nearest-seed one-hot
                # (any bit set = a seed exists). High chg / low copy = the mask marks the right cells.
                awm = outputs["c2_algo_where_maps"].detach().float()
                enclosed = awm[..., 0]
                seeded = (awm[..., 11:21].sum(dim=-1) > 0).float()
                input_tokens = new_carry.current_data["inputs"].long()
                color_labels = labels >= 2
                changed_color = color_labels & (input_tokens != labels)
                unchanged_color = color_labels & (input_tokens == labels)

                def _cov(t: torch.Tensor, mask_: torch.Tensor) -> torch.Tensor:
                    if not mask_.any():
                        return torch.zeros((), device=logits.device)
                    return t[mask_].mean()

                metrics["c2_awm_enclosed_on_changed"] = _cov(enclosed, changed_color) * metric_count
                metrics["c2_awm_enclosed_on_copy"] = _cov(enclosed, unchanged_color) * metric_count
                metrics["c2_awm_seed_on_changed"] = _cov(seeded, changed_color) * metric_count
                metrics["c2_awm_seed_on_copy"] = _cov(seeded, unchanged_color) * metric_count

            if "logits_per_stream" in outputs:
                logits_per_stream = outputs["logits_per_stream"]
                for idx in range(logits_per_stream.shape[0]):
                    preds_k = torch.argmax(logits_per_stream[idx], dim=-1)
                    correct_k = mask & (preds_k == labels)
                    acc_k = torch.where(
                        valid_metrics,
                        (correct_k.to(torch.float32) / loss_divisor).sum(-1),
                        torch.zeros_like(loss_counts, dtype=torch.float32),
                    ).sum()
                    metrics[f"per_stream_acc_{idx}"] = acc_k

            if "z_summary" in outputs:
                z_summary = outputs["z_summary"]
                if z_summary.shape[0] >= 2:
                    cka_vals = []
                    for i in range(z_summary.shape[0]):
                        for j in range(i + 1, z_summary.shape[0]):
                            cka_vals.append(_linear_cka(z_summary[i], z_summary[j]))
                    cka_stack = torch.stack(cka_vals)
                    metrics["cka_max"] = cka_stack.max().detach() * metric_count
                    metrics["cka_mean"] = cka_stack.mean().detach() * metric_count
                else:
                    metrics["cka_max"] = torch.zeros((), device=logits.device)
                    metrics["cka_mean"] = torch.zeros((), device=logits.device)
                stream_stds = z_summary.float().std(dim=1, unbiased=False).mean(dim=-1)
                metrics["stream_var_min"] = stream_stds.min().detach() * metric_count
                metrics["stream_var_max"] = stream_stds.max().detach() * metric_count

        # Read the flag once. Absent (the common non-split path) => no device tensor allocated and no
        # GPU->CPU sync; present => a single .item() sync (inherent to reading a device scalar).
        _muf = outputs.get("c2_main_uses_floor")
        if _muf is None:
            main_uses_floor = False
        elif torch.is_tensor(_muf):
            main_uses_floor = bool(_muf.item())
        else:
            main_uses_floor = bool(_muf)
        dual_head_active = (
            "c2_structure_logits" in outputs
            and "c2_color_logits" in outputs
            and not main_uses_floor
        )
        structure_logits = outputs["c2_structure_logits"].to(torch.float32) if "c2_structure_logits" in outputs else None
        structure_target = torch.zeros_like(labels, dtype=torch.long)
        structure_target = torch.where(labels == 1, torch.ones_like(structure_target), structure_target)
        structure_target = torch.where(labels >= 2, torch.full_like(structure_target, 2), structure_target)

        if dual_head_active:
            color_logits = outputs["c2_color_logits"].to(torch.float32)
            assert structure_logits is not None

            structure_ce = F.cross_entropy(
                structure_logits.reshape(-1, 3),
                structure_target.reshape(-1),
                reduction="none",
            ).view_as(labels)
            color_mask = labels >= 2
            color_target = (labels - 2).clamp_min(0).clamp_max(9).long()
            color_ce = F.cross_entropy(
                color_logits.reshape(-1, 10),
                color_target.reshape(-1),
                reduction="none",
            ).view_as(labels)

            # Factorized NLL: EOS uses structure only; colors use structure + color.
            factor_ce = torch.where(mask, structure_ce, torch.zeros_like(structure_ce))
            factor_ce = factor_ce + torch.where(color_mask, color_ce, torch.zeros_like(color_ce))
            lm_loss = (factor_ce / loss_divisor).sum()
        else:
            lm_loss = (_loss_per_token(self.loss_type, logits, labels, mask) / loss_divisor).sum()

        c2_changed_valid_loss = torch.zeros((), device=logits.device)
        if self.c2_changed_valid_loss_weight > 0:
            input_tokens = new_carry.current_data["inputs"]
            changed_valid_mask = _changed_valid_mask(input_tokens, labels, mask)
            changed_valid_counts = changed_valid_mask.sum(-1)
            changed_valid_rows = changed_valid_counts > 0
            changed_valid_labels = torch.where(
                changed_valid_mask,
                labels,
                torch.full_like(labels, IGNORE_LABEL_ID),
            )
            changed_valid_token_loss = _loss_per_token(
                self.loss_type,
                logits,
                changed_valid_labels,
                changed_valid_mask,
            )
            changed_valid_per_row = (
                changed_valid_token_loss
                / changed_valid_counts.clamp_min(1).unsqueeze(-1).to(changed_valid_token_loss.dtype)
            ).sum(-1)
            c2_changed_valid_raw_loss = torch.where(
                changed_valid_rows,
                changed_valid_per_row,
                torch.zeros_like(changed_valid_per_row),
            ).sum()
            c2_changed_valid_loss = self.c2_changed_valid_loss_weight * c2_changed_valid_raw_loss

            with torch.no_grad():
                changed_correct = changed_valid_mask & (outputs["preds"] == labels)
                changed_token_count = changed_valid_mask.sum().to(torch.float32)
                changed_correct_count = changed_correct.sum().to(torch.float32)
                metric_scale = metrics["count"].clamp_min(1)
                metrics["c2_changed_valid_raw_loss"] = c2_changed_valid_raw_loss.detach()
                metrics["c2_changed_valid_weighted_loss"] = c2_changed_valid_loss.detach()
                metrics["c2_changed_valid_token_count"] = changed_token_count.detach() * metric_scale
                metrics["c2_changed_valid_row_count"] = (
                    changed_valid_rows.sum().to(torch.float32).detach() * metric_scale
                )
                metrics["c2_changed_valid_accuracy"] = (
                    changed_correct_count / changed_token_count.clamp_min(1.0)
                ).detach() * metric_scale

        c2_pad_loss = torch.zeros((), device=logits.device)
        if self.c2_pad_loss_weight > 0:
            valid_rows = loss_counts > 0
            if structure_logits is not None:
                assert labels.ndim == 2, f"Expected labels [B, S], got shape={tuple(labels.shape)}"
                batch_size, seq_len = labels.shape
                grid_side = math.isqrt(seq_len)
                assert grid_side * grid_side == seq_len, f"Expected square grid labels, got S={seq_len}"

                structure_ce = F.cross_entropy(
                    structure_logits.reshape(-1, 3),
                    structure_target.reshape(-1),
                    reduction="none",
                ).view_as(labels)
                content = (labels >= 2).view(batch_size, grid_side, grid_side).to(torch.float32)
                kernel = torch.tensor(
                    [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
                    device=labels.device,
                    dtype=torch.float32,
                ).view(1, 1, 3, 3)
                dilated_content = F.conv2d(content.unsqueeze(1), kernel, padding=1).squeeze(1) > 0
                non_valid = (structure_target.view(batch_size, grid_side, grid_side) != 2)
                boundary_non_valid = dilated_content & non_valid
                boundary_weight = torch.ones((batch_size, grid_side, grid_side), device=labels.device, dtype=structure_ce.dtype)
                boundary_weight = torch.where(
                    boundary_non_valid,
                    torch.full_like(boundary_weight, self.c2_boundary_pad_weight),
                    boundary_weight,
                ).view(batch_size, seq_len)
                row_weight_sum = boundary_weight.sum(-1).clamp_min(1)

                c2_pad_raw_loss = torch.where(
                    valid_rows,
                    (structure_ce * boundary_weight).sum(-1) / row_weight_sum.to(structure_ce.dtype),
                    torch.zeros_like(loss_counts, dtype=structure_ce.dtype),
                ).sum()
                c2_pad_loss = self.c2_pad_loss_weight * c2_pad_raw_loss

                with torch.no_grad():
                    structure_pred = structure_logits.argmax(dim=-1)
                    metric_scale = metrics["count"].clamp_min(1)
                    row_mask = valid_rows.unsqueeze(-1)
                    pad_target = (structure_target == 0) & row_mask
                    valid_target = (structure_target == 2) & row_mask
                    eos_target = (structure_target == 1) & row_mask

                    metrics["c2_pad_loss"] = c2_pad_raw_loss.detach()
                    metrics["c2_pad_weighted_loss"] = c2_pad_loss.detach()
                    metrics["c2_pad_ce"] = c2_pad_raw_loss.detach()
                    metrics["c2_pad_weighted_ce"] = c2_pad_loss.detach()
                    metrics["c2_pad_accuracy"] = (
                        ((structure_pred == 0) & pad_target).sum().to(torch.float32)
                        / pad_target.sum().clamp_min(1).to(torch.float32)
                    ).detach() * metric_scale
                    metrics["c2_valid_token_accuracy"] = (
                        ((structure_pred == 2) & valid_target).sum().to(torch.float32)
                        / valid_target.sum().clamp_min(1).to(torch.float32)
                    ).detach() * metric_scale
                    metrics["c2_eos_token_accuracy"] = (
                        ((structure_pred == 1) & eos_target).sum().to(torch.float32)
                        / eos_target.sum().clamp_min(1).to(torch.float32)
                    ).detach() * metric_scale
                    metrics["c2_boundary_pad_fraction"] = (
                        (boundary_non_valid.view(batch_size, seq_len) & row_mask).sum().to(torch.float32)
                        / row_mask.expand_as(labels).sum().clamp_min(1).to(torch.float32)
                    ).detach() * metric_scale
                    metrics["c2_pad_fraction"] = (
                        pad_target.sum().to(torch.float32)
                        / valid_rows.sum().clamp_min(1).to(torch.float32)
                        / labels.shape[-1]
                    ).detach() * metric_scale
            else:
                pad_mask = (labels == IGNORE_LABEL_ID) & valid_rows.unsqueeze(-1)
                if not pad_mask.any():
                    pass
                else:
                    # Legacy fallback for non-dual ablations only.
                    # EXPLANATION: The dual-head path trains PAD/EOS/VALID through
                    # c2_structure_logits so content cells explicitly push against PAD.
                    # This shared-vocab branch remains only for historical comparisons.
                    pad_logp = F.log_softmax(logits.to(torch.float32), dim=-1)[..., 0]
                    pad_ce = -pad_logp
                    pad_counts = pad_mask.sum(-1)
                    pad_rows = pad_counts > 0
                    c2_pad_raw_loss = torch.where(
                        pad_rows,
                        (pad_ce * pad_mask.to(pad_ce.dtype)).sum(-1) / pad_counts.clamp_min(1).to(pad_ce.dtype),
                        torch.zeros_like(pad_ce.sum(-1)),
                    ).sum()
                    c2_pad_loss = self.c2_pad_loss_weight * c2_pad_raw_loss

                    with torch.no_grad():
                        pad_preds = outputs["preds"][pad_mask]
                        metric_scale = metrics["count"].clamp_min(1)
                        metrics["c2_pad_loss"] = c2_pad_raw_loss.detach()
                        metrics["c2_pad_weighted_loss"] = c2_pad_loss.detach()
                        metrics["c2_pad_ce"] = c2_pad_raw_loss.detach()
                        metrics["c2_pad_weighted_ce"] = c2_pad_loss.detach()
                        metrics["c2_pad_accuracy"] = (pad_preds == 0).float().mean().detach() * metric_scale
                        metrics["c2_pad_fraction"] = (
                            pad_mask.sum().to(torch.float32)
                            / valid_rows.sum().clamp_min(1).to(torch.float32)
                            / labels.shape[-1]
                        ).detach() * metric_scale
        c2_geometry_loss = torch.zeros((), device=logits.device)
        c2_shape_loss = torch.zeros((), device=logits.device)
        if "c2_structure_logits" in outputs and (
            self.c2_valid_mask_loss_weight > 0 or self.c2_eos_loss_weight > 0
        ):
            assert structure_logits is not None

            class_weights = torch.as_tensor(
                [
                    self.c2_valid_mask_loss_weight,
                    self.c2_eos_loss_weight,
                    self.c2_valid_mask_loss_weight,
                ],
                device=structure_logits.device,
                dtype=torch.float32,
            )
            structure_ce = F.cross_entropy(
                structure_logits.reshape(-1, 3),
                structure_target.reshape(-1),
                weight=class_weights,
                reduction="none",
            ).view_as(labels)
            valid_rows = loss_counts > 0
            c2_geometry_loss = torch.where(
                valid_rows,
                structure_ce.sum(-1) / torch.full_like(loss_counts, labels.shape[-1]).clamp_min(1).to(structure_ce.dtype),
                torch.zeros_like(loss_counts, dtype=structure_ce.dtype),
            ).sum()

            with torch.no_grad():
                unweighted_structure_ce = F.cross_entropy(
                    structure_logits.reshape(-1, 3),
                    structure_target.reshape(-1),
                    reduction="none",
                ).view_as(labels)
                row_mask = valid_rows.unsqueeze(-1)
                pad_structure = (structure_target == 0) & row_mask
                eos_structure = (structure_target == 1) & row_mask
                valid_structure = (structure_target == 2) & row_mask

                def _masked_mean(mask_: torch.Tensor) -> torch.Tensor:
                    if not mask_.any():
                        return torch.zeros((), device=structure_logits.device)
                    return unweighted_structure_ce[mask_].mean()

                pad_loss = _masked_mean(pad_structure)
                eos_loss = _masked_mean(eos_structure)
                valid_loss = _masked_mean(valid_structure)
                metric_scale = metrics["count"].clamp_min(1)
                metrics["c2_pad_structure_loss"] = pad_loss.detach() * metric_scale
                metrics["c2_valid_structure_loss"] = valid_loss.detach() * metric_scale
                metrics["c2_valid_mask_loss"] = (0.5 * (pad_loss + valid_loss)).detach() * metric_scale
                metrics["c2_eos_loss"] = eos_loss.detach() * metric_scale
                metrics["c2_geometry_loss"] = c2_geometry_loss.detach()

        if (
            "c2_shape_h_logits" in outputs
            and "c2_shape_w_logits" in outputs
            and "target_height" in new_carry.current_data
            and "target_width" in new_carry.current_data
        ):
            valid_shape = loss_counts > 0
            if valid_shape.any():
                h_target = new_carry.current_data["target_height"].long() - 1
                w_target = new_carry.current_data["target_width"].long() - 1
                assert h_target[valid_shape].min().item() >= 0 and h_target[valid_shape].max().item() < 30
                assert w_target[valid_shape].min().item() >= 0 and w_target[valid_shape].max().item() < 30

                h_logits = outputs["c2_shape_h_logits"].to(torch.float32)
                w_logits = outputs["c2_shape_w_logits"].to(torch.float32)
                h_loss = F.cross_entropy(h_logits[valid_shape], h_target[valid_shape], reduction="sum")
                w_loss = F.cross_entropy(w_logits[valid_shape], w_target[valid_shape], reduction="sum")
                c2_shape_raw_loss = h_loss + w_loss
                c2_shape_loss = self.c2_shape_loss_weight * c2_shape_raw_loss

                with torch.no_grad():
                    h_pred = h_logits.argmax(dim=-1)
                    w_pred = w_logits.argmax(dim=-1)
                    shape_count = valid_shape.sum().clamp_min(1)
                    shape_scale = metrics["count"].clamp_min(1)
                    h_correct = (h_pred == h_target) & valid_shape
                    w_correct = (w_pred == w_target) & valid_shape
                    shape_correct = h_correct & w_correct
                    metrics["c2_shape_loss"] = c2_shape_raw_loss.detach() * shape_scale
                    metrics["c2_shape_weighted_loss"] = c2_shape_loss.detach() * shape_scale
                    metrics["c2_shape_h_acc"] = (h_correct.sum().to(torch.float32) / shape_count) * shape_scale
                    metrics["c2_shape_w_acc"] = (w_correct.sum().to(torch.float32) / shape_count) * shape_scale
                    metrics["c2_shape_exact"] = (shape_correct.sum().to(torch.float32) / shape_count) * shape_scale

        c2_color_force_loss = torch.zeros((), device=logits.device)
        if (self.c2_color_force_weight > 0
                and "c2_color_force_logits" in outputs
                and "c2_aux_labels" in outputs
                and "c2_aux_inputs" in outputs):
            # S3 KEYSTONE: un-detached CE on a pure-z_H colour head over the LODO held-out demo.
            # The LODO pid is blanked -> the answer cannot be memorised -> the gradient forces grid_z
            # (z_H) to carry the cross-demo colour RULE. Lookup-free by construction (reads ONLY z_H),
            # so it is the clean forcing signal the demoted lookup (S5) needs to become relational.
            cf_logits = outputs["c2_color_force_logits"].to(torch.float32)         # [B,L,10]
            cf_labels = outputs["c2_aux_labels"].long()                           # [B,L] token space
            cf_inputs = outputs["c2_aux_inputs"].long()
            cf_color = cf_labels >= 2                                             # colour cells only
            if cf_color.any():
                tgt = (cf_labels - 2).clamp(0, 9)                                # 0..9 colour idx
                ce = F.cross_entropy(
                    cf_logits.reshape(-1, 10), tgt.reshape(-1), reduction="none"
                ).view_as(cf_labels)
                c2_color_force_loss = self.c2_color_force_weight * ce[cf_color].sum()
                with torch.no_grad():
                    pred = cf_logits.argmax(dim=-1)
                    scale = metrics["count"].clamp_min(1)
                    changed = cf_color & (cf_inputs != cf_labels)                 # the TRANSFORM cells
                    correct = pred == tgt
                    metrics["c2_color_force_acc"] = (
                        (correct & cf_color).sum().to(torch.float32) / cf_color.sum().clamp_min(1)) * scale
                    metrics["c2_color_force_changed_acc"] = (
                        (correct & changed).sum().to(torch.float32) / changed.sum().clamp_min(1)) * scale
                    metrics["c2_color_force_loss"] = ce[cf_color].mean().detach() * scale

        if "c2_color_logits" in outputs:
            with torch.no_grad():
                color_mask = labels >= 2
                if color_mask.any():
                    color_preds = torch.argmax(outputs["c2_color_logits"], dim=-1) + 2
                    color_correct = color_mask & (color_preds == labels)
                    color_counts = color_mask.sum(-1).clamp_min(1).unsqueeze(-1)
                    color_acc = (color_correct.float() / color_counts).sum()
                    color_valid_rows = color_mask.sum(-1) > 0
                    color_count = color_valid_rows.sum().clamp_min(1)
                    metrics["c2_color_only_accuracy"] = (color_acc / color_count) * metrics["count"].clamp_min(1)
                    if "c2_task_palette_mask" in outputs:
                        palette = outputs["c2_task_palette_mask"].to(torch.bool)
                        target_idx = (labels - 2).clamp(0, 9).long()
                        pred_idx = (color_preds - 2).clamp(0, 9).long()
                        target_allowed = palette.gather(1, target_idx) & color_mask
                        pred_allowed = palette.gather(1, pred_idx) & color_mask
                        denom = color_mask.sum().clamp_min(1)
                        scale = metrics["count"].clamp_min(1)
                        metrics["c2_palette_target_coverage_pct"] = (
                            target_allowed.sum().to(torch.float32) / denom
                        ).detach() * 100.0 * scale
                        metrics["c2_palette_pred_allowed_pct"] = (
                            pred_allowed.sum().to(torch.float32) / denom
                        ).detach() * 100.0 * scale
                        metrics["c2_palette_pred_disallowed_pct"] = (
                            ((~pred_allowed) & color_mask).sum().to(torch.float32) / denom
                        ).detach() * 100.0 * scale
        lm_loss_aux = torch.zeros((), device=logits.device)
        if self.lambda_aux > 0 and "logits_per_stream" in outputs and outputs["logits_per_stream"].shape[0] > 1:
            per_stream = outputs["logits_per_stream"]
            for idx in range(per_stream.shape[0]):
                lm_loss_aux = lm_loss_aux + (
                    _loss_per_token(self.loss_type, per_stream[idx], labels, mask) / loss_divisor
                ).sum()
            lm_loss_aux = self.lambda_aux * lm_loss_aux / per_stream.shape[0]

        c2_aux_loss = torch.zeros((), device=logits.device)
        if {"c2_aux_logits", "c2_aux_labels", "c2_aux_valid"}.issubset(outputs.keys()):
            c2_aux_logits = outputs["c2_aux_logits"]
            c2_aux_labels = _context_labels_to_loss_labels(outputs["c2_aux_labels"])
            c2_aux_valid = outputs["c2_aux_valid"].to(torch.bool)
            c2_aux_mask = c2_aux_labels != IGNORE_LABEL_ID
            c2_aux_counts = c2_aux_mask.sum(-1)
            c2_aux_valid_metrics = c2_aux_valid & (c2_aux_counts > 0)
            c2_aux_divisor = c2_aux_counts.clamp_min(1).unsqueeze(-1)
            c2_aux_per_row = (_loss_per_token(self.loss_type, c2_aux_logits, c2_aux_labels, c2_aux_mask) / c2_aux_divisor).sum(-1)
            c2_aux_raw_loss = torch.where(
                c2_aux_valid_metrics,
                c2_aux_per_row,
                torch.zeros_like(c2_aux_per_row),
            ).sum()
            c2_aux_weight = outputs.get(
                "c2_aux_weight",
                torch.zeros((), device=logits.device, dtype=torch.float32),
            )
            c2_aux_loss = c2_aux_weight.to(c2_aux_raw_loss.dtype) * c2_aux_raw_loss
            c2_lodo_contrast_loss = torch.zeros((), device=logits.device)

            with torch.no_grad():
                c2_aux_preds = torch.argmax(c2_aux_logits, dim=-1)
                c2_aux_correct = c2_aux_mask & (c2_aux_preds == c2_aux_labels)
                c2_aux_seq_correct = c2_aux_correct.sum(-1) == c2_aux_counts
                c2_aux_count = c2_aux_valid_metrics.sum().clamp_min(1)
                c2_aux_accuracy_sum = torch.where(
                    c2_aux_valid_metrics,
                    (c2_aux_correct.to(torch.float32) / c2_aux_divisor).sum(-1),
                    torch.zeros_like(c2_aux_counts, dtype=torch.float32),
                ).sum()
                c2_aux_exact_sum = (c2_aux_valid_metrics & c2_aux_seq_correct).sum()

                c2_aux_valid_labels = c2_aux_labels[c2_aux_mask & c2_aux_valid_metrics.unsqueeze(-1)]
                if c2_aux_valid_labels.numel() > 0:
                    c2_aux_majority = torch.bincount(c2_aux_valid_labels.long()).max().float() / c2_aux_valid_labels.numel()
                else:
                    c2_aux_majority = torch.zeros((), device=logits.device)

                metrics["c2_aux_loss"] = c2_aux_raw_loss.detach()
                metrics["c2_lodo_real_loss"] = c2_aux_raw_loss.detach()
                metrics["c2_aux_accuracy"] = (c2_aux_accuracy_sum.detach() / c2_aux_count) * metric_count
                metrics["c2_aux_exact_accuracy"] = (c2_aux_exact_sum.detach() / c2_aux_count) * metric_count
                metrics["c2_aux_count"] = c2_aux_valid_metrics.sum().to(torch.float32).detach() * metric_count
                metrics["c2_aux_majority_floor"] = c2_aux_majority.detach() * metric_count

            if {"c2_lodo_shuffle_logits", "c2_lodo_shuffle_labels", "c2_lodo_shuffle_valid"}.issubset(outputs.keys()):
                c2_shuffle_logits = outputs["c2_lodo_shuffle_logits"]
                c2_shuffle_labels = _context_labels_to_loss_labels(outputs["c2_lodo_shuffle_labels"])
                c2_shuffle_valid = outputs["c2_lodo_shuffle_valid"].to(torch.bool)
                c2_shuffle_mask = c2_shuffle_labels != IGNORE_LABEL_ID
                c2_shuffle_counts = c2_shuffle_mask.sum(-1)
                c2_shuffle_valid_metrics = c2_shuffle_valid & (c2_shuffle_counts > 0)
                c2_shuffle_divisor = c2_shuffle_counts.clamp_min(1).unsqueeze(-1)
                c2_shuffle_per_row = (_loss_per_token(self.loss_type, c2_shuffle_logits, c2_shuffle_labels, c2_shuffle_mask) / c2_shuffle_divisor).sum(-1)
                c2_shuffle_raw_loss = torch.where(
                    c2_shuffle_valid_metrics,
                    c2_shuffle_per_row,
                    torch.zeros_like(c2_shuffle_per_row),
                ).sum()
                contrast_valid = c2_aux_valid_metrics & c2_shuffle_valid_metrics
                contrast_count = contrast_valid.sum().clamp_min(1)
                contrast_margin = outputs.get(
                    "c2_lodo_contrast_margin",
                    torch.as_tensor(0.05, device=logits.device, dtype=torch.float32),
                ).to(c2_aux_per_row.dtype)
                contrast_per_row = F.softplus(c2_aux_per_row - c2_shuffle_per_row + contrast_margin)
                c2_lodo_contrast_raw_loss = torch.where(
                    contrast_valid,
                    contrast_per_row,
                    torch.zeros_like(contrast_per_row),
                ).sum()
                contrast_weight = outputs.get(
                    "c2_lodo_contrast_weight",
                    torch.zeros((), device=logits.device, dtype=torch.float32),
                )
                c2_lodo_contrast_loss = contrast_weight.to(c2_lodo_contrast_raw_loss.dtype) * c2_lodo_contrast_raw_loss
                c2_aux_loss = c2_aux_loss + c2_lodo_contrast_loss

                with torch.no_grad():
                    gap = torch.where(
                        contrast_valid,
                        c2_shuffle_per_row - c2_aux_per_row,
                        torch.zeros_like(c2_aux_per_row),
                    ).sum()
                    metrics["c2_lodo_shuffle_loss"] = c2_shuffle_raw_loss.detach()
                    metrics["c2_lodo_loss_gap"] = (gap.detach() / contrast_count) * metric_count
                    metrics["c2_lodo_contrast_loss"] = c2_lodo_contrast_raw_loss.detach()
                    metrics["c2_lodo_contrast_weighted_loss"] = c2_lodo_contrast_loss.detach()
                    metrics["c2_lodo_contrast_count"] = contrast_valid.sum().to(torch.float32).detach() * metric_count

        spec = torch.zeros((), device=logits.device)
        if "z_summary" in outputs and (self.lam_hsic > 0 or self.lam_cov > 0 or self.lam_var > 0):
            spec = specialization_loss(
                outputs["z_summary"],
                lam_hsic=self.lam_hsic,
                lam_cov=self.lam_cov,
                lam_var=self.lam_var,
                gamma=self.gamma,
            )

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum",
        )
        q_continue_loss = torch.zeros((), device=logits.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum",
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        c2_gate_reg = torch.zeros((), device=logits.device)
        gate_l2_weight = float(getattr(self.model.config, "c2_gate_l2_weight", 0.0))
        if gate_l2_weight > 0 and "c2_gate_patch_l2" in outputs:
            c2_gate_reg = gate_l2_weight * outputs["c2_gate_patch_l2"].to(torch.float32)

        # --- Phase-B delta-rule branch: two-region balanced LODO loss on the held-out demo ---
        # c2_aux_logits is the model's blank-pid reconstruction of the held-out demo THROUGH the
        # factored delta branch. This term (NOT the weak c2_aux/leave_one_demo CE) is the
        # cross-demo trainer. Uses RAW aux labels/inputs so PAD is supervised (boundary).
        c2_delta_lodo_loss = torch.zeros((), device=logits.device)
        if self.c2_delta_lodo_weight > 0 and {
            "c2_aux_logits", "c2_aux_labels", "c2_aux_inputs"
        }.issubset(outputs.keys()):
            delta_total, delta_in, delta_out = _ce_canvas_regions(
                outputs["c2_aux_logits"], outputs["c2_aux_labels"], outputs["c2_aux_inputs"],
                self.c2_delta_changed_weight, self.c2_delta_color_weight,
                self.c2_delta_pad_weight, self.c2_delta_eos_weight,
            )
            c2_delta_lodo_loss = self.c2_delta_lodo_weight * delta_total
            with torch.no_grad():
                metrics["c2_delta_lodo_raw"] = delta_total.detach()
                metrics["c2_delta_lodo_inside"] = delta_in.detach()
                metrics["c2_delta_lodo_outside"] = delta_out.detach()

        c2_value_v2_aux_loss = torch.zeros((), device=logits.device)
        if self.c2_value_v2_aux_weight > 0 and {
            "c2_aux_value_v2_logits", "c2_aux_labels", "c2_aux_inputs"
        }.issubset(outputs.keys()):
            v2_total, v2_changed, v2_copy, v2_changed_acc, v2_copy_acc = _value_v2_aux_ce(
                outputs["c2_aux_value_v2_logits"],
                outputs["c2_aux_labels"],
                outputs["c2_aux_inputs"],
                self.c2_value_v2_aux_changed_weight,
                self.c2_value_v2_aux_copy_weight,
            )
            c2_value_v2_aux_loss = self.c2_value_v2_aux_weight * v2_total
            with torch.no_grad():
                v2_logits = outputs["c2_aux_value_v2_logits"].detach().float()
                metrics["c2_value_v2_aux_raw"] = v2_total.detach()
                metrics["c2_value_v2_aux_changed_ce"] = v2_changed.detach()
                metrics["c2_value_v2_aux_copy_ce"] = v2_copy.detach()
                metrics["c2_value_v2_aux_changed_acc"] = v2_changed_acc.detach() * metric_count
                metrics["c2_value_v2_aux_copy_acc"] = v2_copy_acc.detach() * metric_count
                metrics["c2_value_v2_aux_logit_std"] = v2_logits.std()
                metrics["c2_value_v2_aux_logit_abs_mean"] = v2_logits.abs().mean()

        # --- Phase-B two-region CONTRAST: real demos must reconstruct the CHANGED cells better
        # than a DIFFERENT task's demos (hinge). This is the discriminative pressure that forces
        # the cross-demo rule to be TASK-SPECIFIC (the diagnostic showed ALL~ZERO~SHUFFLE without
        # it). Focused on changed-colour cells (the transform), not the task-invariant PAD/copy.
        c2_delta_contrast_loss = torch.zeros((), device=logits.device)
        if self.c2_delta_contrast_weight > 0 and {
            "c2_aux_logits", "c2_lodo_shuffle_logits", "c2_aux_labels", "c2_aux_inputs"
        }.issubset(outputs.keys()):
            if self.c2_delta_contrast_per_row:
                real_row, real_has = _changed_cell_ce(
                    outputs["c2_aux_logits"], outputs["c2_aux_labels"], outputs["c2_aux_inputs"], per_row=True)
                shuf_row, _ = _changed_cell_ce(
                    outputs["c2_lodo_shuffle_logits"], outputs["c2_aux_labels"], outputs["c2_aux_inputs"], per_row=True)
                row_ok = real_has
                if "c2_lodo_shuffle_valid" in outputs:
                    row_ok = row_ok & outputs["c2_lodo_shuffle_valid"].to(torch.bool)
                n_ok = row_ok.float().sum().clamp_min(1.0)
                hinge = (F.relu(self.c2_delta_contrast_margin + real_row - shuf_row) * row_ok.float()).sum() / n_ok
                real_chg = (real_row * row_ok.float()).sum() / n_ok
                shuf_chg = (shuf_row * row_ok.float()).sum() / n_ok
            else:
                real_chg = _changed_cell_ce(outputs["c2_aux_logits"], outputs["c2_aux_labels"], outputs["c2_aux_inputs"])
                shuf_chg = _changed_cell_ce(outputs["c2_lodo_shuffle_logits"], outputs["c2_aux_labels"], outputs["c2_aux_inputs"])
                hinge = F.relu(self.c2_delta_contrast_margin + real_chg - shuf_chg)
            c2_delta_contrast_loss = self.c2_delta_contrast_weight * hinge
            with torch.no_grad():
                # x metric_count: panel aggregators recover the per-batch value by /count (the raw
                # store deflated these by the batch count in the fixed-eval readout).
                metrics["c2_delta_contrast_real_changed"] = real_chg.detach() * metric_count
                metrics["c2_delta_contrast_shuffle_changed"] = shuf_chg.detach() * metric_count
                metrics["c2_delta_contrast_gap"] = (shuf_chg - real_chg).detach() * metric_count  # >0 = task-specific
        elif {"c2_aux_logits", "c2_lodo_shuffle_logits", "c2_aux_labels", "c2_aux_inputs"}.issubset(outputs.keys()):
            # METRIC-ONLY branch (contrast weight 0 but the shuffle build exists, e.g. the
            # quarantine lane): chgCE/gap are the GRADED LODO trend -- the only panel reads that
            # move long before any argmax metric flips (warm-init gaps are 4-8 logits). Gating
            # them behind the contrast LOSS made a contrast-free run fly blind (chgCE=nan while
            # the head trained invisibly). No grad, no loss term -- measurement only.
            with torch.no_grad():
                real_chg = _changed_cell_ce(
                    outputs["c2_aux_logits"], outputs["c2_aux_labels"], outputs["c2_aux_inputs"])
                shuf_chg = _changed_cell_ce(
                    outputs["c2_lodo_shuffle_logits"], outputs["c2_aux_labels"], outputs["c2_aux_inputs"])
                metrics["c2_delta_contrast_real_changed"] = real_chg.detach() * metric_count
                metrics["c2_delta_contrast_shuffle_changed"] = shuf_chg.detach() * metric_count
                metrics["c2_delta_contrast_gap"] = (shuf_chg - real_chg).detach() * metric_count

        # --- Stage 2: preservation KL -- keep base-correct UNCHANGED cells unchanged so the
        # rule branch stops overwriting what the frozen TRM already solves (shape/content fight). ---
        c2_delta_preserve_loss = torch.zeros((), device=logits.device)
        if (self.c2_delta_preserve_weight > 0 or self.c2_delta_diag) and {
            "c2_aux_base_logits", "c2_aux_logits", "c2_aux_labels", "c2_aux_inputs"
        }.issubset(outputs.keys()):
            _on = outputs["c2_aux_logits"]
            if self.c2_delta_preserve_weight <= 0:
                _on = _on.detach()                  # diag-only: no graph
            L_keep, preserved = _preserve_kl(
                outputs["c2_aux_base_logits"], _on,
                outputs["c2_aux_labels"], outputs["c2_aux_inputs"],
            )
            if self.c2_delta_preserve_weight > 0:
                c2_delta_preserve_loss = self.c2_delta_preserve_weight * L_keep
            with torch.no_grad():
                metrics["l_keep"] = L_keep.detach() * metric_count
                metrics["d_kl_keep"] = L_keep.detach() * metric_count
                metrics["d_preserved_correct_frac"] = preserved.detach() * metric_count

        # --- Structured health panel: MAIN / LODO / COUNTS ---
        # One call computes the whole percent panel + denominators; scaled by metric_count so
        # pretrain's /count reduction recovers the true per-batch percentage.
        if self.c2_delta_diag:
            with torch.no_grad():
                for _hk, _hv in _health_metrics(
                    labels, outputs["preds"], new_carry.current_data["inputs"],
                    aux_logits=outputs.get("c2_aux_logits"),
                    aux_floor_logits=outputs.get("c2_aux_base_logits"),
                    aux_labels=outputs.get("c2_aux_labels"),
                    aux_inputs=outputs.get("c2_aux_inputs"),
                ).items():
                    # Scalars are panel metrics (scaled by count so pretrain's /count recovers the
                    # per-batch value). PER-EXAMPLE [B] views must stay RAW 0/1: scaling them by count
                    # inflated eval_selector's percentages by the batch count (up to 800% at batch 8).
                    metrics[_hk] = _hv.detach() * metric_count if _hv.numel() == 1 else _hv.detach()

        total_lm = (
            lm_loss
            + lm_loss_aux
            + c2_aux_loss
            + c2_geometry_loss
            + c2_shape_loss
            + c2_color_force_loss
            + c2_pad_loss
            + c2_changed_valid_loss
            + c2_gate_reg
            + c2_delta_lodo_loss
            + c2_value_v2_aux_loss
            + c2_delta_contrast_loss
            + c2_delta_preserve_loss
        )
        metrics["lm_loss"] = total_lm.detach()
        metrics["lm_loss_main"] = lm_loss.detach()
        metrics["lm_loss_aux"] = lm_loss_aux.detach()
        metrics["c2_aux_weighted_loss"] = c2_aux_loss.detach()
        metrics["c2_delta_lodo_weighted_loss"] = c2_delta_lodo_loss.detach()
        metrics["c2_value_v2_aux_weighted_loss"] = c2_value_v2_aux_loss.detach()
        metrics["c2_delta_contrast_weighted_loss"] = c2_delta_contrast_loss.detach()
        metrics["c2_delta_preserve_weighted_loss"] = c2_delta_preserve_loss.detach()
        metrics["c2_changed_valid_weighted_loss"] = c2_changed_valid_loss.detach()
        metrics["c2_gate_reg_loss"] = c2_gate_reg.detach()
        metrics["spec_loss"] = spec.detach()
        metrics["spec_to_lm"] = (spec.detach() / total_lm.detach().clamp_min(1e-8)) * metrics["count"].clamp_min(1)
        metrics["q_halt_loss"] = q_halt_loss.detach()

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, total_lm + spec + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()
