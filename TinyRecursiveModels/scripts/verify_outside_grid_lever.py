"""Verifier for the §15.9.1 GENERALIZED outside-grid PAD lever (extent_pad_mask + _predicted_extent).

Runs the REAL 518K aux (LODO) forward over MANY batches and separates the three things the realized
`pad` metric conflates: (rule verified) x (mask geometry correct) x (override strong enough).

HARD INVARIANTS (asserted -> process exits non-zero on violation, so this is CI-wireable):
  I1  oracle-size mask reproduces the target-PAD region EXACTLY               (aggregate IoU == 100%)
  I2  mask is EOS-clean                                                       (0 cells convert EOS->PAD)
  I3  same-shape regression: mask PAD == (input==0) on same-shape rows        (100%)

REPORTED (diagnostic, not asserted):
  CEILING   size-rule verify rate from _predicted_extent (== the new pad ceiling; residual -> shape head)
  REALIZED  candidate vs floor pad-win, aggregated over batches
  OVERRIDE  floor colour-over-pad gap vs the ACTIVE override V (read off the model); #cells where gap>V

Run: .\\trm\\Scripts\\python.exe scripts\\verify_outside_grid_lever.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import math

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrain  # noqa: E402
from models.recursive_reasoning.trm_fvr_c2 import extent_eos_mask, extent_pad_mask  # noqa: E402

CONFIG = "checkpoints/TRM-FVR-Experiments/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/all_config.yaml"
DATASET = r"D:\trm_c2\arc1concept-aug-1000"
CKPT = r"D:\trm_c2\step_518071"
MAX_BATCHES = 20        # aggregate over this many aux forwards (~160 examples) -> not an 8-row anecdote


def _tok_extent(tokens: torch.Tensor, side: int):
    """[B,L] tokens -> (h, w, off_r, off_c, has) each [B] in CELLS, from the content bbox (colours >=2).

    The tokenizer fills the ENTIRE HxW box with tokens >=2 (grid+2, incl. background colour 0 -> token 2),
    so no interior all-PAD gaps are possible and rows.sum()/cols.sum() equal the true extent exactly.
    """
    b = tokens.shape[0]
    g = (tokens.reshape(b, side, side) >= 2)
    rows = g.any(dim=2)
    cols = g.any(dim=1)
    has = g.any(dim=(1, 2))
    off_r = torch.argmax(rows.int(), dim=1)
    off_c = torch.argmax(cols.int(), dim=1)
    h = rows.sum(dim=1)
    w = cols.sum(dim=1)
    return h, w, off_r, off_c, has


def build():
    raw = yaml.safe_load(Path(CONFIG).resolve().read_text(encoding="utf-8"))
    raw["load_checkpoint"] = CKPT
    raw["data_paths"] = [DATASET]
    raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0
    raw["global_batch_size"] = 8
    arch = raw.setdefault("arch", {})
    # v3-clean + outside-grid, mirroring run_stage1_local.
    arch["c2_dual_output_head"] = True
    arch["c2_structure_fusion_alpha"] = 0.0
    arch["c2_structure_from_lmhead"] = True
    arch["c2_relmap"] = True
    arch["c2_relmap_structure"] = True
    arch["c2_relmap_outside_grid"] = True
    arch["c2_structure_outside_warm_init"] = True
    arch["c2_structure_outside_warm_init_value"] = 1000.0
    arch["c2_relmap_eos_grid"] = True
    arch["c2_structure_eos_warm_init"] = True
    arch["c2_structure_eos_warm_init_value"] = 1000.0
    arch["c2_lodo_blank_pid"] = True
    arch["c2_lodo_force_build"] = True
    arch["c2_lodo_force_shuffle"] = False
    arch["c2_delta_expose_base_logits"] = True
    config = pretrain.PretrainConfig(**raw)
    loader, meta = pretrain.create_dataloader(
        config, "train", 0, 1, test_set_mode=False, epochs_per_iter=1, global_batch_size=8)
    loss_head, _, _ = pretrain.create_model(config, meta, rank=0, world_size=1)
    loss_head.train()   # the LODO aux forward is built in train mode (mirrors run_stage1_local)
    return loss_head, loader


def main():
    loss_head, loader = build()
    inner = loss_head.model.inner
    sop = getattr(inner, "structure_outside_proj", None)
    if sop is None:
        print("FAIL: structure_outside_proj not built (c2_relmap_outside_grid off?)"); sys.exit(1)
    V = float(sop.weight[0, 0])                         # the ACTIVE override magnitude (not a stale constant)
    print(f"structure_outside_proj built | override V (pad-row W[0,0]) = {V:.1f}")
    sep = getattr(inner, "structure_eos_proj", None)
    if sep is None:
        print("FAIL: structure_eos_proj not built (c2_relmap_eos_grid off?)"); sys.exit(1)
    EV = float(sep.weight[1, 0])
    print(f"structure_eos_proj built     | override V (eos-row W[1,0]) = {EV:.1f}")
    device = next(loss_head.parameters()).device

    a = {k: 0 for k in ("tgt", "floor_win", "cand_win", "inter", "union", "eos_leak",
                        "eos_tgt", "eos_floor_win", "eos_cand_win", "eos_inter", "eos_union", "eos_pad_leak",
                        "ss_num", "ss_den", "conf1", "rows", "gap_n", "gap_over", "off_max")}
    a["gap_sum"] = 0.0
    need = ["c2_aux_logits", "c2_aux_base_logits", "c2_aux_labels", "c2_aux_inputs"]
    nb = 0
    for _s, cb, _g in loader:
        if "context_inputs" not in cb or cb["context_inputs"].shape[1] < 2:
            continue
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
        with torch.no_grad():
            with torch.device(device.type):
                carry = loss_head.initial_carry(batch)
            _c, _l, _m, out, _ = loss_head(carry=carry, batch=batch, return_keys=need)
            if not all(k in out for k in need):
                continue
            cand = out["c2_aux_logits"].float()          # [B,L,12] candidate (with the lever)
            base = out["c2_aux_base_logits"].float()      # [B,L,12] floor
            lab = out["c2_aux_labels"].long()             # [B,L] target
            inp = out["c2_aux_inputs"].long()             # [B,L] aux query input
            side = int(math.isqrt(lab.shape[1]))
            tgt_pad = (lab == 0)
            tgt_eos = (lab == 1)
            a["tgt"] += int(tgt_pad.sum())
            a["floor_win"] += int((base.argmax(-1)[tgt_pad] == 0).sum())
            a["cand_win"] += int((cand.argmax(-1)[tgt_pad] == 0).sum())
            a["eos_tgt"] += int(tgt_eos.sum())
            a["eos_floor_win"] += int((base.argmax(-1)[tgt_eos] == 1).sum())
            a["eos_cand_win"] += int((cand.argmax(-1)[tgt_eos] == 1).sum())
            # OVERRIDE adequacy: floor colour-over-pad gap vs V
            gap = torch.logsumexp(base[..., 2:12], dim=-1) - base[..., 0]
            g = gap[tgt_pad]
            a["gap_sum"] += float(g.sum()); a["gap_n"] += int(g.numel())
            a["gap_over"] += int((g > 2 * V).sum())       # swing is 2V (pad +V, valid -V) -> loses only if gap>2V
            # I1/I2 mask geometry (ORACLE target extent isolates geometry from prediction)
            th, tw, toff_r, toff_c, thas = _tok_extent(lab, side)
            if thas.any():
                a["off_max"] = max(a["off_max"], int(torch.maximum(toff_r[thas], toff_c[thas]).max()))
            mask = extent_pad_mask(inp, th.float(), tw.float(), side) > 0.5
            a["inter"] += int((mask & tgt_pad).sum()); a["union"] += int((mask | tgt_pad).sum())
            a["eos_leak"] += int((mask & (lab == 1)).sum())
            eos_mask = extent_eos_mask(inp, th.float(), tw.float(), side) > 0.5
            a["eos_inter"] += int((eos_mask & tgt_eos).sum()); a["eos_union"] += int((eos_mask | tgt_eos).sum())
            a["eos_pad_leak"] += int((eos_mask & tgt_pad).sum())
            # I3 same-shape regression
            ih, iw, _, _, _ = _tok_extent(inp, side)
            same = (th == ih) & (tw == iw)
            if same.any():
                in_pad = (inp == 0)
                eqrows = (mask == in_pad).reshape(inp.shape[0], -1).all(dim=1)
                a["ss_num"] += int((eqrows & same).sum()); a["ss_den"] += int(same.sum())
            # CEILING: size-rule verify rate (all demos; LODO proxy for the aux ceiling)
            ext = inner._predicted_extent(batch)
            if ext is not None:
                _h, _w, conf = ext
                a["conf1"] += int((conf > 0.5).sum()); a["rows"] += int(conf.numel())
        nb += 1
        if nb >= MAX_BATCHES:
            break

    if nb == 0 or a["tgt"] == 0:
        print("FAIL: no usable aux batches (no context_inputs / aux keys)."); sys.exit(1)
    iou = a["inter"] / max(a["union"], 1)
    eos_iou = a["eos_inter"] / max(a["eos_union"], 1)
    print(f"\nbatches={nb}  aux target-PAD cells={a['tgt']}  max box offset seen={a['off_max']} "
          f"(offset!=0 => translation; invariant must still hold)")
    print(f"[INVARIANT] oracle mask IoU={iou*100:.2f}%  eos-leak={a['eos_leak']}  "
          f"same-shape-regression={a['ss_num']}/{a['ss_den']}")
    print(f"[EOS]       oracle eos IoU={eos_iou*100:.2f}%  pad-leak={a['eos_pad_leak']}  "
          f"eos-WIN floor={a['eos_floor_win']/max(a['eos_tgt'],1)*100:.1f}%  "
          f"candidate={a['eos_cand_win']/max(a['eos_tgt'],1)*100:.1f}%")
    print(f"[CEILING]   size-rule verify rate: {a['conf1']}/{a['rows']} = "
          f"{a['conf1']/max(a['rows'],1)*100:.1f}%  (the pad ceiling; residual needs the shape-head fallback)")
    print(f"[REALIZED]  pad-WIN on target-PAD: floor={a['floor_win']/a['tgt']*100:.1f}%  "
          f"candidate={a['cand_win']/a['tgt']*100:.1f}%")
    print(f"[OVERRIDE]  floor gap mean={a['gap_sum']/max(a['gap_n'],1):.1f}  cells with gap>2V({2*V:.0f})="
          f"{a['gap_over']}  (must be 0; else raise --structure-outside-warm-init-value)")

    # HARD assertions -> non-zero exit on any invariant break (CI-wireable)
    assert a["eos_leak"] == 0, f"I2 EOS-LEAK: {a['eos_leak']} cells convert EOS->PAD (mask not eos-clean)"
    assert iou > 0.999, f"I1 oracle mask IoU={iou:.4f} != 1.0 (extent_pad_mask geometry broken)"
    assert a["eos_pad_leak"] == 0, f"EOS mask leaks onto PAD: {a['eos_pad_leak']} cells"
    assert eos_iou > 0.999, f"EOS oracle mask IoU={eos_iou:.4f} != 1.0 (extent_eos_mask geometry broken)"
    if a["ss_den"] > 0:
        assert a["ss_num"] == a["ss_den"], (
            f"I3 same-shape regression {a['ss_num']}/{a['ss_den']} (mask != (input==0) on a same-shape row)")
    assert a["gap_over"] == 0, f"OVERRIDE too weak: {a['gap_over']} target-pad cells have floor gap > 2V={2*V:.0f}"
    print("\nOK: all hard invariants hold (I1 IoU, I2 eos-clean, I3 same-shape regression, override>gap).")


if __name__ == "__main__":
    main()
