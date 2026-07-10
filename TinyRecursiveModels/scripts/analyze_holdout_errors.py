"""What blocks EXACT on held-out demo reconstruction? (no training)

For the LODO setup (hold out demo 0, blank pid, other demos = context), run the trained
model and, for every held-out demo, compare prediction vs target CELL BY CELL and classify
WHY it misses exact:

  token convention: PAD=0, EOS=1, color = token-2 (colors 0..9). grid = 30x30 = 900.

Per mismatched cell (pred vs tgt):
  extra_content   : tgt=PAD, pred>=color   -> painted OUTSIDE the grid (shape too big)
  missing_content : tgt>=color, pred=PAD   -> left a hole inside     (shape too small)
  eos            : exactly one of pred/tgt is EOS
  color          : both are colors but differ (right cell, WRONG colour)
  other          : anything else

Also:
  - exact_lenient : matches the trainer metric (ignores PAD target cells, tgt!=0)
  - exact_full    : ALL 900 cells must match (the true ARC criterion incl. shape/background)
  - shape_match   : bbox(pred colored) == bbox(tgt colored)  (h,w)
  - closeness     : #wrong cells per held-out demo; how many are 1/2/5 cells from exact
  - primary blocker per non-exact demo : the single category with the most wrong cells

Leak-safe (demo pairs only). Inference only. Reports branch OFF (C2-only, the cleaner model)
and branch ON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrain  # noqa: E402

PAD, EOS, OFF, SIDE = 0, 1, 2, 30


def bbox_hw(grid_row: torch.Tensor):
    """(h,w) of the colored (token>=2) region of a [900] row; (0,0) if none."""
    g = grid_row.reshape(SIDE, SIDE)
    colored = g >= OFF
    if not colored.any():
        return (0, 0)
    rows = torch.where(colored.any(1))[0]; cols = torch.where(colored.any(0))[0]
    return (int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1))


def classify(pred: torch.Tensor, tgt: torch.Tensor):
    """pred,tgt: [900] long. Return dict(category->count) over MISMATCHED cells + n_wrong."""
    mm = pred != tgt
    c = Counter()
    if not mm.any():
        return c, 0
    p = pred[mm]; t = tgt[mm]
    extra = (t == PAD) & (p >= OFF)
    missing = (t >= OFF) & (p == PAD)
    eos = ((p == EOS) | (t == EOS)) & ~extra & ~missing
    color = (p >= OFF) & (t >= OFF) & ~eos
    other = ~(extra | missing | eos | color)
    c["extra_content(shape_too_big)"] = int(extra.sum())
    c["missing_content(shape_too_small)"] = int(missing.sum())
    c["eos"] = int(eos.sum())
    c["color(right_cell_wrong_color)"] = int(color.sum())
    c["other"] = int(other.sum())
    return c, int(mm.sum())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="phaseB_generator_stepN (loss_head state_dict)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=40)
    p.add_argument("--branch", choices=["off", "on", "both"], default="both")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    raw = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    raw["data_paths"] = [args.dataset]; raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0; raw["global_batch_size"] = int(args.global_batch_size)
    raw["run_name"] = "holdout_err"; raw["checkpoint_path"] = str(out_dir / "noop")
    arch = raw.setdefault("arch", {})
    arch["c2_structure_fusion_alpha"] = 0.0
    arch["c2_delta_rule_branch"] = True
    arch["c2_delta_rule_encoder_dim"] = 256
    arch["c2_delta_rule_slots"] = 8
    arch["c2_delta_rule_logit_residual"] = True
    arch["c2_delta_rule_logit_replace"] = True
    arch["c2_delta_rule_slot_attend"] = True
    arch["c2_delta_rule_cell_gate_bias"] = 0.0
    config = pretrain.PretrainConfig(**raw)

    loader, meta = pretrain.create_dataloader(config, "train", 0, 1, test_set_mode=False,
                                              epochs_per_iter=1, global_batch_size=config.global_batch_size)
    loss_head, _, _ = pretrain.create_model(config, meta, rank=0, world_size=1)
    sd = torch.load(Path(args.checkpoint).resolve(), map_location="cpu", weights_only=False)
    miss, unexp = loss_head.load_state_dict(sd, strict=False)
    print(f"[err] loaded checkpoint: missing={len(miss)} unexpected={len(unexp)}")
    core = loss_head.model.to(device).eval()
    for prm in core.parameters():
        prm.requires_grad_(False)
    inner = core.inner
    halt = int(config.arch.halt_max_steps)

    def run_pred(batch):
        with torch.device("cuda"):
            carry = core.initial_carry(batch)
        out = None
        for _ in range(halt):
            carry, out = core(carry=carry, batch=batch)
        return out["logits"].argmax(-1)

    def base_lodo(bb, hold=0):
        lb = dict(bb)
        lb["inputs"] = bb["context_inputs"][:, hold].clone()
        lb["labels"] = bb["context_outputs"][:, hold].clone()
        cm = bb["context_mask"].clone().to(torch.bool); cm[:, hold] = False
        lb["context_mask"] = cm
        lb["puzzle_identifiers"] = torch.zeros_like(bb["puzzle_identifiers"])
        return lb

    def analyze(branch_off: bool):
        inner._force_delta_off = branch_off
        n = 0
        exact_lenient = exact_full = shape_match = 0
        wrong_full_list = []                 # #wrong cells (all 900) per non-exact demo
        cat_total = Counter()                # summed wrong-cell categories (non-exact only)
        primary = Counter()                  # dominant blocker per non-exact demo
        lenient_but_not_full = 0             # exact under tgt!=PAD but NOT all-900 (shape leak)
        with torch.inference_mode():
            nb = 0
            for _s, cb, _g in loader:
                bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
                if "context_inputs" not in bb or bb["context_inputs"].shape[1] < 2:
                    continue
                lb = base_lodo(bb, 0)
                pred = run_pred(lb)                       # [B,900]
                tgt = lb["labels"].long()
                B = pred.shape[0]
                for b in range(B):
                    pr = pred[b].long(); tg = tgt[b]
                    vmask = tg != PAD
                    e_len = bool(((pr == tg) | ~vmask).all())
                    e_full = bool((pr == tg).all())
                    exact_lenient += int(e_len); exact_full += int(e_full)
                    if bbox_hw(pr) == bbox_hw(tg):
                        shape_match += 1
                    if e_len and not e_full:
                        lenient_but_not_full += 1
                    if not e_full:
                        cats, nwrong = classify(pr, tg)
                        wrong_full_list.append(nwrong)
                        cat_total.update(cats)
                        # dominant blocker: collapse shape sub-types
                        shape_ct = cats["extra_content(shape_too_big)"] + cats["missing_content(shape_too_small)"]
                        buckets = {"shape": shape_ct, "color": cats["color(right_cell_wrong_color)"],
                                   "eos": cats["eos"], "other": cats["other"]}
                        primary[max(buckets, key=buckets.get)] += 1
                    n += 1
                nb += 1
                if nb >= args.max_batches:
                    break
        wf = np.array(wrong_full_list) if wrong_full_list else np.array([0])
        denom = max(sum(cat_total.values()), 1)
        rep = {
            "n_holdouts": n,
            "exact_lenient_tgt!=PAD": round(exact_lenient / max(n, 1), 4),
            "exact_full_all900": round(exact_full / max(n, 1), 4),
            "shape_match_rate": round(shape_match / max(n, 1), 4),
            "exact_lenient_but_shape_wrong": lenient_but_not_full,
            "among_non_exact": {
                "count": len(wrong_full_list),
                "median_wrong_cells": float(np.median(wf)),
                "mean_wrong_cells": round(float(wf.mean()), 2),
                "within_1_cell": int((wf <= 1).sum()),
                "within_2_cells": int((wf <= 2).sum()),
                "within_5_cells": int((wf <= 5).sum()),
                "within_10_cells": int((wf <= 10).sum()),
                "wrong_cell_category_pct": {k: round(100 * v / denom, 1) for k, v in cat_total.most_common()},
                "primary_blocker_per_demo": dict(primary.most_common()),
            },
        }
        return rep

    result = {}
    if args.branch in ("off", "both"):
        print("[err] analyzing branch OFF (C2-trained, delta branch disabled)...")
        result["branch_off"] = analyze(True)
    if args.branch in ("on", "both"):
        print("[err] analyzing branch ON (full model)...")
        result["branch_on"] = analyze(False)
    inner._force_delta_off = False
    (out_dir / "holdout_error_breakdown.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
