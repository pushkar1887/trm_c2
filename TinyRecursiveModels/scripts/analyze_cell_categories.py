"""Per-cell accuracy by category on held-out demos (are colour/shape/valid coming?).

For each held-out demo (blank-pid LODO, branch ON), measure cell-level accuracy split by
what the cell IS in the target:
  pad_acc          : tgt=PAD(0)  -> frac pred=PAD            (outside-grid background)
  eos_acc          : tgt=EOS(1)  -> frac pred=EOS            (grid terminator)
  valid_acc        : tgt>=2      -> frac pred>=2             (correctly INSIDE, any colour)
  color_acc        : tgt>=2      -> frac pred==tgt           (EXACT colour, inside)
  changed_color_acc: tgt>=2 & inp!=tgt -> frac pred==tgt     (the actual TRANSFORM cells)
  unchanged_color_acc: tgt>=2 & inp==tgt -> frac pred==tgt
plus structure_acc (3-way PAD/EOS/VALID collapse) and shape_match (bbox h,w).

This separates "where/how-big the grid is" (pad/eos/valid/shape) from "what colour each
cell becomes" (color/changed) so you can see which sub-skills have landed.

Leak-safe (demo pairs only). Inference only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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


def bbox(row):
    g = row.reshape(SIDE, SIDE); col = g >= OFF
    if not bool(col.any()):
        return (0, 0)
    r = torch.where(col.any(1))[0]; c = torch.where(col.any(0))[0]
    return (int(r.max() - r.min() + 1), int(c.max() - c.min() + 1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=40)
    p.add_argument("--factored-head", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    raw = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    raw["data_paths"] = [args.dataset]; raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0; raw["global_batch_size"] = int(args.global_batch_size)
    raw["run_name"] = "cellcat"; raw["checkpoint_path"] = str(out_dir / "noop")
    arch = raw.setdefault("arch", {})
    arch["c2_structure_fusion_alpha"] = 0.0
    arch["c2_delta_rule_branch"] = True
    arch["c2_delta_rule_encoder_dim"] = 256
    arch["c2_delta_rule_slots"] = 8
    arch["c2_delta_rule_logit_residual"] = True
    arch["c2_delta_rule_logit_replace"] = True
    arch["c2_delta_rule_slot_attend"] = True
    arch["c2_delta_rule_factored_head"] = bool(args.factored_head)
    arch["c2_delta_rule_cell_gate_bias"] = 0.0
    config = pretrain.PretrainConfig(**raw)

    loader, meta = pretrain.create_dataloader(config, "train", 0, 1, test_set_mode=False,
                                              epochs_per_iter=1, global_batch_size=config.global_batch_size)
    loss_head, _, _ = pretrain.create_model(config, meta, rank=0, world_size=1)
    sd = torch.load(Path(args.checkpoint).resolve(), map_location="cpu", weights_only=False)
    miss, unexp = loss_head.load_state_dict(sd, strict=False)
    print(f"[cellcat] loaded: missing={len(miss)} unexpected={len(unexp)}")
    assert not any("delta_rule" in k for k in unexp), f"arch mismatch: {[k for k in unexp if 'delta_rule' in k]}"
    core = loss_head.model.to(device).eval()
    for prm in core.parameters():
        prm.requires_grad_(False)
    inner = core.inner
    inner._force_delta_off = False
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

    # accumulate correct/total per category
    cat = {k: [0, 0] for k in ["pad", "eos", "valid", "color", "changed_color", "unchanged_color", "structure"]}
    shape_hits = 0; n = 0
    with torch.inference_mode():
        nb = 0
        for _s, cb, _g in loader:
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
            if "context_inputs" not in bb or bb["context_inputs"].shape[1] < 2:
                continue
            lb = base_lodo(bb, 0)
            pred = run_pred(lb); tgt = lb["labels"].long(); inp = lb["inputs"].long()
            keep = tgt >= 0
            pad = (tgt == PAD) & keep; eos = (tgt == EOS) & keep; color = (tgt >= OFF) & keep
            changed = color & (inp != tgt); unchanged = color & (inp == tgt)
            exact = pred == tgt
            # structure: collapse pred & tgt to {PAD,EOS,VALID}
            def struct(x):
                return torch.where(x >= OFF, torch.full_like(x, 2), x)
            struct_ok = (struct(pred) == struct(tgt)) & keep

            def add(key, mask, hit):
                cat[key][0] += int((hit & mask).sum().item()); cat[key][1] += int(mask.sum().item())
            add("pad", pad, exact)
            add("eos", eos, exact)
            add("valid", color, pred >= OFF)          # correctly inside (any colour)
            add("color", color, exact)                # exact colour
            add("changed_color", changed, exact)
            add("unchanged_color", unchanged, exact)
            add("structure", keep, struct_ok)
            for i in range(pred.shape[0]):
                if bbox(pred[i]) == bbox(tgt[i]):
                    shape_hits += 1
                n += 1
            nb += 1
            if nb >= args.max_batches:
                break

    rep = {"n_holdouts": n, "shape_match": round(shape_hits / max(n, 1), 4)}
    for k, (h, t) in cat.items():
        rep[f"{k}_acc"] = round(h / max(t, 1), 4)
        rep[f"{k}_cells"] = t
    print(json.dumps(rep, indent=2))
    (out_dir / "cell_category_acc.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
