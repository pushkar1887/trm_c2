"""Does cross-demo context help LODO solving, and how much? (no training)

Context-ablation ladder on the CURRENT model. For each held-out demo (blank-pid, so the
model can't recall via puzzle id), reconstruct its output with progressively less context:

  ALL    : all OTHER demos of the same task          (full cross-demo)
  ONE    : exactly ONE other demo                    (minimal cross-demo)
  ZERO   : no demos (context fully masked)           (input->output prior only)
  SHUFFLE: demos from a DIFFERENT task in the batch  (wrong-task context)

Reports exact-reconstruction + changed-cell-acc at each rung. The gaps answer the question:
  helps_amount      = exact(ALL) - exact(ZERO)        how much cross-demo helps at all
  task_specific_amt = exact(ALL) - exact(SHUFFLE)     how much the RIGHT demos help
  marginal_extra    = exact(ALL) - exact(ONE)         value of MORE than one demo

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

IGNORE = 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    raw = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    raw["load_checkpoint"] = str(Path(args.checkpoint).resolve())
    raw["data_paths"] = [args.dataset]; raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0; raw["global_batch_size"] = int(args.global_batch_size)
    raw["run_name"] = "xdemo_lodo"; raw["checkpoint_path"] = str(out_dir / "noop")
    raw.setdefault("arch", {})["c2_structure_fusion_alpha"] = 0.0
    config = pretrain.PretrainConfig(**raw)
    loader, meta = pretrain.create_dataloader(config, "train", 0, 1, test_set_mode=False,
                                              epochs_per_iter=1, global_batch_size=config.global_batch_size)
    loss_head, _, _ = pretrain.create_model(config, meta, rank=0, world_size=1)
    core = loss_head.model.to(device).eval()
    for prm in core.parameters():
        prm.requires_grad_(False)
    halt = int(config.arch.halt_max_steps)

    def reconstruct(lb):
        with torch.device("cuda"):
            carry = core.initial_carry(lb)
        out = None
        for _ in range(halt):
            carry, out = core(carry=carry, batch=lb)
        pred = out["logits"].argmax(-1)
        tgt = lb["labels"]; vmask = tgt != IGNORE
        exact = ((pred == tgt) | ~vmask).all(dim=1).float()                      # [B]
        changed = ((lb["inputs"] != tgt) & vmask)
        cacc = (((pred == tgt) & changed).sum(1).float() / changed.sum(1).clamp_min(1))
        return exact, cacc

    def base_lodo(bb, hold):
        lb = dict(bb)
        lb["inputs"] = bb["context_inputs"][:, hold].clone()
        lb["labels"] = bb["context_outputs"][:, hold].clone()
        lb["puzzle_identifiers"] = torch.zeros_like(bb["puzzle_identifiers"])
        return lb

    rungs = {k: {"exact": [], "cacc": []} for k in ["ALL", "ONE", "ZERO", "SHUFFLE"]}
    nb = 0
    with torch.inference_mode():
        for _s, cb, _g in loader:
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
            if "context_inputs" not in bb or bb["context_inputs"].shape[1] < 2:
                continue
            B, M, L = bb["context_inputs"].shape
            cm = bb["context_mask"].bool()
            hold = 0
            # ALL: every other demo present
            lb = base_lodo(bb, hold)
            m = cm.clone(); m[:, hold] = False
            lb["context_mask"] = m
            e, c = reconstruct(lb); rungs["ALL"]["exact"].append(e); rungs["ALL"]["cacc"].append(c)
            # ONE: keep only the first available other demo
            m1 = torch.zeros_like(cm)
            for b in range(B):
                others = [j for j in range(M) if j != hold and bool(cm[b, j])]
                if others:
                    m1[b, others[0]] = True
            lb1 = base_lodo(bb, hold); lb1["context_mask"] = m1
            e, c = reconstruct(lb1); rungs["ONE"]["exact"].append(e); rungs["ONE"]["cacc"].append(c)
            # ZERO: no demos
            lb0 = base_lodo(bb, hold); lb0["context_mask"] = torch.zeros_like(cm)
            e, c = reconstruct(lb0); rungs["ZERO"]["exact"].append(e); rungs["ZERO"]["cacc"].append(c)
            # SHUFFLE: other task's demos (roll batch by 1), still hold out our demo's target
            roll = (torch.arange(B, device=device) + 1) % B
            lbs = base_lodo(bb, hold)
            lbs["context_inputs"] = bb["context_inputs"][roll].clone()
            lbs["context_outputs"] = bb["context_outputs"][roll].clone()
            sm = cm[roll].clone()
            lbs["context_mask"] = sm
            e, c = reconstruct(lbs); rungs["SHUFFLE"]["exact"].append(e); rungs["SHUFFLE"]["cacc"].append(c)
            nb += 1
            if nb >= args.max_batches:
                break

    def agg(key, field):
        vals = torch.cat(rungs[key][field]).float() if rungs[key][field] else torch.tensor([float("nan")])
        return float(vals.mean().item())

    res = {
        "n_batches": nb,
        "exact": {k: agg(k, "exact") for k in rungs},
        "changed_cell_acc": {k: agg(k, "cacc") for k in rungs},
    }
    eA, eO, eZ, eS = res["exact"]["ALL"], res["exact"]["ONE"], res["exact"]["ZERO"], res["exact"]["SHUFFLE"]
    res["helps_amount_exact (ALL-ZERO)"] = eA - eZ
    res["task_specific_exact (ALL-SHUFFLE)"] = eA - eS
    res["marginal_extra_demos_exact (ALL-ONE)"] = eA - eO
    # verdict
    if eA - eZ > 0.03 and eA - eS > 0.03:
        verdict = (f"CROSS-DEMO HELPS LODO: exact ALL={eA:.3f} vs ZERO={eZ:.3f} (+{eA-eZ:.3f}) and vs "
                   f"SHUFFLE={eS:.3f} (+{eA-eS:.3f}). The model uses the RIGHT task's demos to solve held-out demos.")
    elif eA - eZ > 0.03:
        verdict = (f"CONTEXT HELPS BUT NOT TASK-SPECIFIC: ALL-ZERO=+{eA-eZ:.3f} but ALL-SHUFFLE={eA-eS:+.3f} "
                   f"~0 -> model uses 'demos present' generically, not WHICH task's demos.")
    else:
        verdict = (f"CROSS-DEMO ~USELESS FOR LODO: ALL={eA:.3f} ~ ZERO={eZ:.3f}. The model reconstructs "
                   f"held-out demos from the input alone; demos add little.")
    res["verdict"] = verdict
    (out_dir / "crossdemo_helps_lodo.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
