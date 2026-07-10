"""Are the INSIDE (colour) and OUTSIDE (shape/boundary) objectives hurting each other?

For each held-out demo (blank-pid LODO, branch ON), record two booleans:
  content_ok : ALL non-PAD cells correct (colours + EOS right)   -> the INSIDE job
  shape_ok   : predicted coloured bbox (h,w) == target bbox       -> the OUTSIDE/shape job
Build the 2x2 contingency and the phi correlation. Interpretation:

  both >> 0                      -> they CO-OCCUR; strict~0 is just under-training (train longer).
  both ~ 0 while each-only large -> DISJOINT: a demo gets colours OR shape, never both.
  phi < 0                        -> ANTI-correlated: optimizing one actively costs the other
                                    (true interference -> decouple the heads / separate capacity).

Leak-safe (demo pairs only). Inference only.
"""
from __future__ import annotations

import argparse
import json
import math
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
    p.add_argument("--branch", choices=["on", "off"], default="on")
    p.add_argument("--factored-head", type=int, default=1,
                   help="must match the checkpoint: 1 if trained with factored struct+color heads.")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    raw = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    raw["data_paths"] = [args.dataset]; raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0; raw["global_batch_size"] = int(args.global_batch_size)
    raw["run_name"] = "region_interf"; raw["checkpoint_path"] = str(out_dir / "noop")
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
    print(f"[interf] loaded ckpt: missing={len(miss)} unexpected={len(unexp)}")
    assert not any("delta_rule" in k for k in unexp), (
        f"FATAL: delta_rule keys UNEXPECTED -> model arch mismatch (factored_head wrong?). "
        f"unexpected delta_rule keys: {[k for k in unexp if 'delta_rule' in k]}")
    core = loss_head.model.to(device).eval()
    for prm in core.parameters():
        prm.requires_grad_(False)
    inner = core.inner
    inner._force_delta_off = (args.branch == "off")
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

    # contingency: [content_ok][shape_ok]
    ct = [[0, 0], [0, 0]]
    n = 0
    with torch.inference_mode():
        nb = 0
        for _s, cb, _g in loader:
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}
            if "context_inputs" not in bb or bb["context_inputs"].shape[1] < 2:
                continue
            lb = base_lodo(bb, 0)
            pred = run_pred(lb); tgt = lb["labels"].long()
            for i in range(pred.shape[0]):
                pr = pred[i]; tg = tgt[i]
                content_ok = bool(((pr == tg) | (tg == PAD)).all())   # all non-PAD cells correct
                shape_ok = bbox(pr) == bbox(tg)
                ct[int(content_ok)][int(shape_ok)] += 1
                n += 1
            nb += 1
            if nb >= args.max_batches:
                break

    both = ct[1][1]; content_only = ct[1][0]; shape_only = ct[0][1]; neither = ct[0][0]
    n_content = both + content_only
    n_shape = both + shape_only
    # phi correlation
    a, b, c, d = both, content_only, shape_only, neither
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d)) or 1.0
    phi = (a * d - b * c) / denom
    exp_both_if_indep = (n_content / max(n, 1)) * (n_shape / max(n, 1)) * n

    if both >= max(1.0, 0.5 * exp_both_if_indep) and both > 0:
        verdict = (f"CO-OCCUR: both={both} (expected~{exp_both_if_indep:.1f} if independent). The two "
                   "objectives are NOT fighting; strict~0 is under-training -> train longer.")
    elif n_content > 3 and n_shape > 3 and both == 0:
        verdict = (f"DISJOINT/INTERFERING: {content_only} demos get colours-only, {shape_only} get "
                   f"shape-only, but BOTH=0 (expected~{exp_both_if_indep:.1f} if independent, phi={phi:+.2f}). "
                   "A demo gets colours OR shape, never both -> the single replace-head can't satisfy "
                   "inside+outside together -> DECOUPLE (separate structure/PAD head vs colour head).")
    else:
        verdict = (f"INCONCLUSIVE/sparse: content_ok={n_content}, shape_ok={n_shape}, both={both}, "
                   f"phi={phi:+.2f}. Too few positives to judge; train longer then re-check.")

    rep = {"n_holdouts": n, "branch": args.branch,
           "contingency": {"both": both, "content_only": content_only,
                           "shape_only": shape_only, "neither": neither},
           "content_ok_total": n_content, "shape_ok_total": n_shape,
           "phi_correlation": round(phi, 3),
           "expected_both_if_independent": round(exp_both_if_indep, 2),
           "verdict": verdict}
    (out_dir / "region_interference.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
