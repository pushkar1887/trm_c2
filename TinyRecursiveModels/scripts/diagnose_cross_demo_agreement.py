"""Cross-demo agreement diagnosis: do a task's demos imply the SAME rule?

Answers, per task, across its demos (pure tensor stats on demo pairs; no model):

  COLOR rule   : do demos share the same dominant input->output color transition?
  VALID/SHAPE  : do demos transform the output canvas extent (H,W) the same way?
                 (same output shape? same delta from input shape? same area ratio?)
  CHANGED mask : do demos agree on WHICH KIND of change happens?
                 (fraction of cells changed; add/delete/recolor mix)

For each facet we report a per-task agreement in [0,1] (1 = all demos agree), averaged
over tasks, plus the distribution. This separates "the data is consistent" from "the
learned encoder is consistent" (D0 already showed the learned C2 is shuffle-invariant).

Token convention: PAD=0, EOS=1, color = token-2 (0..9). 900 = 30x30.
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


def grid_shape(tok_row):
    """Infer (h,w) of the colored region of a [900] token row: bbox of token>=2."""
    g = tok_row.reshape(SIDE, SIDE)
    colored = g >= OFF
    if not colored.any():
        return 0, 0
    rows = torch.where(colored.any(dim=1))[0]
    cols = torch.where(colored.any(dim=0))[0]
    return int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--global-batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, default=60)
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    raw = yaml.safe_load(Path(args.config).resolve().read_text(encoding="utf-8"))
    raw["data_paths"] = [args.dataset]; raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0; raw["global_batch_size"] = int(args.global_batch_size)
    raw["run_name"] = "xdemo"; raw["checkpoint_path"] = str(out_dir / "noop")
    raw.setdefault("arch", {})["c2_structure_fusion_alpha"] = 0.0
    config = pretrain.PretrainConfig(**raw)
    loader, _ = pretrain.create_dataloader(config, "train", 0, 1, test_set_mode=False,
                                           epochs_per_iter=1, global_batch_size=config.global_batch_size)

    color_agree, shape_agree, dshape_agree, arearatio_cv, changed_cv = [], [], [], [], []
    addmix_agree = []
    n_tasks = 0
    n_batches = 0
    for _s, cb, _g in loader:
        ci, co, cm = cb["context_inputs"], cb["context_outputs"], cb["context_mask"].bool()
        B, M, L = ci.shape
        for b in range(B):
            demos = [m for m in range(M) if bool(cm[b, m])]
            if len(demos) < 2:
                continue
            n_tasks += 1
            # ---- COLOR rule: top transition per demo ----
            tops = []
            for m in demos:
                x = ci[b, m].long(); y = co[b, m].long()
                ch = (x >= OFF) & (y >= OFF) & (x != y)
                if ch.any():
                    pair = ((x - OFF).clamp(0, 9) * 10 + (y - OFF).clamp(0, 9))[ch]
                    vals, cnts = torch.unique(pair, return_counts=True)
                    tops.append(int(vals[cnts.argmax()]))
            if len(tops) >= 2:
                vv, cc = np.unique(np.array(tops), return_counts=True)
                color_agree.append(float(cc.max()) / len(tops))
            # ---- VALID/SHAPE rule ----
            out_shapes = [grid_shape(co[b, m]) for m in demos]
            in_shapes = [grid_shape(ci[b, m]) for m in demos]
            # (a) do demos share the SAME output shape?
            uniq_out = len(set(out_shapes))
            shape_agree.append(1.0 if uniq_out == 1 else 1.0 / uniq_out)
            # (b) do demos share the same SHAPE DELTA (out-in)?
            deltas = [(o[0] - i[0], o[1] - i[1]) for o, i in zip(out_shapes, in_shapes)]
            uniq_d = len(set(deltas))
            dshape_agree.append(1.0 if uniq_d == 1 else 1.0 / uniq_d)
            # (c) area-ratio consistency (coefficient of variation; low = consistent)
            ratios = [((o[0] * o[1]) / max(i[0] * i[1], 1)) for o, i in zip(out_shapes, in_shapes)]
            r = np.array(ratios, float)
            arearatio_cv.append(float(r.std() / (r.mean() + 1e-6)))
            # ---- CHANGED mask: change-rate consistency + add/delete/recolor mix ----
            crates, addmix = [], []
            for m in demos:
                x = ci[b, m].long(); y = co[b, m].long()
                valid = (x >= OFF) | (y >= OFF)
                changed = (x != y) & valid
                crates.append(float(changed.float().sum() / valid.float().sum().clamp_min(1)))
                add = ((x < OFF) & (y >= OFF)).float().sum()
                dele = ((x >= OFF) & (y < OFF)).float().sum()
                recolor = ((x >= OFF) & (y >= OFF) & (x != y)).float().sum()
                tot = (add + dele + recolor).clamp_min(1)
                # dominant change type id: 0=add,1=del,2=recolor
                addmix.append(int(torch.tensor([add, dele, recolor]).argmax()))
            cr = np.array(crates, float)
            changed_cv.append(float(cr.std() / (cr.mean() + 1e-6)))
            if len(addmix) >= 2:
                vv, cc = np.unique(np.array(addmix), return_counts=True)
                addmix_agree.append(float(cc.max()) / len(addmix))
        n_batches += 1
        if n_batches >= args.max_batches:
            break

    def stat(xs):
        a = np.array(xs, float)
        return {"mean": float(a.mean()), "median": float(np.median(a)), "n": len(a)} if len(a) else {"mean": float("nan"), "n": 0}

    rep = {
        "n_tasks": n_tasks,
        "COLOR_rule": {
            "desc": "fraction of demos sharing the task's modal top color-transition (1=all agree)",
            **stat(color_agree),
        },
        "VALID_output_shape": {
            "desc": "1 if all demos produce identical output (H,W) shape, else 1/distinct",
            **stat(shape_agree),
        },
        "SHAPE_delta": {
            "desc": "1 if all demos share the same (out-in) shape delta, else 1/distinct",
            **stat(dshape_agree),
        },
        "area_ratio_CV": {
            "desc": "coeff of variation of out/in area ratio across demos (LOW=consistent)",
            **stat(arearatio_cv),
        },
        "CHANGED_rate_CV": {
            "desc": "coeff of variation of changed-cell fraction across demos (LOW=consistent)",
            **stat(changed_cv),
        },
        "CHANGE_TYPE_agree": {
            "desc": "fraction of demos sharing the dominant change type add/del/recolor (1=all agree)",
            **stat(addmix_agree),
        },
    }
    (out_dir / "cross_demo_agreement.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
