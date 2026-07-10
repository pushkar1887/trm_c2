"""SEE why LODO held-out reconstructions are not exact -- actual cells, in ARC colours.

Trains the full stack a few steps (so the model is in its real working state), then runs the
LODO held-out-demo reconstruction and writes an SVG with, per FAILING demo, four grids:
    INPUT (held-out demo input) | TARGET (correct output) | PRED (model output) | ERRORS
ERRORS shows ONLY the wrong cells: the cell is filled with the colour the MODEL produced, given a
thick red border, and a small dot of the colour it SHOULD have been. So each wrong cell reads as
"model said <big colour>, should be <dot colour>".

It also prints the COMMON pattern of every wrong cell (across all failing demos, not just shown):
  - wrong cells that are CHANGED (input!=target) vs COPY (input==target)
  - of CHANGED-wrong: how many the model COPIED (pred==input) i.e. failed to recolour
  - of COPY-wrong:    how many the model RECOLOURED (pred!=input) i.e. corrupted a copy
  - top (target_colour -> predicted_colour) error pairs
  - boundary (touching PAD/EOS) vs interior wrong cells

CMD (from D:\\trm_c2\\TinyRecursiveModels):
  trm\\Scripts\\python.exe scripts\\lodo_visualize.py --train-steps 200 --examples 8
Output: reports\\lodo_vis\\lodo_failures.svg  (+ a printed common-pattern summary)
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pretrain  # noqa: E402

CONFIG = "checkpoints/TRM-FVR-Experiments/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/all_config.yaml"
CKPT = r"D:\trm_c2\step_518071"
DATASET = r"D:\trm_c2\arc1concept-aug-1000"
SIDE = 30
PAD, EOS, OFF = 0, 1, 2
ARC = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
       "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]
CNAME = ["black", "blue", "red", "green", "yellow", "grey", "magenta", "orange", "cyan", "maroon"]


def tok_name(t: int) -> str:
    if t == PAD:
        return "PAD"
    if t == EOS:
        return "EOS"
    return CNAME[(t - OFF) % 10]


def fill(t: int) -> str:
    if t == PAD:
        return "#f5f5f5"
    if t == EOS:
        return "#cccccc"
    return ARC[(t - OFF) % 10]


def bbox(mask2d: torch.Tensor):
    rows = torch.where(mask2d.any(1))[0]
    cols = torch.where(mask2d.any(0))[0]
    if rows.numel() == 0:
        return 0, 1, 0, 1
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def grid_svg(g2d, r0, r1, c0, c1, ox, oy, cs, error_overlay=None):
    """error_overlay: optional target grid; when given, g2d is the PRED grid and only wrong cells
    are drawn (model colour + red border + target-colour dot), correct cells faint white."""
    out = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            x = ox + (c - c0) * cs
            y = oy + (r - r0) * cs
            p = int(g2d[r, c])
            if error_overlay is None:
                out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cs}" height="{cs}" '
                           f'fill="{fill(p)}" stroke="#999" stroke-width="0.5"/>')
            else:
                t = int(error_overlay[r, c])
                if t == p:
                    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cs}" height="{cs}" '
                               f'fill="#ffffff" stroke="#eeeeee" stroke-width="0.5"/>')
                else:
                    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cs}" height="{cs}" '
                               f'fill="{fill(p)}" stroke="#ff0000" stroke-width="1.6"/>')
                    d = cs * 0.42
                    out.append(f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{d:.1f}" height="{d:.1f}" '
                               f'fill="{fill(t)}" stroke="#000000" stroke-width="0.5"/>')
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--examples", type=int, default=8, help="failing demos to draw (closest first)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--prior-gate", type=float, default=3.0)
    ap.add_argument("--cell", type=int, default=14)
    ap.add_argument("--out", type=str, default="reports/lodo_vis/lodo_failures.svg")
    args = ap.parse_args()

    raw = yaml.safe_load(Path(CONFIG).resolve().read_text(encoding="utf-8"))
    raw["load_checkpoint"] = CKPT
    raw["data_paths"] = [DATASET]
    raw["data_paths_test"] = []
    raw["dataloader_num_workers"] = 0
    raw["global_batch_size"] = args.batch
    raw["run_name"] = "lodo_vis"
    raw["checkpoint_path"] = "reports/lodo_vis_noop"
    arch = raw.setdefault("arch", {})
    arch["c2_structure_fusion_alpha"] = 0.0
    arch["c2_delta_rule_branch"] = True
    arch["c2_delta_rule_encoder_dim"] = 256
    arch["c2_delta_rule_slots"] = 8
    arch["c2_delta_rule_logit_residual"] = True
    arch["c2_delta_rule_logit_replace"] = True
    arch["c2_delta_rule_slot_attend"] = True
    arch["c2_delta_rule_factored_head"] = True
    arch["c2_delta_rule_cell_gate_bias"] = -2.0
    arch["c2_delta_expose_rule_vec"] = True
    arch["c2_delta_expose_base_logits"] = True
    arch["c2_color_transition_bank"] = True
    arch["c2_color_prior"] = True
    arch["c2_color_prior_gate_init"] = args.prior_gate
    arch["c2_lodo_blank_pid"] = True
    arch["c2_leave_one_demo_weight"] = 0.0
    arch["c2_lodo_force_build"] = True
    arch["c2_lodo_force_shuffle"] = True
    arch["c2_lodo_contrast_weight"] = 0.0
    loss = arch.setdefault("loss", {})
    loss["c2_delta_lodo_weight"] = 0.5
    loss["c2_delta_changed_weight"] = 5.0
    loss["c2_delta_color_weight"] = 1.0
    loss["c2_delta_pad_weight"] = 1.0
    loss["c2_delta_eos_weight"] = 3.0
    loss["c2_delta_contrast_weight"] = 1.0
    loss["c2_delta_diag"] = True
    loss["c2_delta_nce_weight"] = 2.0
    loss["c2_delta_cons_weight"] = 0.5
    loss["c2_changed_valid_loss_weight"] = 0.3

    config = pretrain.PretrainConfig(**raw)
    loader, meta = pretrain.create_dataloader(
        config, "train", 0, 1, test_set_mode=False,
        epochs_per_iter=1, global_batch_size=config.global_batch_size)
    loss_head, _, _ = pretrain.create_model(config, meta, rank=0, world_size=1)
    loss_head.train()
    device = torch.device("cuda")
    params = [p for n, p in loss_head.named_parameters() if p.requires_grad and "puzzle_emb" not in n]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    aux_keys = ["c2_aux_logits", "c2_aux_labels", "c2_aux_inputs"]

    def batches():
        while True:
            for _s, cb, _g in loader:
                if "context_inputs" in cb and cb["context_inputs"].shape[1] >= 2:
                    yield {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cb.items()}

    bgen = batches()
    print(f"[lodo-vis] training {args.train_steps} steps (prior_gate={args.prior_gate})...")
    for step in range(args.train_steps):
        batch = next(bgen)
        with torch.device("cuda"):
            carry = loss_head.initial_carry(batch)
        carry, loss_val, _m, _d, _h = loss_head(carry=carry, batch=batch, return_keys=[])
        loss_head.zero_grad(set_to_none=True)
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

    # ---- collect failing held-out demos ----
    print("[lodo-vis] collecting failing held-out demos...")
    found = []  # (n_wrong, input2d, target2d, pred2d)
    seen = 0
    with torch.no_grad():
        for _ in range(12):
            batch = next(bgen)
            with torch.device("cuda"):
                carry = loss_head.initial_carry(batch)
            carry, _l, _m, dout, _h = loss_head(carry=carry, batch=batch, return_keys=aux_keys)
            if "c2_aux_logits" not in dout:
                continue
            logit = dout["c2_aux_logits"]
            lab = dout["c2_aux_labels"].long()
            inp = dout["c2_aux_inputs"].long()
            prd = logit.argmax(-1).long()
            B = lab.shape[0]
            for b in range(B):
                t = lab[b]
                keep = t >= 0
                if (t >= OFF).sum() < 6:        # skip near-empty grids
                    continue
                wrong = (prd[b] != t) & keep
                nw = int(wrong.sum())
                seen += 1
                if nw == 0:                      # already exact -> not a failure
                    continue
                found.append((nw,
                              inp[b].view(SIDE, SIDE).cpu(),
                              t.view(SIDE, SIDE).cpu(),
                              prd[b].view(SIDE, SIDE).cpu()))
            if len(found) >= 24:
                break

    found.sort(key=lambda r: r[0])              # closest failures first
    shown = found[: args.examples]
    print(f"[lodo-vis] held-out demos seen={seen}  failing={len(found)}  drawing closest {len(shown)}")

    # ---- COMMON PATTERN over ALL failing wrong cells ----
    n_wrong = n_changed_wrong = n_copy_wrong = 0
    n_changed_copied = 0   # changed cell, model output == input (failed to recolour)
    n_copy_recoloured = 0  # copy cell, model output != input (corrupted a copy)
    n_boundary = 0         # wrong cell adjacent to a PAD/EOS cell in target
    pair = Counter()       # (target_name -> pred_name)
    for _nw, inp2d, tgt2d, prd2d in found:
        keep = tgt2d >= 0
        wrong = (prd2d != tgt2d) & keep
        ys, xs = torch.where(wrong)
        for y, x in zip(ys.tolist(), xs.tolist()):
            t = int(tgt2d[y, x]); p = int(prd2d[y, x]); i = int(inp2d[y, x])
            n_wrong += 1
            pair[(tok_name(t), tok_name(p))] += 1
            changed = (i != t) and (t >= OFF) and (i >= OFF)
            if changed:
                n_changed_wrong += 1
                if p == i:
                    n_changed_copied += 1
            elif (i == t) and (t >= OFF):
                n_copy_wrong += 1
                if p != i:
                    n_copy_recoloured += 1
            nb = False
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < SIDE and 0 <= xx < SIDE and int(tgt2d[yy, xx]) in (PAD, EOS):
                    nb = True
            n_boundary += int(nb)

    def pc(n):
        return 100.0 * n / max(n_wrong, 1)

    print("\n================ COMMON PATTERN OF WRONG CELLS ================")
    print(f"total wrong cells (over {len(found)} failing demos): {n_wrong}")
    print(f"  CHANGED-cell errors (input!=target): {n_changed_wrong} ({pc(n_changed_wrong):.1f}%)")
    print(f"     of those, model COPIED the input (failed to recolour): "
          f"{n_changed_copied} ({100.0*n_changed_copied/max(n_changed_wrong,1):.1f}% of changed-wrong)")
    print(f"  COPY-cell errors  (input==target): {n_copy_wrong} ({pc(n_copy_wrong):.1f}%)")
    print(f"     of those, model RECOLOURED a copy (corrupted): "
          f"{n_copy_recoloured} ({100.0*n_copy_recoloured/max(n_copy_wrong,1):.1f}% of copy-wrong)")
    print(f"  boundary-adjacent wrong cells (touch PAD/EOS): {n_boundary} ({pc(n_boundary):.1f}%)")
    print("  top target->pred error pairs:")
    for (tn, pn), c in pair.most_common(10):
        print(f"     {tn:>8} -> {pn:<8}  x{c}")
    print("==============================================================\n")

    # ---- render SVG ----
    cs = args.cell
    pad_x, pad_y, gap = 24, 40, 26
    panels = ["INPUT", "TARGET", "PRED", "ERRORS"]
    rows_svg = []
    max_w = 0
    yo = pad_y
    for idx, (nw, inp2d, tgt2d, prd2d) in enumerate(shown):
        content = (tgt2d >= OFF) | (prd2d >= OFF)
        r0, r1, c0, c1 = bbox(content)
        gw = (c1 - c0) * cs
        gh = (r1 - r0) * cs
        xo = pad_x
        rows_svg.append(f'<text x="{xo}" y="{yo - 8}" font-size="13" font-family="monospace">'
                        f'demo {idx}  (wrong cells: {nw})</text>')
        for pi, name in enumerate(panels):
            lbl_x = xo + 2
            rows_svg.append(f'<text x="{lbl_x}" y="{yo + gh + 14}" font-size="11" '
                            f'font-family="monospace" fill="#444">{name}</text>')
            if name == "INPUT":
                rows_svg.append(grid_svg(inp2d, r0, r1, c0, c1, xo, yo, cs))
            elif name == "TARGET":
                rows_svg.append(grid_svg(tgt2d, r0, r1, c0, c1, xo, yo, cs))
            elif name == "PRED":
                rows_svg.append(grid_svg(prd2d, r0, r1, c0, c1, xo, yo, cs))
            else:
                rows_svg.append(grid_svg(prd2d, r0, r1, c0, c1, xo, yo, cs, error_overlay=tgt2d))
            xo += gw + gap
        max_w = max(max_w, xo)
        yo += gh + gap + 24
    total_h = yo + pad_y
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{max_w + pad_x}" height="{total_h}" '
           f'font-family="monospace">'
           f'<rect width="100%" height="100%" fill="#ffffff"/>'
           f'<text x="{pad_x}" y="22" font-size="15">LODO held-out reconstruction failures '
           f'(ERRORS: big=model colour, dot=correct colour, red border=wrong cell)</text>'
           + "".join(rows_svg) + "</svg>")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    print(f"[lodo-vis] wrote {out_path.resolve()}  ({len(shown)} demos)")


if __name__ == "__main__":
    main()
