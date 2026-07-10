import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain
from models.losses import IGNORE_LABEL_ID


FIELDS = [
    "task_id",
    "puzzle_id",
    "bucket",
    "exact_accuracy",
    "content_accuracy",
    "count",
    "n_steps",
    "close_miss",
    "failed_band",
    "shape_exact",
    "height_acc",
    "width_acc",
    "valid_mask_exact",
    "eos_mask_exact",
    "outside_canvas_fpr",
    "inside_canvas_color_acc",
    "majority_floor_content",
    "labelmasked_wrong_cells",
    "labelmasked_valid_to_valid_wrong_cells",
]

METRIC_FIELDS = FIELDS[3:]
SUMMARY_METRICS = [
    "exact_accuracy",
    "content_accuracy",
    "shape_exact",
    "height_acc",
    "width_acc",
    "valid_mask_exact",
    "eos_mask_exact",
    "outside_canvas_fpr",
    "inside_canvas_color_acc",
    "close_miss",
    "failed_band",
]
BUCKETS = ["both_fail", "both_pass", "trm_only", "varc_only"]


def parse_alphas(raw: str) -> List[float]:
    alphas = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not alphas:
        raise ValueError("At least one alpha is required.")
    if 0.0 not in alphas:
        raise ValueError("Alpha sweep must include 0.0 for the no-op control.")
    return alphas


def alpha_tag(alpha: float) -> str:
    return f"alpha_{alpha:.4f}".replace(".", "p").replace("-", "m")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def crop_shape(seq: np.ndarray) -> tuple[int, int]:
    grid = seq.reshape(30, 30)
    num_c = 30
    max_area = 0
    max_shape = (0, 0)
    for num_r in range(1, 31):
        for c in range(1, num_c + 1):
            x = int(grid[num_r - 1, c - 1])
            if x < 2 or x > 11:
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_shape = (num_r, num_c)
    return max_shape


def majority_floor(label_seq: np.ndarray) -> float:
    grid = label_seq.reshape(30, 30)
    true_valid = (grid >= 2) & (grid <= 11)
    vals = grid[true_valid]
    if vals.size == 0:
        return 0.0
    _, counts = np.unique(vals, return_counts=True)
    return float(counts.max()) / float(vals.size)


def row_metrics(pred_seq: np.ndarray, raw_label_seq: np.ndarray, n_steps: int) -> Dict[str, float]:
    label_mask = raw_label_seq != IGNORE_LABEL_ID
    label_seq = np.where(label_mask, raw_label_seq, 0)
    true_h, true_w = crop_shape(label_seq)
    pred_h, pred_w = crop_shape(pred_seq)

    true_grid = label_seq.reshape(30, 30)
    pred_grid = pred_seq.reshape(30, 30)
    true_valid = (true_grid >= 2) & (true_grid <= 11)
    pred_valid = (pred_grid >= 2) & (pred_grid <= 11)
    true_eos = true_grid == 1
    pred_eos = pred_grid == 1
    outside = ~true_valid

    raw_n = max(int(label_mask.sum()), 1)
    inside_n = max(int(true_valid.sum()), 1)
    outside_n = max(int(outside.sum()), 1)
    content = float((pred_seq[label_mask] == label_seq[label_mask]).sum()) / raw_n
    exact = float(np.array_equal(pred_seq[label_mask], label_seq[label_mask]))
    labelmasked_wrong_cells = int(((pred_seq != label_seq) & label_mask).sum())
    labelmasked_valid_to_valid_wrong_cells = int(
        (
            (pred_seq != label_seq)
            & label_mask
            & (label_seq >= 2)
            & (label_seq <= 11)
            & (pred_seq >= 2)
            & (pred_seq <= 11)
        ).sum()
    )

    return {
        "exact_accuracy": exact,
        "content_accuracy": content,
        "count": 1.0,
        "n_steps": float(n_steps),
        "close_miss": float(exact == 0.0 and content >= 0.9),
        "failed_band": float(content < 0.3),
        "shape_exact": float(pred_h == true_h and pred_w == true_w),
        "height_acc": float(pred_h == true_h),
        "width_acc": float(pred_w == true_w),
        "valid_mask_exact": float(np.array_equal(pred_valid, true_valid)),
        "eos_mask_exact": float(np.array_equal(pred_eos, true_eos)),
        "outside_canvas_fpr": float((pred_valid & outside).sum()) / outside_n,
        "inside_canvas_color_acc": float(((pred_grid == true_grid) & true_valid).sum()) / inside_n,
        "majority_floor_content": majority_floor(label_seq),
        "labelmasked_wrong_cells": float(labelmasked_wrong_cells),
        "labelmasked_valid_to_valid_wrong_cells": float(labelmasked_valid_to_valid_wrong_cells),
    }


def summarize(rows: List[Dict[str, object]]) -> Dict[str, float]:
    return {
        key: sum(float(row[key]) for row in rows) / max(len(rows), 1)
        for key in SUMMARY_METRICS
    }


def wrong_cell_diagnostics(rows: List[Dict[str, object]]) -> Dict[str, int]:
    close_rows = [row for row in rows if float(row["close_miss"]) > 0]
    return {
        "total_labelmasked_valid_to_valid_wrong_cells": int(
            sum(float(row.get("labelmasked_valid_to_valid_wrong_cells", 0.0)) for row in rows)
        ),
        "close_miss_le1_scored_wrong_cells": sum(
            int(float(row.get("labelmasked_wrong_cells", 0.0)) <= 1.0) for row in close_rows
        ),
        "close_miss_le2_scored_wrong_cells": sum(
            int(float(row.get("labelmasked_wrong_cells", 0.0)) <= 2.0) for row in close_rows
        ),
        "close_miss_le3_scored_wrong_cells": sum(
            int(float(row.get("labelmasked_wrong_cells", 0.0)) <= 3.0) for row in close_rows
        ),
    }


def movement_rows(alpha: float, reference_rows: List[Dict[str, str]], rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ref_by_task = {row["task_id"]: row for row in reference_rows}
    row_by_task = {str(row["task_id"]): row for row in rows}
    out = []
    for bucket in BUCKETS:
        rec = {
            "alpha": alpha,
            "bucket": bucket,
            "content_improved": 0,
            "content_degraded": 0,
            "content_equal": 0,
            "closemiss_gain": 0,
            "closemiss_loss": 0,
            "closemiss_retained": 0,
            "exact_gain": 0,
            "exact_loss": 0,
            "exact_retained": 0,
            "labelmasked_valid_to_valid_wrong_cells": 0,
            "close_miss_le1_scored_wrong_cells": 0,
            "close_miss_le2_scored_wrong_cells": 0,
            "close_miss_le3_scored_wrong_cells": 0,
        }
        for task_id, ref in ref_by_task.items():
            if ref["bucket"] != bucket:
                continue
            cur = row_by_task[task_id]
            ref_content = float(ref["content_accuracy"])
            cur_content = float(cur["content_accuracy"])
            if cur_content > ref_content:
                rec["content_improved"] += 1
            elif cur_content < ref_content:
                rec["content_degraded"] += 1
            else:
                rec["content_equal"] += 1

            ref_close = float(ref["close_miss"]) > 0
            cur_close = float(cur["close_miss"]) > 0
            if cur_close and not ref_close:
                rec["closemiss_gain"] += 1
            elif ref_close and not cur_close:
                rec["closemiss_loss"] += 1
            elif ref_close and cur_close:
                rec["closemiss_retained"] += 1

            ref_exact = float(ref["exact_accuracy"]) > 0
            cur_exact = float(cur["exact_accuracy"]) > 0
            if cur_exact and not ref_exact:
                rec["exact_gain"] += 1
            elif ref_exact and not cur_exact:
                rec["exact_loss"] += 1
            elif ref_exact and cur_exact:
                rec["exact_retained"] += 1
            rec["labelmasked_valid_to_valid_wrong_cells"] += int(
                float(cur.get("labelmasked_valid_to_valid_wrong_cells", 0.0))
            )
            if cur_close:
                scored_wrong = float(cur.get("labelmasked_wrong_cells", 0.0))
                rec["close_miss_le1_scored_wrong_cells"] += int(scored_wrong <= 1.0)
                rec["close_miss_le2_scored_wrong_cells"] += int(scored_wrong <= 2.0)
                rec["close_miss_le3_scored_wrong_cells"] += int(scored_wrong <= 3.0)
        out.append(rec)
    return out


def per_task_delta_rows(alpha: float, reference_rows: List[Dict[str, str]], rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    row_by_task = {str(row["task_id"]): row for row in rows}
    out = []
    for ref in reference_rows:
        cur = row_by_task[ref["task_id"]]
        out.append(
            {
                "alpha": alpha,
                "task_id": ref["task_id"],
                "puzzle_id": ref["puzzle_id"],
                "bucket": ref["bucket"],
                "ref_exact": ref["exact_accuracy"],
                "alpha_exact": cur["exact_accuracy"],
                "exact_delta": float(cur["exact_accuracy"]) - float(ref["exact_accuracy"]),
                "ref_content": ref["content_accuracy"],
                "alpha_content": cur["content_accuracy"],
                "content_delta": float(cur["content_accuracy"]) - float(ref["content_accuracy"]),
                "ref_outside_fpr": ref["outside_canvas_fpr"],
                "alpha_outside_fpr": cur["outside_canvas_fpr"],
                "outside_fpr_delta": float(cur["outside_canvas_fpr"]) - float(ref["outside_canvas_fpr"]),
                "ref_close_miss": ref["close_miss"],
                "alpha_close_miss": cur["close_miss"],
                "alpha_labelmasked_wrong_cells": cur.get("labelmasked_wrong_cells", ""),
                "alpha_labelmasked_valid_to_valid_wrong_cells": cur.get(
                    "labelmasked_valid_to_valid_wrong_cells",
                    "",
                ),
            }
        )
    return out


def assert_alpha_zero_matches(reference_rows: List[Dict[str, str]], alpha_zero_rows: List[Dict[str, object]]) -> None:
    if len(reference_rows) != len(alpha_zero_rows):
        raise AssertionError(f"Row count mismatch: ref={len(reference_rows)}, alpha0={len(alpha_zero_rows)}")
    row_by_task = {str(row["task_id"]): row for row in alpha_zero_rows}
    max_diff = 0.0
    worst = None
    for ref in reference_rows:
        cur = row_by_task.get(ref["task_id"])
        if cur is None:
            raise AssertionError(f"Missing alpha=0 task: {ref['task_id']}")
        for key in ("task_id", "puzzle_id", "bucket"):
            if str(ref[key]) != str(cur[key]):
                raise AssertionError(f"Alpha=0 {key} mismatch for {ref['task_id']}: {ref[key]} != {cur[key]}")
        for key in METRIC_FIELDS:
            if key not in ref:
                continue
            diff = abs(float(ref[key]) - float(cur[key]))
            if diff > max_diff:
                max_diff = diff
                worst = (ref["task_id"], key, ref[key], cur[key])
    print(f"[alpha0-check] max_diff={max_diff} worst={worst}")
    if max_diff != 0.0:
        raise AssertionError(
            "alpha=0 does not reproduce C0 exactly. "
            f"max_diff={max_diff}, worst={worst}"
        )


def evaluate_alpha(
    alpha: float,
    core_model: torch.nn.Module,
    eval_batches: List[Dict[str, torch.Tensor]],
    config: pretrain.PretrainConfig,
    eval_ids: List[str],
    reference_by_task: Dict[str, Dict[str, str]],
    reference_order: List[str],
    device: torch.device,
) -> List[Dict[str, object]]:
    for cfg in (getattr(core_model, "config", None), getattr(getattr(core_model, "inner", None), "config", None)):
        if cfg is not None and hasattr(cfg, "c2_structure_fusion_alpha"):
            setattr(cfg, "c2_structure_fusion_alpha", float(alpha))
    core_model.eval()

    by_task: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    with torch.inference_mode():
        for batch_idx, cpu_batch in enumerate(eval_batches, start=1):
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            labels = batch["labels"]
            puzzle_ids = batch["puzzle_identifiers"].detach().cpu().numpy().tolist()
            with torch.device(device.type):
                carry = core_model.initial_carry(batch)
            outputs = None
            for _step in range(1, config.arch.halt_max_steps + 1):
                carry, outputs = core_model(carry=carry, batch=batch)
            preds = torch.argmax(outputs["logits"], dim=-1).detach().cpu().numpy()
            raw_labels = labels.detach().cpu().numpy()
            for row_idx, pid in enumerate(puzzle_ids):
                pid = int(pid)
                if pid <= 0 or pid >= len(eval_ids):
                    continue
                task_id = eval_ids[pid]
                if task_id in by_task:
                    continue
                ref = reference_by_task[task_id]
                metrics = row_metrics(preds[row_idx], raw_labels[row_idx], n_steps=config.arch.halt_max_steps)
                row: Dict[str, object] = {
                    "task_id": task_id,
                    "puzzle_id": ref["puzzle_id"],
                    "bucket": ref["bucket"],
                }
                for key in FIELDS[3:]:
                    row[key] = metrics[key]
                row["majority_floor_content"] = ref["majority_floor_content"]
                by_task[task_id] = row
            if batch_idx % 25 == 0:
                print(f"[alpha={alpha}] batches={batch_idx}, tasks={len(by_task)}")
    missing = [task_id for task_id in reference_order if task_id not in by_task]
    if missing:
        raise RuntimeError(f"Missing evaluated tasks for alpha={alpha}: {missing[:10]} count={len(missing)}")
    return [by_task[task_id] for task_id in reference_order]


def decide(summary_rows: List[Dict[str, object]], movement: List[Dict[str, object]]) -> tuple[str, float | None, str]:
    by_alpha = {float(row["alpha"]): row for row in summary_rows}
    gains_losses = {}
    hard_bucket_gains = {}
    for row in movement:
        alpha = float(row["alpha"])
        gains_losses.setdefault(alpha, {"gains": 0, "losses": 0})
        hard_bucket_gains.setdefault(alpha, 0)
        gains_losses[alpha]["gains"] += int(row["exact_gain"])
        gains_losses[alpha]["losses"] += int(row["exact_loss"])
        if row["bucket"] in ("both_fail", "trm_only"):
            hard_bucket_gains[alpha] += int(row["exact_gain"])

    accepted = []
    for alpha, row in by_alpha.items():
        if alpha == 0.0:
            continue
        exact_count = float(row["exact_accuracy"]) * 400.0
        gains = gains_losses[alpha]["gains"]
        losses = gains_losses[alpha]["losses"]
        hard_gains = hard_bucket_gains[alpha]
        strong = exact_count >= 128 and gains > losses
        hard = exact_count >= 125 and hard_gains > 0 and losses <= gains
        if strong or hard:
            accepted.append((exact_count, gains - losses, hard_gains, alpha))
    if accepted:
        accepted.sort(reverse=True)
        alpha = accepted[0][3]
        return (
            "KEEP",
            alpha,
            "Accepted: alpha satisfies exact/hard-bucket gate against C0.",
        )
    return (
        "REJECT",
        None,
        "Rejected: no positive alpha met exact gain/loss and hard-bucket gates.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structure-fusion alpha sweep on a C0 VALID005 PID401 checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-ledger", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alphas", default="0,0.001,0.0025,0.005,0.01,0.02,0.05,0.10")
    parser.add_argument("--global-batch-size", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    reference_path = Path(args.reference_ledger).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = parse_alphas(args.alphas)

    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    raw_config["load_checkpoint"] = str(checkpoint_path)
    raw_config["data_paths"] = ["data/arc-agi-evaluation-full400-seed0"]
    raw_config["data_paths_test"] = []
    raw_config["eval_save_outputs"] = []
    raw_config["dataloader_num_workers"] = 0
    raw_config["checkpoint_path"] = str(out_dir / "noop_checkpoints")
    raw_config["run_name"] = "structfuse_alpha_sweep"
    raw_config["arch"]["c2_structure_fusion_alpha"] = 0.0
    if args.global_batch_size is not None:
        raw_config["global_batch_size"] = int(args.global_batch_size)

    config_copy = out_dir / "alpha_sweep_base_config.yaml"
    config_copy.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    config = pretrain.PretrainConfig(**raw_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this TRM eval path.")

    train_loader, train_metadata = pretrain.create_dataloader(
        config,
        "train",
        0,
        1,
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    eval_loader, _ = pretrain.create_dataloader(
        config,
        "test",
        0,
        1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    del train_loader

    eval_batches = []
    for _set_name, batch, _global_batch_size in eval_loader:
        eval_batches.append({key: value.cpu() for key, value in batch.items()})
    print(f"[setup] cached_eval_batches={len(eval_batches)}")

    loss_head, _, _ = pretrain.create_model(config, train_metadata, rank=0, world_size=1)
    core_model = loss_head.model

    repo_root = Path(__file__).resolve().parents[1]
    eval_ids = json.loads((repo_root / "data" / "arc-agi-evaluation-full400-seed0" / "identifiers.json").read_text(encoding="utf-8"))
    reference_rows = read_csv(reference_path)
    reference_by_task = {row["task_id"]: row for row in reference_rows}
    reference_order = [row["task_id"] for row in reference_rows]

    all_summary = []
    all_movement = []
    all_delta = []
    alpha_zero_rows = None
    for alpha in alphas:
        rows = evaluate_alpha(
            alpha=alpha,
            core_model=core_model,
            eval_batches=eval_batches,
            config=config,
            eval_ids=eval_ids,
            reference_by_task=reference_by_task,
            reference_order=reference_order,
            device=device,
        )
        tag = alpha_tag(alpha)
        ledger_path = out_dir / f"{tag}_17col_ledger.csv"
        solved_path = out_dir / f"{tag}_solved_ids.txt"
        write_csv(ledger_path, rows, FIELDS)
        solved_ids = [str(row["task_id"]) for row in rows if float(row["exact_accuracy"]) > 0]
        solved_path.write_text("\n".join(solved_ids) + ("\n" if solved_ids else ""), encoding="utf-8")
        print(f"[write] {ledger_path}")

        if alpha == 0.0:
            alpha_zero_rows = rows
            assert_alpha_zero_matches(reference_rows, alpha_zero_rows)

        summary = summarize(rows)
        diagnostics = wrong_cell_diagnostics(rows)
        move = movement_rows(alpha, reference_rows, rows)
        delta = per_task_delta_rows(alpha, reference_rows, rows)
        exact_gain = sum(int(row["exact_gain"]) for row in move)
        exact_loss = sum(int(row["exact_loss"]) for row in move)
        both_fail_gain = sum(int(row["exact_gain"]) for row in move if row["bucket"] == "both_fail")
        trm_only_gain = sum(int(row["exact_gain"]) for row in move if row["bucket"] == "trm_only")
        summary_row: Dict[str, object] = {"alpha": alpha}
        summary_row.update(summary)
        summary_row.update(diagnostics)
        summary_row.update(
            {
                "exact_count": int(round(summary["exact_accuracy"] * len(rows))),
                "exact_gain_vs_ref": exact_gain,
                "exact_loss_vs_ref": exact_loss,
                "both_fail_exact_gain_vs_ref": both_fail_gain,
                "trm_only_exact_gain_vs_ref": trm_only_gain,
            }
        )
        all_summary.append(summary_row)
        all_movement.extend(move)
        all_delta.extend(delta)

    if alpha_zero_rows is None:
        raise RuntimeError("Internal error: alpha=0 rows were not evaluated.")

    summary_fields = [
        "alpha",
        *SUMMARY_METRICS,
        "total_labelmasked_valid_to_valid_wrong_cells",
        "close_miss_le1_scored_wrong_cells",
        "close_miss_le2_scored_wrong_cells",
        "close_miss_le3_scored_wrong_cells",
        "exact_count",
        "exact_gain_vs_ref",
        "exact_loss_vs_ref",
        "both_fail_exact_gain_vs_ref",
        "trm_only_exact_gain_vs_ref",
    ]
    write_csv(out_dir / "alpha_sweep_summary.csv", all_summary, summary_fields)
    write_csv(out_dir / "alpha_sweep_task_movement.csv", all_movement, list(all_movement[0].keys()))
    write_csv(out_dir / "alpha_sweep_per_task_delta.csv", all_delta, list(all_delta[0].keys()))

    verdict, best_alpha, reason = decide(all_summary, all_movement)
    report = [
        f"verdict: {verdict}",
        f"best_alpha: {best_alpha}",
        f"reason: {reason}",
        "",
        "reference: C0 VALID005 EOS0 AUG1000 seed0",
        "target: both_fail exact conversion without solved-task loss",
        "",
        "summary:",
    ]
    for row in all_summary:
        report.append(
            "alpha={alpha}: exact={exact_count}/400, gains={exact_gain_vs_ref}, "
            "losses={exact_loss_vs_ref}, both_fail_gains={both_fail_exact_gain_vs_ref}, "
            "trm_only_gains={trm_only_exact_gain_vs_ref}, outside_fpr={outside_canvas_fpr:.6f}, "
            "close_miss={close_miss:.6f}".format(**row)
        )
    report_path = out_dir / "rejection_or_keep.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[write] {out_dir / 'alpha_sweep_summary.csv'}")
    print(f"[write] {out_dir / 'alpha_sweep_task_movement.csv'}")
    print(f"[write] {out_dir / 'alpha_sweep_per_task_delta.csv'}")
    print(f"[write] {report_path}")
    print("\n".join(report))


if __name__ == "__main__":
    main()
