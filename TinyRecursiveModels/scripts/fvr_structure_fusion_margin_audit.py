import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
from scripts.fvr_structfuse_alpha_sweep import read_csv, write_csv


EPS = 1e-12
INF = 1.0e30


def finite(value: float) -> str:
    if value <= -INF / 2:
        return "-inf"
    if value >= INF / 2:
        return "inf"
    return f"{value:.10g}"


def alpha_tag(value: float | None) -> str:
    if value is None:
        return ""
    return finite(value)


def intersect_constraint(lo: float, hi: float, margin: float, delta: float, want_true_wins: bool) -> Tuple[float, float, bool]:
    """Intersect alpha interval with margin + alpha * delta >= 0."""
    if abs(delta) < EPS:
        return lo, hi, margin >= -EPS
    threshold = -margin / delta
    if delta > 0:
        lo = max(lo, threshold)
    else:
        hi = min(hi, threshold)
    return lo, hi, lo <= hi


def safety_interval_for_correct_cell(logits: np.ndarray, bias: np.ndarray, true_class: int) -> Tuple[float, float, bool]:
    lo, hi = -INF, INF
    ly = float(logits[true_class])
    by = float(bias[true_class])
    for cls in range(logits.shape[-1]):
        if cls == true_class:
            continue
        margin = ly - float(logits[cls])
        delta = by - float(bias[cls])
        lo, hi, ok = intersect_constraint(lo, hi, margin, delta, want_true_wins=True)
        if not ok:
            return lo, hi, False
    return lo, hi, True


def correction_interval_for_wrong_cell(logits: np.ndarray, bias: np.ndarray, true_class: int) -> Tuple[float, float, bool]:
    lo, hi = -INF, INF
    ly = float(logits[true_class])
    by = float(bias[true_class])
    for cls in range(logits.shape[-1]):
        if cls == true_class:
            continue
        margin = ly - float(logits[cls])
        delta = by - float(bias[cls])
        lo, hi, ok = intersect_constraint(lo, hi, margin, delta, want_true_wins=True)
        if not ok:
            return lo, hi, False
    return lo, hi, True


def interval_intersection(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return max(a[0], b[0]), min(a[1], b[1])


def has_positive(interval: Tuple[float, float]) -> bool:
    lo, hi = interval
    return lo <= hi and hi > max(lo, 0.0)


def has_negative(interval: Tuple[float, float]) -> bool:
    lo, hi = interval
    return lo <= hi and lo < min(hi, 0.0)


def structure_bias(structure_logits: torch.Tensor, vocab_size: int) -> torch.Tensor:
    valid_ref = structure_logits[..., 2:3].to(torch.float32)
    bias = torch.zeros((*structure_logits.shape[:-1], vocab_size), device=structure_logits.device, dtype=torch.float32)
    bias[..., 0:1] = structure_logits[..., 0:1].to(torch.float32) - valid_ref
    bias[..., 1:2] = structure_logits[..., 1:2].to(torch.float32) - valid_ref
    return bias


def load_config(config_path: Path, checkpoint_path: Path, out_dir: Path, batch_size: int) -> pretrain.PretrainConfig:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["load_checkpoint"] = str(checkpoint_path)
    raw_config["data_paths"] = ["data/arc-agi-evaluation-full400-seed0"]
    raw_config["data_paths_test"] = []
    raw_config["eval_save_outputs"] = []
    raw_config["dataloader_num_workers"] = 0
    raw_config["checkpoint_path"] = str(out_dir / "noop_checkpoints")
    raw_config["run_name"] = "structure_fusion_margin_audit"
    raw_config["global_batch_size"] = int(batch_size)
    raw_config.setdefault("arch", {})["c2_structure_fusion_alpha"] = 0.0
    (out_dir / "margin_audit_config.yaml").write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    return pretrain.PretrainConfig(**raw_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether structure fusion margins can correct C0 close-miss tasks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-ledger", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--global-batch-size", type=int, default=8)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    reference_path = Path(args.reference_ledger).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_rows = read_csv(reference_path)
    reference_by_task = {row["task_id"]: row for row in reference_rows}
    close_miss_tasks = {
        row["task_id"]
        for row in reference_rows
        if float(row["exact_accuracy"]) == 0.0 and float(row["close_miss"]) > 0.0
    }
    exact_tasks = {
        row["task_id"]
        for row in reference_rows
        if float(row["exact_accuracy"]) > 0.0
    }
    if not close_miss_tasks:
        raise RuntimeError("No close-miss tasks found in reference ledger.")

    config = load_config(config_path, checkpoint_path, out_dir, args.global_batch_size)
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

    loss_head, _, _ = pretrain.create_model(config, train_metadata, rank=0, world_size=1)
    core_model = loss_head.model
    core_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this TRM eval path.")

    repo_root = Path(__file__).resolve().parents[1]
    eval_ids = json.loads((repo_root / "data" / "arc-agi-evaluation-full400-seed0" / "identifiers.json").read_text(encoding="utf-8"))
    vocab_size = int(config.arch.__pydantic_extra__.get("vocab_size", 12) or 12)
    halt_max_steps = int(config.arch.__pydantic_extra__.get("halt_max_steps", 16))

    global_exact_safe = (-INF, INF)
    exact_safe_ok = True
    task_rows: List[Dict[str, object]] = []
    wrong_rows: List[Dict[str, object]] = []
    seen_close = set()
    seen_exact = set()

    with torch.inference_mode():
        for batch_idx, (_set_name, cpu_batch, _global_batch_size) in enumerate(eval_loader, start=1):
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            with torch.device(device.type):
                carry = core_model.initial_carry(batch)
            outputs = None
            for _step in range(1, halt_max_steps + 1):
                carry, outputs = core_model(carry=carry, batch=batch)
            if outputs is None or "c2_structure_logits" not in outputs:
                raise RuntimeError("Model output does not contain c2_structure_logits; c2_geometry_aux_head must be true.")

            logits_t = outputs["logits"].detach().to(torch.float32)
            struct_t = outputs["c2_structure_logits"].detach().to(torch.float32)
            bias_t = structure_bias(struct_t, vocab_size=logits_t.shape[-1])
            preds_t = torch.argmax(logits_t, dim=-1)
            labels_t = batch["labels"].detach()
            pids = batch["puzzle_identifiers"].detach().cpu().numpy().tolist()

            logits_np = logits_t.cpu().numpy()
            bias_np = bias_t.cpu().numpy()
            preds_np = preds_t.cpu().numpy()
            labels_np = labels_t.cpu().numpy()

            for row_idx, pid in enumerate(pids):
                pid = int(pid)
                if pid <= 0 or pid >= len(eval_ids):
                    continue
                task_id = eval_ids[pid]
                if task_id not in reference_by_task:
                    continue
                raw_label = labels_np[row_idx]
                label_mask = raw_label != IGNORE_LABEL_ID
                label_seq = np.where(label_mask, raw_label, 0).astype(np.int64)
                pred_seq = preds_np[row_idx].astype(np.int64)
                logits = logits_np[row_idx]
                bias = bias_np[row_idx]

                if task_id in exact_tasks and task_id not in seen_exact:
                    seen_exact.add(task_id)
                    for pos in np.where(label_mask & (pred_seq == label_seq))[0]:
                        lo, hi, ok = safety_interval_for_correct_cell(logits[pos], bias[pos], int(label_seq[pos]))
                        if not ok:
                            exact_safe_ok = False
                        global_exact_safe = interval_intersection(global_exact_safe, (lo, hi))

                if task_id not in close_miss_tasks or task_id in seen_close:
                    continue
                seen_close.add(task_id)
                ref = reference_by_task[task_id]
                wrong_positions = np.where(pred_seq != label_seq)[0]
                label_wrong_positions = np.where(label_mask & (pred_seq != label_seq))[0]

                wrong_interval = (-INF, INF)
                same_task_safe_interval = (-INF, INF)
                correctable_any = 0
                correctable_pos = 0
                correctable_neg = 0
                zero_delta_wrong = 0
                impossible_wrong = 0
                pos_direction = 0
                neg_direction = 0

                for pos in np.where(label_mask & (pred_seq == label_seq))[0]:
                    lo, hi, ok = safety_interval_for_correct_cell(logits[pos], bias[pos], int(label_seq[pos]))
                    if ok:
                        same_task_safe_interval = interval_intersection(same_task_safe_interval, (lo, hi))

                for pos in wrong_positions:
                    y = int(label_seq[pos])
                    pred = int(pred_seq[pos])
                    margin = float(logits[pos, y] - logits[pos, pred])
                    delta_margin = float(bias[pos, y] - bias[pos, pred])
                    if delta_margin > EPS:
                        pos_direction += 1
                    elif delta_margin < -EPS:
                        neg_direction += 1
                    else:
                        zero_delta_wrong += 1

                    lo, hi, ok = correction_interval_for_wrong_cell(logits[pos], bias[pos], y)
                    if not ok:
                        impossible_wrong += 1
                    else:
                        cell_interval = (lo, hi)
                        if has_positive(cell_interval):
                            correctable_pos += 1
                        if has_negative(cell_interval):
                            correctable_neg += 1
                        if has_positive(cell_interval) or has_negative(cell_interval):
                            correctable_any += 1
                        wrong_interval = interval_intersection(wrong_interval, cell_interval)

                    wrong_rows.append(
                        {
                            "task_id": task_id,
                            "bucket": ref["bucket"],
                            "position": int(pos),
                            "label_mask": int(bool(label_mask[pos])),
                            "true_class": y,
                            "pred_class": pred,
                            "lm_margin_true_minus_pred": margin,
                            "structure_delta_margin_true_minus_pred": delta_margin,
                            "cell_alpha_lo": finite(lo),
                            "cell_alpha_hi": finite(hi),
                            "cell_has_positive_alpha": int(ok and has_positive((lo, hi))),
                            "cell_has_negative_alpha": int(ok and has_negative((lo, hi))),
                            "cell_impossible": int(not ok),
                        }
                    )

                all_wrong_cells_structurally_correctable = (
                    wrong_positions.size > 0
                    and impossible_wrong == 0
                    and wrong_interval[0] <= wrong_interval[1]
                )
                task_correction_interval = wrong_interval if all_wrong_cells_structurally_correctable else (1.0, 0.0)
                task_full_interval = interval_intersection(task_correction_interval, same_task_safe_interval)
                task_full_interval = interval_intersection(task_full_interval, global_exact_safe)
                task_rows.append(
                    {
                        "task_id": task_id,
                        "bucket": ref["bucket"],
                        "wrong_cells_all900": int(wrong_positions.size),
                        "wrong_cells_labelmask": int(label_wrong_positions.size),
                        "wrong_cells_with_any_structural_interval": int(correctable_any),
                        "wrong_cells_positive_direction_vs_pred": int(pos_direction),
                        "wrong_cells_negative_direction_vs_pred": int(neg_direction),
                        "wrong_cells_zero_delta_vs_pred": int(zero_delta_wrong),
                        "wrong_cells_impossible_vs_all_classes": int(impossible_wrong),
                        "all_wrong_cells_structurally_correctable": int(all_wrong_cells_structurally_correctable),
                        "wrong_interval_lo": finite(task_correction_interval[0]),
                        "wrong_interval_hi": finite(task_correction_interval[1]),
                        "same_task_safe_interval_lo": finite(same_task_safe_interval[0]),
                        "same_task_safe_interval_hi": finite(same_task_safe_interval[1]),
                        "full_safe_interval_lo": finite(task_full_interval[0]),
                        "full_safe_interval_hi": finite(task_full_interval[1]),
                        "feasible_positive_alpha": int(has_positive(task_full_interval)),
                        "feasible_negative_alpha": int(has_negative(task_full_interval)),
                    }
                )

            if batch_idx % 25 == 0:
                print(f"[audit] batches={batch_idx}, close_seen={len(seen_close)}, exact_seen={len(seen_exact)}")

    missing_close = sorted(close_miss_tasks - seen_close)
    missing_exact = sorted(exact_tasks - seen_exact)
    if missing_close:
        raise RuntimeError(f"Missing close-miss tasks from eval: {missing_close[:10]} count={len(missing_close)}")
    if missing_exact:
        raise RuntimeError(f"Missing exact tasks from eval: {missing_exact[:10]} count={len(missing_exact)}")

    write_csv(out_dir / "margin_audit_task_summary.csv", task_rows, list(task_rows[0].keys()))
    write_csv(out_dir / "margin_audit_wrong_cells.csv", wrong_rows, list(wrong_rows[0].keys()))

    by_bucket: Dict[str, Counter] = defaultdict(Counter)
    for row in task_rows:
        bucket = str(row["bucket"])
        by_bucket[bucket]["close_miss_tasks"] += 1
        by_bucket[bucket]["tasks_with_any_structurally_correctable_wrong_cell"] += int(row["wrong_cells_with_any_structural_interval"]) > 0
        by_bucket[bucket]["tasks_all_wrong_cells_structurally_correctable"] += int(row["all_wrong_cells_structurally_correctable"]) > 0
        by_bucket[bucket]["tasks_feasible_positive_alpha"] += int(row["feasible_positive_alpha"]) > 0
        by_bucket[bucket]["tasks_feasible_negative_alpha"] += int(row["feasible_negative_alpha"]) > 0

    bucket_rows = []
    for bucket in sorted(by_bucket):
        rec = {"bucket": bucket}
        rec.update(by_bucket[bucket])
        bucket_rows.append(rec)
    write_csv(out_dir / "margin_audit_bucket_summary.csv", bucket_rows, list(bucket_rows[0].keys()))

    total_close = len(task_rows)
    any_correctable = sum(int(row["wrong_cells_with_any_structural_interval"]) > 0 for row in task_rows)
    all_correctable = sum(int(row["all_wrong_cells_structurally_correctable"]) > 0 for row in task_rows)
    pos_feasible = sum(int(row["feasible_positive_alpha"]) > 0 for row in task_rows)
    neg_feasible = sum(int(row["feasible_negative_alpha"]) > 0 for row in task_rows)
    hard_pos = sum(
        int(row["feasible_positive_alpha"]) > 0 and row["bucket"] in ("both_fail", "trm_only")
        for row in task_rows
    )
    hard_neg = sum(
        int(row["feasible_negative_alpha"]) > 0 and row["bucket"] in ("both_fail", "trm_only")
        for row in task_rows
    )

    verdict = "REJECT"
    reason = "No feasible scalar-alpha correction interval found for at least 2 hard-bucket close-miss tasks."
    if max(hard_pos, hard_neg) >= 2:
        verdict = "INSPECT"
        reason = "At least 2 hard-bucket close-miss tasks have a feasible scalar-alpha interval; inspect task rows before training."

    report = [
        f"verdict: {verdict}",
        f"reason: {reason}",
        "",
        f"global_exact_safe_interval: [{finite(global_exact_safe[0])}, {finite(global_exact_safe[1])}]",
        f"global_exact_safe_ok: {int(exact_safe_ok and global_exact_safe[0] <= global_exact_safe[1])}",
        "",
        f"close_miss_tasks: {total_close}",
        f"tasks_with_at_least_one_structurally_correctable_wrong_cell: {any_correctable}",
        f"tasks_where_all_wrong_cells_structurally_correctable: {all_correctable}",
        f"tasks_with_feasible_positive_alpha_interval: {pos_feasible}",
        f"tasks_with_feasible_negative_alpha_interval: {neg_feasible}",
        f"hard_bucket_feasible_positive_alpha_tasks: {hard_pos}",
        f"hard_bucket_feasible_negative_alpha_tasks: {hard_neg}",
        "",
        "by_bucket:",
    ]
    for row in bucket_rows:
        report.append(
            "{bucket}: close={close_miss_tasks}, any_cell={tasks_with_any_structurally_correctable_wrong_cell}, "
            "all_cells={tasks_all_wrong_cells_structurally_correctable}, pos_feasible={tasks_feasible_positive_alpha}, "
            "neg_feasible={tasks_feasible_negative_alpha}".format(**row)
        )
    (out_dir / "margin_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
