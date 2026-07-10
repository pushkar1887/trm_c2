import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain
from scripts.fvr_structfuse_alpha_sweep import (
    BUCKETS,
    FIELDS,
    SUMMARY_METRICS,
    evaluate_alpha,
    movement_rows,
    per_task_delta_rows,
    read_csv,
    summarize,
    wrong_cell_diagnostics,
    write_csv,
)


def decide(reference_rows: List[Dict[str, str]], rows: List[Dict[str, object]], movement: List[Dict[str, object]]) -> tuple[str, str, str]:
    ref_exact = sum(float(row["exact_accuracy"]) for row in reference_rows)
    cur_exact = sum(float(row["exact_accuracy"]) for row in rows)
    exact_gain = sum(int(row["exact_gain"]) for row in movement)
    exact_loss = sum(int(row["exact_loss"]) for row in movement)
    both_fail_gain = sum(int(row["exact_gain"]) for row in movement if row["bucket"] == "both_fail")
    both_fail_loss = sum(int(row["exact_loss"]) for row in movement if row["bucket"] == "both_fail")

    if cur_exact > ref_exact:
        return "KEEP", "exact total beats C0 parent.", "Promote this setting as current parent and rerun both-fail diagnostics."
    if both_fail_gain > 0 and exact_loss <= exact_gain:
        return "KEEP", "both_fail exact gains exist without net exact loss.", "Promote cautiously and inspect gained/lost task identities."
    if both_fail_gain > 0:
        return "REJECT", "both_fail gains are outweighed by exact losses.", "Do not continue this setting; inspect loss cases before Stage 4."
    return "REJECT", "no both_fail exact conversion and no exact improvement.", "Proceed to the next gated stage."


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PID401 checkpoint into a 17-column ledger and compare to C0.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-ledger", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    reference_path = Path(args.reference_ledger).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = Path(args.dataset) if args.dataset else Path("data/arc-agi-evaluation-full400-seed0")
    dataset_abs = dataset_path if dataset_path.is_absolute() else repo_root / dataset_path
    if not dataset_abs.exists() and args.dataset is None:
        fallback = repo_root / "data" / "arc1concept-aug-0"
        if fallback.exists():
            dataset_path = Path("data/arc1concept-aug-0")
            dataset_abs = fallback
    if not dataset_abs.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {dataset_abs}")

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["load_checkpoint"] = str(checkpoint_path)
    raw_config["data_paths"] = [str(dataset_path)]
    raw_config["data_paths_test"] = []
    raw_config["eval_save_outputs"] = []
    raw_config["dataloader_num_workers"] = 0
    raw_config["checkpoint_path"] = str(out_dir / "noop_checkpoints")
    raw_config["run_name"] = args.run_label
    raw_config["global_batch_size"] = int(args.global_batch_size)
    raw_config.setdefault("arch", {})["c2_structure_fusion_alpha"] = 0.0
    (out_dir / "eval_config.yaml").write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    config = pretrain.PretrainConfig(**raw_config)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this TRM eval path.")

    eval_ids = json.loads((dataset_abs / "identifiers.json").read_text(encoding="utf-8"))
    reference_rows = read_csv(reference_path)
    reference_by_task = {row["task_id"]: row for row in reference_rows}
    reference_order = [row["task_id"] for row in reference_rows]

    rows = evaluate_alpha(
        alpha=0.0,
        core_model=core_model,
        eval_batches=eval_batches,
        config=config,
        eval_ids=eval_ids,
        reference_by_task=reference_by_task,
        reference_order=reference_order,
        device=device,
    )
    ledger_path = out_dir / f"{args.run_label}_17col_ledger.csv"
    write_csv(ledger_path, rows, FIELDS)
    solved_ids = [str(row["task_id"]) for row in rows if float(row["exact_accuracy"]) > 0]
    (out_dir / "solved_ids.txt").write_text("\n".join(solved_ids) + ("\n" if solved_ids else ""), encoding="utf-8")

    summary = summarize(rows)
    diagnostics = wrong_cell_diagnostics(rows)
    movement = movement_rows(0.0, reference_rows, rows)
    delta = per_task_delta_rows(0.0, reference_rows, rows)
    exact_gain = sum(int(row["exact_gain"]) for row in movement)
    exact_loss = sum(int(row["exact_loss"]) for row in movement)
    both_fail_gain = sum(int(row["exact_gain"]) for row in movement if row["bucket"] == "both_fail")
    both_fail_loss = sum(int(row["exact_loss"]) for row in movement if row["bucket"] == "both_fail")
    close_loss = sum(int(row["closemiss_loss"]) for row in movement)
    close_gain = sum(int(row["closemiss_gain"]) for row in movement)

    summary_row: Dict[str, object] = {
        "run_label": args.run_label,
        **summary,
        **diagnostics,
        "exact_count": int(round(summary["exact_accuracy"] * len(rows))),
        "exact_gain_vs_c0": exact_gain,
        "exact_loss_vs_c0": exact_loss,
        "both_fail_exact_gain_vs_c0": both_fail_gain,
        "both_fail_exact_loss_vs_c0": both_fail_loss,
        "closemiss_gain_vs_c0": close_gain,
        "closemiss_loss_vs_c0": close_loss,
    }
    write_csv(out_dir / "summary_vs_c0.csv", [summary_row], list(summary_row.keys()))
    write_csv(out_dir / "bucket_movement.csv", movement, list(movement[0].keys()))
    write_csv(out_dir / "per_task_delta.csv", delta, list(delta[0].keys()))

    verdict, reason, next_stage = decide(reference_rows, rows, movement)
    ref_outside = sum(float(row["outside_canvas_fpr"]) for row in reference_rows) / max(len(reference_rows), 1)
    report = [
        f"verdict: {verdict}",
        f"exact gained: {exact_gain}",
        f"exact lost: {exact_loss}",
        f"both_fail exact gained: {both_fail_gain}",
        f"both_fail exact lost: {both_fail_loss}",
        f"close_miss gained/lost: {close_gain}/{close_loss}",
        f"outside_fpr change: {summary['outside_canvas_fpr'] - ref_outside:.6f}",
        f"total_labelmasked_valid_to_valid_wrong_cells: {diagnostics['total_labelmasked_valid_to_valid_wrong_cells']}",
        "close_miss <=1/<=2/<=3 scored wrong cells: "
        f"{diagnostics['close_miss_le1_scored_wrong_cells']}/"
        f"{diagnostics['close_miss_le2_scored_wrong_cells']}/"
        f"{diagnostics['close_miss_le3_scored_wrong_cells']}",
        f"reason: {reason}",
        f"next stage: {next_stage}",
        "",
        "summary:",
        f"exact: {summary_row['exact_count']}/400",
    ]
    for key in SUMMARY_METRICS:
        report.append(f"{key}: {summary[key]:.6f}")
    (out_dir / "rejection_or_keep.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
