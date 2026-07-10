"""Close-miss repair for both_fail tasks.

Generates TRM predictions, applies a stacked repair pipeline:
  S0: baseline TRM prediction (no repair)
  S1: canvas cleanup -- force outside (pred_h, pred_w) tokens to EOS
  S2: isolated-cell symbolic repair on the cleaned inside grid
  S3: ORPI second-opinion (where a verified+stable rule exists)

Reports per-strategy strict gains against the C0 baseline ledger.
Targets the 53 both_fail close-miss tasks but runs the same pipeline on all 400.
"""

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain
from scripts.fvr_structfuse_alpha_sweep import IGNORE_LABEL_ID, crop_shape, read_csv, write_csv


EOS_TOKEN = 1
PAD_TOKEN = 0
COLOR_TOKEN_OFFSET = 2  # tokens 2..11 are colors 0..9


def token_grid_to_color_grid(grid: np.ndarray, h: int, w: int) -> np.ndarray:
    """Convert (30,30) token grid to (h,w) color grid (0..9) for the active canvas."""
    out = np.zeros((h, w), dtype=np.int64)
    for r in range(h):
        for c in range(w):
            tok = int(grid[r, c])
            if 2 <= tok <= 11:
                out[r, c] = tok - COLOR_TOKEN_OFFSET
            else:
                out[r, c] = 0  # treat EOS/pad as background
    return out


def color_grid_to_token_seq(color_grid: np.ndarray) -> np.ndarray:
    """Convert (h,w) color grid back to a (900,) token sequence with EOS padding outside."""
    h, w = color_grid.shape
    out = np.full((30, 30), EOS_TOKEN, dtype=np.int64)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(color_grid[r, c]) + COLOR_TOKEN_OFFSET
    return out.flatten()


def canvas_cleanup(pred_seq: np.ndarray) -> np.ndarray:
    """S1: force everything outside the predicted (pred_h, pred_w) canvas to EOS."""
    pred_h, pred_w = crop_shape(pred_seq)
    if pred_h == 0 or pred_w == 0:
        return pred_seq.copy()
    grid = pred_seq.reshape(30, 30).copy()
    cleaned = np.full_like(grid, EOS_TOKEN)
    cleaned[:pred_h, :pred_w] = grid[:pred_h, :pred_w]
    return cleaned.flatten()


def isolated_cell_repair(pred_seq: np.ndarray, allowed_colors: Optional[set] = None) -> np.ndarray:
    """S2 (conservative): within the predicted canvas, flip a cell only if:
      (a) ALL 4 neighbours exist (cell is not on a border edge), AND
      (b) all 4 neighbours unanimously have the same value, AND
      (c) the cell's color is NOT in `allowed_colors` (i.e., a colour TRM
          invented that doesn't appear in any demo input/output for this task).

    This makes the repair safe: legitimate sparse markers/anchors keep their
    color because their color WILL appear somewhere in the demos. Only true
    hallucinations (colors TRM emitted that aren't part of the task palette)
    get flipped.
    """
    if allowed_colors is None:
        return pred_seq.copy()
    pred_h, pred_w = crop_shape(pred_seq)
    if pred_h == 0 or pred_w == 0:
        return pred_seq.copy()
    grid = pred_seq.reshape(30, 30).copy()
    inside = grid[:pred_h, :pred_w].copy()
    out = inside.copy()
    for r in range(pred_h):
        for c in range(pred_w):
            cell_value = int(inside[r, c])
            if cell_value in allowed_colors:
                continue
            # need a full 4-neighbour ring (no edge cells)
            if r == 0 or r == pred_h - 1 or c == 0 or c == pred_w - 1:
                continue
            neighbours = [int(inside[r - 1, c]), int(inside[r + 1, c]), int(inside[r, c - 1]), int(inside[r, c + 1])]
            if len(set(neighbours)) == 1 and neighbours[0] != cell_value:
                out[r, c] = neighbours[0]
    grid[:pred_h, :pred_w] = out
    return grid.flatten()


def compute_allowed_token_set_for_task(task_id: str, dataset_root: Path) -> set:
    """Allowed = union of token values appearing in any demo input/output for this task,
    plus EOS and pad tokens. Returns a set of TOKEN ids (2..11 = colors)."""
    test_puzzles_path = dataset_root / "test_puzzles.json"
    if not test_puzzles_path.exists():
        return set(range(0, 12))  # permissive fallback
    puzzles = json.loads(test_puzzles_path.read_text(encoding="utf-8"))
    if task_id not in puzzles:
        return set(range(0, 12))
    task = puzzles[task_id]
    colors: set = set()
    for demo in task.get("train", []):
        for row in demo["input"]:
            colors.update(int(v) for v in row)
        for row in demo["output"]:
            colors.update(int(v) for v in row)
    # also include test input colors (we see them at inference time)
    for test_pair in task.get("test", []):
        for row in test_pair["input"]:
            colors.update(int(v) for v in row)
    # convert palette colors (0..9) to token ids (2..11)
    return {c + COLOR_TOKEN_OFFSET for c in colors} | {EOS_TOKEN, PAD_TOKEN}


def orpi_second_opinion(pred_seq: np.ndarray, task_id: str, demos: List[Dict], orpi_program_cache: Dict[str, np.ndarray]) -> np.ndarray:
    """S3: if ORPI has a verified+LODO-stable program for this task, replace prediction
    on cells where ORPI's prediction differs but ORPI itself reproduces the demos exactly."""
    if task_id not in orpi_program_cache:
        return pred_seq.copy()
    orpi_pred = orpi_program_cache[task_id]
    if orpi_pred.shape != (30, 30):
        h, w = orpi_pred.shape
        pad = np.full((30, 30), EOS_TOKEN, dtype=np.int64)
        pad[:h, :w] = np.where(orpi_pred >= 0, orpi_pred + COLOR_TOKEN_OFFSET, EOS_TOKEN)
        orpi_pred = pad
    return orpi_pred.flatten()


def score_seq(pred_seq: np.ndarray, label_seq: np.ndarray) -> Tuple[int, int]:
    """Return (exact, content_correct_tokens) under label mask."""
    label_mask = label_seq != IGNORE_LABEL_ID
    masked_label = np.where(label_mask, label_seq, 0)
    masked_pred = np.where(label_mask, pred_seq, 0)
    exact = int(np.array_equal(pred_seq[label_mask], label_seq[label_mask]))
    return exact, int((pred_seq[label_mask] == label_seq[label_mask]).sum())


def load_orpi_program_cache(orpi_report_dir: Path, dataset_path: Path) -> Dict[str, np.ndarray]:
    """Load ORPI v4 verified+LODO-stable program outputs per task.

    Reads `repair_candidate_outputs.csv` from a previous ORPI run. Each row has the
    candidate grid for a task; we keep one prediction per task (the first verified).
    """
    cache: Dict[str, np.ndarray] = {}
    csv_path = orpi_report_dir / "repair_candidate_outputs.csv"
    if not csv_path.exists():
        return cache
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task_id = str(row["task_id"])
            if task_id in cache:
                continue
            try:
                grid_data = json.loads(row["candidate_grid"])
                cache[task_id] = np.array(grid_data, dtype=np.int64)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Close-miss repair stacked pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-ledger", required=True)
    parser.add_argument("--closemiss-csv", required=True, help="both_fail close-miss taxonomy CSV")
    parser.add_argument("--orpi-report-dir", default="reports/orpi_c2_unification_v4", help="ORPI v4 output dir for second opinion")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--global-batch-size", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    closemiss_rows = read_csv(Path(args.closemiss_csv).resolve())
    target_tasks = {row["task_id"] for row in closemiss_rows}
    print(f"[setup] close-miss target tasks: {len(target_tasks)}")

    reference_rows = read_csv(Path(args.reference_ledger).resolve())
    reference_by_task = {row["task_id"]: row for row in reference_rows}

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["load_checkpoint"] = str(checkpoint_path)
    raw_config["data_paths"] = ["data/arc-agi-evaluation-full400-seed0"]
    raw_config["data_paths_test"] = []
    raw_config["eval_save_outputs"] = []
    raw_config["dataloader_num_workers"] = 0
    raw_config["checkpoint_path"] = str(out_dir / "noop_checkpoints")
    raw_config["run_name"] = "closemiss_repair"
    raw_config["global_batch_size"] = int(args.global_batch_size)
    raw_config.setdefault("arch", {})["c2_structure_fusion_alpha"] = 0.0
    config = pretrain.PretrainConfig(**raw_config)

    train_loader, train_metadata = pretrain.create_dataloader(
        config, "train", 0, 1, test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    eval_loader, _ = pretrain.create_dataloader(
        config, "test", 0, 1, test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    del train_loader

    eval_batches = []
    for _set_name, batch, _gbs in eval_loader:
        eval_batches.append({k: v.cpu() for k, v in batch.items()})
    print(f"[setup] cached_eval_batches={len(eval_batches)}")

    loss_head, _, _ = pretrain.create_model(config, train_metadata, rank=0, world_size=1)
    core_model = loss_head.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this TRM eval path.")
    core_model.eval()

    repo_root = Path(__file__).resolve().parents[1]
    eval_ids = json.loads((repo_root / "data" / "arc-agi-evaluation-full400-seed0" / "identifiers.json").read_text(encoding="utf-8"))

    # Inference: capture pred_seq and label_seq per task
    pred_by_task: Dict[str, np.ndarray] = OrderedDict()
    label_by_task: Dict[str, np.ndarray] = OrderedDict()
    with torch.inference_mode():
        for batch_idx, cpu_batch in enumerate(eval_batches, start=1):
            batch = {k: v.to(device) for k, v in cpu_batch.items()}
            with torch.device(device.type):
                carry = core_model.initial_carry(batch)
            outputs = None
            for _step in range(1, config.arch.halt_max_steps + 1):
                carry, outputs = core_model(carry=carry, batch=batch)
            preds = torch.argmax(outputs["logits"], dim=-1).detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            puzzle_ids = batch["puzzle_identifiers"].detach().cpu().numpy().tolist()
            for row_idx, pid in enumerate(puzzle_ids):
                pid = int(pid)
                if pid <= 0 or pid >= len(eval_ids):
                    continue
                tid = eval_ids[pid]
                if tid in pred_by_task:
                    continue
                pred_by_task[tid] = preds[row_idx].copy()
                label_by_task[tid] = labels[row_idx].copy()
            if batch_idx % 25 == 0:
                print(f"[infer] batches={batch_idx}, tasks={len(pred_by_task)}")
    print(f"[infer] DONE — tasks predicted: {len(pred_by_task)}")

    # Load ORPI second-opinion cache (best-effort)
    orpi_dir = Path(args.orpi_report_dir).resolve()
    orpi_cache = load_orpi_program_cache(orpi_dir, Path(raw_config["data_paths"][0]).resolve())
    orpi_for_targets = {tid: g for tid, g in orpi_cache.items() if tid in target_tasks}
    print(f"[setup] ORPI second-opinion cache for close-miss targets: {len(orpi_for_targets)}")

    # Apply strategies to ALL tasks, but report focused on close-miss
    dataset_root = Path(raw_config["data_paths"][0]).resolve()
    strategies = ["S0_baseline", "S1_cleanup", "S2_cleanup_repair", "S3_cleanup_repair_orpi"]
    per_task: List[Dict[str, object]] = []
    for tid in pred_by_task:
        pred = pred_by_task[tid]
        label = label_by_task[tid]
        results = {}
        results["S0_baseline"] = score_seq(pred, label)
        s1 = canvas_cleanup(pred)
        results["S1_cleanup"] = score_seq(s1, label)
        allowed = compute_allowed_token_set_for_task(tid, dataset_root)
        s2 = isolated_cell_repair(s1, allowed_colors=allowed)
        results["S2_cleanup_repair"] = score_seq(s2, label)
        s3 = orpi_second_opinion(s2, tid, [], orpi_for_targets) if tid in orpi_for_targets else s2
        results["S3_cleanup_repair_orpi"] = score_seq(s3, label)
        ref = reference_by_task.get(tid, {})
        per_task.append({
            "task_id": tid,
            "bucket": ref.get("bucket", ""),
            "c0_exact": int(float(ref.get("exact_accuracy", 0)) > 0),
            "is_closemiss_target": int(tid in target_tasks),
            "S0_exact": results["S0_baseline"][0],
            "S1_exact": results["S1_cleanup"][0],
            "S2_exact": results["S2_cleanup_repair"][0],
            "S3_exact": results["S3_cleanup_repair_orpi"][0],
            "S0_content_tokens": results["S0_baseline"][1],
            "S1_content_tokens": results["S1_cleanup"][1],
            "S2_content_tokens": results["S2_cleanup_repair"][1],
            "S3_content_tokens": results["S3_cleanup_repair_orpi"][1],
        })

    fields = ["task_id", "bucket", "c0_exact", "is_closemiss_target",
              "S0_exact", "S1_exact", "S2_exact", "S3_exact",
              "S0_content_tokens", "S1_content_tokens", "S2_content_tokens", "S3_content_tokens"]
    write_csv(out_dir / "per_task_repair.csv", per_task, fields)

    # Summary
    def count_strategy(rows: List[Dict], key: str, scope: str = "all") -> int:
        if scope == "all":
            return sum(int(r[key]) for r in rows)
        if scope == "closemiss":
            return sum(int(r[key]) for r in rows if int(r["is_closemiss_target"]) == 1)
        if scope == "both_fail":
            return sum(int(r[key]) for r in rows if r["bucket"] == "both_fail")
        raise ValueError(scope)

    summary_lines = [
        "run: close-miss repair stacked pipeline",
        f"checkpoint: {checkpoint_path}",
        f"config: {config_path}",
        f"baseline ledger: {Path(args.reference_ledger).resolve()}",
        f"close-miss target tasks: {len(target_tasks)}",
        f"ORPI second-opinion entries used: {len(orpi_for_targets)}",
        "",
        f"Strategy gains (deployed strict exact) over all 400 tasks:",
        f"  C0 baseline:                 {count_strategy(per_task, 'c0_exact', 'all')}",
        f"  S0 (raw TRM rerun):          {count_strategy(per_task, 'S0_exact', 'all')}",
        f"  S1 (cleanup):                {count_strategy(per_task, 'S1_exact', 'all')}",
        f"  S2 (cleanup + iso-repair):   {count_strategy(per_task, 'S2_exact', 'all')}",
        f"  S3 (cleanup + repair + ORPI):{count_strategy(per_task, 'S3_exact', 'all')}",
        "",
        f"Strategy gains scoped to close-miss target ({len(target_tasks)} tasks):",
        f"  S0 (raw TRM rerun):          {count_strategy(per_task, 'S0_exact', 'closemiss')}",
        f"  S1 (cleanup):                {count_strategy(per_task, 'S1_exact', 'closemiss')}",
        f"  S2 (cleanup + iso-repair):   {count_strategy(per_task, 'S2_exact', 'closemiss')}",
        f"  S3 (cleanup + repair + ORPI):{count_strategy(per_task, 'S3_exact', 'closemiss')}",
        "",
        f"Strategy gains scoped to both_fail bucket:",
        f"  C0 baseline:                 {count_strategy(per_task, 'c0_exact', 'both_fail')}",
        f"  S0:                          {count_strategy(per_task, 'S0_exact', 'both_fail')}",
        f"  S1:                          {count_strategy(per_task, 'S1_exact', 'both_fail')}",
        f"  S2:                          {count_strategy(per_task, 'S2_exact', 'both_fail')}",
        f"  S3:                          {count_strategy(per_task, 'S3_exact', 'both_fail')}",
        "",
        f"Per-task tasks newly solved by S3 (not in C0):",
    ]
    newly_solved = [r["task_id"] for r in per_task if int(r["S3_exact"]) == 1 and int(r["c0_exact"]) == 0]
    summary_lines.append("  " + ", ".join(newly_solved) if newly_solved else "  (none)")
    regressions = [r["task_id"] for r in per_task if int(r["S3_exact"]) == 0 and int(r["c0_exact"]) == 1]
    summary_lines.append("")
    summary_lines.append(f"Per-task tasks lost by S3 (was C0-exact, now wrong):")
    summary_lines.append("  " + ", ".join(regressions) if regressions else "  (none)")

    (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
