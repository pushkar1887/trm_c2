import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain
from dataset.build_arc_dataset import (
    ARCMaxGridSize,
    PuzzleIdSeparator,
    arc_grid_to_np,
    grid_hash,
)
from dataset.common import dihedral_transform
from evaluators.arc import _crop
from puzzle_dataset import VisualFeatureCache
from scripts.fvr_structfuse_alpha_sweep import FIELDS, row_metrics


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_yaml_config(ref: str) -> dict:
    if ref.startswith("http://") or ref.startswith("https://"):
        with urllib.request.urlopen(ref) as resp:
            return yaml.safe_load(resp.read().decode("utf-8"))
    with Path(ref).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_aug_name(name: str) -> Tuple[str, int | None, np.ndarray | None]:
    if PuzzleIdSeparator not in name:
        return name, None, None
    base, trans_id, perm = name.split(PuzzleIdSeparator)
    return base, int(trans_id[1:]), np.array([int(ch) for ch in perm], dtype=np.uint8)


def inverse_aug_grid(name: str, grid: np.ndarray) -> np.ndarray:
    _base, trans_id, mapping = split_aug_name(name)
    if trans_id is None or mapping is None:
        return grid
    inv_mapping = np.argsort(mapping).astype(np.uint8)
    from dataset.common import inverse_dihedral_transform

    return inv_mapping[inverse_dihedral_transform(grid, trans_id)]


def grid_to_seq(grid: np.ndarray) -> np.ndarray:
    assert grid.ndim == 2
    assert grid.shape[0] <= ARCMaxGridSize and grid.shape[1] <= ARCMaxGridSize
    seq = np.zeros((ARCMaxGridSize, ARCMaxGridSize), dtype=np.uint8)
    h, w = grid.shape
    seq[:h, :w] = grid + 2
    if h < ARCMaxGridSize:
        seq[h, :w] = 1
    if w < ARCMaxGridSize:
        seq[:h, w] = 1
    return seq.reshape(-1)


def build_exact_candidate_examples(
    aug_data_path: Path,
    target_tasks: List[str],
    k: int,
) -> Tuple[List[Dict[str, object]], Dict[str, dict]]:
    aug_ids = json.loads((aug_data_path / "identifiers.json").read_text(encoding="utf-8"))
    test_puzzles = json.loads((aug_data_path / "test_puzzles.json").read_text(encoding="utf-8"))
    target_set = set(target_tasks)

    pid_arr = np.load(aug_data_path / "test" / "all__puzzle_identifiers.npy", mmap_mode="r")
    puzzle_indices = np.load(aug_data_path / "test" / "all__puzzle_indices.npy", mmap_mode="r")
    inputs = np.load(aug_data_path / "test" / "all__inputs.npy", mmap_mode="r")
    labels = np.load(aug_data_path / "test" / "all__labels.npy", mmap_mode="r")

    target_info = {}
    for task_id in target_tasks:
        puzzle = test_puzzles[task_id]
        first_pair = puzzle["test"][0]
        label_grid = arc_grid_to_np(first_pair["output"])
        target_info[task_id] = {
            "input_hash": grid_hash(arc_grid_to_np(first_pair["input"])),
            "label_grid": label_grid,
            "label_hash": grid_hash(label_grid),
        }

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for puzzle_index in range(len(puzzle_indices) - 1):
        aug_pid = int(pid_arr[puzzle_index])
        aug_name = aug_ids[aug_pid]
        base_name, _trans_id, _mapping = split_aug_name(aug_name)
        if base_name not in target_set:
            continue
        if len(grouped[base_name]) >= k:
            continue

        row_start = int(puzzle_indices[puzzle_index])
        row_end = int(puzzle_indices[puzzle_index + 1])
        for row_idx in range(row_start, row_end):
            inv_input = inverse_aug_grid(aug_name, _crop(np.asarray(inputs[row_idx])))
            if grid_hash(inv_input) != target_info[base_name]["input_hash"]:
                continue
            grouped[base_name].append(
                {
                    "task_id": base_name,
                    "aug_name": aug_name,
                    # Critical for original TRM: use the Aug-1000 puzzle identifier
                    # that indexes the checkpoint's 876406-row puzzle embedding.
                    "puzzle_identifier": aug_pid,
                    "input": np.asarray(inputs[row_idx], dtype=np.int32),
                    "label": np.asarray(labels[row_idx], dtype=np.int32),
                    "candidate_index": len(grouped[base_name]),
                }
            )
            break

    empty = [task_id for task_id in target_tasks if len(grouped[task_id]) == 0]
    if empty:
        raise RuntimeError(f"No candidates found for {len(empty)} tasks; first={empty[:10]}")

    examples = []
    for task_id in target_tasks:
        target_info[task_id]["available_candidates"] = len(grouped[task_id])
        target_info[task_id]["used_candidates"] = min(k, len(grouped[task_id]))
        examples.extend(grouped[task_id][:k])
    return examples, target_info


def run_exact_candidates(
    examples: List[Dict[str, object]],
    target_info: Dict[str, dict],
    core_model: torch.nn.Module,
    halt_max_steps: int,
    batch_size: int,
    device: torch.device,
    visual_cache: VisualFeatureCache | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    predictions: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    core_model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            input_np = np.stack([x["input"] for x in chunk])
            batch = {
                "inputs": torch.from_numpy(input_np).to(device),
                "labels": torch.from_numpy(np.stack([x["label"] for x in chunk])).to(device),
                "puzzle_identifiers": torch.tensor(
                    [int(x["puzzle_identifier"]) for x in chunk],
                    dtype=torch.int32,
                    device=device,
                ),
            }
            if visual_cache is not None:
                batch["input_visual_features"] = torch.from_numpy(visual_cache.lookup_batch(input_np)).to(device)
            with torch.device(device.type):
                carry = core_model.initial_carry(batch)
            outputs = None
            for _step in range(1, halt_max_steps + 1):
                carry, outputs = core_model(carry=carry, batch=batch)
            assert outputs is not None

            pred_tokens = torch.argmax(outputs["logits"], dim=-1).detach().cpu().numpy()
            for row, item in enumerate(chunk):
                task_id = str(item["task_id"])
                aug_name = str(item["aug_name"])
                pred_grid = inverse_aug_grid(aug_name, _crop(pred_tokens[row]))
                pred_seq = grid_to_seq(pred_grid)
                label_seq = grid_to_seq(target_info[task_id]["label_grid"])
                metrics = row_metrics(pred_seq, label_seq, n_steps=halt_max_steps)
                pred_hash = grid_hash(pred_grid)
                label_hash = target_info[task_id]["label_hash"]
                row_out = {
                    "task_id": task_id,
                    "candidate_index": int(item["candidate_index"]),
                    "aug_name": aug_name,
                    "puzzle_identifier": int(item["puzzle_identifier"]),
                    "pred_hash": pred_hash,
                    "label_hash": label_hash,
                    "is_exact": int(pred_hash == label_hash),
                }
                row_out.update(metrics)
                predictions[task_id].append(row_out)
            done = min(start + batch_size, len(examples))
            if done % max(batch_size * 25, 1) == 0 or done == len(examples):
                print(f"[exact-candidate] candidates={done}/{len(examples)}", flush=True)
    return predictions


def main() -> None:
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Original TRM exact-candidate@K diagnostic. No voting.")
    parser.add_argument("--config", required=True, help="Local YAML path or HTTP(S) YAML URL.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--aug-data-path", required=True)
    parser.add_argument("--canonical-data", default="data/arc-agi-evaluation-full400-seed0")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--visual-cache", default=None)
    parser.add_argument("--reference-ledger", default=None)
    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be positive")

    checkpoint_path = Path(args.checkpoint).resolve()
    aug_data_path = Path(args.aug_data_path).resolve()
    canonical_data = Path(args.canonical_data).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_ids = json.loads((canonical_data / "identifiers.json").read_text(encoding="utf-8"))
    target_tasks = [task_id for task_id in target_ids if task_id != "<blank>"]
    if len(target_tasks) != 400:
        raise RuntimeError(f"Expected 400 canonical tasks, got {len(target_tasks)} from {canonical_data}")

    raw_config = load_yaml_config(args.config)
    raw_config["load_checkpoint"] = str(checkpoint_path)
    raw_config["data_paths"] = [str(aug_data_path)]
    raw_config["data_paths_test"] = []
    raw_config["global_batch_size"] = int(args.global_batch_size)
    raw_config["dataloader_num_workers"] = 0
    raw_config["checkpoint_path"] = str(out_dir / "noop_checkpoints")
    raw_config["run_name"] = "original_trm_exact_candidate_k"
    raw_config["ema"] = False
    raw_config["checkpoint_every_eval"] = False

    config_copy = out_dir / "original_trm_exact_candidate_config.yaml"
    config_copy.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    config = pretrain.PretrainConfig(**raw_config)
    visual_cache = VisualFeatureCache(args.visual_cache) if args.visual_cache else None

    examples, target_info = build_exact_candidate_examples(
        aug_data_path=aug_data_path,
        target_tasks=target_tasks,
        k=int(args.k),
    )
    print(
        f"[setup] tasks={len(target_tasks)} k={args.k} candidates={len(examples)} "
        f"checkpoint={checkpoint_path}",
        flush=True,
    )

    train_loader, train_metadata = pretrain.create_dataloader(
        config,
        "train",
        0,
        1,
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    del train_loader
    loss_head, _, _ = pretrain.create_model(config, train_metadata, rank=0, world_size=1)
    core_model = loss_head.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this TRM eval path.")

    halt_max_steps = int(config.arch.__pydantic_extra__.get("halt_max_steps", 16))
    eval_start = time.perf_counter()
    predictions = run_exact_candidates(
        examples=examples,
        target_info=target_info,
        core_model=core_model,
        halt_max_steps=halt_max_steps,
        batch_size=int(args.global_batch_size),
        device=device,
        visual_cache=visual_cache,
    )
    eval_seconds = time.perf_counter() - eval_start

    task_rows = []
    candidate_rows = []
    exact_task_ids = []
    for task_id in target_tasks:
        preds = predictions[task_id]
        exact_candidates = [p for p in preds if int(p["is_exact"]) > 0]
        if exact_candidates:
            exact_task_ids.append(task_id)
        task_rows.append(
            {
                "task_id": task_id,
                "k": int(args.k),
                "available_candidates": target_info[task_id].get("available_candidates", ""),
                "used_candidates": len(preds),
                "exact_candidate_found": int(bool(exact_candidates)),
                "n_exact_candidates": len(exact_candidates),
                "exact_candidate_indices": "|".join(str(p["candidate_index"]) for p in exact_candidates),
                "exact_aug_names": "|".join(str(p["aug_name"]) for p in exact_candidates),
            }
        )
        candidate_rows.extend(preds)

    summary = [
        {
            "k": int(args.k),
            "tasks": len(target_tasks),
            "candidates": len(examples),
            "exact_candidate_found": len(exact_task_ids),
            "exact_candidate_rate": len(exact_task_ids) / len(target_tasks),
            "tasks_with_full_k": sum(int(target_info[t].get("available_candidates", 0)) >= int(args.k) for t in target_tasks),
            "halt_max_steps": halt_max_steps,
            "candidate_eval_seconds": eval_seconds,
            "total_runtime_seconds": time.perf_counter() - start_time,
            "checkpoint": str(checkpoint_path),
            "config_source": args.config,
            "aug_data_path": str(aug_data_path),
            "voting": "false",
            "visual_cache": str(Path(args.visual_cache).resolve()) if args.visual_cache else "",
        }
    ]

    write_csv(out_dir / "exact_candidate_summary.csv", summary, list(summary[0].keys()))
    write_csv(out_dir / "exact_candidate_task_detail.csv", task_rows, list(task_rows[0].keys()))
    write_csv(out_dir / "exact_candidate_candidates.csv", candidate_rows, list(candidate_rows[0].keys()))
    (out_dir / "exact_candidate_task_ids.txt").write_text(
        "\n".join(exact_task_ids) + ("\n" if exact_task_ids else ""),
        encoding="utf-8",
    )
    if args.reference_ledger:
        with Path(args.reference_ledger).open("r", newline="", encoding="utf-8") as f:
            ref_rows = list(csv.DictReader(f))
        ref_by_task = {row["task_id"]: row for row in ref_rows}
        ledger_rows = []
        for ref in ref_rows:
            task_id = ref["task_id"]
            if task_id not in predictions or not predictions[task_id]:
                continue
            pred = predictions[task_id][0]
            row = {
                "task_id": task_id,
                "puzzle_id": ref.get("puzzle_id", ""),
                "bucket": ref.get("bucket", ""),
            }
            for key in FIELDS[3:]:
                row[key] = pred[key]
            row["majority_floor_content"] = ref.get(
                "majority_floor_content",
                row.get("majority_floor_content", 0.0),
            )
            ledger_rows.append(row)
        write_csv(out_dir / "17col_ledger.csv", ledger_rows, FIELDS)
        (out_dir / "solved_ids.txt").write_text(
            "\n".join(str(row["task_id"]) for row in ledger_rows if float(row["exact_accuracy"]) > 0)
            + ("\n" if ledger_rows else ""),
            encoding="utf-8",
        )

    report = [
        "run: original TRM exact-candidate@K",
        f"checkpoint: {checkpoint_path}",
        f"config_source: {args.config}",
        f"aug_data_path: {aug_data_path}",
        f"canonical_data: {canonical_data}",
        "training: false",
        "checkpoint_mutated: false",
        "voting: false",
        f"visual_cache: {Path(args.visual_cache).resolve() if args.visual_cache else ''}",
        f"k: {args.k}",
        f"halt_max_steps: {halt_max_steps}",
        f"tasks: {len(target_tasks)}",
        f"candidates: {len(examples)}",
        f"exact_candidate_found: {len(exact_task_ids)}/{len(target_tasks)}",
        f"candidate_eval_seconds: {eval_seconds:.2f}",
        f"total_runtime_seconds: {summary[0]['total_runtime_seconds']:.2f}",
        "",
        "exact_candidate_task_ids:",
        *exact_task_ids,
    ]
    (out_dir / "run_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
