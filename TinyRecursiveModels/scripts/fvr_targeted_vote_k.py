import argparse
import csv
import json
import os
import sys
import time
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
from dataset.build_arc_dataset import (
    ARCMaxGridSize,
    PuzzleIdSeparator,
    arc_grid_to_np,
    grid_hash,
)
from dataset.common import dihedral_transform
from evaluators.arc import _crop
from scripts.fvr_structfuse_alpha_sweep import (
    BUCKETS,
    FIELDS,
    SUMMARY_METRICS,
    row_metrics,
    summarize,
    write_csv,
)
from scripts.model_candidate_dump import write_model_dump


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_ks(raw: str) -> List[int]:
    ks = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not ks:
        raise ValueError("At least one K is required.")
    if min(ks) <= 0:
        raise ValueError(f"K values must be positive: {ks}")
    return sorted(set(ks))


def split_aug_name(name: str) -> Tuple[str, int | None, np.ndarray | None]:
    if PuzzleIdSeparator not in name:
        return name, None, None
    base, trans_id, perm = name.split(PuzzleIdSeparator)
    return base, int(trans_id[1:]), np.array([int(ch) for ch in perm], dtype=np.uint8)


def forward_aug_grid(name: str, grid: np.ndarray) -> np.ndarray:
    _base, trans_id, mapping = split_aug_name(name)
    if trans_id is None or mapping is None:
        return grid
    return dihedral_transform(mapping[grid], trans_id)


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


def context_for_aug_name(name: str, puzzle: dict, c2_num_context: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = np.zeros((c2_num_context, ARCMaxGridSize * ARCMaxGridSize), dtype=np.int32)
    outputs = np.zeros_like(inputs)
    mask = np.zeros((c2_num_context,), dtype=np.int32)
    for idx, example in enumerate(puzzle.get("train", [])[:c2_num_context]):
        inp = forward_aug_grid(name, arc_grid_to_np(example["input"]))
        out = forward_aug_grid(name, arc_grid_to_np(example["output"]))
        inputs[idx] = grid_to_seq(inp).astype(np.int32)
        outputs[idx] = grid_to_seq(out).astype(np.int32)
        mask[idx] = 1
    return inputs, outputs, mask


def build_target_examples(
    aug_data_path: Path,
    eval_ids: List[str],
    target_tasks: List[str],
    max_k: int,
    c2_num_context: int,
) -> Tuple[List[Dict[str, object]], Dict[str, dict]]:
    aug_ids = json.loads((aug_data_path / "identifiers.json").read_text(encoding="utf-8"))
    test_puzzles = json.loads((aug_data_path / "test_puzzles.json").read_text(encoding="utf-8"))
    base_to_eval_pid = {name: idx for idx, name in enumerate(eval_ids)}
    target_set = set(target_tasks)

    pid_arr = np.load(aug_data_path / "test" / "all__puzzle_identifiers.npy", mmap_mode="r")
    puzzle_indices = np.load(aug_data_path / "test" / "all__puzzle_indices.npy", mmap_mode="r")
    inputs = np.load(aug_data_path / "test" / "all__inputs.npy", mmap_mode="r")
    labels = np.load(aug_data_path / "test" / "all__labels.npy", mmap_mode="r")

    target_info = {}
    for task_id in target_tasks:
        puzzle = test_puzzles[task_id]
        first_pair = puzzle["test"][0]
        target_info[task_id] = {
            "input_hash": grid_hash(arc_grid_to_np(first_pair["input"])),
            "label_grid": arc_grid_to_np(first_pair["output"]),
            "label_hash": grid_hash(arc_grid_to_np(first_pair["output"])),
        }

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    context_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for puzzle_index in range(len(puzzle_indices) - 1):
        aug_pid = int(pid_arr[puzzle_index])
        aug_name = aug_ids[aug_pid]
        base_name, _trans_id, _mapping = split_aug_name(aug_name)
        if base_name not in target_set:
            continue
        if len(grouped[base_name]) >= max_k:
            continue

        row_start = int(puzzle_indices[puzzle_index])
        row_end = int(puzzle_indices[puzzle_index + 1])
        eval_pid = base_to_eval_pid[base_name]
        if aug_name not in context_cache:
            context_cache[aug_name] = context_for_aug_name(
                aug_name,
                test_puzzles[base_name],
                c2_num_context=c2_num_context,
            )
        ctx_inputs, ctx_outputs, ctx_mask = context_cache[aug_name]
        for row_idx in range(row_start, row_end):
            inv_input = inverse_aug_grid(aug_name, _crop(np.asarray(inputs[row_idx])))
            if grid_hash(inv_input) != target_info[base_name]["input_hash"]:
                continue
            grouped[base_name].append(
                {
                    "task_id": base_name,
                    "aug_name": aug_name,
                    "input": np.asarray(inputs[row_idx], dtype=np.int32),
                    "label": np.asarray(labels[row_idx], dtype=np.int32),
                    "puzzle_identifier": int(eval_pid),
                    "context_inputs": ctx_inputs,
                    "context_outputs": ctx_outputs,
                    "context_mask": ctx_mask,
                    "candidate_index": len(grouped[base_name]),
                }
            )
            break

    empty = [task_id for task_id in target_tasks if len(grouped[task_id]) == 0]
    if empty:
        raise RuntimeError(
            f"No augmented candidates for {len(empty)} target tasks; "
            f"first empty={empty[:10]}"
        )

    examples = []
    for task_id in target_tasks:
        available = len(grouped[task_id])
        target_info[task_id]["available_candidates"] = available
        target_info[task_id]["used_candidate_cap"] = min(max_k, available)
        examples.extend(grouped[task_id][: min(max_k, available)])
    return examples, target_info


def run_candidates(
    examples: List[Dict[str, object]],
    target_info: Dict[str, dict],
    core_model: torch.nn.Module,
    halt_max_steps: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, List[Dict[str, object]]]:
    predictions: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    core_model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            batch = {
                "inputs": torch.from_numpy(np.stack([x["input"] for x in chunk])).to(device),
                "labels": torch.from_numpy(np.stack([x["label"] for x in chunk])).to(device),
                "puzzle_identifiers": torch.tensor(
                    [int(x["puzzle_identifier"]) for x in chunk],
                    dtype=torch.int32,
                    device=device,
                ),
                "context_inputs": torch.from_numpy(np.stack([x["context_inputs"] for x in chunk])).to(device),
                "context_outputs": torch.from_numpy(np.stack([x["context_outputs"] for x in chunk])).to(device),
                "context_mask": torch.from_numpy(np.stack([x["context_mask"] for x in chunk])).to(device),
            }
            with torch.device(device.type):
                carry = core_model.initial_carry(batch)
            outputs = None
            for _step in range(1, halt_max_steps + 1):
                carry, outputs = core_model(carry=carry, batch=batch)
            assert outputs is not None
            pred_tokens = torch.argmax(outputs["logits"], dim=-1).detach().cpu().numpy()
            q_values = outputs["q_halt_logits"].detach().sigmoid().cpu().numpy()
            for row, item in enumerate(chunk):
                task_id = str(item["task_id"])
                aug_name = str(item["aug_name"])
                pred_grid = inverse_aug_grid(aug_name, _crop(pred_tokens[row]))
                pred_hash = grid_hash(pred_grid)
                label_hash = target_info[task_id]["label_hash"]
                predictions[task_id].append(
                    {
                        "candidate_index": int(item["candidate_index"]),
                        "aug_name": aug_name,
                        "pred_hash": pred_hash,
                        "pred_grid": pred_grid,
                        "q": float(q_values[row]),
                        "is_exact": int(pred_hash == label_hash),
                    }
                )
            done = min(start + batch_size, len(examples))
            if done % max(batch_size * 25, 1) == 0 or done == len(examples):
                print(f"[vote] candidates={done}/{len(examples)}")
    return predictions


def vote_for_k(task_preds: List[Dict[str, object]], k: int) -> Tuple[Dict[str, object], int, int]:
    candidates = task_preds[:k]
    counts: Dict[str, List[object]] = {}
    for cand in candidates:
        h = str(cand["pred_hash"])
        if h not in counts:
            counts[h] = [0, 0.0, cand["pred_grid"]]
        counts[h][0] = int(counts[h][0]) + 1
        counts[h][1] = float(counts[h][1]) + float(cand["q"])
    ranked = []
    for h, stats in counts.items():
        count = int(stats[0])
        avg_q = float(stats[1]) / max(count, 1)
        ranked.append((h, count, avg_q, stats[2]))
    ranked.sort(key=lambda item: ([item[1], item[2]], item[0]), reverse=True)
    top_hash, top_count, top_q, top_grid = ranked[0]
    return {"pred_hash": top_hash, "count": top_count, "avg_q": top_q, "pred_grid": top_grid}, len(ranked), len(candidates)


def model_dump_records(
    predictions: Dict[str, List[Dict[str, object]]],
    target_tasks: List[str],
    k: int,
) -> List[Dict[str, object]]:
    """Build top-K distinct TRM vote records for scripts/model_candidate_dump.py.

    This is the V3 bridge producer: keep the candidate grids and vote counts
    that the existing voting diagnostic already computed, so eval_arc_agi1.py
    can run Verify/Select over the model pool instead of only the top vote.
    """
    records: List[Dict[str, object]] = []
    for task_id in target_tasks:
        counts: Dict[str, List[object]] = {}
        for cand in predictions[task_id][:k]:
            h = str(cand["pred_hash"])
            if h not in counts:
                counts[h] = [0, 0.0, cand["pred_grid"]]
            counts[h][0] = int(counts[h][0]) + 1
            counts[h][1] = float(counts[h][1]) + float(cand["q"])
        ranked = []
        for h, stats in counts.items():
            count = int(stats[0])
            avg_q = float(stats[1]) / max(count, 1)
            ranked.append((h, count, avg_q, stats[2]))
        ranked.sort(key=lambda item: ([item[1], item[2]], item[0]), reverse=True)
        if not ranked:
            continue
        records.append(
            {
                "task_id": task_id,
                # This canonical diagnostic has one target question per task.
                # Raw multi-test ARC tasks need a producer that emits each test_index.
                "test_index": 0,
                "candidates": [grid_to_seq(np.asarray(grid)) for _h, _count, _q, grid in ranked],
                "vote_counts": [count for _h, count, _q, _grid in ranked],
            }
        )
    return records


def movement_for_subset(k: int, reference_rows: List[Dict[str, str]], rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ref_by_task = {row["task_id"]: row for row in reference_rows}
    row_by_task = {str(row["task_id"]): row for row in rows}
    out = []
    for bucket in BUCKETS:
        rec = {
            "K": k,
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
        }
        for task_id, cur in row_by_task.items():
            ref = ref_by_task[task_id]
            if ref["bucket"] != bucket:
                continue
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
        out.append(rec)
    return out


def main() -> None:
    total_start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description="Augmentation vote test on C0 canonical evaluation tasks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-ledger", required=True)
    parser.add_argument("--aug-data-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ks", default="8,16,32,64")
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument(
        "--model-dump-out",
        default=None,
        help="Optional V3 bridge output: write top-K distinct TRM candidate grids and vote counts to .npz.",
    )
    parser.add_argument(
        "--target-scope",
        choices=["both_fail_closemiss", "all400"],
        default="both_fail_closemiss",
        help="Task subset to evaluate. all400 uses every task in the reference ledger.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    reference_path = Path(args.reference_ledger).resolve()
    aug_data_path = Path(args.aug_data_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = parse_ks(args.ks)
    max_k = max(ks)

    reference_rows = read_csv(reference_path)
    reference_by_task = {row["task_id"]: row for row in reference_rows}
    if args.target_scope == "all400":
        target_tasks = [row["task_id"] for row in reference_rows]
    else:
        target_tasks = [
            row["task_id"]
            for row in reference_rows
            if row["bucket"] == "both_fail"
            and float(row["exact_accuracy"]) == 0.0
            and float(row["close_miss"]) > 0.0
        ]
    if not target_tasks:
        raise RuntimeError(f"No target tasks found for target_scope={args.target_scope}.")

    with config_path.open("r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    raw_config["load_checkpoint"] = str(checkpoint_path)
    raw_config["data_paths"] = ["data/arc-agi-evaluation-full400-seed0"]
    raw_config["data_paths_test"] = []
    raw_config["dataloader_num_workers"] = 0
    raw_config["checkpoint_path"] = str(out_dir / "noop_checkpoints")
    raw_config["run_name"] = "targeted_vote_k"
    raw_config["global_batch_size"] = int(args.global_batch_size)
    raw_config.setdefault("arch", {})["c2_structure_fusion_alpha"] = 0.0
    config_copy = out_dir / "targeted_vote_config.yaml"
    config_copy.write_text(yaml.safe_dump(raw_config, sort_keys=False), encoding="utf-8")
    config = pretrain.PretrainConfig(**raw_config)

    repo_root = Path(__file__).resolve().parents[1]
    eval_ids = json.loads((repo_root / "data" / "arc-agi-evaluation-full400-seed0" / "identifiers.json").read_text(encoding="utf-8"))
    c2_num_context = int((config.arch.__pydantic_extra__ or {}).get("c2_num_context", 0))
    examples, target_info = build_target_examples(
        aug_data_path=aug_data_path,
        eval_ids=eval_ids,
        target_tasks=target_tasks,
        max_k=max_k,
        c2_num_context=c2_num_context,
    )
    short_tasks = [
        task_id
        for task_id in target_tasks
        if int(target_info[task_id].get("available_candidates", 0)) < max_k
    ]
    print(
        f"[setup] target_scope={args.target_scope} target_tasks={len(target_tasks)} "
        f"candidates={len(examples)} max_k={max_k} short_tasks={len(short_tasks)}"
    )
    if short_tasks:
        preview = ", ".join(
            f"{task_id}:{target_info[task_id]['available_candidates']}" for task_id in short_tasks[:10]
        )
        print(f"[setup] capped_tasks first={preview}")

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

    eval_start_time = time.perf_counter()
    predictions = run_candidates(
        examples=examples,
        target_info=target_info,
        core_model=core_model,
        halt_max_steps=int(config.arch.__pydantic_extra__.get("halt_max_steps", 16)),
        batch_size=int(args.global_batch_size),
        device=device,
    )
    candidate_eval_seconds = time.perf_counter() - eval_start_time
    if args.model_dump_out:
        dump_path = Path(args.model_dump_out).resolve()
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        records = model_dump_records(predictions, target_tasks, max_k)
        write_model_dump(dump_path, records)
        print(f"[model-dump] wrote {len(records)} records to {dump_path}")

    all_summary = []
    all_movement = []
    all_delta = []
    best_k = None
    best_vote_count = -1
    best_rows: List[Dict[str, object]] = []
    detail_rows = []
    for k in ks:
        rows = []
        any_hit_count = 0
        pass_at_k_count = 0
        tasks_with_full_k = 0
        total_used_candidates = 0
        for task_id in target_tasks:
            task_preds = predictions[task_id]
            label_hash = target_info[task_id]["label_hash"]
            any_hit = any(int(c["is_exact"]) for c in task_preds[:k])
            vote, unique_candidates, used_candidates = vote_for_k(task_preds, k)
            available_candidates = int(target_info[task_id].get("available_candidates", len(task_preds)))
            tasks_with_full_k += int(available_candidates >= k)
            total_used_candidates += int(used_candidates)
            ranked_hashes = []
            counts = Counter(str(c["pred_hash"]) for c in task_preds[:k])
            for pred_hash, _count in counts.most_common(k):
                ranked_hashes.append(pred_hash)
            pass_at_k = label_hash in ranked_hashes[:k]
            vote_exact = vote["pred_hash"] == label_hash
            any_hit_count += int(any_hit)
            pass_at_k_count += int(pass_at_k)

            label_seq = grid_to_seq(target_info[task_id]["label_grid"])
            pred_seq = grid_to_seq(vote["pred_grid"])
            metrics = row_metrics(pred_seq, label_seq, n_steps=int(config.arch.__pydantic_extra__.get("halt_max_steps", 16)))
            ref = reference_by_task[task_id]
            row: Dict[str, object] = {
                "task_id": task_id,
                "puzzle_id": ref["puzzle_id"],
                "bucket": ref["bucket"],
            }
            for field in FIELDS[3:]:
                row[field] = metrics[field]
            row["majority_floor_content"] = ref["majority_floor_content"]
            rows.append(row)

            detail_rows.append(
                {
                    "K": k,
                    "task_id": task_id,
                    "bucket": ref["bucket"],
                    "any_hit": int(any_hit),
                    "pass_at_k": int(pass_at_k),
                    "vote_exact": int(vote_exact),
                    "unique_candidates": unique_candidates,
                    "available_candidates": available_candidates,
                    "used_candidates": used_candidates,
                    "has_full_k": int(available_candidates >= k),
                    "vote_count": vote["count"],
                    "vote_avg_q": vote["avg_q"],
                    "ref_content": ref["content_accuracy"],
                    "vote_content": metrics["content_accuracy"],
                    "ref_outside_fpr": ref["outside_canvas_fpr"],
                    "vote_outside_fpr": metrics["outside_canvas_fpr"],
                }
            )

        ledger_path = out_dir / f"vote_k{k}_17col_ledger.csv"
        write_csv(ledger_path, rows, FIELDS)
        solved_ids = [str(row["task_id"]) for row in rows if float(row["exact_accuracy"]) > 0]
        (out_dir / f"vote_k{k}_solved_ids.txt").write_text(
            "\n".join(solved_ids) + ("\n" if solved_ids else ""),
            encoding="utf-8",
        )
        movement = movement_for_subset(k, reference_rows, rows)
        all_movement.extend(movement)
        for row in rows:
            ref = reference_by_task[str(row["task_id"])]
            all_delta.append(
                {
                    "K": k,
                    "task_id": row["task_id"],
                    "bucket": row["bucket"],
                    "ref_exact": ref["exact_accuracy"],
                    "vote_exact": row["exact_accuracy"],
                    "exact_delta": float(row["exact_accuracy"]) - float(ref["exact_accuracy"]),
                    "ref_content": ref["content_accuracy"],
                    "vote_content": row["content_accuracy"],
                    "content_delta": float(row["content_accuracy"]) - float(ref["content_accuracy"]),
                    "ref_close_miss": ref["close_miss"],
                    "vote_close_miss": row["close_miss"],
                }
            )
        summary = summarize(rows)
        vote_exact_count = int(round(summary["exact_accuracy"] * len(rows)))
        exact_gain_count = sum(int(rec["exact_gain"]) for rec in movement)
        exact_loss_count = sum(int(rec["exact_loss"]) for rec in movement)
        exact_retained_count = sum(int(rec["exact_retained"]) for rec in movement)
        both_fail_movement = next(rec for rec in movement if rec["bucket"] == "both_fail")
        both_fail_vote_exact_count = sum(
            1 for row in rows if row["bucket"] == "both_fail" and float(row["exact_accuracy"]) > 0
        )
        summary_row = {
            "K": k,
            "target_scope": args.target_scope,
            "target_tasks": len(target_tasks),
            "tasks_with_full_k": tasks_with_full_k,
            "total_used_candidates": total_used_candidates,
            "mean_used_candidates": total_used_candidates / len(target_tasks),
            "any_hit_count": any_hit_count,
            "any_hit_rate": any_hit_count / len(target_tasks),
            "vote_exact_count": vote_exact_count,
            "vote_exact_rate": vote_exact_count / len(target_tasks),
            "pass_at_k_count": pass_at_k_count,
            "pass_at_k_rate": pass_at_k_count / len(target_tasks),
            "exact_gain_vs_c0": exact_gain_count,
            "exact_loss_vs_c0": exact_loss_count,
            "exact_retained_vs_c0": exact_retained_count,
            "net_exact_vs_c0": exact_gain_count - exact_loss_count,
            "both_fail_vote_exact_count": both_fail_vote_exact_count,
            "both_fail_exact_gained": int(both_fail_movement["exact_gain"]),
            "both_fail_exact_lost": int(both_fail_movement["exact_loss"]),
            "candidate_eval_seconds": candidate_eval_seconds,
            **summary,
        }
        all_summary.append(summary_row)
        if vote_exact_count > best_vote_count:
            best_vote_count = vote_exact_count
            best_k = k
            best_rows = rows
        print(
            f"[K={k}] any_hit={any_hit_count}/{len(target_tasks)} "
            f"vote_exact={vote_exact_count}/{len(target_tasks)} "
            f"gains={exact_gain_count} losses={exact_loss_count} "
            f"both_fail_gains={int(both_fail_movement['exact_gain'])} "
            f"full_k={tasks_with_full_k}/{len(target_tasks)}"
        )

    write_csv(out_dir / "candidate_vote_summary.csv", all_summary, list(all_summary[0].keys()))
    write_csv(out_dir / "bucket_movement.csv", all_movement, list(all_movement[0].keys()))
    write_csv(out_dir / "per_task_delta.csv", all_delta, list(all_delta[0].keys()))
    write_csv(out_dir / "candidate_vote_task_detail.csv", detail_rows, list(detail_rows[0].keys()))
    first_hit_rows = []
    for task_id in target_tasks:
        task_details = [row for row in detail_rows if row["task_id"] == task_id]
        first_any_hit_k = next((row["K"] for row in task_details if int(row["any_hit"]) > 0), "")
        first_vote_exact_k = next((row["K"] for row in task_details if int(row["vote_exact"]) > 0), "")
        ref = reference_by_task[task_id]
        first_hit_rows.append(
            {
                "task_id": task_id,
                "bucket": ref["bucket"],
                "ref_exact": ref["exact_accuracy"],
                "available_candidates": target_info[task_id].get("available_candidates", ""),
                "first_any_hit_k": first_any_hit_k,
                "first_vote_exact_k": first_vote_exact_k,
            }
        )
    write_csv(out_dir / "first_hit_k.csv", first_hit_rows, list(first_hit_rows[0].keys()))
    (out_dir / "solved_ids.txt").write_text(
        "\n".join(str(row["task_id"]) for row in best_rows if float(row["exact_accuracy"]) > 0)
        + ("\n" if best_vote_count > 0 else ""),
        encoding="utf-8",
    )

    max_any_hit = max(int(row["any_hit_count"]) for row in all_summary)
    max_vote_exact = max(int(row["vote_exact_count"]) for row in all_summary)
    max_exact_gain = max(int(row["exact_gain_vs_c0"]) for row in all_summary)
    max_both_fail_gain = max(int(row["both_fail_exact_gained"]) for row in all_summary)
    if args.target_scope == "all400":
        if max_both_fail_gain > 0 or max_exact_gain > 0:
            verdict = "DIAGNOSTIC_GAIN"
            reason = "Augmented voting produced at least one new exact solve under top-vote selection."
            next_stage = "Inspect first_hit_k.csv and candidate_vote_task_detail.csv; build reranker if any-hit exceeds vote-selected gains."
        else:
            verdict = "DIAGNOSTIC_REJECT"
            reason = "Augmented voting did not produce new top-vote exact solves over C0."
            next_stage = "Use any-hit rows to decide whether reranking has headroom; otherwise stop increasing K."
    elif max_vote_exact >= 15:
        verdict = "KEEP"
        reason = "Targeted voting converts at least 15 both_fail close-miss tasks by top-vote exact."
        next_stage = "Run full-400 augmented voting and then build verifier/reranker if solved retention is acceptable."
    elif max_any_hit >= 15:
        verdict = "REJECT_AS_FINAL"
        reason = "Oracle any-hit is high but top vote does not convert enough tasks; this is a selection/reranking problem."
        next_stage = "Build verifier/reranker before more training."
    else:
        verdict = "REJECT"
        reason = "Oracle any-hit is too low for the 15-20 both_fail conversion target; this is a representation/training problem."
        next_stage = "Proceed to Stage 3 LODO-Light."

    total_seconds = time.perf_counter() - total_start_time
    report = [
        f"verdict: {verdict}",
        f"target_scope: {args.target_scope}",
        f"checkpoint: {checkpoint_path}",
        f"config: {config_path}",
        f"aug_data_path: {aug_data_path}",
        f"reference_ledger: {reference_path}",
        f"training: false",
        f"checkpoint mutated: false",
        f"halt_max_steps: {int(config.arch.__pydantic_extra__.get('halt_max_steps', 16))}",
        f"best_k: {best_k}",
        f"exact gained: {max_exact_gain}",
        f"exact lost: {max(int(row['exact_loss_vs_c0']) for row in all_summary)}",
        f"both_fail exact gained: {max_both_fail_gain}",
        f"both_fail exact lost: {max(int(row['both_fail_exact_lost']) for row in all_summary)}",
        f"close_miss gained/lost: see per_task_delta.csv and bucket_movement.csv",
        f"outside_fpr change: subset diagnostic, see candidate_vote_summary.csv",
        f"candidate_eval_seconds: {candidate_eval_seconds:.2f}",
        f"total_runtime_seconds: {total_seconds:.2f}",
        f"reason: {reason}",
        f"next stage: {next_stage}",
        "",
        "summary:",
    ]
    for row in all_summary:
        report.append(
            "K={K}: any_hit={any_hit_count}/{target_tasks}, vote_exact={vote_exact_count}/{target_tasks}, "
            "exact_gain={exact_gain_vs_c0}, exact_loss={exact_loss_vs_c0}, net_exact={net_exact_vs_c0}, "
            "both_fail_gain={both_fail_exact_gained}, both_fail_vote_exact={both_fail_vote_exact_count}, "
            "tasks_with_full_k={tasks_with_full_k}/{target_tasks}, mean_used_candidates={mean_used_candidates:.2f}, "
            "outside_fpr={outside_canvas_fpr:.6f}, content={content_accuracy:.6f}".format(**row)
        )
    (out_dir / "rejection_or_keep.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
