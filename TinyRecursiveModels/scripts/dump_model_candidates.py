"""Dump TRM top-K hypotheses for LOOCV verification and test-time selection.

The output is the typed `model_candidate_dump.py` format:
  record_kind="fold": held-out support demo j, context excludes j
  record_kind="test": real target test input, context uses support demos only

No target-test outputs are loaded or needed. The only labels used are support
demo outputs for fold reconstruction metadata.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain
from dataset.build_arc_dataset import arc_grid_to_np, np_grid_to_seq_translational_augment
from model_candidate_dump import write_model_dump
from puzzle_dataset import _derive_target_hw_from_label_tokens


def _arch_value(config: pretrain.PretrainConfig, name: str, default: Any = None) -> Any:
    if hasattr(config.arch, name):
        return getattr(config.arch, name)
    extra = getattr(config.arch, "__pydantic_extra__", None) or {}
    return extra.get(name, default)


def _repo_path(path: str | Path, repo_root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _pair_to_seq(inp_grid: list[list[int]], out_grid: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    inp = arc_grid_to_np(inp_grid)
    out = arc_grid_to_np(out_grid)
    x, y = np_grid_to_seq_translational_augment(inp, out, do_translation=False)
    return x.astype(np.int32), y.astype(np.int32)


def _input_to_seq(inp_grid: list[list[int]]) -> np.ndarray:
    x, _ = _pair_to_seq(inp_grid, inp_grid)
    return x


def _episode_jobs_for_task(
    task_id: str,
    pid: int,
    task: dict[str, Any],
    context_limit: int,
    blank_pid: int,
    blank_fold_pid: bool,
    include_folds: bool,
    include_tests: bool,
) -> list[dict[str, Any]]:
    train_pairs = [(_pair_to_seq(p["input"], p["output"])) for p in task.get("train", [])]
    jobs: list[dict[str, Any]] = []

    def _context(exclude: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ci = np.zeros((context_limit, train_pairs[0][0].shape[0]), dtype=np.int32)
        co = np.zeros_like(ci)
        cm = np.zeros((context_limit,), dtype=np.int32)
        out_slot = 0
        for j, (x, y) in enumerate(train_pairs):
            if exclude is not None and j == exclude:
                continue
            if out_slot >= context_limit:
                break
            ci[out_slot] = x
            co[out_slot] = y
            cm[out_slot] = 1
            out_slot += 1
        return ci, co, cm

    if context_limit > 0 and train_pairs:
        if include_folds and len(train_pairs) >= 2:
            for j, (x, y) in enumerate(train_pairs):
                ci, co, cm = _context(exclude=j)
                jobs.append(
                    {
                        "task_id": task_id,
                        "test_index": j,
                        "record_kind": "fold",
                        "inputs": x,
                        # Leakage guard: the held-out output is the verifier's
                        # target, not a model input. Use an input-shaped label
                        # placeholder so target_height/target_width and any
                        # forward-side batch plumbing cannot see the answer.
                        "labels": x.copy(),
                        "puzzle_identifier": blank_pid if blank_fold_pid else pid,
                        "context_inputs": ci,
                        "context_outputs": co,
                        "context_mask": cm,
                    }
                )
        if include_tests:
            ci, co, cm = _context(exclude=None)
            for t, test in enumerate(task.get("test", [])):
                x = _input_to_seq(test["input"]).astype(np.int32)
                # No test output is available here by design. Labels are a shape
                # placeholder for code paths that expect the key; model forward
                # itself does not score against them.
                jobs.append(
                    {
                        "task_id": task_id,
                        "test_index": t,
                        "record_kind": "test",
                        "inputs": x,
                        "labels": x.copy(),
                        "puzzle_identifier": pid,
                        "context_inputs": ci.copy(),
                        "context_outputs": co.copy(),
                        "context_mask": cm.copy(),
                    }
                )
    return jobs


def _attach_derived_features(tensors: dict[str, torch.Tensor], config: pretrain.PretrainConfig) -> None:
    side = int(np.sqrt(tensors["inputs"].shape[-1]))
    if bool(_arch_value(config, "c2_relmap", False)):
        from models.recursive_reasoning.object_bank import relational_maps

        with torch.no_grad():
            tensors["rel_maps"] = relational_maps(tensors["inputs"], side=side)
            if "context_inputs" in tensors:
                b, c, l = tensors["context_inputs"].shape
                ctx = tensors["context_inputs"].reshape(b * c, l)
                tensors["context_rel_maps"] = relational_maps(ctx, side=side).reshape(b, c, l, -1)
                cout = tensors["context_outputs"].reshape(b * c, l)
                tensors["context_output_rel_maps"] = relational_maps(cout, side=side).reshape(b, c, l, -1)
    if bool(_arch_value(config, "c2_frame_hint", False)):
        from models.recursive_reasoning.object_rule_bank import task_frame_label

        b = tensors["inputs"].shape[0]
        labels = torch.zeros(b, dtype=torch.long)
        with torch.no_grad():
            for i in range(b):
                labels[i] = task_frame_label(tensors["context_inputs"][i], tensors["context_outputs"][i], side)
        tensors["frame_label"] = labels


def _batch_from_jobs(jobs: list[dict[str, Any]], config: pretrain.PretrainConfig) -> dict[str, torch.Tensor]:
    inputs = np.stack([j["inputs"] for j in jobs]).astype(np.int32)
    labels = np.stack([j["labels"] for j in jobs]).astype(np.int32)
    target_h, target_w = _derive_target_hw_from_label_tokens(labels)
    batch = {
        "inputs": torch.from_numpy(inputs),
        "labels": torch.from_numpy(labels),
        "puzzle_identifiers": torch.tensor([int(j["puzzle_identifier"]) for j in jobs], dtype=torch.int32),
        "context_inputs": torch.from_numpy(np.stack([j["context_inputs"] for j in jobs]).astype(np.int32)),
        "context_outputs": torch.from_numpy(np.stack([j["context_outputs"] for j in jobs]).astype(np.int32)),
        "context_mask": torch.from_numpy(np.stack([j["context_mask"] for j in jobs]).astype(np.int32)),
        "target_height": torch.from_numpy(target_h).long(),
        "target_width": torch.from_numpy(target_w).long(),
    }
    _attach_derived_features(batch, config)
    return batch


def _mean_argmax_conf(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.amax(dim=-1).mean(dim=-1)


def _candidate_records_from_outputs(
    jobs: list[dict[str, Any]],
    outputs: dict[str, torch.Tensor],
    side: int,
    k: int,
) -> list[dict[str, Any]]:
    floor_logits = outputs.get("c2_floor_logits", outputs["logits"])
    source_logits: list[tuple[str, torch.Tensor]] = []
    for name in ("c2_candidate_logits", "c2_factored_candidate_logits", "logits"):
        if name in outputs:
            source_logits.append((name, outputs[name]))

    floor_tokens = floor_logits.argmax(dim=-1).detach().cpu()
    floor_scores = _mean_argmax_conf(floor_logits).detach().cpu()
    source_preds = [(name, logits.argmax(dim=-1).detach().cpu(), _mean_argmax_conf(logits).detach().cpu())
                    for name, logits in source_logits]

    records = []
    for row, job in enumerate(jobs):
        floor = floor_tokens[row].long()
        candidates = [floor.numpy().astype(np.int64)]
        votes = [float(floor_scores[row].item())]
        nonfloor: list[tuple[float, np.ndarray]] = []
        for _name, pred, score in source_preds:
            cand = pred[row].long()
            if bool(torch.equal(cand, floor)):
                continue
            if any(np.array_equal(cand.numpy(), existing) for existing in candidates):
                continue
            nonfloor.append((float(score[row].item()), cand.numpy().astype(np.int64)))
        nonfloor.sort(key=lambda item: item[0], reverse=True)
        for score, cand in nonfloor:
            if len(candidates) >= k:
                break
            if not any(np.array_equal(cand, existing) for existing in candidates):
                candidates.append(cand)
                votes.append(score)
        records.append(
            {
                "task_id": job["task_id"],
                "test_index": job["test_index"],
                "record_kind": job["record_kind"],
                "candidates": np.stack(candidates, axis=0).reshape(len(candidates), side * side),
                "vote_counts": np.asarray(votes, dtype=np.float32),
            }
        )
    return records


def _run_model_batch(
    core_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    config: pretrain.PretrainConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    gpu_batch = {k: v.to(device) for k, v in batch.items()}
    with torch.device(device.type):
        carry = core_model.initial_carry(gpu_batch)
    outputs = None
    with torch.inference_mode():
        for _ in range(int(config.arch.halt_max_steps)):
            carry, outputs = core_model(carry=carry, batch=gpu_batch)
    assert outputs is not None
    return {k: v.detach().cpu() for k, v in outputs.items() if k.endswith("logits") or k == "logits"}


def _load_config(config_path: Path, checkpoint: Path, dataset_path: Path, global_batch_size: int) -> pretrain.PretrainConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["load_checkpoint"] = str(checkpoint)
    raw["data_paths"] = [str(dataset_path)]
    raw["data_paths_test"] = []
    raw["eval_save_outputs"] = []
    raw["dataloader_num_workers"] = 0
    raw["checkpoint_path"] = None
    raw["global_batch_size"] = int(global_batch_size)
    raw["disable_compile"] = True
    return pretrain.PretrainConfig(**raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump typed TRM top-K fold/test candidates for V3 verify-select.")
    parser.add_argument("--config", required=True, help="YAML config/all_config used to instantiate the model.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--dataset", default="data/arc-agi-evaluation-full400-seed0-pid401aligned")
    parser.add_argument("--out", required=True, help="Output .npz path.")
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--no-folds", action="store_true", help="Do not write LOOCV fold records.")
    parser.add_argument("--no-tests", action="store_true", help="Do not write target-test records.")
    parser.add_argument("--no-blank-fold-pid", action="store_true", help="Use task PID in fold rows instead of blank PID.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = _repo_path(args.config, repo_root).resolve()
    checkpoint = _repo_path(args.checkpoint, repo_root).resolve()
    dataset_path = _repo_path(args.dataset, repo_root).resolve()
    out_path = _repo_path(args.out, repo_root).resolve()
    if not (dataset_path / "identifiers.json").exists() or not (dataset_path / "test_puzzles.json").exists():
        raise FileNotFoundError(f"Dataset must contain identifiers.json and test_puzzles.json: {dataset_path}")

    config = _load_config(config_path, checkpoint, dataset_path, args.global_batch_size)
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
    core_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for TRM candidate dumping.")

    identifiers = json.loads((dataset_path / "identifiers.json").read_text(encoding="utf-8"))
    test_puzzles = json.loads((dataset_path / "test_puzzles.json").read_text(encoding="utf-8"))
    context_limit = int(_arch_value(config, "c2_num_context", 0))
    if context_limit <= 0:
        raise ValueError("Model config has c2_num_context <= 0; no support-conditioned dump can be built.")

    records = []
    pending: list[dict[str, Any]] = []
    side = int(np.sqrt(train_metadata.seq_len))

    def flush() -> None:
        nonlocal pending, records
        if not pending:
            return
        batch = _batch_from_jobs(pending, config)
        outputs = _run_model_batch(core_model, batch, config, device)
        records.extend(_candidate_records_from_outputs(pending, outputs, side=side, k=args.k))
        pending = []

    n_tasks = 0
    for pid, task_id in enumerate(identifiers):
        task = test_puzzles.get(task_id)
        if task is None:
            continue
        n_tasks += 1
        pending.extend(
            _episode_jobs_for_task(
                task_id=task_id,
                pid=pid,
                task=task,
                context_limit=context_limit,
                blank_pid=int(train_metadata.blank_identifier_id),
                blank_fold_pid=not args.no_blank_fold_pid,
                include_folds=not args.no_folds,
                include_tests=not args.no_tests,
            )
        )
        while len(pending) >= config.global_batch_size:
            chunk, pending = pending[: config.global_batch_size], pending[config.global_batch_size :]
            batch = _batch_from_jobs(chunk, config)
            outputs = _run_model_batch(core_model, batch, config, device)
            records.extend(_candidate_records_from_outputs(chunk, outputs, side=side, k=args.k))
        if args.max_tasks and n_tasks >= args.max_tasks:
            break

    flush()
    if not records:
        raise RuntimeError("No candidate records were produced.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_model_dump(out_path, records, side=side)
    kinds = {}
    for rec in records:
        kinds[rec["record_kind"]] = kinds.get(rec["record_kind"], 0) + 1
    print(f"[dump-model-candidates] wrote {len(records)} records to {out_path}")
    print(f"[dump-model-candidates] kind counts: {kinds}")


if __name__ == "__main__":
    main()
