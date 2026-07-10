import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain
from models.losses import IGNORE_LABEL_ID


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in batch.items()}


def _reduce_metric_sums(total: Dict[str, float], metrics: Dict[str, torch.Tensor]):
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value.detach().cpu())


def _normalize(total: Dict[str, float]) -> Dict[str, float]:
    count = max(total.get("count", 0.0), 1.0)
    out = {"count": total.get("count", 0.0)}
    for key, value in total.items():
        if key == "count":
            continue
        out[key] = value if key.endswith("loss") else value / count
    return out


def _wrong_task_indices(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float]:
    puzzle_ids = batch["puzzle_identifiers"]
    batch_size = puzzle_ids.shape[0]
    indices = torch.arange(batch_size, device=puzzle_ids.device)
    if batch_size <= 1:
        return indices, 0.0

    chosen = torch.roll(indices, shifts=1)
    for shift in range(1, batch_size):
        candidate = torch.roll(indices, shifts=shift)
        need_wrong = puzzle_ids[chosen] == puzzle_ids
        candidate_wrong = puzzle_ids[candidate] != puzzle_ids
        chosen = torch.where(need_wrong & candidate_wrong, candidate, chosen)
    wrong_rate = (puzzle_ids[chosen] != puzzle_ids).float().mean()
    return chosen, float(wrong_rate.detach().cpu())


def _apply_context_mode(batch: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
    if not {"context_inputs", "context_outputs", "context_mask"}.issubset(batch.keys()):
        raise RuntimeError("Batch has no C2 context fields. Use arch.c2_num_context > 0.")

    out = _clone_batch(batch)
    if mode == "real":
        return out
    if mode == "zero":
        out["context_inputs"].zero_()
        out["context_outputs"].zero_()
        out["context_mask"].zero_()
        return out
    if mode == "input_only":
        out["context_outputs"].zero_()
        return out
    if mode == "permute":
        out["context_inputs"] = torch.flip(out["context_inputs"], dims=[1])
        out["context_outputs"] = torch.flip(out["context_outputs"], dims=[1])
        out["context_mask"] = torch.flip(out["context_mask"], dims=[1])
        return out
    if mode == "shuffle":
        wrong_indices, _wrong_rate = _wrong_task_indices(batch)
        out["context_inputs"] = batch["context_inputs"][wrong_indices].clone()
        out["context_outputs"] = batch["context_outputs"][wrong_indices].clone()
        out["context_mask"] = batch["context_mask"][wrong_indices].clone()
        return out
    raise ValueError(f"Unknown context mode: {mode}")


def _run_loss_head_on_batch(loss_head, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    with torch.device("cuda"):
        carry = loss_head.initial_carry(batch)
    while True:
        carry, _loss, metrics, _preds, all_finish = loss_head(
            carry=carry,
            batch=batch,
            return_keys=[],
        )
        if all_finish:
            return metrics


def _run_core_logits(core_model, batch: Dict[str, torch.Tensor], halt_max_steps: int) -> torch.Tensor:
    with torch.device("cuda"):
        carry = core_model.initial_carry(batch)
    outputs = None
    for _ in range(halt_max_steps):
        carry, outputs = core_model(carry=carry, batch=batch)
    assert outputs is not None
    return outputs["logits"]


def run_context_eval(loss_head, eval_loader, device: torch.device, mode: str, max_batches: int | None) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    processed = 0
    with torch.inference_mode():
        for _set_name, batch, _global_batch_size in eval_loader:
            processed += 1
            batch = _to_device(batch, device)
            batch = _apply_context_mode(batch, mode)
            metrics = _run_loss_head_on_batch(loss_head, batch)
            _reduce_metric_sums(totals, metrics)
            if max_batches is not None and processed >= max_batches:
                break
    return _normalize(totals)


def measure_leakage(eval_loader, device: torch.device, max_batches: int | None) -> float:
    leak_count = 0.0
    context_count = 0.0
    processed = 0
    with torch.inference_mode():
        for _set_name, batch, _global_batch_size in eval_loader:
            processed += 1
            batch = _to_device(batch, device)
            if not {"context_inputs", "context_outputs", "context_mask"}.issubset(batch.keys()):
                return 0.0
            target_outputs = torch.where(
                batch["labels"] == IGNORE_LABEL_ID,
                torch.zeros_like(batch["labels"]),
                batch["labels"],
            )
            same_input = (batch["context_inputs"] == batch["inputs"].unsqueeze(1)).all(dim=-1)
            same_output = (batch["context_outputs"] == target_outputs.unsqueeze(1)).all(dim=-1)
            valid_context = batch["context_mask"].to(torch.bool)
            leaks = same_input & same_output & valid_context
            leak_count += float(leaks.sum().detach().cpu())
            context_count += float(valid_context.sum().detach().cpu())
            if max_batches is not None and processed >= max_batches:
                break
    return leak_count / max(context_count, 1.0)


def measure_shuffle_wrong_rate(eval_loader, device: torch.device, max_batches: int | None) -> float:
    total = 0.0
    count = 0
    processed = 0
    with torch.inference_mode():
        for _set_name, batch, _global_batch_size in eval_loader:
            processed += 1
            batch = _to_device(batch, device)
            _indices, wrong_rate = _wrong_task_indices(batch)
            total += wrong_rate
            count += 1
            if max_batches is not None and processed >= max_batches:
                break
    return total / max(count, 1)


def measure_permutation_diff(core_model, eval_loader, device: torch.device, halt_max_steps: int, max_batches: int | None) -> float:
    max_diff = 0.0
    processed = 0
    with torch.inference_mode():
        for _set_name, batch, _global_batch_size in eval_loader:
            processed += 1
            batch = _to_device(batch, device)
            permuted = _apply_context_mode(batch, "permute")
            logits = _run_core_logits(core_model, batch, halt_max_steps)
            perm_logits = _run_core_logits(core_model, permuted, halt_max_steps)
            diff = (logits.float() - perm_logits.float()).abs().max()
            max_diff = max(max_diff, float(diff.detach().cpu()))
            if max_batches is not None and processed >= max_batches:
                break
    return max_diff


def _lodo_batch(batch: Dict[str, torch.Tensor], demo_idx: int, shuffle_context: bool) -> Dict[str, torch.Tensor] | None:
    context_mask = batch["context_mask"].to(torch.bool)
    valid_counts = context_mask.sum(dim=-1)
    valid_rows = context_mask[:, demo_idx] & (valid_counts >= 2)
    if not valid_rows.any():
        return None

    out = _clone_batch(batch)
    out["inputs"] = batch["context_inputs"][:, demo_idx].clone()
    labels = batch["context_outputs"][:, demo_idx].clone()
    labels = torch.where(labels == 0, torch.full_like(labels, IGNORE_LABEL_ID), labels)
    labels[~valid_rows] = IGNORE_LABEL_ID
    out["labels"] = labels

    out["context_mask"] = batch["context_mask"].clone()
    out["context_mask"][:, demo_idx] = 0
    out["context_mask"][~valid_rows] = 0
    if shuffle_context:
        out = _apply_context_mode(out, "shuffle")
    return out


def run_lodo_eval(loss_head, eval_loader, device: torch.device, max_batches: int | None, shuffle_context: bool) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    processed = 0
    with torch.inference_mode():
        for _set_name, batch, _global_batch_size in eval_loader:
            processed += 1
            batch = _to_device(batch, device)
            if not {"context_inputs", "context_outputs", "context_mask"}.issubset(batch.keys()):
                raise RuntimeError("Batch has no C2 context fields. Use arch.c2_num_context > 0.")
            for demo_idx in range(batch["context_inputs"].shape[1]):
                lodo_batch = _lodo_batch(batch, demo_idx, shuffle_context=shuffle_context)
                if lodo_batch is None:
                    continue
                metrics = _run_loss_head_on_batch(loss_head, lodo_batch)
                _reduce_metric_sums(totals, metrics)
            if max_batches is not None and processed >= max_batches:
                break
    return _normalize(totals)


def main():
    parser = argparse.ArgumentParser(description="Direct diagnostics for demo-conditioned C2 use.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TRM model construction currently requires CUDA.")

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["load_checkpoint"] = args.checkpoint
    raw.setdefault("checkpoint_path", None)
    config = pretrain.PretrainConfig(**raw)

    train_loader, train_metadata = pretrain.create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )
    eval_loader, _ = pretrain.create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )
    del train_loader

    loss_head, _, _ = pretrain.create_model(config, train_metadata, rank=0, world_size=1)
    loss_head.eval()
    core_model = loss_head.model
    core_model.eval()

    real = run_context_eval(loss_head, eval_loader, device, "real", args.max_batches)
    zero = run_context_eval(loss_head, eval_loader, device, "zero", args.max_batches)
    shuffled = run_context_eval(loss_head, eval_loader, device, "shuffle", args.max_batches)
    input_only = run_context_eval(loss_head, eval_loader, device, "input_only", args.max_batches)
    lodo = run_lodo_eval(loss_head, eval_loader, device, args.max_batches, shuffle_context=False)
    lodo_shuffle = run_lodo_eval(loss_head, eval_loader, device, args.max_batches, shuffle_context=True)
    leak_rate = measure_leakage(eval_loader, device, args.max_batches)
    shuffle_wrong_rate = measure_shuffle_wrong_rate(eval_loader, device, args.max_batches)
    perm_max_diff = measure_permutation_diff(core_model, eval_loader, device, config.arch.halt_max_steps, args.max_batches)

    result = {
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "leak_rate": leak_rate,
        "shuffle_wrong_task_rate": shuffle_wrong_rate,
        "context_count_mean": real.get("c2_context_count", 0.0),
        "perm_max_logit_diff": perm_max_diff,
        "real_ctx_content": real.get("content_accuracy", real.get("accuracy", 0.0)),
        "zero_ctx_content": zero.get("content_accuracy", zero.get("accuracy", 0.0)),
        "shuffle_ctx_content": shuffled.get("content_accuracy", shuffled.get("accuracy", 0.0)),
        "input_only_content": input_only.get("content_accuracy", input_only.get("accuracy", 0.0)),
        "real_ctx_exact": real.get("exact_accuracy", 0.0),
        "zero_ctx_exact": zero.get("exact_accuracy", 0.0),
        "shuffle_ctx_exact": shuffled.get("exact_accuracy", 0.0),
        "input_only_exact": input_only.get("exact_accuracy", 0.0),
        "delta_real_zero": real.get("content_accuracy", real.get("accuracy", 0.0)) - zero.get("content_accuracy", zero.get("accuracy", 0.0)),
        "delta_real_shuffle": real.get("content_accuracy", real.get("accuracy", 0.0)) - shuffled.get("content_accuracy", shuffled.get("accuracy", 0.0)),
        "delta_real_inputonly": real.get("content_accuracy", real.get("accuracy", 0.0)) - input_only.get("content_accuracy", input_only.get("accuracy", 0.0)),
        "lodo_content": lodo.get("content_accuracy", lodo.get("accuracy", 0.0)),
        "lodo_exact": lodo.get("exact_accuracy", 0.0),
        "lodo_majority_floor": lodo.get("majority_floor", 0.0),
        "lodo_shuffle_content": lodo_shuffle.get("content_accuracy", lodo_shuffle.get("accuracy", 0.0)),
        "lodo_real_shuffle_delta": lodo.get("content_accuracy", lodo.get("accuracy", 0.0)) - lodo_shuffle.get("content_accuracy", lodo_shuffle.get("accuracy", 0.0)),
        "c2_gate_patch_abs": real.get("c2_gate_patch_abs", abs(real.get("c2_gate_patch", 0.0))),
        "c2_gate_global_abs": real.get("c2_gate_global_abs", abs(real.get("c2_gate_global", 0.0))),
        "c2_update_norm_ratio": real.get("c2_update_norm_ratio", 0.0),
        "c2_patch_update_norm_ratio": real.get("c2_patch_update_norm_ratio", 0.0),
        "c2_global_update_norm_ratio": real.get("c2_global_update_norm_ratio", 0.0),
        "c2_rule_bank_token_count": real.get("c2_rule_bank_token_count", 0.0),
        "real_ctx": real,
        "zero_ctx": zero,
        "shuffle_ctx": shuffled,
        "input_only_ctx": input_only,
        "lodo": lodo,
        "lodo_shuffle": lodo_shuffle,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
