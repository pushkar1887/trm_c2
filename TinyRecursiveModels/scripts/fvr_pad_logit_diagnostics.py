import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn.functional as F
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain
from models.losses import IGNORE_LABEL_ID


def _load_module_from_file(module_name: str, source_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_name} from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _load_checkpoint_source(checkpoint_path: str | None) -> None:
    if checkpoint_path is None:
        return
    checkpoint_dir = Path(checkpoint_path).resolve().parent
    source_map = {
        "models.recursive_reasoning.trm_fvr_c2": checkpoint_dir / "trm_fvr_c2.py",
        "models.losses_fvr": checkpoint_dir / "losses_fvr.py",
    }
    for module_name, source_path in source_map.items():
        if source_path.exists():
            _load_module_from_file(module_name, source_path)


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def _wrong_task_indices(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    puzzle_ids = batch["puzzle_identifiers"]
    indices = torch.arange(puzzle_ids.shape[0], device=puzzle_ids.device)
    if indices.numel() <= 1:
        return indices
    chosen = torch.roll(indices, shifts=1)
    for shift in range(1, indices.numel()):
        candidate = torch.roll(indices, shifts=shift)
        need_wrong = puzzle_ids[chosen] == puzzle_ids
        candidate_wrong = puzzle_ids[candidate] != puzzle_ids
        chosen = torch.where(need_wrong & candidate_wrong, candidate, chosen)
    return chosen


def _apply_context_mode(batch: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
    if mode == "real":
        return batch
    if not {"context_inputs", "context_outputs", "context_mask"}.issubset(batch.keys()):
        raise RuntimeError("Batch has no C2 context fields. Use arch.c2_num_context > 0.")
    out = _clone_batch(batch)
    if mode == "zero":
        out["context_inputs"].zero_()
        out["context_outputs"].zero_()
        out["context_mask"].zero_()
        return out
    if mode == "input_only":
        out["context_outputs"].zero_()
        return out
    if mode == "shuffle":
        wrong_indices = _wrong_task_indices(batch)
        out["context_inputs"] = batch["context_inputs"][wrong_indices].clone()
        out["context_outputs"] = batch["context_outputs"][wrong_indices].clone()
        out["context_mask"] = batch["context_mask"][wrong_indices].clone()
        return out
    raise ValueError(f"Unknown context mode: {mode}")


def _mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().detach().cpu())


def _std(values: torch.Tensor) -> float:
    if values.numel() <= 1:
        return 0.0
    return float(values.float().std(unbiased=False).detach().cpu())


def _frac(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().detach().cpu())


def _class_hist(preds: torch.Tensor, mask: torch.Tensor, vocab_size: int, top_k: int) -> list[Dict[str, float]]:
    if not mask.any():
        return []
    values = preds[mask].detach().cpu().tolist()
    count = len(values)
    return [
        {"token": int(token), "count": int(freq), "fraction": float(freq / max(count, 1))}
        for token, freq in Counter(values).most_common(top_k)
        if 0 <= int(token) < vocab_size
    ]


def _position_agreement(inputs: torch.Tensor, labels: torch.Tensor, valid_rows: torch.Tensor) -> Dict[str, float]:
    row_mask = valid_rows.unsqueeze(-1)
    input_pad = (inputs == 0) & row_mask
    output_pad = (labels == IGNORE_LABEL_ID) & row_mask
    input_nonpad = (inputs != 0) & row_mask
    output_nonpad = (labels != IGNORE_LABEL_ID) & row_mask

    output_pad_count = output_pad.sum().clamp_min(1)
    input_pad_count = input_pad.sum().clamp_min(1)
    row_token_count = row_mask.expand_as(labels).sum().clamp_min(1)
    return {
        "input_pad_fraction": float(input_pad.sum().detach().cpu()) / float(row_token_count.detach().cpu()),
        "output_pad_fraction": float(output_pad.sum().detach().cpu()) / float(row_token_count.detach().cpu()),
        "output_pad_with_input_pad_frac": float((output_pad & input_pad).sum().detach().cpu()) / float(output_pad_count.detach().cpu()),
        "output_pad_with_input_nonpad_frac": float((output_pad & input_nonpad).sum().detach().cpu()) / float(output_pad_count.detach().cpu()),
        "input_pad_with_output_pad_frac": float((input_pad & output_pad).sum().detach().cpu()) / float(input_pad_count.detach().cpu()),
        "input_pad_with_output_nonpad_frac": float((input_pad & output_nonpad).sum().detach().cpu()) / float(input_pad_count.detach().cpu()),
    }


def _masked_logit_stats(
    logits: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    prefix: str,
    top_k: int,
) -> Dict[str, object]:
    vocab_size = logits.shape[-1]
    if not mask.any():
        return {
            f"{prefix}_count": 0,
            f"{prefix}_fraction": 0.0,
        }

    local = logits[mask].float()
    local_preds = preds[mask]
    pad_logit = local[:, 0]
    eos_logit = local[:, 1] if vocab_size > 1 else torch.zeros_like(pad_logit)
    color_logits = local[:, 2:] if vocab_size > 2 else local[:, :0]
    best_color_logit = color_logits.max(dim=-1).values if color_logits.numel() else torch.zeros_like(pad_logit)
    best_nonpad_logit = local[:, 1:].max(dim=-1).values if vocab_size > 1 else torch.full_like(pad_logit, -1e9)
    pad_rank = 1 + (local > pad_logit.unsqueeze(-1)).sum(dim=-1)
    prob = torch.softmax(local, dim=-1)

    out: Dict[str, object] = {
        f"{prefix}_count": int(mask.sum().detach().cpu()),
        f"{prefix}_pad_logit_mean": _mean(pad_logit),
        f"{prefix}_pad_logit_std": _std(pad_logit),
        f"{prefix}_eos_logit_mean": _mean(eos_logit),
        f"{prefix}_best_color_logit_mean": _mean(best_color_logit),
        f"{prefix}_best_nonpad_logit_mean": _mean(best_nonpad_logit),
        f"{prefix}_pad_margin_vs_best_nonpad_mean": _mean(pad_logit - best_nonpad_logit),
        f"{prefix}_pad_margin_vs_best_color_mean": _mean(pad_logit - best_color_logit),
        f"{prefix}_eos_margin_vs_pad_mean": _mean(eos_logit - pad_logit),
        f"{prefix}_best_color_margin_vs_pad_mean": _mean(best_color_logit - pad_logit),
        f"{prefix}_pad_prob_mean": _mean(prob[:, 0]),
        f"{prefix}_pad_rank_mean": _mean(pad_rank.float()),
        f"{prefix}_pad_top1_rate": _frac(pad_rank == 1),
        f"{prefix}_pad_top2_rate": _frac(pad_rank <= 2),
        f"{prefix}_pred_pad_rate": _frac(local_preds == 0),
        f"{prefix}_pred_eos_rate": _frac(local_preds == 1),
        f"{prefix}_pred_color_rate": _frac(local_preds >= 2),
        f"{prefix}_pred_hist": _class_hist(preds, mask, vocab_size, top_k),
    }

    safe_labels = labels[mask].long()
    if prefix == "pad":
        out[f"{prefix}_target_ce"] = float(F.cross_entropy(local, torch.zeros_like(safe_labels)).detach().cpu())
    elif prefix == "eos":
        out[f"{prefix}_target_ce"] = float(F.cross_entropy(local, torch.ones_like(safe_labels)).detach().cpu())
    elif prefix == "color":
        out[f"{prefix}_target_ce"] = float(F.cross_entropy(local, safe_labels).detach().cpu())
        out[f"{prefix}_target_accuracy"] = _frac(local_preds == safe_labels)
        out[f"{prefix}_pred_pad_on_target_rate"] = _frac(local_preds == 0)
    return out


def _step_stats(logits: torch.Tensor, batch: Dict[str, torch.Tensor], top_k: int) -> Dict[str, object]:
    labels = batch["labels"]
    inputs = batch["inputs"]
    preds = logits.argmax(dim=-1)
    valid_rows = (labels != IGNORE_LABEL_ID).sum(dim=-1) > 0
    row_mask = valid_rows.unsqueeze(-1)
    pad_mask = (labels == IGNORE_LABEL_ID) & row_mask
    eos_mask = (labels == 1) & row_mask
    color_mask = (labels >= 2) & row_mask
    row_token_count = row_mask.expand_as(labels).sum().clamp_min(1)

    out: Dict[str, object] = {
        "rows": int(valid_rows.sum().detach().cpu()),
        "tokens_per_row": int(labels.shape[-1]),
        "pad_fraction": float(pad_mask.sum().detach().cpu()) / float(row_token_count.detach().cpu()),
        "eos_fraction": float(eos_mask.sum().detach().cpu()) / float(row_token_count.detach().cpu()),
        "color_fraction": float(color_mask.sum().detach().cpu()) / float(row_token_count.detach().cpu()),
    }
    out.update(_position_agreement(inputs, labels, valid_rows))
    out.update(_masked_logit_stats(logits, preds, labels, pad_mask, "pad", top_k))
    out.update(_masked_logit_stats(logits, preds, labels, eos_mask, "eos", top_k))
    out.update(_masked_logit_stats(logits, preds, labels, color_mask, "color", top_k))
    return out


def _run_steps(core_model, batch: Dict[str, torch.Tensor], halt_max_steps: int, steps: Iterable[int], top_k: int) -> Dict[str, Dict[str, object]]:
    wanted = set(int(step) for step in steps)
    if not wanted:
        wanted = {halt_max_steps}
    max_step = max(wanted)
    with torch.device("cuda"):
        carry = core_model.initial_carry(batch)
    outputs = None
    by_step = {}
    for step in range(1, max_step + 1):
        carry, outputs = core_model(carry=carry, batch=batch)
        if step in wanted:
            assert outputs is not None
            by_step[str(step)] = _step_stats(outputs["logits"], batch, top_k)
    return by_step


def _metric_weight_key(key: str) -> str:
    if key.startswith("pad_") and key != "pad_count" and not key.endswith("_hist"):
        return "pad_count"
    if key.startswith("eos_") and key != "eos_count" and not key.endswith("_hist"):
        return "eos_count"
    if key.startswith("color_") and key != "color_count" and not key.endswith("_hist"):
        return "color_count"
    return "rows"


def _merge_step_stats(accum: Dict[str, Dict[str, object]], incoming: Dict[str, Dict[str, object]]) -> None:
    for step, values in incoming.items():
        if step not in accum:
            accum[step] = {"_weights": {}}
        weights = accum[step]["_weights"]
        for key, value in values.items():
            if isinstance(value, list):
                accum[step].setdefault(key, [])
                accum[step][key].extend(value)
            elif isinstance(value, (int, float)):
                if key in ("rows", "pad_count", "eos_count", "color_count"):
                    accum[step][key] = accum[step].get(key, 0.0) + float(value)
                elif key == "tokens_per_row":
                    accum[step][key] = float(value)
                else:
                    weight_key = _metric_weight_key(key)
                    weight = float(values.get(weight_key, values.get("rows", 1.0)))
                    accum[step][key] = accum[step].get(key, 0.0) + float(value) * weight
                    weights[key] = weights.get(key, 0.0) + weight
            else:
                accum[step][key] = value


def _finalize(accum: Dict[str, Dict[str, object]], top_k: int) -> Dict[str, Dict[str, object]]:
    finalized = {}
    for step, values in sorted(accum.items(), key=lambda item: int(item[0])):
        weights = values.pop("_weights", {})
        out: Dict[str, object] = {}
        for key, value in values.items():
            if isinstance(value, list):
                if key.endswith("_hist"):
                    token_counts: Counter[int] = Counter()
                    total = 0
                    for row in value:
                        token = int(row["token"])
                        count = int(row["count"])
                        token_counts[token] += count
                        total += count
                    out[key] = [
                        {"token": int(token), "count": int(count), "fraction": float(count / max(total, 1))}
                        for token, count in token_counts.most_common(top_k)
                    ]
                else:
                    out[key] = value
            else:
                if key in ("rows", "pad_count", "eos_count", "color_count", "tokens_per_row"):
                    out[key] = value
                else:
                    out[key] = value / max(float(weights.get(key, 1.0)), 1.0)
        finalized[step] = out
    return finalized


def _round_payload(value: object, digits: int | None) -> object:
    if digits is None or digits < 0:
        return value
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round_payload(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_payload(item, digits) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="C2-only PAD/EOS/color logit diagnostics. No blank-PID A5 path.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--context-mode", choices=["real", "zero", "shuffle", "input_only"], default="real")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--steps", default="final", help="Comma-separated ACT steps, 'all', or 'final'.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--round-digits", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TRM model construction currently requires CUDA.")

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if args.checkpoint:
        raw["load_checkpoint"] = args.checkpoint
    else:
        raw["load_checkpoint"] = None
    raw.setdefault("checkpoint_path", None)
    config = pretrain.PretrainConfig(**raw)
    _load_checkpoint_source(args.checkpoint)

    train_loader, train_metadata = pretrain.create_dataloader(
        config,
        "train",
        rank=0,
        world_size=1,
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    eval_loader, _ = pretrain.create_dataloader(
        config,
        "test",
        rank=0,
        world_size=1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    del train_loader

    loss_head, _, _ = pretrain.create_model(config, train_metadata, rank=0, world_size=1)
    core_model = loss_head.model
    core_model.eval()

    if args.steps == "all":
        steps = list(range(1, config.arch.halt_max_steps + 1))
    elif args.steps == "final":
        steps = [config.arch.halt_max_steps]
    else:
        steps = [int(item.strip()) for item in args.steps.split(",") if item.strip()]

    accum: Dict[str, Dict[str, object]] = {}
    processed = 0
    with torch.inference_mode():
        for _set_name, batch, _global_batch_size in eval_loader:
            processed += 1
            batch = _to_device(batch, device)
            batch = _apply_context_mode(batch, args.context_mode)
            step_stats = _run_steps(core_model, batch, config.arch.halt_max_steps, steps, args.top_k)
            _merge_step_stats(accum, step_stats)
            if args.max_batches is not None and processed >= args.max_batches:
                break

    payload = {
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)) if args.checkpoint else None,
        "context_mode": args.context_mode,
        "note": "C2-only PAD/EOS/color logit diagnostic. It does not run blank-puzzle-ID A5 evaluation.",
        "steps": _finalize(accum, args.top_k),
    }
    payload = _round_payload(payload, args.round_digits)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
