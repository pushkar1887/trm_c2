import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain
from models.losses import IGNORE_LABEL_ID


def _step_metrics(outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> Dict[str, float]:
    preds = torch.argmax(outputs["logits"], dim=-1)
    mask = labels != IGNORE_LABEL_ID
    loss_counts = mask.sum(-1)
    valid = loss_counts > 0
    correct = mask & (preds == labels)
    seq_correct = correct.sum(-1) == loss_counts
    count = float(valid.sum().detach().cpu())
    if count <= 0:
        return {"count": 0.0, "accuracy": 0.0, "exact_accuracy": 0.0}
    accuracy = torch.where(
        valid,
        (correct.float() / loss_counts.clamp_min(1).unsqueeze(-1)).sum(-1),
        torch.zeros_like(loss_counts, dtype=torch.float32),
    ).sum()
    valid_pred_values = preds[mask]
    unique_preds = torch.unique(valid_pred_values).numel() if valid_pred_values.numel() else 0
    return {
        "count": count,
        "accuracy": float(accuracy.detach().cpu()) / count,
        "exact_accuracy": float((valid & seq_correct).sum().detach().cpu()) / count,
        "pred_unique_classes": float(unique_preds),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure per-ACT-step prediction quality.")
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
    core_model = loss_head.model
    core_model.eval()

    totals: Dict[int, Dict[str, float]] = {}
    processed = 0
    with torch.inference_mode():
        for _set_name, batch, _global_batch_size in eval_loader:
            processed += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            with torch.device("cuda"):
                carry = core_model.initial_carry(batch)
            for step in range(1, config.arch.halt_max_steps + 1):
                carry, outputs = core_model(carry=carry, batch=batch)
                metrics = _step_metrics(outputs, labels)
                dst = totals.setdefault(step, {"count": 0.0, "accuracy": 0.0, "exact_accuracy": 0.0, "pred_unique_classes": 0.0})
                count = metrics["count"]
                dst["accuracy"] += metrics["accuracy"] * count
                dst["exact_accuracy"] += metrics["exact_accuracy"] * count
                dst["pred_unique_classes"] += metrics["pred_unique_classes"] * count
                dst["count"] += count
            if args.max_batches is not None and processed >= args.max_batches:
                break

    results = {}
    for step, values in sorted(totals.items()):
        count = max(values["count"], 1.0)
        results[str(step)] = {
            "count": values["count"],
            "accuracy": values["accuracy"] / count,
            "exact_accuracy": values["exact_accuracy"] / count,
            "pred_unique_classes": values["pred_unique_classes"] / count,
        }

    payload = {
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "steps": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
