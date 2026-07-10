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


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _reduce_metric_sums(total: Dict[str, float], metrics: Dict[str, torch.Tensor]):
    for key, value in metrics.items():
        total[key] = total.get(key, 0.0) + float(value.detach().cpu())


def _normalize(total: Dict[str, float]) -> Dict[str, float]:
    count = max(total.get("count", 0.0), 1.0)
    out = {"count": total.get("count", 0.0)}
    for key, value in total.items():
        if key == "count":
            continue
        out[key] = value / count if not key.endswith("loss") else value
    return out


def run_eval(config: pretrain.PretrainConfig, blank_pid: bool, max_batches: int | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TRM pretrain.create_model currently constructs models on CUDA; run this on a CUDA machine.")

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

    model, _, _ = pretrain.create_model(config, train_metadata, rank=0, world_size=1)
    model.eval()

    totals: Dict[str, float] = {}
    processed = 0
    with torch.inference_mode():
        for _set_name, batch, _global_batch_size in eval_loader:
            processed += 1
            batch = _to_device(batch, device)
            if blank_pid:
                batch["puzzle_identifiers"] = torch.full_like(
                    batch["puzzle_identifiers"],
                    train_metadata.blank_identifier_id,
                )
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            while True:
                carry, _loss, metrics, _preds, all_finish = model(
                    carry=carry,
                    batch=batch,
                    return_keys=[],
                )
                if all_finish:
                    break
            _reduce_metric_sums(totals, metrics)
            if max_batches is not None and processed >= max_batches:
                break

    return _normalize(totals)


def main():
    parser = argparse.ArgumentParser(description="Evaluate real-PID vs blank-PID behavior with ACTLossHead metrics.")
    parser.add_argument("--config", required=True, help="Path to all_config.yaml or a pretrain config dump.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path to load.")
    parser.add_argument("--out", required=True, help="JSON output path.")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["load_checkpoint"] = args.checkpoint
    raw.setdefault("checkpoint_path", None)
    config = pretrain.PretrainConfig(**raw)

    real = run_eval(config, blank_pid=False, max_batches=args.max_batches)
    blank = run_eval(config, blank_pid=True, max_batches=args.max_batches)
    result = {
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "real_pid": real,
        "blank_pid": blank,
        "delta_exact_accuracy": real.get("exact_accuracy", 0.0) - blank.get("exact_accuracy", 0.0),
        "delta_accuracy": real.get("accuracy", 0.0) - blank.get("accuracy", 0.0),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
