import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util

import numpy as np
import torch
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


def _load_checkpoint_source(checkpoint_path: str) -> None:
    checkpoint_dir = Path(checkpoint_path).resolve().parent
    source_map = {
        "models.recursive_reasoning.trm_fvr_c2": checkpoint_dir / "trm_fvr_c2.py",
        "models.losses_fvr": checkpoint_dir / "losses_fvr.py",
    }
    for module_name, source_path in source_map.items():
        if source_path.exists():
            _load_module_from_file(module_name, source_path)


def _crop_shape(seq: np.ndarray) -> Tuple[int, int]:
    grid = seq.reshape(30, 30)
    num_c = 30
    max_area = 0
    max_shape = (0, 0)
    for num_r in range(1, 31):
        for c in range(1, num_c + 1):
            x = int(grid[num_r - 1, c - 1])
            if x < 2 or x > 11:
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_shape = (num_r, num_c)
    return max_shape


def _row_metrics(preds: torch.Tensor, labels: torch.Tensor) -> List[Dict[str, float]]:
    valid_rows = (labels != IGNORE_LABEL_ID).sum(-1) > 0
    preds_np = preds[valid_rows].detach().cpu().numpy()
    raw_labels = labels[valid_rows].detach().cpu().numpy()
    label_masks = raw_labels != IGNORE_LABEL_ID
    labels_np = np.where(label_masks, raw_labels, 0)

    rows = []
    for pred_seq, label_seq, label_mask in zip(preds_np, labels_np, label_masks):
        true_h, true_w = _crop_shape(label_seq)
        pred_h, pred_w = _crop_shape(pred_seq)
        true_grid = label_seq.reshape(30, 30)
        pred_grid = pred_seq.reshape(30, 30)
        label_mask_grid = label_mask.reshape(30, 30)

        true_valid = (true_grid >= 2) & (true_grid <= 11)
        pred_valid = (pred_grid >= 2) & (pred_grid <= 11)
        true_eos = true_grid == 1
        pred_eos = pred_grid == 1
        outside = ~true_valid
        raw_n = max(int(label_mask.sum()), 1)
        inside_n = max(int(true_valid.sum()), 1)
        outside_n = max(int(outside.sum()), 1)
        oracle_pred_inside = pred_grid[:true_h, :true_w]
        oracle_true_inside = true_grid[:true_h, :true_w]
        oracle_inside_n = max(int(oracle_true_inside.size), 1)

        rows.append(
            {
                "raw_exact": float(np.array_equal(pred_seq[label_mask], label_seq[label_mask])),
                "raw_content_accuracy": float((pred_seq[label_mask] == label_seq[label_mask]).sum()) / raw_n,
                "oracle_shape_inside_exact": float(np.array_equal(oracle_pred_inside, oracle_true_inside)),
                "oracle_shape_inside_accuracy": float((oracle_pred_inside == oracle_true_inside).sum()) / oracle_inside_n,
                "shape_exact": float(pred_h == true_h and pred_w == true_w),
                "height_accuracy": float(pred_h == true_h),
                "width_accuracy": float(pred_w == true_w),
                "valid_mask_exact": float(np.array_equal(pred_valid, true_valid)),
                "eos_mask_exact": float(np.array_equal(pred_eos, true_eos)),
                "inside_canvas_color_accuracy": float(((pred_grid == true_grid) & true_valid).sum()) / inside_n,
                "outside_canvas_false_positive_rate": float((pred_valid & outside).sum()) / outside_n,
                "outside_boundary_accuracy": float((~pred_valid & outside).sum()) / outside_n,
                "full_sequence_exact": float(np.array_equal(pred_grid[label_mask_grid], true_grid[label_mask_grid])),
                "black_fraction_true_inside": float(((true_grid == 2) & true_valid).sum()) / inside_n,
                "black_fraction_pred_inside": float(((pred_grid == 2) & true_valid).sum()) / inside_n,
                "true_shape": [int(true_h), int(true_w)],
                "pred_shape": [int(pred_h), int(pred_w)],
            }
        )
    return rows


def _summarize(rows: List[Dict[str, float]], top_k: int) -> Dict[str, object]:
    if not rows:
        return {"count": 0}

    scalar_keys = [
        "raw_exact",
        "raw_content_accuracy",
        "oracle_shape_inside_exact",
        "oracle_shape_inside_accuracy",
        "shape_exact",
        "height_accuracy",
        "width_accuracy",
        "valid_mask_exact",
        "eos_mask_exact",
        "inside_canvas_color_accuracy",
        "outside_canvas_false_positive_rate",
        "outside_boundary_accuracy",
        "full_sequence_exact",
        "black_fraction_true_inside",
        "black_fraction_pred_inside",
    ]
    out: Dict[str, object] = {"count": len(rows)}
    for key in scalar_keys:
        out[key] = sum(float(row[key]) for row in rows) / len(rows)

    true_shapes = Counter(tuple(row["true_shape"]) for row in rows)
    true_heights = Counter(int(row["true_shape"][0]) for row in rows)
    true_widths = Counter(int(row["true_shape"][1]) for row in rows)
    pred_shapes = Counter(tuple(row["pred_shape"]) for row in rows)
    count = len(rows)
    out["height_majority_floor"] = max(true_heights.values()) / count
    out["width_majority_floor"] = max(true_widths.values()) / count
    out["shape_pair_majority_floor"] = max(true_shapes.values()) / count
    out["true_height_accuracy_floor"] = out["height_majority_floor"]
    out["true_width_accuracy_floor"] = out["width_majority_floor"]
    out["true_shape_pair_majority_floor"] = out["shape_pair_majority_floor"]
    out["oracle_shape_inside_note"] = "Diagnostic only: evaluates colors inside ground-truth H/W and ignores official canvas/EOS/PAD behavior."
    out["top_true_shapes"] = [
        {"shape": list(shape), "count": int(count)}
        for shape, count in true_shapes.most_common(top_k)
    ]
    out["top_pred_shapes"] = [
        {"shape": list(shape), "count": int(count)}
        for shape, count in pred_shapes.most_common(top_k)
    ]
    return out


def main():
    parser = argparse.ArgumentParser(description="Measure ARC output geometry and canvas errors per ACT step.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("TRM model construction currently requires CUDA.")

    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw["load_checkpoint"] = args.checkpoint
    config = pretrain.PretrainConfig(**raw)
    _load_checkpoint_source(args.checkpoint)

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

    by_step: Dict[int, List[Dict[str, float]]] = {
        step: [] for step in range(1, config.arch.halt_max_steps + 1)
    }
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
                preds = torch.argmax(outputs["logits"], dim=-1)
                by_step[step].extend(_row_metrics(preds, labels))
            if args.max_batches is not None and processed >= args.max_batches:
                break

    payload = {
        "config": str(Path(args.config)),
        "checkpoint": str(Path(args.checkpoint)),
        "steps": {
            str(step): _summarize(rows, top_k=args.top_k)
            for step, rows in sorted(by_step.items())
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
