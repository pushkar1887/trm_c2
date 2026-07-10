import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain
from models.losses import IGNORE_LABEL_ID
from scripts.fvr_structfuse_alpha_sweep import crop_shape, read_csv, row_metrics, write_csv


DEFAULT_TASKS = [
    "73c3b0d8",
    "c35c1b4c",
    "dd2401ed",
    "e5790162",
    "ecaa0ec1",
    "ce039d91",
    "29700607",
    "64a7c07e",
    "1acc24af",
]

ARC_COLORS = {
    -4: "#ffffff",  # blank / unused
    -3: "#f1f5f9",  # ignored
    -2: "#2f1b1b",  # EOS
    -1: "#cbd5e1",  # PAD
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}


def parse_tasks(raw: str) -> List[str]:
    if not raw.strip():
        return list(DEFAULT_TASKS)
    return [x.strip() for x in raw.split(",") if x.strip()]


def token_name(token: int) -> str:
    token = int(token)
    if token == IGNORE_LABEL_ID:
        return "IGNORE"
    if token == 0:
        return "PAD"
    if token == 1:
        return "EOS"
    if 2 <= token <= 11:
        return str(token - 2)
    return f"T{token}"


def token_to_display(token: int) -> int:
    token = int(token)
    if token == IGNORE_LABEL_ID:
        return -3
    if token == 0:
        return -1
    if token == 1:
        return -2
    if 2 <= token <= 11:
        return token - 2
    return -4


def seq_label(raw_label_seq: np.ndarray) -> np.ndarray:
    return np.where(raw_label_seq != IGNORE_LABEL_ID, raw_label_seq, 0)


def seq_to_display_grid(seq: np.ndarray, height: Optional[int] = None, width: Optional[int] = None) -> np.ndarray:
    if height is None or width is None:
        height, width = crop_shape(seq)
    height = max(int(height), 1)
    width = max(int(width), 1)
    grid = seq.reshape(30, 30)[:height, :width]
    return np.vectorize(token_to_display)(grid)


def arc_grid_to_display(grid: List[List[int]]) -> np.ndarray:
    if not grid:
        return np.zeros((1, 1), dtype=np.int64) - 4
    return np.asarray(grid, dtype=np.int64)


def valid_token(token: int) -> bool:
    return 2 <= int(token) <= 11


def error_kind(true_token: int, pred_token: int) -> str:
    if valid_token(true_token) and valid_token(pred_token):
        return "valid_to_valid"
    if valid_token(true_token) and int(pred_token) == 0:
        return "true_valid_pred_pad"
    if valid_token(true_token) and int(pred_token) == 1:
        return "true_valid_pred_eos"
    if int(true_token) == 1 and valid_token(pred_token):
        return "true_eos_pred_valid"
    if int(true_token) == 0 and valid_token(pred_token):
        return "true_pad_pred_valid"
    return "other"


def wrong_cell_rows(
    task_id: str,
    model_label: str,
    pred_seq: np.ndarray,
    raw_label_seq: np.ndarray,
    input_seq: np.ndarray,
) -> List[Dict[str, object]]:
    label_mask = raw_label_seq != IGNORE_LABEL_ID
    label_seq = seq_label(raw_label_seq)
    pred_grid = pred_seq.reshape(30, 30)
    label_grid = label_seq.reshape(30, 30)
    input_grid = input_seq.reshape(30, 30)
    wrong = (pred_seq != label_seq) & label_mask
    rows: List[Dict[str, object]] = []
    for flat_idx in np.nonzero(wrong)[0].tolist():
        r = int(flat_idx // 30)
        c = int(flat_idx % 30)
        true_token = int(label_grid[r, c])
        pred_token = int(pred_grid[r, c])
        input_token = int(input_grid[r, c])
        rows.append(
            {
                "task_id": task_id,
                "model": model_label,
                "row": r,
                "col": c,
                "input_token": input_token,
                "true_token": true_token,
                "pred_token": pred_token,
                "input_color": token_name(input_token),
                "true_color": token_name(true_token),
                "pred_color": token_name(pred_token),
                "transition": f"{token_name(true_token)}->{token_name(pred_token)}",
                "error_type": error_kind(true_token, pred_token),
                "changed_valid": bool(valid_token(true_token) and input_token != true_token),
            }
        )
    return rows


def suggested_classification(rows: List[Dict[str, object]], metrics: Dict[str, float]) -> str:
    wrong = len(rows)
    if wrong == 0:
        return "already_exact"
    valid_to_valid = sum(row["error_type"] == "valid_to_valid" for row in rows)
    missing = sum(row["error_type"] in {"true_valid_pred_pad", "true_valid_pred_eos"} for row in rows)
    extra = sum(row["error_type"] in {"true_pad_pred_valid", "true_eos_pred_valid"} for row in rows)
    shape_ok = float(metrics["shape_exact"]) > 0.5

    if shape_ok and valid_to_valid / max(wrong, 1) >= 0.75:
        return "recolour_mapping"
    if missing / max(wrong, 1) >= 0.50:
        return "missing_copy_or_object"
    if extra / max(wrong, 1) >= 0.50 or float(metrics["outside_canvas_fpr"]) > 0:
        return "extra_paint_or_failed_deletion"
    if not shape_ok:
        return "wrong_spatial_transform"
    return "mixed_or_unclear"


def summarize_task_prediction(
    task_id: str,
    model_label: str,
    pred_seq: np.ndarray,
    raw_label_seq: np.ndarray,
    input_seq: np.ndarray,
    bucket: str,
) -> Dict[str, object]:
    metrics = row_metrics(pred_seq, raw_label_seq, n_steps=16)
    wrong_rows = wrong_cell_rows(task_id, model_label, pred_seq, raw_label_seq, input_seq)
    transition_counts: Dict[str, int] = {}
    for row in wrong_rows:
        transition = str(row["transition"])
        transition_counts[transition] = transition_counts.get(transition, 0) + 1

    return {
        "task_id": task_id,
        "model": model_label,
        "bucket": bucket,
        "exact_accuracy": metrics["exact_accuracy"],
        "content_accuracy": metrics["content_accuracy"],
        "shape_exact": metrics["shape_exact"],
        "valid_mask_exact": metrics["valid_mask_exact"],
        "eos_mask_exact": metrics["eos_mask_exact"],
        "outside_canvas_fpr": metrics["outside_canvas_fpr"],
        "inside_canvas_color_acc": metrics["inside_canvas_color_acc"],
        "scored_wrong_cells": int(metrics["labelmasked_wrong_cells"]),
        "valid_to_valid_wrong_cells": int(metrics["labelmasked_valid_to_valid_wrong_cells"]),
        "changed_valid_wrong_cells": sum(bool(row["changed_valid"]) for row in wrong_rows),
        "transition_counts": "; ".join(f"{k}:{v}" for k, v in sorted(transition_counts.items())),
        "classification_label": "manual_pending",
        "suggested_classification": suggested_classification(wrong_rows, metrics),
    }


def prepare_config(config_path: Path, checkpoint_path: Path, out_dir: Path, batch_size: int) -> pretrain.PretrainConfig:
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["load_checkpoint"] = str(checkpoint_path)
    raw_config["data_paths"] = ["data/arc-agi-evaluation-full400-seed0"]
    raw_config["data_paths_test"] = []
    raw_config["eval_save_outputs"] = []
    raw_config["dataloader_num_workers"] = 0
    raw_config["checkpoint_path"] = str(out_dir / "noop_checkpoints")
    raw_config["run_name"] = "bothfail_forensics"
    raw_config["global_batch_size"] = int(batch_size)
    raw_config.setdefault("arch", {})["c2_structure_fusion_alpha"] = 0.0
    return pretrain.PretrainConfig(**raw_config)


def collect_predictions(
    model_label: str,
    config_path: Path,
    checkpoint_path: Path,
    task_ids: List[str],
    out_dir: Path,
    batch_size: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    print(f"[{model_label}] loading config={config_path}")
    config = prepare_config(config_path, checkpoint_path, out_dir, batch_size)
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
    eval_loader, _ = pretrain.create_dataloader(
        config,
        "test",
        0,
        1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    )
    loss_head, optimizers, _ = pretrain.create_model(config, train_metadata, rank=0, world_size=1)
    del optimizers
    core_model = loss_head.model
    core_model.eval()
    setattr(core_model.config, "c2_structure_fusion_alpha", 0.0)
    setattr(core_model.inner.config, "c2_structure_fusion_alpha", 0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this TRM eval path.")

    repo_root = Path(__file__).resolve().parents[1]
    eval_ids = json.loads((repo_root / "data" / "arc-agi-evaluation-full400-seed0" / "identifiers.json").read_text(encoding="utf-8"))
    wanted = set(task_ids)
    found: Dict[str, Dict[str, np.ndarray]] = {}

    with torch.inference_mode():
        for batch_idx, (_set_name, cpu_batch, _global_batch_size) in enumerate(eval_loader, start=1):
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            puzzle_ids = batch["puzzle_identifiers"].detach().cpu().numpy().tolist()
            candidate_rows = []
            for row_idx, pid in enumerate(puzzle_ids):
                pid = int(pid)
                if 0 < pid < len(eval_ids) and eval_ids[pid] in wanted and eval_ids[pid] not in found:
                    candidate_rows.append(row_idx)
            if not candidate_rows:
                continue

            with torch.device(device.type):
                carry = core_model.initial_carry(batch)
            outputs = None
            for _step in range(1, config.arch.halt_max_steps + 1):
                carry, outputs = core_model(carry=carry, batch=batch)
            preds = torch.argmax(outputs["logits"], dim=-1).detach().cpu().numpy()
            raw_labels = batch["labels"].detach().cpu().numpy()
            inputs = batch["inputs"].detach().cpu().numpy()

            for row_idx in candidate_rows:
                task_id = eval_ids[int(puzzle_ids[row_idx])]
                found[task_id] = {
                    "pred": preds[row_idx].copy(),
                    "label": raw_labels[row_idx].copy(),
                    "input": inputs[row_idx].copy(),
                }
                print(f"[{model_label}] captured {task_id} ({len(found)}/{len(wanted)})")
            if len(found) == len(wanted):
                break
            if batch_idx % 50 == 0:
                print(f"[{model_label}] batches={batch_idx}, found={len(found)}/{len(wanted)}")

    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"{model_label} missing tasks: {missing}")

    del loss_head, core_model
    torch.cuda.empty_cache()
    return found


def svg_text(x: int, y: int, text: str, size: int = 12, weight: str = "normal") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="#0f172a">{html.escape(text)}</text>'
    )


def panel_svg(x: int, y: int, title: str, grid: np.ndarray, cell: int = 14, max_side: int = 30) -> tuple[str, int, int]:
    rows, cols = grid.shape
    cell = max(6, min(cell, max_side * cell // max(max(rows, cols), 1)))
    title_h = 18
    width = cols * cell
    height = rows * cell + title_h
    parts = [svg_text(x, y + 12, title, size=12, weight="bold")]
    y0 = y + title_h
    for r in range(rows):
        for c in range(cols):
            val = int(grid[r, c])
            color = ARC_COLORS.get(val, "#ffffff")
            stroke = "#334155" if cell >= 8 else "#64748b"
            parts.append(
                f'<rect x="{x + c * cell}" y="{y0 + r * cell}" width="{cell}" height="{cell}" '
                f'fill="{color}" stroke="{stroke}" stroke-width="0.5"/>'
            )
    parts.append(
        f'<rect x="{x}" y="{y0}" width="{width}" height="{rows * cell}" '
        f'fill="none" stroke="#0f172a" stroke-width="1"/>'
    )
    return "\n".join(parts), width, height


def error_grid(pred_seq: np.ndarray, raw_label_seq: np.ndarray) -> np.ndarray:
    label_seq = seq_label(raw_label_seq)
    true_h, true_w = crop_shape(label_seq)
    pred_grid = pred_seq.reshape(30, 30)[:true_h, :true_w]
    label_grid = label_seq.reshape(30, 30)[:true_h, :true_w]
    mask_grid = raw_label_seq.reshape(30, 30)[:true_h, :true_w] != IGNORE_LABEL_ID
    err = np.zeros((true_h, true_w), dtype=np.int64)
    err[(pred_grid != label_grid) & mask_grid] = 2
    return err


def write_task_svg(
    path: Path,
    task_id: str,
    raw_task: Dict[str, object],
    predictions: Dict[str, Dict[str, np.ndarray]],
    summaries: List[Dict[str, object]],
) -> None:
    cell = 14
    margin = 18
    x = margin
    y = margin
    max_width = 1
    parts = [svg_text(x, y, f"Task {task_id}", size=18, weight="bold")]
    y += 24

    for demo_idx, demo in enumerate(raw_task["train"], start=1):
        in_grid = arc_grid_to_display(demo["input"])
        out_grid = arc_grid_to_display(demo["output"])
        p1, w1, h1 = panel_svg(x, y, f"demo {demo_idx} input", in_grid, cell=cell)
        p2, w2, h2 = panel_svg(x + w1 + 18, y, f"demo {demo_idx} output", out_grid, cell=cell)
        parts.extend([p1, p2])
        max_width = max(max_width, x + w1 + 18 + w2 + margin)
        y += max(h1, h2) + 14

    test_case = raw_task["test"][0]
    panels: List[tuple[str, np.ndarray]] = [
        ("test input", arc_grid_to_display(test_case["input"])),
        ("ground truth", arc_grid_to_display(test_case["output"])),
    ]
    for label, pred_info in predictions.items():
        label_seq = seq_label(pred_info["label"])
        pred_seq = pred_info["pred"]
        pred_h, pred_w = crop_shape(pred_seq)
        true_h, true_w = crop_shape(label_seq)
        panels.append((f"{label} prediction", seq_to_display_grid(pred_seq, pred_h, pred_w)))
        panels.append((f"{label} error mask", error_grid(pred_seq, pred_info["label"])))
        if pred_h != true_h or pred_w != true_w:
            panels.append((f"{label} pred-on-true-crop", seq_to_display_grid(pred_seq, true_h, true_w)))

    row_x = x
    row_h = 0
    wrap_width = 1300
    for title, grid in panels:
        p, w, h = panel_svg(row_x, y, title, grid, cell=cell)
        if row_x != x and row_x + w + margin > wrap_width:
            y += row_h + 18
            row_x = x
            row_h = 0
            p, w, h = panel_svg(row_x, y, title, grid, cell=cell)
        parts.append(p)
        row_x += w + 18
        row_h = max(row_h, h)
        max_width = max(max_width, row_x + margin)
    y += row_h + 20

    parts.append(svg_text(x, y, "Model summaries", size=14, weight="bold"))
    y += 18
    for summary in summaries:
        line = (
            f"{summary['model']}: wrong={summary['scored_wrong_cells']}, "
            f"valid_to_valid={summary['valid_to_valid_wrong_cells']}, "
            f"changed_valid={summary['changed_valid_wrong_cells']}, "
            f"suggested={summary['suggested_classification']}, manual={summary['classification_label']}"
        )
        parts.append(svg_text(x, y, line, size=12))
        y += 16

    width = int(max(max_width, 900))
    height = int(y + margin)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        "\n".join(parts),
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def write_report(
    out_dir: Path,
    task_ids: List[str],
    summary_rows: List[Dict[str, object]],
    c0_checkpoint: Path,
    c4_checkpoint: Optional[Path],
) -> None:
    by_task: Dict[str, List[Dict[str, object]]] = {}
    for row in summary_rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)

    lines = [
        "# Both-Fail Wrong-Cell Forensics",
        "",
        "This is an inference-only diagnostic. The listed eval tasks must not be used for training or loss-weight selection.",
        "",
        f"C0 checkpoint: `{c0_checkpoint}`",
        f"C4 checkpoint: `{c4_checkpoint}`" if c4_checkpoint else "C4 checkpoint: not provided",
        "",
        "## Task Figures",
        "",
    ]
    for task_id in task_ids:
        lines.append(f"### {task_id}")
        lines.append("")
        lines.append(f"![{task_id}]({task_id}_forensics.svg)")
        lines.append("")
        lines.append("| model | bucket | wrong | valid_to_valid | changed_valid | suggested | manual |")
        lines.append("|---|---:|---:|---:|---:|---|---|")
        for row in by_task.get(task_id, []):
            lines.append(
                f"| {row['model']} | {row['bucket']} | {row['scored_wrong_cells']} | "
                f"{row['valid_to_valid_wrong_cells']} | {row['changed_valid_wrong_cells']} | "
                f"{row['suggested_classification']} | {row['classification_label']} |"
            )
        lines.append("")

    (out_dir / "forensics_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render qualitative wrong-cell forensics for selected ARC both_fail tasks.")
    parser.add_argument("--c0-config", required=True)
    parser.add_argument("--c0-checkpoint", required=True)
    parser.add_argument("--c4-config")
    parser.add_argument("--c4-checkpoint")
    parser.add_argument("--reference-ledger", required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--global-batch-size", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    task_ids = parse_tasks(args.tasks)

    repo_root = Path(__file__).resolve().parents[1]
    raw_tasks = json.loads((repo_root / "data" / "arc-agi-evaluation-full400-seed0" / "test_puzzles.json").read_text(encoding="utf-8"))
    missing_raw = [task_id for task_id in task_ids if task_id not in raw_tasks]
    if missing_raw:
        raise RuntimeError(f"Tasks not found in canonical eval JSON: {missing_raw}")

    reference_rows = read_csv(Path(args.reference_ledger).resolve())
    ref_by_task = {row["task_id"]: row for row in reference_rows}

    all_predictions: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    all_predictions["C0"] = collect_predictions(
        "C0",
        Path(args.c0_config).resolve(),
        Path(args.c0_checkpoint).resolve(),
        task_ids,
        out_dir,
        args.global_batch_size,
    )

    c4_checkpoint = None
    if args.c4_config and args.c4_checkpoint:
        c4_checkpoint = Path(args.c4_checkpoint).resolve()
        all_predictions["C4"] = collect_predictions(
            "C4",
            Path(args.c4_config).resolve(),
            c4_checkpoint,
            task_ids,
            out_dir,
            args.global_batch_size,
        )

    wrong_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for task_id in task_ids:
        task_predictions = {model: preds[task_id] for model, preds in all_predictions.items()}
        task_summaries: List[Dict[str, object]] = []
        for model_label, pred_info in task_predictions.items():
            bucket = ref_by_task.get(task_id, {}).get("bucket", "unknown")
            summary = summarize_task_prediction(
                task_id,
                model_label,
                pred_info["pred"],
                pred_info["label"],
                pred_info["input"],
                bucket,
            )
            task_summaries.append(summary)
            summary_rows.append(summary)
            wrong_rows.extend(
                wrong_cell_rows(
                    task_id,
                    model_label,
                    pred_info["pred"],
                    pred_info["label"],
                    pred_info["input"],
                )
            )
        write_task_svg(
            out_dir / f"{task_id}_forensics.svg",
            task_id,
            raw_tasks[task_id],
            task_predictions,
            task_summaries,
        )

    summary_fields = [
        "task_id",
        "model",
        "bucket",
        "exact_accuracy",
        "content_accuracy",
        "shape_exact",
        "valid_mask_exact",
        "eos_mask_exact",
        "outside_canvas_fpr",
        "inside_canvas_color_acc",
        "scored_wrong_cells",
        "valid_to_valid_wrong_cells",
        "changed_valid_wrong_cells",
        "transition_counts",
        "classification_label",
        "suggested_classification",
    ]
    wrong_fields = [
        "task_id",
        "model",
        "row",
        "col",
        "input_token",
        "true_token",
        "pred_token",
        "input_color",
        "true_color",
        "pred_color",
        "transition",
        "error_type",
        "changed_valid",
    ]
    write_csv(out_dir / "task_summary.csv", summary_rows, summary_fields)
    write_csv(out_dir / "wrong_cells.csv", wrong_rows, wrong_fields)
    write_report(
        out_dir,
        task_ids,
        summary_rows,
        Path(args.c0_checkpoint).resolve(),
        c4_checkpoint,
    )

    print(f"[done] wrote {len(task_ids)} SVG figures")
    print(f"[done] task_summary={out_dir / 'task_summary.csv'}")
    print(f"[done] wrong_cells={out_dir / 'wrong_cells.csv'}")
    print(f"[done] report={out_dir / 'forensics_report.md'}")


if __name__ == "__main__":
    main()
