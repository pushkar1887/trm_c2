import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.build_arc_dataset import arc_grid_to_np, np_grid_to_seq_translational_augment


DEFAULT_CACHE_DIR = Path("visual_cache/dinov2_large_arc1concept_aug1000")
DEFAULT_AUG_DATA = Path("D:/trm_c2/arc1concept-aug-1000")
DEFAULT_EVAL_DATA = Path("data/arc-agi-evaluation-full400-seed0")
DEFAULT_CANONICAL_DATA = Path("data/arc-agi-evaluation-full400-seed0")
MODEL_NAME = "facebook/dinov2-large"
MODEL_SPECS: Dict[str, Dict[str, object]] = {
    "facebook/dinov2-large": {
        "family": "dino",
        "feature_shape": [256, 1024],
        "image_side": 224,
        "cache_level": "dino_patch_16x16",
        "title": "DINO Visual Cache",
    },
    "google/siglip2-base-patch16-256": {
        "family": "siglip2",
        "feature_shape": [256, 768],
        "image_side": 256,
        "cache_level": "siglip2_patch_16x16",
        "title": "SigLIP2 Visual Cache",
    },
}


def model_spec(model_name: str) -> Dict[str, object]:
    if model_name not in MODEL_SPECS:
        raise ValueError(
            f"Unsupported visual model {model_name!r}. "
            f"Supported: {', '.join(sorted(MODEL_SPECS))}"
        )
    return MODEL_SPECS[model_name]


def feature_shape_for(model_name: str) -> List[int]:
    return list(model_spec(model_name)["feature_shape"])  # type: ignore[index,return-value]


def feature_bytes_per_grid(model_name: str) -> int:
    shape = feature_shape_for(model_name)
    return int(shape[0]) * int(shape[1]) * 2


def sha1_tokens(tokens: np.ndarray) -> str:
    arr = np.asarray(tokens, dtype=np.uint8).reshape(-1)
    if arr.shape[0] != 900:
        raise ValueError(f"Expected 900-token grid, got {arr.shape[0]}")
    return hashlib.sha1(arr.tobytes()).hexdigest()


def iter_token_sources(
    dataset_root: Path,
    splits: Iterable[str],
    fields: Iterable[str] = ("inputs", "labels"),
) -> Iterable[Tuple[str, Path, int, np.ndarray]]:
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for field in fields:
            path = split_dir / f"all__{field}.npy"
            if not path.exists():
                continue
            arr = np.load(path, mmap_mode="r")
            if arr.ndim != 2 or arr.shape[1] != 900:
                raise ValueError(f"Expected {path} shape [N,900], got {arr.shape}")
            for row_idx in range(arr.shape[0]):
                yield f"{dataset_root.name}:{split}:{field}", path, row_idx, np.asarray(arr[row_idx])


def iter_eval_json_context_sources(eval_data: Path) -> Iterable[Tuple[str, Path, int, np.ndarray]]:
    """Yield the exact demo context grids loaded by PuzzleDataset._load_test_context."""
    path = eval_data / "test_puzzles.json"
    if not path.exists():
        return
    puzzles = json.loads(path.read_text(encoding="utf-8"))
    row_idx = 0
    for task_id, puzzle in puzzles.items():
        for demo_idx, example in enumerate(puzzle.get("train", [])):
            inp, out = np_grid_to_seq_translational_augment(
                arc_grid_to_np(example["input"]),
                arc_grid_to_np(example["output"]),
                do_translation=False,
            )
            yield f"{eval_data.name}:test_puzzles:{task_id}:train:{demo_idx}:input", path, row_idx, inp
            row_idx += 1
            yield f"{eval_data.name}:test_puzzles:{task_id}:train:{demo_idx}:output", path, row_idx, out
            row_idx += 1


def collect_unique_sources(sources: Iterable[Tuple[str, Path, int, np.ndarray]]) -> Dict[str, Dict[str, object]]:
    unique: Dict[str, Dict[str, object]] = {}
    token_min = 999
    token_max = -999
    rows_seen = 0
    for source_name, path, row_idx, tokens in sources:
        rows_seen += 1
        token_min = min(token_min, int(tokens.min()))
        token_max = max(token_max, int(tokens.max()))
        digest = sha1_tokens(tokens)
        if digest not in unique:
            row: Dict[str, object] = {
                "source": source_name,
                "token_file": str(path),
                "row_index": int(row_idx),
            }
            if path.name == "test_puzzles.json" or not path.exists():
                row["tokens"] = np.asarray(tokens, dtype=np.uint8).reshape(-1).tolist()
            unique[digest] = row
        if rows_seen % 250_000 == 0:
            print(f"[census] rows={rows_seen:,} unique={len(unique):,}", flush=True)
    if token_min < 0 or token_max > 11:
        raise ValueError(f"Token range must be 0..11, got min={token_min} max={token_max}")
    print(f"[census] rows={rows_seen:,} unique={len(unique):,} token_range={token_min}..{token_max}", flush=True)
    return unique


def write_census(cache_dir: Path, scope_name: str, unique_count: int, model_name: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    spec = model_spec(model_name)
    feature_shape = feature_shape_for(model_name)
    estimated_bytes = unique_count * feature_bytes_per_grid(model_name)
    free_bytes = shutil.disk_usage(cache_dir.resolve().anchor or ".").free
    payload = {
        "scope": scope_name,
        "model": model_name,
        "unique_grid_count": unique_count,
        "feature_shape": feature_shape,
        "cache_level": spec["cache_level"],
        "dtype": "float16",
        "estimated_cache_bytes": estimated_bytes,
        "estimated_cache_gib": estimated_bytes / (1024**3),
        "disk_free_bytes": free_bytes,
        "disk_free_gib": free_bytes / (1024**3),
        "fits_current_disk": estimated_bytes < free_bytes,
    }
    (cache_dir / "cache_census.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Visual Cache Census",
        "",
        f"- scope: `{scope_name}`",
        f"- model: `{model_name}`",
        f"- unique grids: `{unique_count}`",
        f"- feature shape: `{feature_shape}`",
        f"- cache level: `{spec['cache_level']}`",
        "- dtype: `float16`",
        f"- estimated cache size: `{payload['estimated_cache_gib']:.2f} GiB`",
        f"- disk free: `{payload['disk_free_gib']:.2f} GiB`",
        f"- fits current disk: `{payload['fits_current_disk']}`",
    ]
    (cache_dir / "cache_census.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_tokens(path: Path, row_idx: int) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    row = np.asarray(arr[row_idx], dtype=np.uint8)
    if row.shape != (900,):
        raise ValueError(f"Expected row shape (900,), got {row.shape} from {path}:{row_idx}")
    if row.min() < 0 or row.max() > 11:
        raise ValueError(f"Token range must be 0..11 in {path}:{row_idx}")
    return row


def load_tokens_from_meta(meta: Dict[str, object]) -> np.ndarray:
    if "tokens" in meta:
        row = np.asarray(meta["tokens"], dtype=np.uint8)
        if row.shape != (900,):
            raise ValueError(f"Expected inline token shape (900,), got {row.shape}")
        if row.min() < 0 or row.max() > 11:
            raise ValueError("Inline token range must be 0..11")
        return row
    return load_tokens(Path(str(meta["token_file"])), int(meta["row_index"]))


def tokens_to_pixel_values(tokens: torch.Tensor, processor, image_side: int = 224) -> torch.Tensor:
    # tokens: [B,900] in 0..11
    palette = torch.tensor(
        [
            [0, 0, 0],
            [255, 255, 255],
            [0, 0, 0],
            [0, 116, 217],
            [255, 65, 54],
            [46, 204, 64],
            [255, 220, 0],
            [170, 170, 170],
            [240, 18, 190],
            [255, 133, 27],
            [127, 219, 255],
            [135, 12, 37],
        ],
        dtype=torch.float32,
        device=tokens.device,
    ) / 255.0
    rgb = palette[tokens.long()]
    rgb = rgb.reshape(tokens.shape[0], 30, 30, 3).permute(0, 3, 1, 2).contiguous()
    rgb = F.interpolate(rgb, size=(image_side, image_side), mode="nearest")
    image_mean = getattr(processor, "image_mean", [0.5, 0.5, 0.5])
    image_std = getattr(processor, "image_std", [0.5, 0.5, 0.5])
    mean = torch.tensor(image_mean, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    std = torch.tensor(image_std, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    return (rgb - mean) / std


def load_visual_model(model_name: str, device: torch.device):
    spec = model_spec(model_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    if spec["family"] == "siglip2":
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_name)
        full_model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, use_safetensors=True)
        if not hasattr(full_model, "vision_model"):
            raise TypeError(f"Expected {model_name} AutoModel to expose vision_model.")
        model = full_model.vision_model
    else:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, use_safetensors=True)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return processor, model


def load_dino(device: torch.device):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype, use_safetensors=True)
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return processor, model


@torch.inference_mode()
def compute_visual_features(
    tokens_np: np.ndarray,
    processor,
    model,
    device: torch.device,
    model_name: str,
) -> torch.Tensor:
    spec = model_spec(model_name)
    feature_shape = tuple(feature_shape_for(model_name))
    image_side = int(spec["image_side"])
    tokens = torch.from_numpy(np.ascontiguousarray(tokens_np, dtype=np.uint8).copy()).to(device)
    pixel_values = tokens_to_pixel_values(tokens, processor, image_side=image_side).to(
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    out = model(pixel_values=pixel_values).last_hidden_state
    if out.shape[1] == feature_shape[0] + 1:
        out = out[:, 1:, :]
    if tuple(out.shape[1:]) != feature_shape:
        raise RuntimeError(
            f"Expected {model_name} output [B,{feature_shape[0]},{feature_shape[1]}], got {tuple(out.shape)}"
        )
    if not torch.isfinite(out).all():
        raise RuntimeError(f"{model_name} produced non-finite features.")
    return out.detach().cpu().to(torch.float16)


@torch.inference_mode()
def compute_dino_features(tokens_np: np.ndarray, processor, model, device: torch.device) -> torch.Tensor:
    return compute_visual_features(tokens_np, processor, model, device, MODEL_NAME)


def build_cache(
    cache_dir: Path,
    unique_sources: Dict[str, Dict[str, object]],
    batch_size: int,
    shard_size: int,
    force: bool,
    model_name: str,
) -> None:
    manifest_path = cache_dir / "manifest.jsonl"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"Cache manifest already exists: {manifest_path}. Use --force to overwrite.")
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_visual_model(model_name, device)
    spec = model_spec(model_name)
    feature_shape = feature_shape_for(model_name)

    entries = list(unique_sources.items())
    manifest_rows: List[Dict[str, object]] = []
    shard_tensors: List[torch.Tensor] = []
    shard_idx = 0
    shard_row = 0
    start_time = time.perf_counter()

    for start in range(0, len(entries), batch_size):
        chunk = entries[start : start + batch_size]
        tokens_np = np.stack([
            load_tokens_from_meta(meta)
            for _digest, meta in chunk
        ])
        feats = compute_visual_features(tokens_np, processor, model, device, model_name)
        for local_idx, (digest, meta) in enumerate(chunk):
            shard_tensors.append(feats[local_idx : local_idx + 1])
            manifest_rows.append(
                {
                    "sha1": digest,
                    "feature_file": f"features_{shard_idx:05d}.pt",
                    "row": shard_row,
                    "shape": feature_shape,
                    "dtype": "float16",
                    "model": model_name,
                    "visual_batch_size": int(batch_size),
                    "cache_level": spec["cache_level"],
                    **meta,
                }
            )
            shard_row += 1
            if shard_row >= shard_size:
                torch.save(torch.cat(shard_tensors, dim=0), cache_dir / f"features_{shard_idx:05d}.pt")
                shard_idx += 1
                shard_row = 0
                shard_tensors = []

        done = min(start + batch_size, len(entries))
        if done % max(batch_size * 25, 1) == 0 or done == len(entries):
            elapsed = time.perf_counter() - start_time
            print(f"[cache] features={done:,}/{len(entries):,} elapsed_s={elapsed:.1f}", flush=True)

    if shard_tensors:
        torch.save(torch.cat(shard_tensors, dim=0), cache_dir / f"features_{shard_idx:05d}.pt")

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    total_bytes = sum(path.stat().st_size for path in cache_dir.glob("features_*.pt"))
    summary = {
        "model": model_name,
        "feature_count": len(entries),
        "feature_shape": feature_shape,
        "cache_level": spec["cache_level"],
        "dtype": "float16",
        "visual_batch_size": int(batch_size),
        "feature_files": len(list(cache_dir.glob("features_*.pt"))),
        "total_feature_bytes": total_bytes,
        "total_feature_gib": total_bytes / (1024**3),
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    (cache_dir / "cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (cache_dir / "cache_summary.md").write_text(
        "\n".join(
            [
                f"# {spec['title']}",
                "",
                f"- model: `{model_name}`",
                f"- features: `{summary['feature_count']}`",
                f"- feature shape: `{feature_shape}`",
                f"- cache level: `{spec['cache_level']}`",
                "- dtype: `float16`",
                f"- visual batch size: `{int(batch_size)}`",
                f"- shard files: `{summary['feature_files']}`",
                f"- feature size: `{summary['total_feature_gib']:.2f} GiB`",
                f"- elapsed seconds: `{summary['elapsed_seconds']:.1f}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def iter_original_k1_sources(
    aug_data: Path,
    canonical_data: Path,
) -> Iterable[Tuple[str, Path, int, np.ndarray]]:
    from scripts.fvr_original_trm_exact_candidate_k import build_exact_candidate_examples

    target_ids = json.loads((canonical_data / "identifiers.json").read_text(encoding="utf-8"))
    target_tasks = [task_id for task_id in target_ids if task_id != "<blank>"]
    examples, _target_info = build_exact_candidate_examples(aug_data, target_tasks, k=1)
    pseudo_path = Path("__original_trm_k1_eval__.inline")
    for row_idx, example in enumerate(examples):
        yield (
            f"{aug_data.name}:original_trm_k1:{example['task_id']}:input",
            pseudo_path,
            row_idx,
            np.asarray(example["input"], dtype=np.uint8),
        )


def sources_for_scope(scope: str, aug_data: Path, eval_data: Path, canonical_data: Path):
    if scope == "eval":
        def eval_sources():
            yield from iter_token_sources(eval_data, ("train", "test"))
            yield from iter_eval_json_context_sources(eval_data)
        return eval_sources()
    if scope == "eval_original_k1":
        def eval_original_k1_sources():
            yield from iter_token_sources(eval_data, ("train", "test"))
            yield from iter_eval_json_context_sources(eval_data)
            yield from iter_original_k1_sources(aug_data, canonical_data)
        return eval_original_k1_sources()
    if scope == "aug":
        return iter_token_sources(aug_data, ("train", "test"))
    if scope == "all":
        def chained():
            yield from iter_token_sources(aug_data, ("train", "test"))
            yield from iter_token_sources(eval_data, ("train", "test"))
            yield from iter_eval_json_context_sources(eval_data)
            yield from iter_original_k1_sources(aug_data, canonical_data)
        return chained()
    raise ValueError(f"Unsupported scope={scope!r}")


def self_test() -> None:
    row = np.arange(900, dtype=np.uint16) % 12
    digest = sha1_tokens(row)
    assert digest == sha1_tokens(row.astype(np.uint8))
    class Proc:
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]
    pix = tokens_to_pixel_values(torch.from_numpy(row.reshape(1, -1).astype(np.uint8)), Proc())
    assert tuple(pix.shape) == (1, 3, 224, 224)
    assert torch.isfinite(pix).all()
    pix256 = tokens_to_pixel_values(
        torch.from_numpy(row.reshape(1, -1).astype(np.uint8)),
        Proc(),
        image_side=256,
    )
    assert tuple(pix256.shape) == (1, 3, 256, 256)
    assert feature_shape_for("google/siglip2-base-patch16-256") == [256, 768]
    print("[self-test] PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen visual feature cache for TRM token grids.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--model-name", default=MODEL_NAME, choices=tuple(sorted(MODEL_SPECS)))
    parser.add_argument("--aug-data", default=str(DEFAULT_AUG_DATA))
    parser.add_argument("--eval-data", default=str(DEFAULT_EVAL_DATA))
    parser.add_argument("--canonical-data", default=str(DEFAULT_CANONICAL_DATA))
    parser.add_argument("--scope", choices=("eval", "eval_original_k1", "aug", "all"), default="eval")
    parser.add_argument("--census-only", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    cache_dir = Path(args.cache_dir)
    aug_data = Path(args.aug_data)
    eval_data = Path(args.eval_data)
    canonical_data = Path(args.canonical_data)
    unique = collect_unique_sources(sources_for_scope(args.scope, aug_data, eval_data, canonical_data))
    write_census(cache_dir, args.scope, len(unique), args.model_name)

    if args.census_only and not args.build:
        return
    if not args.build:
        print("[info] Census complete. Pass --build to materialize visual features.")
        return
    build_cache(
        cache_dir=cache_dir,
        unique_sources=unique,
        batch_size=int(args.batch_size),
        shard_size=int(args.shard_size),
        force=bool(args.force),
        model_name=str(args.model_name),
    )


if __name__ == "__main__":
    main()
