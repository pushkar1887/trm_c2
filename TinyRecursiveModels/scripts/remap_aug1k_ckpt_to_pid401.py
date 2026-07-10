"""Extract PID401 puzzle embeddings from a raw Aug1000-shaped checkpoint.

This is different from remap_ckpt_pid401_for_current_dataset.py:

* this script converts a training-side checkpoint whose puzzle embedding is
  [876406, D] into an eval-side checkpoint whose puzzle embedding is [401, D];
* the older script only permutes an already-PID401-shaped [401, D] checkpoint.

Only model.inner.puzzle_emb.weights is changed. All other checkpoint tensors are
copied verbatim.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


PUZZLE_KEY = "model.inner.puzzle_emb.weights"
COMPILED_PUZZLE_KEY = "_orig_mod.model.inner.puzzle_emb.weights"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _find_puzzle_key(state: dict[str, Any]) -> str:
    candidates = [k for k in (PUZZLE_KEY, COMPILED_PUZZLE_KEY) if k in state]
    if len(candidates) == 1:
        return candidates[0]
    suffix_matches = [k for k in state if k.endswith("model.inner.puzzle_emb.weights")]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise RuntimeError(
        "Could not uniquely identify puzzle embedding key. "
        f"candidates={suffix_matches[:10]}"
    )


def _load_ids(path: Path) -> list[str]:
    ids = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not ids or ids[0] != "<blank>":
        raise RuntimeError(f"{path} must be an identifiers.json list with '<blank>' at row 0")
    return ids


def _load_mapping(path: Path) -> dict[int, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[int, dict[str, Any]] = {}
    for key, value in raw.items():
        pid = int(key)
        if not isinstance(value, dict) or "aug1k_idx" not in value or "identifier" not in value:
            raise RuntimeError(f"Bad mapping entry for pid {pid}: {value!r}")
        mapping[pid] = value
    return mapping


def build_pid401_embedding(
    old_weights: torch.Tensor,
    mapping: dict[int, dict[str, Any]],
    dst_ids: list[str],
    *,
    allow_small_source: bool = False,
) -> tuple[torch.Tensor, list[int]]:
    """Return [401, D] weights and the source row used for each destination row."""
    if old_weights.ndim != 2:
        raise RuntimeError(f"Expected 2D puzzle embedding, got shape {tuple(old_weights.shape)}")
    if len(dst_ids) != 401:
        raise RuntimeError(f"Expected 401 destination identifiers, got {len(dst_ids)}")
    if set(mapping) != set(range(1, 401)):
        missing = sorted(set(range(1, 401)) - set(mapping))
        extra = sorted(set(mapping) - set(range(1, 401)))
        raise RuntimeError(f"Mapping must contain pids 1..400. missing={missing[:5]} extra={extra[:5]}")
    if old_weights.shape[0] == len(dst_ids) and not allow_small_source:
        raise RuntimeError(
            "Source checkpoint is already PID401-shaped. Use "
            "scripts/remap_ckpt_pid401_for_current_dataset.py for 401-to-401 permutation."
        )
    if old_weights.shape[0] <= max(int(mapping[i]["aug1k_idx"]) for i in mapping):
        raise RuntimeError(
            f"Source puzzle_emb has only {old_weights.shape[0]} rows, but mapping references "
            f"row {max(int(mapping[i]['aug1k_idx']) for i in mapping)}"
        )

    source_rows = [0]
    rows = [old_weights[0]]
    for pid in range(1, 401):
        expected_id = dst_ids[pid]
        entry = mapping[pid]
        mapped_id = str(entry["identifier"])
        if mapped_id != expected_id:
            raise RuntimeError(
                f"Mapping/destination mismatch at pid {pid}: mapping has {mapped_id}, "
                f"dst_ids has {expected_id}"
            )
        src_idx = int(entry["aug1k_idx"])
        source_rows.append(src_idx)
        rows.append(old_weights[src_idx])
    return torch.stack(rows, dim=0).contiguous(), source_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-ckpt", required=True, help="raw Aug1000-shaped source checkpoint")
    parser.add_argument(
        "--mapping",
        default="reports/pid401_to_aug1k_mapping.json",
        help="PID row -> Aug1000 row mapping JSON",
    )
    parser.add_argument(
        "--dst-ids",
        default="data/arc-agi-evaluation-full400-seed0-pid401aligned/identifiers.json",
        help="destination PID401 identifiers.json",
    )
    parser.add_argument("--out", required=True, help="output checkpoint path")
    parser.add_argument("--copy-config", default=None, help="optional all_config.yaml to copy beside output")
    parser.add_argument(
        "--allow-small-source",
        action="store_true",
        help="testing escape hatch; production remaps should not need this",
    )
    args = parser.parse_args()

    src_ckpt = _resolve_repo_path(args.src_ckpt)
    mapping_path = _resolve_repo_path(args.mapping)
    dst_ids_path = _resolve_repo_path(args.dst_ids)
    out_path = _resolve_repo_path(args.out)

    print(f"[remap-aug1k] src_ckpt: {src_ckpt}")
    print(f"[remap-aug1k] mapping: {mapping_path}")
    print(f"[remap-aug1k] dst_ids: {dst_ids_path}")
    print(f"[remap-aug1k] out: {out_path}")

    state = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    puzzle_key = _find_puzzle_key(state)
    old_weights = state[puzzle_key]
    mapping = _load_mapping(mapping_path)
    dst_ids = _load_ids(dst_ids_path)

    new_weights, source_rows = build_pid401_embedding(
        old_weights,
        mapping,
        dst_ids,
        allow_small_source=bool(args.allow_small_source),
    )
    state[puzzle_key] = new_weights

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)
    print(
        "[remap-aug1k] puzzle_emb: "
        f"{puzzle_key} {tuple(old_weights.shape)} -> {tuple(new_weights.shape)}"
    )
    print(f"[remap-aug1k] wrote checkpoint -> {out_path}")

    if args.copy_config:
        cfg_src = _resolve_repo_path(args.copy_config)
        cfg_dst = out_path.parent / "all_config.yaml"
        shutil.copyfile(cfg_src, cfg_dst)
        print(f"[remap-aug1k] copied config -> {cfg_dst}")

    summary = {
        "src_ckpt": str(src_ckpt),
        "mapping": str(mapping_path),
        "dst_ids": str(dst_ids_path),
        "out": str(out_path),
        "puzzle_key": puzzle_key,
        "old_puzzle_emb_shape": list(old_weights.shape),
        "new_puzzle_emb_shape": list(new_weights.shape),
        "n_eval_tasks": 400,
        "first_eval_ids": dst_ids[1:6],
        "first_source_rows": source_rows[1:6],
    }
    summary_path = out_path.parent / "remap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[remap-aug1k] summary -> {summary_path}")


if __name__ == "__main__":
    main()
