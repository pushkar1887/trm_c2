"""Permute a PID401-shaped checkpoint's puzzle_emb to align with the current
data/arc-agi-evaluation-full400-seed0 ordering.

Why: codex rebuilt the eval dataset with a different task-id ordering than the
one the checkpoint was originally trained against. The checkpoint's puzzle_emb
row N still holds the learned vector for some task_id, just not the task_id
that's at row N in the rebuilt dataset. This script permutes the rows so the
mapping matches.

Inputs:
    --src-ckpt     PID401-shaped checkpoint (puzzle_emb shape [401, D])
    --src-task-order  text file or csv listing the task_ids in the OLD order
                      (line/row i = task_id at OLD puzzle_id i, 1-indexed)
                      Default: the c2_geomaux ledger.
    --dst-ids      identifiers.json of the rebuilt dataset (= NEW order)
                      Default: data/arc-agi-evaluation-full400-seed0/identifiers.json
    --out          where to write the remapped checkpoint
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch


def load_old_order_from_ledger(ledger_path: Path) -> list:
    """Return task_ids in OLD puzzle_id order (1-indexed; row 0 reserved for <blank>)."""
    with ledger_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return ["<blank>"] + [r["task_id"] for r in rows]


def load_new_order_from_ids(ids_path: Path) -> list:
    return json.loads(ids_path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src-ckpt", required=True)
    p.add_argument("--src-ledger",
                   default="reports/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401_17col_ledger.csv",
                   help="ledger CSV defining the OLD task ordering (default: c2_geomaux 17col)")
    p.add_argument("--dst-ids",
                   default="data/arc-agi-evaluation-full400-seed0/identifiers.json",
                   help="identifiers.json of the rebuilt eval dataset (NEW order)")
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--copy-config", default=None,
                   help="optional: also copy an all_config.yaml alongside the new ckpt")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_ckpt = Path(args.src_ckpt).resolve()
    src_ledger = (repo_root / args.src_ledger) if not Path(args.src_ledger).is_absolute() else Path(args.src_ledger)
    dst_ids = (repo_root / args.dst_ids) if not Path(args.dst_ids).is_absolute() else Path(args.dst_ids)
    out_path = Path(args.out).resolve()

    print(f"[remap] src_ckpt: {src_ckpt}")
    print(f"[remap] src_ledger (OLD order): {src_ledger}")
    print(f"[remap] dst_ids (NEW order): {dst_ids}")
    print(f"[remap] out: {out_path}")

    old_order = load_old_order_from_ledger(src_ledger)
    new_order = load_new_order_from_ids(dst_ids)

    if len(old_order) != len(new_order):
        raise RuntimeError(
            f"OLD/NEW order length mismatch: old={len(old_order)} new={len(new_order)}"
        )
    if set(old_order[1:]) != set(new_order[1:]):
        missing_in_new = sorted(set(old_order[1:]) - set(new_order[1:]))
        missing_in_old = sorted(set(new_order[1:]) - set(old_order[1:]))
        raise RuntimeError(
            f"OLD/NEW task SETS differ. Missing in new: {missing_in_new[:5]}... "
            f"Missing in old: {missing_in_old[:5]}..."
        )

    # Build permutation: for each NEW row, find the OLD row that holds the same task_id.
    old_index = {tid: i for i, tid in enumerate(old_order)}
    perm = [old_index[tid] for tid in new_order]
    assert perm[0] == 0, "blank row should map to itself"
    print(f"[remap] permutation built; {sum(1 for i, p in enumerate(perm) if i != p)} rows move")

    # Apply to puzzle_emb only; copy everything else verbatim.
    state = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    key = "model.inner.puzzle_emb.weights"
    if key not in state:
        raise RuntimeError(f"checkpoint missing key {key!r}; available: {list(state)[:10]}...")
    old_w = state[key]
    if old_w.shape[0] != len(old_order):
        raise RuntimeError(
            f"puzzle_emb shape {tuple(old_w.shape)} doesn't match old_order length {len(old_order)}"
        )
    perm_t = torch.tensor(perm, dtype=torch.long)
    new_w = old_w[perm_t].contiguous()
    state[key] = new_w
    print(f"[remap] permuted puzzle_emb: old shape {tuple(old_w.shape)} -> new {tuple(new_w.shape)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_path)
    print(f"[remap] wrote remapped checkpoint -> {out_path}")

    if args.copy_config:
        cfg_src = Path(args.copy_config).resolve()
        cfg_dst = out_path.parent / "all_config.yaml"
        shutil.copyfile(cfg_src, cfg_dst)
        print(f"[remap] copied config -> {cfg_dst}")

    # Sanity report.
    summary = {
        "src_ckpt": str(src_ckpt),
        "src_ledger": str(src_ledger),
        "dst_ids": str(dst_ids),
        "out": str(out_path),
        "n_rows": len(perm),
        "n_rows_moved": int(sum(1 for i, p in enumerate(perm) if i != p)),
        "old_puzzle_emb_shape": list(old_w.shape),
        "new_puzzle_emb_shape": list(new_w.shape),
    }
    (out_path.parent / "remap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[remap] summary -> {out_path.parent / 'remap_summary.json'}")


if __name__ == "__main__":
    main()
