"""Rebuild data/arc-agi-evaluation-full400-seed0 so its task ordering matches
the c2_geomaux / stage4b PID401 checkpoints' puzzle_emb ordering.

Background:
    The c2_geomaux step670 PID401 checkpoint scored 125/400 against a dataset
    version dated 2026-05-22. On 2026-05-28 the dataset was rebuilt with a
    different task ordering (alphabetical-by-task_id) and the original was
    overwritten. The checkpoint's puzzle_emb is still aligned to the May-22
    ordering. Without a backup, we recovered the May-22 ordering by reverse-
    engineering from the base checkpoint step_518071 (which holds the full
    876k-row puzzle_emb): every PID401 row r matches exactly (cos sim 1.0000)
    one bare-task_id row in aug1k. The mapping is saved at
    reports/pid401_to_aug1k_mapping.json.

This script:
    1. Loads the mapping (PID401 row -> aug1k bare task_id) and constructs the
       NEW task ordering (list of 400 task_ids in PID401 row order).
    2. Reads the current (alphabetical) dataset's arrays.
    3. For each NEW puzzle_id r (1..400), finds the OLD puzzle_id where the
       task_id matches, and copies that task's input/label/shape samples to
       the NEW position.
    4. Writes the rebuilt dataset to a new directory (default:
       data/arc-agi-evaluation-full400-seed0-pid401aligned).

After this rebuild, running the eval against the rebuilt dataset with the
PID401 checkpoint should restore ~125/400.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load_pid401_to_taskid(mapping_path: Path) -> list[str]:
    """Return list of 400 task_ids in PID401 row order (row 1 -> idx 0)."""
    raw = json.loads(mapping_path.read_text(encoding="utf-8"))
    # raw is a dict keyed by str(r); each value has 'identifier' = aug1k row text
    # The aug1k identifier for a bare row IS the task_id.
    ordered = []
    for r in range(1, 401):
        entry = raw[str(r)]
        ident = entry["identifier"]
        if "|||" in ident:
            tid = ident.split("|||")[0]
        else:
            tid = ident
        ordered.append(tid)
    assert len(ordered) == 400
    return ordered


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/arc-agi-evaluation-full400-seed0",
                   help="current (misaligned) dataset to permute")
    p.add_argument("--mapping", default="reports/pid401_to_aug1k_mapping.json",
                   help="PID401 -> aug1k mapping (PID401-row-keyed JSON)")
    p.add_argument("--out", default="data/arc-agi-evaluation-full400-seed0-pid401aligned",
                   help="output dataset directory (rebuilt)")
    p.add_argument("--splits", default="train,test",
                   help="comma-separated splits to rebuild (default: train,test). "
                        "The eval harness reads the TRAIN split, so it must be included.")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src = (repo_root / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src)
    mapping_path = (repo_root / args.mapping).resolve() if not Path(args.mapping).is_absolute() else Path(args.mapping)
    out = (repo_root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"[rebuild] src: {src}")
    print(f"[rebuild] mapping: {mapping_path}")
    print(f"[rebuild] out: {out}")
    print(f"[rebuild] splits: {splits}")

    # NEW task ordering (PID401 row order)
    new_order = load_pid401_to_taskid(mapping_path)
    print(f"[rebuild] NEW order length: {len(new_order)}")
    print(f"[rebuild]   NEW[0..3]: {new_order[:4]}")

    # OLD task ordering (current dataset's identifiers.json[1..400])
    old_ids_path = src / "identifiers.json"
    old_ids = json.loads(old_ids_path.read_text(encoding="utf-8"))
    assert old_ids[0] == "<blank>", f"expected blank at row 0, got {old_ids[0]!r}"
    old_order = old_ids[1:]
    print(f"[rebuild] OLD order length: {len(old_order)}")
    print(f"[rebuild]   OLD[0..3]: {old_order[:4]}")

    # Sanity: same set of task_ids
    if set(new_order) != set(old_order):
        miss_in_new = sorted(set(old_order) - set(new_order))
        miss_in_old = sorted(set(new_order) - set(old_order))
        raise RuntimeError(
            f"task_id sets differ. Missing in NEW: {miss_in_new[:5]}... Missing in OLD: {miss_in_old[:5]}..."
        )

    # Build permutation: for each NEW idx i (0..399), find old_idx where old_order[old_idx] == new_order[i]
    old_pos = {tid: idx for idx, tid in enumerate(old_order)}
    perm_old_idx = [old_pos[tid] for tid in new_order]   # length 400
    print(f"[rebuild] permutation built (sample): NEW0 <- OLD{perm_old_idx[0]}, NEW1 <- OLD{perm_old_idx[1]}, ...")

    # Build NEW identifiers.json (shared across splits)
    new_ids = ["<blank>"] + new_order
    assert len(new_ids) == 401
    out.mkdir(parents=True, exist_ok=True)
    (out / "identifiers.json").write_text(json.dumps(new_ids), encoding="utf-8")

    n_rows_per_split = {}

    for split in splits:
        src_split = src / split
        if not src_split.exists():
            print(f"[rebuild] WARNING: split {split!r} not found at {src_split}; skipping")
            continue

        inputs = np.load(src_split / "all__inputs.npy")
        labels = np.load(src_split / "all__labels.npy")
        puzz_idx = np.load(src_split / "all__puzzle_indices.npy")   # length=401 (cumulative)
        grp_idx = np.load(src_split / "all__group_indices.npy")     # length=401
        th = np.load(src_split / "all__target_height.npy")
        tw = np.load(src_split / "all__target_width.npy")

        print(f"[rebuild][{split}] OLD shapes: inputs={inputs.shape} labels={labels.shape} "
              f"puzz_idx={puzz_idx.shape} grp_idx={grp_idx.shape}")

        # For each OLD puzzle p (0..399), its example slice is inputs[puzz_idx[p]:puzz_idx[p+1]].
        # Reassemble the slices in NEW order.
        new_inputs_parts, new_labels_parts, new_th_parts, new_tw_parts = [], [], [], []
        new_puzz_idx = [0]
        new_pids = []
        for new_i in range(400):
            old_i = perm_old_idx[new_i]   # OLD puzzle position whose task_id == new_order[new_i]
            s = int(puzz_idx[old_i])
            e = int(puzz_idx[old_i + 1])
            new_inputs_parts.append(inputs[s:e])
            new_labels_parts.append(labels[s:e])
            new_th_parts.append(th[s:e])
            new_tw_parts.append(tw[s:e])
            new_puzz_idx.append(new_puzz_idx[-1] + (e - s))
            new_pids.append(new_i + 1)   # NEW identifier index (1..400)

        new_inputs = np.concatenate(new_inputs_parts, axis=0)
        new_labels = np.concatenate(new_labels_parts, axis=0)
        new_th = np.concatenate(new_th_parts, axis=0)
        new_tw = np.concatenate(new_tw_parts, axis=0)
        new_puzz_idx = np.array(new_puzz_idx, dtype=puzz_idx.dtype)
        new_pids = np.array(new_pids, dtype=np.int32)
        # group_indices structure is preserved (one group per puzzle); copy dtype/shape from source
        new_grp_idx = np.arange(len(grp_idx), dtype=grp_idx.dtype)

        print(f"[rebuild][{split}] NEW shapes: inputs={new_inputs.shape} labels={new_labels.shape}")
        n_rows_per_split[split] = int(new_inputs.shape[0])

        (out / split).mkdir(parents=True, exist_ok=True)
        np.save(out / split / "all__inputs.npy", new_inputs)
        np.save(out / split / "all__labels.npy", new_labels)
        np.save(out / split / "all__puzzle_identifiers.npy", new_pids)
        np.save(out / split / "all__puzzle_indices.npy", new_puzz_idx)
        np.save(out / split / "all__group_indices.npy", new_grp_idx)
        np.save(out / split / "all__target_height.npy", new_th)
        np.save(out / split / "all__target_width.npy", new_tw)

        # Carry dataset.json forward unchanged (row permutation does not change counts).
        ds_json = src_split / "dataset.json"
        if ds_json.exists():
            shutil.copyfile(ds_json, out / split / "dataset.json")
            print(f"[rebuild][{split}] copied dataset.json")
        else:
            print(f"[rebuild][{split}] WARNING: no dataset.json in source split")

    # Copy test_puzzles.json if present
    for aux in ["test_puzzles.json", "train_puzzles.json"]:
        if (src / aux).exists():
            shutil.copyfile(src / aux, out / aux)
            print(f"[rebuild] copied {aux}")

    # Also copy dataset.json if exists
    for aux in ["dataset.json"]:
        if (src / aux).exists():
            shutil.copyfile(src / aux, out / aux)
            print(f"[rebuild] copied {aux}")

    # Summary report
    summary = {
        "src": str(src),
        "mapping": str(mapping_path),
        "out": str(out),
        "splits": splits,
        "n_puzzles": 400,
        "n_input_rows_per_split": n_rows_per_split,
        "new_order_first10": new_order[:10],
        "old_order_first10": old_order[:10],
        "n_rows_moved": int(sum(1 for i, p in enumerate(perm_old_idx) if i != p)),
    }
    (out / "rebuild_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[rebuild] wrote summary -> {out / 'rebuild_summary.json'}")
    print(f"[rebuild] DONE.")


if __name__ == "__main__":
    main()
