"""Builds the (candidate_grid, label) dataset for verifier training.

v2 DESIGN — LODO-on-demos (LEAK-FREE relative to the test pair):
  For each task with demos D_0..D_{N-1}:
    For each i in 0..N-1:
      Hold out demo D_i. The OUTPUT of D_i becomes the positive candidate.
      Negatives are:
        - D4 transforms of D_i.output that CHANGE the grid ("d4_confuser_*").
          These match the inference-time confuser distribution exactly:
          eval candidates = {canonical prediction} ∪ {its 7 D4 transforms}.
        - synthetic perturbations of D_i.output (cell flip / colour swap / …).
        - outputs of other demos D_j (j != i)  → cross-demo hard negatives.
        - ORPI program-repair candidates for this task.
      D4 transforms that LEAVE the grid invariant (symmetric outputs) stay
      positive. Records are de-duplicated per (task, held demo) with positives
      winning collisions, so no grid appears with conflicting labels.
      A downstream verifier trainer must honor `held_out_demo_idx` and mask
      D_i from the C2 context — so the verifier learns to predict
      D_i.output FROM D_j outputs, not from D_i itself.

  Why D4-as-negative matters: v1 (and the first v2 draft) labelled ALL D4
  variants of the correct output as POSITIVE. At inference the verifier then
  could not tell the canonical prediction apart from its own rotations
  (D4-commit exact rate was 7%), which is why v1 was REJECTED with zero gains.

CRITICAL: this builder NEVER accesses `task["test"][...]["output"]`. The
test pair is held out for evaluation. The verifier therefore cannot memorise
the eval set's GT answer grids — even when train and eval share task IDs.

Caveat: the verifier still trains on the demo distributions of these specific
task IDs. For TRUE held-out task-level generalization, supply --train-data
with task IDs disjoint from --eval-data (e.g., the ARC-AGI-1 training subset
which is NOT in the public eval set).

Record schema (verifier_dataset.jsonl, one JSON per line):
  task_id              str
  held_out_demo_idx    int  — which demo this record's positive/negative is for
  source               str  — provenance label
  label                int  — 1=positive, 0=negative
  candidate_seq        list[int]  — 900-token candidate output
  true_h, true_w       int  — original (h, w) before padding to 30x30
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Token vocabulary (must match TRM's encoding)
EOS_TOKEN = 1
PAD_TOKEN = 0
COLOR_TOKEN_OFFSET = 2
TOKEN_GRID_SIDE = 30
TOKEN_GRID_LEN = TOKEN_GRID_SIDE * TOKEN_GRID_SIDE


def color_grid_to_token_seq(color_grid: np.ndarray) -> np.ndarray:
    """Convert (h, w) colour grid (0..9) to (900,) token sequence, padded
    with EOS outside the canvas."""
    h, w = color_grid.shape
    out = np.full((TOKEN_GRID_SIDE, TOKEN_GRID_SIDE), EOS_TOKEN, dtype=np.int64)
    out[:h, :w] = color_grid.astype(np.int64) + COLOR_TOKEN_OFFSET
    return out.flatten()


def token_seq_to_color_grid(seq: np.ndarray, h: int, w: int) -> np.ndarray:
    """Inverse of color_grid_to_token_seq."""
    grid = seq.reshape(TOKEN_GRID_SIDE, TOKEN_GRID_SIDE)
    inside = grid[:h, :w]
    return np.clip(inside - COLOR_TOKEN_OFFSET, 0, 9).astype(np.int64)


# ---------------- D4 augmentation ----------------

_D4_NAMES = ("identity", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "anti_transpose")


def apply_d4(grid: np.ndarray, name: str) -> np.ndarray:
    if name == "identity": return grid.copy()
    if name == "rot90": return np.rot90(grid, k=1).copy()
    if name == "rot180": return np.rot90(grid, k=2).copy()
    if name == "rot270": return np.rot90(grid, k=3).copy()
    if name == "flip_h": return np.flip(grid, axis=1).copy()
    if name == "flip_v": return np.flip(grid, axis=0).copy()
    if name == "transpose": return grid.T.copy()
    if name == "anti_transpose": return np.flip(np.flip(grid, axis=0), axis=1).T.copy()
    raise ValueError(f"Unknown D4 transform: {name}")


# ---------------- Synthetic negatives ----------------

def perturb_one_cell(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flip one random cell to a different colour from the palette in use."""
    out = grid.copy()
    palette = sorted(int(c) for c in np.unique(grid) if int(c) != 0)
    if not palette:
        palette = [1]
    h, w = grid.shape
    r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
    current = int(out[r, c])
    choices = [p for p in palette if p != current] or [0]
    out[r, c] = int(rng.choice(choices))
    return out


def perturb_color_swap(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Swap two used colours across the whole grid."""
    out = grid.copy()
    palette = sorted(int(c) for c in np.unique(grid) if int(c) != 0)
    if len(palette) < 2:
        return perturb_one_cell(grid, rng)
    a, b = rng.choice(palette, size=2, replace=False).tolist()
    mask_a, mask_b = (out == a), (out == b)
    out[mask_a] = int(b)
    out[mask_b] = int(a)
    return out


def perturb_translate(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Translate by +-1 cell in a random direction (background-fill)."""
    out = np.zeros_like(grid)
    dr, dc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
    h, w = grid.shape
    sr0, sr1 = max(0, dr), min(h, h + dr)
    dr0, dr1 = max(0, -dr), min(h, h - dr)
    sc0, sc1 = max(0, dc), min(w, w + dc)
    dc0, dc1 = max(0, -dc), min(w, w - dc)
    out[sr0:sr1, sc0:sc1] = grid[dr0:dr1, dc0:dc1]
    return out


def perturb_reflect(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    axis = int(rng.choice([0, 1]))
    return np.flip(grid, axis=axis).copy()


PERTURBATIONS = (perturb_one_cell, perturb_color_swap, perturb_translate, perturb_reflect)


# ---------------- Dataset builder ----------------

def load_task_ids(data_root: Path) -> List[str]:
    """Load task IDs from a dataset's identifiers.json."""
    ids_path = data_root / "identifiers.json"
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    # Strip augmentation suffixes like '|||t7|||0123456789' to get base task_id
    return sorted({i.split("|||")[0] for i in ids if i and i != "<blank>"})


def load_test_puzzles(data_root: Path) -> Dict[str, dict]:
    p = data_root / "test_puzzles.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verifier training dataset.")
    parser.add_argument("--train-data", default="D:/trm_c2/arc1concept-aug-1000",
                        help="Training dataset root (used for task list).")
    parser.add_argument("--eval-data", default="data/arc1concept-aug-0",
                        help="Eval dataset root (FORBIDDEN task IDs).")
    parser.add_argument("--eval-id-range", default="",
                        help="Index range into eval-data identifiers.json that contains the "
                             "FORBIDDEN eval task IDs (e.g. '401:801' = 400 tasks). "
                             "Default '' = no filtering (warn loudly; rely on user-supplied "
                             "task-level CV at eval time). Pass empty to use the full identifier list as eval (strict leak guard).")
    parser.add_argument("--orpi-candidates",
                        default="reports/orpi_c2_unification_v4/repair_candidate_outputs.csv",
                        help="ORPI candidate dump (optional hard negatives).")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-d4-augs-per-task", type=int, default=8)
    parser.add_argument("--max-synthetic-negatives-per-positive", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=0,
                        help="Cap number of training tasks (0 = use all).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    repo_root = Path(__file__).resolve().parents[1]
    train_root = Path(args.train_data).resolve()
    eval_root = (repo_root / args.eval_data).resolve()

    train_task_ids = set(load_task_ids(train_root))
    print(f"[dataset] train task IDs: {len(train_task_ids)}")

    # ⚠️ CONTAMINATION CHECK — surface hidden-label leakage loudly.
    # If the train_root's test_puzzles.json contains tasks that are also in
    # eval, the GT test outputs of those tasks will end up as positives in
    # this dataset → direct label leakage at verifier-eval time.
    train_puzzles_path = train_root / "test_puzzles.json"
    eval_puzzles_path = eval_root / "test_puzzles.json"
    if train_puzzles_path.exists() and eval_puzzles_path.exists():
        tp = set(json.loads(train_puzzles_path.read_text(encoding="utf-8")).keys())
        ep = set(json.loads(eval_puzzles_path.read_text(encoding="utf-8")).keys())
        gt_overlap = tp & ep
        if gt_overlap:
            print("=" * 70)
            print(f"[INFO] train_puzzles and eval_puzzles share {len(gt_overlap)} task IDs.")
            print(f"[INFO] This dataset builder uses LODO-on-demos (task['train'][i].output)")
            print(f"[INFO] as positives — NEVER task['test'][...]['output']. The held-out test")
            print(f"[INFO] pair is preserved for evaluation.")
            print(f"[INFO] Caveat: the verifier still trains on demo distributions from these")
            print(f"[INFO] 400 specific task IDs. For a TRULY held-out task-level generalization")
            print(f"[INFO] study, supply --train-data with DISJOINT task IDs.")
            print("=" * 70)

    # The eval task IDs may be a SUBSET of eval-data identifiers (per the
    # codex-discovered PID401 remap convention: eval is full_ids[401:801]).
    # If --eval-id-range is empty, no automatic leak filtering is applied
    # — the caller is responsible for task-level CV at eval time.
    eval_task_ids: set[str] = set()
    ids_path = eval_root / "identifiers.json"
    if args.eval_id_range:
        try:
            lo, hi = [int(x) for x in args.eval_id_range.split(":")]
            raw_ids_with_blank = json.loads(ids_path.read_text(encoding="utf-8"))
            eval_task_ids = {raw_ids_with_blank[i].split("|||")[0]
                              for i in range(lo, hi)
                              if 0 <= i < len(raw_ids_with_blank)
                              and raw_ids_with_blank[i] not in (None, "", "<blank>")}
            print(f"[dataset] eval task IDs (forbidden via --eval-id-range): {len(eval_task_ids)}")
        except Exception as e:
            print(f"[dataset] WARN: failed to parse --eval-id-range={args.eval_id_range!r}: {e}")
    else:
        print(f"[dataset] WARN: --eval-id-range empty; NO automatic leak filtering. "
              f"Apply task-level CV at eval time.")

    overlap = train_task_ids & eval_task_ids
    if overlap:
        print(f"[dataset] excluding {len(overlap)} train tasks that overlap eval set")
    safe_train_ids = sorted(train_task_ids - eval_task_ids)
    print(f"[dataset] safe training tasks (train - eval): {len(safe_train_ids)}")
    if not safe_train_ids:
        raise RuntimeError(
            "No training tasks available after removing eval overlap. "
            "Verify --train-data and --eval-id-range correctly partition the dataset."
        )

    if args.max_tasks > 0:
        safe_train_ids = safe_train_ids[: int(args.max_tasks)]
        print(f"[dataset] capped to first {len(safe_train_ids)} tasks")

    train_puzzles = load_test_puzzles(train_root)
    if not train_puzzles:
        raise RuntimeError(f"No test_puzzles.json at {train_root}")

    # Optionally load ORPI candidates keyed by task_id.
    orpi_by_task: Dict[str, List[np.ndarray]] = {}
    orpi_path = Path(args.orpi_candidates)
    if orpi_path.exists():
        with orpi_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid = str(row.get("task_id", ""))
                if tid in eval_task_ids:
                    continue  # CRITICAL: discard ORPI candidates from eval tasks
                try:
                    grid = np.array(json.loads(row["candidate_grid"]), dtype=np.int64)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
                orpi_by_task.setdefault(tid, []).append(grid)
        print(f"[dataset] loaded ORPI candidates for {len(orpi_by_task)} non-eval tasks")
    else:
        print(f"[dataset] ORPI candidates not found at {orpi_path}; skipping source (c)")

    # Build records via LODO-on-demos: per task, per demo, hold out that demo's
    # (input, output) pair and treat its OUTPUT as the verifier-training label.
    # The other demos serve as C2 context at training time. Any downstream
    # verifier trainer must honor `held_out_demo_idx`.
    #
    # CRITICAL: NEVER access task["test"][0]["output"]. The test pair is held
    # out for evaluation. Only task["train"] (demo pairs) is used here.
    n_pos, n_neg, n_skipped, n_demo_skipped = 0, 0, 0, 0
    records: List[Dict[str, object]] = []
    for tid in safe_train_ids:
        if tid not in train_puzzles:
            n_skipped += 1
            continue
        task = train_puzzles[tid]
        demos = task.get("train", [])
        if len(demos) < 2:
            # Need at least 2 demos so there's something to use as C2 context
            # after holding one out.
            n_demo_skipped += 1
            continue

        for hold_idx, held_demo in enumerate(demos):
            true_out = np.array(held_demo["output"], dtype=np.int64)
            other_demo_outputs = [
                np.array(d["output"], dtype=np.int64)
                for j, d in enumerate(demos) if j != hold_idx
            ]

            # Per-(task, held demo) dedup. The verifier sees ONE candidate set per
            # task at inference; identical grids must not appear with conflicting
            # labels. Positives are registered first and win any collision: a later
            # "negative" grid that exactly equals the positive is dropped.
            pos_seq_keys: set = set()
            emitted_seq_keys: set = set()

            def _emit(grid: np.ndarray, source: str, label: int) -> bool:
                """Append one record with dedup. Returns True if appended."""
                nonlocal n_pos, n_neg
                seq = color_grid_to_token_seq(grid)
                key = seq.tobytes()
                if label == 1:
                    if key in pos_seq_keys:
                        return False  # duplicate positive (e.g. symmetric transform)
                    pos_seq_keys.add(key)
                else:
                    # never let a negative duplicate the known-correct grid
                    if key in pos_seq_keys:
                        return False
                    if key in emitted_seq_keys:
                        return False  # duplicate negative
                emitted_seq_keys.add(key)
                records.append({
                    "task_id": tid,
                    "held_out_demo_idx": int(hold_idx),
                    "source": source,
                    "label": int(label),
                    "candidate_seq": seq.tolist(),
                    "true_h": int(grid.shape[0]),
                    "true_w": int(grid.shape[1]),
                })
                if label == 1:
                    n_pos += 1
                else:
                    n_neg += 1
                return True

            # --- POSITIVE: the held-out demo's exact output (only guaranteed-correct grid) ---
            _emit(true_out, "lodo_demo_gt_identity", 1)

            # --- D4 transforms of the true output ---
            # KEY FIX over v1/early-v2: a transform that LEAVES the grid invariant is
            # still the correct answer (positive); a transform that CHANGES the grid is a
            # geometrically-plausible WRONG answer — exactly the confuser the verifier must
            # reject at inference, where candidates = {canonical} ∪ {its 7 D4 transforms}.
            # v1 labeled all D4 variants as positive and so could not discriminate the
            # canonical prediction from its rotations (D4-commit exact rate was 7%).
            for name in _D4_NAMES[1:]:  # skip identity (already emitted)
                augmented = apply_d4(true_out, name)
                invariant = (augmented.shape == true_out.shape
                             and np.array_equal(augmented, true_out))
                if invariant:
                    _emit(augmented, f"lodo_demo_gt_sym_{name}", 1)
                else:
                    _emit(augmented, f"d4_confuser_{name}", 0)

            # --- NEGATIVES: synthetic perturbations of the held-out output ---
            for _ in range(int(args.max_synthetic_negatives_per_positive)):
                perturb_fn = PERTURBATIONS[int(rng.integers(0, len(PERTURBATIONS)))]
                wrong = perturb_fn(true_out, rng)
                if np.array_equal(wrong, true_out):
                    continue
                _emit(wrong, f"synthetic_{perturb_fn.__name__}", 0)

            # --- NEGATIVES: cross-demo hard negatives (other demo outputs) ---
            # Another demo's output is a plausible-looking but wrong answer for this
            # held-out demo. Strong "looks-like-an-ARC-output" negative.
            for other_out in other_demo_outputs:
                _emit(other_out, "cross_demo_output", 0)

            # --- NEGATIVES: ORPI candidates (still valid; they don't touch test pair) ---
            for cand in orpi_by_task.get(tid, []):
                _emit(cand, "orpi_candidate", 0)

    # Write JSONL.
    out_jsonl = out_dir / "verifier_dataset.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # Write a manifest summary.
    manifest = {
        "n_records": len(records),
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "n_tasks": len(set(r["task_id"] for r in records)),
        "n_eval_overlap_excluded": len(overlap),
        "n_train_tasks_in_train_root": len(train_task_ids),
        "n_eval_tasks_in_eval_root": len(eval_task_ids),
        "max_d4_augs_per_task": int(args.max_d4_augs_per_task),
        "max_synthetic_negatives_per_positive": int(args.max_synthetic_negatives_per_positive),
        "orpi_candidate_tasks": len(orpi_by_task),
        "seed": int(args.seed),
        "train_data": str(train_root),
        "eval_data": str(eval_root),
        "skipped_tasks_missing_puzzle": n_skipped,
        "skipped_tasks_too_few_demos": n_demo_skipped,
        "training_signal": "LODO-on-demos (task['train'][i].output as positive, others as C2 context)",
        "test_pair_access": "NEVER — task['test'] is held out for evaluation only",
        "leak_guard": "no task['test'][...]['output'] access in record generation",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[dataset] wrote {len(records)} records ({n_pos} pos, {n_neg} neg) -> {out_jsonl}")
    print(f"[dataset] manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
