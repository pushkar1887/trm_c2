import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pretrain
from scripts.fvr_structfuse_alpha_sweep import BUCKETS, crop_shape, read_csv, write_csv
from scripts.fvr_verified_program_repair import Component, connected_components, grid_key, most_common_color


Grid = np.ndarray
FAMILIES = ["fill", "translate", "recolour", "reflect", "extend"]
PREREG_TEXT = """# D0-OBJ Oracle Probe Preregistration

This file is written before running D0-OBJ candidate scoring.

Acceptance thresholds:

- oracle_any new exact on both_fail >= 8
- at least one primitive solo gain on both_fail >= 3
- combined replace_loss on currently solved C0 tasks = 0

Scientific boundary:

Public eval tasks have informed this probe. Any improvement is engineering/development evidence only, not an unbiased paper generalization claim.
"""


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def preregister(out_dir: Path, repo_root: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg_path = out_dir / "preregister.md"
    if not prereg_path.exists():
        prereg_path.write_text(PREREG_TEXT, encoding="utf-8")
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0 or result.stdout.strip().lower() != "true":
        (out_dir / "git_unavailable.txt").write_text(
            "D:\\trm_c2\\TinyRecursiveModels is not a git repository.\n"
            f"preregister_sha256={sha256_text(prereg_path.read_text(encoding='utf-8'))}\n",
            encoding="utf-8",
        )
        return
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--short", str(prereg_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if status.stdout.strip():
        subprocess.run(["git", "-C", str(repo_root), "add", str(prereg_path)], check=True)
        subprocess.run(
            ["git", "-C", str(repo_root), "commit", "-m", "docs: preregister d0 obj oracle probe"],
            check=True,
        )


def load_tasks(dataset: Path) -> Dict[str, Dict[str, object]]:
    path = dataset / "test_puzzles.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def task_output(task: Dict[str, object]) -> Grid:
    return np.asarray(task["test"][0]["output"], dtype=np.uint8)


def task_input(task: Dict[str, object]) -> Grid:
    return np.asarray(task["test"][0]["input"], dtype=np.uint8)


def token_seq_to_grid(seq: np.ndarray) -> Grid:
    h, w = crop_shape(seq)
    if h <= 0 or w <= 0:
        return np.zeros((1, 1), dtype=np.uint8)
    token_grid = seq.reshape(30, 30)[:h, :w]
    out = np.zeros((h, w), dtype=np.uint8)
    valid = (token_grid >= 2) & (token_grid <= 11)
    out[valid] = (token_grid[valid] - 2).astype(np.uint8)
    return out


def generate_c0_cache(
    cache_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    batch_size: int,
) -> Dict[str, Grid]:
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        task_ids = data["task_ids"].astype(str).tolist()
        preds = data["preds"]
        return {task_id: token_seq_to_grid(preds[idx]) for idx, task_id in enumerate(task_ids)}

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["load_checkpoint"] = str(checkpoint_path)
    raw_config["data_paths"] = ["data/arc-agi-evaluation-full400-seed0"]
    raw_config["data_paths_test"] = []
    raw_config["eval_save_outputs"] = []
    raw_config["dataloader_num_workers"] = 0
    raw_config["checkpoint_path"] = str(cache_path.parent / "noop_checkpoints")
    raw_config["run_name"] = "d0_obj_c0_cache"
    raw_config["global_batch_size"] = int(batch_size)
    raw_config.setdefault("arch", {})["c2_structure_fusion_alpha"] = 0.0
    config = pretrain.PretrainConfig(**raw_config)

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
        raise RuntimeError("CUDA is required to regenerate C0 predictions.")

    repo_root = Path(__file__).resolve().parents[1]
    eval_ids = json.loads((repo_root / "data" / "arc-agi-evaluation-full400-seed0" / "identifiers.json").read_text(encoding="utf-8"))
    by_task: Dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for batch_idx, (_set_name, cpu_batch, _global_batch_size) in enumerate(eval_loader, start=1):
            batch = {key: value.to(device) for key, value in cpu_batch.items()}
            pids = batch["puzzle_identifiers"].detach().cpu().numpy().tolist()
            with torch.device(device.type):
                carry = core_model.initial_carry(batch)
            outputs = None
            for _step in range(1, int(config.arch.halt_max_steps) + 1):
                carry, outputs = core_model(carry=carry, batch=batch)
            preds = torch.argmax(outputs["logits"], dim=-1).detach().cpu().numpy()
            for row_idx, pid in enumerate(pids):
                pid = int(pid)
                if 0 < pid < len(eval_ids):
                    task_id = eval_ids[pid]
                    by_task.setdefault(task_id, preds[row_idx].copy())
            if batch_idx % 50 == 0:
                print(f"[c0-cache] batches={batch_idx}, tasks={len(by_task)}")
    task_ids = np.asarray(sorted(by_task), dtype="<U32")
    pred_arr = np.stack([by_task[task_id] for task_id in task_ids], axis=0)
    np.savez_compressed(cache_path, task_ids=task_ids, preds=pred_arr)
    del loss_head, core_model
    torch.cuda.empty_cache()
    return {task_id: token_seq_to_grid(by_task[task_id]) for task_id in by_task}


def unique_grids(grids: Iterable[Grid], limit: int = 200) -> List[Grid]:
    seen = set()
    out = []
    for grid in grids:
        arr = np.asarray(grid, dtype=np.uint8)
        key = grid_key(arr)
        if key in seen:
            continue
        seen.add(key)
        out.append(arr)
        if len(out) >= limit:
            break
    return out


def exact_any(candidates: List[Grid], target: Grid) -> bool:
    return any(candidate.shape == target.shape and np.array_equal(candidate, target) for candidate in candidates)


def flood_external_background(grid: Grid, background: int) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if int(grid[r, c]) == background and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if int(grid[r, c]) == background and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == background:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def oracle_fill(c0_grid: Grid) -> List[Grid]:
    background = most_common_color(c0_grid)
    external = flood_external_background(c0_grid, background)
    holes = (c0_grid == background) & (~external)
    if not holes.any():
        return []
    h, w = c0_grid.shape
    labels = np.zeros((h, w), dtype=np.int32)
    comps = []
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not holes[r, c] or labels[r, c]:
                continue
            comp_id += 1
            stack = [(r, c)]
            labels[r, c] = comp_id
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and holes[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        stack.append((nr, nc))
            comps.append(cells)
    candidates = []
    for cells in comps:
        neighbour_colors = []
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and int(c0_grid[nr, nc]) != background:
                    neighbour_colors.append(int(c0_grid[nr, nc]))
        for color, _count in Counter(neighbour_colors).most_common(3):
            out = c0_grid.copy()
            for r, c in cells:
                out[r, c] = color
            candidates.append(out)
    return unique_grids(candidates)


def rank_components(grid: Grid, key: str) -> List[Component]:
    comps = [c for c in connected_components(grid, include_background=True) if c.color != most_common_color(grid)]
    if key == "area_desc":
        return sorted(comps, key=lambda c: (-c.area, c.centroid[0], c.centroid[1], c.color))
    if key == "row":
        return sorted(comps, key=lambda c: (c.centroid[0], c.centroid[1], -c.area, c.color))
    if key == "col":
        return sorted(comps, key=lambda c: (c.centroid[1], c.centroid[0], -c.area, c.color))
    raise ValueError(key)


def derive_recolour_rules(demos: List[Dict[str, object]]) -> List[Tuple[str, int, int]]:
    rules = []
    for rank_key in ("area_desc", "row", "col"):
        per_demo = []
        valid = True
        for demo in demos:
            inp = np.asarray(demo["input"], dtype=np.uint8)
            out = np.asarray(demo["output"], dtype=np.uint8)
            if inp.shape != out.shape:
                valid = False
                break
            inp_comps = rank_components(inp, rank_key)
            changed = []
            for idx, comp in enumerate(inp_comps):
                vals = np.asarray([out[r, c] for r, c in comp.cells], dtype=np.uint8)
                if vals.size == 0:
                    continue
                uniq = np.unique(vals)
                if uniq.size == 1 and int(uniq[0]) != comp.color:
                    changed.append((idx, int(uniq[0])))
            per_demo.append(changed)
        if not valid or not per_demo:
            continue
        common = set(per_demo[0])
        for changed in per_demo[1:]:
            common &= set(changed)
        for idx, color in common:
            rules.append((rank_key, idx, color))
    return sorted(set(rules))


def oracle_recolour(demos: List[Dict[str, object]], test_grid: Grid) -> List[Grid]:
    candidates = []
    for rank_key, idx, color in derive_recolour_rules(demos):
        comps = rank_components(test_grid, rank_key)
        if idx >= len(comps):
            continue
        out = test_grid.copy()
        for r, c in comps[idx].cells:
            out[r, c] = color
        candidates.append(out)
    return unique_grids(candidates)


def derive_translation_vectors(demos: List[Dict[str, object]]) -> List[Tuple[int, int]]:
    per_demo = []
    for demo in demos:
        inp = np.asarray(demo["input"], dtype=np.uint8)
        out = np.asarray(demo["output"], dtype=np.uint8)
        inp_comps = [c for c in connected_components(inp, include_background=True) if c.color != most_common_color(inp)]
        out_comps = [c for c in connected_components(out, include_background=True) if c.color != most_common_color(out)]
        vectors = []
        for a in inp_comps:
            for b in out_comps:
                if a.color == b.color and a.area == b.area and a.shape_signature == b.shape_signature:
                    dr = b.bbox[0] - a.bbox[0]
                    dc = b.bbox[1] - a.bbox[1]
                    if dr != 0 or dc != 0:
                        vectors.append((dr, dc))
        per_demo.append(set(vectors))
    if not per_demo:
        return []
    common = set(per_demo[0])
    for vectors in per_demo[1:]:
        common &= vectors
    return sorted(common)


def paste_component(out: Grid, comp: Component, dr: int, dc: int, erase: bool) -> Optional[Grid]:
    background = most_common_color(out)
    result = out.copy()
    if erase:
        for r, c in comp.cells:
            result[r, c] = background
    for r, c in comp.cells:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < result.shape[0] and 0 <= nc < result.shape[1]):
            return None
        result[nr, nc] = comp.color
    return result


def oracle_translate(demos: List[Dict[str, object]], test_grid: Grid) -> List[Grid]:
    vectors = derive_translation_vectors(demos)
    comps = [c for c in connected_components(test_grid, include_background=True) if c.color != most_common_color(test_grid)]
    candidates = []
    for dr, dc in vectors:
        for erase in (False, True):
            out = test_grid.copy()
            ok = True
            for comp in comps:
                pasted = paste_component(out, comp, dr, dc, erase=erase)
                if pasted is None:
                    ok = False
                    break
                out = pasted
            if ok:
                candidates.append(out)
        for comp in comps:
            for erase in (False, True):
                pasted = paste_component(test_grid, comp, dr, dc, erase=erase)
                if pasted is not None:
                    candidates.append(pasted)
    return unique_grids(candidates)


def separator_axes(grid: Grid) -> List[Tuple[str, float, int]]:
    axes = []
    h, w = grid.shape
    background = most_common_color(grid)
    for r in range(h):
        vals = np.unique(grid[r, :])
        if vals.size == 1 and int(vals[0]) != background:
            axes.append(("h", float(r), int(vals[0])))
    for c in range(w):
        vals = np.unique(grid[:, c])
        if vals.size == 1 and int(vals[0]) != background:
            axes.append(("v", float(c), int(vals[0])))
    return axes


def reflect_cell(r: int, c: int, axis: str, index: float) -> Optional[Tuple[int, int]]:
    if axis == "h":
        nr = int(round(2 * index - r))
        nc = c
        if abs((2 * index - r) - nr) > 1e-6:
            return None
    else:
        nr = r
        nc = int(round(2 * index - c))
        if abs((2 * index - c) - nc) > 1e-6:
            return None
    return nr, nc


def oracle_reflect(demos: List[Dict[str, object]], test_grid: Grid) -> List[Grid]:
    demo_axis_kinds = []
    for demo in demos:
        axes = separator_axes(np.asarray(demo["input"], dtype=np.uint8))
        demo_axis_kinds.append({(axis, color) for axis, _idx, color in axes})
    if not demo_axis_kinds:
        return []
    allowed = set(demo_axis_kinds[0])
    for kinds in demo_axis_kinds[1:]:
        allowed &= kinds
    candidates = []
    comps = [c for c in connected_components(test_grid, include_background=True) if c.color != most_common_color(test_grid)]
    for axis, idx, color in separator_axes(test_grid):
        if (axis, color) not in allowed:
            continue
        for comp in comps:
            if comp.color == color:
                continue
            out = test_grid.copy()
            ok = True
            for r, c in comp.cells:
                reflected = reflect_cell(r, c, axis, idx)
                if reflected is None:
                    ok = False
                    break
                nr, nc = reflected
                if not (0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]):
                    ok = False
                    break
                out[nr, nc] = comp.color
            if ok:
                candidates.append(out)
    return unique_grids(candidates)


def connect_same_color(grid: Grid, axis: str, color: Optional[int]) -> Grid:
    out = grid.copy()
    background = most_common_color(grid)
    colors = [color] if color is not None else [int(x) for x in np.unique(grid) if int(x) != background]
    h, w = grid.shape
    for cval in colors:
        if axis == "row":
            for r in range(h):
                positions = [c for c in range(w) if int(grid[r, c]) == int(cval)]
                if len(positions) >= 2:
                    for a, b in zip(positions[:-1], positions[1:]):
                        if np.all((grid[r, a : b + 1] == background) | (grid[r, a : b + 1] == cval)):
                            out[r, a : b + 1] = cval
        else:
            for c in range(w):
                positions = [r for r in range(h) if int(grid[r, c]) == int(cval)]
                if len(positions) >= 2:
                    for a, b in zip(positions[:-1], positions[1:]):
                        if np.all((grid[a : b + 1, c] == background) | (grid[a : b + 1, c] == cval)):
                            out[a : b + 1, c] = cval
    return out


def oracle_extend(demos: List[Dict[str, object]], test_grid: Grid) -> List[Grid]:
    colors = set()
    for demo in demos:
        inp = np.asarray(demo["input"], dtype=np.uint8)
        out = np.asarray(demo["output"], dtype=np.uint8)
        if inp.shape == out.shape:
            changed_to = np.unique(out[inp != out]).tolist()
            colors.update(int(x) for x in changed_to if int(x) != most_common_color(inp))
    candidates = []
    for axis in ("row", "col"):
        candidates.append(connect_same_color(test_grid, axis, None))
        for color in colors:
            candidates.append(connect_same_color(test_grid, axis, color))
    return unique_grids(candidates)


def candidate_rows_for_family(task_id: str, family: str, candidates: List[Grid], target: Grid) -> List[Dict[str, object]]:
    rows = []
    for idx, candidate in enumerate(candidates):
        rows.append(
            {
                "task_id": task_id,
                "family": family,
                "candidate_index": idx,
                "candidate_hash": grid_key(candidate),
                "candidate_shape": f"{candidate.shape[0]}x{candidate.shape[1]}",
                "candidate_exact": int(candidate.shape == target.shape and np.array_equal(candidate, target)),
                "candidate_grid": json.dumps(candidate.astype(int).tolist(), separators=(",", ":")),
            }
        )
    return rows


def score_task(
    task_id: str,
    bucket: str,
    c0_exact: int,
    c0_grid: Grid,
    task: Dict[str, object],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    demos = list(task["train"])
    test_grid = task_input(task)
    target = task_output(task)
    family_candidates = {
        "fill": oracle_fill(c0_grid),
        "translate": oracle_translate(demos, test_grid),
        "recolour": oracle_recolour(demos, test_grid),
        "reflect": oracle_reflect(demos, test_grid),
        "extend": oracle_extend(demos, test_grid),
    }
    rows = []
    exact_by_family = {}
    replace_loss_by_family = {}
    for family, candidates in family_candidates.items():
        exact_by_family[family] = int(exact_any(candidates, target))
        unique_replace = len(candidates) == 1
        replace_exact = exact_by_family[family] if unique_replace else c0_exact
        replace_loss_by_family[family] = int(c0_exact and unique_replace and not replace_exact)
        rows.extend(candidate_rows_for_family(task_id, family, candidates, target))
    oracle_recovered_by = [family for family in FAMILIES if exact_by_family[family]]
    any_candidate_exact = int(bool(oracle_recovered_by))
    oracle_any_exact = int(c0_exact or any_candidate_exact)
    ledger = {
        "task_id": task_id,
        "bucket": bucket,
        "c0_exact": c0_exact,
        "fill_exact": exact_by_family["fill"],
        "translate_exact": exact_by_family["translate"],
        "recolour_exact": exact_by_family["recolour"],
        "reflect_exact": exact_by_family["reflect"],
        "extend_exact": exact_by_family["extend"],
        "oracle_any_exact": oracle_any_exact,
        "fill_replace_loss": replace_loss_by_family["fill"],
        "translate_replace_loss": replace_loss_by_family["translate"],
        "recolour_replace_loss": replace_loss_by_family["recolour"],
        "reflect_replace_loss": replace_loss_by_family["reflect"],
        "extend_replace_loss": replace_loss_by_family["extend"],
        "oracle_gain": int((not c0_exact) and any_candidate_exact),
        "oracle_recovered_by": ";".join(oracle_recovered_by),
        "n_fill_candidates": len(family_candidates["fill"]),
        "n_translate_candidates": len(family_candidates["translate"]),
        "n_recolour_candidates": len(family_candidates["recolour"]),
        "n_reflect_candidates": len(family_candidates["reflect"]),
        "n_extend_candidates": len(family_candidates["extend"]),
    }
    return ledger, rows


def summarize(task_rows: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    bucket_rows = []
    for bucket in ["ALL"] + BUCKETS:
        rows = task_rows if bucket == "ALL" else [r for r in task_rows if r["bucket"] == bucket]
        if not rows:
            continue
        rec = {
            "bucket": bucket,
            "tasks": len(rows),
            "c0_exact": sum(int(r["c0_exact"]) for r in rows),
            "oracle_any_exact": sum(int(r["oracle_any_exact"]) for r in rows),
            "oracle_gain": sum(int(r["oracle_gain"]) for r in rows),
        }
        for family in FAMILIES:
            rec[f"{family}_exact"] = sum(int(r[f"{family}_exact"]) for r in rows)
            rec[f"{family}_replace_loss"] = sum(int(r[f"{family}_replace_loss"]) for r in rows)
        bucket_rows.append(rec)

    confusion_rows = []
    for bucket in BUCKETS:
        rows = [r for r in task_rows if r["bucket"] == bucket]
        for family in FAMILIES:
            confusion_rows.append(
                {
                    "bucket": bucket,
                    "family": family,
                    "repair_exact": sum(int(r[f"{family}_exact"]) for r in rows),
                    "new_exact": sum(int((not int(r["c0_exact"])) and int(r[f"{family}_exact"])) for r in rows),
                    "replace_loss": sum(int(r[f"{family}_replace_loss"]) for r in rows),
                }
            )
    return bucket_rows, confusion_rows


def write_verdict(out_dir: Path, task_rows: List[Dict[str, object]], bucket_rows: List[Dict[str, object]], confusion_rows: List[Dict[str, object]]) -> None:
    both_fail_rows = [r for r in task_rows if r["bucket"] == "both_fail"]
    both_fail_oracle_gain = sum(int(r["oracle_gain"]) for r in both_fail_rows)
    best_primitive = 0
    best_family = ""
    for family in FAMILIES:
        gains = sum(int((not int(r["c0_exact"])) and int(r[f"{family}_exact"])) for r in both_fail_rows)
        if gains > best_primitive:
            best_primitive = gains
            best_family = family
    c0_solved = [r for r in task_rows if int(r["c0_exact"])]
    combined_replace_loss = sum(
        int(any(int(r[f"{family}_replace_loss"]) for family in FAMILIES))
        for r in c0_solved
    )
    overall_c0 = sum(int(r["c0_exact"]) for r in task_rows)
    overall_oracle = sum(int(r["oracle_any_exact"]) for r in task_rows)
    keep = both_fail_oracle_gain >= 8 and best_primitive >= 3 and combined_replace_loss == 0
    verdict = "KEEP" if keep else "REJECT"
    reason = (
        "object-centric oracle primitives clear the preregistered headroom gates."
        if keep
        else "object-centric oracle primitives do not clear preregistered both_fail headroom gates."
    )
    next_step = (
        "Freeze primitive interface and design primitive-conditioned residual branch."
        if keep
        else "Do not build primitive-conditioned C2 branch from this evidence; inspect candidate misses or move to richer program search."
    )
    lines = [
        f"verdict: {verdict}",
        f"both_fail oracle_any gains: {both_fail_oracle_gain}",
        f"best primitive both_fail solo gains: {best_primitive} ({best_family or 'none'})",
        f"combined replace_loss on C0 solved tasks: {combined_replace_loss}",
        f"overall C0 exact: {overall_c0}/400",
        f"overall oracle_any exact: {overall_oracle}/400",
        f"reason: {reason}",
        f"next step: {next_step}",
        "",
        "scientific note: public eval tasks informed this probe; treat as development evidence only.",
    ]
    (out_dir / "rejection_or_keep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def run_probe(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir).resolve()
    preregister(out_dir, repo_root)
    dataset = Path(args.dataset).resolve()
    tasks = load_tasks(dataset)
    ledger = read_csv(Path(args.c0_ledger).resolve())
    c0_grids = generate_c0_cache(
        out_dir / "c0_prediction_cache.npz",
        Path(args.c0_config).resolve(),
        Path(args.c0_checkpoint).resolve(),
        int(args.global_batch_size),
    )

    task_rows = []
    candidate_rows = []
    for idx, ref in enumerate(ledger, start=1):
        task_id = ref["task_id"]
        if task_id not in tasks:
            raise KeyError(f"Missing task in canonical JSON: {task_id}")
        if task_id not in c0_grids:
            raise KeyError(f"Missing C0 prediction cache for task: {task_id}")
        row, candidates = score_task(
            task_id=task_id,
            bucket=ref["bucket"],
            c0_exact=int(float(ref["exact_accuracy"]) > 0.5),
            c0_grid=c0_grids[task_id],
            task=tasks[task_id],
        )
        task_rows.append(row)
        candidate_rows.extend(candidates)
        if idx % 50 == 0:
            print(f"[d0-obj] scored tasks={idx}, candidates={len(candidate_rows)}")

    task_fields = [
        "task_id",
        "bucket",
        "c0_exact",
        "fill_exact",
        "translate_exact",
        "recolour_exact",
        "reflect_exact",
        "extend_exact",
        "oracle_any_exact",
        "fill_replace_loss",
        "translate_replace_loss",
        "recolour_replace_loss",
        "reflect_replace_loss",
        "extend_replace_loss",
        "oracle_gain",
        "oracle_recovered_by",
        "n_fill_candidates",
        "n_translate_candidates",
        "n_recolour_candidates",
        "n_reflect_candidates",
        "n_extend_candidates",
    ]
    candidate_fields = [
        "task_id",
        "family",
        "candidate_index",
        "candidate_hash",
        "candidate_shape",
        "candidate_exact",
        "candidate_grid",
    ]
    bucket_rows, confusion_rows = summarize(task_rows)
    write_csv(out_dir / "d0_obj_task_ledger.csv", task_rows, task_fields)
    write_csv(out_dir / "d0_obj_candidate_outputs.csv", candidate_rows, candidate_fields)
    write_csv(out_dir / "d0_obj_primitive_confusion.csv", confusion_rows, list(confusion_rows[0].keys()))
    write_csv(out_dir / "d0_obj_bucket_summary.csv", bucket_rows, list(bucket_rows[0].keys()))
    write_verdict(out_dir, task_rows, bucket_rows, confusion_rows)


def assert_grid_equal(actual: Grid, expected: List[List[int]]) -> None:
    exp = np.asarray(expected, dtype=np.uint8)
    assert actual.shape == exp.shape and np.array_equal(actual, exp), f"{actual.tolist()} != {exp.tolist()}"


def self_test() -> None:
    grid = np.asarray([[0, 1, 1], [0, 1, 0], [2, 0, 2]], dtype=np.uint8)
    comps = connected_components(grid, include_background=True)
    assert len([c for c in comps if c.color == 1]) == 1
    assert len([c for c in comps if c.color == 2]) == 2

    fill_grid = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0],
            [0, 3, 0, 3, 0],
            [0, 3, 3, 3, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    fill_candidates = oracle_fill(fill_grid)
    assert any(c[2, 2] == 3 for c in fill_candidates)

    recolour_demos = [
        {"input": [[1, 1, 0, 0], [1, 0, 2, 0], [0, 0, 0, 0]], "output": [[5, 5, 0, 0], [5, 0, 2, 0], [0, 0, 0, 0]]},
        {"input": [[1, 1, 0, 0], [1, 0, 2, 0], [0, 0, 0, 0]], "output": [[5, 5, 0, 0], [5, 0, 2, 0], [0, 0, 0, 0]]},
    ]
    recolour_out = oracle_recolour(recolour_demos, np.asarray([[1, 1, 0, 0], [1, 0, 2, 0], [0, 0, 0, 0]], dtype=np.uint8))
    assert any(np.array_equal(c, np.asarray([[5, 5, 0, 0], [5, 0, 2, 0], [0, 0, 0, 0]], dtype=np.uint8)) for c in recolour_out)

    translate_demos = [
        {"input": [[4, 4, 0, 0, 0], [4, 0, 0, 0, 0], [0, 0, 0, 0, 0]], "output": [[4, 4, 0, 4, 4], [4, 0, 0, 4, 0], [0, 0, 0, 0, 0]]},
        {"input": [[4, 4, 0, 0, 0], [4, 0, 0, 0, 0], [0, 0, 0, 0, 0]], "output": [[4, 4, 0, 4, 4], [4, 0, 0, 4, 0], [0, 0, 0, 0, 0]]},
    ]
    assert derive_translation_vectors(translate_demos)

    reflect_demos = [
        {
            "input": [[1, 0, 9, 0, 0], [1, 0, 9, 0, 0]],
            "output": [[1, 0, 9, 0, 1], [1, 0, 9, 0, 1]],
        },
        {
            "input": [[1, 0, 9, 0, 0], [1, 0, 9, 0, 0]],
            "output": [[1, 0, 9, 0, 1], [1, 0, 9, 0, 1]],
        },
    ]
    reflect_out = oracle_reflect(reflect_demos, np.asarray([[1, 0, 9, 0, 0], [1, 0, 9, 0, 0]], dtype=np.uint8))
    assert any(np.array_equal(c, np.asarray([[1, 0, 9, 0, 1], [1, 0, 9, 0, 1]], dtype=np.uint8)) for c in reflect_out)

    extend_demos = [
        {"input": [[0, 0, 0], [2, 0, 2], [0, 0, 0]], "output": [[0, 0, 0], [2, 2, 2], [0, 0, 0]]},
        {"input": [[0, 0, 0], [2, 0, 2], [0, 0, 0]], "output": [[0, 0, 0], [2, 2, 2], [0, 0, 0]]},
    ]
    extend_out = oracle_extend(extend_demos, np.asarray([[0, 0, 0], [2, 0, 2], [0, 0, 0]], dtype=np.uint8))
    assert any(np.array_equal(c, np.asarray([[0, 0, 0], [2, 2, 2], [0, 0, 0]], dtype=np.uint8)) for c in extend_out)

    c0_exact = 0
    candidates = [np.asarray([[1]], dtype=np.uint8)]
    target = np.asarray([[1]], dtype=np.uint8)
    assert exact_any(candidates, target)
    assert int((not c0_exact) and exact_any(candidates, target)) == 1
    print("[self-test] PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="D0-OBJ oracle coverage probe.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dataset", default="data/arc-agi-evaluation-full400-seed0")
    parser.add_argument("--c0-ledger")
    parser.add_argument("--c0-config")
    parser.add_argument("--c0-checkpoint")
    parser.add_argument("--out-dir", default="reports/d0_obj_oracle_probe")
    parser.add_argument("--global-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    missing = [name for name in ("c0_ledger", "c0_config", "c0_checkpoint") if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required arguments: {missing}")
    run_probe(args)


if __name__ == "__main__":
    main()
