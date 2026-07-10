"""REAL ARC-AGI-1 benchmark for V3 floor-safe committed solving.

Without --model-dump this is the deterministic DSL committed-solve benchmark.
With --model-dump it consumes TRM top-K candidate grids and evaluates the V3
candidate pool:
    {TRM top-K candidates} union {deterministic DSL candidate} union {floor=TRM majority}

NON-PEEKING: candidates are selected from support demos and model votes only.
The solution is loaded only after candidate selection, for scoring.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse import COLOR_OFFSET, _parse_same
from solve import _default_geo_ops, _committed_solve
from model_candidate_dump import load_model_dump

COMBINED = Path(r"D:\trm_c2\TinyRecursiveModels\kaggle\combined")


def _grid(g):
    return torch.tensor(g, dtype=torch.long) + COLOR_OFFSET


def _eq(p, g):
    return p is not None and p.shape == g.shape and bool((p == g).all())


def _same_grid(a, b):
    return a is not None and b is not None and a.shape == b.shape and bool((a == b).all())


def _flat_tokens_to_grid(tokens, side=30):
    """Flat [900] TRM tokens -> compact token grid by colour-cell extent.

    PAD/EOS are structure markers, not ARC colours. The compact grid is the
    top-left rectangle covering all colour tokens (>= COLOR_OFFSET).
    """
    t = torch.as_tensor(tokens, dtype=torch.long).reshape(side, side)
    valid = t >= COLOR_OFFSET
    if not bool(valid.any()):
        return t[:0, :0].clone()
    nz = valid.nonzero(as_tuple=False)
    h = int(nz[:, 0].max().item()) + 1
    w = int(nz[:, 1].max().item()) + 1
    return t[:h, :w].clone()


def _first_distinct(candidates, reference, side=30):
    for cand in candidates:
        g = _flat_tokens_to_grid(cand, side=side)
        if not _same_grid(g, reference):
            return g
    return None


def _select_attempts(det_a1, det_floor, dump_record=None, side=30):
    """Return (attempt1, attempt2, source, model_floor_grid).

    If TRM top-K is present, attempt2 is always the TRM majority/floor. Attempt1
    is the safest distinct extra attempt: first a deterministic candidate that
    agrees with one of the TRM minority grids, otherwise the highest-vote
    minority grid, otherwise the deterministic candidate, otherwise floor.
    """
    if dump_record is None:
        return det_a1, det_floor, "deterministic", None

    candidates = dump_record["candidates"]
    model_floor = _flat_tokens_to_grid(candidates[0], side=side)
    minority = [_flat_tokens_to_grid(c, side=side) for c in candidates[1:]]

    if det_a1 is not None and not _same_grid(det_a1, model_floor):
        if any(_same_grid(det_a1, m) for m in minority):
            return det_a1, model_floor, "dsl_and_model_agree", model_floor

    nonfloor = _first_distinct(candidates[1:], model_floor, side=side)
    if nonfloor is not None:
        return nonfloor, model_floor, "model_topk_nonfloor", model_floor

    if det_a1 is not None and not _same_grid(det_a1, model_floor):
        return det_a1, model_floor, "dsl", model_floor

    return model_floor, model_floor, "model_floor", model_floor


def evaluate(split, combined=COMBINED, model_dump=None, side=30, quiet=False):
    combined = Path(combined)
    ch = json.loads((combined / f"arc-agi_{split}_challenges.json").read_text(encoding="utf-8"))
    sol = json.loads((combined / f"arc-agi_{split}_solutions.json").read_text(encoding="utf-8"))
    dump = load_model_dump(model_dump, side=side, kind="test") if model_dump is not None else {}
    if model_dump is not None and not dump:
        raise ValueError(f"model dump has zero target-test records: {model_dump}")
    geo_ops = _default_geo_ops()
    n = exact = floor = a1_only = model_floor = dump_hits = 0
    n_same = exact_same = n_diff = exact_diff = 0           # split by in-place (test out-shape == in-shape) vs size-change
    sources: dict[str, int] = {}
    for tid, task in ch.items():
        demos = task["train"]
        gin = {i: _grid(d["input"]) for i, d in enumerate(demos)}
        gout = {i: _grid(d["output"]) for i, d in enumerate(demos)}
        labels = {i: _parse_same(gin[i]) for i in gin}
        sup = list(range(len(demos)))
        for t, test in enumerate(task["test"]):
            h = len(demos) + t
            gin[h] = _grid(test["input"])
            labels[h] = _parse_same(gin[h])
            det_a1, det_floor = _committed_solve(gin, gout, sup, h, geo_ops, labels)
            target = _grid(sol[tid][t])
            record = dump.get((tid, t))
            if record is not None:
                dump_hits += 1
            a1, a2, source, mf = _select_attempts(det_a1, det_floor, record, side=side)
            sources[source] = sources.get(source, 0) + 1
            ok = _eq(a1, target) or _eq(a2, target)
            n += 1
            exact += int(ok)
            floor += int(_eq(a2, target))
            a1_only += int(_eq(a1, target))
            model_floor += int(_eq(mf, target)) if mf is not None else 0
            if gin[h].shape == target.shape:
                n_same += 1; exact_same += int(ok)
            else:
                n_diff += 1; exact_diff += int(ok)
            del gin[h], labels[h]
    pc = lambda x, d: 100.0 * x / max(d, 1)
    res = {
        "tasks": len(ch),
        "n": n,
        "exact": exact,
        "floor": floor,
        "model_floor": model_floor,
        "a1_only": a1_only,
        "dump_hits": dump_hits,
        "exact_same": exact_same,
        "n_same": n_same,
        "exact_diff": exact_diff,
        "n_diff": n_diff,
        "sources": sources,
    }
    if not quiet:
        mode = "MODEL-DUMP+DSL" if model_dump is not None else "DETERMINISTIC-DSL"
        print(f"\n[ARC-AGI-1 {split.upper()} | {mode}]  tasks={len(ch)}  test-questions={n}")
        if model_dump is not None:
            print(f"  model-dump coverage          = {pc(dump_hits, n):5.2f}%   ({dump_hits}/{n})")
            print(f"  TRM-floor-only               = {pc(model_floor, dump_hits):5.2f}%   ({model_floor}/{dump_hits})")
        print(f"  COMMITTED-exact (2-attempt)  = {pc(exact, n):5.2f}%   ({exact}/{n})")
        print(f"  FLOOR-only (attempt2)        = {pc(floor, n):5.2f}%   (committed >= floor by construction: {exact >= floor})")
        print(f"  attempt1-only                = {pc(a1_only, n):5.2f}%")
        print(f"  in-place (out-shape==in)     = {pc(exact_same, n_same):5.2f}%  ({exact_same}/{n_same})")
        print(f"  size-change (out-shape!=in)  = {pc(exact_diff, n_diff):5.2f}%  ({exact_diff}/{n_diff})")
        if sources:
            print("  attempt1 source counts:")
            for src, cnt in sorted(sources.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"    {src:22s} {cnt}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="*", default=["evaluation", "training"])
    ap.add_argument("--combined", type=Path, default=COMBINED)
    ap.add_argument("--model-dump", type=Path, default=None)
    ap.add_argument("--side", type=int, default=30)
    args = ap.parse_args()
    for s in args.splits:
        evaluate(s, combined=args.combined, model_dump=args.model_dump, side=args.side)
