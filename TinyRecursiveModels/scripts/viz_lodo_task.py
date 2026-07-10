"""Dump ONE real INTRA-task LODO solve as compact JSON so it can be rendered + audited by eye.

For a real task: build the cell-colour recolour map from the SUPPORT demos of the SAME task ONLY,
apply it to the HELD-OUT demo's INPUT, compare to the held-out OUTPUT. This is the exact computation
check_extraction does -- emitted as JSON (grids as colour-int arrays + the coverage tally) so a viewer
can confirm it is intra-task (the other demos of the SAME task solve the held-out one).

Run:  trm\\Scripts\\python.exe scripts\\viz_lodo_task.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("DISABLE_COMPILE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.oracle_eval import (load_real_eval_batches, _task_recolour_consistency,
                                      COLOR_OFFSET, N_COLORS, GRID_SIDE)

modal = lambda lst: max(set(lst), key=lst.count)


def bbox(tok):
    g = tok.view(GRID_SIDE, GRID_SIDE)
    m = g >= COLOR_OFFSET
    rs = torch.where(m.any(1))[0]; cs = torch.where(m.any(0))[0]
    return rs.min().item(), cs.min().item(), rs.max().item(), cs.max().item()


def to_colours(tok2d):
    """token grid -> 2D list of colour ints (0..9), or -1 for PAD/EOS."""
    return [[(int(t) - COLOR_OFFSET if int(t) >= COLOR_OFFSET else -1) for t in row] for row in tok2d]


def cell_map(support):
    votes = {}
    for xin, yout in support:
        ch = (xin != yout) & (xin >= COLOR_OFFSET) & (yout >= COLOR_OFFSET)
        for i in torch.nonzero(ch).flatten().tolist():
            votes.setdefault(int(xin[i]) - COLOR_OFFSET, []).append(int(yout[i]) - COLOR_OFFSET)
    return {a: modal(v) for a, v in votes.items()}


def crop(tok, box):
    r0, c0, r1, c1 = box
    return tok.view(GRID_SIDE, GRID_SIDE)[r0:r1 + 1, c0:c1 + 1]


def build_panel(ci, co, valid, title):
    h = valid[-1]                       # hold out the LAST demo; support = the rest (same task)
    sup = valid[:-1]
    support = []
    for m in sup:
        box = bbox(ci[m])
        support.append({"in": to_colours(crop(ci[m], box)), "out": to_colours(crop(co[m], box))})
    cm = cell_map([(ci[m], co[m]) for m in sup])
    box = bbox(ci[h])
    gi, go = crop(ci[h], box), crop(co[h], box)
    H, W = gi.shape
    pred = gi.clone()
    for r in range(H):
        for c in range(W):
            t = int(gi[r, c])
            if t >= COLOR_OFFSET and (t - COLOR_OFFSET) in cm:
                pred[r, c] = cm[t - COLOR_OFFSET] + COLOR_OFFSET
    chg = (gi != go) & (gi >= COLOR_OFFSET) & (go >= COLOR_OFFSET)
    changed = [[1 if bool(chg[r, c]) else 0 for c in range(W)] for r in range(H)]
    correct = [[1 if (bool(chg[r, c]) and int(pred[r, c]) == int(go[r, c])) else 0
                for c in range(W)] for r in range(H)]
    corr = sum(sum(row) for row in correct)
    tot = sum(sum(row) for row in changed)
    return {
        "title": title, "n_support": len(sup), "n_total": len(valid),
        "map": {str(k): v for k, v in sorted(cm.items())},
        "support": support,
        "heldout": {"input": to_colours(gi), "pred": to_colours(pred), "actual": to_colours(go),
                    "changed": changed, "correct": correct,
                    "corr": corr, "tot": tot, "cov": round(100.0 * corr / max(tot, 1), 1)},
    }


def main():
    batches, _pids = load_real_eval_batches(24, 8)   # loader now also returns per-task identifiers
    clean = cond = None
    for ci, co, dv in batches:
        for b in range(ci.shape[0]):
            valid = [m for m in range(ci.shape[1]) if bool(dv[b, m])]
            if len(valid) < 3:
                continue
            r = _task_recolour_consistency(ci[b], co[b], dv[b])
            if r is None:
                continue
            if r[0] >= 0.97 and clean is None:
                clean = (ci[b], co[b], valid)
            if r[0] < 0.55 and cond is None:
                cond = (ci[b], co[b], valid)
    panels = []
    if clean:
        panels.append(build_panel(*clean, "CLEAN-RECOLOUR task  (gate ACCEPTS -> colour head fires)"))
    if cond:
        panels.append(build_panel(*cond, "CONDITIONAL task  (gate REJECTS -> no fixed colour map fits)"))
    print(json.dumps({"side": GRID_SIDE, "panels": panels}))


if __name__ == "__main__":
    main()
