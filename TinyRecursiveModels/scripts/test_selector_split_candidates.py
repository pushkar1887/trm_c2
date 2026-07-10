"""Self-tests for split deterministic candidates in the LODO selector.

Run:
  trm\Scripts\python.exe scripts\test_selector_split_candidates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_and_select_candidates
from parse import COLOR_OFFSET


def _seq(g: torch.Tensor, side: int) -> torch.Tensor:
    out = torch.zeros(side, side, dtype=torch.long)
    h, w = g.shape
    out[:h, :w] = g
    return out.reshape(-1)


def _set_cover_task():
    """Two-rule task: border-touching objects -> 7 and frame-contained objects -> 8."""
    oc = 4 + COLOR_OFFSET

    def frame(g, r0, r1, c0, c1):
        g[r0, c0:c1 + 1] = 3 + COLOR_OFFSET
        g[r1, c0:c1 + 1] = 3 + COLOR_OFFSET
        g[r0:r1 + 1, c0] = 3 + COLOR_OFFSET
        g[r0:r1 + 1, c1] = 3 + COLOR_OFFSET

    def mk(a, f, b, n):
        g = torch.full((14, 14), COLOR_OFFSET, dtype=torch.long)
        frame(g, *f)
        for r, c in (a, b, n):
            g[r:r + 2, c:c + 2] = oc
        o = g.clone()
        o[a[0]:a[0] + 2, a[1]:a[1] + 2] = 7 + COLOR_OFFSET
        o[b[0]:b[0] + 2, b[1]:b[1] + 2] = 8 + COLOR_OFFSET
        return g, o

    specs = [
        ((0, 0), (4, 9, 4, 9), (6, 6), (11, 11)),
        ((12, 0), (3, 8, 6, 11), (5, 8), (10, 2)),
        ((0, 12), (5, 10, 2, 7), (7, 4), (11, 10)),
    ]
    return [mk(*spec) for spec in specs]


def _largest_colour_copy_task():
    """Relation task: every smaller object copies the largest object's colour."""

    def mk(big_colour, big_box, smalls):
        g = torch.full((12, 12), COLOR_OFFSET, dtype=torch.long)
        g[big_box[0]:big_box[1] + 1, big_box[2]:big_box[3] + 1] = big_colour + COLOR_OFFSET
        for box, col in smalls:
            g[box[0]:box[1] + 1, box[2]:box[3] + 1] = col + COLOR_OFFSET
        o = g.clone()
        for box, _ in smalls:
            o[box[0]:box[1] + 1, box[2]:box[3] + 1] = big_colour + COLOR_OFFSET
        return g, o

    specs = [
        (4, (1, 4, 1, 4), [((1, 2, 7, 8), 5), ((7, 8, 2, 3), 6)]),
        (8, (6, 10, 6, 10), [((1, 2, 1, 2), 5), ((1, 2, 9, 10), 3)]),
        (3, (2, 6, 2, 6), [((9, 10, 9, 10), 5), ((9, 10, 1, 2), 7)]),
    ]
    return [mk(*spec) for spec in specs]


def test_selector_scores_set_cover_as_own_candidate() -> None:
    side = 16
    pairs = _set_cover_task()
    demos = [(_seq(pairs[i][0], side), _seq(pairs[i][1], side)) for i in (0, 2)]

    res = verify_and_select_candidates.evaluate_task(demos, side, task_id="synthetic-set-cover")

    assert "set_cover" in res["scores"]
    assert res["scores"]["set_cover"][0] == 1.0
    assert res["selector_exact"] == 1.0


def test_selector_scores_relation_copy_as_own_candidate() -> None:
    side = 16
    demos = [(_seq(x, side), _seq(y, side)) for x, y in _largest_colour_copy_task()]

    res = verify_and_select_candidates.evaluate_task(demos, side, task_id="synthetic-rel-copy")

    assert "rel_largest" in res["scores"]
    assert res["scores"]["rel_largest"][0] == 1.0
    assert res["winner"] == "rel_largest"


def test_candidate_row_reports_coverage_denominator() -> None:
    row = verify_and_select_candidates.format_candidate_row("refined_recipe", 1.0, 1.0, 1, 100)

    assert "1/100" in row
    assert "100.0%" in row


if __name__ == "__main__":
    test_selector_scores_set_cover_as_own_candidate()
    test_selector_scores_relation_copy_as_own_candidate()
    test_candidate_row_reports_coverage_denominator()
    print("test_selector_split_candidates PASS")
