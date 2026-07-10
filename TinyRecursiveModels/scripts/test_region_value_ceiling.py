"""Tests for the region-WHERE x VALUE ceiling probe.

Run with:
  trm\Scripts\python.exe scripts\test_region_value_ceiling.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import csv
import json
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import verify_and_select_candidates as vsc


def _hole_task(side: int, fill_colour: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A colour-4 ring encloses colour-0 background; output fills only the hole."""
    g = torch.full((side, side), 0 + vsc.COLOR_OFFSET, dtype=torch.long)
    g[1:6, 1] = 4 + vsc.COLOR_OFFSET
    g[1:6, 5] = 4 + vsc.COLOR_OFFSET
    g[1, 1:6] = 4 + vsc.COLOR_OFFSET
    g[5, 1:6] = 4 + vsc.COLOR_OFFSET
    out = g.clone()
    out[2:5, 2:5] = fill_colour + vsc.COLOR_OFFSET
    return g.reshape(-1), out.reshape(-1)


def test_region_where_solves_enclosed_background_fill() -> None:
    side = 8
    demos = [_hole_task(side, fill_colour=7) for _ in range(3)]

    floor = vsc.score_lodo(demos, vsc.floor_predict, side)
    obj = vsc.score_lodo(demos, vsc.object_where_recolor_predict, side)
    region = vsc.score_lodo(demos, vsc.region_where_fill_predict, side)

    assert floor is not None and floor[0] < 1.0, f"floor must fail background fill, got {floor}"
    assert obj is not None and obj[0] == 0.0, f"object-WHERE must not solve background fill, got {obj}"
    assert region is not None and region[0] >= 1.0 - 1e-9, (
        f"region-WHERE must solve enclosed background fill, got {region}"
    )


def test_constant_fill_solves_enclosed_background_fill() -> None:
    side = 8
    demos = [_hole_task(side, fill_colour=7) for _ in range(3)]

    floor = vsc.score_lodo(demos, vsc.floor_predict, side)
    const = vsc.score_lodo(demos, vsc.constant_fill_predict, side)

    assert floor is not None and floor[0] < 1.0, f"floor must fail background fill, got {floor}"
    assert const is not None and const[0] >= 1.0 - 1e-9, (
        f"constant_fill_predict must solve the verified constant-fill task, got {const}"
    )


def test_conditional_recolor_microscope_identifies_enclosed_background_fill() -> None:
    side = 8
    demos = [_hole_task(side, fill_colour=7) for _ in range(3)]

    row = vsc._conditional_recolor_microscope_row("toy", "conditional_recolor", demos, side)

    assert row["n_changed"] == 27
    assert abs(row["changed_background_frac"] - 1.0) < 1e-6
    assert abs(row["changed_foreground_frac"] - 0.0) < 1e-6
    assert abs(row["changed_enclosed_frac"] - 1.0) < 1e-6
    assert row["best_explaining_family"] == "enclosed_background"


def test_microscope_can_include_all_categories() -> None:
    challenge = {
        "cond": {"train": [{"input": [[0, 0], [0, 0]], "output": [[1, 0], [0, 0]]}]},
        "other": {"train": [{"input": [[2, 2], [2, 2]], "output": [[3, 3], [3, 3]]}]},
    }
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        chal = root / "challenges.json"
        cats = root / "categories.csv"
        out_dir = root / "report"
        chal.write_text(json.dumps(challenge), encoding="utf-8")
        cats.write_text(
            "task_id,category\ncond,conditional_recolor\nother,other\n",
            encoding="utf-8",
        )

        vsc.run_conditional_recolor_microscope(
            side=4,
            challenges=chal,
            categories_csv=cats,
            out_dir=out_dir,
            include_all=True,
        )

        rows = list(csv.DictReader(open(out_dir / "per_task.csv", encoding="utf-8")))
    assert [r["task_id"] for r in rows] == ["cond", "other"]
    assert {r["category"] for r in rows} == {"conditional_recolor", "other"}


if __name__ == "__main__":
    test_region_where_solves_enclosed_background_fill()
    test_constant_fill_solves_enclosed_background_fill()
    test_conditional_recolor_microscope_identifies_enclosed_background_fill()
    test_microscope_can_include_all_categories()
    print("test_region_value_ceiling PASS")
