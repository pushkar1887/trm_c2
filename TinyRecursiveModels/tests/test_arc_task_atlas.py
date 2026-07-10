import sys
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATASET_DIR = ROOT / "data" / "arc1concept-aug-0" / "train"
sys.path.insert(0, str(SCRIPTS))

from arc_task_atlas_analysis import (  # noqa: E402
    DecodedGrid,
    ExampleRecord,
    TaskRecord,
    analyze_task,
    decode_canvas,
    load_dataset,
)
from arc_task_atlas_render import (  # noqa: E402
    ARC_PALETTE,
    render_index_html,
    render_summary_svg,
    render_task_svg,
)
from build_arc_task_atlas import write_atlas  # noqa: E402


def encode_fixture(grid: np.ndarray, row: int, col: int) -> np.ndarray:
    canvas = np.zeros((30, 30), dtype=np.uint8)
    height, width = grid.shape
    canvas[row : row + height, col : col + width] = grid + 2
    if row + height < 30:
        canvas[row + height, col : col + width] = 1
    if col + width < 30:
        canvas[row : row + height, col + width] = 1
    return canvas.reshape(-1)


def test_decode_canvas_preserves_black_cells_and_offset():
    logical = np.array([[0, 0, 2], [0, 3, 0]], dtype=np.uint8)
    canvas = encode_fixture(logical, row=7, col=11)

    decoded = decode_canvas(canvas)

    assert decoded.offset == (7, 11)
    assert np.array_equal(decoded.grid, logical)


def test_load_tasks_uses_puzzle_indices():
    tasks = load_dataset(DATASET_DIR)

    assert len(tasks) == 960
    assert sum(len(task.examples) for task in tasks) == 3988
    assert tasks[0].identifier == "8be77c9e"
    assert len(tasks[0].examples) == 4


def make_task(pairs: list[tuple[np.ndarray, np.ndarray]], identifier: str = "fixture") -> TaskRecord:
    examples = []
    for index, (inp, out) in enumerate(pairs):
        examples.append(
            ExampleRecord(
                index=index,
                input=DecodedGrid(np.asarray(inp, dtype=np.uint8), (0, 0)),
                output=DecodedGrid(np.asarray(out, dtype=np.uint8), (0, 0)),
                target_height=int(out.shape[0]),
                target_width=int(out.shape[1]),
            )
        )
    return TaskRecord(ordinal=1, identifier_index=1, identifier=identifier, examples=tuple(examples))


def recolor(grid: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    result = grid.copy()
    for source, target in mapping.items():
        result[grid == source] = target
    return result


def test_global_recolor_is_proven():
    a = np.array([[0, 2, 2], [0, 1, 2]], dtype=np.uint8)
    b = np.array([[2, 0], [1, 2]], dtype=np.uint8)
    result = analyze_task(make_task([(a, recolor(a, {2: 7})), (b, recolor(b, {2: 7}))]))

    assert result.family == "clean_recolor"
    assert result.confidence == "proven"
    assert result.evidence["global_color_map"]["2"] == 7


def test_dihedral_is_proven():
    a = np.array([[1, 1, 0], [2, 0, 0]], dtype=np.uint8)
    b = np.array([[3, 3, 0], [4, 0, 0]], dtype=np.uint8)
    result = analyze_task(make_task([(a, np.fliplr(a)), (b, np.fliplr(b))]))

    assert result.family == "dihedral"
    assert result.confidence == "proven"
    assert result.operation == "flip_h"


def test_tiling_is_proven():
    a = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    b = np.array([[4, 0, 4]], dtype=np.uint8)
    result = analyze_task(make_task([(a, np.tile(a, (2, 3))), (b, np.tile(b, (2, 3)))]))

    assert result.family == "tile"
    assert result.confidence == "proven"
    assert result.operation == "repeat_2x3"


def test_size_change_is_probable_not_proven():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)
    b = np.array([[2, 2, 0], [2, 0, 0], [0, 0, 0]], dtype=np.uint8)
    result = analyze_task(make_task([(a, a[:2, :2]), (b, b[:2, :2])]))

    assert result.family == "size_change"
    assert result.confidence == "probable"


def test_task_svg_is_valid_xml_and_contains_every_pair():
    a = np.array([[0, 2, 2], [0, 1, 2]], dtype=np.uint8)
    b = np.array([[2, 0], [1, 2]], dtype=np.uint8)
    analysis = analyze_task(
        make_task([(a, recolor(a, {2: 7})), (b, recolor(b, {2: 7}))], identifier="svg-fixture")
    )

    svg = render_task_svg(analysis)

    ET.fromstring(svg)
    assert "svg-fixture" in svg
    assert svg.count("Demo pair") == 2
    assert all(color in svg for color in ARC_PALETTE)


def test_fixture_atlas_builds_index_summary_json_and_task_svgs():
    a = np.array([[0, 2], [1, 2]], dtype=np.uint8)
    recolor_analysis = analyze_task(make_task([(a, recolor(a, {2: 7}))], identifier="recolor-fixture"))
    resize_analysis = analyze_task(make_task([(a, np.tile(a, (2, 2)))], identifier="tile-fixture"))

    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "atlas"
        write_atlas((recolor_analysis, resize_analysis), output)

        records = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
        index = (output / "index.html").read_text(encoding="utf-8")
        summary = (output / "summary.svg").read_text(encoding="utf-8")
        ET.fromstring(summary)
        assert len(records) == 2
        assert "family-filter" in index
        assert "confidence-filter" in index
        assert "recolor-fixture" in index
        assert len(list((output / "tasks").glob("*.svg"))) == 2


def main():
    test_decode_canvas_preserves_black_cells_and_offset()
    test_load_tasks_uses_puzzle_indices()
    test_global_recolor_is_proven()
    test_dihedral_is_proven()
    test_tiling_is_proven()
    test_size_change_is_probable_not_proven()
    test_task_svg_is_valid_xml_and_contains_every_pair()
    test_fixture_atlas_builds_index_summary_json_and_task_svgs()
    print("test_arc_task_atlas: PASS")


if __name__ == "__main__":
    main()
