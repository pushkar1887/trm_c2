"""Dataset decoding and evidence analysis for the static ARC task atlas."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


GRID_SIDE = 30
COLOR_OFFSET = 2
N_COLORS = 10


@dataclass(frozen=True)
class DecodedGrid:
    grid: np.ndarray
    offset: tuple[int, int]


@dataclass(frozen=True)
class ExampleRecord:
    index: int
    input: DecodedGrid
    output: DecodedGrid
    target_height: int
    target_width: int


@dataclass(frozen=True)
class TaskRecord:
    ordinal: int
    identifier_index: int
    identifier: str
    examples: tuple[ExampleRecord, ...]


@dataclass(frozen=True)
class TaskAnalysis:
    task: TaskRecord
    family: str
    confidence: str
    operation: str
    description: str
    evidence: dict[str, Any]
    example_statistics: tuple[dict[str, Any], ...]
    capability: dict[str, str]


CAPABILITY_BY_FAMILY: dict[str, dict[str, str]] = {
    "identity": {
        "existing": "Base TRM copy behavior",
        "support": "general",
        "addition": "No dedicated applier required",
    },
    "clean_recolor": {
        "existing": "ColorTransitionBank, support-consistency router, color-only repair head",
        "support": "implemented; training not assessed",
        "addition": "Validate router and unchanged-cell preservation end to end",
    },
    "conditional_recolor": {
        "existing": "Local/object gate, object features, learned rule vector",
        "support": "partial; training not assessed",
        "addition": "Relational object-color binding and stronger WHERE reasoning",
    },
    "dihedral": {
        "existing": "General TRM and diagnostic detector",
        "support": "no dedicated production applier",
        "addition": "Add deterministic applier only if aggregate coverage justifies it",
    },
    "translate": {
        "existing": "General TRM and diagnostic detector",
        "support": "no dedicated production applier",
        "addition": "Object/whole-grid movement applier if coverage is material",
    },
    "tile": {
        "existing": "General TRM and diagnostic detector",
        "support": "no dedicated production applier",
        "addition": "Repetition applier if coverage is material",
    },
    "size_change": {
        "existing": "Structure and shape heads",
        "support": "partial; training not assessed",
        "addition": "Explicit output-canvas, crop, extract, or resize applier",
    },
    "rearrangement": {
        "existing": "Base TRM and learned rule path",
        "support": "general only; training not assessed",
        "addition": "Object matching and movement applier",
    },
    "structural_other": {
        "existing": "Base TRM recursion",
        "support": "general only; training not assessed",
        "addition": "Cluster repeated evidence before adding a specialized mechanism",
    },
    "unknown": {
        "existing": "Base TRM recursion",
        "support": "not characterized",
        "addition": "Manual inspection or a new validated detector",
    },
}


def decode_canvas(tokens: np.ndarray) -> DecodedGrid:
    """Decode one translated 30x30 PAD/EOS canvas into its logical ARC grid."""
    array = np.asarray(tokens)
    if array.shape != (GRID_SIDE * GRID_SIDE,):
        raise ValueError(f"expected 900 tokens, got shape {array.shape}")
    canvas = array.reshape(GRID_SIDE, GRID_SIDE)
    color_mask = canvas >= COLOR_OFFSET
    if not color_mask.any():
        raise ValueError("canvas has no ARC grid cells")
    rows = np.flatnonzero(color_mask.any(axis=1))
    cols = np.flatnonzero(color_mask.any(axis=0))
    row0, row1 = int(rows[0]), int(rows[-1]) + 1
    col0, col1 = int(cols[0]), int(cols[-1]) + 1
    region = canvas[row0:row1, col0:col1]
    if np.any(region < COLOR_OFFSET):
        raise ValueError("decoded grid rectangle contains PAD or EOS tokens")
    return DecodedGrid(grid=(region - COLOR_OFFSET).astype(np.uint8), offset=(row0, col0))


def _load_array(dataset_dir: Path, name: str) -> np.ndarray:
    path = dataset_dir / f"all__{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r")


def load_dataset(dataset_dir: str | Path) -> tuple[TaskRecord, ...]:
    """Load task-grouped input/output pairs from a processed ARC training split."""
    dataset_dir = Path(dataset_dir)
    inputs = _load_array(dataset_dir, "inputs")
    labels = _load_array(dataset_dir, "labels")
    puzzle_ids = _load_array(dataset_dir, "puzzle_identifiers")
    boundaries = _load_array(dataset_dir, "puzzle_indices")
    target_heights = _load_array(dataset_dir, "target_height")
    target_widths = _load_array(dataset_dir, "target_width")

    example_count = len(inputs)
    if not (len(labels) == len(target_heights) == len(target_widths) == example_count):
        raise ValueError("example arrays have inconsistent lengths")
    if len(boundaries) != len(puzzle_ids) + 1:
        raise ValueError("puzzle_indices must contain one more boundary than puzzle identifiers")
    if int(boundaries[0]) != 0 or int(boundaries[-1]) != example_count:
        raise ValueError("puzzle boundaries do not span the complete example arrays")

    identifiers_path = dataset_dir.parent / "identifiers.json"
    with identifiers_path.open("r", encoding="utf-8") as handle:
        identifiers = json.load(handle)
    if not isinstance(identifiers, list):
        raise ValueError("identifiers.json must be an index-ordered list")

    tasks: list[TaskRecord] = []
    for ordinal, raw_identifier_index in enumerate(puzzle_ids):
        identifier_index = int(raw_identifier_index)
        if not 0 <= identifier_index < len(identifiers):
            raise ValueError(f"task {ordinal} has invalid identifier index {identifier_index}")
        start, end = int(boundaries[ordinal]), int(boundaries[ordinal + 1])
        if end <= start:
            raise ValueError(f"task {ordinal} has no examples")
        examples: list[ExampleRecord] = []
        for example_index in range(start, end):
            decoded_input = decode_canvas(inputs[example_index])
            decoded_output = decode_canvas(labels[example_index])
            target_height = int(target_heights[example_index])
            target_width = int(target_widths[example_index])
            if decoded_output.grid.shape != (target_height, target_width):
                raise ValueError(
                    f"example {example_index} output shape {decoded_output.grid.shape} "
                    f"does not match target {(target_height, target_width)}"
                )
            examples.append(
                ExampleRecord(
                    index=example_index,
                    input=decoded_input,
                    output=decoded_output,
                    target_height=target_height,
                    target_width=target_width,
                )
            )
        tasks.append(
            TaskRecord(
                ordinal=ordinal + 1,
                identifier_index=identifier_index,
                identifier=str(identifiers[identifier_index]),
                examples=tuple(examples),
            )
        )
    return tuple(tasks)


def global_color_map(inp: np.ndarray, out: np.ndarray) -> dict[int, int] | None:
    if inp.shape != out.shape:
        return None
    mapping: dict[int, int] = {}
    for raw_source, raw_target in zip(inp.flat, out.flat):
        source, target = int(raw_source), int(raw_target)
        previous = mapping.get(source)
        if previous is not None and previous != target:
            return None
        mapping[source] = target
    return mapping


def common_global_color_map(examples: Sequence[ExampleRecord]) -> dict[int, int] | None:
    merged: dict[int, int] = {}
    for example in examples:
        current = global_color_map(example.input.grid, example.output.grid)
        if current is None:
            return None
        for source, target in current.items():
            previous = merged.get(source)
            if previous is not None and previous != target:
                return None
            merged[source] = target
    return merged


DIHEDRAL_TRANSFORMS = {
    "rot90": lambda grid: np.rot90(grid, 1),
    "rot180": lambda grid: np.rot90(grid, 2),
    "rot270": lambda grid: np.rot90(grid, 3),
    "flip_h": np.fliplr,
    "flip_v": np.flipud,
    "transpose": lambda grid: grid.T,
    "antitranspose": lambda grid: np.flipud(np.fliplr(grid.T)),
}


def common_dihedral(examples: Sequence[ExampleRecord]) -> str | None:
    candidates = set(DIHEDRAL_TRANSFORMS)
    for example in examples:
        candidates = {
            name
            for name in candidates
            if DIHEDRAL_TRANSFORMS[name](example.input.grid).shape == example.output.grid.shape
            and np.array_equal(DIHEDRAL_TRANSFORMS[name](example.input.grid), example.output.grid)
        }
    return sorted(candidates)[0] if candidates else None


def _mode_color(grid: np.ndarray) -> int:
    return int(np.bincount(grid.ravel(), minlength=N_COLORS).argmax())


def shift_grid(grid: np.ndarray, dr: int, dc: int) -> np.ndarray:
    result = np.full_like(grid, _mode_color(grid))
    height, width = grid.shape
    src_r0, src_r1 = max(0, -dr), min(height, height - dr)
    src_c0, src_c1 = max(0, -dc), min(width, width - dc)
    dst_r0, dst_r1 = max(0, dr), min(height, height + dr)
    dst_c0, dst_c1 = max(0, dc), min(width, width + dc)
    if src_r1 > src_r0 and src_c1 > src_c0:
        result[dst_r0:dst_r1, dst_c0:dst_c1] = grid[src_r0:src_r1, src_c0:src_c1]
    return result


def common_translation(examples: Sequence[ExampleRecord], radius: int = GRID_SIDE - 1) -> tuple[int, int] | None:
    candidates: set[tuple[int, int]] | None = None
    for example in examples:
        inp, out = example.input.grid, example.output.grid
        if inp.shape != out.shape:
            return None
        height, width = inp.shape
        local = {
            (dr, dc)
            for dr in range(-min(radius, height - 1), min(radius, height - 1) + 1)
            for dc in range(-min(radius, width - 1), min(radius, width - 1) + 1)
            if (dr or dc) and np.array_equal(shift_grid(inp, dr, dc), out)
        }
        candidates = local if candidates is None else candidates & local
        if not candidates:
            return None
    return sorted(candidates)[0] if candidates else None


def common_tiling(examples: Sequence[ExampleRecord]) -> tuple[int, int] | None:
    factor: tuple[int, int] | None = None
    for example in examples:
        inp, out = example.input.grid, example.output.grid
        input_height, input_width = inp.shape
        output_height, output_width = out.shape
        if output_height % input_height or output_width % input_width:
            return None
        current = (output_height // input_height, output_width // input_width)
        if current == (1, 1) or not np.array_equal(np.tile(inp, current), out):
            return None
        if factor is not None and factor != current:
            return None
        factor = current
    return factor


def example_statistics(example: ExampleRecord) -> dict[str, Any]:
    inp, out = example.input.grid, example.output.grid
    same_shape = inp.shape == out.shape
    changed_cells = int(np.count_nonzero(inp != out)) if same_shape else None
    input_histogram = np.bincount(inp.ravel(), minlength=N_COLORS)
    output_histogram = np.bincount(out.ravel(), minlength=N_COLORS)
    return {
        "input_shape": [int(v) for v in inp.shape],
        "output_shape": [int(v) for v in out.shape],
        "input_offset": list(example.input.offset),
        "output_offset": list(example.output.offset),
        "input_palette": sorted(int(v) for v in np.unique(inp)),
        "output_palette": sorted(int(v) for v in np.unique(out)),
        "changed_cells": changed_cells,
        "input_cells": int(inp.size),
        "output_cells": int(out.size),
        "histogram_preserved": bool(same_shape and np.array_equal(input_histogram, output_histogram)),
    }


def _analysis(
    task: TaskRecord,
    family: str,
    confidence: str,
    operation: str,
    description: str,
    evidence: dict[str, Any],
) -> TaskAnalysis:
    return TaskAnalysis(
        task=task,
        family=family,
        confidence=confidence,
        operation=operation,
        description=description,
        evidence=evidence,
        example_statistics=tuple(example_statistics(example) for example in task.examples),
        capability=dict(CAPABILITY_BY_FAMILY[family]),
    )


def analyze_task(task: TaskRecord) -> TaskAnalysis:
    """Assign an evidence-backed broad operation family to one task."""
    examples = task.examples
    if all(np.array_equal(example.input.grid, example.output.grid) for example in examples):
        return _analysis(task, "identity", "proven", "identity", "Output exactly copies input.", {})

    color_map = common_global_color_map(examples)
    if color_map is not None and any(source != target for source, target in color_map.items()):
        rendered_map = {str(source): int(target) for source, target in sorted(color_map.items())}
        return _analysis(
            task,
            "clean_recolor",
            "proven",
            "global_color_map",
            "One position-preserving color map reconstructs every output.",
            {"global_color_map": rendered_map},
        )

    dihedral = common_dihedral(examples)
    if dihedral is not None:
        return _analysis(
            task,
            "dihedral",
            "proven",
            dihedral,
            f"The same {dihedral} transform reconstructs every output.",
            {"dihedral": dihedral},
        )

    translation = common_translation(examples)
    if translation is not None:
        dr, dc = translation
        return _analysis(
            task,
            "translate",
            "proven",
            f"shift_{dr}_{dc}",
            f"The same translation (row {dr:+d}, column {dc:+d}) reconstructs every output.",
            {"translation": [dr, dc]},
        )

    tiling = common_tiling(examples)
    if tiling is not None:
        vertical, horizontal = tiling
        return _analysis(
            task,
            "tile",
            "proven",
            f"repeat_{vertical}x{horizontal}",
            f"Repeating each input {vertical}x{horizontal} reconstructs every output.",
            {"tiling": [vertical, horizontal]},
        )

    stats = tuple(example_statistics(example) for example in examples)
    same_shape = [stat["input_shape"] == stat["output_shape"] for stat in stats]
    any_changed = any(stat["changed_cells"] for stat in stats if stat["changed_cells"] is not None)
    if not all(same_shape):
        return _analysis(
            task,
            "size_change",
            "probable",
            "size_change",
            "At least one output has different dimensions; no tested exact size operation explains all pairs.",
            {"shape_preserved_pairs": int(sum(same_shape)), "pair_count": len(examples)},
        )

    hist_preserved = [bool(stat["histogram_preserved"]) for stat in stats]
    if any_changed and not all(hist_preserved):
        return _analysis(
            task,
            "conditional_recolor",
            "probable",
            "conditional_in_place_change",
            "Dimensions are preserved and colors change, but no single global color map explains every pair.",
            {"histogram_preserved_pairs": int(sum(hist_preserved)), "pair_count": len(examples)},
        )
    if any_changed and all(hist_preserved):
        return _analysis(
            task,
            "rearrangement",
            "probable",
            "histogram_preserving_rearrangement",
            "Dimensions and color counts are preserved while cell positions change.",
            {"histogram_preserved_pairs": len(examples), "pair_count": len(examples)},
        )
    if any_changed:
        return _analysis(
            task,
            "structural_other",
            "probable",
            "unresolved_structure",
            "The task changes structure but matches no tested exact operation.",
            {"pair_count": len(examples)},
        )
    return _analysis(
        task,
        "unknown",
        "unknown",
        "unknown",
        "No tested operation produced stable evidence.",
        {"pair_count": len(examples)},
    )


def analysis_to_dict(result: TaskAnalysis, svg_path: str | None = None) -> dict[str, Any]:
    return {
        "ordinal": result.task.ordinal,
        "identifier_index": result.task.identifier_index,
        "identifier": result.task.identifier,
        "example_count": len(result.task.examples),
        "family": result.family,
        "confidence": result.confidence,
        "operation": result.operation,
        "description": result.description,
        "evidence": result.evidence,
        "examples": list(result.example_statistics),
        "capability": result.capability,
        "svg_path": svg_path,
    }
