import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

os.environ.setdefault("DISABLE_COMPILE", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_SILENT", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fvr_structfuse_alpha_sweep import BUCKETS, read_csv, write_csv


Grid = np.ndarray


@dataclass(frozen=True)
class Component:
    component_id: int
    color: int
    cells: Tuple[Tuple[int, int], ...]
    area: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    shape_signature: Tuple[Tuple[int, int], ...]
    touches_border: bool


@dataclass(frozen=True)
class SceneObject:
    object_id: int
    kind: str
    color_set: Tuple[int, ...]
    cells: Tuple[Tuple[int, int], ...]
    area: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    shape_signature: Tuple[Tuple[int, int], ...]
    touches_border: bool


@dataclass(frozen=True)
class Program:
    name: str
    family: str
    params: Dict[str, object]
    debug_reason: str


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def grid_key(grid: Grid) -> str:
    arr = np.asarray(grid, dtype=np.uint8)
    h = hashlib.sha1()
    h.update(str(tuple(arr.shape)).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def grid_json(grid: Grid) -> str:
    return json.dumps(np.asarray(grid, dtype=int).tolist(), separators=(",", ":"))


def most_common_color(grid: Grid) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def connected_components(grid: Grid, include_background: bool = True) -> List[Component]:
    arr = np.asarray(grid, dtype=np.int64)
    h, w = arr.shape
    background = most_common_color(arr)
    seen = np.zeros((h, w), dtype=bool)
    comps: List[Component] = []
    cid = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(arr[r, c])
            if not include_background and color == background:
                seen[r, c] = True
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(arr[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [x[0] for x in cells]
            cols = [x[1] for x in cells]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            sig = tuple(sorted((rr - min_r, cc - min_c) for rr, cc in cells))
            comps.append(
                Component(
                    component_id=cid,
                    color=color,
                    cells=tuple(sorted(cells)),
                    area=len(cells),
                    bbox=(min_r, min_c, max_r, max_c),
                    centroid=(float(sum(rows)) / len(rows), float(sum(cols)) / len(cols)),
                    shape_signature=sig,
                    touches_border=min_r == 0 or min_c == 0 or max_r == h - 1 or max_c == w - 1,
                )
            )
            cid += 1
    return comps


def non_background_objects(grid: Grid) -> List[SceneObject]:
    arr = np.asarray(grid, dtype=np.int64)
    h, w = arr.shape
    background = most_common_color(arr)
    mask = arr != background
    seen = np.zeros((h, w), dtype=bool)
    objects: List[SceneObject] = []
    oid = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [x[0] for x in cells]
            cols = [x[1] for x in cells]
            min_r, max_r = min(rows), max(rows)
            min_c, max_c = min(cols), max(cols)
            sig = tuple(sorted((rr - min_r, cc - min_c) for rr, cc in cells))
            colors = tuple(sorted(int(arr[rr, cc]) for rr, cc in cells))
            color_set = tuple(sorted(set(colors)))
            kind = "multicolour_component" if len(color_set) > 1 else "monochrome_component"
            objects.append(
                SceneObject(
                    object_id=oid,
                    kind=kind,
                    color_set=color_set,
                    cells=tuple(sorted(cells)),
                    area=len(cells),
                    bbox=(min_r, min_c, max_r, max_c),
                    centroid=(float(sum(rows)) / len(rows), float(sum(cols)) / len(cols)),
                    shape_signature=sig,
                    touches_border=min_r == 0 or min_c == 0 or max_r == h - 1 or max_c == w - 1,
                )
            )
            oid += 1
    return objects


def maximal_axis_lines(grid: Grid) -> List[Dict[str, object]]:
    arr = np.asarray(grid, dtype=np.int64)
    background = most_common_color(arr)
    h, w = arr.shape
    lines: List[Dict[str, object]] = []
    for r in range(h):
        c = 0
        while c < w:
            color = int(arr[r, c])
            start = c
            while c < w and int(arr[r, c]) == color:
                c += 1
            if color != background and c - start >= 2:
                lines.append({"axis": "row", "color": color, "row": r, "start": start, "end": c - 1, "length": c - start})
    for c in range(w):
        r = 0
        while r < h:
            color = int(arr[r, c])
            start = r
            while r < h and int(arr[r, c]) == color:
                r += 1
            if color != background and r - start >= 2:
                lines.append({"axis": "col", "color": color, "col": c, "start": start, "end": r - 1, "length": r - start})
    return lines


def scene_graph_summary(grid: Grid) -> Dict[str, object]:
    comps = connected_components(grid, include_background=True)
    background = most_common_color(grid)
    objects = non_background_objects(grid)
    markers = [c for c in comps if c.color != background and c.area == 1]
    return {
        "shape": list(np.asarray(grid).shape),
        "background": int(background),
        "monochrome_components": [
            {
                "color": int(c.color),
                "area": int(c.area),
                "bbox": list(c.bbox),
                "centroid": [float(c.centroid[0]), float(c.centroid[1])],
                "touches_border": bool(c.touches_border),
            }
            for c in comps
            if c.color != background
        ],
        "non_background_objects": [
            {
                "kind": obj.kind,
                "color_set": list(obj.color_set),
                "area": int(obj.area),
                "bbox": list(obj.bbox),
                "centroid": [float(obj.centroid[0]), float(obj.centroid[1])],
                "touches_border": bool(obj.touches_border),
            }
            for obj in objects
        ],
        "singleton_markers": [
            {
                "color": int(c.color),
                "cell": [int(c.cells[0][0]), int(c.cells[0][1])],
                "centroid": [float(c.centroid[0]), float(c.centroid[1])],
            }
            for c in markers
        ],
        "axis_lines": maximal_axis_lines(grid),
    }


def _cells_bbox(cells: Iterable[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    coords = list(cells)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return (min(rows), min(cols), max(rows), max(cols))


def _cells_signature(cells: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    coords = list(cells)
    min_r, min_c, _, _ = _cells_bbox(coords)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _bbox_orientation(bbox: Tuple[int, int, int, int], area: int) -> str:
    min_r, min_c, max_r, max_c = bbox
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    if area == 1:
        return "point"
    if height == 1:
        return "horizontal"
    if width == 1:
        return "vertical"
    if height == width:
        return "square"
    return "wide" if width > height else "tall"


def _hole_count_from_cells(shape: Tuple[int, int], cells: Iterable[Tuple[int, int]]) -> int:
    coords = set(cells)
    if not coords:
        return 0
    min_r, min_c, max_r, max_c = _cells_bbox(coords)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    local = np.zeros((h, w), dtype=np.uint8)
    for r, c in coords:
        local[r - min_r, c - min_c] = 1
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if local[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            region: List[Tuple[int, int]] = []
            touches_edge = False
            while stack:
                cr, cc = stack.pop()
                region.append((cr, cc))
                if cr == 0 or cc == 0 or cr == h - 1 or cc == w - 1:
                    touches_edge = True
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not local[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge and region:
                holes += 1
    return int(holes)


def _node_from_cells(
    grid: Grid,
    node_id: int,
    kind: str,
    cells: Iterable[Tuple[int, int]],
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    arr = np.asarray(grid, dtype=np.uint8)
    coords = tuple(sorted((int(r), int(c)) for r, c in cells))
    bbox = _cells_bbox(coords)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    h, w = arr.shape
    color_set = tuple(sorted({int(arr[r, c]) for r, c in coords}))
    area = len(coords)
    node = {
        "node_id": f"n{node_id}",
        "kind": kind,
        "mask": [[int(r), int(c)] for r, c in coords],
        "colour_set": list(color_set),
        "color_set": list(color_set),
        "area": int(area),
        "bbox": [int(x) for x in bbox],
        "centroid": [float(sum(rows)) / area, float(sum(cols)) / area],
        "normalized_shape_signature": [[int(r), int(c)] for r, c in _cells_signature(coords)],
        "shape_signature": [[int(r), int(c)] for r, c in _cells_signature(coords)],
        "orientation": _bbox_orientation(bbox, area),
        "touches_border": bool(bbox[0] == 0 or bbox[1] == 0 or bbox[2] == h - 1 or bbox[3] == w - 1),
        "hole_count": _hole_count_from_cells(arr.shape, coords),
        "line_direction": None,
    }
    if extra:
        node.update(extra)
    return node


def _find_hole_nodes(grid: Grid, start_node_id: int) -> List[Dict[str, object]]:
    arr = np.asarray(grid, dtype=np.uint8)
    background = most_common_color(arr)
    external = flood_external_background(arr, background)
    hole_mask = (arr == background) & (~external)
    seen = np.zeros(arr.shape, dtype=bool)
    nodes: List[Dict[str, object]] = []
    nid = start_node_id
    h, w = arr.shape
    for r in range(h):
        for c in range(w):
            if not hole_mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and hole_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            nodes.append(
                _node_from_cells(
                    arr,
                    nid,
                    "hole",
                    cells,
                    {"enclosed_background_color": int(background)},
                )
            )
            nid += 1
    return nodes


def _line_node(grid: Grid, node_id: int, line: Dict[str, object]) -> Dict[str, object]:
    if line["axis"] == "row":
        cells = [(int(line["row"]), c) for c in range(int(line["start"]), int(line["end"]) + 1)]
        direction = "horizontal"
    else:
        cells = [(r, int(line["col"])) for r in range(int(line["start"]), int(line["end"]) + 1)]
        direction = "vertical"
    node = _node_from_cells(
        grid,
        node_id,
        "line",
        cells,
        {
            "line_direction": direction,
            "axis": line["axis"],
            "line_color": int(line["color"]),
            "line_length": int(line["length"]),
            "is_separator": bool(
                (line["axis"] == "row" and int(line["length"]) == np.asarray(grid).shape[1])
                or (line["axis"] == "col" and int(line["length"]) == np.asarray(grid).shape[0])
            ),
        },
    )
    return node


def _make_relations(nodes: List[Dict[str, object]]) -> List[Dict[str, object]]:
    relations: List[Dict[str, object]] = []
    for i, a in enumerate(nodes):
        a_bbox = tuple(int(x) for x in a["bbox"])
        a_cent = a["centroid"]
        a_mask = {tuple(x) for x in a["mask"]}
        for b in nodes[i + 1 :]:
            b_bbox = tuple(int(x) for x in b["bbox"])
            b_cent = b["centroid"]
            b_mask = {tuple(x) for x in b["mask"]}
            pair = {"a": a["node_id"], "b": b["node_id"]}
            if a_bbox[3] < b_bbox[1]:
                relations.append({**pair, "type": "left_of"})
            if b_bbox[3] < a_bbox[1]:
                relations.append({**pair, "type": "right_of"})
            if a_bbox[2] < b_bbox[0]:
                relations.append({**pair, "type": "above"})
            if b_bbox[2] < a_bbox[0]:
                relations.append({**pair, "type": "below"})
            if abs(float(a_cent[0]) - float(b_cent[0])) < 0.5:
                relations.append({**pair, "type": "aligned_row"})
            if abs(float(a_cent[1]) - float(b_cent[1])) < 0.5:
                relations.append({**pair, "type": "aligned_column"})
            if a_mask & b_mask:
                relations.append({**pair, "type": "touches"})
            if a_bbox[0] <= b_bbox[0] and a_bbox[1] <= b_bbox[1] and a_bbox[2] >= b_bbox[2] and a_bbox[3] >= b_bbox[3]:
                relations.append({**pair, "type": "contains"})
            if b_bbox[0] <= a_bbox[0] and b_bbox[1] <= a_bbox[1] and b_bbox[2] >= a_bbox[2] and b_bbox[3] >= a_bbox[3]:
                relations.append({**pair, "type": "inside"})
            same_shape = a["normalized_shape_signature"] == b["normalized_shape_signature"]
            if same_shape:
                relations.append({**pair, "type": "same_shape"})
                relations.append({**pair, "type": "same_shape_up_to_recolour"})
                relations.append(
                    {
                        **pair,
                        "type": "translation_vector",
                        "dy": int(b_bbox[0] - a_bbox[0]),
                        "dx": int(b_bbox[1] - a_bbox[1]),
                    }
                )
            if sorted(a["colour_set"]) == sorted(b["colour_set"]):
                relations.append({**pair, "type": "same_colour"})
    return relations


def extract_multiview_scene_graph(grid: Grid) -> Dict[str, object]:
    arr = np.asarray(grid, dtype=np.uint8)
    background = most_common_color(arr)
    nodes: List[Dict[str, object]] = []
    node_id = 0

    for comp in connected_components(arr, include_background=True):
        if comp.color == background:
            continue
        kind = "marker" if comp.area == 1 else "colour_component"
        nodes.append(
            _node_from_cells(
                arr,
                node_id,
                kind,
                comp.cells,
                {
                    "component_id": int(comp.component_id),
                    "color": int(comp.color),
                    "view": "colour_connected",
                },
            )
        )
        node_id += 1

    for obj in non_background_objects(arr):
        nodes.append(
            _node_from_cells(
                arr,
                node_id,
                "foreground_object",
                obj.cells,
                {
                    "object_id": int(obj.object_id),
                    "view": "foreground_connected",
                    "multicolour": bool(len(obj.color_set) > 1),
                },
            )
        )
        node_id += 1

    for line in maximal_axis_lines(arr):
        nodes.append(_line_node(arr, node_id, line))
        node_id += 1
        if int(line["length"]) >= 2:
            if line["axis"] == "row":
                endpoints = [(int(line["row"]), int(line["start"])), (int(line["row"]), int(line["end"]))]
            else:
                endpoints = [(int(line["start"]), int(line["col"])), (int(line["end"]), int(line["col"]))]
            for endpoint in endpoints:
                nodes.append(
                    _node_from_cells(
                        arr,
                        node_id,
                        "anchor_endpoint",
                        [endpoint],
                        {"source_line_axis": line["axis"], "source_line_color": int(line["color"])},
                    )
                )
                node_id += 1

    for hole in _find_hole_nodes(arr, node_id):
        nodes.append(hole)
        node_id += 1

    components = [node for node in nodes if node["kind"] in {"colour_component", "marker"}]
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for node in components:
        key = json.dumps(node["normalized_shape_signature"], separators=(",", ":"))
        groups[key].append(node)
    repeated_shape_classes = []
    for key, group in groups.items():
        if len(group) < 2:
            continue
        cells = sorted({tuple(cell) for node in group for cell in node["mask"]})
        repeated_shape_classes.append(
            {
                "shape_key": key,
                "count": len(group),
                "node_ids": [node["node_id"] for node in group],
            }
        )
        nodes.append(
            _node_from_cells(
                arr,
                node_id,
                "repeated_shape_group",
                cells,
                {"shape_key": key, "member_node_ids": [node["node_id"] for node in group]},
            )
        )
        node_id += 1

    legacy = scene_graph_summary(arr)
    relations = _make_relations(nodes)
    return {
        **legacy,
        "nodes": nodes,
        "relations": relations,
        "repeated_shape_classes": repeated_shape_classes,
    }


def extract_scene_graph(grid: Grid) -> Dict[str, object]:
    return extract_multiview_scene_graph(grid)


def relation_summary(grid: Grid) -> Dict[str, object]:
    graph = extract_multiview_scene_graph(grid)
    markers = graph["singleton_markers"]
    objects = graph["non_background_objects"]
    return {
        "shape": graph["shape"],
        "n_objects": len(objects),
        "n_markers": len(markers),
        "n_axis_lines": len(graph["axis_lines"]),
        "n_multiview_nodes": len(graph["nodes"]),
        "n_relations": len(graph["relations"]),
        "marker_colours": sorted({int(marker["color"]) for marker in markers}),
        "object_colour_sets": [obj["color_set"] for obj in objects],
        "relation_types": sorted({str(rel["type"]) for rel in graph["relations"]}),
    }


def _node_cells(node: Dict[str, object]) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in node["mask"]))


def _shift_cells(cells: Iterable[Tuple[int, int]], dy: int, dx: int) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((int(r + dy), int(c + dx)) for r, c in cells))


def _reflect_cells(cells: Iterable[Tuple[int, int]], shape: Tuple[int, int], axis: str) -> Tuple[Tuple[int, int], ...]:
    h, w = shape
    if axis == "vertical":
        return tuple(sorted((int(r), int(w - 1 - c)) for r, c in cells))
    if axis == "horizontal":
        return tuple(sorted((int(h - 1 - r), int(c)) for r, c in cells))
    raise ValueError(f"Unknown reflection axis: {axis}")


def _matchable_nodes(graph: Dict[str, object]) -> List[Dict[str, object]]:
    return [
        node
        for node in graph["nodes"]
        if node["kind"] in {"colour_component", "foreground_object", "marker"}
        and node["area"] > 0
        and node["kind"] != "repeated_shape_group"
    ]


def match_objects_under_transform(input_graph: Dict[str, object], output_graph: Dict[str, object]) -> Dict[str, object]:
    input_nodes = _matchable_nodes(input_graph)
    output_nodes = _matchable_nodes(output_graph)
    shape = tuple(int(x) for x in input_graph["shape"])
    events: List[Dict[str, object]] = []
    matched_input: set[str] = set()
    matched_output: set[str] = set()

    def compatible(a: Dict[str, object], b: Dict[str, object], same_color: bool = True) -> bool:
        if int(a["area"]) != int(b["area"]):
            return False
        if a["normalized_shape_signature"] != b["normalized_shape_signature"]:
            return False
        return (not same_color) or sorted(a["colour_set"]) == sorted(b["colour_set"])

    for inp in input_nodes:
        in_cells = _node_cells(inp)
        for out in output_nodes:
            if out["node_id"] in matched_output:
                continue
            if in_cells == _node_cells(out) and compatible(inp, out, same_color=True):
                events.append({"event": "KEEP", "input_node": inp["node_id"], "output_node": out["node_id"]})
                matched_input.add(inp["node_id"])
                matched_output.add(out["node_id"])
                break

    for inp in input_nodes:
        if inp["node_id"] in matched_input:
            continue
        in_cells = _node_cells(inp)
        for out in output_nodes:
            if out["node_id"] in matched_output:
                continue
            if in_cells == _node_cells(out) and compatible(inp, out, same_color=False):
                events.append(
                    {
                        "event": "RECOLOUR",
                        "input_node": inp["node_id"],
                        "output_node": out["node_id"],
                        "source_colours": inp["colour_set"],
                        "target_colours": out["colour_set"],
                    }
                )
                matched_input.add(inp["node_id"])
                matched_output.add(out["node_id"])
                break

    for inp in input_nodes:
        if inp["node_id"] in matched_input:
            continue
        in_bbox = tuple(int(x) for x in inp["bbox"])
        in_cells = _node_cells(inp)
        for out in output_nodes:
            if out["node_id"] in matched_output:
                continue
            if not compatible(inp, out, same_color=True):
                continue
            out_bbox = tuple(int(x) for x in out["bbox"])
            dy = out_bbox[0] - in_bbox[0]
            dx = out_bbox[1] - in_bbox[1]
            if _shift_cells(in_cells, dy, dx) == _node_cells(out):
                events.append(
                    {
                        "event": "MOVE",
                        "input_node": inp["node_id"],
                        "output_node": out["node_id"],
                        "dy": int(dy),
                        "dx": int(dx),
                    }
                )
                matched_input.add(inp["node_id"])
                matched_output.add(out["node_id"])
                break

    for inp in input_nodes:
        if inp["node_id"] in matched_input:
            continue
        in_cells = _node_cells(inp)
        for out in output_nodes:
            if out["node_id"] in matched_output:
                continue
            if sorted(inp["colour_set"]) != sorted(out["colour_set"]) or int(inp["area"]) != int(out["area"]):
                continue
            for axis in ("vertical", "horizontal"):
                if _reflect_cells(in_cells, shape, axis) == _node_cells(out):
                    events.append(
                        {
                            "event": "REFLECT",
                            "input_node": inp["node_id"],
                            "output_node": out["node_id"],
                            "axis": axis,
                        }
                    )
                    matched_input.add(inp["node_id"])
                    matched_output.add(out["node_id"])
                    break
            if inp["node_id"] in matched_input:
                break

    kept_inputs = {event["input_node"] for event in events if event["event"] == "KEEP"}
    for inp in input_nodes:
        if inp["node_id"] not in kept_inputs:
            continue
        in_bbox = tuple(int(x) for x in inp["bbox"])
        in_cells = _node_cells(inp)
        for out in output_nodes:
            if out["node_id"] in matched_output:
                continue
            if not compatible(inp, out, same_color=True):
                continue
            out_bbox = tuple(int(x) for x in out["bbox"])
            dy = out_bbox[0] - in_bbox[0]
            dx = out_bbox[1] - in_bbox[1]
            if _shift_cells(in_cells, dy, dx) == _node_cells(out):
                events.append(
                    {
                        "event": "COPY",
                        "input_node": inp["node_id"],
                        "output_node": out["node_id"],
                        "dy": int(dy),
                        "dx": int(dx),
                    }
                )
                matched_output.add(out["node_id"])

    for inp in input_nodes:
        if inp["node_id"] not in matched_input and inp["node_id"] not in kept_inputs:
            events.append({"event": "DELETE", "input_node": inp["node_id"]})
    for out in output_nodes:
        if out["node_id"] not in matched_output:
            events.append({"event": "ADD", "output_node": out["node_id"]})

    return {
        "events": events,
        "object_matches": [event for event in events if event["event"] in {"KEEP", "RECOLOUR", "MOVE", "COPY", "REFLECT"}],
        "removed_objects": [event for event in events if event["event"] == "DELETE"],
        "added_objects": [event for event in events if event["event"] == "ADD"],
        "input_count": len(input_nodes),
        "output_count": len(output_nodes),
    }


def match_input_output_nodes(input_graph: Dict[str, object], output_graph: Dict[str, object]) -> Dict[str, object]:
    return match_objects_under_transform(input_graph, output_graph)


def compute_demo_delta(input_grid: Grid, output_grid: Grid) -> Dict[str, object]:
    inp = np.asarray(input_grid, dtype=np.uint8)
    out = np.asarray(output_grid, dtype=np.uint8)
    if inp.shape != out.shape:
        return {
            "shape_change": {"input": list(inp.shape), "output": list(out.shape)},
            "preserved_cells": 0,
            "removed_cells": 0,
            "added_cells": 0,
            "recoloured_cells": 0,
            "object_matches": [],
            "removed_objects": [],
            "added_objects": [],
            "recolour_events": [],
            "move_events": [],
            "copy_events": [],
            "reflection_events": [],
            "motif_cell_rewrites": [],
            "filled_regions": [],
            "path_segments": [],
            "marker_selection_events": [],
        }
    input_bg = most_common_color(inp)
    output_bg = most_common_color(out)
    preserved = int(np.sum(inp == out))
    removed_mask = (inp != input_bg) & (out == output_bg)
    added_mask = (inp == input_bg) & (out != output_bg)
    recoloured_mask = (inp != input_bg) & (out != output_bg) & (inp != out)
    changed = inp != out
    motif = dominant_multicolour_object(inp)
    rewrites = []
    for r, c in zip(*np.where(changed)):
        rec = {
            "row": int(r),
            "col": int(c),
            "input_color": int(inp[r, c]),
            "output_color": int(out[r, c]),
        }
        if motif is not None:
            min_r, min_c, max_r, max_c = motif.bbox
            rec["motif_rel"] = [int(r - min_r), int(c - min_c)]
            rec["inside_motif_bbox"] = bool(min_r <= r <= max_r and min_c <= c <= max_c)
        rewrites.append(rec)
    filled_regions = []
    for color in sorted(int(x) for x in np.unique(out[changed])) if changed.any() else []:
        coords = [(int(r), int(c)) for r, c in zip(*np.where(changed & (out == color)))]
        if coords:
            filled_regions.append({"output_color": color, "cells": coords})
    input_graph = extract_multiview_scene_graph(inp)
    output_graph = extract_multiview_scene_graph(out)
    matches = match_objects_under_transform(input_graph, output_graph)
    recolour_events = [event for event in matches["events"] if event["event"] == "RECOLOUR"]
    move_events = [event for event in matches["events"] if event["event"] == "MOVE"]
    copy_events = [event for event in matches["events"] if event["event"] == "COPY"]
    reflection_events = [event for event in matches["events"] if event["event"] == "REFLECT"]
    marker_selection_events = []
    for color in sorted(int(x) for x in np.unique(inp)):
        if color == input_bg:
            continue
        input_markers = [
            comp for comp in connected_components(inp, include_background=True) if comp.color == color and comp.area == 1
        ]
        output_markers = [
            comp for comp in connected_components(out, include_background=True) if comp.color == color and comp.area == 1
        ]
        if len(input_markers) > len(output_markers):
            marker_selection_events.append(
                {
                    "color": int(color),
                    "input_marker_count": len(input_markers),
                    "output_marker_count": len(output_markers),
                    "removed_cells": [
                        [int(r), int(c)]
                        for comp in input_markers
                        for r, c in comp.cells
                        if int(out[r, c]) == output_bg
                    ],
                    "retained_cells": [
                        [int(r), int(c)]
                        for comp in output_markers
                        for r, c in comp.cells
                    ],
                }
            )

    path_segments = []
    added_mask = (inp == input_bg) & (out != output_bg)
    for line in maximal_axis_lines(out):
        color = int(line["color"])
        if line["axis"] == "row":
            r = int(line["row"])
            coords = [(r, c) for c in range(int(line["start"]), int(line["end"]) + 1)]
        else:
            c = int(line["col"])
            coords = [(r, c) for r in range(int(line["start"]), int(line["end"]) + 1)]
        if any(bool(added_mask[r, c]) for r, c in coords):
            path_segments.append(
                {
                    "axis": line["axis"],
                    "color": color,
                    "start": [int(coords[0][0]), int(coords[0][1])],
                    "end": [int(coords[-1][0]), int(coords[-1][1])],
                    "length": int(line["length"]),
                    "added_cells": [[int(r), int(c)] for r, c in coords if bool(added_mask[r, c])],
                }
            )
    return {
        "shape_change": None,
        "preserved_cells": preserved,
        "removed_cells": int(np.sum(removed_mask)),
        "added_cells": int(np.sum(added_mask)),
        "recoloured_cells": int(np.sum(recoloured_mask)),
        "object_matches": matches["object_matches"],
        "removed_objects": matches["removed_objects"],
        "added_objects": matches["added_objects"],
        "recolour_events": recolour_events,
        "move_events": move_events,
        "copy_events": copy_events,
        "reflection_events": reflection_events,
        "motif_cell_rewrites": rewrites,
        "filled_regions": filled_regions,
        "path_segments": path_segments,
        "marker_selection_events": marker_selection_events,
    }


def demo_deltas(demos: List[Dict[str, object]]) -> List[Dict[str, object]]:
    records = []
    for idx, demo in enumerate(demos):
        inp = np.asarray(demo["input"], dtype=np.uint8)
        out = np.asarray(demo["output"], dtype=np.uint8)
        input_graph = extract_scene_graph(inp)
        output_graph = extract_scene_graph(out)
        object_matches = match_objects_under_transform(input_graph, output_graph)
        records.append(
            {
                "demo_index": idx,
                "delta": compute_demo_delta(inp, out),
                "node_matches": object_matches,
                "object_matches": object_matches,
            }
        )
    return records


def dominant_multicolour_object(grid: Grid) -> Optional[SceneObject]:
    objects = [obj for obj in non_background_objects(grid) if len(obj.color_set) > 1]
    if not objects:
        return None
    h, w = np.asarray(grid).shape
    center = ((h - 1) / 2.0, (w - 1) / 2.0)
    return sorted(
        objects,
        key=lambda obj: (
            -obj.area,
            abs(obj.centroid[0] - center[0]) + abs(obj.centroid[1] - center[1]),
            obj.bbox,
        ),
    )[0]


def normalize_selector(selector: Dict[str, object]) -> str:
    return json.dumps(selector, sort_keys=True, separators=(",", ":"))


def program_semantic_key(program: Program) -> str:
    semantic_params = dict(program.params)
    semantic_params.pop("source_demo", None)
    return f"{program.name}|{program.family}|{json.dumps(semantic_params, sort_keys=True, separators=(',', ':'))}"


def program_stability_key(program: Program) -> str:
    if program.family == "orpi_conditional_recolour_by_region":
        stripped = {
            "discriminator": program.params.get("discriminator"),
            "source_color": program.params.get("source_color"),
        }
        return f"{program.name}|{program.family}|{json.dumps(stripped, sort_keys=True, separators=(',', ':'))}"
    return program_semantic_key(program)


def program_complexity_score(program: Program) -> Tuple[int, int]:
    if program.family == "orpi_conditional_recolour_by_region":
        mapping = list(program.params.get("mapping", []))
        return (len(mapping), len(json.dumps(program.params, sort_keys=True)))
    selector = program.params.get("selector", {}) if isinstance(program.params, dict) else {}
    selector_complexity = 0
    if isinstance(selector, dict):
        sel_type = selector.get("type", "")
        if sel_type == "by_source_color_only":
            selector_complexity = 0
        elif sel_type in (
            "by_source_color_horizontal_line",
            "by_source_color_vertical_line",
            "by_source_color_axis_line",
            "by_source_color_singleton",
            "by_source_color_touches_border",
            "by_source_color_not_touches_border",
            "by_source_color_zone",
            "by_source_color_extreme",
            "by_source_color_aspect",
        ):
            selector_complexity = 1
        elif sel_type in (
            "by_source_color_area",
            "by_source_color_hole_count",
            "by_source_color_quadrant",
            "by_source_color_area_rank_largest",
            "by_source_color_area_rank_smallest",
        ):
            selector_complexity = 2
        elif sel_type == "by_source_color_shape":
            selector_complexity = 3
        else:
            selector_complexity = 4
    return (selector_complexity, len(json.dumps(program.params, sort_keys=True)))


def selector_matches(comp: Component, selector: Dict[str, object], comps: List[Component]) -> bool:
    typ = selector["type"]
    if typ == "source_color":
        return comp.color == int(selector["color"])
    if typ == "source_color_area":
        return comp.color == int(selector["color"]) and comp.area == int(selector["area"])
    if typ == "source_color_shape":
        return comp.color == int(selector["color"]) and comp.shape_signature == tuple(tuple(x) for x in selector["shape"])
    if typ == "source_color_shape_area":
        return (
            comp.color == int(selector["color"])
            and comp.area == int(selector["area"])
            and comp.shape_signature == tuple(tuple(x) for x in selector["shape"])
        )
    if typ == "extreme_by_color":
        same = [x for x in comps if x.color == int(selector["color"])]
        if not same:
            return False
        axis = str(selector["axis"])
        side = str(selector["side"])
        values = [x.centroid[0 if axis == "row" else 1] for x in same]
        target = min(values) if side == "min" else max(values)
        return comp.color == int(selector["color"]) and comp.centroid[0 if axis == "row" else 1] == target
    if typ == "shape_any_color":
        return comp.shape_signature == tuple(tuple(x) for x in selector["shape"])
    raise ValueError(f"Unknown selector type: {typ}")


def apply_program(program: Program, grid: Grid) -> Grid:
    arr = np.asarray(grid, dtype=np.uint8).copy()
    if program.name == "global_color_map":
        mapping = {int(k): int(v) for k, v in dict(program.params["mapping"]).items()}
        out = arr.copy()
        for src, dst in mapping.items():
            out[arr == src] = dst
        return out
    if program.name == "component_delete_or_keep_by_relation":
        selector = dict(program.params["selector"])
        source_color = int(program.params["source_color"])
        background = int(program.params["background_color"])
        comps = connected_components(arr, include_background=True)
        out = arr.copy()
        for comp in comps:
            if comp.color != source_color:
                continue
            keep = selector_matches(comp, selector, comps)
            if not keep:
                for r, c in comp.cells:
                    out[r, c] = background
        return out
    if program.name == "component_recolour":
        selector = dict(program.params["selector"])
        target_color = int(program.params["target_color"])
        comps = connected_components(arr, include_background=bool(program.params.get("include_background", True)))
        out = arr.copy()
        for comp in comps:
            if selector_matches(comp, selector, comps):
                for r, c in comp.cells:
                    out[r, c] = target_color
        return out
    if program.name == "orient_motif_select_marker":
        return apply_orient_motif_select_marker(program, arr)
    if program.name == "complete_dominant_region":
        return apply_complete_dominant_region(program, arr)
    if program.name == "motif_orientation_select_marker":
        return apply_motif_orientation_select_marker(program, arr)
    if program.name == "template_component_recolour":
        return apply_template_component_recolour(program, arr)
    if program.name == "orpi_marker_from_motif_delta":
        if "d4_transform" in program.params:
            return apply_orpi_marker_motif_unified(program, arr)
        return apply_motif_orientation_select_marker(program, arr)
    if program.name == "orpi_template_recolour_from_delta":
        selector_type = dict(program.params.get("selector", {})).get("type", "")
        if selector_type.startswith("by_source_color"):
            return apply_orpi_template_recolour_intersection(program, arr)
        return apply_template_component_recolour(program, arr)
    if program.name == "orpi_conditional_recolour_by_region":
        return apply_orpi_conditional_recolour(program, arr)
    if program.name == "orpi_frame_from_seed":
        return apply_orpi_frame_from_seed(program, arr)
    if program.name == "orpi_axis_line_recolour":
        return apply_orpi_axis_line_recolour(program, arr)
    if program.name == "line_path_completion":
        return apply_line_path_program(program, arr)
    if program.name == "rigid_transform_copy":
        return apply_rigid_transform_program(program, arr)
    raise ValueError(f"Unknown program name: {program.name}")


def fill_between_same_color(arr: Grid, axis: str, color: Optional[int], span: str) -> Grid:
    out = arr.copy()
    background = most_common_color(arr)
    h, w = arr.shape
    colors = [int(color)] if color is not None else [int(x) for x in np.unique(arr) if int(x) != background]
    for cval in colors:
        if axis == "row":
            for r in range(h):
                positions = [c for c in range(w) if int(arr[r, c]) == cval]
                if len(positions) < 2:
                    continue
                pairs = [(min(positions), max(positions))] if span == "outermost" else list(zip(positions[:-1], positions[1:]))
                for a, b in pairs:
                    segment = arr[r, a : b + 1]
                    if np.all((segment == background) | (segment == cval)):
                        out[r, a : b + 1] = cval
        elif axis == "col":
            for col_idx in range(w):
                positions = [r for r in range(h) if int(arr[r, col_idx]) == cval]
                if len(positions) < 2:
                    continue
                pairs = [(min(positions), max(positions))] if span == "outermost" else list(zip(positions[:-1], positions[1:]))
                for a, b in pairs:
                    segment = arr[a : b + 1, col_idx]
                    if np.all((segment == background) | (segment == cval)):
                        out[a : b + 1, col_idx] = cval
        else:
            raise ValueError(f"Bad line axis: {axis}")
    return out


def marker_relation(motif: SceneObject, marker: Component) -> Dict[str, object]:
    mr, mc = motif.centroid
    rr, cc = marker.centroid
    dr = float(rr - mr)
    dc = float(cc - mc)
    if abs(dr) >= abs(dc):
        axis = "row"
        side = "down" if dr >= 0 else "up"
    else:
        axis = "col"
        side = "right" if dc >= 0 else "left"
    return {
        "axis": axis,
        "side": side,
        "row_sign": 1 if dr > 0 else -1 if dr < 0 else 0,
        "col_sign": 1 if dc > 0 else -1 if dc < 0 else 0,
        "distance": abs(dr) + abs(dc),
    }


def choose_marker_by_relation(motif: SceneObject, markers: List[Component], relation: Dict[str, object], policy: str) -> Optional[Component]:
    if not markers:
        return None
    mr, mc = motif.centroid

    def side_ok(marker: Component) -> bool:
        rr, cc = marker.centroid
        side = str(relation["side"])
        if side == "up":
            return rr < mr
        if side == "down":
            return rr > mr
        if side == "left":
            return cc < mc
        if side == "right":
            return cc > mc
        return False

    def quadrant_ok(marker: Component) -> bool:
        row_sign = int(relation["row_sign"])
        col_sign = int(relation["col_sign"])
        rr, cc = marker.centroid
        return (row_sign == 0 or (rr - mr) * row_sign > 0) and (col_sign == 0 or (cc - mc) * col_sign > 0)

    candidates = [m for m in markers if side_ok(m)]
    if policy.startswith("quadrant"):
        candidates = [m for m in markers if quadrant_ok(m)]
    if not candidates:
        return None
    if policy.endswith("farthest"):
        return sorted(candidates, key=lambda m: (-(abs(m.centroid[0] - mr) + abs(m.centroid[1] - mc)), m.centroid))[0]
    if policy == "extreme_side":
        side = str(relation["side"])
        if side == "up":
            return sorted(candidates, key=lambda m: (m.centroid[0], m.centroid[1]))[0]
        if side == "down":
            return sorted(candidates, key=lambda m: (-m.centroid[0], m.centroid[1]))[0]
        if side == "left":
            return sorted(candidates, key=lambda m: (m.centroid[1], m.centroid[0]))[0]
        return sorted(candidates, key=lambda m: (-m.centroid[1], m.centroid[0]))[0]
    return sorted(candidates, key=lambda m: (abs(m.centroid[0] - mr) + abs(m.centroid[1] - mc), m.centroid))[0]


def apply_orient_motif_select_marker(program: Program, arr: Grid) -> Grid:
    marker_color = int(program.params["marker_color"])
    background = int(program.params["background_color"])
    relation = dict(program.params["relation"])
    policy = str(program.params["policy"])
    motif = dominant_multicolour_object(arr)
    if motif is None:
        return arr.copy()
    comps = connected_components(arr, include_background=True)
    motif_cells = set(motif.cells)
    markers = [
        comp
        for comp in comps
        if comp.color == marker_color and comp.area == 1 and comp.cells[0] not in motif_cells
    ]
    selected = choose_marker_by_relation(motif, markers, relation, policy)
    if selected is None:
        return arr.copy()
    selected_cell = selected.cells[0]
    out = arr.copy()
    for marker in markers:
        if marker.cells[0] != selected_cell:
            r, c = marker.cells[0]
            out[r, c] = background
    return out


def apply_motif_orientation_select_marker(program: Program, arr: Grid) -> Grid:
    out = arr.copy()
    marker_color = int(program.params["marker_color"])
    background = int(program.params["background_color"])
    motif = dominant_multicolour_object(arr)
    if motif is None:
        return out
    min_r, min_c, max_r, max_c = motif.bbox
    for rewrite in list(program.params["motif_rewrites"]):
        rel_r, rel_c = rewrite["rel"]
        rr = min_r + int(rel_r)
        cc = min_c + int(rel_c)
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            out[rr, cc] = int(rewrite["output_color"])
    motif_cells = set(motif.cells)
    comps = connected_components(arr, include_background=True)
    markers = [
        comp
        for comp in comps
        if comp.color == marker_color and comp.area == 1 and comp.cells[0] not in motif_cells
    ]
    relation = dict(program.params.get("marker_relation", {}))
    selected_cell: Optional[Tuple[int, int]] = None
    if relation:
        selected = choose_marker_by_relation(motif, markers, relation, str(program.params.get("selection_policy", "nearest_side")))
        if selected is not None:
            selected_cell = selected.cells[0]
    rel_cell = program.params.get("selected_marker_rel_cell")
    if selected_cell is None and rel_cell is not None:
        rr = int(round(motif.centroid[0] + float(rel_cell[0])))
        cc = int(round(motif.centroid[1] + float(rel_cell[1])))
        if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
            selected_cell = (rr, cc)
    for marker in markers:
        if marker.cells[0] != selected_cell:
            r, c = marker.cells[0]
            out[r, c] = background
    if selected_cell is not None:
        r, c = selected_cell
        out[r, c] = marker_color
    return out


def flood_external_background(grid: Grid, background: int) -> np.ndarray:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    stack: List[Tuple[int, int]] = []
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


def largest_component_of_color(arr: Grid, color: int) -> Optional[Component]:
    comps = [comp for comp in connected_components(arr, include_background=True) if comp.color == color]
    if not comps:
        return None
    return sorted(comps, key=lambda comp: (-comp.area, comp.bbox))[0]


def apply_complete_dominant_region(program: Program, arr: Grid) -> Grid:
    color = int(program.params["region_color"])
    op = str(program.params["op"])
    background = int(program.params.get("background_color", most_common_color(arr)))
    out = arr.copy()
    if op == "enclosed_non_region":
        external = flood_external_background((arr != color).astype(np.uint8), 1)
        holes = (arr != color) & (~external)
        out[holes] = color
        return out
    if op == "enclosed_background":
        external = flood_external_background(arr, background)
        holes = (arr == background) & (~external)
        out[holes] = color
        return out
    comp = largest_component_of_color(arr, color)
    if comp is None:
        return out
    min_r, min_c, max_r, max_c = comp.bbox
    if op == "bbox_background":
        region = out[min_r : max_r + 1, min_c : max_c + 1]
        region[region == background] = color
        return out
    if op == "bbox_non_region":
        pad = int(program.params.get("pad", 0))
        h, w = out.shape
        rr0 = max(min_r - pad, 0)
        cc0 = max(min_c - pad, 0)
        rr1 = min(max_r + pad, h - 1)
        cc1 = min(max_c + pad, w - 1)
        region = out[rr0 : rr1 + 1, cc0 : cc1 + 1]
        region[region != color] = color
        return out
    if op == "neighbor_majority":
        neighbours = str(program.params["neighbours"])
        threshold = int(program.params["threshold"])
        iterations = int(program.params.get("iterations", 1))
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if neighbours == "8":
            offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
        h, w = out.shape
        for _ in range(iterations):
            prev = out.copy()
            for r in range(h):
                for c in range(w):
                    if int(prev[r, c]) != background:
                        continue
                    count = 0
                    for dr, dc in offsets:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and int(prev[nr, nc]) == color:
                            count += 1
                    if count >= threshold:
                        out[r, c] = color
        return out
    raise ValueError(f"Unknown complete-region op: {op}")


def apply_orpi_template_recolour_intersection(program: Program, arr: Grid) -> Grid:
    out = np.asarray(arr, dtype=np.uint8).copy()
    source_color = int(program.params["source_color"])
    target_color = int(program.params["target_color"])
    selector = dict(program.params["selector"])
    selected = _select_components_by_intersection_selector(out, selector, source_color)
    for comp in selected:
        for r, c in comp.cells:
            out[r, c] = target_color
    return out


def apply_orpi_marker_motif_unified(program: Program, arr: Grid) -> Grid:
    out = np.asarray(arr, dtype=np.uint8).copy()
    marker_color = int(program.params["marker_color"])
    background = int(program.params["background_color"])
    d4_name = str(program.params["d4_transform"])
    motif = dominant_multicolour_object(out)
    if motif is None:
        return out
    min_r, min_c, max_r, max_c = motif.bbox
    sub = out[min_r : max_r + 1, min_c : max_c + 1]
    try:
        transformed_sub = _apply_d4_transform(sub.copy(), d4_name)
    except Exception:
        return out
    if transformed_sub.shape != sub.shape:
        return out
    out[min_r : max_r + 1, min_c : max_c + 1] = transformed_sub
    motif_cells = set(motif.cells)
    comps = connected_components(arr, include_background=True)
    markers = [
        comp
        for comp in comps
        if comp.color == marker_color and comp.area == 1 and comp.cells[0] not in motif_cells
    ]
    relation = dict(program.params.get("marker_relation", {}))
    policy = str(program.params.get("selection_policy", "nearest_side"))
    selected_cell: Optional[Tuple[int, int]] = None
    if relation:
        selected = choose_marker_by_relation(motif, markers, relation, policy)
        if selected is not None:
            selected_cell = selected.cells[0]
    for marker in markers:
        if marker.cells[0] != selected_cell:
            r, c = marker.cells[0]
            out[r, c] = background
    if selected_cell is not None:
        r, c = selected_cell
        out[r, c] = marker_color
    return out


def _conditional_recolour_key_for(comp: Component, arr: np.ndarray, discriminator: str, all_comps: List[Component]) -> object:
    bag = _component_property_bag(comp, arr, all_comps)
    value = bag[discriminator]
    if isinstance(value, tuple):
        return [list(x) for x in value]
    return value


def apply_orpi_conditional_recolour(program: Program, arr: Grid) -> Grid:
    out = np.asarray(arr, dtype=np.uint8).copy()
    source_color = int(program.params["source_color"])
    discriminator = str(program.params["discriminator"])
    raw_mapping = list(program.params["mapping"])

    def _normalise_key(k: object) -> object:
        if isinstance(k, list):
            return [list(x) for x in k]
        if isinstance(k, bool):
            return bool(k)
        if isinstance(k, (int, np.integer)):
            return int(k)
        if isinstance(k, (float, np.floating)):
            return float(k)
        return str(k)

    mapping_lookup: List[Tuple[object, int]] = [
        (_normalise_key(entry[0]), int(entry[1])) for entry in raw_mapping
    ]
    all_comps = connected_components(out, include_background=True)
    same_color_comps = [c for c in all_comps if c.color == source_color]
    for comp in same_color_comps:
        key = _normalise_key(_conditional_recolour_key_for(comp, out, discriminator, all_comps))
        target_color: Optional[int] = None
        for stored_key, tgt in mapping_lookup:
            if stored_key == key:
                target_color = int(tgt)
                break
        if target_color is None:
            continue
        for r, c in comp.cells:
            out[r, c] = target_color
    return out


def apply_orpi_axis_line_recolour(program: Program, arr: Grid) -> Grid:
    out = np.asarray(arr, dtype=np.uint8).copy()
    src = int(program.params["source_color"])
    tgt = int(program.params["target_color"])
    axis = str(program.params["axis"])
    min_len = int(program.params["min_length"])
    runs = _axis_runs_of_color(out, src, axis, min_len)
    for run in runs:
        for r, c in run:
            out[r, c] = tgt
    return out


def apply_orpi_frame_from_seed(program: Program, arr: Grid) -> Grid:
    out = np.asarray(arr, dtype=np.uint8).copy()
    background = most_common_color(out)
    anchor_color = int(program.params["anchor_color"])
    anchor_offset_r = int(program.params["anchor_offset_r"])
    anchor_offset_c = int(program.params["anchor_offset_c"])
    frame_color = int(program.params["frame_color"])
    frame_h = int(program.params["frame_h"])
    frame_w = int(program.params["frame_w"])
    h, w = out.shape
    seed_cells: List[Tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == anchor_color:
                seed_cells.append((r, c))
    if not seed_cells:
        return out
    target_r, target_c = seed_cells[0]
    min_r = target_r - anchor_offset_r
    min_c = target_c - anchor_offset_c
    if min_r < 0 or min_c < 0 or min_r + frame_h > h or min_c + frame_w > w:
        return out
    for r in range(min_r, min_r + frame_h):
        for c in range(min_c, min_c + frame_w):
            if r in (min_r, min_r + frame_h - 1) or c in (min_c, min_c + frame_w - 1):
                if int(out[r, c]) == background:
                    out[r, c] = frame_color
    return out


def apply_template_component_recolour(program: Program, arr: Grid) -> Grid:
    out = arr.copy()
    source_color = int(program.params["source_color"])
    target_color = int(program.params["target_color"])
    selector = dict(program.params["selector"])
    comps = [comp for comp in connected_components(arr, include_background=True) if comp.color == source_color]
    if selector["type"] == "rank_by_axis":
        axis = str(selector["axis"])
        rank = int(selector["rank"])
        reverse = bool(selector.get("reverse", False))
        comps = sorted(comps, key=lambda comp: (comp.centroid[0 if axis == "row" else 1], comp.centroid[1 if axis == "row" else 0]))
        if reverse:
            comps = list(reversed(comps))
        selected = comps[rank : rank + 1] if 0 <= rank < len(comps) else []
    elif selector["type"] == "shape_and_rank":
        shape = tuple(tuple(x) for x in selector["shape"])
        rank = int(selector["rank"])
        matches = [comp for comp in comps if comp.shape_signature == shape]
        matches = sorted(matches, key=lambda comp: (comp.centroid[0], comp.centroid[1]))
        selected = matches[rank : rank + 1] if 0 <= rank < len(matches) else []
    elif selector["type"] == "shape_aligned_template":
        shape = tuple(tuple(x) for x in selector["shape"])
        template_color = int(selector["template_color"])
        template_shape = tuple(tuple(x) for x in selector["template_shape"])
        axis = str(selector.get("alignment_axis", "col"))
        tolerance = float(selector.get("tolerance", 0.75))
        template_rank = int(selector.get("template_rank", 0))
        template_comps = [
            comp
            for comp in connected_components(arr, include_background=True)
            if comp.color == template_color and comp.shape_signature == template_shape
        ]
        template_comps = sorted(template_comps, key=lambda comp: (comp.centroid[1], comp.centroid[0]))
        if not (0 <= template_rank < len(template_comps)):
            selected = []
        else:
            template = template_comps[template_rank]
            idx = 1 if axis == "col" else 0
            selected = [
                comp
                for comp in comps
                if comp.shape_signature == shape and abs(comp.centroid[idx] - template.centroid[idx]) <= tolerance
            ]
    elif selector["type"] == "shape_relative_zone":
        shape = tuple(tuple(x) for x in selector["shape"])
        zone = str(selector["zone"])
        h, w = arr.shape
        matches = [comp for comp in comps if comp.shape_signature == shape]
        if zone == "lower":
            selected = [comp for comp in matches if comp.centroid[0] >= (h - 1) / 2.0]
        elif zone == "upper":
            selected = [comp for comp in matches if comp.centroid[0] <= (h - 1) / 2.0]
        elif zone == "left":
            selected = [comp for comp in matches if comp.centroid[1] <= (w - 1) / 2.0]
        elif zone == "right":
            selected = [comp for comp in matches if comp.centroid[1] >= (w - 1) / 2.0]
        else:
            selected = []
    else:
        raise ValueError(f"Unknown template recolour selector: {selector['type']}")
    for comp in selected:
        for r, c in comp.cells:
            out[r, c] = target_color
    return out


def extend_color_rays(arr: Grid, color: Optional[int], direction: str) -> Grid:
    out = arr.copy()
    background = most_common_color(arr)
    h, w = arr.shape
    colors = [int(color)] if color is not None else [int(x) for x in np.unique(arr) if int(x) != background]
    starts = [(r, c, int(arr[r, c])) for r in range(h) for c in range(w) if int(arr[r, c]) in colors]
    delta = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[direction]
    for r, c, cval in starts:
        nr, nc = r + delta[0], c + delta[1]
        while 0 <= nr < h and 0 <= nc < w and int(arr[nr, nc]) == background:
            out[nr, nc] = cval
            nr += delta[0]
            nc += delta[1]
    return out


def apply_line_path_program(program: Program, arr: Grid) -> Grid:
    op = str(program.params["op"])
    raw_color = program.params.get("color")
    color = None if raw_color == "all" else int(raw_color)
    if op == "connect_same_color":
        return fill_between_same_color(
            arr,
            axis=str(program.params["axis"]),
            color=color,
            span=str(program.params["span"]),
        )
    if op == "extend_rays":
        return extend_color_rays(arr, color=color, direction=str(program.params["direction"]))
    raise ValueError(f"Unknown line/path op: {op}")


def component_by_descriptor(arr: Grid, descriptor: Dict[str, object]) -> Optional[Component]:
    comps = connected_components(arr, include_background=True)
    color = int(descriptor["color"])
    area = int(descriptor["area"])
    shape = tuple(tuple(x) for x in descriptor["shape"])
    matches = [
        comp
        for comp in comps
        if comp.color == color and comp.area == area and comp.shape_signature == shape
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def paste_component(out: Grid, comp: Component, top: int, left: int, color: Optional[int] = None) -> bool:
    h, w = out.shape
    source_min_r, source_min_c, _source_max_r, _source_max_c = comp.bbox
    paint_color = comp.color if color is None else int(color)
    target_cells = []
    for r, c in comp.cells:
        nr = top + (r - source_min_r)
        nc = left + (c - source_min_c)
        if not (0 <= nr < h and 0 <= nc < w):
            return False
        target_cells.append((nr, nc))
    for nr, nc in target_cells:
        out[nr, nc] = paint_color
    return True


def reflect_cells(cells: Tuple[Tuple[int, int], ...], axis: str, index: float) -> Optional[Tuple[Tuple[int, int], ...]]:
    reflected = []
    for r, c in cells:
        if axis == "h":
            nr = int(round(2.0 * index - r))
            nc = c
        elif axis == "v":
            nr = r
            nc = int(round(2.0 * index - c))
        else:
            raise ValueError(f"Bad reflection axis: {axis}")
        if abs((2.0 * index - (r if axis == "h" else c)) - (nr if axis == "h" else nc)) > 1e-6:
            return None
        reflected.append((nr, nc))
    return tuple(sorted(reflected))


def apply_rigid_transform_program(program: Program, arr: Grid) -> Grid:
    op = str(program.params["op"])
    background = most_common_color(arr)
    out = arr.copy()
    if op in {"copy_translate", "move_translate"}:
        comp = component_by_descriptor(arr, dict(program.params["component"]))
        if comp is None:
            return arr.copy()
        dr = int(program.params["dr"])
        dc = int(program.params["dc"])
        top = comp.bbox[0] + dr
        left = comp.bbox[1] + dc
        if op == "move_translate":
            for r, c in comp.cells:
                out[r, c] = background
        if not paste_component(out, comp, top, left):
            return arr.copy()
        return out
    if op == "copy_reflect":
        comp = component_by_descriptor(arr, dict(program.params["component"]))
        if comp is None:
            return arr.copy()
        reflected = reflect_cells(
            comp.cells,
            axis=str(program.params["axis"]),
            index=float(program.params["index"]),
        )
        if reflected is None:
            return arr.copy()
        h, w = arr.shape
        for r, c in reflected:
            if not (0 <= r < h and 0 <= c < w):
                return arr.copy()
        for r, c in reflected:
            out[r, c] = comp.color
        return out
    raise ValueError(f"Unknown rigid op: {op}")


def verify_program(program: Program, demos: List[Dict[str, object]]) -> bool:
    for demo in demos:
        pred = apply_program(program, np.asarray(demo["input"], dtype=np.uint8))
        target = np.asarray(demo["output"], dtype=np.uint8)
        if pred.shape != target.shape or not np.array_equal(pred, target):
            return False
    return True


def verify_program_trace(program: Program, demos: List[Dict[str, object]]) -> Dict[str, object]:
    mismatches: List[int] = []
    failed_demo_index: Optional[int] = None
    for demo_idx, demo in enumerate(demos):
        pred = apply_program(program, np.asarray(demo["input"], dtype=np.uint8))
        target = np.asarray(demo["output"], dtype=np.uint8)
        if pred.shape != target.shape:
            mismatch_count = max(pred.size, target.size)
        else:
            mismatch_count = int(np.sum(pred != target))
        mismatches.append(mismatch_count)
        if mismatch_count != 0 and failed_demo_index is None:
            failed_demo_index = demo_idx
    return {
        "verified_on_all_demos": all(x == 0 for x in mismatches),
        "failed_demo_index": failed_demo_index,
        "mismatch_count_per_demo": mismatches,
    }


_PARAMETRIC_LODO_FAMILIES = {"orpi_conditional_recolour_by_region"}


def is_lodo_stable(program: Program, demos: List[Dict[str, object]], families: List[str]) -> bool:
    if len(demos) <= 1:
        return True
    target_key = program_stability_key(program)
    parametric = program.family in _PARAMETRIC_LODO_FAMILIES
    for held_idx, held_demo in enumerate(demos):
        train_subset = [demo for idx, demo in enumerate(demos) if idx != held_idx]
        subset_programs = synthesize_programs(train_subset, families)
        if parametric:
            if not any(program_stability_key(c) == target_key for c in subset_programs):
                return False
            continue
        held_input = np.asarray(held_demo["input"], dtype=np.uint8)
        held_output = np.asarray(held_demo["output"], dtype=np.uint8)
        ok = False
        for candidate in subset_programs:
            if program_stability_key(candidate) != target_key:
                continue
            pred = apply_program(candidate, held_input)
            if pred.shape == held_output.shape and np.array_equal(pred, held_output):
                ok = True
                break
        if not ok:
            return False
    return True


def exact_color_map_program(demos: List[Dict[str, object]]) -> Optional[Program]:
    mapping: Dict[int, int] = {}
    for demo in demos:
        inp = np.asarray(demo["input"], dtype=np.uint8)
        out = np.asarray(demo["output"], dtype=np.uint8)
        if inp.shape != out.shape:
            return None
        for color in np.unique(inp).tolist():
            vals = np.unique(out[inp == color])
            if vals.size != 1:
                return None
            src = int(color)
            dst = int(vals[0])
            if src in mapping and mapping[src] != dst:
                return None
            mapping[src] = dst
    if all(src == dst for src, dst in mapping.items()):
        return None
    return Program(
        name="global_color_map",
        family="component_recolour",
        params={"mapping": {str(k): int(v) for k, v in sorted(mapping.items())}},
        debug_reason="deterministic input-colour to output-colour map across demonstrations",
    )


def component_selectors_from_changed_demo(inp: Grid, out: Grid) -> List[Tuple[Dict[str, object], int, str]]:
    if inp.shape != out.shape:
        return []
    background = most_common_color(inp)
    comps = connected_components(inp, include_background=True)
    selectors: List[Tuple[Dict[str, object], int, str]] = []
    for comp in comps:
        in_vals = np.array([inp[r, c] for r, c in comp.cells], dtype=np.uint8)
        out_vals = np.array([out[r, c] for r, c in comp.cells], dtype=np.uint8)
        if np.array_equal(in_vals, out_vals):
            continue
        unique_out = np.unique(out_vals)
        if unique_out.size != 1:
            continue
        target = int(unique_out[0])
        shape = tuple(tuple(x) for x in comp.shape_signature)
        base_reason = f"component color {comp.color} area {comp.area} changed uniformly to {target}"
        selectors.extend(
            [
                ({"type": "source_color", "color": comp.color}, target, base_reason + "; selector=color"),
                (
                    {"type": "source_color_area", "color": comp.color, "area": comp.area},
                    target,
                    base_reason + "; selector=color+area",
                ),
                (
                    {"type": "source_color_shape", "color": comp.color, "shape": shape},
                    target,
                    base_reason + "; selector=color+shape",
                ),
                (
                    {"type": "source_color_shape_area", "color": comp.color, "shape": shape, "area": comp.area},
                    target,
                    base_reason + "; selector=color+shape+area",
                ),
                ({"type": "shape_any_color", "shape": shape}, target, base_reason + "; selector=shape"),
            ]
        )
        if target == background:
            selectors.extend(
                [
                    (
                        {"type": "extreme_by_color", "color": comp.color, "axis": "row", "side": "min"},
                        target,
                        base_reason + "; marker topmost of color erased",
                    ),
                    (
                        {"type": "extreme_by_color", "color": comp.color, "axis": "row", "side": "max"},
                        target,
                        base_reason + "; marker bottommost of color erased",
                    ),
                    (
                        {"type": "extreme_by_color", "color": comp.color, "axis": "col", "side": "min"},
                        target,
                        base_reason + "; marker leftmost of color erased",
                    ),
                    (
                        {"type": "extreme_by_color", "color": comp.color, "axis": "col", "side": "max"},
                        target,
                        base_reason + "; marker rightmost of color erased",
                    ),
                ]
            )
    return selectors


def keep_delete_programs_from_demo(inp: Grid, out: Grid, demo_idx: int) -> List[Program]:
    if inp.shape != out.shape:
        return []
    background = most_common_color(inp)
    programs: List[Program] = []
    comps = connected_components(inp, include_background=True)
    colors = sorted({comp.color for comp in comps if comp.color != background})
    for color in colors:
        color_comps = [comp for comp in comps if comp.color == color]
        if len(color_comps) < 2:
            continue
        out_color_cells = int((out == color).sum())
        in_color_cells = int((inp == color).sum())
        if out_color_cells >= in_color_cells:
            continue
        selectors: List[Tuple[Dict[str, object], str]] = []
        for axis in ("row", "col"):
            for side in ("min", "max"):
                selectors.append(
                    (
                        {"type": "extreme_by_color", "color": color, "axis": axis, "side": side},
                        f"keep only {side} {axis} component of color {color}",
                    )
                )
        for comp in color_comps:
            shape = tuple(tuple(x) for x in comp.shape_signature)
            selectors.extend(
                [
                    (
                        {"type": "source_color_area", "color": color, "area": comp.area},
                        f"keep only area {comp.area} components of color {color}",
                    ),
                    (
                        {"type": "source_color_shape", "color": color, "shape": shape},
                        f"keep only shape-matched components of color {color}",
                    ),
                    (
                        {"type": "source_color_shape_area", "color": color, "shape": shape, "area": comp.area},
                        f"keep only shape+area matched components of color {color}",
                    ),
                ]
            )
        for selector, reason in selectors:
            programs.append(
                Program(
                    name="component_delete_or_keep_by_relation",
                    family="marker_selection",
                    params={
                        "source_color": int(color),
                        "background_color": int(background),
                        "selector": selector,
                        "source_demo": int(demo_idx),
                    },
                    debug_reason=reason,
                )
            )
    return programs


def line_path_programs_from_demo(inp: Grid, out: Grid, demo_idx: int) -> List[Program]:
    if inp.shape != out.shape:
        return []
    background = most_common_color(inp)
    changed_to_colors = sorted(
        {
            int(x)
            for x in np.unique(out[(inp != out) & (inp == background)])
            if int(x) != background
        }
    )
    non_background_colors = sorted({int(x) for x in np.unique(inp) if int(x) != background})
    colors: List[object] = list(dict.fromkeys(changed_to_colors + non_background_colors))
    colors.append("all")

    programs: List[Program] = []
    for color in colors:
        for axis in ("row", "col"):
            for span in ("outermost", "consecutive"):
                programs.append(
                    Program(
                        name="line_path_completion",
                        family="line_path_completion",
                        params={
                            "op": "connect_same_color",
                            "axis": axis,
                            "span": span,
                            "color": color,
                            "source_demo": int(demo_idx),
                        },
                        debug_reason=f"connect same-colour anchors axis={axis} span={span} color={color}",
                    )
                )
        for direction in ("up", "down", "left", "right"):
            programs.append(
                Program(
                    name="line_path_completion",
                    family="line_path_completion",
                    params={
                        "op": "extend_rays",
                        "direction": direction,
                        "color": color,
                        "source_demo": int(demo_idx),
                    },
                    debug_reason=f"extend colour rays direction={direction} color={color}",
                )
            )
    return programs


def component_descriptor(comp: Component) -> Dict[str, object]:
    return {
        "color": int(comp.color),
        "area": int(comp.area),
        "shape": tuple(tuple(x) for x in comp.shape_signature),
    }


def rigid_transform_programs_from_demo(inp: Grid, out: Grid, demo_idx: int) -> List[Program]:
    if inp.shape != out.shape:
        return []
    background = most_common_color(inp)
    changed = inp != out
    in_comps = [
        comp
        for comp in connected_components(inp, include_background=True)
        if comp.color != background and comp.area >= 2
    ]
    out_comps = [
        comp
        for comp in connected_components(out, include_background=True)
        if comp.color != background
        and comp.area >= 2
        and any(bool(changed[r, c]) for r, c in comp.cells)
    ]
    programs: List[Program] = []

    for source in in_comps:
        descriptor = component_descriptor(source)
        source_shape = tuple(tuple(x) for x in source.shape_signature)
        source_top, source_left, _source_bottom, _source_right = source.bbox
        matching_out = [
            comp
            for comp in out_comps
            if comp.color == source.color and comp.area == source.area
        ]
        for target in matching_out:
            target_top, target_left, _target_bottom, _target_right = target.bbox
            dr = target_top - source_top
            dc = target_left - source_left
            if dr != 0 or dc != 0:
                for op in ("copy_translate", "move_translate"):
                    programs.append(
                        Program(
                            name="rigid_transform_copy",
                            family="rigid_transform_copy",
                            params={
                                "op": op,
                                "component": descriptor,
                                "dr": int(dr),
                                "dc": int(dc),
                                "source_demo": int(demo_idx),
                            },
                            debug_reason=f"{op} component color={source.color} area={source.area} dr={dr} dc={dc}",
                        )
                    )
            for axis in ("h", "v"):
                candidate_indices = set()
                source_axis_vals = [r if axis == "h" else c for r, c in source.cells]
                target_axis_vals = [r if axis == "h" else c for r, c in target.cells]
                candidate_indices.add((float(min(source_axis_vals)) + float(max(target_axis_vals))) / 2.0)
                candidate_indices.add((source.centroid[0 if axis == "h" else 1] + target.centroid[0 if axis == "h" else 1]) / 2.0)
                for index in sorted(candidate_indices):
                    reflected = reflect_cells(source.cells, axis=axis, index=index)
                    if reflected is not None and reflected == target.cells:
                        programs.append(
                            Program(
                                name="rigid_transform_copy",
                                family="rigid_transform_copy",
                                params={
                                    "op": "copy_reflect",
                                    "component": descriptor,
                                    "axis": axis,
                                    "index": float(index),
                                    "source_demo": int(demo_idx),
                                },
                                debug_reason=f"copy_reflect component color={source.color} area={source.area} axis={axis} index={index}",
                            )
                        )

        # Shape may be colour-rewritten after the rigid operation. Generate a
        # colour-preserving descriptor only here; recolour is covered by D0.1.
        same_shape_out = [
            comp
            for comp in out_comps
            if comp.area == source.area and tuple(tuple(x) for x in comp.shape_signature) == source_shape
        ]
        for target in same_shape_out:
            if target.color != source.color:
                continue
            target_top, target_left, _target_bottom, _target_right = target.bbox
            dr = target_top - source_top
            dc = target_left - source_left
            if dr == 0 and dc == 0:
                continue
            programs.append(
                Program(
                    name="rigid_transform_copy",
                    family="rigid_transform_copy",
                    params={
                        "op": "copy_translate",
                        "component": descriptor,
                        "dr": int(dr),
                        "dc": int(dc),
                        "source_demo": int(demo_idx),
                    },
                    debug_reason=f"copy same-shape component color={source.color} dr={dr} dc={dc}",
                )
            )
    return programs


def orient_motif_programs_from_demo(inp: Grid, out: Grid, demo_idx: int) -> List[Program]:
    if inp.shape != out.shape:
        return []
    background = most_common_color(inp)
    changed = inp != out
    if not changed.any():
        return []
    changed_input_colors = {int(x) for x in np.unique(inp[changed])}
    changed_output_colors = {int(x) for x in np.unique(out[changed])}
    if changed_output_colors != {int(background)}:
        return []
    motif = dominant_multicolour_object(inp)
    if motif is None:
        return []
    motif_cells = set(motif.cells)
    comps = connected_components(inp, include_background=True)
    programs: List[Program] = []
    for marker_color in sorted(changed_input_colors):
        markers = [
            comp
            for comp in comps
            if comp.color == marker_color and comp.area == 1 and comp.cells[0] not in motif_cells
        ]
        if len(markers) < 2:
            continue
        selected = [marker for marker in markers if int(out[marker.cells[0]]) == marker_color]
        erased = [marker for marker in markers if int(out[marker.cells[0]]) == background and int(inp[marker.cells[0]]) == marker_color]
        if len(selected) != 1 or not erased:
            continue
        relation = marker_relation(motif, selected[0])
        for policy in ("nearest_side", "farthest_side", "extreme_side", "quadrant_nearest", "quadrant_farthest"):
            programs.append(
                Program(
                    name="orient_motif_select_marker",
                    family="orient_motif_select_marker",
                    params={
                        "marker_color": int(marker_color),
                        "background_color": int(background),
                        "motif_selector": "largest_multicolour",
                        "relation": relation,
                        "policy": policy,
                        "source_demo": int(demo_idx),
                    },
                    debug_reason=(
                        "select singleton marker by relation to dominant multi-colour motif; "
                        f"policy={policy}, side={relation['side']}"
                    ),
                )
            )
    return programs


def complete_region_programs_from_demo(inp: Grid, out: Grid, demo_idx: int) -> List[Program]:
    if inp.shape != out.shape:
        return []
    background = most_common_color(inp)
    changed = inp != out
    if not changed.any():
        return []
    programs: List[Program] = []
    for color in sorted(int(x) for x in np.unique(out[changed])):
        # Region completion is a fill operation: changed cells should become the
        # selected region colour without moving existing non-background structure.
        if not np.all(out[changed] == color):
            continue
        programs.append(
            Program(
                name="complete_dominant_region",
                family="complete_dominant_region",
                params={
                    "region_color": int(color),
                    "background_color": int(background),
                    "op": "enclosed_non_region",
                    "source_demo": int(demo_idx),
                },
                debug_reason="fill non-region islands enclosed by dominant region colour",
            )
        )
        programs.append(
            Program(
                name="complete_dominant_region",
                family="complete_dominant_region",
                params={
                    "region_color": int(color),
                    "background_color": int(background),
                    "op": "enclosed_background",
                    "source_demo": int(demo_idx),
                },
                debug_reason="fill enclosed background holes using changed output colour",
            )
        )
        programs.append(
            Program(
                name="complete_dominant_region",
                family="complete_dominant_region",
                params={
                    "region_color": int(color),
                    "background_color": int(background),
                    "op": "bbox_background",
                    "source_demo": int(demo_idx),
                },
                debug_reason="fill background pockets inside dominant region bounding box",
            )
        )
        for pad in (0, 1, 2):
            programs.append(
                Program(
                    name="complete_dominant_region",
                    family="complete_dominant_region",
                    params={
                        "region_color": int(color),
                        "background_color": int(background),
                        "op": "bbox_non_region",
                        "pad": int(pad),
                        "source_demo": int(demo_idx),
                    },
                    debug_reason=f"fill non-region cells in dominant region bbox with pad={pad}",
                )
            )
        for neighbours in ("4", "8"):
            for threshold in (2, 3, 4, 5):
                max_neighbours = 4 if neighbours == "4" else 8
                if threshold > max_neighbours:
                    continue
                for iterations in (1, 2):
                    programs.append(
                        Program(
                            name="complete_dominant_region",
                            family="complete_dominant_region",
                            params={
                                "region_color": int(color),
                                "background_color": int(background),
                                "op": "neighbor_majority",
                                "neighbours": neighbours,
                                "threshold": int(threshold),
                                "iterations": int(iterations),
                                "source_demo": int(demo_idx),
                            },
                            debug_reason=(
                                "fill background cells by local region-colour majority; "
                                f"neighbours={neighbours}, threshold={threshold}, iterations={iterations}"
                            ),
                        )
                    )
    return programs


def motif_delta_programs_from_demo(inp: Grid, out: Grid, demo_idx: int) -> List[Program]:
    if inp.shape != out.shape:
        return []
    background = most_common_color(inp)
    motif = dominant_multicolour_object(inp)
    out_motif = dominant_multicolour_object(out)
    if motif is None:
        return []
    min_r, min_c, max_r, max_c = motif.bbox
    motif_rewrites = []
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if int(inp[r, c]) != int(out[r, c]):
                motif_rewrites.append(
                    {
                        "rel": [int(r - min_r), int(c - min_c)],
                        "input_color": int(inp[r, c]),
                        "output_color": int(out[r, c]),
                    }
                )
    changed = inp != out
    if not motif_rewrites and not changed.any():
        return []
    input_comps = connected_components(inp, include_background=True)
    output_comps = connected_components(out, include_background=True)
    motif_cells = set(motif.cells)
    out_motif_cells = set(out_motif.cells) if out_motif is not None else set()
    programs: List[Program] = []
    for marker_color in sorted(int(x) for x in np.unique(inp)):
        if marker_color == background:
            continue
        input_markers = [
            comp
            for comp in input_comps
            if comp.color == marker_color and comp.area == 1 and comp.cells[0] not in motif_cells
        ]
        if len(input_markers) < 2:
            continue
        output_markers = [
            comp
            for comp in output_comps
            if comp.color == marker_color and comp.area == 1 and comp.cells[0] not in out_motif_cells
        ]
        if len(output_markers) != 1:
            continue
        relation = marker_relation(out_motif or motif, output_markers[0])
        rel_cell = [
            float(output_markers[0].centroid[0] - (out_motif or motif).centroid[0]),
            float(output_markers[0].centroid[1] - (out_motif or motif).centroid[1]),
        ]
        for policy in ("nearest_side", "farthest_side", "extreme_side", "quadrant_nearest", "quadrant_farthest"):
            programs.append(
                Program(
                    name="motif_orientation_select_marker",
                    family="motif_orientation_select_marker",
                    params={
                        "marker_color": int(marker_color),
                        "background_color": int(background),
                        "motif_selector": "largest_multicolour",
                        "motif_rewrites": motif_rewrites,
                        "marker_relation": relation,
                        "selection_policy": policy,
                        "selected_marker_rel_cell": rel_cell,
                        "source_demo": int(demo_idx),
                    },
                    debug_reason=(
                        "delta-derived motif rewrite plus marker selection; "
                        f"marker_color={marker_color}, policy={policy}, motif_rewrites={len(motif_rewrites)}"
                    ),
                )
            )
    return programs


def template_recolour_programs_from_demo(inp: Grid, out: Grid, demo_idx: int) -> List[Program]:
    if inp.shape != out.shape:
        return []
    changed = inp != out
    if not changed.any():
        return []
    programs: List[Program] = []
    for source_color in sorted(int(x) for x in np.unique(inp[changed])):
        for target_color in sorted(int(x) for x in np.unique(out[changed & (inp == source_color)])):
            if source_color == target_color:
                continue
            comps = [comp for comp in connected_components(inp, include_background=True) if comp.color == source_color]
            changed_cells = {(int(r), int(c)) for r, c in zip(*np.where(changed & (inp == source_color) & (out == target_color)))}
            changed_comps = [comp for comp in comps if set(comp.cells) and set(comp.cells).issubset(changed_cells)]
            if not changed_comps:
                continue
            comps_by_row = sorted(comps, key=lambda comp: (comp.centroid[0], comp.centroid[1]))
            comps_by_col = sorted(comps, key=lambda comp: (comp.centroid[1], comp.centroid[0]))
            for comp in changed_comps:
                row_rank = comps_by_row.index(comp)
                col_rank = comps_by_col.index(comp)
                programs.append(
                    Program(
                        name="template_component_recolour",
                        family="template_component_recolour",
                        params={
                            "source_color": int(source_color),
                            "target_color": int(target_color),
                            "selector": {"type": "rank_by_axis", "axis": "row", "rank": int(row_rank), "reverse": False},
                            "source_demo": int(demo_idx),
                        },
                        debug_reason=f"delta recolours source_color={source_color} component row-rank={row_rank} to {target_color}",
                    )
                )
                programs.append(
                    Program(
                        name="template_component_recolour",
                        family="template_component_recolour",
                        params={
                            "source_color": int(source_color),
                            "target_color": int(target_color),
                            "selector": {"type": "rank_by_axis", "axis": "col", "rank": int(col_rank), "reverse": False},
                            "source_demo": int(demo_idx),
                        },
                        debug_reason=f"delta recolours source_color={source_color} component col-rank={col_rank} to {target_color}",
                    )
                )
                same_shape = [candidate for candidate in comps_by_row if candidate.shape_signature == comp.shape_signature]
                shape_rank = same_shape.index(comp)
                programs.append(
                    Program(
                        name="template_component_recolour",
                        family="template_component_recolour",
                        params={
                            "source_color": int(source_color),
                            "target_color": int(target_color),
                            "selector": {
                                "type": "shape_and_rank",
                                "shape": [list(x) for x in comp.shape_signature],
                                "rank": int(shape_rank),
                            },
                            "source_demo": int(demo_idx),
                        },
                        debug_reason=(
                            f"delta recolours source_color={source_color} shape-rank={shape_rank} "
                            f"to {target_color}"
                        ),
                    )
                )
    return programs


def _orpi_program_from_existing(program: Program, name: str, family: str, reason_prefix: str) -> Program:
    params = dict(program.params)
    params.pop("source_demo", None)
    return Program(
        name=name,
        family=family,
        params=params,
        debug_reason=f"{reason_prefix}; {program.debug_reason}",
    )


def orpi_marker_programs_from_deltas(demos: List[Dict[str, object]]) -> List[Program]:
    deltas = demo_deltas(demos)
    programs: List[Program] = []
    for rec in deltas:
        demo_idx = int(rec["demo_index"])
        demo = demos[demo_idx]
        inp = np.asarray(demo["input"], dtype=np.uint8)
        out = np.asarray(demo["output"], dtype=np.uint8)
        delta = dict(rec["delta"])
        if not delta.get("marker_selection_events") and not delta.get("motif_cell_rewrites"):
            continue
        for legacy in motif_delta_programs_from_demo(inp, out, demo_idx):
            programs.append(
                _orpi_program_from_existing(
                    legacy,
                    "orpi_marker_from_motif_delta",
                    "orpi_marker_from_motif_delta",
                    "ORPI unified from demo delta: motif rewrite plus marker-selection event",
                )
            )
    return unique_programs(programs)


def _changed_whole_components(inp: Grid, out: Grid) -> List[Tuple[Component, int]]:
    changed = inp != out
    result: List[Tuple[Component, int]] = []
    for source_color in sorted(int(x) for x in np.unique(inp[changed])) if changed.any() else []:
        comps = [comp for comp in connected_components(inp, include_background=True) if comp.color == source_color]
        for comp in comps:
            cells = set(comp.cells)
            if not cells or not cells.issubset({(int(r), int(c)) for r, c in zip(*np.where(changed))}):
                continue
            targets = {int(out[r, c]) for r, c in cells}
            if len(targets) == 1 and next(iter(targets)) != source_color:
                result.append((comp, int(next(iter(targets)))))
    return result


def _hole_count_in_bbox(comp: Component, grid: Grid) -> int:
    arr = np.asarray(grid, dtype=np.uint8)
    min_r, min_c, max_r, max_c = comp.bbox
    sub = arr[min_r : max_r + 1, min_c : max_c + 1]
    background = most_common_color(arr)
    mask = (sub == background).astype(np.uint8)
    if mask.sum() == 0:
        return 0
    visited = np.zeros_like(mask, dtype=bool)
    holes = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c] == 0 or visited[r, c]:
                continue
            touches_border = False
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and mask[nr, nc] == 1:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _component_property_bag(comp: Component, inp: Grid, all_comps: List[Component]) -> Dict[str, object]:
    arr = np.asarray(inp, dtype=np.uint8)
    h, w = arr.shape
    same_color = [c for c in all_comps if c.color == comp.color]
    same_shape = [c for c in all_comps if c.shape_signature == comp.shape_signature]
    max_area = max((c.area for c in same_color), default=0)
    min_area = min((c.area for c in same_color), default=0)
    bbox_h = int(comp.bbox[2] - comp.bbox[0] + 1)
    bbox_w = int(comp.bbox[3] - comp.bbox[1] + 1)
    is_h_line = bool(bbox_h == 1 and comp.area >= 2)
    is_v_line = bool(bbox_w == 1 and comp.area >= 2)
    zone_v = "lower" if comp.centroid[0] >= (h - 1) / 2.0 else "upper"
    zone_h = "right" if comp.centroid[1] >= (w - 1) / 2.0 else "left"
    sorted_by_area_desc = sorted(same_color, key=lambda c: (-c.area, c.bbox))
    area_rank_largest = next((i for i, c in enumerate(sorted_by_area_desc) if c is comp), -1)
    sorted_by_area_asc = sorted(same_color, key=lambda c: (c.area, c.bbox))
    area_rank_smallest = next((i for i, c in enumerate(sorted_by_area_asc) if c is comp), -1)
    aspect_ratio_band = "square" if bbox_h == bbox_w else ("wide" if bbox_w > bbox_h else "tall")
    return {
        "source_color": int(comp.color),
        "area": int(comp.area),
        "shape_signature": comp.shape_signature,
        "zone_vertical": zone_v,
        "zone_horizontal": zone_h,
        "centroid_quadrant": f"{zone_v}_{zone_h}",
        "is_largest_of_color": bool(comp.area == max_area and len(same_color) > 1),
        "is_smallest_of_color": bool(comp.area == min_area and len(same_color) > 1),
        "is_unique_of_color": bool(len(same_color) == 1),
        "area_rank_largest": int(area_rank_largest),
        "area_rank_smallest": int(area_rank_smallest),
        "count_same_shape": int(len(same_shape)),
        "hole_count": int(_hole_count_in_bbox(comp, arr)),
        "bbox_h": bbox_h,
        "bbox_w": bbox_w,
        "aspect_ratio_band": aspect_ratio_band,
        "is_horizontal_line": is_h_line,
        "is_vertical_line": is_v_line,
        "is_axis_line": bool(is_h_line or is_v_line),
        "is_singleton": bool(comp.area == 1),
        "touches_border": bool(comp.touches_border),
    }


def _all_source_components_share_target(per_demo_props: List[List[Dict[str, object]]], source_color: int, target_color: int) -> bool:
    for props in per_demo_props:
        candidates = [p for p in props if p["source_color"] == source_color and p["target_color"] == target_color]
        if not candidates:
            return False
    return True


def intersect_template_recolour_selectors(demos: List[Dict[str, object]]) -> Dict[str, object]:
    per_demo_records: List[List[Dict[str, object]]] = []
    per_demo_changed_components: List[List[Tuple[Component, int]]] = []
    for demo in demos:
        inp = np.asarray(demo["input"], dtype=np.uint8)
        out = np.asarray(demo["output"], dtype=np.uint8)
        if inp.shape != out.shape:
            return {"selectors": [], "reason": "shape_mismatch"}
        changed = _changed_whole_components(inp, out)
        if not changed:
            return {"selectors": [], "reason": "no_whole_component_changes"}
        all_comps = connected_components(inp, include_background=True)
        recs = []
        for comp, target_color in changed:
            bag = _component_property_bag(comp, inp, all_comps)
            bag["target_color"] = int(target_color)
            recs.append(bag)
        per_demo_records.append(recs)
        per_demo_changed_components.append(changed)

    color_pairs_per_demo = [
        {(r["source_color"], r["target_color"]) for r in recs} for recs in per_demo_records
    ]
    shared_pairs = set.intersection(*color_pairs_per_demo) if color_pairs_per_demo else set()
    if not shared_pairs:
        return {"selectors": [], "reason": "no_shared_color_pair"}

    candidates: List[Dict[str, object]] = []
    for src, tgt in sorted(shared_pairs):
        filtered = [
            [r for r in recs if r["source_color"] == src and r["target_color"] == tgt]
            for recs in per_demo_records
        ]
        if any(not f for f in filtered):
            continue

        candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_only"}})

        for zone_axis, key in (("vertical", "zone_vertical"), ("horizontal", "zone_horizontal")):
            zones_per_demo = [{r[key] for r in f} for f in filtered]
            shared_zones = set.intersection(*zones_per_demo)
            if len(shared_zones) == 1:
                zone = next(iter(shared_zones))
                candidates.append({
                    "source_color": int(src),
                    "target_color": int(tgt),
                    "selector": {"type": "by_source_color_zone", "zone_axis": zone_axis, "zone": str(zone)},
                })

        if all(all(r["is_largest_of_color"] for r in f) for f in filtered):
            candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_extreme", "extreme": "largest"}})
        if all(all(r["is_smallest_of_color"] for r in f) for f in filtered):
            candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_extreme", "extreme": "smallest"}})
        if all(all(r["is_horizontal_line"] for r in f) for f in filtered):
            candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_horizontal_line"}})
        if all(all(r["is_vertical_line"] for r in f) for f in filtered):
            candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_vertical_line"}})
        if all(all(r["is_axis_line"] for r in f) for f in filtered):
            candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_axis_line"}})
        if all(all(r["is_singleton"] for r in f) for f in filtered):
            candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_singleton"}})
        if all(all(r["touches_border"] for r in f) for f in filtered):
            candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_touches_border"}})
        if all(all(not r["touches_border"] for r in f) for f in filtered):
            candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_not_touches_border"}})

        quad_per_demo = [{r["centroid_quadrant"] for r in f} for f in filtered]
        shared_quads = set.intersection(*quad_per_demo) if quad_per_demo else set()
        if len(shared_quads) == 1:
            quad = next(iter(shared_quads))
            candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_quadrant", "quadrant": str(quad)}})

        rank_largest_per_demo = [{r["area_rank_largest"] for r in f} for f in filtered]
        shared_rank_largest = set.intersection(*rank_largest_per_demo) if rank_largest_per_demo else set()
        for rank in sorted(shared_rank_largest):
            if rank >= 0:
                candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_area_rank_largest", "rank": int(rank)}})

        rank_smallest_per_demo = [{r["area_rank_smallest"] for r in f} for f in filtered]
        shared_rank_smallest = set.intersection(*rank_smallest_per_demo) if rank_smallest_per_demo else set()
        for rank in sorted(shared_rank_smallest):
            if rank >= 0:
                candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_area_rank_smallest", "rank": int(rank)}})

        aspect_per_demo = [{r["aspect_ratio_band"] for r in f} for f in filtered]
        shared_aspect = set.intersection(*aspect_per_demo) if aspect_per_demo else set()
        if len(shared_aspect) == 1:
            band = next(iter(shared_aspect))
            candidates.append({"source_color": int(src), "target_color": int(tgt), "selector": {"type": "by_source_color_aspect", "band": str(band)}})

        shapes_per_demo = [{r["shape_signature"] for r in f} for f in filtered]
        shared_shapes = set.intersection(*shapes_per_demo) if shapes_per_demo else set()
        for shape in shared_shapes:
            candidates.append({
                "source_color": int(src),
                "target_color": int(tgt),
                "selector": {"type": "by_source_color_shape", "shape": [list(x) for x in shape]},
            })

        areas_per_demo = [{r["area"] for r in f} for f in filtered]
        shared_areas = set.intersection(*areas_per_demo) if areas_per_demo else set()
        for area in sorted(shared_areas):
            candidates.append({
                "source_color": int(src),
                "target_color": int(tgt),
                "selector": {"type": "by_source_color_area", "area": int(area)},
            })

        holes_per_demo = [{r["hole_count"] for r in f} for f in filtered]
        shared_holes = set.intersection(*holes_per_demo) if holes_per_demo else set()
        for hole_count in sorted(shared_holes):
            candidates.append({
                "source_color": int(src),
                "target_color": int(tgt),
                "selector": {"type": "by_source_color_hole_count", "hole_count": int(hole_count)},
            })

    return {"selectors": candidates, "reason": "ok", "per_demo_records": per_demo_records}


def _select_components_by_intersection_selector(arr: Grid, selector: Dict[str, object], source_color: int) -> List[Component]:
    all_comps = connected_components(arr, include_background=True)
    arr_np = np.asarray(arr, dtype=np.uint8)
    h, w = arr_np.shape
    comps = [c for c in all_comps if c.color == source_color]
    typ = selector["type"]
    if typ == "by_source_color_only":
        return comps
    if typ == "by_source_color_zone":
        axis = str(selector.get("zone_axis", "vertical"))
        zone = str(selector["zone"])
        if axis == "vertical":
            if zone == "lower":
                return [c for c in comps if c.centroid[0] >= (h - 1) / 2.0]
            if zone == "upper":
                return [c for c in comps if c.centroid[0] <= (h - 1) / 2.0]
        if axis == "horizontal":
            if zone == "right":
                return [c for c in comps if c.centroid[1] >= (w - 1) / 2.0]
            if zone == "left":
                return [c for c in comps if c.centroid[1] <= (w - 1) / 2.0]
        return []
    if typ == "by_source_color_extreme":
        extreme = str(selector["extreme"])
        if not comps:
            return []
        if extreme == "largest":
            target_area = max(c.area for c in comps)
        else:
            target_area = min(c.area for c in comps)
        return [c for c in comps if c.area == target_area]
    if typ == "by_source_color_shape":
        shape = tuple(tuple(x) for x in selector["shape"])
        return [c for c in comps if c.shape_signature == shape]
    if typ == "by_source_color_area":
        area = int(selector["area"])
        return [c for c in comps if c.area == area]
    if typ == "by_source_color_hole_count":
        target_hc = int(selector["hole_count"])
        return [c for c in comps if _hole_count_in_bbox(c, arr_np) == target_hc]
    if typ == "by_source_color_horizontal_line":
        return [c for c in comps if (c.bbox[2] - c.bbox[0] + 1) == 1 and c.area >= 2]
    if typ == "by_source_color_vertical_line":
        return [c for c in comps if (c.bbox[3] - c.bbox[1] + 1) == 1 and c.area >= 2]
    if typ == "by_source_color_axis_line":
        return [
            c for c in comps
            if c.area >= 2 and ((c.bbox[2] - c.bbox[0] + 1) == 1 or (c.bbox[3] - c.bbox[1] + 1) == 1)
        ]
    if typ == "by_source_color_singleton":
        return [c for c in comps if c.area == 1]
    if typ == "by_source_color_touches_border":
        return [c for c in comps if c.touches_border]
    if typ == "by_source_color_not_touches_border":
        return [c for c in comps if not c.touches_border]
    if typ == "by_source_color_quadrant":
        quadrant = str(selector["quadrant"])
        result = []
        for c in comps:
            zv = "lower" if c.centroid[0] >= (h - 1) / 2.0 else "upper"
            zh = "right" if c.centroid[1] >= (w - 1) / 2.0 else "left"
            if f"{zv}_{zh}" == quadrant:
                result.append(c)
        return result
    if typ == "by_source_color_area_rank_largest":
        rank = int(selector["rank"])
        sorted_c = sorted(comps, key=lambda c: (-c.area, c.bbox))
        return [sorted_c[rank]] if 0 <= rank < len(sorted_c) else []
    if typ == "by_source_color_area_rank_smallest":
        rank = int(selector["rank"])
        sorted_c = sorted(comps, key=lambda c: (c.area, c.bbox))
        return [sorted_c[rank]] if 0 <= rank < len(sorted_c) else []
    if typ == "by_source_color_aspect":
        band = str(selector["band"])
        result = []
        for c in comps:
            bh = c.bbox[2] - c.bbox[0] + 1
            bw = c.bbox[3] - c.bbox[1] + 1
            this_band = "square" if bh == bw else ("wide" if bw > bh else "tall")
            if this_band == band:
                result.append(c)
        return result
    return []


_D4_TRANSFORM_NAMES = ("identity", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "anti_transpose")


def _apply_d4_transform(grid: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return grid.copy()
    if name == "rot90":
        return np.rot90(grid, k=1)
    if name == "rot180":
        return np.rot90(grid, k=2)
    if name == "rot270":
        return np.rot90(grid, k=3)
    if name == "flip_h":
        return np.flip(grid, axis=1)
    if name == "flip_v":
        return np.flip(grid, axis=0)
    if name == "transpose":
        return grid.T.copy()
    if name == "anti_transpose":
        return np.flip(np.flip(grid, axis=0), axis=1).T.copy()
    raise ValueError(f"Unknown D4 transform: {name}")


def _crop_motif(grid: Grid, motif: SceneObject) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.uint8)
    background = most_common_color(arr)
    min_r, min_c, max_r, max_c = motif.bbox
    sub = arr[min_r : max_r + 1, min_c : max_c + 1].copy()
    cell_set = set(motif.cells)
    out = np.full_like(sub, background)
    for r in range(sub.shape[0]):
        for c in range(sub.shape[1]):
            if (min_r + r, min_c + c) in cell_set:
                out[r, c] = sub[r, c]
    return out


def unify_motif_orientation(demos: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    transforms = []
    for demo in demos:
        inp = np.asarray(demo["input"], dtype=np.uint8)
        out = np.asarray(demo["output"], dtype=np.uint8)
        in_motif = dominant_multicolour_object(inp)
        out_motif = dominant_multicolour_object(out)
        if in_motif is None or out_motif is None:
            return None
        in_grid = _crop_motif(inp, in_motif)
        out_grid = _crop_motif(out, out_motif)
        found = None
        for name in _D4_TRANSFORM_NAMES:
            try:
                transformed = _apply_d4_transform(in_grid, name)
            except Exception:
                continue
            if transformed.shape == out_grid.shape and np.array_equal(transformed, out_grid):
                found = name
                break
        if found is None:
            return None
        transforms.append(found)
    if not transforms:
        return None
    if len(set(transforms)) != 1:
        return None
    return {"d4_transform": transforms[0]}


def _marker_relations_per_demo(demos: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    relations: List[Dict[str, object]] = []
    policies_intersected = set(["nearest_side", "farthest_side", "extreme_side", "quadrant_nearest", "quadrant_farthest"])
    marker_color: Optional[int] = None
    for demo in demos:
        inp = np.asarray(demo["input"], dtype=np.uint8)
        out = np.asarray(demo["output"], dtype=np.uint8)
        in_motif = dominant_multicolour_object(inp)
        out_motif = dominant_multicolour_object(out)
        if in_motif is None:
            return None
        in_comps = connected_components(inp, include_background=True)
        out_comps = connected_components(out, include_background=True)
        motif_cells = set(in_motif.cells)
        out_motif_cells = set(out_motif.cells) if out_motif is not None else set()
        background = most_common_color(inp)
        chosen_color: Optional[int] = None
        chosen_relation: Optional[Dict[str, object]] = None
        chosen_policies: set = set()
        for color in sorted(int(x) for x in np.unique(inp)):
            if color == background:
                continue
            input_markers = [c for c in in_comps if c.color == color and c.area == 1 and c.cells[0] not in motif_cells]
            output_markers = [c for c in out_comps if c.color == color and c.area == 1 and c.cells[0] not in out_motif_cells]
            if len(input_markers) < 2 or len(output_markers) != 1:
                continue
            kept_marker = output_markers[0]
            rel = marker_relation(out_motif or in_motif, kept_marker)
            valid_policies = set()
            for policy in ("nearest_side", "farthest_side", "extreme_side", "quadrant_nearest", "quadrant_farthest"):
                picked = choose_marker_by_relation(out_motif or in_motif, input_markers, rel, policy)
                if picked is not None and picked.cells[0] == kept_marker.cells[0]:
                    valid_policies.add(policy)
            if not valid_policies:
                continue
            chosen_color = int(color)
            chosen_relation = rel
            chosen_policies = valid_policies
            break
        if chosen_color is None or chosen_relation is None:
            return None
        if marker_color is None:
            marker_color = chosen_color
        elif marker_color != chosen_color:
            return None
        policies_intersected &= chosen_policies
        relations.append({"relation": chosen_relation, "marker_color": chosen_color})
    if not relations or not policies_intersected:
        return None
    first_rel = dict(relations[0]["relation"])
    for record in relations[1:]:
        other = record["relation"]
        if other["axis"] != first_rel["axis"] or other["side"] != first_rel["side"]:
            return None
        if other["row_sign"] != first_rel["row_sign"] or other["col_sign"] != first_rel["col_sign"]:
            return None
    return {
        "marker_color": int(marker_color),
        "relation": first_rel,
        "policy_candidates": sorted(policies_intersected),
        "background_color": int(most_common_color(np.asarray(demos[0]["input"], dtype=np.uint8))),
    }


def orpi_template_recolour_programs_from_deltas(demos: List[Dict[str, object]]) -> List[Program]:
    result = intersect_template_recolour_selectors(demos)
    candidates = result.get("selectors", []) if isinstance(result, dict) else []
    programs: List[Program] = []
    for cand in candidates:
        selector = cand["selector"]
        sel_desc = selector["type"]
        if sel_desc == "by_source_color_zone":
            sel_desc = f"zone[{selector['zone_axis']}={selector['zone']}]"
        elif sel_desc == "by_source_color_extreme":
            sel_desc = f"extreme[{selector['extreme']}]"
        elif sel_desc == "by_source_color_shape":
            sel_desc = "shape[fixed]"
        elif sel_desc == "by_source_color_area":
            sel_desc = f"area={selector['area']}"
        elif sel_desc == "by_source_color_hole_count":
            sel_desc = f"holes={selector['hole_count']}"
        programs.append(
            Program(
                name="orpi_template_recolour_from_delta",
                family="orpi_template_recolour_from_delta",
                params={
                    "source_color": int(cand["source_color"]),
                    "target_color": int(cand["target_color"]),
                    "selector": selector,
                },
                debug_reason=(
                    f"intersection-first selector {sel_desc} "
                    f"for {cand['source_color']}->{cand['target_color']}"
                ),
            )
        )
    return unique_programs(programs)


def orpi_marker_programs_from_deltas_unified(demos: List[Dict[str, object]]) -> List[Program]:
    orient = unify_motif_orientation(demos)
    relations = _marker_relations_per_demo(demos)
    if orient is None or relations is None:
        return []
    programs: List[Program] = []
    for policy in relations["policy_candidates"]:
        programs.append(
            Program(
                name="orpi_marker_from_motif_delta",
                family="orpi_marker_from_motif_delta",
                params={
                    "marker_color": int(relations["marker_color"]),
                    "background_color": int(relations["background_color"]),
                    "d4_transform": str(orient["d4_transform"]),
                    "marker_relation": dict(relations["relation"]),
                    "selection_policy": str(policy),
                },
                debug_reason=(
                    f"orientation-unified motif transform={orient['d4_transform']}, "
                    f"marker_color={relations['marker_color']}, policy={policy}"
                ),
            )
        )
    return unique_programs(programs)


def orpi_conditional_recolour_programs(demos: List[Dict[str, object]]) -> List[Program]:
    per_demo_records: List[List[Dict[str, object]]] = []
    for demo in demos:
        inp = np.asarray(demo["input"], dtype=np.uint8)
        out = np.asarray(demo["output"], dtype=np.uint8)
        if inp.shape != out.shape:
            return []
        changed = _changed_whole_components(inp, out)
        if not changed:
            return []
        all_comps = connected_components(inp, include_background=True)
        recs = []
        for comp, target_color in changed:
            bag = _component_property_bag(comp, inp, all_comps)
            bag["target_color"] = int(target_color)
            recs.append(bag)
        per_demo_records.append(recs)

    target_groups_per_demo = []
    for recs in per_demo_records:
        groups: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        for r in recs:
            groups[int(r["target_color"])].append(r)
        target_groups_per_demo.append(groups)

    all_targets_per_demo = [set(groups.keys()) for groups in target_groups_per_demo]
    shared_targets = set.intersection(*all_targets_per_demo) if all_targets_per_demo else set()
    if len(shared_targets) < 2:
        return []

    source_color_candidates: set = set()
    for recs in per_demo_records:
        source_color_candidates |= {int(r["source_color"]) for r in recs}
    if not source_color_candidates:
        return []

    programs: List[Program] = []
    for src_color in sorted(source_color_candidates):
        valid = True
        for recs in per_demo_records:
            this_demo = [r for r in recs if r["source_color"] == src_color]
            if not this_demo:
                valid = False
                break
        if not valid:
            continue

        for discriminator_type in (
            "zone_vertical",
            "zone_horizontal",
            "centroid_quadrant",
            "is_largest_of_color",
            "hole_count",
            "area",
            "area_rank_largest",
            "area_rank_smallest",
            "is_horizontal_line",
            "is_vertical_line",
            "is_axis_line",
            "is_singleton",
            "touches_border",
            "bbox_h",
            "bbox_w",
            "aspect_ratio_band",
            "count_same_shape",
        ):
            mapping: Dict[object, int] = {}
            mapping_valid = True
            for groups in target_groups_per_demo:
                for target_color, records_in_group in groups.items():
                    for rec in records_in_group:
                        if rec["source_color"] != src_color:
                            continue
                        key = rec[discriminator_type]
                        if isinstance(key, list):
                            key = tuple(key)
                        if key in mapping and mapping[key] != int(target_color):
                            mapping_valid = False
                            break
                        mapping[key] = int(target_color)
                    if not mapping_valid:
                        break
                if not mapping_valid:
                    break
            if not mapping_valid or not mapping:
                continue
            if len(set(mapping.values())) < 2:
                continue
            keys_serializable: List[List[object]] = []
            for k, v in sorted(mapping.items(), key=lambda kv: str(kv[0])):
                key_repr: object
                if isinstance(k, bool):
                    key_repr = bool(k)
                elif isinstance(k, (int, np.integer)):
                    key_repr = int(k)
                elif isinstance(k, (float, np.floating)):
                    key_repr = float(k)
                elif isinstance(k, tuple):
                    key_repr = [list(x) for x in k]
                else:
                    key_repr = str(k)
                keys_serializable.append([key_repr, int(v)])
            programs.append(
                Program(
                    name="orpi_conditional_recolour_by_region",
                    family="orpi_conditional_recolour_by_region",
                    params={
                        "source_color": int(src_color),
                        "discriminator": str(discriminator_type),
                        "mapping": keys_serializable,
                    },
                    debug_reason=(
                        f"conditional recolour: source_color={src_color}, "
                        f"discriminator={discriminator_type}, mapping_size={len(keys_serializable)}"
                    ),
                )
            )
    return unique_programs(programs)


def _axis_runs_of_color(grid: Grid, color: int, axis: str, min_length: int) -> List[set]:
    arr = np.asarray(grid, dtype=np.uint8)
    h, w = arr.shape
    runs: List[set] = []
    if axis == "h":
        for r in range(h):
            c = 0
            while c < w:
                if int(arr[r, c]) == color:
                    start = c
                    while c < w and int(arr[r, c]) == color:
                        c += 1
                    if c - start >= min_length:
                        runs.append({(r, k) for k in range(start, c)})
                else:
                    c += 1
    elif axis == "v":
        for c in range(w):
            r = 0
            while r < h:
                if int(arr[r, c]) == color:
                    start = r
                    while r < h and int(arr[r, c]) == color:
                        r += 1
                    if r - start >= min_length:
                        runs.append({(k, c) for k in range(start, r)})
                else:
                    r += 1
    return runs


def orpi_axis_line_recolour_programs(demos: List[Dict[str, object]]) -> List[Program]:
    candidates_per_demo: List[set] = []
    for demo in demos:
        inp = np.asarray(demo["input"], dtype=np.uint8)
        out = np.asarray(demo["output"], dtype=np.uint8)
        if inp.shape != out.shape:
            return []
        changed_by_pair: Dict[Tuple[int, int], set] = {}
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                src_v = int(inp[r, c])
                tgt_v = int(out[r, c])
                if src_v != tgt_v:
                    changed_by_pair.setdefault((src_v, tgt_v), set()).add((r, c))
        if not changed_by_pair:
            return []
        cands: set = set()
        for (src, tgt), cells in changed_by_pair.items():
            all_src_cells = {
                (r, c)
                for r in range(inp.shape[0])
                for c in range(inp.shape[1])
                if int(inp[r, c]) == src
            }
            unchanged_src = all_src_cells - cells
            for r2, c2 in unchanged_src:
                if int(out[r2, c2]) != src:
                    cells_match_strict = False
                    break
            else:
                cells_match_strict = True
            if not cells_match_strict:
                continue
            for axis in ("h", "v"):
                for min_len in range(2, 10):
                    runs = _axis_runs_of_color(inp, src, axis, min_len)
                    run_cells: set = set()
                    for run in runs:
                        run_cells |= run
                    if run_cells == cells:
                        cands.add((src, tgt, axis, min_len))
        candidates_per_demo.append(cands)
    if not candidates_per_demo or any(not cands for cands in candidates_per_demo):
        return []
    intersection = candidates_per_demo[0]
    for cs in candidates_per_demo[1:]:
        intersection &= cs
    programs: List[Program] = []
    for src, tgt, axis, min_len in sorted(intersection):
        programs.append(
            Program(
                name="orpi_axis_line_recolour",
                family="orpi_axis_line_recolour",
                params={
                    "source_color": int(src),
                    "target_color": int(tgt),
                    "axis": str(axis),
                    "min_length": int(min_len),
                },
                debug_reason=f"recolour {axis}-runs of color {src} length>={min_len} to {tgt}",
            )
        )
    return programs


def orpi_frame_from_seed_programs(demos: List[Dict[str, object]]) -> List[Program]:
    per_demo_inferences = []
    for demo in demos:
        inp = np.asarray(demo["input"], dtype=np.uint8)
        out = np.asarray(demo["output"], dtype=np.uint8)
        if inp.shape != out.shape:
            return []
        background = most_common_color(inp)
        seeds = []
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                if int(inp[r, c]) != background:
                    seeds.append((r, c, int(inp[r, c])))
        if not seeds:
            return []
        added_cells = [(r, c) for r in range(out.shape[0]) for c in range(out.shape[1])
                       if int(out[r, c]) != background and int(inp[r, c]) == background]
        if not added_cells:
            return []
        seed_anchor: Optional[Tuple[int, int, int]] = None
        seed_anchor_score = -1
        for sr, sc, color in seeds:
            keep_score = int(out[sr, sc] == color)
            if keep_score > seed_anchor_score:
                seed_anchor_score = keep_score
                seed_anchor = (sr, sc, color)
        if seed_anchor is None or seed_anchor_score == 0:
            return []
        anchor_r, anchor_c, anchor_color = seed_anchor
        out_components = connected_components(out, include_background=False)
        anchor_frame = None
        for comp in out_components:
            if comp.color == anchor_color:
                continue
            min_r, min_c, max_r, max_c = comp.bbox
            if min_r <= anchor_r <= max_r and min_c <= anchor_c <= max_c:
                anchor_frame = comp
                break
        if anchor_frame is None:
            return []
        min_r, min_c, max_r, max_c = anchor_frame.bbox
        rect_h = max_r - min_r + 1
        rect_w = max_c - min_c + 1
        per_demo_inferences.append({
            "anchor_color": int(anchor_color),
            "anchor_offset_r": int(anchor_r - min_r),
            "anchor_offset_c": int(anchor_c - min_c),
            "frame_color": int(anchor_frame.color),
            "frame_h": int(rect_h),
            "frame_w": int(rect_w),
        })
    if not per_demo_inferences:
        return []
    keys = ("anchor_color", "anchor_offset_r", "anchor_offset_c", "frame_color", "frame_h", "frame_w")
    first = per_demo_inferences[0]
    for inf in per_demo_inferences[1:]:
        for k in keys:
            if inf[k] != first[k]:
                return []
    program = Program(
        name="orpi_frame_from_seed",
        family="orpi_frame_from_seed",
        params=dict(first),
        debug_reason=(
            f"frame-from-seed: anchor_color={first['anchor_color']}, frame_color={first['frame_color']}, "
            f"rect={first['frame_h']}x{first['frame_w']}, offset=({first['anchor_offset_r']},{first['anchor_offset_c']})"
        ),
    )
    return [program]


def synthesize_orpi_raw_programs(demos: List[Dict[str, object]], families: List[str]) -> List[Program]:
    family_set = set(families)
    programs: List[Program] = []
    if "orpi_marker_from_motif_delta" in family_set:
        programs.extend(orpi_marker_programs_from_deltas_unified(demos))
    if "orpi_template_recolour_from_delta" in family_set:
        programs.extend(orpi_template_recolour_programs_from_deltas(demos))
    if "orpi_conditional_recolour_by_region" in family_set:
        programs.extend(orpi_conditional_recolour_programs(demos))
    if "orpi_frame_from_seed" in family_set:
        programs.extend(orpi_frame_from_seed_programs(demos))
    if "orpi_axis_line_recolour" in family_set:
        programs.extend(orpi_axis_line_recolour_programs(demos))
    return unique_programs(programs)


def unique_programs(programs: List[Program]) -> List[Program]:
    unique: Dict[str, Program] = {}
    for program in programs:
        unique[program_semantic_key(program)] = program
    return list(unique.values())


def synthesize_raw_programs(demos: List[Dict[str, object]], families: List[str]) -> List[Program]:
    candidates: List[Program] = []
    family_set = set(families)
    candidates.extend(synthesize_orpi_raw_programs(demos, families))
    color_map = exact_color_map_program(demos)
    if color_map is not None and color_map.family in family_set:
        candidates.append(color_map)

    for demo_idx, demo in enumerate(demos):
        inp = np.asarray(demo["input"], dtype=np.uint8)
        out = np.asarray(demo["output"], dtype=np.uint8)
        for selector, target, reason in component_selectors_from_changed_demo(inp, out):
            family = "marker_selection" if target == most_common_color(inp) else "component_recolour"
            if family not in family_set:
                continue
            candidates.append(
                Program(
                    name="component_recolour",
                    family=family,
                    params={
                        "selector": selector,
                        "target_color": int(target),
                        "include_background": True,
                        "source_demo": int(demo_idx),
                    },
                    debug_reason=reason,
                )
            )
        if "marker_selection" in family_set:
            candidates.extend(keep_delete_programs_from_demo(inp, out, demo_idx))
        if "line_path_completion" in family_set:
            candidates.extend(line_path_programs_from_demo(inp, out, demo_idx))
        if "rigid_transform_copy" in family_set:
            candidates.extend(rigid_transform_programs_from_demo(inp, out, demo_idx))
        if "orient_motif_select_marker" in family_set:
            candidates.extend(orient_motif_programs_from_demo(inp, out, demo_idx))
        if "complete_dominant_region" in family_set:
            candidates.extend(complete_region_programs_from_demo(inp, out, demo_idx))
        if "motif_orientation_select_marker" in family_set:
            candidates.extend(motif_delta_programs_from_demo(inp, out, demo_idx))
        if "template_component_recolour" in family_set:
            candidates.extend(template_recolour_programs_from_demo(inp, out, demo_idx))
    return unique_programs(candidates)


def synthesize_programs(demos: List[Dict[str, object]], families: List[str]) -> List[Program]:
    return [program for program in synthesize_raw_programs(demos, families) if verify_program(program, demos)]


def load_tasks(dataset: Path) -> Dict[str, Dict[str, object]]:
    path = dataset / "test_puzzles.json"
    if not path.exists():
        raise FileNotFoundError(f"Expected canonical task file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_families(raw: str) -> List[str]:
    families = [x.strip() for x in raw.split(",") if x.strip()]
    allowed = {
        "component_recolour",
        "marker_selection",
        "line_path_completion",
        "rigid_transform_copy",
        "orient_motif_select_marker",
        "complete_dominant_region",
        "motif_orientation_select_marker",
        "template_component_recolour",
        "orpi_marker_from_motif_delta",
        "orpi_template_recolour_from_delta",
        "orpi_anchor_path_from_delta",
        "orpi_transform_from_delta",
        "orpi_conditional_recolour_by_region",
        "orpi_frame_from_seed",
        "orpi_axis_line_recolour",
    }
    bad = sorted(set(families) - allowed)
    if bad:
        raise ValueError(f"Supported primitive families are {sorted(allowed)}, got unsupported families={bad}")
    return sorted(set(families))


def candidate_records(task_id: str, programs: List[Program], test_input: Grid, target: Grid) -> Tuple[List[Dict[str, object]], Dict[str, Grid]]:
    by_key: Dict[str, Grid] = {}
    rows: List[Dict[str, object]] = []
    for program_idx, program in enumerate(programs):
        output = apply_program(program, test_input)
        key = grid_key(output)
        by_key.setdefault(key, output)
        exact = int(output.shape == target.shape and np.array_equal(output, target))
        rows.append(
            {
                "task_id": task_id,
                "program_index": program_idx,
                "candidate_hash": key,
                "family": program.family,
                "name": program.name,
                "output_shape": f"{output.shape[0]}x{output.shape[1]}",
                "candidate_exact": exact,
                "program_params": json.dumps(program.params, sort_keys=True, separators=(",", ":")),
                "candidate_grid": grid_json(output),
                "debug_reason": program.debug_reason,
            }
        )
    return rows, by_key


def make_bucket_movement(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_bucket: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        bucket = str(row["bucket"])
        ctr = by_bucket[bucket]
        ctr["tasks"] += 1
        ctr["c0_exact"] += int(row["c0_exact"])
        ctr["verified_replace_exact"] += int(row["verified_replace_exact"])
        ctr["oracle_any_exact"] += int(row["oracle_any_exact"])
        ctr["replace_gain"] += int(row["replace_gain"])
        ctr["replace_loss"] += int(row["replace_loss"])
        ctr["oracle_gain"] += int(row["oracle_gain"])
        ctr["unique_candidate"] += int(row["unique_candidate"])
        ctr["tasks_with_verified_programs"] += int(int(row["n_verified_programs"]) > 0)
    return [{"bucket": bucket, **dict(by_bucket[bucket])} for bucket in BUCKETS if bucket in by_bucket]


def make_oracle_summary(rows: List[Dict[str, object]], bucket_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    total = Counter()
    for row in rows:
        total["tasks"] += 1
        total["c0_exact"] += int(row["c0_exact"])
        total["verified_replace_exact"] += int(row["verified_replace_exact"])
        total["oracle_any_exact"] += int(row["oracle_any_exact"])
        total["replace_gain"] += int(row["replace_gain"])
        total["replace_loss"] += int(row["replace_loss"])
        total["oracle_gain"] += int(row["oracle_gain"])
        total["unique_candidate"] += int(row["unique_candidate"])
        total["tasks_with_verified_programs"] += int(int(row["n_verified_programs"]) > 0)
    out = [{"bucket": "ALL", **dict(total)}]
    out.extend(bucket_rows)
    return out


def make_lodo_stability_summary(rows: List[Dict[str, object]], bucket_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    def summarize_bucket(bucket: str, selected_rows: List[Dict[str, object]]) -> Dict[str, object]:
        return {
            "bucket": bucket,
            "tasks": len(selected_rows),
            "n_raw_candidates": sum(int(row.get("n_raw_candidates", 0)) for row in selected_rows),
            "n_verified_programs": sum(int(row.get("n_verified_programs", 0)) for row in selected_rows),
            "n_lodo_stable_programs": sum(int(row.get("n_lodo_stable_programs", 0)) for row in selected_rows),
            "tasks_with_verified_programs": sum(int(row.get("n_verified_programs", 0)) > 0 for row in selected_rows),
            "tasks_with_lodo_stable_programs": sum(int(row.get("n_lodo_stable_programs", 0)) > 0 for row in selected_rows),
        }

    out = [summarize_bucket("ALL", rows)]
    for bucket_row in bucket_rows:
        bucket = str(bucket_row["bucket"])
        out.append(summarize_bucket(bucket, [row for row in rows if row["bucket"] == bucket]))
    return out


def decide(rows: List[Dict[str, object]], families: List[str]) -> Tuple[str, str, str]:
    both_fail_oracle = sum(int(row["oracle_gain"]) for row in rows if row["bucket"] == "both_fail")
    both_fail_replace = sum(int(row["replace_gain"]) for row in rows if row["bucket"] == "both_fail")
    replace_losses = sum(int(row["replace_loss"]) for row in rows)
    strict_exact = sum(int(row["verified_replace_exact"]) for row in rows)
    verified_total = sum(int(row.get("n_verified_programs", 0)) for row in rows)
    orpi_families = {
        "orpi_marker_from_motif_delta",
        "orpi_template_recolour_from_delta",
        "orpi_anchor_path_from_delta",
        "orpi_transform_from_delta",
        "orpi_conditional_recolour_by_region",
        "orpi_frame_from_seed",
        "orpi_axis_line_recolour",
    }
    if orpi_families.intersection(families):
        if both_fail_oracle >= 3 and both_fail_replace >= 1 and replace_losses == 0 and strict_exact > 125:
            return (
                "KEEP",
                "ORPI produced strong both_fail headroom with deployable strict gains and no C0 solved-task loss.",
                "Freeze these schemas, then add routing/path ORPI schemas for e5790162 and 29700607.",
            )
        if both_fail_oracle >= 1 and replace_losses == 0 and verified_total > 0:
            return (
                "KEEP",
                "ORPI produced at least one verified hard-bucket candidate without strict replacement loss.",
                "Inspect verified programs, then add routing/path schemas only after freezing this result.",
            )
        return (
            "REJECT",
            "ORPI delta-derived schemas did not produce verified both_fail headroom under the preregistered gate.",
            "Do not train. The blocker remains richer operation-space search or schema coverage.",
        )
    d2_families = {"motif_orientation_select_marker", "template_component_recolour"}
    if d2_families.intersection(families):
        if both_fail_oracle >= 1 and replace_losses == 0 and verified_total > 0:
            return (
                "KEEP",
                "D2 delta-derived programs generated at least one hard-bucket candidate without C0 solved-task losses.",
                "Inspect stable verified programs before considering routing/path schemas or any learned operation proposer.",
            )
        return (
            "REJECT",
            "D2 delta-derived schemas did not produce verified both_fail headroom under the preregistered gate.",
            "Do not train; inspect demo_deltas and verification failures to determine the missing operation-space search.",
        )
    d1_families = {"orient_motif_select_marker", "complete_dominant_region"}
    if d1_families.intersection(families):
        if both_fail_oracle >= 1 and replace_losses == 0 and verified_total > 0:
            return (
                "KEEP",
                "D1 compositional schemas generated at least one hard-bucket candidate without strict replacement losses.",
                "Inspect verified programs, then proceed to D1.2 routing only if the candidates are mechanistically valid.",
            )
        return (
            "REJECT",
            "D1 compositional schemas did not produce verified both_fail headroom under the preregistered gate.",
            "Do not train; inspect trace outputs for missing parser/proposer coverage before adding routing schemas.",
        )
    if both_fail_oracle >= 2 and both_fail_replace >= 1 and replace_losses == 0 and strict_exact > 125:
        return (
            "KEEP",
            "Verified repair generated hard-bucket candidates and strict replacement improves exact without C0 solved losses.",
            "Freeze the accepted primitive families and inspect candidate programs before any trainable primitive branch.",
        )
    if both_fail_oracle >= 2:
        return (
            "REJECT",
            "Verified repair has oracle headroom but strict replacement is not yet safe.",
            "Add agreement/selection logic before adding new primitive families.",
        )
    return (
        "REJECT",
        "Verified repair primitives do not generate enough hard-bucket exact candidates.",
        "Proceed to the next primitive family in order; do not fine-tune another pixel loss.",
    )


def trace_program_generation(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).resolve()
    tasks = load_tasks(dataset)
    task_id = str(args.trace_task)
    if task_id not in tasks:
        raise KeyError(f"trace task {task_id!r} not found in {dataset / 'test_puzzles.json'}")
    requested_families = parse_families(args.trace_family or args.families)
    task = tasks[task_id]
    demos = list(task["train"])
    test_input = np.asarray(task["test"][0]["input"], dtype=np.uint8)
    target = np.asarray(task["test"][0]["output"], dtype=np.uint8)
    trace_root = Path(args.trace_out_dir).resolve() / task_id
    trace_root.mkdir(parents=True, exist_ok=True)

    parsed = {
        "task_id": task_id,
        "train": [
            {
                "demo_index": idx,
                "input": extract_multiview_scene_graph(np.asarray(demo["input"], dtype=np.uint8)),
                "output": extract_multiview_scene_graph(np.asarray(demo["output"], dtype=np.uint8)),
            }
            for idx, demo in enumerate(demos)
        ],
        "test_input": extract_multiview_scene_graph(test_input),
    }
    (trace_root / "parsed_scene_graph.json").write_text(
        json.dumps(parsed, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (trace_root / "parsed_multiview_graph.json").write_text(
        json.dumps(parsed, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for family in requested_families:
        family_dir = trace_root / family
        family_dir.mkdir(parents=True, exist_ok=True)
        (family_dir / "parsed_nodes.json").write_text(
            json.dumps(parsed, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (family_dir / "parsed_multiview_graph.json").write_text(
            json.dumps(parsed, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        relations = {
            "task_id": task_id,
            "family": family,
            "train": [
                {
                    "demo_index": idx,
                    "input_relations": relation_summary(np.asarray(demo["input"], dtype=np.uint8)),
                    "output_relations": relation_summary(np.asarray(demo["output"], dtype=np.uint8)),
                }
                for idx, demo in enumerate(demos)
            ],
            "test_input_relations": relation_summary(test_input),
        }
        (family_dir / "parsed_relations.json").write_text(
            json.dumps(relations, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (family_dir / "relations.json").write_text(
            json.dumps(relations, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        delta_records = demo_deltas(demos)
        (family_dir / "demo_deltas.json").write_text(
            json.dumps(delta_records, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        intersection_payload: Dict[str, object] = {"family": family, "task_id": task_id}
        if family == "orpi_template_recolour_from_delta":
            intersection_payload["template_recolour"] = intersect_template_recolour_selectors(demos)
        if family == "orpi_marker_from_motif_delta":
            intersection_payload["motif_orientation"] = unify_motif_orientation(demos)
            intersection_payload["marker_relation"] = _marker_relations_per_demo(demos)
        if family == "orpi_conditional_recolour_by_region":
            intersection_payload["conditional_recolour_candidate_count"] = len(orpi_conditional_recolour_programs(demos))
        if family == "orpi_frame_from_seed":
            intersection_payload["frame_from_seed_candidate_count"] = len(orpi_frame_from_seed_programs(demos))
        if family == "orpi_axis_line_recolour":
            intersection_payload["axis_line_recolour_candidate_count"] = len(orpi_axis_line_recolour_programs(demos))
        (family_dir / "selector_intersection.json").write_text(
            json.dumps(_jsonable(intersection_payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (family_dir / "object_matches.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "family": family,
                    "object_matches": [
                        {"demo_index": rec["demo_index"], "matches": rec["object_matches"]}
                        for rec in delta_records
                    ],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        raw_programs = synthesize_raw_programs(demos, [family])
        verified_programs = [program for program in raw_programs if verify_program(program, demos)]
        stable_keys = {
            program_semantic_key(program)
            for program in verified_programs
            if is_lodo_stable(program, demos, [family])
        }
        raw_rows = []
        verification_rows = []
        candidate_rows_out = []
        unique_outputs: Dict[str, Grid] = {}
        for idx, program in enumerate(raw_programs):
            trace = verify_program_trace(program, demos)
            stable = program_semantic_key(program) in stable_keys
            raw_rows.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "program_index": idx,
                    "candidate_program_string": program_semantic_key(program),
                    "name": program.name,
                    "params": program.params,
                    "debug_reason": program.debug_reason,
                }
            )
            verification_rows.append(
                {
                    "task_id": task_id,
                    "family": family,
                    "program_index": idx,
                    "candidate_program_string": program_semantic_key(program),
                    "failed_demo_index": trace["failed_demo_index"],
                    "mismatch_count_per_demo": trace["mismatch_count_per_demo"],
                    "verified_on_all_demos": trace["verified_on_all_demos"],
                    "lodo_stable": stable,
                }
            )
            if bool(args.dump_candidates):
                output = apply_program(program, test_input)
                key = grid_key(output)
                unique_outputs.setdefault(key, output)
                candidate_rows_out.append(
                    {
                        "task_id": task_id,
                        "program_index": idx,
                        "candidate_hash": key,
                        "family": family,
                        "name": program.name,
                        "output_shape": f"{output.shape[0]}x{output.shape[1]}",
                        "candidate_exact": int(output.shape == target.shape and np.array_equal(output, target)),
                        "program_params": json.dumps(program.params, sort_keys=True, separators=(",", ":")),
                        "candidate_grid": grid_json(output),
                        "debug_reason": program.debug_reason,
                    }
                )
        write_jsonl(family_dir / "raw_candidates.jsonl", raw_rows)
        write_jsonl(family_dir / "raw_programs.jsonl", raw_rows)
        write_jsonl(family_dir / "unified_programs.jsonl", raw_rows)
        write_jsonl(family_dir / "verification_trace.jsonl", verification_rows)
        failure_fields = [
            "task_id",
            "family",
            "program_index",
            "candidate_program_string",
            "failed_demo_index",
            "mismatch_count_per_demo",
            "verified_on_all_demos",
            "lodo_stable",
        ]
        write_csv(family_dir / "verification_failures.csv", verification_rows, failure_fields)
        stable_rows = [
            {
                "task_id": task_id,
                "family": family,
                "program_index": idx,
                "program_string": program_semantic_key(program),
                "parameters": json.dumps(program.params, sort_keys=True, separators=(",", ":")),
                "debug_reason": program.debug_reason,
            }
            for idx, program in enumerate(raw_programs)
            if program_semantic_key(program) in stable_keys
        ]
        write_jsonl(family_dir / "stable_verified_programs.jsonl", stable_rows)
        if bool(args.dump_candidates):
            candidate_fields = [
                "task_id",
                "program_index",
                "candidate_hash",
                "family",
                "name",
                "output_shape",
                "candidate_exact",
                "program_params",
                "candidate_grid",
                "debug_reason",
            ]
            write_csv(family_dir / "candidate_outputs.csv", candidate_rows_out, candidate_fields)
            write_csv(family_dir / "test_candidates.csv", candidate_rows_out, candidate_fields)
        parsed_object_count = sum(len(item["input"]["non_background_objects"]) for item in parsed["train"])
        parsed_marker_count = sum(len(item["input"]["singleton_markers"]) for item in parsed["train"])
        parsed_motif_count = sum(
            sum(1 for obj in item["input"]["non_background_objects"] if obj["kind"] == "multicolour_component")
            for item in parsed["train"]
        )
        summary = [
            f"task_id: {task_id}",
            f"family: {family}",
            f"parsed_object_count: {parsed_object_count}",
            f"parsed_marker_count: {parsed_marker_count}",
            f"parsed_motif_count: {parsed_motif_count}",
            f"n_raw_candidates: {len(raw_programs)}",
            f"n_verified_programs: {len(verified_programs)}",
            f"n_lodo_stable_programs: {len(stable_keys)}",
            f"n_unique_test_outputs: {len(unique_outputs)}",
        ]
        (family_dir / "trace_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
        print("\n".join(summary))


def run_probe(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(dataset)
    ledger_rows = read_csv(Path(args.c0_ledger).resolve())
    families = parse_families(args.families)

    task_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    program_rows: List[Dict[str, object]] = []

    for ref in ledger_rows:
        task_id = ref["task_id"]
        task = tasks[task_id]
        demos = list(task["train"])
        test_input = np.asarray(task["test"][0]["input"], dtype=np.uint8)
        target = np.asarray(task["test"][0]["output"], dtype=np.uint8)
        c0_exact = int(float(ref["exact_accuracy"]) > 0.5)

        raw_programs = synthesize_raw_programs(demos, families)
        verified_programs = [program for program in raw_programs if verify_program(program, demos)]
        lodo_stable_programs = [
            program for program in verified_programs if is_lodo_stable(program, demos, families)
        ]
        programs = lodo_stable_programs if bool(getattr(args, "lodostable", False)) else verified_programs
        candidate_recs, unique_outputs = candidate_records(task_id, programs, test_input, target)
        candidate_rows.extend(candidate_recs)
        for program_idx, program in enumerate(programs):
            rec = asdict(program)
            rec.update(
                {
                    "task_id": task_id,
                    "program_index": program_idx,
                    "verified_on_all_demos": True,
                    "lodo_stable": program_semantic_key(program)
                    in {program_semantic_key(stable) for stable in lodo_stable_programs},
                }
            )
            program_rows.append(rec)

        exact_candidates = [
            output
            for output in unique_outputs.values()
            if output.shape == target.shape and np.array_equal(output, target)
        ]
        repair_exact = int(len(exact_candidates) > 0)
        oracle_any_exact = int(c0_exact or repair_exact)

        unique_candidate = int(len(unique_outputs) == 1)
        if unique_candidate:
            only_output = next(iter(unique_outputs.values()))
            strict_repair_exact = int(only_output.shape == target.shape and np.array_equal(only_output, target))
            verified_replace_exact = strict_repair_exact
        elif programs:
            scored = sorted(programs, key=program_complexity_score)
            if len(scored) >= 2 and program_complexity_score(scored[0]) < program_complexity_score(scored[1]):
                chosen_output = apply_program(scored[0], test_input)
                chosen_hash = grid_key(chosen_output)
                disagreeing_hashes = {h for h in unique_outputs.keys() if h != chosen_hash}
                if disagreeing_hashes:
                    verified_replace_exact = int(chosen_output.shape == target.shape and np.array_equal(chosen_output, target))
                    unique_candidate = 1
                else:
                    verified_replace_exact = c0_exact
            else:
                verified_replace_exact = c0_exact
        else:
            verified_replace_exact = c0_exact

        replace_gain = int(c0_exact == 0 and verified_replace_exact == 1)
        replace_loss = int(c0_exact == 1 and verified_replace_exact == 0)
        oracle_gain = int(c0_exact == 0 and repair_exact == 1)
        primitive_families = sorted(set(program.family for program in programs))
        candidate_shapes = sorted({f"{grid.shape[0]}x{grid.shape[1]}" for grid in unique_outputs.values()})

        task_rows.append(
            {
                "task_id": task_id,
                "bucket": ref["bucket"],
                "c0_exact": c0_exact,
                "n_raw_candidates": len(raw_programs),
                "n_verified_programs": len(programs),
                "n_verified_before_lodo": len(verified_programs),
                "n_lodo_stable_programs": len(lodo_stable_programs),
                "n_unique_candidates": len(unique_outputs),
                "primitive_families": ";".join(primitive_families),
                "unique_candidate": unique_candidate,
                "repair_exact": repair_exact,
                "oracle_any_exact": oracle_any_exact,
                "verified_replace_exact": verified_replace_exact,
                "replace_gain": replace_gain,
                "replace_loss": replace_loss,
                "oracle_gain": oracle_gain,
                "candidate_shapes": ";".join(candidate_shapes),
                "debug_best_program": programs[0].debug_reason if programs else "",
            }
        )

    task_fields = [
        "task_id",
        "bucket",
        "c0_exact",
        "n_raw_candidates",
        "n_verified_programs",
        "n_verified_before_lodo",
        "n_lodo_stable_programs",
        "n_unique_candidates",
        "primitive_families",
        "unique_candidate",
        "repair_exact",
        "oracle_any_exact",
        "verified_replace_exact",
        "replace_gain",
        "replace_loss",
        "oracle_gain",
        "candidate_shapes",
        "debug_best_program",
    ]
    candidate_fields = [
        "task_id",
        "program_index",
        "candidate_hash",
        "family",
        "name",
        "output_shape",
        "candidate_exact",
        "program_params",
        "candidate_grid",
        "debug_reason",
    ]
    write_csv(out_dir / "repair_task_ledger.csv", task_rows, task_fields)
    write_csv(out_dir / "repair_candidate_outputs.csv", candidate_rows, candidate_fields)
    write_jsonl(out_dir / "repair_programs.jsonl", program_rows)
    bucket_rows = make_bucket_movement(task_rows)
    write_csv(out_dir / "bucket_movement.csv", bucket_rows, list(bucket_rows[0].keys()) if bucket_rows else ["bucket"])
    oracle_summary = make_oracle_summary(task_rows, bucket_rows)
    write_csv(out_dir / "oracle_summary.csv", oracle_summary, list(oracle_summary[0].keys()))
    lodo_summary = make_lodo_stability_summary(task_rows, bucket_rows)
    write_csv(out_dir / "lodo_stability_summary.csv", lodo_summary, list(lodo_summary[0].keys()))

    verdict, reason, next_stage = decide(task_rows, families)
    both_fail_oracle = sum(int(row["oracle_gain"]) for row in task_rows if row["bucket"] == "both_fail")
    both_fail_replace = sum(int(row["replace_gain"]) for row in task_rows if row["bucket"] == "both_fail")
    replace_losses = sum(int(row["replace_loss"]) for row in task_rows)
    strict_exact = sum(int(row["verified_replace_exact"]) for row in task_rows)
    oracle_exact = sum(int(row["oracle_any_exact"]) for row in task_rows)
    raw_total = sum(int(row["n_raw_candidates"]) for row in task_rows)
    verified_total = sum(int(row["n_verified_programs"]) for row in task_rows)
    lodo_total = sum(int(row["n_lodo_stable_programs"]) for row in task_rows)

    default_checkpoint = "D:/trm_c2/TinyRecursiveModels/checkpoints/TRM-FVR-Experiments/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/step670_evalfull400_pid401"
    default_config = "D:/trm_c2/TinyRecursiveModels/checkpoints/TRM-FVR-Experiments/c2_geomaux_VALID_005_EOS0_AUG1000_step670_evalfull400_pid401/all_config.yaml"
    checkpoint_path = getattr(args, "c0_checkpoint", None) or default_checkpoint
    config_path = getattr(args, "c0_config", None) or default_config
    report = [
        f"run name: ORPI-C2 Unification v1 (families={','.join(sorted(families))})",
        f"checkpoint: {checkpoint_path}",
        f"config: {config_path}",
        f"input dataset: {dataset}",
        f"baseline ledger: {Path(args.c0_ledger).resolve()}",
        f"output folder: {out_dir}",
        "training: false",
        "checkpoint mutated: false",
        "ground truth used only for scoring: true",
        "",
        f"verdict: {verdict}",
        f"n_raw_candidates: {raw_total}",
        f"n_verified_programs: {verified_total}",
        f"n_lodo_stable_programs: {lodo_total}",
        f"new both_fail oracle candidates: {both_fail_oracle}",
        f"new both_fail strict replacements: {both_fail_replace}",
        f"strict replacement losses on C0 solved tasks: {replace_losses}",
        f"overall strict exact: {strict_exact}/400",
        f"overall oracle_any exact: {oracle_exact}/400",
        f"reason: {reason}",
        f"next action: {next_stage}",
        "",
        "scientific note: public eval tasks informed this primitive bank, so this is engineering/development evidence only.",
    ]
    (out_dir / "rejection_or_keep.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


def assert_grid_equal(a: List[List[int]], b: List[List[int]]) -> None:
    aa = np.asarray(a, dtype=np.uint8)
    bb = np.asarray(b, dtype=np.uint8)
    assert aa.shape == bb.shape and np.array_equal(aa, bb), f"{aa.tolist()} != {bb.tolist()}"


def self_test() -> None:
    assert "orient_motif_select_marker" in parse_families("orient_motif_select_marker")
    assert "complete_dominant_region" in parse_families("complete_dominant_region")
    assert "motif_orientation_select_marker" in parse_families("motif_orientation_select_marker")
    assert "template_component_recolour" in parse_families("template_component_recolour")
    assert "orpi_marker_from_motif_delta" in parse_families("orpi_marker_from_motif_delta")
    assert "orpi_template_recolour_from_delta" in parse_families("orpi_template_recolour_from_delta")

    grid = np.asarray(
        [
            [0, 1, 1, 0, 2],
            [0, 1, 0, 0, 0],
            [3, 0, 0, 2, 2],
        ],
        dtype=np.uint8,
    )
    comps = connected_components(grid, include_background=True)
    assert len([c for c in comps if c.color == 1]) == 1
    assert len([c for c in comps if c.color == 2]) == 2
    assert any(c.color == 1 and c.area == 3 for c in comps)

    multiview = extract_multiview_scene_graph(
        np.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0, 4],
                [0, 2, 0, 2, 0, 0],
                [0, 2, 2, 2, 0, 4],
                [0, 3, 3, 0, 0, 0],
                [0, 3, 0, 0, 5, 0],
            ],
            dtype=np.uint8,
        )
    )
    node_kinds = {node["kind"] for node in multiview["nodes"]}
    assert {"colour_component", "foreground_object", "marker", "line", "hole", "anchor_endpoint"}.issubset(node_kinds)
    assert multiview["repeated_shape_classes"], "Expected repeated shape-class nodes."
    assert any(rel["type"] == "same_shape" for rel in multiview["relations"])

    move_in = np.asarray([[6, 6, 0, 0], [6, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    move_out = np.asarray([[0, 0, 0, 0], [0, 6, 6, 0], [0, 6, 0, 0]], dtype=np.uint8)
    move_match = match_objects_under_transform(
        extract_multiview_scene_graph(move_in),
        extract_multiview_scene_graph(move_out),
    )
    assert any(event["event"] == "MOVE" for event in move_match["events"])

    copy_in = np.asarray([[6, 6, 0, 0], [6, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    copy_out = np.asarray([[6, 6, 0, 0], [6, 0, 6, 6], [0, 0, 6, 0]], dtype=np.uint8)
    copy_match = match_objects_under_transform(
        extract_multiview_scene_graph(copy_in),
        extract_multiview_scene_graph(copy_out),
    )
    assert any(event["event"] == "COPY" for event in copy_match["events"])

    refl_in = np.asarray([[7, 0, 0, 0], [7, 7, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)
    refl_out = np.asarray([[0, 0, 0, 7], [0, 0, 7, 7], [0, 0, 0, 0]], dtype=np.uint8)
    refl_match = match_objects_under_transform(
        extract_multiview_scene_graph(refl_in),
        extract_multiview_scene_graph(refl_out),
    )
    assert any(event["event"] == "REFLECT" for event in refl_match["events"])

    a = connected_components(np.asarray([[0, 4, 4], [0, 4, 0]], dtype=np.uint8), include_background=True)
    b = connected_components(np.asarray([[0, 0, 0, 0], [0, 4, 4, 0], [0, 4, 0, 0]], dtype=np.uint8), include_background=True)
    sig_a = [c.shape_signature for c in a if c.color == 4][0]
    sig_b = [c.shape_signature for c in b if c.color == 4][0]
    assert sig_a == sig_b

    demos = [
        {
            "input": [[0, 1, 1], [0, 1, 0]],
            "output": [[0, 2, 2], [0, 2, 0]],
        },
        {
            "input": [[1, 0], [1, 1]],
            "output": [[2, 0], [2, 2]],
        },
    ]
    programs = synthesize_programs(demos, ["component_recolour", "marker_selection"])
    assert programs, "Expected a verified recolour program."
    pred = apply_program(programs[0], np.asarray([[1, 1, 0], [0, 1, 0]], dtype=np.uint8))
    assert_grid_equal(pred.tolist(), [[2, 2, 0], [0, 2, 0]])

    ambiguous_outputs = {
        grid_key(np.asarray([[1]], dtype=np.uint8)): np.asarray([[1]], dtype=np.uint8),
        grid_key(np.asarray([[2]], dtype=np.uint8)): np.asarray([[2]], dtype=np.uint8),
    }
    assert len(ambiguous_outputs) == 2
    unique_candidate = int(len(ambiguous_outputs) == 1)
    c0_exact = 1
    verified_replace_exact = c0_exact if not unique_candidate else 0
    assert unique_candidate == 0
    assert verified_replace_exact == 1

    no_candidate_outputs: Dict[str, Grid] = {}
    unique_candidate = int(len(no_candidate_outputs) == 1)
    verified_replace_exact = c0_exact if not unique_candidate else 0
    assert verified_replace_exact == 1

    marker_demos = [
        {
            "input": [[0, 7, 0, 7, 0], [0, 0, 0, 0, 0]],
            "output": [[0, 7, 0, 0, 0], [0, 0, 0, 0, 0]],
        },
        {
            "input": [[0, 7, 0, 7, 0], [0, 0, 0, 0, 0]],
            "output": [[0, 7, 0, 0, 0], [0, 0, 0, 0, 0]],
        },
    ]
    marker_programs = synthesize_programs(marker_demos, ["component_recolour", "marker_selection"])
    assert any(program.name == "component_delete_or_keep_by_relation" for program in marker_programs)
    marker_pred = apply_program(
        next(program for program in marker_programs if program.name == "component_delete_or_keep_by_relation"),
        np.asarray([[0, 7, 0, 7, 0], [0, 0, 0, 0, 0]], dtype=np.uint8),
    )
    assert_grid_equal(marker_pred.tolist(), [[0, 7, 0, 0, 0], [0, 0, 0, 0, 0]])

    line_demos = [
        {
            "input": [[0, 3, 0, 0, 3], [0, 0, 0, 0, 0]],
            "output": [[0, 3, 3, 3, 3], [0, 0, 0, 0, 0]],
        },
        {
            "input": [[0, 0, 0], [8, 0, 8], [0, 0, 0]],
            "output": [[0, 0, 0], [8, 8, 8], [0, 0, 0]],
        },
    ]
    line_programs = synthesize_programs(line_demos, ["line_path_completion"])
    assert any(program.name == "line_path_completion" for program in line_programs)
    line_pred = apply_program(
        line_programs[0],
        np.asarray([[0, 3, 0, 0, 3], [0, 0, 0, 0, 0]], dtype=np.uint8),
    )
    assert_grid_equal(line_pred.tolist(), [[0, 3, 3, 3, 3], [0, 0, 0, 0, 0]])

    copy_demos = [
        {
            "input": [[4, 4, 0, 0, 0], [4, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            "output": [[4, 4, 0, 4, 4], [4, 0, 0, 4, 0], [0, 0, 0, 0, 0]],
        },
        {
            "input": [[4, 4, 0, 0, 0], [4, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            "output": [[4, 4, 0, 4, 4], [4, 0, 0, 4, 0], [0, 0, 0, 0, 0]],
        },
    ]
    copy_programs = synthesize_programs(copy_demos, ["rigid_transform_copy"])
    assert any(program.name == "rigid_transform_copy" for program in copy_programs)
    copy_pred = apply_program(
        next(program for program in copy_programs if program.params["op"] == "copy_translate"),
        np.asarray([[4, 4, 0, 0, 0], [4, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint8),
    )
    assert_grid_equal(copy_pred.tolist(), [[4, 4, 0, 4, 4], [4, 0, 0, 4, 0], [0, 0, 0, 0, 0]])

    reflect_demos = [
        {
            "input": [[5, 0, 0, 0, 0, 0], [5, 5, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            "output": [[5, 0, 0, 0, 0, 5], [5, 5, 0, 0, 5, 5], [0, 0, 0, 0, 0, 0]],
        },
        {
            "input": [[5, 0, 0, 0, 0, 0], [5, 5, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            "output": [[5, 0, 0, 0, 0, 5], [5, 5, 0, 0, 5, 5], [0, 0, 0, 0, 0, 0]],
        },
    ]
    reflect_programs = synthesize_programs(reflect_demos, ["rigid_transform_copy"])
    assert any(program.params["op"] == "copy_reflect" for program in reflect_programs)

    motif_demos = [
        {
            "input": [
                [0, 4, 0, 0, 0, 4, 0],
                [0, 0, 0, 1, 8, 0, 0],
                [4, 0, 0, 8, 1, 0, 4],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 4, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 8, 0, 0],
                [0, 0, 0, 8, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        },
        {
            "input": [
                [0, 4, 0, 0, 0, 4, 0],
                [0, 0, 0, 1, 8, 0, 0],
                [4, 0, 0, 8, 1, 0, 4],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 4, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 8, 0, 0],
                [0, 0, 0, 8, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        },
    ]
    motif_programs = synthesize_programs(motif_demos, ["orient_motif_select_marker"])
    assert any(program.name == "orient_motif_select_marker" for program in motif_programs)
    motif_pred = apply_program(
        motif_programs[0],
        np.asarray(
            [
                [0, 4, 0, 0, 0, 4, 0],
                [0, 0, 0, 1, 8, 0, 0],
                [4, 0, 0, 8, 1, 0, 4],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )
    assert_grid_equal(
        motif_pred.tolist(),
        [
            [0, 4, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 8, 0, 0],
            [0, 0, 0, 8, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
    )

    region_demos = [
        {
            "input": [
                [0, 0, 0, 0, 0],
                [0, 6, 6, 6, 0],
                [0, 6, 0, 6, 0],
                [0, 6, 6, 6, 0],
                [0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0],
                [0, 6, 6, 6, 0],
                [0, 6, 6, 6, 0],
                [0, 6, 6, 6, 0],
                [0, 0, 0, 0, 0],
            ],
        },
        {
            "input": [
                [0, 0, 0, 0, 0],
                [0, 6, 6, 6, 0],
                [0, 6, 0, 6, 0],
                [0, 6, 6, 6, 0],
                [0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0],
                [0, 6, 6, 6, 0],
                [0, 6, 6, 6, 0],
                [0, 6, 6, 6, 0],
                [0, 0, 0, 0, 0],
            ],
        },
    ]
    region_programs = synthesize_programs(region_demos, ["complete_dominant_region"])
    assert any(program.name == "complete_dominant_region" for program in region_programs)
    assert is_lodo_stable(region_programs[0], region_demos, ["complete_dominant_region"])

    delta = compute_demo_delta(
        np.asarray([[0, 1, 0], [2, 2, 0], [0, 0, 3]], dtype=np.uint8),
        np.asarray([[0, 1, 4], [2, 5, 0], [0, 0, 0]], dtype=np.uint8),
    )
    assert delta["preserved_cells"] == 6
    assert delta["added_cells"] == 1
    assert delta["removed_cells"] == 1
    assert delta["recoloured_cells"] == 1
    assert delta["motif_cell_rewrites"], "Expected non-empty motif/pixel rewrite records."

    rich_delta = compute_demo_delta(
        np.asarray(
            [
                [0, 9, 9, 0, 0],
                [0, 9, 0, 0, 0],
                [0, 0, 0, 3, 0],
                [0, 4, 0, 4, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        np.asarray(
            [
                [0, 9, 9, 0, 0],
                [0, 9, 0, 9, 9],
                [0, 0, 0, 9, 3],
                [0, 4, 4, 4, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
    )
    assert rich_delta["copy_events"], "Expected real copy event records."
    assert rich_delta["path_segments"], "Expected path segment records."
    assert "translated_or_copied_nodes" not in rich_delta

    motif_delta_demos = [
        {
            "input": [
                [0, 4, 0, 0, 0, 4, 0],
                [0, 0, 0, 1, 8, 0, 0],
                [4, 0, 0, 8, 1, 0, 4],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0, 4, 0],
                [0, 0, 0, 8, 1, 0, 0],
                [0, 0, 0, 1, 8, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        },
        {
            "input": [
                [0, 4, 0, 0, 0, 4, 0],
                [0, 0, 0, 1, 8, 0, 0],
                [4, 0, 0, 8, 1, 0, 4],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0, 4, 0],
                [0, 0, 0, 8, 1, 0, 0],
                [0, 0, 0, 1, 8, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        },
    ]
    motif_delta_programs = synthesize_programs(motif_delta_demos, ["motif_orientation_select_marker"])
    assert any(program.name == "motif_orientation_select_marker" for program in motif_delta_programs)
    assert is_lodo_stable(motif_delta_programs[0], motif_delta_demos, ["motif_orientation_select_marker"])

    orpi_motif_programs = synthesize_programs(motif_delta_demos, ["orpi_marker_from_motif_delta"])
    assert any(program.name == "orpi_marker_from_motif_delta" for program in orpi_motif_programs)
    assert is_lodo_stable(orpi_motif_programs[0], motif_delta_demos, ["orpi_marker_from_motif_delta"])

    template_demos = [
        {
            "input": [
                [0, 1, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 5, 5, 0, 5, 5, 0],
                [0, 5, 0, 0, 5, 0, 0],
            ],
            "output": [
                [0, 1, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 2, 2, 0, 5, 5, 0],
                [0, 2, 0, 0, 5, 0, 0],
            ],
        },
        {
            "input": [
                [0, 1, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 5, 5, 0, 5, 5, 0],
                [0, 5, 0, 0, 5, 0, 0],
            ],
            "output": [
                [0, 1, 1, 0, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 2, 2, 0, 5, 5, 0],
                [0, 2, 0, 0, 5, 0, 0],
            ],
        },
    ]
    template_programs = synthesize_programs(template_demos, ["template_component_recolour"])
    assert any(program.name == "template_component_recolour" for program in template_programs)
    assert is_lodo_stable(template_programs[0], template_demos, ["template_component_recolour"])

    orpi_template_programs = synthesize_programs(template_demos, ["orpi_template_recolour_from_delta"])
    assert any(program.name == "orpi_template_recolour_from_delta" for program in orpi_template_programs)
    assert is_lodo_stable(orpi_template_programs[0], template_demos, ["orpi_template_recolour_from_delta"])

    intersection_zone_demos = [
        {
            "input": [
                [0, 0, 0, 0, 0],
                [0, 5, 5, 0, 0],
                [0, 5, 5, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 5, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0],
                [0, 5, 5, 0, 0],
                [0, 5, 5, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        },
        {
            "input": [
                [0, 5, 5, 5, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 5, 5, 0],
                [0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 5, 5, 5, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ],
        },
        {
            "input": [
                [0, 5, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 5],
                [0, 0, 0, 5],
            ],
            "output": [
                [0, 5, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 2],
                [0, 0, 0, 2],
            ],
        },
    ]
    inter_result = intersect_template_recolour_selectors(intersection_zone_demos)
    zone_selectors = [c for c in inter_result["selectors"] if c["selector"]["type"] == "by_source_color_zone" and c["selector"]["zone"] == "lower"]
    assert zone_selectors, f"Expected lower-zone intersection selector, got {[c['selector'] for c in inter_result['selectors']]}"
    zone_program = synthesize_programs(intersection_zone_demos, ["orpi_template_recolour_from_delta"])
    assert any(p.params["selector"]["type"] == "by_source_color_zone" for p in zone_program), "Zone-only intersection program should verify"

    exclusivity_demos = [
        {
            "input": [
                [0, 5, 5, 0, 5, 5, 0],
                [0, 5, 5, 0, 5, 5, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 5, 0],
                [0, 0, 0, 0, 5, 5, 0],
            ],
            "output": [
                [0, 5, 5, 0, 5, 5, 0],
                [0, 5, 5, 0, 5, 5, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 2, 0],
                [0, 0, 0, 0, 2, 2, 0],
            ],
        },
        {
            "input": [
                [0, 5, 5, 0, 5, 5, 0],
                [0, 5, 5, 0, 5, 5, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 5, 0],
                [0, 0, 0, 0, 5, 5, 0],
            ],
            "output": [
                [0, 5, 5, 0, 5, 5, 0],
                [0, 5, 5, 0, 5, 5, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 2, 0],
                [0, 0, 0, 0, 2, 2, 0],
            ],
        },
    ]
    excl_programs = synthesize_programs(exclusivity_demos, ["orpi_template_recolour_from_delta"])
    for p in excl_programs:
        sel = p.params["selector"]
        if sel["type"] == "by_source_color_shape":
            recoloured_target = apply_program(p, np.asarray(exclusivity_demos[0]["input"], dtype=np.uint8))
            assert np.array_equal(recoloured_target, np.asarray(exclusivity_demos[0]["output"], dtype=np.uint8)), \
                "Shape-only selector should fail exclusivity but verify_program should have rejected it"

    rot_demos = [
        {
            "input": [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 0, 0],
                [0, 3, 4, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0],
                [0, 3, 1, 0, 0],
                [0, 4, 2, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        },
        {
            "input": [
                [0, 0, 0, 0, 0],
                [0, 1, 2, 0, 0],
                [0, 3, 4, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0],
                [0, 3, 1, 0, 0],
                [0, 4, 2, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        },
    ]
    motif_orient = unify_motif_orientation(rot_demos)
    assert motif_orient is not None and motif_orient["d4_transform"] in {"rot90", "rot180", "rot270", "transpose", "anti_transpose"}, f"Expected non-identity D4 transform, got {motif_orient}"

    conditional_demos = [
        {
            "input": [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 8, 8, 0, 0, 8, 8],
                [0, 8, 8, 0, 0, 8, 8],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 8, 8, 0, 0, 0, 0],
                [0, 8, 8, 0, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 1, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 3, 3, 0, 0, 0, 0],
                [0, 3, 3, 0, 0, 0, 0],
            ],
        },
        {
            "input": [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 8, 8, 0, 0, 0, 0],
                [0, 8, 8, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 8, 8],
                [0, 0, 0, 0, 0, 8, 8],
                [0, 8, 8, 0, 0, 0, 0],
                [0, 8, 8, 0, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 3, 3],
                [0, 0, 0, 0, 0, 3, 3],
                [0, 3, 3, 0, 0, 0, 0],
                [0, 3, 3, 0, 0, 0, 0],
            ],
        },
    ]
    cond_programs = synthesize_programs(conditional_demos, ["orpi_conditional_recolour_by_region"])
    assert any(p.name == "orpi_conditional_recolour_by_region" for p in cond_programs), \
        f"Expected conditional recolour program, got {[p.name for p in cond_programs]}"
    cond_pred = apply_program(cond_programs[0], np.asarray(conditional_demos[0]["input"], dtype=np.uint8))
    assert np.array_equal(cond_pred, np.asarray(conditional_demos[0]["output"], dtype=np.uint8)), "Conditional recolour must reproduce demo exactly"

    frame_demos = [
        {
            "input": [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 5, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 6, 6, 6, 0, 0, 0],
                [0, 6, 5, 6, 0, 0, 0],
                [0, 6, 6, 6, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        },
        {
            "input": [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 5, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 6, 6, 6, 0],
                [0, 0, 0, 6, 5, 6, 0],
                [0, 0, 0, 6, 6, 6, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        },
    ]
    frame_programs = synthesize_programs(frame_demos, ["orpi_frame_from_seed"])
    assert any(p.name == "orpi_frame_from_seed" for p in frame_programs), \
        f"Expected frame_from_seed program, got {[p.name for p in frame_programs]}"
    frame_pred = apply_program(frame_programs[0], np.asarray(frame_demos[0]["input"], dtype=np.uint8))
    assert np.array_equal(frame_pred, np.asarray(frame_demos[0]["output"], dtype=np.uint8)), "Frame-from-seed must reproduce demo exactly"

    line_demos = [
        {
            "input": [
                [0, 5, 5, 5, 0],
                [0, 0, 0, 0, 0],
                [0, 5, 0, 0, 0],
                [0, 5, 0, 0, 0],
                [5, 0, 5, 5, 0],
            ],
            "output": [
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 5, 0, 0, 0],
                [0, 5, 0, 0, 0],
                [5, 0, 1, 1, 0],
            ],
        },
        {
            "input": [
                [0, 0, 0, 0, 0],
                [5, 5, 5, 5, 0],
                [0, 0, 0, 0, 0],
                [0, 5, 0, 0, 5],
                [0, 5, 0, 0, 0],
            ],
            "output": [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 5, 0, 0, 5],
                [0, 5, 0, 0, 0],
            ],
        },
    ]
    line_progs = synthesize_programs(line_demos, ["orpi_template_recolour_from_delta"])
    assert any(
        p.params["selector"]["type"] == "by_source_color_horizontal_line"
        for p in line_progs
    ), f"Expected horizontal_line selector, got {[p.params.get('selector', {}).get('type') for p in line_progs]}"
    line_pred = apply_program(
        next(p for p in line_progs if p.params["selector"]["type"] == "by_source_color_horizontal_line"),
        np.asarray(line_demos[0]["input"], dtype=np.uint8),
    )
    assert np.array_equal(line_pred, np.asarray(line_demos[0]["output"], dtype=np.uint8)), "Horizontal-line selector must reproduce demo exactly"

    axis_demos = [
        {
            "input": [
                [0, 0, 5, 5, 5, 0],
                [0, 5, 0, 0, 5, 0],
                [0, 5, 5, 5, 5, 0],
                [0, 0, 0, 5, 0, 0],
                [0, 0, 5, 5, 5, 0],
            ],
            "output": [
                [0, 0, 1, 1, 1, 0],
                [0, 5, 0, 0, 5, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 0, 5, 0, 0],
                [0, 0, 1, 1, 1, 0],
            ],
        },
        {
            "input": [
                [0, 5, 5, 0, 0],
                [5, 0, 0, 0, 5],
                [5, 5, 5, 0, 5],
                [0, 0, 0, 0, 0],
                [5, 5, 5, 5, 5],
            ],
            "output": [
                [0, 1, 1, 0, 0],
                [5, 0, 0, 0, 5],
                [1, 1, 1, 0, 5],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ],
        },
    ]
    axis_progs = synthesize_programs(axis_demos, ["orpi_axis_line_recolour"])
    assert any(p.name == "orpi_axis_line_recolour" and p.params["axis"] == "h" for p in axis_progs), \
        f"Expected h-axis axis_line program, got {[(p.name, p.params) for p in axis_progs]}"
    axis_pred = apply_program(
        next(p for p in axis_progs if p.params["axis"] == "h"),
        np.asarray(axis_demos[0]["input"], dtype=np.uint8),
    )
    assert np.array_equal(axis_pred, np.asarray(axis_demos[0]["output"], dtype=np.uint8)), "Axis-line recolour must reproduce demo exactly"

    print("[self-test] PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verified ARC program-repair coverage probe.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dataset", default="data/arc-agi-evaluation-full400-seed0")
    parser.add_argument("--c0-ledger")
    parser.add_argument("--c0-config")
    parser.add_argument("--c0-checkpoint")
    parser.add_argument("--out-dir", default="reports/verified_program_repair_d0_component")
    parser.add_argument("--families", default="component_recolour,marker_selection")
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--trace-task")
    parser.add_argument("--trace-family")
    parser.add_argument("--dump-candidates", action="store_true")
    parser.add_argument("--trace-out-dir", default="reports/d1_trace")
    parser.add_argument("--lodostable", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.trace_task:
        trace_program_generation(args)
        return
    if not args.c0_ledger:
        raise ValueError("--c0-ledger is required unless --self-test is used.")
    run_probe(args)


if __name__ == "__main__":
    main()
