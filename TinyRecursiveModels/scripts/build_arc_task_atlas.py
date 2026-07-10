"""Build the static SVG/HTML atlas for processed ARC training tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

from arc_task_atlas_analysis import TaskAnalysis, analysis_to_dict, analyze_task, load_dataset
from arc_task_atlas_render import render_index_html, render_summary_svg, render_task_svg


def _safe_identifier(identifier: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier).strip("._")
    return cleaned or "task"


def write_atlas(analyses: tuple[TaskAnalysis, ...], output_dir: str | Path) -> list[dict[str, object]]:
    """Write a complete atlas into an empty or new output directory."""
    output_dir = Path(output_dir)
    task_dir = output_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for analysis in analyses:
        filename = f"{analysis.task.ordinal:04d}_{_safe_identifier(analysis.task.identifier)}.svg"
        relative_svg = f"tasks/{filename}"
        svg = render_task_svg(analysis)
        ET.fromstring(svg)
        (task_dir / filename).write_text(svg, encoding="utf-8")
        records.append(analysis_to_dict(analysis, relative_svg))

    (output_dir / "analysis.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    summary = render_summary_svg(analyses)
    ET.fromstring(summary)
    (output_dir / "summary.svg").write_text(summary, encoding="utf-8")
    (output_dir / "index.html").write_text(render_index_html(records), encoding="utf-8")
    return records


def validate_atlas(output_dir: str | Path, expected_tasks: int, expected_examples: int) -> dict[str, int]:
    output_dir = Path(output_dir)
    records = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
    if len(records) != expected_tasks:
        raise ValueError(f"analysis contains {len(records)} tasks, expected {expected_tasks}")
    example_count = sum(int(record["example_count"]) for record in records)
    if example_count != expected_examples:
        raise ValueError(f"analysis contains {example_count} examples, expected {expected_examples}")
    svg_paths = sorted((output_dir / "tasks").glob("*.svg"))
    if len(svg_paths) != expected_tasks:
        raise ValueError(f"found {len(svg_paths)} task SVGs, expected {expected_tasks}")
    for svg_path in svg_paths:
        ET.parse(svg_path)
    ET.parse(output_dir / "summary.svg")
    if not (output_dir / "index.html").is_file():
        raise ValueError("index.html is missing")
    return {"tasks": len(records), "examples": example_count, "svgs": len(svg_paths)}


def build_atomic(dataset_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    dataset_dir = Path(dataset_dir).resolve()
    output_dir = Path(output_dir).resolve()
    temporary = output_dir.with_name(output_dir.name + ".tmp")
    backup = output_dir.with_name(output_dir.name + ".previous")
    if temporary.exists():
        shutil.rmtree(temporary)
    tasks = load_dataset(dataset_dir)
    analyses = tuple(analyze_task(task) for task in tasks)
    write_atlas(analyses, temporary)
    result = validate_atlas(
        temporary,
        expected_tasks=len(tasks),
        expected_examples=sum(len(task.examples) for task in tasks),
    )
    if backup.exists():
        shutil.rmtree(backup)
    if output_dir.exists():
        output_dir.rename(backup)
    try:
        temporary.rename(output_dir)
    except Exception:
        if backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/arc1concept-aug-0/train"),
        help="Processed ARC split containing all__*.npy arrays.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/arc_task_atlas"),
        help="Completed static atlas directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_atomic(args.dataset, args.output)
    print(
        f"tasks={result['tasks']} examples={result['examples']} "
        f"svgs={result['svgs']} decode_errors=0"
    )
    print(f"index={args.output.resolve() / 'index.html'}")


if __name__ == "__main__":
    main()
