"""Static SVG and HTML rendering for the ARC task atlas."""

from __future__ import annotations

from collections import Counter
import html
import json
import math
import textwrap
from typing import Any, Sequence

import numpy as np

from arc_task_atlas_analysis import TaskAnalysis, analysis_to_dict


ARC_PALETTE = (
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
)

FAMILY_COLORS = {
    "identity": "#64748b",
    "clean_recolor": "#15803d",
    "conditional_recolor": "#a16207",
    "dihedral": "#0369a1",
    "translate": "#0e7490",
    "tile": "#6d28d9",
    "size_change": "#be123c",
    "rearrangement": "#c2410c",
    "structural_other": "#7c3aed",
    "unknown": "#475569",
}


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(
    x: float,
    y: float,
    value: object,
    size: int = 13,
    weight: int = 400,
    fill: str = "#1f2937",
) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Segoe UI,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{_escape(value)}</text>'
    )


def svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str = "none",
    radius: float = 0,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'fill="{fill}" stroke="{stroke}" rx="{radius}"/>'
    )


def _wrapped_text(
    x: float,
    y: float,
    value: object,
    width: int,
    line_height: int = 18,
    size: int = 13,
    fill: str = "#374151",
    weight: int = 400,
) -> tuple[list[str], int]:
    lines = textwrap.wrap(str(value), width=width, break_long_words=False, break_on_hyphens=False) or [""]
    parts = [svg_text(x, y + i * line_height, line, size=size, weight=weight, fill=fill) for i, line in enumerate(lines)]
    return parts, len(lines) * line_height


def render_grid(grid: np.ndarray, x: int, y: int, cell: int, title: str) -> tuple[list[str], int, int]:
    height, width = grid.shape
    parts = [svg_text(x, y - 10, title, size=13, weight=700, fill="#334155")]
    parts.append(svg_rect(x - 1, y - 1, width * cell + 2, height * cell + 2, "#ffffff", "#475569", 1))
    for row in range(height):
        for col in range(width):
            color = ARC_PALETTE[int(grid[row, col])]
            parts.append(
                svg_rect(x + col * cell, y + row * cell, cell, cell, color, "#475569", 0)
            )
    return parts, width * cell, height * cell


def _palette_legend(x: int, y: int) -> list[str]:
    parts = [svg_text(x, y, "ARC palette", size=12, weight=700, fill="#475569")]
    cursor = x + 82
    for index, color in enumerate(ARC_PALETTE):
        parts.append(svg_rect(cursor, y - 13, 18, 18, color, "#64748b", 2))
        parts.append(svg_text(cursor + 6, y + 1, index, size=9, weight=700, fill="#ffffff" if index in {0, 1, 2, 6, 9} else "#111827"))
        cursor += 25
    return parts


def _format_evidence(evidence: dict[str, Any]) -> list[str]:
    if not evidence:
        return ["No additional detector parameters."]
    lines = []
    for key, value in evidence.items():
        if isinstance(value, dict):
            formatted = ", ".join(f"{source}->{target}" for source, target in value.items())
        elif isinstance(value, list):
            formatted = ", ".join(map(str, value))
        else:
            formatted = str(value)
        lines.append(f"{key.replace('_', ' ')}: {formatted}")
    return lines


def render_task_svg(result: TaskAnalysis) -> str:
    task = result.task
    max_dimension = max(
        max(example.input.grid.shape + example.output.grid.shape)
        for example in task.examples
    )
    cell = max(8, min(18, 390 // max_dimension))
    pair_heights = [
        max(example.input.grid.shape[0], example.output.grid.shape[0]) * cell + 70
        for example in task.examples
    ]
    width = 1440
    header_height = 188
    footer_height = 40
    height = header_height + sum(pair_heights) + footer_height
    family_color = FAMILY_COLORS.get(result.family, FAMILY_COLORS["unknown"])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, "#f8fafc"),
        svg_rect(0, 0, width, 88, "#0f172a"),
        svg_text(34, 42, f"Task {task.ordinal:04d}  |  {task.identifier}", size=25, weight=800, fill="#ffffff"),
        svg_text(34, 70, f"{len(task.examples)} stored demonstration pairs", size=14, weight=500, fill="#cbd5e1"),
        svg_rect(1015, 24, 380, 42, family_color, "none", 6),
        svg_text(1032, 51, f"{result.family}  /  {result.confidence}", size=17, weight=800, fill="#ffffff"),
    ]
    parts.extend(_palette_legend(34, 118))

    panel_x = 995
    parts.append(svg_rect(panel_x, 104, 405, height - 134, "#ffffff", "#cbd5e1", 6))
    panel_y = 138
    parts.append(svg_text(panel_x + 20, panel_y, "Operation evidence", size=16, weight=800))
    panel_y += 25
    wrapped, used = _wrapped_text(panel_x + 20, panel_y, result.description, 48)
    parts.extend(wrapped)
    panel_y += used + 8
    parts.append(svg_text(panel_x + 20, panel_y, f"Operation: {result.operation}", size=13, weight=700, fill=family_color))
    panel_y += 24
    for evidence_line in _format_evidence(result.evidence):
        wrapped, used = _wrapped_text(panel_x + 20, panel_y, evidence_line, 48, size=12)
        parts.extend(wrapped)
        panel_y += used + 3

    panel_y += 18
    parts.append(svg_text(panel_x + 20, panel_y, "Repository capability", size=16, weight=800))
    panel_y += 26
    for label, key in (("Existing", "existing"), ("Status", "support"), ("Add", "addition")):
        parts.append(svg_text(panel_x + 20, panel_y, label, size=12, weight=800, fill="#475569"))
        panel_y += 17
        wrapped, used = _wrapped_text(panel_x + 20, panel_y, result.capability[key], 48, size=12)
        parts.extend(wrapped)
        panel_y += used + 8

    y = header_height
    for pair_number, (example, stats, row_height) in enumerate(
        zip(task.examples, result.example_statistics, pair_heights), start=1
    ):
        parts.append(svg_text(34, y + 4, f"Demo pair {pair_number}", size=16, weight=800, fill="#0f172a"))
        grid_y = y + 30
        input_parts, input_width, _ = render_grid(example.input.grid, 34, grid_y, cell, "Input")
        parts.extend(input_parts)
        arrow_x = 34 + input_width + 28
        parts.append(svg_text(arrow_x, grid_y + 24, "->", size=22, weight=800, fill="#64748b"))
        output_x = arrow_x + 54
        output_parts, _, _ = render_grid(example.output.grid, output_x, grid_y, cell, "Output")
        parts.extend(output_parts)
        meta_y = y + row_height - 18
        changed = stats["changed_cells"] if stats["changed_cells"] is not None else "n/a"
        parts.append(
            svg_text(
                34,
                meta_y,
                f"input {stats['input_shape'][0]}x{stats['input_shape'][1]}  |  "
                f"output {stats['output_shape'][0]}x{stats['output_shape'][1]}  |  changed cells {changed}",
                size=12,
                fill="#64748b",
            )
        )
        parts.append(svg_rect(24, y + row_height - 2, 920, 1, "#e2e8f0"))
        y += row_height

    parts.append(
        svg_text(
            34,
            height - 16,
            "Confidence contract: proven = exact reconstruction across every pair; probable = broad structural evidence; unknown = insufficient evidence.",
            size=11,
            fill="#64748b",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_summary_svg(results: Sequence[TaskAnalysis]) -> str:
    family_counts = Counter(result.family for result in results)
    confidence_counts = Counter(result.confidence for result in results)
    example_count = sum(len(result.task.examples) for result in results)
    families = sorted(family_counts, key=lambda name: (-family_counts[name], name))
    width = 1180
    row_height = 36
    chart_y = 190
    height = chart_y + len(families) * row_height + 230
    maximum = max(family_counts.values(), default=1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, "#f8fafc"),
        svg_rect(0, 0, width, 104, "#0f172a"),
        svg_text(36, 48, "ARC Task Atlas Summary", size=28, weight=800, fill="#ffffff"),
        svg_text(36, 78, f"{len(results)} tasks  |  {example_count} stored demonstration pairs", size=15, weight=500, fill="#cbd5e1"),
        svg_text(36, 140, "Operation families", size=19, weight=800),
    ]
    y = chart_y
    for family in families:
        count = family_counts[family]
        bar_width = int(680 * count / maximum)
        parts.append(svg_text(36, y + 19, family, size=13, weight=700, fill="#334155"))
        parts.append(svg_rect(240, y, 700, 24, "#e2e8f0", radius=4))
        parts.append(svg_rect(240, y, bar_width, 24, FAMILY_COLORS.get(family, "#475569"), radius=4))
        parts.append(svg_text(956, y + 18, count, size=13, weight=800))
        y += row_height

    y += 28
    parts.append(svg_text(36, y, "Confidence", size=19, weight=800))
    y += 30
    confidence_colors = {"proven": "#15803d", "probable": "#a16207", "unknown": "#475569"}
    x = 36
    for confidence in ("proven", "probable", "unknown"):
        count = confidence_counts.get(confidence, 0)
        parts.append(svg_rect(x, y, 260, 74, "#ffffff", "#cbd5e1", 6))
        parts.append(svg_rect(x, y, 8, 74, confidence_colors[confidence], radius=6))
        parts.append(svg_text(x + 22, y + 28, confidence, size=13, weight=700, fill="#475569"))
        parts.append(svg_text(x + 22, y + 58, count, size=25, weight=800))
        x += 282
    footer_y = height - 52
    parts.append(
        svg_text(
            36,
            footer_y,
            "Methodology: proven labels exactly reconstruct every stored pair; probable labels are broad structural evidence, not semantic ARC ground truth.",
            size=12,
            fill="#64748b",
        )
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render_index_html(records: Sequence[dict[str, Any]]) -> str:
    families = sorted({record["family"] for record in records})
    confidences = sorted({record["confidence"] for record in records})
    data_json = json.dumps(list(records), ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    family_options = "".join(f'<option value="{_escape(name)}">{_escape(name)}</option>' for name in families)
    confidence_options = "".join(
        f'<option value="{_escape(name)}">{_escape(name)}</option>' for name in confidences
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ARC Task Atlas</title>
  <style>
    :root {{ color-scheme: light; font-family: Segoe UI, Arial, sans-serif; color: #172033; background: #eef2f4; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; }}
    header {{ background: #101827; color: white; padding: 20px 28px; display: flex; align-items: end; justify-content: space-between; gap: 24px; }}
    h1 {{ margin: 0; font-size: 25px; letter-spacing: 0; }}
    header p {{ margin: 6px 0 0; color: #cbd5e1; }}
    .summary-link {{ color: white; font-weight: 700; }}
    .toolbar {{ position: sticky; top: 0; z-index: 2; display: grid; grid-template-columns: minmax(220px,1fr) 220px 180px auto; gap: 10px; padding: 14px 28px; background: white; border-bottom: 1px solid #cbd5e1; }}
    input, select {{ width: 100%; min-height: 40px; border: 1px solid #aeb9c7; border-radius: 4px; padding: 8px 10px; background: white; font: inherit; }}
    #count {{ align-self: center; color: #526174; font-weight: 700; text-align: right; }}
    main {{ padding: 20px 28px 40px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill,minmax(310px,1fr)); gap: 12px; }}
    .task {{ display: grid; grid-template-columns: 92px 1fr; gap: 12px; min-height: 142px; background: white; border: 1px solid #cbd5e1; border-radius: 6px; overflow: hidden; color: inherit; text-decoration: none; }}
    .task:hover {{ border-color: #475569; box-shadow: 0 2px 8px #0f172a18; }}
    .preview {{ position: relative; background: #e8edf0; border-right: 1px solid #d4dce3; overflow: hidden; }}
    .preview img {{ width: 470px; max-width: none; transform: scale(.19); transform-origin: top left; }}
    .body {{ padding: 12px 12px 12px 0; min-width: 0; }}
    .id {{ font-weight: 800; margin-bottom: 7px; overflow-wrap: anywhere; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 8px; }}
    .badge {{ background: #e8edf0; color: #334155; border-radius: 3px; padding: 3px 6px; font-size: 11px; font-weight: 700; }}
    .description {{ color: #526174; font-size: 13px; line-height: 1.35; }}
    .empty {{ padding: 60px 10px; text-align: center; color: #64748b; }}
    @media (max-width: 760px) {{
      header {{ align-items: start; flex-direction: column; }}
      .toolbar {{ grid-template-columns: 1fr 1fr; padding: 12px; }}
      .toolbar input {{ grid-column: 1 / -1; }}
      #count {{ text-align: left; }}
      main {{ padding: 12px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <div><h1>ARC Task Atlas</h1><p>Processed aug-0 training tasks; every stored pair is shown as a demonstration.</p></div>
    <a class="summary-link" href="summary.svg">Open summary SVG</a>
  </header>
  <section class="toolbar">
    <input id="search" type="search" placeholder="Search task number or ARC identifier">
    <select id="family-filter"><option value="">All operation families</option>{family_options}</select>
    <select id="confidence-filter"><option value="">All confidence levels</option>{confidence_options}</select>
    <div id="count"></div>
  </section>
  <main><div id="tasks" class="grid"></div></main>
  <script>
    const records = {data_json};
    const tasks = document.getElementById('tasks');
    const search = document.getElementById('search');
    const family = document.getElementById('family-filter');
    const confidence = document.getElementById('confidence-filter');
    const count = document.getElementById('count');
    const esc = value => String(value).replace(/[&<>\"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[ch]));
    function render() {{
      const query = search.value.trim().toLowerCase();
      const visible = records.filter(record =>
        (!query || String(record.ordinal).includes(query) || record.identifier.toLowerCase().includes(query)) &&
        (!family.value || record.family === family.value) &&
        (!confidence.value || record.confidence === confidence.value));
      count.textContent = `${{visible.length}} / ${{records.length}} tasks`;
      tasks.innerHTML = visible.length ? visible.map(record => `
        <a class="task" href="${{esc(record.svg_path)}}">
          <div class="preview"><img loading="lazy" src="${{esc(record.svg_path)}}" alt=""></div>
          <div class="body">
            <div class="id">${{String(record.ordinal).padStart(4,'0')}} | ${{esc(record.identifier)}}</div>
            <div class="badges"><span class="badge">${{esc(record.family)}}</span><span class="badge">${{esc(record.confidence)}}</span><span class="badge">${{record.example_count}} pairs</span></div>
            <div class="description">${{esc(record.description)}}</div>
          </div>
        </a>`).join('') : '<div class="empty">No tasks match these filters.</div>';
    }}
    search.addEventListener('input', render);
    family.addEventListener('change', render);
    confidence.addEventListener('change', render);
    render();
  </script>
</body>
</html>"""
