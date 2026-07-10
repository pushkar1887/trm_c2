import argparse
import csv
import html
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def svg_text(x: int, y: int, text: object, size: int = 14, weight: str = "400", fill: str = "#111827") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Inter,Segoe UI,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(text)}</text>'
    )


def svg_rect(x: int, y: int, w: int, h: int, fill: str, stroke: str = "none", rx: int = 4) -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def summary_card(x: int, y: int, w: int, title: str, value: str, subtitle: str, color: str) -> str:
    return "\n".join(
        [
            svg_rect(x, y, w, 92, "#ffffff", "#d1d5db", 6),
            svg_rect(x, y, 7, 92, color, "none", 6),
            svg_text(x + 18, y + 24, title, 13, "700", "#374151"),
            svg_text(x + 18, y + 57, value, 28, "800", "#111827"),
            svg_text(x + 18, y + 78, subtitle, 12, "400", "#6b7280"),
        ]
    )


def bar_chart(
    x: int,
    y: int,
    title: str,
    items: List[tuple[str, int]],
    width: int = 560,
    bar_height: int = 22,
    color: str = "#2563eb",
) -> tuple[str, int]:
    max_value = max([value for _label, value in items] or [1])
    parts = [svg_text(x, y, title, 16, "800")]
    top = y + 24
    label_w = 250
    for idx, (label, value) in enumerate(items):
        yy = top + idx * (bar_height + 13)
        bw = int((width - label_w - 72) * value / max(max_value, 1))
        parts.append(svg_text(x, yy + 16, label, 12, "600", "#374151"))
        parts.append(svg_rect(x + label_w, yy, width - label_w - 72, bar_height, "#f3f4f6", "none", 4))
        parts.append(svg_rect(x + label_w, yy, bw, bar_height, color, "none", 4))
        parts.append(svg_text(x + width - 58, yy + 16, value, 12, "700", "#111827"))
    return "\n".join(parts), top + len(items) * (bar_height + 13)


def table_svg(x: int, y: int, rows: List[Dict[str, str]]) -> tuple[str, int]:
    headers = ["task_id", "wrong", "color", "eos", "valid", "outside", "failure_type"]
    widths = [92, 58, 58, 58, 58, 70, 315]
    parts = [svg_text(x, y, "Closest both_fail close misses by scored wrong tokens", 16, "800")]
    y += 18
    parts.append(svg_rect(x, y, sum(widths), 28, "#111827", "none", 4))
    cx = x
    for h, w in zip(headers, widths):
        parts.append(svg_text(cx + 8, y + 19, h, 12, "700", "#ffffff"))
        cx += w
    y += 28
    for idx, row in enumerate(rows):
        fill = "#ffffff" if idx % 2 == 0 else "#f9fafb"
        parts.append(svg_rect(x, y, sum(widths), 26, fill, "#e5e7eb", 0))
        values = [
            row["task_id"],
            row["wrong_label_tokens"],
            row["wrong_color_tokens"],
            row["wrong_eos_tokens"],
            row["wrong_valid_mask_tokens"],
            row["outside_color_fp_tokens"],
            row["failure_type"],
        ]
        cx = x
        for value, w in zip(values, widths):
            parts.append(svg_text(cx + 8, y + 18, value, 11, "500", "#111827"))
            cx += w
        y += 26
    return "\n".join(parts), y


def histogram_counts(values: List[int]) -> List[tuple[str, int]]:
    bins = [
        ("1", lambda x: x == 1),
        ("2", lambda x: x == 2),
        ("3", lambda x: x == 3),
        ("4-5", lambda x: 4 <= x <= 5),
        ("6-10", lambda x: 6 <= x <= 10),
        ("11-20", lambda x: 11 <= x <= 20),
        (">20", lambda x: x > 20),
    ]
    return [(name, sum(1 for value in values if pred(value))) for name, pred in bins]


def build_svg(summary_rows: List[Dict[str, str]], taxonomy_rows: List[Dict[str, str]], variant: str) -> str:
    both_fail_summary = [r for r in summary_rows if r["bucket"] == "both_fail"]
    selected = [r for r in both_fail_summary if r["variant"] == variant]
    if len(selected) != 1:
        raise ValueError(f"Expected one both_fail summary row for {variant}, got {len(selected)}")
    summary = selected[0]
    rows = [r for r in taxonomy_rows if r["bucket"] == "both_fail" and r["variant"] == variant]
    if not rows:
        raise ValueError(f"No both_fail taxonomy rows for {variant}")

    wrong_values = [int(r["wrong_label_tokens"]) for r in rows]
    failure_counts = Counter(r["failure_type"] for r in rows)
    failure_items = sorted(failure_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    token_items = [
        ("wrong color tokens", int(summary["sum_wrong_color_tokens"])),
        ("wrong EOS tokens", int(summary["sum_wrong_eos_tokens"])),
        ("wrong valid-mask tokens", int(summary["sum_wrong_valid_mask_tokens"])),
        ("outside colour FP tokens", int(summary["sum_outside_color_fp_tokens"])),
    ]
    closest = sorted(rows, key=lambda r: (int(r["wrong_label_tokens"]), r["task_id"]))[:16]

    width = 1340
    height = 1360
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, "#f8fafc", "none", 0),
        svg_text(34, 48, "Close-Miss Failure Taxonomy: both_fail only", 26, "850"),
        svg_text(34, 75, f"variant: {variant}", 14, "600", "#4b5563"),
        svg_text(34, 96, "Input: closemiss_failure_summary_by_variant_bucket.csv + eos0 taxonomy CSV", 12, "400", "#6b7280"),
    ]

    card_y = 122
    parts.append(summary_card(34, card_y, 220, "close-miss tasks", summary["close_miss_count"], "both_fail bucket", "#2563eb"))
    parts.append(summary_card(274, card_y, 220, "median wrong labels", summary["median_wrong_label_tokens"], "scored output tokens", "#7c3aed"))
    parts.append(summary_card(514, card_y, 220, "color-only failures", str(failure_counts.get("color_error_only", 0)), "exact-blocking content only", "#059669"))
    dominant_name, dominant_count = failure_items[0]
    parts.append(summary_card(754, card_y, 420, "dominant failure type", f"{dominant_count}/{len(rows)}", dominant_name, "#dc2626"))

    chart1, chart1_bottom = bar_chart(34, 270, "Failure-type distribution", failure_items, width=590, color="#dc2626")
    chart2, chart2_bottom = bar_chart(660, 270, "Token-error totals", token_items, width=560, color="#2563eb")
    parts.append(chart1)
    parts.append(chart2)

    hist, hist_bottom = bar_chart(34, max(chart1_bottom, chart2_bottom) + 50, "Wrong-label-token histogram", histogram_counts(wrong_values), width=590, color="#7c3aed")
    parts.append(hist)

    comp_y = max(chart1_bottom, chart2_bottom) + 50
    parts.append(svg_text(660, comp_y, "Variant comparison inside both_fail", 16, "800"))
    comp_y += 28
    comp_headers = ["variant", "close", "median", "color", "eos", "valid", "outside"]
    comp_widths = [235, 58, 66, 58, 70, 70, 74]
    parts.append(svg_rect(660, comp_y, sum(comp_widths), 28, "#111827", "none", 4))
    cx = 660
    for header, w in zip(comp_headers, comp_widths):
        parts.append(svg_text(cx + 8, comp_y + 19, header, 12, "700", "#ffffff"))
        cx += w
    comp_y += 28
    for idx, row in enumerate(both_fail_summary):
        fill = "#ffffff" if idx % 2 == 0 else "#f9fafb"
        parts.append(svg_rect(660, comp_y, sum(comp_widths), 28, fill, "#e5e7eb", 0))
        vals = [
            row["variant"],
            row["close_miss_count"],
            row["median_wrong_label_tokens"],
            row["sum_wrong_color_tokens"],
            row["sum_wrong_eos_tokens"],
            row["sum_wrong_valid_mask_tokens"],
            row["sum_outside_color_fp_tokens"],
        ]
        cx = 660
        for value, w in zip(vals, comp_widths):
            parts.append(svg_text(cx + 8, comp_y + 19, value, 11, "500", "#111827"))
            cx += w
        comp_y += 28

    table, table_bottom = table_svg(34, max(hist_bottom, comp_y) + 46, closest)
    parts.append(table)

    footer_y = table_bottom + 34
    parts.append(svg_text(34, footer_y, f"Both_fail close-miss count: {len(rows)} | median wrong labels: {median(wrong_values)} | <=1/<=2/<=3 wrong tokens: "
                          f"{sum(v <= 1 for v in wrong_values)}/{sum(v <= 2 for v in wrong_values)}/{sum(v <= 3 for v in wrong_values)}",
                          13, "700", "#374151"))
    parts.append(svg_text(34, footer_y + 22, "Interpretation: most both_fail close misses are not shape failures; they still carry large EOS/valid/outside-mask disagreements, while exact blockers include small valid-colour mistakes.", 12, "400", "#6b7280"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build close-miss failure taxonomy SVGs.")
    parser.add_argument("--report-dir", default="reports/closemiss_content_diagnostics_aug1000")
    parser.add_argument("--variant", default="eos0_VALID005_EOS0_AUG1000")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    summary_path = report_dir / "closemiss_failure_summary_by_variant_bucket.csv"
    taxonomy_path = report_dir / f"{args.variant}_closemiss_failure_taxonomy.csv"

    summary_rows = read_csv(summary_path)
    taxonomy_rows = read_csv(taxonomy_path)
    both_fail_summary = [r for r in summary_rows if r["bucket"] == "both_fail"]
    both_fail_taxonomy = [
        r for r in taxonomy_rows
        if r["bucket"] == "both_fail" and r["variant"] == args.variant
    ]

    write_csv(
        report_dir / "closemiss_failure_summary_by_variant_bucket_both_fail.csv",
        both_fail_summary,
        summary_rows[0].keys(),
    )
    write_csv(
        report_dir / f"{args.variant}_closemiss_failure_taxonomy_both_fail.csv",
        both_fail_taxonomy,
        taxonomy_rows[0].keys(),
    )

    svg = build_svg(summary_rows, taxonomy_rows, args.variant)
    svg_path = report_dir / f"{args.variant}_both_fail_closemiss_failure_taxonomy.svg"
    svg_path.write_text(svg, encoding="utf-8")
    md_path = report_dir / f"{args.variant}_both_fail_closemiss_failure_taxonomy.md"
    md_path.write_text(
        "\n".join(
            [
                f"# {args.variant} both_fail close-miss failure taxonomy",
                "",
                f"![both_fail taxonomy]({svg_path.name})",
                "",
                "Source slices:",
                f"- `closemiss_failure_summary_by_variant_bucket_both_fail.csv`",
                f"- `{args.variant}_closemiss_failure_taxonomy_both_fail.csv`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[write] {svg_path}")
    print(f"[write] {md_path}")


if __name__ == "__main__":
    main()
