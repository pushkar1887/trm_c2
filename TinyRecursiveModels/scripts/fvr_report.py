import argparse
import json
from pathlib import Path


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Create a short markdown report from FVR diagnostic JSON.")
    parser.add_argument("--a5-json", default=None)
    parser.add_argument("--step-json", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--decision", default="inconclusive")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    lines = ["# FVR/TRM Experiment Report", ""]
    lines.append(f"- decision: `{args.decision}`")
    if args.notes:
        lines.append(f"- notes: {args.notes}")

    if args.a5_json:
        data = json.loads(Path(args.a5_json).read_text(encoding="utf-8"))
        lines.extend([
            "",
            "## A5 Real vs Blank PID",
            f"- config: `{data.get('config')}`",
            f"- checkpoint: `{data.get('checkpoint')}`",
            f"- real exact: `{_fmt(data.get('real_pid', {}).get('exact_accuracy'))}`",
            f"- blank exact: `{_fmt(data.get('blank_pid', {}).get('exact_accuracy'))}`",
            f"- delta exact: `{_fmt(data.get('delta_exact_accuracy'))}`",
            f"- real token/content accuracy: `{_fmt(data.get('real_pid', {}).get('accuracy'))}`",
            f"- blank token/content accuracy: `{_fmt(data.get('blank_pid', {}).get('accuracy'))}`",
        ])

    if args.step_json:
        data = json.loads(Path(args.step_json).read_text(encoding="utf-8"))
        lines.extend(["", "## Step Saturation", f"- config: `{data.get('config')}`"])
        steps = data.get("steps", {})
        for step in sorted(steps, key=lambda x: int(x)):
            row = steps[step]
            lines.append(
                f"- step {step}: exact=`{_fmt(row.get('exact_accuracy'))}`, "
                f"accuracy=`{_fmt(row.get('accuracy'))}`, "
                f"unique=`{_fmt(row.get('pred_unique_classes'))}`"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
