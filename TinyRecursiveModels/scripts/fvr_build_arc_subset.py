import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.build_arc_dataset import DataProcessConfig, convert_dataset


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _select_task_ids(input_file_prefix: str, subsets: list[str], subset_n: int, seed: int) -> dict[str, list[str]]:
    all_ids: list[tuple[str, str]] = []
    per_subset: dict[str, list[str]] = {}
    for subset in subsets:
        challenges_path = Path(f"{input_file_prefix}_{subset}_challenges.json")
        challenges = _load_json(challenges_path)
        task_ids = sorted(challenges.keys())
        per_subset[subset] = task_ids
        all_ids.extend((subset, task_id) for task_id in task_ids)

    if subset_n <= 0:
        raise ValueError("--subset-n must be positive")
    if subset_n > len(all_ids):
        raise ValueError(f"--subset-n={subset_n} exceeds available tasks={len(all_ids)}")

    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(len(all_ids), size=subset_n, replace=False)
    selected_pairs = [all_ids[int(i)] for i in selected_indices]
    selected_pairs.sort(key=lambda x: (subsets.index(x[0]), x[1]))

    selected: dict[str, list[str]] = {subset: [] for subset in subsets}
    for subset, task_id in selected_pairs:
        selected[subset].append(task_id)
    return selected


def _write_filtered_arc_files(input_file_prefix: str, tmp_prefix: Path, selected: dict[str, list[str]]):
    for subset, task_ids in selected.items():
        challenges_path = Path(f"{input_file_prefix}_{subset}_challenges.json")
        solutions_path = Path(f"{input_file_prefix}_{subset}_solutions.json")

        challenges = _load_json(challenges_path)
        filtered_challenges = {task_id: challenges[task_id] for task_id in task_ids}
        _write_json(Path(f"{tmp_prefix}_{subset}_challenges.json"), filtered_challenges)

        if solutions_path.exists():
            solutions = _load_json(solutions_path)
            filtered_solutions = {task_id: solutions[task_id] for task_id in task_ids if task_id in solutions}
            _write_json(Path(f"{tmp_prefix}_{subset}_solutions.json"), filtered_solutions)


def main():
    parser = argparse.ArgumentParser(description="Build a deterministic canonical num_aug=0 ARC subset for FVR confidence runs.")
    parser.add_argument("--input-file-prefix", required=True, help="Prefix before _training_challenges.json, e.g. kaggle/combined/arc-agi")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset-n", type=int, required=True)
    parser.add_argument("--subsets", nargs="+", default=["training"])
    parser.add_argument("--test-set-name", default="training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--puzzle-identifiers-start", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite to rebuild it.")
        shutil.rmtree(output_dir)

    selected = _select_task_ids(
        input_file_prefix=args.input_file_prefix,
        subsets=args.subsets,
        subset_n=args.subset_n,
        seed=args.seed,
    )

    with tempfile.TemporaryDirectory(prefix="fvr_arc_subset_") as tmp:
        tmp_prefix = Path(tmp) / "arc_subset"
        _write_filtered_arc_files(args.input_file_prefix, tmp_prefix, selected)
        convert_dataset(DataProcessConfig(
            input_file_prefix=str(tmp_prefix),
            output_dir=str(output_dir),
            subsets=args.subsets,
            test_set_name=args.test_set_name,
            seed=args.seed,
            num_aug=0,
            puzzle_identifiers_start=args.puzzle_identifiers_start,
        ))

    audit = {
        "input_file_prefix": args.input_file_prefix,
        "subset_n": args.subset_n,
        "seed": args.seed,
        "subsets": args.subsets,
        "test_set_name": args.test_set_name,
        "selected": selected,
        "task_ids": [task_id for subset in args.subsets for task_id in selected[subset]],
    }
    _write_json(output_dir / "subset_task_ids.json", audit)
    print(output_dir)


if __name__ == "__main__":
    main()
