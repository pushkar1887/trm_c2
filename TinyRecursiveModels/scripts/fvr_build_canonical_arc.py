import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.build_arc_dataset import DataProcessConfig, convert_dataset


def main():
    parser = argparse.ArgumentParser(description="Build a canonical num_aug=0 ARC dataset for clean FVR/TRM measurements.")
    parser.add_argument("--input-file-prefix", required=True, help="Prefix before _training_challenges.json, e.g. kaggle/combined/arc-agi")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subsets", nargs="+", default=["training"])
    parser.add_argument("--test-set-name", default="training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--puzzle-identifiers-start", type=int, default=1)
    args = parser.parse_args()

    convert_dataset(DataProcessConfig(
        input_file_prefix=args.input_file_prefix,
        output_dir=args.output_dir,
        subsets=args.subsets,
        test_set_name=args.test_set_name,
        seed=args.seed,
        num_aug=0,
        puzzle_identifiers_start=args.puzzle_identifiers_start,
    ))
    print(args.output_dir)


if __name__ == "__main__":
    main()
