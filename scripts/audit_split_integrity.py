import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.split_integrity import run_split_integrity_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit dataset split integrity, subject leakage, labels, image folders, and prediction alignment."
    )
    parser.add_argument("--split-file", required=True, help="Path to train/val/test split JSON.")
    parser.add_argument("--label-dir", required=True, help="Directory containing *_Depression.csv label files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for split integrity diagnostics.")
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional root directory containing aligned image folders. Enables image-folder existence checks.",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Optional prediction CSV, usually test_predictions.csv, to validate split alignment.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    generated_files = run_split_integrity_audit(
        split_file=Path(args.split_file),
        label_dir=Path(args.label_dir),
        output_dir=Path(args.output_dir),
        image_root=Path(args.image_root) if args.image_root else None,
        predictions_csv=Path(args.predictions) if args.predictions else None,
    )
    print("[SPLIT_INTEGRITY_AUDIT] generated files:")
    for path in generated_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
