import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.behavior_comparison import compare_behavior_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Compare RGB/MTL-Lite and behavior-only prediction CSV files.")
    parser.add_argument("--rgb-predictions", required=True, help="RGB/MTL-Lite prediction CSV path.")
    parser.add_argument("--behavior-predictions", required=True, help="Behavior-only prediction CSV path.")
    parser.add_argument("--output-dir", required=True, help="Directory for comparison CSV outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    comparison_path, summary_path = compare_behavior_predictions(
        rgb_prediction_csv=Path(args.rgb_predictions),
        behavior_prediction_csv=Path(args.behavior_predictions),
        output_dir=Path(args.output_dir),
    )
    print(f"[COMPARE] Per-sample comparison saved to: {comparison_path}")
    print(f"[COMPARE] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
