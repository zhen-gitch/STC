import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.black_artifacts import run_black_artifact_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run offline diagnostics for black-padding artifacts in OpenFace aligned frames."
    )
    parser.add_argument("--predictions", required=True, help="Path to test_predictions.csv.")
    parser.add_argument("--image-root", required=True, help="Root directory containing aligned video frame folders.")
    parser.add_argument("--output-dir", required=True, help="Output directory for black artifact diagnostics.")
    parser.add_argument(
        "--black-threshold",
        type=int,
        default=8,
        help="Pixel is black when all RGB channels are <= this threshold.",
    )
    parser.add_argument(
        "--border-fraction",
        type=float,
        default=0.15,
        help="Outer image fraction treated as border for border black-pixel statistics.",
    )
    parser.add_argument("--sample-step", type=int, default=10, help="Frame sampling step for artifact summary.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum sampled frames per video.")
    return parser


def main():
    args = build_parser().parse_args()
    generated_files = run_black_artifact_audit(
        predictions_csv=Path(args.predictions),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        black_threshold=args.black_threshold,
        border_fraction=args.border_fraction,
        sample_step=args.sample_step,
        max_frames=args.max_frames,
    )
    print("[BLACK_ARTIFACT_AUDIT] generated files:")
    for path in generated_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
