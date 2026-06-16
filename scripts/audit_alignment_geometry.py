import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.alignment_geometry import run_alignment_geometry_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run offline diagnostics for OpenFace landmark alignment geometry and prediction errors."
    )
    parser.add_argument("--predictions", required=True, help="Path to test_predictions.csv.")
    parser.add_argument("--openface-root", required=True, help="Root directory containing OpenFace CSV files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for alignment geometry diagnostics.")
    parser.add_argument(
        "--frame-width",
        type=float,
        default=112.0,
        help="Aligned frame width used to normalize face center offsets and scale.",
    )
    parser.add_argument(
        "--frame-height",
        type=float,
        default=112.0,
        help="Aligned frame height used to normalize face center offsets and scale.",
    )
    parser.add_argument("--sample-step", type=int, default=1, help="Frame sampling step for OpenFace geometry summary.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum sampled frames per video.")
    return parser


def main():
    args = build_parser().parse_args()
    generated_files = run_alignment_geometry_audit(
        predictions_csv=Path(args.predictions),
        openface_root=Path(args.openface_root),
        output_dir=Path(args.output_dir),
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        sample_step=args.sample_step,
        max_frames=args.max_frames,
    )
    print("[ALIGNMENT_GEOMETRY_AUDIT] generated files:")
    for path in generated_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
