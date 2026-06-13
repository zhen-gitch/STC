import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.openface_quality import DEFAULT_LOW_CONFIDENCE_THRESHOLD
from src.diagnostics.shortcut_audit import run_shortcut_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run offline Shortcut Audit diagnostics for OpenFace aligned AVEC runs."
    )
    parser.add_argument("--predictions", required=True, help="Path to predictions.csv.")
    parser.add_argument("--openface-root", required=True, help="OpenFace CSV file or root directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for shortcut audit diagnostics.")
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        help="Frames below this confidence are counted as low-confidence frames.",
    )
    parser.add_argument(
        "--max-heatmap-features",
        type=int,
        default=40,
        help="Maximum shortcut features to include in the correlation heatmap.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    generated_files = run_shortcut_audit(
        predictions_csv=Path(args.predictions),
        openface_root=Path(args.openface_root),
        output_dir=Path(args.output_dir),
        low_confidence_threshold=args.low_confidence_threshold,
        max_heatmap_features=args.max_heatmap_features,
    )
    print("[SHORTCUT_AUDIT] generated files:")
    for path in generated_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
