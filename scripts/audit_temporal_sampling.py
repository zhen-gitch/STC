import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.temporal_sampling import run_temporal_sampling_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run offline diagnostics for video length, temporal sampling, truncation, and padding."
    )
    parser.add_argument("--predictions", required=True, help="Path to test_predictions.csv.")
    parser.add_argument("--image-root", required=True, help="Root directory containing aligned video frame folders.")
    parser.add_argument("--output-dir", required=True, help="Output directory for temporal sampling diagnostics.")
    parser.add_argument(
        "--sample-step",
        type=int,
        default=10,
        help="Dataset PROCESS_TEMPORAL.SAMPLE_STEP used during training/evaluation.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2000,
        help="Dataset PROCESS_TEMPORAL.MAX_SEQ_LEN used during training/evaluation.",
    )
    parser.add_argument(
        "--sampling-strategy",
        default="stride_head",
        help="Dataset PROCESS_TEMPORAL.SAMPLING_STRATEGY used during training/evaluation.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    generated_files = run_temporal_sampling_audit(
        predictions_csv=Path(args.predictions),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        sample_step=args.sample_step,
        max_seq_len=args.max_seq_len,
        sampling_strategy=args.sampling_strategy,
    )
    print("[TEMPORAL_SAMPLING_AUDIT] generated files:")
    for path in generated_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
