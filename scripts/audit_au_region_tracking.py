import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.au_region_tracking import run_frame_contract_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the AU-T0a read-only frame contract inventory for aligned JPG "
            "frames and OpenFace CSV rows."
        )
    )
    parser.add_argument("--image-root", required=True, help="Root containing aligned frame directories.")
    parser.add_argument("--openface-root", required=True, help="Root containing per-video OpenFace CSV files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for AU-T0a tables and report.")
    parser.add_argument("--sample-step", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=2000)
    parser.add_argument("--sampling-strategy", default="stride_head")
    parser.add_argument(
        "--frame-id-regex",
        default=None,
        help="Optional regex used to extract a frame id from the JPG stem; first capture group wins.",
    )
    parser.add_argument("--join-threshold", type=float, default=0.995)
    parser.add_argument("--max-videos", type=int, default=None, help="Optional deterministic debug limit.")
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_frame_contract_audit(
        image_root=Path(args.image_root),
        openface_root=Path(args.openface_root),
        output_dir=Path(args.output_dir),
        sample_step=args.sample_step,
        max_seq_len=args.max_seq_len,
        sampling_strategy=args.sampling_strategy,
        frame_id_regex=args.frame_id_regex,
        join_threshold=args.join_threshold,
        max_videos=args.max_videos,
    )
    print("[AU-T0a] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
