#!/usr/bin/env python
"""Generate non-authorizing before/after exposure contact sheets."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.frame_recovery import write_exposure_previews


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Preview audited log/inverse-log exposure mappings as BEFORE/AFTER contact sheets. "
            "This command does not modify source images, materialize a dataset, or approve the "
            "exposure review gate."
        )
    )
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--image-integrity-comparison-summary",
        default=None,
        help=(
            "Required when --image-root differs from the audited root. Must be an EXACT_PASS "
            "comparison whose candidate inventory records this relocated root."
        ),
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=None,
        help="Preview one audited exposure candidate; repeat for multiple videos. Default: all candidates.",
    )
    parser.add_argument("--frames-per-video", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=224)
    parser.add_argument(
        "--underexposed-target-luma",
        type=float,
        default=None,
        help="Preview-only target override; does not change the frozen audit manifest.",
    )
    parser.add_argument(
        "--overexposed-target-luma",
        type=float,
        default=None,
        help="Preview-only target override; does not change the frozen audit manifest.",
    )
    parser.add_argument("--safe-low-luma", type=float, default=None)
    parser.add_argument("--safe-high-luma", type=float, default=None)
    parser.add_argument(
        "--scan-all-frames",
        action="store_true",
        help="Evaluate every visible frame while keeping contact sheets limited to representative frames.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    generated = write_exposure_previews(
        audit_dir=Path(args.audit_dir),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        image_integrity_comparison_summary=(
            Path(args.image_integrity_comparison_summary)
            if args.image_integrity_comparison_summary
            else None
        ),
        video_ids=args.video_id,
        frames_per_video=args.frames_per_video,
        columns=args.columns,
        thumb_width=args.thumb_width,
        underexposed_target_luma=args.underexposed_target_luma,
        overexposed_target_luma=args.overexposed_target_luma,
        safe_low_luma=args.safe_low_luma,
        safe_high_luma=args.safe_high_luma,
        scan_all_frames=args.scan_all_frames,
        project_root=PROJECT_ROOT,
    )
    print("[EXPOSURE_PREVIEW] generated files:")
    for path in generated:
        print(f"  - {path}")
    print(f"  - {Path(args.output_dir).expanduser().resolve() / 'contact_sheets'}")


if __name__ == "__main__":
    main()
