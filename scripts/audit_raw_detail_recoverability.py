#!/usr/bin/env python
"""Audit whether raw video retains detail lost in aligned JPG frames."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.raw_detail_recoverability import run_raw_detail_recoverability_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare same-frame raw-video and aligned-JPG face evidence using existing "
            "landmarks. This command never runs OpenFace or modifies source data."
        )
    )
    parser.add_argument("--frame-audit-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--raw-openface-root", required=True)
    parser.add_argument("--aligned-openface-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-integrity-comparison-summary", default=None)
    parser.add_argument("--video-id", action="append", default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--candidate-frames-per-video", type=int, default=32)
    parser.add_argument("--reference-frames-per-video", type=int, default=4)
    parser.add_argument("--contact-frames-per-video", type=int, default=8)
    parser.add_argument("--contact-sheet-columns", type=int, default=4)
    parser.add_argument("--calibration-quantile", type=float, default=0.95)
    parser.add_argument("--min-practical-clip-advantage", type=float, default=0.01)
    parser.add_argument("--min-transform-inlier-ratio", type=float, default=0.50)
    parser.add_argument("--max-transform-rmse-px", type=float, default=2.50)
    parser.add_argument("--ransac-threshold-px", type=float, default=2.0)
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_raw_detail_recoverability_audit(
        frame_audit_dir=Path(args.frame_audit_dir),
        dataset_root=Path(args.dataset_root),
        image_root=Path(args.image_root),
        raw_openface_root=Path(args.raw_openface_root),
        aligned_openface_root=Path(args.aligned_openface_root),
        output_dir=Path(args.output_dir),
        image_integrity_comparison_summary=(
            Path(args.image_integrity_comparison_summary)
            if args.image_integrity_comparison_summary
            else None
        ),
        video_ids=args.video_id,
        max_videos=args.max_videos,
        candidate_frames_per_video=args.candidate_frames_per_video,
        reference_frames_per_video=args.reference_frames_per_video,
        contact_frames_per_video=args.contact_frames_per_video,
        contact_sheet_columns=args.contact_sheet_columns,
        calibration_quantile=args.calibration_quantile,
        min_practical_clip_advantage=args.min_practical_clip_advantage,
        min_transform_inlier_ratio=args.min_transform_inlier_ratio,
        max_transform_rmse_px=args.max_transform_rmse_px,
        ransac_threshold_px=args.ransac_threshold_px,
        project_root=PROJECT_ROOT,
    )
    print("[RAW_DETAIL_AUDIT] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
