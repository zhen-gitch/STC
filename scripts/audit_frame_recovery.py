#!/usr/bin/env python
"""Audit existing OpenFace failures and optionally materialize RGB repairs."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.frame_recovery import (
    FrameRecoveryPolicy,
    materialize_frame_repairs,
    run_frame_failure_audit,
)
from src.diagnostics.raw_frame_warp import run_raw_frame_warp_smoke


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create a frame-level OpenFace failure/exposure manifest, then optionally "
            "materialize non-destructive RGB repairs. No FaceLandmark/OpenFace process "
            "is launched by this script."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Read existing CSV/JPG files and write repair planning tables.")
    audit.add_argument("--image-root", required=True)
    audit.add_argument("--openface-root", required=True, help="Existing aligned-image OpenFace CSV root.")
    audit.add_argument("--dataset-split-file", required=True)
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--workers", type=int, default=None)
    audit.add_argument("--max-videos", type=int, default=None, help="Deterministic debug limit.")
    audit.add_argument("--black-threshold", type=int, default=8)
    audit.add_argument("--pure-black-max-nonblack-ratio", type=float, default=0.01)
    audit.add_argument("--pure-black-max-visible-ratio", type=float, default=0.01)
    audit.add_argument("--pure-black-max-mean-luma", type=float, default=1.0)
    audit.add_argument("--underexposed-frame-median", type=float, default=35.0)
    audit.add_argument("--underexposed-frame-q90", type=float, default=90.0)
    audit.add_argument("--overexposed-frame-median", type=float, default=220.0)
    audit.add_argument("--overexposed-frame-q90", type=float, default=250.0)
    audit.add_argument("--exposure-sample-frames", type=int, default=32)
    audit.add_argument("--exposure-video-ratio", type=float, default=0.50)
    audit.add_argument("--exposure-safe-low-quantile", type=float, default=0.20)
    audit.add_argument("--exposure-target-low-quantile", type=float, default=0.25)
    audit.add_argument("--exposure-target-high-quantile", type=float, default=0.75)
    audit.add_argument("--exposure-safe-high-quantile", type=float, default=0.80)
    audit.add_argument("--max-exposure-curve-parameter", type=float, default=32.0)
    audit.add_argument("--max-optical-flow-gap", type=int, default=3)
    audit.add_argument("--max-copy-gap", type=int, default=1)

    materialize = subparsers.add_parser(
        "materialize",
        help="Write repaired frames to a separate derived root using the frozen audit policy.",
    )
    materialize.add_argument("--audit-dir", required=True)
    materialize.add_argument("--image-root", required=True)
    materialize.add_argument("--output-root", required=True)
    materialize.add_argument(
        "--source-review-manifest",
        required=True,
        help=(
            "Completed REVIEWED segment manifest from audit_source_presence.py. "
            "Aligned-neighbor black-frame synthesis remains disabled."
        ),
    )
    materialize.add_argument(
        "--exposure-review-manifest",
        required=True,
        help=(
            "Completed REVIEWED exposure segment manifest derived from "
            "tables/exposure_review_template.csv. PENDING or uncovered candidate intervals abort."
        ),
    )
    materialize.add_argument(
        "--layout",
        choices=("sparse_overlay", "mirror"),
        default="sparse_overlay",
        help="Sparse writes only changed frames; mirror also links/copies every unchanged frame.",
    )
    materialize.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How mirror layout represents unchanged files.",
    )
    materialize.add_argument("--max-videos", type=int, default=None, help="Deterministic sparse debug limit.")
    materialize.add_argument(
        "--video-id",
        action="append",
        default=None,
        help="Materialize only this audited video; repeat for multiple videos.",
    )

    raw_warp = subparsers.add_parser(
        "raw-warp-smoke",
        help="Warp 1-3 real raw-video frames into aligned space for review; never installs them.",
    )
    raw_warp.add_argument("--audit-dir", required=True)
    raw_warp.add_argument("--dataset-root", required=True, help="Raw AVEC2014 video root.")
    raw_warp.add_argument("--image-root", required=True, help="Byte-audited aligned JPG root.")
    raw_warp.add_argument("--source-presence-gate", required=True)
    raw_warp.add_argument("--raw-openface-root", required=True)
    raw_warp.add_argument("--aligned-openface-root", required=True)
    raw_warp.add_argument("--output-dir", required=True)
    raw_warp.add_argument("--image-integrity-comparison-summary", default=None)
    raw_warp.add_argument("--max-frames", type=int, default=3)
    raw_warp.add_argument("--split", default="train", choices=("train",))
    raw_warp.add_argument(
        "--target-frame",
        action="append",
        default=None,
        help="Optional explicit VIDEO_ID:FRAME_ID; repeat for up to three reviewed targets.",
    )
    raw_warp.add_argument("--max-anchor-gap", type=int, default=8)
    raw_warp.add_argument("--max-fb-error-px", type=float, default=3.0)
    raw_warp.add_argument("--min-tracking-valid-ratio", type=float, default=0.60)
    raw_warp.add_argument("--min-transform-inlier-ratio", type=float, default=0.50)
    raw_warp.add_argument("--max-transform-rmse-px", type=float, default=2.50)
    raw_warp.add_argument("--ransac-threshold-px", type=float, default=2.0)
    raw_warp.add_argument("--max-transform-disagreement-px", type=float, default=5.0)
    raw_warp.add_argument("--min-warped-landmark-in-bounds-ratio", type=float, default=0.80)
    return parser


def _policy_from_args(args):
    return FrameRecoveryPolicy(
        black_threshold=args.black_threshold,
        pure_black_max_nonblack_ratio=args.pure_black_max_nonblack_ratio,
        pure_black_max_visible_ratio=args.pure_black_max_visible_ratio,
        pure_black_max_mean_luma=args.pure_black_max_mean_luma,
        underexposed_frame_median=args.underexposed_frame_median,
        underexposed_frame_q90=args.underexposed_frame_q90,
        overexposed_frame_median=args.overexposed_frame_median,
        overexposed_frame_q90=args.overexposed_frame_q90,
        exposure_sample_frames=args.exposure_sample_frames,
        exposure_video_ratio=args.exposure_video_ratio,
        exposure_safe_low_quantile=args.exposure_safe_low_quantile,
        exposure_reference_low_quantile=args.exposure_target_low_quantile,
        exposure_reference_high_quantile=args.exposure_target_high_quantile,
        exposure_safe_high_quantile=args.exposure_safe_high_quantile,
        max_exposure_curve_parameter=args.max_exposure_curve_parameter,
        max_optical_flow_gap=args.max_optical_flow_gap,
        max_copy_gap=args.max_copy_gap,
    )


def main():
    args = build_parser().parse_args()
    if args.command == "audit":
        generated = run_frame_failure_audit(
            image_root=Path(args.image_root),
            openface_root=Path(args.openface_root),
            dataset_split_file=Path(args.dataset_split_file),
            output_dir=Path(args.output_dir),
            policy=_policy_from_args(args),
            workers=args.workers,
            max_videos=args.max_videos,
            project_root=PROJECT_ROOT,
        )
        prefix = "FRAME_RECOVERY_AUDIT"
    elif args.command == "materialize":
        generated = materialize_frame_repairs(
            audit_dir=Path(args.audit_dir),
            image_root=Path(args.image_root),
            output_root=Path(args.output_root),
            source_review_manifest=Path(args.source_review_manifest),
            exposure_review_manifest=Path(args.exposure_review_manifest),
            layout=args.layout,
            link_mode=args.link_mode,
            max_videos=args.max_videos,
            video_ids=args.video_id,
            project_root=PROJECT_ROOT,
        )
        prefix = "FRAME_RECOVERY_MATERIALIZE"
    else:
        generated = run_raw_frame_warp_smoke(
            audit_dir=Path(args.audit_dir),
            dataset_root=Path(args.dataset_root),
            image_root=Path(args.image_root),
            source_presence_gate=Path(args.source_presence_gate),
            raw_openface_root=Path(args.raw_openface_root),
            aligned_openface_root=Path(args.aligned_openface_root),
            output_dir=Path(args.output_dir),
            image_integrity_comparison_summary=(
                Path(args.image_integrity_comparison_summary)
                if args.image_integrity_comparison_summary
                else None
            ),
            max_frames=args.max_frames,
            split=args.split,
            target_frames=args.target_frame,
            max_anchor_gap=args.max_anchor_gap,
            max_fb_error_px=args.max_fb_error_px,
            min_tracking_valid_ratio=args.min_tracking_valid_ratio,
            min_transform_inlier_ratio=args.min_transform_inlier_ratio,
            max_transform_rmse_px=args.max_transform_rmse_px,
            ransac_threshold_px=args.ransac_threshold_px,
            max_transform_disagreement_px=args.max_transform_disagreement_px,
            min_warped_landmark_in_bounds_ratio=args.min_warped_landmark_in_bounds_ratio,
            project_root=PROJECT_ROOT,
        )
        prefix = "RAW_FRAME_WARP_SMOKE"
    print(f"[{prefix}] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
