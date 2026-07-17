#!/usr/bin/env python
"""Run FACE-S1 phase-1 face usability distributions and review sheets."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.face_usability import run_face_usability_distribution_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read aligned JPGs and aligned-space 68-point landmarks to produce "
            "FACE-S1 phase-1 distributions/contact sheets. No thresholds, labels, "
            "predictions, AU values, OpenFace process, or training outputs are used."
        )
    )
    parser.add_argument("--dataset-split-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--aligned-landmark-root", required=True)
    parser.add_argument("--frame-audit-dir", required=True)
    parser.add_argument("--source-presence-gate", required=True)
    parser.add_argument("--exposure-review-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-integrity-comparison-summary", default=None)
    parser.add_argument("--video-id", action="append", default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--canonical-sample-step", type=int, default=30)
    parser.add_argument("--contact-frames-per-reason", type=int, default=12)
    parser.add_argument("--contact-sheet-columns", type=int, default=4)
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_face_usability_distribution_audit(
        dataset_split_file=Path(args.dataset_split_file),
        image_root=Path(args.image_root),
        aligned_landmark_root=Path(args.aligned_landmark_root),
        frame_audit_dir=Path(args.frame_audit_dir),
        source_presence_gate=Path(args.source_presence_gate),
        exposure_review_manifest=Path(args.exposure_review_manifest),
        output_dir=Path(args.output_dir),
        image_integrity_comparison_summary=(
            Path(args.image_integrity_comparison_summary)
            if args.image_integrity_comparison_summary
            else None
        ),
        video_ids=args.video_id,
        max_videos=args.max_videos,
        canonical_sample_step=args.canonical_sample_step,
        contact_frames_per_reason=args.contact_frames_per_reason,
        contact_sheet_columns=args.contact_sheet_columns,
        project_root=PROJECT_ROOT,
    )
    print("[FACE_USABILITY_PHASE1] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
