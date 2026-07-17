#!/usr/bin/env python
"""Calibrate train-only full-frame temporal exposure stability."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.exposure_temporal_stability import (
    run_exposure_temporal_stability_audit,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Scan train-normal and audited exposure-candidate JPGs at full frame rate, "
            "freeze the train-normal luma-IQR q90 review threshold, and write read-only "
            "triage tables. No BDI labels, predictions, OpenFace run, or image "
            "materialization are used."
        )
    )
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--image-integrity-comparison-summary",
        default=None,
        help=(
            "Required when --image-root differs from the audited root. Must prove "
            "EXACT_PASS for the relocated image tree."
        ),
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--stability-quantile",
        type=float,
        default=0.90,
        help="Frozen at 0.90; other values fail closed.",
    )
    parser.add_argument("--max-reference-videos", type=int, default=None)
    parser.add_argument("--max-candidate-videos", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_exposure_temporal_stability_audit(
        audit_dir=Path(args.audit_dir),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        image_integrity_comparison_summary=(
            Path(args.image_integrity_comparison_summary)
            if args.image_integrity_comparison_summary
            else None
        ),
        workers=args.workers,
        stability_quantile=args.stability_quantile,
        max_reference_videos=args.max_reference_videos,
        max_candidate_videos=args.max_candidate_videos,
        project_root=PROJECT_ROOT,
    )
    print("[EXPOSURE_TEMPORAL_STABILITY] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
