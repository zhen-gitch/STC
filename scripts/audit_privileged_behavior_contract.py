#!/usr/bin/env python
"""Run the PB-P0 aligned-JPG AU/head data-contract audit."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.privileged_behavior_p0 import run_privileged_behavior_p0  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact aligned-JPG/OpenFace AU+head targets for PB-P0. "
            "Landmark-only schemas emit a complete BLOCKED result."
        )
    )
    parser.add_argument("--dataset-split-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--openface-root", required=True)
    parser.add_argument(
        "--source-run-manifest",
        default=None,
        help="OpenFace aligned-image extraction run_manifest.json (default: <openface-root>/_audit/run_manifest.json).",
    )
    parser.add_argument(
        "--source-video-contract",
        required=True,
        help="source_video_contract.csv; only exact identity/frame-count/FPS metadata is read.",
    )
    parser.add_argument(
        "--coverage-policy-decision",
        default=None,
        help="Full PB-P0A2 coverage_policy_decision.json; required with its SHA and policy when legacy strict coverage fails.",
    )
    parser.add_argument(
        "--coverage-policy-decision-sha256",
        default=None,
        help="Expected SHA-256 of --coverage-policy-decision.",
    )
    parser.add_argument(
        "--coverage-policy",
        default=None,
        help="Frozen PB-P0A2 coverage policy JSON bound by the supplied decision.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument(
        "--max-pose-velocity-dt-seconds",
        "--max-dt-seconds",
        dest="max_pose_velocity_dt_seconds",
        type=float,
        default=0.1,
    )
    return parser


def main():
    args = build_parser().parse_args()
    openface_root = Path(args.openface_root)
    source_run_manifest = (
        Path(args.source_run_manifest)
        if args.source_run_manifest
        else openface_root / "_audit" / "run_manifest.json"
    )
    generated = run_privileged_behavior_p0(
        dataset_split_file=Path(args.dataset_split_file),
        image_root=Path(args.image_root),
        openface_root=openface_root,
        source_run_manifest=source_run_manifest,
        source_video_contract=Path(args.source_video_contract),
        output_dir=Path(args.output_dir),
        coverage_policy_decision=(
            Path(args.coverage_policy_decision)
            if args.coverage_policy_decision
            else None
        ),
        coverage_policy_decision_sha256=args.coverage_policy_decision_sha256,
        coverage_policy=(
            Path(args.coverage_policy) if args.coverage_policy else None
        ),
        confidence_threshold=args.confidence_threshold,
        max_pose_velocity_dt_seconds=args.max_pose_velocity_dt_seconds,
        project_root=PROJECT_ROOT,
    )
    print("[PB-P0] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
