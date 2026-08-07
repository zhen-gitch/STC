#!/usr/bin/env python
"""Generate REGION-P0A candidate geometry without approving a crop policy."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.landmark_rgb_region_contract import (
    CANDIDATE_MODES,
    run_landmark_rgb_region_contract,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Audit candidate eye/brow, nose/cheek, and mouth/lower-face RGB "
            "regions from aligned-space 68-point landmarks. This command does "
            "not read AU values, labels, predictions, or checkpoints and does "
            "not approve a training policy."
        )
    )
    parser.add_argument("--dataset-split-file", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--aligned-landmark-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-id", action="append", default=None)
    parser.add_argument(
        "--selection-manifest",
        default=None,
        help="Versioned physical-train pilot selection and runtime parameter lock.",
    )
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--canonical-sample-step", type=int, default=30)
    parser.add_argument("--stabilization-window", type=int, default=5)
    parser.add_argument(
        "--candidate-margin-ratio",
        type=float,
        default=0.0,
        help=(
            "Unfrozen candidate margin relative to image size. It expands x "
            "and only the outer top/bottom edges, never shared region boundaries."
        ),
    )
    parser.add_argument(
        "--candidate-mode",
        action="append",
        choices=CANDIDATE_MODES,
        default=None,
        help="Repeat to select modes; defaults to all three candidate modes.",
    )
    parser.add_argument("--overlay-frames-per-mode", type=int, default=2)
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_landmark_rgb_region_contract(
        dataset_split_file=Path(args.dataset_split_file),
        image_root=Path(args.image_root),
        aligned_landmark_root=Path(args.aligned_landmark_root),
        output_dir=Path(args.output_dir),
        video_ids=args.video_id,
        max_videos=args.max_videos,
        confidence_threshold=args.confidence_threshold,
        canonical_sample_step=args.canonical_sample_step,
        stabilization_window=args.stabilization_window,
        candidate_margin_ratio=args.candidate_margin_ratio,
        candidate_modes=args.candidate_mode,
        overlay_frames_per_mode=args.overlay_frames_per_mode,
        project_root=PROJECT_ROOT,
        selection_manifest=args.selection_manifest,
    )
    print("[REGION] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
