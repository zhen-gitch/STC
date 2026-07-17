#!/usr/bin/env python
"""Audit validity-aware temporal slicing without modifying the dataset."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.validity_aware_slicing import run_validity_aware_slicing_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Join the existing failure and source-presence gates, then measure "
            "continuous valid runs and label-free long-video clip candidates."
        )
    )
    parser.add_argument("--dataset-split-file", required=True)
    parser.add_argument("--video-failure-summary", required=True)
    parser.add_argument("--frame-failure-manifest", required=True)
    parser.add_argument("--source-presence-gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-step", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=2000)
    parser.add_argument(
        "--window-frames",
        type=int,
        nargs="+",
        default=[600, 1200, 2000],
        help="Candidate raw-frame clip ceilings.",
    )
    parser.add_argument(
        "--min-clip-frames",
        type=int,
        default=600,
        help="Report clips below this duration; the audit never silently removes them.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_validity_aware_slicing_audit(
        dataset_split_file=Path(args.dataset_split_file),
        video_failure_summary=Path(args.video_failure_summary),
        frame_failure_manifest=Path(args.frame_failure_manifest),
        source_presence_gate=Path(args.source_presence_gate),
        output_dir=Path(args.output_dir),
        sample_step=args.sample_step,
        max_seq_len=args.max_seq_len,
        window_frames_values=args.window_frames,
        min_clip_frames=args.min_clip_frames,
        project_root=PROJECT_ROOT,
    )
    print("[VALIDITY_AWARE_SLICING_AUDIT] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
