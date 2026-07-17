#!/usr/bin/env python
"""Generate train-only paired review panels for FACE-S1 landmark jumps."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.face_usability_temporal_review import run_landmark_jump_pair_review


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Read a completed FACE-S1 phase-1 contact manifest and generate exact "
            "t-1/t paired panels for train-only landmark-jump candidates."
        )
    )
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--aligned-landmark-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--columns", type=int, default=4)
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_landmark_jump_pair_review(
        source_run_dir=Path(args.source_run_dir),
        aligned_landmark_root=Path(args.aligned_landmark_root),
        output_dir=Path(args.output_dir),
        columns=args.columns,
        project_root=PROJECT_ROOT,
    )
    print("[FACE_USABILITY_TEMPORAL_REVIEW] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
