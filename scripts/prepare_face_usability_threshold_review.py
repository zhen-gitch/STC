#!/usr/bin/env python
"""Prepare the train-only dual-lane FACE-S1 threshold review template."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.face_usability_threshold_review import run_threshold_review_template


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate a completed FACE-S1 train-only contact review into separate "
            "global-face, local-geometry, and temporal-boundary PENDING decisions."
        )
    )
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--temporal-review-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_threshold_review_template(
        source_run_dir=Path(args.source_run_dir),
        temporal_review_dir=Path(args.temporal_review_dir),
        output_dir=Path(args.output_dir),
        project_root=PROJECT_ROOT,
    )
    print("[FACE_USABILITY_THRESHOLD_REVIEW] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
