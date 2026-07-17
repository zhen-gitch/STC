#!/usr/bin/env python
"""Validate and summarize a completed source-video presence review."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.source_presence import run_source_presence_review_report


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate full pure-black source-presence coverage and write frame/video reports."
    )
    parser.add_argument("--frame-failure-manifest", required=True)
    parser.add_argument("--source-run-manifest", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_source_presence_review_report(
        frame_failure_manifest=Path(args.frame_failure_manifest),
        source_run_manifest=Path(args.source_run_manifest),
        review_manifest=Path(args.review_manifest),
        output_dir=Path(args.output_dir),
        project_root=PROJECT_ROOT,
    )
    print("[SOURCE_PRESENCE_REVIEW] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
