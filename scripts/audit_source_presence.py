#!/usr/bin/env python
"""Audit original-video presence for pure-black aligned placeholders."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.source_presence import run_source_presence_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Join aligned pure-black failure runs to original videos and create "
            "read-only human-review artifacts. This script never launches OpenFace."
        )
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--frame-failure-manifest", required=True)
    parser.add_argument("--video-failure-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--raw-openface-root",
        default=None,
        help="Optional existing original-video OpenFace CSV root; no detector is launched.",
    )
    parser.add_argument("--video-id", action="append", default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-run-samples", type=int, default=7)
    parser.add_argument("--contact-sheet-columns", type=int, default=4)
    parser.add_argument("--no-contact-sheets", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_source_presence_audit(
        dataset_root=Path(args.dataset_root),
        frame_failure_manifest=Path(args.frame_failure_manifest),
        video_failure_summary=Path(args.video_failure_summary),
        output_dir=Path(args.output_dir),
        raw_openface_root=Path(args.raw_openface_root) if args.raw_openface_root else None,
        video_ids=args.video_id,
        max_videos=args.max_videos,
        max_run_samples=args.max_run_samples,
        contact_sheet_columns=args.contact_sheet_columns,
        write_contact_sheets=not args.no_contact_sheets,
        project_root=PROJECT_ROOT,
    )
    print("[SOURCE_PRESENCE_AUDIT] generated files:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
