#!/usr/bin/env python
"""Generate matched-only artifact weak-label correlations.

This is a post-processing utility for Stage A3 outputs. It consumes existing
``artifact_weaklabel_summary.csv`` files, keeps only rows with prediction/error
targets, then writes ``*_matched`` summary, correlation and report files.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_named_path(value):
    if "=" in value:
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise argparse.ArgumentTypeError("Named summary must use NAME=PATH.")
        return name, Path(path)
    path = Path(value)
    return None, path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate matched-only artifact weak-label correlations."
    )
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        type=_parse_named_path,
        help=(
            "Artifact weak-label summary CSV. Can be repeated. Use NAME=PATH "
            "when writing multiple summaries into one --output-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional output directory. If omitted, writes *_matched files next "
            "to each input summary."
        ),
    )
    parser.add_argument(
        "--coupling-threshold",
        type=float,
        default=0.2,
        help="|corr with abs_error| threshold used in the report (default 0.2).",
    )
    parser.add_argument(
        "--required-field",
        action="append",
        default=None,
        help=(
            "Required field for matched rows. Can be repeated. Default: "
            "pred_bdi, residual, abs_error."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    from src.diagnostics.artifact_weaklabels import (
        DEFAULT_MATCH_REQUIRED_FIELDS,
        run_matched_weaklabel_correlation,
    )

    required_fields = tuple(args.required_field or DEFAULT_MATCH_REQUIRED_FIELDS)
    all_generated = []
    for name, summary_path in args.summary:
        output_dir = None
        if args.output_dir:
            if len(args.summary) > 1:
                output_dir = Path(args.output_dir) / (name or summary_path.stem)
            else:
                output_dir = Path(args.output_dir)
        generated = run_matched_weaklabel_correlation(
            summary_csv=summary_path,
            output_dir=output_dir,
            name=name if args.output_dir and len(args.summary) == 1 else None,
            coupling_threshold=args.coupling_threshold,
            required_fields=required_fields,
        )
        all_generated.extend(generated)
        print(f"[MATCHED-WEAKLABELS] {summary_path}")
        for path in generated:
            print(f"  - {path}")

    print(f"[MATCHED-WEAKLABELS] Generated {len(all_generated)} files.")


if __name__ == "__main__":
    main()
