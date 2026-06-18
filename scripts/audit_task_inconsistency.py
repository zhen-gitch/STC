#!/usr/bin/env python
"""Offline entry point for the task inconsistency mixed-factor audit.

This audit pairs Freeform and Northwind predictions for the same subject and
investigates which artifact / quality / geometry / temporal variables are
associated with large prediction differences.  See
``docs/RGB_OVERFITTING_AUDIT_PLAN.md`` (P0-G).

Example:

    python scripts/audit_task_inconsistency.py \
        --predictions experiment/default/rgb/diagnostics/test/regression/test_predictions.csv \
        --black-artifacts-summary experiment/default/rgb/diagnostics/black_artifacts/tables/black_artifact_summary.csv \
        --openface-quality-summary experiment/default/rgb/diagnostics/shortcut_audit/tables/openface_quality_summary.csv \
        --alignment-geometry-summary experiment/default/rgb/diagnostics/alignment_geometry/tables/alignment_geometry_summary.csv \
        --temporal-sampling-summary experiment/default/rgb/diagnostics/temporal_sampling/tables/temporal_sampling_summary.csv \
        --output-dir experiment/default/rgb/diagnostics/task_inconsistency \
        --top-n 20
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.task_inconsistency import run_task_inconsistency_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description="Audit Freeform/Northwind task inconsistency and its mixed factors."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to test_predictions.csv with task_name column.",
    )
    parser.add_argument(
        "--black-artifacts-summary",
        default=None,
        help="Optional black_artifact_summary.csv.",
    )
    parser.add_argument(
        "--openface-quality-summary",
        default=None,
        help="Optional openface_quality_summary.csv.",
    )
    parser.add_argument(
        "--alignment-geometry-summary",
        default=None,
        help="Optional alignment_geometry_summary.csv.",
    )
    parser.add_argument(
        "--temporal-sampling-summary",
        default=None,
        help="Optional temporal_sampling_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for task_inconsistency_manifest.csv, task_artifact_correlation.csv, and report.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top inconsistent subjects to highlight (default 20).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_task_inconsistency_audit(
        predictions_csv=args.predictions,
        output_dir=args.output_dir,
        black_artifacts_summary=args.black_artifacts_summary,
        openface_quality_summary=args.openface_quality_summary,
        alignment_geometry_summary=args.alignment_geometry_summary,
        temporal_sampling_summary=args.temporal_sampling_summary,
        top_n=args.top_n,
    )
    print("[TASK INCONSISTENCY] Generated files:")
    for path in generated:
        print(f"  - {path}")
    print(f"[TASK INCONSISTENCY] Output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
