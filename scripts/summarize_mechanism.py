#!/usr/bin/env python
"""Build the unified RGB overfitting mechanism summary table.

This entry point merges the per-run summaries produced by the prediction,
identity retrieval, and severity calibration audits into a single mechanism
view (``docs/RGB_OVERFITTING_AUDIT_PLAN.md`` P0-G.1).

Example:

    python scripts/summarize_mechanism.py \
        --prediction-summary analysis_outputs/rgb_input_ablation_summary/tables/prediction_run_summary.csv \
        --severity-bias-summary analysis_outputs/rgb_input_ablation_summary/tables/severity_bias_summary.csv \
        --task-consistency-summary analysis_outputs/rgb_input_ablation_summary/tables/task_consistency_summary.csv \
        --identity-summary analysis_outputs/identity_retrieval_summary/tables/identity_retrieval_run_summary.csv \
        --calibration-summary analysis_outputs/severity_calibration_summary/tables/severity_calibration_run_summary.csv \
        --output-dir analysis_outputs/mechanism_summary
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.mechanism_summary import build_mechanism_summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Merge prediction, identity, and calibration summaries into one mechanism table."
    )
    parser.add_argument(
        "--prediction-summary",
        required=True,
        help="Path to prediction_run_summary.csv produced by scripts/summarize_predictions.py.",
    )
    parser.add_argument(
        "--severity-bias-summary",
        default=None,
        help="Optional severity_bias_summary.csv for minimal/severe residuals.",
    )
    parser.add_argument(
        "--task-consistency-summary",
        default=None,
        help="Optional task_consistency_summary.csv for Freeform/Northwind prediction diff.",
    )
    parser.add_argument(
        "--identity-summary",
        default=None,
        help="Optional identity_retrieval_run_summary.csv for same-subject retrieval.",
    )
    parser.add_argument(
        "--calibration-summary",
        default=None,
        help="Optional severity_calibration_run_summary.csv for calibration deltas.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for mechanism_summary.csv and mechanism_report.md.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    generated = build_mechanism_summary(
        prediction_summary_csv=args.prediction_summary,
        output_dir=args.output_dir,
        severity_bias_summary_csv=args.severity_bias_summary,
        task_consistency_summary_csv=args.task_consistency_summary,
        identity_summary_csv=args.identity_summary,
        calibration_summary_csv=args.calibration_summary,
    )
    print("[MECHANISM SUMMARY] Generated files:")
    for path in generated:
        print(f"  - {path}")
    print(f"[MECHANISM SUMMARY] Output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
