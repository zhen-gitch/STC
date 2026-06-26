#!/usr/bin/env python
"""RPDF Stage A4: severity imbalance / prediction compression summary.

Joins prediction_run_summary.csv, severity_bias_summary.csv and
severity_calibration_run_summary.csv to decide whether severity-balanced
regression is a Stage B baseline or a Stage D side-branch.  See
``docs/SHORTCUT_AUDIT_DESIGN.md`` section 13.4.

Example:

    python scripts/summarize_severity_imbalance.py \\
        --prediction-summary analysis_outputs/rgb_input_ablation_summary/tables/prediction_run_summary.csv \\
        --severity-bias analysis_outputs/rgb_input_ablation_summary/tables/severity_bias_summary.csv \\
        --calibration-summary analysis_outputs/severity_calibration_summary/tables/severity_calibration_run_summary.csv \\
        --output-dir analysis_outputs/severity_imbalance_summary
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        description="RPDF Stage A4: severity imbalance / prediction compression summary."
    )
    parser.add_argument("--prediction-summary", default=None, help="prediction_run_summary.csv")
    parser.add_argument("--severity-bias", default=None, help="severity_bias_summary.csv")
    parser.add_argument("--calibration-summary", default=None, help="severity_calibration_run_summary.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Explicit run name to include (can be given multiple times).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    from src.diagnostics.severity_imbalance import run_severity_imbalance_audit

    runs = args.run if args.run else None
    generated = run_severity_imbalance_audit(
        output_dir=Path(args.output_dir),
        prediction_summary_csv=Path(args.prediction_summary) if args.prediction_summary else None,
        severity_bias_csv=Path(args.severity_bias) if args.severity_bias else None,
        calibration_summary_csv=Path(args.calibration_summary) if args.calibration_summary else None,
        runs=runs,
    )
    print("[SEVERITY-IMBALANCE] Generated files:")
    for path in generated:
        print(f"  - {path}")
    print(f"[SEVERITY-IMBALANCE] Output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
