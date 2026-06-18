#!/usr/bin/env python
"""Offline entry point for severity calibration verification.

Fits a post-hoc linear calibration ``pred_calibrated = a * pred + b`` on
validation predictions only, then applies the fixed mapping to test predictions
and reports the change in metrics and severity group bias.

This is a diagnostic-only audit and must not use test labels to fit the
calibration.  See ``docs/RGB_OVERFITTING_AUDIT_PLAN.md`` (P0-F).

Example:

    python scripts/audit_severity_calibration.py \
        --val-predictions experiment/default/rgb/version_0/diagnostics/val/regression/val_predictions.csv \
        --test-predictions experiment/default/rgb/version_0/diagnostics/test/regression/test_predictions.csv \
        --output-dir experiment/default/rgb/version_0/diagnostics/severity_calibration
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.severity_calibration import run_severity_calibration_audit


def build_parser():
    parser = argparse.ArgumentParser(
        description="Verify whether MTL-Lite test errors are mainly prediction compression."
    )
    parser.add_argument(
        "--val-predictions",
        required=True,
        help="Path to val_predictions.csv (used to fit calibration).",
    )
    parser.add_argument(
        "--test-predictions",
        required=True,
        help="Path to test_predictions.csv (evaluated with fixed calibration).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where tables, figures, and reports are written.",
    )
    parser.add_argument(
        "--max-score",
        type=int,
        default=63,
        help="Maximum BDI score for scatter plot axes (default 63).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_severity_calibration_audit(
        val_predictions=args.val_predictions,
        test_predictions=args.test_predictions,
        output_dir=args.output_dir,
        max_score=args.max_score,
    )
    print("[SEVERITY CALIBRATION] Generated files:")
    for path in generated:
        print(f"  - {path}")
    print(f"[SEVERITY CALIBRATION] Output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
