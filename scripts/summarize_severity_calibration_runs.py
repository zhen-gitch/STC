#!/usr/bin/env python
"""Aggregate multiple severity calibration audits into one comparison table.

Example:

    python scripts/summarize_severity_calibration_runs.py \
        --output-dir analysis_outputs/severity_calibration_summary \
        --run rgb=experiment/default/rgb/diagnostics/severity_calibration \
        --run center_mask=experiment/default/center_mask/diagnostics/severity_calibration \
        --run border_black_feather=experiment/default/border_black_feather/diagnostics/severity_calibration

Each ``--run`` value is ``NAME=PATH``.  PATH may be the output directory of
``scripts/audit_severity_calibration.py`` (containing
``tables/severity_calibration_fit.csv``) or the fit CSV itself.
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.severity_calibration_runs import summarize_severity_calibration_runs


def _parse_run_spec(spec):
    if "=" not in spec:
        raise ValueError(f"Run spec must be NAME=PATH, got: {spec}")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Run spec has empty name: {spec}")
    return name, raw_path.strip()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize and compare multiple severity calibration audit runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Severity calibration run as NAME=PATH. PATH may be a fit CSV or a run directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the multi-run summary tables and report.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    run_specs = [_parse_run_spec(spec) for spec in args.run]
    generated = summarize_severity_calibration_runs(
        run_specs=run_specs,
        output_dir=args.output_dir,
    )
    print("[SEVERITY CALIBRATION RUNS] Generated files:")
    for path in generated:
        print(f"  - {path}")
    print(f"[SEVERITY CALIBRATION RUNS] Output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
