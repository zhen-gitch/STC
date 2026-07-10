#!/usr/bin/env python
"""Aggregate multiple subject-attacker audits into one comparison table.

Example::

    python scripts/summarize_subject_attacker_runs.py \\
        --output-dir logs/stage_b/aggregate/subject_attacker \\
        --run e0=<RUN_DIR_E0> \\
        --run e1=<RUN_DIR_E1> \\
        --run e1_lambda0.02=<RUN_DIR_E1_L02> \\
        --run e3=<RUN_DIR_E3>

Each ``--run`` value is ``NAME=PATH``.  PATH may be the output directory of
``scripts/audit_subject_attacker.py`` (containing
``tables/subject_attacker_summary.csv``) or the summary CSV itself.  Runs whose
summary is missing are skipped (not an error) so partial diagnoses aggregate.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.subject_attacker import summarize_subject_attacker_runs


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
        description="Summarize and compare multiple subject-attacker audit runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Attacker run as NAME=PATH. PATH may be a summary CSV or a run directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the multi-run summary table and report.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    run_specs = [_parse_run_spec(spec) for spec in args.run]
    generated = summarize_subject_attacker_runs(run_specs, output_dir=args.output_dir)
    print("[SUBJECT_ATTACKER_RUNS] generated files:")
    for path in generated:
        print(f"  - {path}")
    print(f"[SUBJECT_ATTACKER_RUNS] output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
