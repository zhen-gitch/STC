#!/usr/bin/env python
"""Aggregate multiple embedding identity retrieval audits into one comparison table.

Example:

    python scripts/summarize_identity_retrieval_runs.py \
        --output-dir analysis_outputs/identity_retrieval_summary \
        --run rgb=<LOG_DIR>/default/rgb/version_0/diagnostics/identity_retrieval \
        --run center_mask=<LOG_DIR>/default/center_mask/version_0/diagnostics/identity_retrieval \
        --run border_black_feather=<LOG_DIR>/default/border_black_feather/version_0/diagnostics/identity_retrieval

Each ``--run`` value is ``NAME=PATH``.  PATH may be the output directory of
``scripts/audit_identity_retrieval.py`` (containing
``tables/embedding_identity_summary.csv``) or the summary CSV itself.
"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.identity_retrieval_runs import summarize_identity_retrieval_runs


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
        description="Summarize and compare multiple identity retrieval audit runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Identity retrieval run as NAME=PATH. PATH may be a summary CSV or a run directory.",
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
    generated = summarize_identity_retrieval_runs(
        run_specs=run_specs,
        output_dir=args.output_dir,
    )
    print("[IDENTITY RETRIEVAL RUNS] Generated files:")
    for path in generated:
        print(f"  - {path}")
    print(f"[IDENTITY RETRIEVAL RUNS] Output directory: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
