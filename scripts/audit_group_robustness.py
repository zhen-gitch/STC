#!/usr/bin/env python
"""Run Stage C train-thresholded validation group robustness analysis."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Read-only Stage C validation group robustness analysis."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("NAME", "TRAIN_PREDICTIONS", "VAL_PREDICTIONS"),
        required=True,
        help="Repeat for each run. Train paths are recorded for provenance.",
    )
    parser.add_argument(
        "--axis",
        action="append",
        nargs=4,
        metavar=("NAME", "TRAIN_CSV", "VAL_CSV", "VALUE_COLUMN"),
        default=[],
        help="Optional continuous audit axis; threshold is fit from train CSV only.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-run", default="C-REF")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    from src.diagnostics.group_robustness import run_group_robustness_analysis

    generated = run_group_robustness_analysis(
        run_specs=args.run,
        output_dir=args.output_dir,
        axis_specs=args.axis,
        reference_run=args.reference_run,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print("[GROUP_ROBUSTNESS] generated:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
