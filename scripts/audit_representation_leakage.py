#!/usr/bin/env python
"""Run Stage C train-to-validation representation leakage analysis."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Read-only Stage C representation leakage analysis."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("NAME", "TRAIN_NPZ", "VAL_NPZ"),
        required=True,
        help="Repeat for each run.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    return parser


def main():
    args = build_parser().parse_args()
    from src.diagnostics.representation_leakage import (
        run_representation_leakage_analysis,
    )

    generated = run_representation_leakage_analysis(
        run_specs=args.run,
        output_dir=args.output_dir,
        ridge_alpha=args.ridge_alpha,
    )
    print("[REPRESENTATION_LEAKAGE] generated:")
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
