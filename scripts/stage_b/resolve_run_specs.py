#!/usr/bin/env python
"""Version-aware Stage B run resolver (CLI).

Enumerates every ``version_N`` dir for each experiment and labels it by its
sweep config (read from the version's ``resolved_config.yaml``), so aggregation
can compare base configs AND lambda/power sweeps in one table.

Labels:
    base (lambda_id==0.05, power==0.5)          -> eN
    lambda sweep (lambda_id!=0.05, power==0.5)  -> eN_lambda<L>
    power sweep (power!=0.5, lambda_id==0.05)   -> eN_power<P>
    both non-base                               -> eN_lambda<L>_power<P>

Multiple versions with the same sweep signature are de-duplicated to the latest.

Usage:
    python scripts/stage_b/resolve_run_specs.py e0 e1 e2 e3   # version-aware
    python scripts/stage_b/resolve_run_specs.py --latest-only e1   # old behavior

Prints ``NAME=PATH`` lines (one per resolved run) to stdout.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.stage_b_runs import resolve_stage_b_run_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "names",
        nargs="*",
        default=["e0", "e1", "e2", "e3"],
        help="Experiment ids to resolve (default: e0 e1 e2 e3).",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Only the latest version per experiment, labeled eN (pre-version-aware behavior).",
    )
    args = parser.parse_args()

    specs = resolve_stage_b_run_specs(args.names, latest_only=args.latest_only)
    for label, path in specs:
        print(f"{label}={path}")


if __name__ == "__main__":
    main()
