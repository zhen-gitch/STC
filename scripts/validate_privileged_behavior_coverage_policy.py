#!/usr/bin/env python
"""Validate one immutable OpenFace rich extraction against PB-P0A2 policy."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.privileged_behavior_coverage_policy import (  # noqa: E402
    validate_behavior_source_coverage,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Apply the frozen label-blind PB-P0A2 mask-aware source/coverage policy. "
            "This never authorizes full-rich, P0B or training."
        )
    )
    parser.add_argument("--policy", required=True)
    parser.add_argument("--dataset-split-file", required=True)
    parser.add_argument("--extraction-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    generated = validate_behavior_source_coverage(
        policy_path=Path(args.policy),
        dataset_split_file=Path(args.dataset_split_file),
        extraction_root=Path(args.extraction_root),
        output_dir=Path(args.output_dir),
        project_root=PROJECT_ROOT,
    )
    decision_path = Path(args.output_dir) / "coverage_policy_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    print(f"[PB-P0A2] status={decision['status']} next={decision['next_action']}")
    for path in generated:
        print(f"  - {path}")
    if decision["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
