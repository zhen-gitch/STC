#!/usr/bin/env python
"""Run the frozen PB-P0C core-AU raw/aligned fidelity audit."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diagnostics.privileged_behavior_au_fidelity import (  # noqa: E402
    run_privileged_behavior_au_fidelity,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare immutable raw-video and aligned-JPG OpenFace AU12/14/15 "
            "sequences. Physical train alone determines eligibility; val/test "
            "are report-only. This never authorizes P0D, P0E, models, or training."
        )
    )
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--dataset-split-file", required=True)
    parser.add_argument("--source-video-contract", required=True)
    parser.add_argument("--aligned-openface-root", required=True)
    parser.add_argument("--aligned-run-manifest", required=True)
    parser.add_argument("--raw-openface-root", required=True)
    parser.add_argument("--raw-run-manifest", required=True)
    parser.add_argument("--raw-run-manifest-sha256", required=True)
    parser.add_argument("--p0b-run-manifest", required=True)
    parser.add_argument("--p0b-selected-target-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    generated = run_privileged_behavior_au_fidelity(
        policy_path=Path(args.policy),
        policy_sha256=args.policy_sha256,
        dataset_split_file=Path(args.dataset_split_file),
        source_video_contract=Path(args.source_video_contract),
        aligned_openface_root=Path(args.aligned_openface_root),
        aligned_run_manifest=Path(args.aligned_run_manifest),
        raw_openface_root=Path(args.raw_openface_root),
        raw_run_manifest=Path(args.raw_run_manifest),
        raw_run_manifest_sha256=args.raw_run_manifest_sha256,
        p0b_run_manifest=Path(args.p0b_run_manifest),
        p0b_selected_target_manifest=Path(args.p0b_selected_target_manifest),
        output_dir=Path(args.output_dir),
        project_root=PROJECT_ROOT,
        command_line=" ".join(sys.argv),
    )
    decision_path = Path(args.output_dir).expanduser().resolve() / "au_fidelity_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    print(
        f"[PB-P0C] audit={decision['audit_status']} "
        f"eligibility={decision['eligibility_status']} next={decision['next_action']}"
    )
    for path in generated:
        print(f"  - {path}")
    if decision["audit_status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
