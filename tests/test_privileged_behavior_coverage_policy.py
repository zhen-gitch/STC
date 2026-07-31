import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_privileged_behavior_coverage_policy import build_parser
from src.diagnostics.privileged_behavior_coverage_policy import (
    CoveragePolicyError,
    load_coverage_policy,
    validate_behavior_source_coverage,
)


AU_COLUMNS = ["AU12_r", "AU14_r", "AU15_r"]
POSE_COLUMNS = ["pose_Rx", "pose_Ry", "pose_Rz"]
CSV_COLUMNS = ["frame", "confidence", "success", *AU_COLUMNS, *POSE_COLUMNS]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _video_rows(video_id, valid_count, total=4):
    rows = []
    for frame in range(1, total + 1):
        valid = frame <= valid_count
        rows.append(
            {
                "frame": frame,
                "confidence": 0.9 if valid else 0.2,
                "success": 1 if valid else 0,
                **{column: 0.1 * frame for column in AU_COLUMNS + POSE_COLUMNS},
            }
        )
    return rows


def _build_pilot(
    tmp_path, *, freeform_valid=3, northwind_valid=4, northwind_split="train"
):
    root = tmp_path / "pilot"
    audit = root / "_audit"
    audit.mkdir(parents=True)
    videos = {
        "203_1_Freeform_video_aligned": freeform_valid,
        "203_1_Northwind_video_aligned": northwind_valid,
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "train": ["203_1_Freeform_video_aligned"],
                "val": ["203_1_Northwind_video_aligned"]
                if northwind_split == "val"
                else [],
                "test": [],
            }
            | (
                {"train": list(videos), "val": [], "test": []}
                if northwind_split == "train"
                else {}
            )
        ),
        encoding="utf-8",
    )

    content_rows = []
    summary_rows = []
    contract_rows = []
    total_success = 0
    for video_id, valid_count in videos.items():
        csv_path = root / f"{video_id}.csv"
        rows = _video_rows(video_id, valid_count)
        _write_csv(csv_path, CSV_COLUMNS, rows)
        ratio = valid_count / len(rows)
        content_rows.append(
            {
                "video_id": video_id,
                "csv_rows": len(rows),
                "schema_sha256": "schema",
                "csv_size_bytes": csv_path.stat().st_size,
                "csv_sha256": _sha(csv_path),
                "status": "FAIL" if ratio < 0.995 else "PASS",
            }
        )
        summary_rows.append(
            {
                "video_id": video_id,
                "image_count": len(rows),
                "success_count": valid_count,
                "success_ratio": ratio,
                "schema_column_count": len(CSV_COLUMNS),
                "status": "FAIL" if ratio < 0.995 else "PASS",
                "issues": "" if ratio >= 0.995 else f"success_ratio:{ratio}<0.995",
            }
        )
        contract_rows.append(
            {
                "video_id": video_id,
                "image_count": len(rows),
                "selected_for_run": True,
                "status": "PASS",
                "issues": "",
            }
        )
        total_success += valid_count

    content_path = audit / "csv_content_manifest.csv"
    _write_csv(
        content_path,
        ["video_id", "csv_rows", "schema_sha256", "csv_size_bytes", "csv_sha256", "status"],
        content_rows,
    )
    video_summary_path = audit / "video_run_summary.csv"
    _write_csv(
        video_summary_path,
        [
            "video_id",
            "image_count",
            "success_count",
            "success_ratio",
            "schema_column_count",
            "status",
            "issues",
        ],
        summary_rows,
    )
    input_contract_path = audit / "input_frame_contract.csv"
    _write_csv(
        input_contract_path,
        ["video_id", "image_count", "selected_for_run", "status", "issues"],
        contract_rows,
    )
    summary_path = audit / "extraction_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "video_count": len(videos),
                "total_images": 8,
                "total_csv_rows": 8,
                "total_successes": total_success,
                "status": "FAIL" if total_success < 8 else "PASS",
                "csv_content_manifest_sha256": _sha(content_path),
                "schema_count": 1,
                "schema_consistent": True,
                "hog_file_count": 0,
                "tracked_video_file_count": 0,
                "regenerated_aligned_image_count": 0,
            }
        ),
        encoding="utf-8",
    )
    manifest_path = audit / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_scope": "debug_subset",
                "feature_profile": "test_profile",
                "git_commit": "abc1234",
                "git_status_available": True,
                "git_status_short": "",
                "gaze_requested": False,
                "hog_enabled": False,
                "tracked_video_enabled": False,
                "aligned_image_generation_enabled": False,
                "source_video_contract_sha256": "source-contract",
                "selected_video_count": len(videos),
                "input_frame_contract_sha256": _sha(input_contract_path),
                "csv_content_manifest_sha256": _sha(content_path),
                "extraction_summary_sha256": _sha(summary_path),
                "min_success_ratio": 0.995,
            }
        ),
        encoding="utf-8",
    )

    policy = {
        "policy_id": "test_policy_v1",
        "policy_version": 1,
        "status": "FROZEN",
        "label_blind": True,
        "source_contract": {
            "feature_profile": "test_profile",
            "confidence_threshold": 0.8,
            "legacy_strict_success_ratio": 0.995,
            "legacy_strict_status_is_diagnostic_only": True,
            "expected_csv_column_count": len(CSV_COLUMNS),
            "required_au_columns": AU_COLUMNS,
            "required_pose_columns": POSE_COLUMNS,
            "required_source_video_contract_sha256": "source-contract",
            "require_clean_source_git": True,
            "require_complete_rows_schema_hashes": True,
            "require_no_gaze_request": True,
        },
        "pilot_evidence": {
            "source_git_commit": "abc1234",
            "dataset_split_sha256": _sha(split_path),
            "run_manifest_sha256": _sha(manifest_path),
            "extraction_summary_sha256": _sha(summary_path),
            "video_run_summary_sha256": _sha(video_summary_path),
            "input_frame_contract_sha256": _sha(input_contract_path),
            "csv_content_manifest_sha256": _sha(content_path),
            "expected_video_count": 2,
            "expected_frame_count": 8,
        },
        "full_source_expectations": {
            "expected_video_count": 2,
            "expected_frame_count": 8,
            "require_exact_dataset_split_video_set": True,
        },
        "pilot_feasibility_thresholds": {
            "aggregate_ratio_policy": "REPORT_ONLY_INCOMPLETE_PHYSICAL_TRAIN",
            "enforcement_split": "train",
            "minimum_au_valid_frames_per_video": 2,
            "minimum_head_valid_pairs_per_video": 1,
            "require_each_group_nonzero": True,
        },
        "physical_train_full_thresholds": {
            "minimum_overall_quality_valid_ratio": 0.5,
            "minimum_task_quality_valid_ratio": 0.5,
            "maximum_task_quality_valid_ratio_gap": 0.5,
            "minimum_au_valid_frames_per_video": 2,
            "minimum_head_valid_pairs_per_video": 1,
            "require_each_group_nonzero": True,
        },
        "validation_test_policy": "REPORT_ONLY_NO_THRESHOLD_CALLBACK",
        "decision_contract": {
            "pilot_pass_status": "PASS_PILOT_MASK_AWARE_FEASIBILITY",
            "full_pass_status": "PASS_FULL_SOURCE_COVERAGE",
            "blocked_status": "BLOCKED",
            "full_rich_authorized": False,
            "p0b_authorized": False,
            "training_authorized": False,
            "pilot_pass_next_action": "REQUEST_SEPARATE_FULL",
        },
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return policy_path, split_path, root


def _promote_fixture_to_full(root):
    manifest_path = root / "_audit" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_scope"] = "full_dataset"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_pilot_strict_failure_can_pass_mask_aware_feasibility(tmp_path):
    policy, split, root = _build_pilot(tmp_path)
    output = tmp_path / "output"
    validate_behavior_source_coverage(
        policy_path=policy,
        dataset_split_file=split,
        extraction_root=root,
        output_dir=output,
        project_root=tmp_path,
    )
    decision = json.loads((output / "coverage_policy_decision.json").read_text())
    assert decision["status"] == "PASS_PILOT_MASK_AWARE_FEASIBILITY"
    assert decision["legacy_strict_status_observed"] == "FAIL"
    assert decision["legacy_strict_status_used_as_gate"] is False
    assert decision["next_action"] == "REQUEST_SEPARATE_FULL"
    assert decision["full_rich_authorized"] is False
    assert decision["p0b_authorized"] is False
    assert decision["training_authorized"] is False
    assert decision["label_access"] == {
        "bdi_label_access_count": 0,
        "prediction_access_count": 0,
        "checkpoint_access_count": 0,
        "validation_test_threshold_callback_count": 0,
    }


def test_pilot_train_video_minimum_failure_blocks(tmp_path):
    policy, split, root = _build_pilot(tmp_path, freeform_valid=1, northwind_valid=4)
    output = tmp_path / "output"
    validate_behavior_source_coverage(
        policy_path=policy,
        dataset_split_file=split,
        extraction_root=root,
        output_dir=output,
        project_root=tmp_path,
    )
    decision = json.loads((output / "coverage_policy_decision.json").read_text())
    issues = (output / "tables" / "coverage_policy_issues.csv").read_text()
    assert decision["status"] == "BLOCKED"
    assert decision["next_action"] == "STOP_AND_REVIEW_POLICY_OR_SOURCE_ISSUES"
    assert "au_valid_frames_below_minimum" in issues


def test_full_pass_uses_backward_compatible_next_action_without_policy_mutation(
    tmp_path,
):
    policy, split, root = _build_pilot(tmp_path)
    _promote_fixture_to_full(root)
    output = tmp_path / "full_output"

    validate_behavior_source_coverage(
        policy_path=policy,
        dataset_split_file=split,
        extraction_root=root,
        output_dir=output,
        project_root=tmp_path,
    )

    decision = json.loads((output / "coverage_policy_decision.json").read_text())
    assert decision["status"] == "PASS_FULL_SOURCE_COVERAGE"
    assert decision["next_action"] == "REQUEST_SEPARATE_P0B_AUTHORIZATION"
    assert decision["full_rich_authorized"] is False
    assert decision["p0b_authorized"] is False
    assert decision["training_authorized"] is False


def test_pilot_validation_coverage_is_report_only(tmp_path):
    policy, split, root = _build_pilot(
        tmp_path, freeform_valid=4, northwind_valid=0, northwind_split="val"
    )
    output = tmp_path / "output"
    validate_behavior_source_coverage(
        policy_path=policy,
        dataset_split_file=split,
        extraction_root=root,
        output_dir=output,
        project_root=tmp_path,
    )
    decision = json.loads((output / "coverage_policy_decision.json").read_text())
    video_table = (output / "tables" / "video_coverage_policy.csv").read_text()
    assert decision["status"] == "PASS_PILOT_MASK_AWARE_FEASIBILITY"
    assert decision["full_train_aggregate_thresholds_evaluated"] is False
    assert "val,203_1_Northwind_video_aligned" in video_table
    assert ",REPORT_ONLY," in video_table


def test_pinned_source_mutation_blocks(tmp_path):
    policy, split, root = _build_pilot(tmp_path)
    with (root / "_audit" / "video_run_summary.csv").open("a", encoding="utf-8") as handle:
        handle.write("mutated\n")
    output = tmp_path / "output"
    validate_behavior_source_coverage(
        policy_path=policy,
        dataset_split_file=split,
        extraction_root=root,
        output_dir=output,
        project_root=tmp_path,
    )
    decision = json.loads((output / "coverage_policy_decision.json").read_text())
    issues = (output / "tables" / "coverage_policy_issues.csv").read_text()
    assert decision["status"] == "BLOCKED"
    assert "pilot_video_run_summary_sha256_mismatch" in issues


def test_content_manifest_status_must_match_legacy_summary(tmp_path):
    policy, split, root = _build_pilot(tmp_path)
    content_path = root / "_audit" / "csv_content_manifest.csv"
    rows = list(csv.DictReader(content_path.open(newline="", encoding="utf-8-sig")))
    rows[0]["status"] = "PASS"
    _write_csv(content_path, rows[0].keys(), rows)

    summary_path = root / "_audit" / "extraction_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["csv_content_manifest_sha256"] = _sha(content_path)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path = root / "_audit" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["csv_content_manifest_sha256"] = _sha(content_path)
    manifest["extraction_summary_sha256"] = _sha(summary_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    policy_payload = json.loads(policy.read_text())
    policy_payload["pilot_evidence"]["csv_content_manifest_sha256"] = _sha(content_path)
    policy_payload["pilot_evidence"]["extraction_summary_sha256"] = _sha(summary_path)
    policy_payload["pilot_evidence"]["run_manifest_sha256"] = _sha(manifest_path)
    policy.write_text(json.dumps(policy_payload), encoding="utf-8")

    output = tmp_path / "output"
    validate_behavior_source_coverage(
        policy_path=policy,
        dataset_split_file=split,
        extraction_root=root,
        output_dir=output,
        project_root=tmp_path,
    )
    issues = (output / "tables" / "coverage_policy_issues.csv").read_text()
    assert "csv_content_manifest_status_mismatch" in issues


def test_policy_requires_frozen_label_blind_contract(tmp_path):
    policy, _, _ = _build_pilot(tmp_path)
    payload = json.loads(policy.read_text())
    payload["label_blind"] = False
    policy.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CoveragePolicyError, match="FROZEN and label_blind"):
        load_coverage_policy(policy)


def test_cli_exposes_no_label_prediction_checkpoint_or_training_inputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--policy",
            "/policy.json",
            "--dataset-split-file",
            "/split.json",
            "--extraction-root",
            "/rich",
            "--output-dir",
            "/audit",
        ]
    )
    keys = set(vars(args))
    assert keys == {"policy", "dataset_split_file", "extraction_root", "output_dir"}
    assert not keys & {"labels", "predictions", "checkpoint", "train"}
