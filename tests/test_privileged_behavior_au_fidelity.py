import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_privileged_behavior_au_fidelity import build_parser
from src.diagnostics.privileged_behavior_au_fidelity import (
    CORE_AU_COLUMNS,
    AuFidelityError,
    OpenFaceSequence,
    _read_openface_sequence,
    _resolve_aligned_status_compatibility,
    aggregate_au_fidelity_metrics,
    build_au_fidelity_decision,
    compute_video_au_fidelity,
    load_au_fidelity_policy,
    run_privileged_behavior_au_fidelity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "behavior_alignment"
    / "privileged_behavior_au_fidelity_policy_v1.json"
)
POLICY_SHA256 = "2f605261958c0cb7f82c8f871c33c05c81e12c146f73ddd0b4b25d13e9276d2d"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _policy():
    return load_au_fidelity_policy(POLICY_PATH)


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mask_aware_p0b_evidence(
    *,
    video_count,
    frame_count,
    legacy_pass_count,
):
    decision_sha256 = "a" * 64
    policy_sha256 = "b" * 64
    p0b_run = {
        "status": "PASS",
        "training_authorized": False,
        "counts": {
            "video_count": video_count,
            "frame_count": frame_count,
            "core_audited_video_count": video_count,
            "rich_schema_video_count": video_count,
            "exact_join_video_count": video_count,
            "blocking_issue_count": 0,
        },
        "inputs": {
            "coverage_policy_decision": {
                "provided": True,
                "expected_sha256": decision_sha256,
                "sha256": decision_sha256,
            },
            "coverage_policy": {
                "provided": True,
                "sha256": policy_sha256,
            },
        },
    }
    selected = {
        "rich_provenance": {
            "mode": "strict_rich",
            "status": "PASS",
            "legacy_strict_status_observed": "FAIL",
            "legacy_strict_pass_count": legacy_pass_count,
            "legacy_strict_fail_count": video_count - legacy_pass_count,
            "legacy_strict_status_used_as_gate": False,
            "mask_aware_decision_required": True,
            "coverage_policy_binding": {
                "required": True,
                "provided": {
                    "decision": True,
                    "decision_sha256": True,
                    "policy": True,
                },
                "status": "PASS",
                "decision": {
                    "expected_sha256": decision_sha256,
                    "observed_sha256": decision_sha256,
                    "status": "PASS_FULL_SOURCE_COVERAGE",
                    "scope": "full",
                    "policy_sha256": policy_sha256,
                },
                "policy": {"sha256": policy_sha256},
                "observed_counts": {
                    "video_count": video_count,
                    "frame_count": frame_count,
                },
            },
        }
    }
    return p0b_run, selected


def _sequence(values, *, invalid=()):
    values = np.asarray(values, dtype=float)
    success = np.ones(values.size, dtype=bool)
    confidence = np.full(values.size, 0.95, dtype=float)
    for index in invalid:
        success[index] = False
        confidence[index] = 0.1
    return OpenFaceSequence(
        frames=np.arange(1, values.size + 1, dtype=np.int64),
        success=success,
        confidence=confidence,
        au_values={au: values.copy() for au in CORE_AU_COLUMNS},
        schema_sha256="schema",
        schema_column_count=9,
    )


def _decision_rows(subject_count=50, *, direction_by_au=None, split="train"):
    direction_by_au = direction_by_au or {au: 0.9 for au in CORE_AU_COLUMNS}
    subject_rows = []
    video_rows = []
    lag_rows = []
    for au in CORE_AU_COLUMNS:
        for index in range(subject_count):
            subject_id = f"{200 + index}_1"
            subject_rows.append(
                {
                    "split": split,
                    "subject_id": subject_id,
                    "au": au,
                    "rho0": 0.9,
                    "direction_rate": direction_by_au[au],
                    "amplitude_ratio": 1.0,
                }
            )
            for task in ("Freeform", "Northwind"):
                video_rows.append(
                    {
                        "split": split,
                        "subject_id": subject_id,
                        "video_id": f"{subject_id}_{task}_video_aligned",
                        "task_name": task,
                        "au": au,
                        "rho0": 0.9,
                        "amplitude_ratio": 1.0,
                        "lag_curve_estimable": True,
                    }
                )
        for lag in range(-5, 6):
            lag_rows.append(
                {
                    "split": split,
                    "aggregation": "split_subject_median",
                    "au": au,
                    "lag_frames": lag,
                    "rho": 0.95 if lag == 0 else 0.8,
                    "subject_count": subject_count,
                }
            )
    return video_rows, subject_rows, lag_rows


def test_frozen_policy_is_core_au_only_and_label_blind():
    policy = _policy()

    assert _sha(POLICY_PATH) == POLICY_SHA256
    assert tuple(policy["scope"]["au_columns"]) == CORE_AU_COLUMNS
    assert policy["scope"]["head_motion_in_eligibility"] is False
    assert policy["scope"]["gaze_in_eligibility"] is False
    assert policy["scope"]["extension_au_in_eligibility"] is False
    assert policy["split_policy"] == {
        "eligibility_split": "train",
        "validation_test_policy": "REPORT_ONLY_NO_THRESHOLD_CALLBACK",
        "labels_accessed": False,
    }
    statistics = policy["statistics"]
    assert statistics["minimum_matched_frames_per_video"] == 120
    assert statistics["minimum_train_subjects"] == 30
    assert statistics["minimum_train_videos_per_task"] == 20
    assert statistics["spearman"] == {
        "tie_method": "average_rank",
        "minimum_subject_median": 0.7,
        "bootstrap_samples": 10000,
        "bootstrap_seed": 42,
        "bootstrap_confidence_level": 0.95,
        "minimum_bootstrap_lower_bound": 0.5,
        "lower_bound_comparison": "strict_greater_than",
    }
    assert statistics["lag"]["minimum_lag_frames"] == -5
    assert statistics["lag"]["maximum_lag_frames"] == 5
    assert statistics["lag"]["canonical_zero_tolerance"] == 0.02


def test_all_pass_aligned_content_does_not_require_compatibility_evidence():
    result = _resolve_aligned_status_compatibility(
        {"video": {"status": "PASS"}},
        {},
        {},
        {},
    )

    assert result == {
        "enabled": False,
        "reason": "all_aligned_content_status_pass",
        "aligned_content_status_counts": {"PASS": 1},
        "accepted_legacy_fail_count": 0,
    }


def test_legacy_fail_requires_exact_mask_aware_p0b_binding():
    p0b_run, selected = _mask_aware_p0b_evidence(
        video_count=2,
        frame_count=16,
        legacy_pass_count=1,
    )
    policy = {"source_contract": {"expected_video_count": 2, "expected_frame_count": 16}}
    content = {
        "pass_video": {"status": "PASS"},
        "legacy_fail_video": {"status": "FAIL"},
    }

    result = _resolve_aligned_status_compatibility(
        content,
        p0b_run,
        selected,
        policy,
    )

    assert result["enabled"] is True
    assert result["accepted_aligned_status"] == "FAIL"
    assert result["accepted_legacy_fail_count"] == 1
    assert result["aligned_content_status_counts"] == {"FAIL": 1, "PASS": 1}


def test_legacy_fail_rejects_incomplete_or_weakened_mask_aware_binding():
    policy = {"source_contract": {"expected_video_count": 2, "expected_frame_count": 16}}
    content = {
        "pass_video": {"status": "PASS"},
        "legacy_fail_video": {"status": "FAIL"},
    }
    mutations = (
        (
            lambda selected: selected.pop("rich_provenance"),
            "P0B rich provenance must be an object",
        ),
        (
            lambda selected: selected["rich_provenance"].update(
                {"legacy_strict_status_used_as_gate": True}
            ),
            "P0B legacy strict gate mismatch",
        ),
        (
            lambda selected: selected["rich_provenance"]["coverage_policy_binding"][
                "decision"
            ].update({"status": "FAIL"}),
            "P0B coverage policy decision status mismatch",
        ),
    )

    for mutate, match in mutations:
        p0b_run, selected = _mask_aware_p0b_evidence(
            video_count=2,
            frame_count=16,
            legacy_pass_count=1,
        )
        mutate(selected)
        with pytest.raises(AuFidelityError, match=match):
            _resolve_aligned_status_compatibility(
                content,
                p0b_run,
                selected,
                policy,
            )


def test_aligned_content_rejects_unknown_status_even_with_mask_aware_binding():
    p0b_run, selected = _mask_aware_p0b_evidence(
        video_count=1,
        frame_count=8,
        legacy_pass_count=0,
    )

    with pytest.raises(AuFidelityError, match="unsupported statuses"):
        _resolve_aligned_status_compatibility(
            {"video": {"status": "ERROR"}},
            p0b_run,
            selected,
            {"source_contract": {"expected_video_count": 1, "expected_frame_count": 8}},
        )


def test_video_metrics_use_joint_mask_and_never_bridge_invalid_frame():
    values = (np.arange(240) % 20) * 0.25
    raw = _sequence(values, invalid=(100,))
    aligned = _sequence(values, invalid=(100,))

    coverage, metrics, lag_rows, cross_rows = compute_video_au_fidelity(
        raw=raw,
        aligned=aligned,
        policy=_policy(),
        video_id="203_1_Freeform_video_aligned",
        physical_split="train",
        task_name="Freeform",
    )

    assert len(coverage) == 3
    assert coverage[0]["matched_valid_count"] == 239
    assert coverage[0]["consecutive_matched_pair_count"] == 237
    assert metrics[0]["rho0"] == pytest.approx(1.0)
    assert metrics[0]["direction_rate"] == pytest.approx(1.0)
    assert metrics[0]["amplitude_ratio"] == pytest.approx(1.0)
    assert metrics[0]["lag_curve_estimable"] is True
    assert len(lag_rows) == 3 * 11
    assert len(cross_rows) == 3 * 3 * 3


def test_exact_raw_aligned_frame_identity_is_required():
    values = (np.arange(120) % 20) * 0.25
    raw = _sequence(values)
    aligned = _sequence(values)
    aligned = OpenFaceSequence(
        frames=aligned.frames + 1,
        success=aligned.success,
        confidence=aligned.confidence,
        au_values=aligned.au_values,
        schema_sha256=aligned.schema_sha256,
        schema_column_count=aligned.schema_column_count,
    )

    with pytest.raises(AuFidelityError, match="offsets are forbidden"):
        compute_video_au_fidelity(
            raw=raw,
            aligned=aligned,
            policy=_policy(),
            video_id="203_1_Freeform_video_aligned",
            physical_split="train",
            task_name="Freeform",
        )


def test_subject_aggregation_gives_tasks_equal_weight():
    video_rows = [
        {
            "split": "train",
            "subject_id": "203_1",
            "video_id": "203_1_Freeform_video_aligned",
            "task_name": "Freeform",
            "au": "AU12_r",
            "rho0": 0.9,
            "direction_rate": 0.8,
            "amplitude_ratio": 1.5,
            "lag_curve_estimable": True,
        },
        {
            "split": "train",
            "subject_id": "203_1",
            "video_id": "203_1_Northwind_video_aligned",
            "task_name": "Northwind",
            "au": "AU12_r",
            "rho0": 0.1,
            "direction_rate": 0.4,
            "amplitude_ratio": 0.5,
            "lag_curve_estimable": True,
        },
    ]

    subject_rows, _, _ = aggregate_au_fidelity_metrics(video_rows, [], [])

    assert subject_rows[0]["rho0"] == pytest.approx(0.5)
    assert subject_rows[0]["direction_rate"] == pytest.approx(0.6)
    assert subject_rows[0]["amplitude_ratio"] == pytest.approx(1.0)
    assert subject_rows[0]["task_count"] == 2
    assert subject_rows[0]["single_task_only"] is False
    assert subject_rows[0]["status"] == "ESTIMABLE"


def test_subject_aggregation_marks_metric_with_only_one_estimable_task():
    video_rows = [
        {
            "split": "train",
            "subject_id": "203_1",
            "video_id": f"203_1_{task}_video_aligned",
            "task_name": task,
            "au": "AU12_r",
            "rho0": 0.8 if task == "Freeform" else None,
            "direction_rate": None,
            "amplitude_ratio": None,
            "lag_curve_estimable": False,
        }
        for task in ("Freeform", "Northwind")
    ]

    subject_rows, _, _ = aggregate_au_fidelity_metrics(video_rows, [], [])

    assert subject_rows[0]["task_count"] == 2
    assert subject_rows[0]["rho_task_count"] == 1
    assert subject_rows[0]["single_task_only"] is True
    assert subject_rows[0]["single_task_metrics"] == "rho"
    assert subject_rows[0]["status"] == "PARTIAL_ESTIMABLE"


def test_subject_aggregation_is_partial_when_one_metric_is_entirely_unavailable():
    video_rows = [
        {
            "split": "train",
            "subject_id": "203_1",
            "video_id": f"203_1_{task}_video_aligned",
            "task_name": task,
            "au": "AU12_r",
            "rho0": 0.8,
            "direction_rate": 0.9,
            "amplitude_ratio": 1.0,
            "lag_curve_estimable": False,
        }
        for task in ("Freeform", "Northwind")
    ]

    subject_rows, _, _ = aggregate_au_fidelity_metrics(video_rows, [], [])

    assert subject_rows[0]["lag_task_count"] == 0
    assert subject_rows[0]["status"] == "PARTIAL_ESTIMABLE"


def test_train_only_decision_passes_all_frozen_gates():
    video_rows, subject_rows, lag_rows = _decision_rows()

    decision = build_au_fidelity_decision(
        policy=_policy(),
        video_rows=video_rows,
        subject_rows=subject_rows,
        lag_rows=lag_rows,
    )

    assert decision["audit_status"] == "PASS"
    assert decision["eligibility_status"] == "ELIGIBLE"
    assert decision["group_decision"]["eligible"] is True
    assert {row["status"] for row in decision["au_decisions"].values()} == {"PASS"}
    assert decision["au_decisions"]["AU12_r"]["observed"]["canonical_peak_lag"] == 0
    assert decision["training_authorized"] is False


def test_val_test_values_cannot_change_train_eligibility():
    video_rows, subject_rows, lag_rows = _decision_rows()
    baseline = build_au_fidelity_decision(
        policy=_policy(),
        video_rows=video_rows,
        subject_rows=subject_rows,
        lag_rows=lag_rows,
    )
    val_video, val_subject, val_lag = _decision_rows(
        subject_count=50,
        direction_by_au={au: 0.0 for au in CORE_AU_COLUMNS},
        split="val",
    )
    contaminated = build_au_fidelity_decision(
        policy=_policy(),
        video_rows=[*video_rows, *val_video],
        subject_rows=[*subject_rows, *val_subject],
        lag_rows=[*lag_rows, *val_lag],
    )

    assert contaminated["au_decisions"] == baseline["au_decisions"]
    assert contaminated["group_decision"] == baseline["group_decision"]
    assert contaminated["split_reports"]["val"]["AU12_r"]["decision_role"] == "REPORT_ONLY"


def test_metric_failure_and_evidence_block_are_distinct():
    video_rows, subject_rows, lag_rows = _decision_rows(
        direction_by_au={"AU12_r": 0.5, "AU14_r": 0.9, "AU15_r": 0.9}
    )
    failed = build_au_fidelity_decision(
        policy=_policy(),
        video_rows=video_rows,
        subject_rows=subject_rows,
        lag_rows=lag_rows,
    )
    small_video, small_subject, small_lag = _decision_rows(subject_count=10)
    blocked = build_au_fidelity_decision(
        policy=_policy(),
        video_rows=small_video,
        subject_rows=small_subject,
        lag_rows=small_lag,
    )

    assert failed["au_decisions"]["AU12_r"]["status"] == "FAIL_METRIC"
    assert failed["eligibility_status"] == "INELIGIBLE_METRIC"
    assert {row["status"] for row in blocked["au_decisions"].values()} == {
        "BLOCKED_EVIDENCE"
    }
    assert blocked["eligibility_status"] == "BLOCKED_INCOMPLETE_EVIDENCE"


def test_csv_projection_does_not_parse_pose_gaze_or_extension_values(tmp_path):
    path = tmp_path / "projected.csv"
    fieldnames = [
        "frame",
        "success",
        "confidence",
        *CORE_AU_COLUMNS,
        "pose_Rx",
        "gaze_angle_x",
        "AU04_r",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame in range(1, 4):
            writer.writerow(
                {
                    "frame": frame,
                    "success": 1,
                    "confidence": 0.9,
                    **{au: 0.25 * frame for au in CORE_AU_COLUMNS},
                    "pose_Rx": "must-not-parse",
                    "gaze_angle_x": "must-not-parse",
                    "AU04_r": "must-not-parse",
                }
            )
    schema = hashlib.sha256(",".join(fieldnames).encode("utf-8")).hexdigest()
    access_counts = Counter()

    sequence = _read_openface_sequence(
        path=path,
        expected_frames=3,
        content_record={
            "csv_size_bytes": path.stat().st_size,
            "csv_sha256": _sha(path),
            "csv_rows": 3,
            "schema_sha256": schema,
        },
        au_columns=CORE_AU_COLUMNS,
        value_minimum=0.0,
        value_maximum=5.0,
        source_name="raw",
        access_counts=access_counts,
    )

    assert sequence.frames.tolist() == [1, 2, 3]
    assert not any("pose" in key or "gaze" in key or "AU04" in key for key in access_counts)
    assert access_counts["raw:AU12_r"] == 3


def test_input_failure_writes_complete_blocked_artifacts(tmp_path):
    output_dir = tmp_path / "blocked"

    generated = run_privileged_behavior_au_fidelity(
        policy_path=POLICY_PATH,
        policy_sha256=_sha(POLICY_PATH),
        dataset_split_file=tmp_path / "missing_split.json",
        source_video_contract=tmp_path / "missing_source.csv",
        aligned_openface_root=tmp_path / "missing_aligned",
        aligned_run_manifest=tmp_path / "missing_aligned" / "_audit" / "run_manifest.json",
        raw_openface_root=tmp_path / "missing_raw",
        raw_run_manifest=tmp_path / "missing_raw" / "_audit" / "run_manifest.json",
        raw_run_manifest_sha256="0" * 64,
        p0b_run_manifest=tmp_path / "missing_p0b.json",
        p0b_selected_target_manifest=tmp_path / "missing_selected.json",
        output_dir=output_dir,
        project_root=PROJECT_ROOT,
        command_line="synthetic blocked audit",
    )

    assert len(generated) == 10
    assert all(path.is_file() for path in generated)
    decision = json.loads((output_dir / "au_fidelity_decision.json").read_text())
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert decision["audit_status"] == "BLOCKED"
    assert decision["eligibility_status"] == "NOT_EVALUABLE"
    assert decision["training_authorized"] is False
    assert manifest["status"] == "BLOCKED"
    assert manifest["source_data_modified"] is False

    with pytest.raises(FileExistsError, match="refusing overwrite/resume"):
        run_privileged_behavior_au_fidelity(
            policy_path=POLICY_PATH,
            policy_sha256=_sha(POLICY_PATH),
            dataset_split_file=tmp_path / "missing_split.json",
            source_video_contract=tmp_path / "missing_source.csv",
            aligned_openface_root=tmp_path / "missing_aligned",
            aligned_run_manifest=tmp_path / "missing_aligned" / "_audit" / "run_manifest.json",
            raw_openface_root=tmp_path / "missing_raw",
            raw_run_manifest=tmp_path / "missing_raw" / "_audit" / "run_manifest.json",
            raw_run_manifest_sha256="0" * 64,
            p0b_run_manifest=tmp_path / "missing_p0b.json",
            p0b_selected_target_manifest=tmp_path / "missing_selected.json",
            output_dir=output_dir,
            project_root=PROJECT_ROOT,
        )


def test_synthetic_end_to_end_binds_manifests_and_exact_pairs(tmp_path):
    split = {
        "train": [
            "203_1_Freeform_video_aligned",
            "203_1_Northwind_video_aligned",
        ],
        "val": [
            "303_1_Freeform_video_aligned",
            "303_1_Northwind_video_aligned",
        ],
        "test": [
            "403_1_Freeform_video_aligned",
            "403_1_Northwind_video_aligned",
        ],
    }
    split_path = tmp_path / "split.json"
    _write_json(split_path, split)
    video_ids = [video_id for physical_split in split.values() for video_id in physical_split]
    source_path = tmp_path / "source.csv"
    source_fields = [
        "split",
        "video_id",
        "raw_video_id",
        "task_name",
        "raw_video_path",
        "raw_openface_csv",
        "aligned_frame_count",
        "raw_frame_count",
        "frame_count_match",
        "raw_fps",
        "contract_status",
        "issues",
    ]
    source_rows = []
    for physical_split, split_videos in split.items():
        for video_id in split_videos:
            raw_id = video_id.removesuffix("_aligned")
            task = "Freeform" if "_Freeform_" in video_id else "Northwind"
            source_rows.append(
                {
                    "split": physical_split,
                    "video_id": video_id,
                    "raw_video_id": raw_id,
                    "task_name": task,
                    "raw_video_path": f"/not/accessed/{raw_id}.mp4",
                    "raw_openface_csv": "must-not-access",
                    "aligned_frame_count": 8,
                    "raw_frame_count": 8,
                    "frame_count_match": 1,
                    "raw_fps": 30,
                    "contract_status": "PASS",
                    "issues": "",
                }
            )
    _write_csv(source_path, source_fields, source_rows)

    aligned_root = tmp_path / "aligned"
    raw_root = tmp_path / "raw"
    aligned_content = []
    raw_content = []
    fieldnames = ["frame", "success", "confidence", *CORE_AU_COLUMNS]
    values = [0.0, 0.5, 1.0, 1.5, 1.0, 0.5, 0.0, 0.5]
    schema = hashlib.sha256(",".join(fieldnames).encode("utf-8")).hexdigest()
    for video_id in video_ids:
        raw_id = video_id.removesuffix("_aligned")
        rows = [
            {
                "frame": frame,
                "success": 1,
                "confidence": 0.95,
                **{au: values[frame - 1] for au in CORE_AU_COLUMNS},
            }
            for frame in range(1, 9)
        ]
        aligned_csv = aligned_root / f"{video_id}.csv"
        raw_csv = raw_root / f"{raw_id}.csv"
        _write_csv(aligned_csv, fieldnames, rows)
        _write_csv(raw_csv, fieldnames, rows)
        common_aligned = {
            "video_id": video_id,
            "csv_rows": 8,
            "schema_sha256": schema,
            "csv_size_bytes": aligned_csv.stat().st_size,
            "csv_sha256": _sha(aligned_csv),
            "status": "FAIL" if video_id == video_ids[0] else "PASS",
        }
        common_raw = {
            "video_id": video_id,
            "raw_video_id": raw_id,
            "csv_rows": 8,
            "schema_sha256": schema,
            "csv_size_bytes": raw_csv.stat().st_size,
            "csv_sha256": _sha(raw_csv),
            "status": "PASS",
        }
        aligned_content.append(common_aligned)
        raw_content.append(common_raw)
    content_fields = [
        "video_id",
        "csv_rows",
        "schema_sha256",
        "csv_size_bytes",
        "csv_sha256",
        "status",
    ]
    aligned_content_path = aligned_root / "_audit" / "csv_content_manifest.csv"
    raw_content_path = raw_root / "_audit" / "csv_content_manifest.csv"
    _write_csv(aligned_content_path, content_fields, aligned_content)
    _write_csv(
        raw_content_path,
        ["video_id", "raw_video_id", *content_fields[1:]],
        raw_content,
    )
    raw_summary_path = raw_root / "_audit" / "extraction_summary.json"
    _write_json(
        raw_summary_path,
        {
            "status": "PASS",
            "run_scope": "full_dataset",
            "video_count": 6,
            "fail_count": 0,
            "total_source_frames": 48,
            "total_csv_rows": 48,
            "schema_count": 1,
            "schema_consistent": True,
            "hog_file_count": 0,
            "tracked_video_file_count": 0,
            "generated_image_count": 0,
            "provenance_stable": True,
        },
    )

    package_fields = ["relative_path", "size", "sha256"]
    package_rows = [
        {"relative_path": "FeatureExtraction.exe", "size": 123, "sha256": "a" * 64}
    ]
    for root in (aligned_root, raw_root):
        _write_csv(root / "_audit" / "binary_manifest.csv", package_fields, package_rows)
        _write_csv(root / "_audit" / "model_manifest.csv", package_fields, package_rows)
    binary_manifest_sha256 = _sha(aligned_root / "_audit" / "binary_manifest.csv")
    model_manifest_sha256 = _sha(aligned_root / "_audit" / "model_manifest.csv")

    dummy_hashes = {
        "feature_extraction_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "openface_readme_sha256": "3" * 64,
        "au_predictor_manifest_sha256": "4" * 64,
        "source_video_contract_sha256": _sha(source_path),
    }
    common_manifest = {
        "feature_profile": "quality_2d_pose_au_no_gaze_no_hog_v1",
        "feature_arguments": ["-2Dfp", "-pose", "-aus"],
        **dummy_hashes,
        "gaze_requested": False,
        "hog_enabled": False,
        "tracked_video_enabled": False,
        "aligned_image_generation_enabled": False,
        "run_scope": "full_dataset",
        "selected_video_count": 6,
        "input_frame_count": 48,
        "binary_manifest_sha256": binary_manifest_sha256,
        "model_manifest_sha256": model_manifest_sha256,
    }
    aligned_manifest_path = aligned_root / "_audit" / "run_manifest.json"
    _write_json(
        aligned_manifest_path,
        {
            **common_manifest,
            "csv_content_manifest_sha256": _sha(aligned_content_path),
        },
    )
    raw_manifest_path = raw_root / "_audit" / "run_manifest.json"
    _write_json(
        raw_manifest_path,
        {
            **common_manifest,
            "input_mode": "raw_video",
            "status": "PASS",
            "selected_frame_count": 48,
            "output_root_must_be_new": True,
            "raw_openface_csv_path_field_access_count": 0,
            "historical_raw_openface_feature_file_open_count": 0,
            "selected_value_columns": list(CORE_AU_COLUMNS),
            "pose_value_access_count": 0,
            "gaze_value_access_count": 0,
            "extension_au_value_access_count": 0,
            "git_status_available": True,
            "provenance_stable": True,
            "script_unchanged": True,
            "source_video_contract_unchanged": True,
            "binary_files_unchanged": True,
            "model_files_unchanged": True,
            "git_provenance_unchanged": True,
            "script_sha256": _sha(
                PROJECT_ROOT
                / "scripts"
                / "run_openface_raw_au_fidelity_reference.ps1"
            ),
            "git_status_short": "",
            "csv_content_manifest_sha256": _sha(raw_content_path),
            "extraction_summary_sha256": _sha(raw_summary_path),
        },
    )

    p0b_run_path = tmp_path / "p0b_run.json"
    p0b_run, selected = _mask_aware_p0b_evidence(
        video_count=6,
        frame_count=48,
        legacy_pass_count=5,
    )
    _write_json(p0b_run_path, p0b_run)
    selected_path = tmp_path / "selected.json"
    selected.update(
        {
            "status": "PASS",
            "training_authorized": False,
            "groups": {"au": list(CORE_AU_COLUMNS)},
            "provenance": {
                "dataset_split_sha256": _sha(split_path),
                "source_run_manifest_sha256": _sha(aligned_manifest_path),
                "source_video_contract_sha256": _sha(source_path),
            },
            "raw_openface": {
                "feature_file_open_count": 0,
                "feature_value_access_count": 0,
            },
        }
    )
    _write_json(selected_path, selected)

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    policy["source_contract"].update(
        {
            "expected_video_count": 6,
            "expected_frame_count": 48,
            "required_dataset_split_sha256": _sha(split_path),
            "required_source_video_contract_sha256": _sha(source_path),
            "required_aligned_run_manifest_sha256": _sha(aligned_manifest_path),
            "required_p0b_run_manifest_sha256": _sha(p0b_run_path),
            "required_p0b_selected_target_manifest_sha256": _sha(selected_path),
            "required_feature_extraction_sha256": "1" * 64,
            "required_model_sha256": "2" * 64,
            "required_openface_readme_sha256": "3" * 64,
            "required_au_predictor_manifest_sha256": "4" * 64,
        }
    )
    policy["statistics"].update(
        {
            "minimum_matched_frames_per_video": 4,
            "minimum_unique_values_per_side": 2,
            "minimum_train_subjects": 1,
            "minimum_train_videos_per_task": 1,
        }
    )
    policy["statistics"]["spearman"]["bootstrap_samples"] = 10
    policy["statistics"]["direction"].update(
        {"minimum_events_per_video": 1, "minimum_train_subjects": 1}
    )
    policy["statistics"]["lag"].update(
        {"minimum_lag_frames": -1, "maximum_lag_frames": 1}
    )
    policy_path = tmp_path / "policy.json"
    _write_json(policy_path, policy)
    output_dir = tmp_path / "audit"

    generated = run_privileged_behavior_au_fidelity(
        policy_path=policy_path,
        policy_sha256=_sha(policy_path),
        dataset_split_file=split_path,
        source_video_contract=source_path,
        aligned_openface_root=aligned_root,
        aligned_run_manifest=aligned_manifest_path,
        raw_openface_root=raw_root,
        raw_run_manifest=raw_manifest_path,
        raw_run_manifest_sha256=_sha(raw_manifest_path),
        p0b_run_manifest=p0b_run_path,
        p0b_selected_target_manifest=selected_path,
        output_dir=output_dir,
        project_root=PROJECT_ROOT,
        command_line="synthetic complete audit",
    )

    assert len(generated) == 10
    decision = json.loads((output_dir / "au_fidelity_decision.json").read_text())
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    assert decision["audit_status"] == "PASS"
    assert decision["counts"]["processed_video_count"] == 6
    assert decision["counts"]["processed_frame_count"] == 48
    assert manifest["access_contract"]["pose_value_access_count"] == 0
    assert manifest["access_contract"]["extension_au_value_access_count"] == 0
    assert manifest["access_contract"]["openface_value_access_counts"]["raw:AU12_r"] == 48
    compatibility = manifest["aligned_mask_aware_compatibility"]
    assert compatibility["enabled"] is True
    assert compatibility["accepted_legacy_fail_count"] == 1
    assert compatibility["aligned_content_status_counts"] == {"FAIL": 1, "PASS": 5}
    assert (output_dir / "tables" / "au_fidelity_issues.csv").read_text().count("\n") == 1


def test_audit_output_cannot_overlap_an_openface_input_root(tmp_path):
    aligned_root = (tmp_path / "aligned").resolve()
    raw_root = (tmp_path / "raw").resolve()

    with pytest.raises(AuFidelityError, match="output must not overlap"):
        run_privileged_behavior_au_fidelity(
            policy_path=POLICY_PATH,
            policy_sha256=_sha(POLICY_PATH),
            dataset_split_file=tmp_path / "missing_split.json",
            source_video_contract=tmp_path / "missing_source.csv",
            aligned_openface_root=aligned_root,
            aligned_run_manifest=aligned_root / "_audit" / "run_manifest.json",
            raw_openface_root=raw_root,
            raw_run_manifest=raw_root / "_audit" / "run_manifest.json",
            raw_run_manifest_sha256="0" * 64,
            p0b_run_manifest=tmp_path / "missing_p0b.json",
            p0b_selected_target_manifest=tmp_path / "missing_selected.json",
            output_dir=raw_root / "audit_output",
            project_root=PROJECT_ROOT,
        )

    assert not (raw_root / "audit_output").exists()


def test_cli_requires_every_provenance_input():
    parser = build_parser()
    required = {
        action.dest
        for action in parser._actions
        if getattr(action, "required", False)
    }
    assert required == {
        "policy",
        "policy_sha256",
        "dataset_split_file",
        "source_video_contract",
        "aligned_openface_root",
        "aligned_run_manifest",
        "raw_openface_root",
        "raw_run_manifest",
        "raw_run_manifest_sha256",
        "p0b_run_manifest",
        "p0b_selected_target_manifest",
        "output_dir",
    }
