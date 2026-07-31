import csv
import hashlib
import json
from pathlib import Path

import pytest

import src.diagnostics.privileged_behavior_p0 as p0_module
from scripts.audit_privileged_behavior_contract import build_parser
from src.diagnostics.privileged_behavior_contract import REQUIRED_OPENFACE_COLUMNS
from src.diagnostics.privileged_behavior_p0 import (
    FROZEN_BEHAVIOR_PROFILE,
    FROZEN_FEATURE_EXTRACTION_SHA256,
    FROZEN_MIN_SUCCESS_RATIO,
    FROZEN_MODEL_SHA256,
    FROZEN_OPENFACE_PACKAGE,
    FROZEN_OPENFACE_README_SHA256,
    _paths_cross_platform_equivalent,
    run_privileged_behavior_p0,
)


VIDEO_ID = "203_1_Freeform_video_aligned"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_split(path):
    path.write_text(
        json.dumps({"train": [VIDEO_ID], "val": [], "test": []}),
        encoding="utf-8",
    )


def _write_images(root, count=3, video_id=VIDEO_ID):
    video_dir = root / video_id
    video_dir.mkdir(parents=True)
    for frame in range(1, count + 1):
        (video_dir / f"frame_det_00_{frame:06d}.jpg").write_bytes(b"audit-only")


def _write_source_contract(path, count=3, fps=30.0):
    fields = [
        "split",
        "source_split",
        "video_id",
        "raw_video_id",
        "task_name",
        "raw_video_path",
        "raw_openface_csv",
        "aligned_frame_count",
        "raw_frame_count",
        "frame_count_match",
        "raw_width",
        "raw_height",
        "raw_fps",
        "pure_black_run_count",
        "pure_black_frame_count",
        "contract_status",
        "issues",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "source_split": "train",
                "video_id": VIDEO_ID,
                "raw_video_id": VIDEO_ID.removesuffix("_aligned"),
                "task_name": "Freeform",
                "raw_video_path": "/must/not/be/opened/raw.mp4",
                "raw_openface_csv": "/must/not/be/opened/raw_openface.csv",
                "aligned_frame_count": count,
                "raw_frame_count": count,
                "frame_count_match": 1,
                "raw_width": 640,
                "raw_height": 480,
                "raw_fps": fps,
                "pure_black_run_count": 0,
                "pure_black_frame_count": 0,
                "contract_status": "PASS",
                "issues": "",
            }
        )


def _write_source_manifest(path, image_root, source_contract, *, behavior=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audit": "aligned image OpenFace extraction",
        "image_root": str(image_root),
        "arguments": ["-fdir", "<video_dir>", "-out_dir", "<output>", "-2Dfp"],
        "feature_extraction_sha256": "feature-sha",
        "model_sha256": "model-sha",
        "integrity_status": "EXACT_PASS",
    }
    if behavior:
        openface_root = path.parent.parent
        project_root = path.parents[2]
        candidate_manifest = path.parent / "image_integrity_candidate_manifest.csv"
        image_rows = []
        for video_dir in sorted(item for item in image_root.iterdir() if item.is_dir()):
            for image_path in sorted(
                item
                for item in video_dir.iterdir()
                if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg"}
            ):
                image_rows.append(
                    {
                        "relative_path": image_path.relative_to(image_root).as_posix(),
                        "file_size": image_path.stat().st_size,
                        "file_sha256": _sha(image_path),
                    }
                )
        with candidate_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("relative_path", "file_size", "file_sha256"),
            )
            writer.writeheader()
            writer.writerows(image_rows)
        candidate_manifest_sha256 = _sha(candidate_manifest)
        comparison_summary = path.parent / "image_integrity_comparison_summary.json"
        comparison_summary.write_text(
            json.dumps(
                {
                    "audit": "aligned image cross-machine comparison",
                    "status": "EXACT_PASS",
                    "reference_manifest": candidate_manifest.relative_to(
                        project_root
                    ).as_posix(),
                    "candidate_manifest": candidate_manifest.relative_to(
                        project_root
                    ).as_posix(),
                    "reference_manifest_sha256": candidate_manifest_sha256,
                    "candidate_manifest_sha256": candidate_manifest_sha256,
                    "reference_count": len(image_rows),
                    "candidate_count": len(image_rows),
                    "status_counts": {"EXACT_MATCH": len(image_rows)},
                }
            ),
            encoding="utf-8",
        )
        csv_paths = sorted(openface_root.glob("*.csv"))
        content_rows = []
        total_rows = 0
        schema_hashes = set()
        for csv_path in csv_paths:
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.reader(handle, skipinitialspace=True)
                header = [value.strip() for value in next(reader)]
                row_count = sum(1 for _row in reader)
            schema_sha256 = _sha_text(",".join(header))
            schema_hashes.add(schema_sha256)
            total_rows += row_count
            content_rows.append(
                {
                    "video_id": csv_path.stem,
                    "csv_rows": row_count,
                    "schema_sha256": schema_sha256,
                    "csv_size_bytes": csv_path.stat().st_size,
                    "csv_sha256": _sha(csv_path),
                    "status": "PASS",
                }
            )
        content_manifest = path.parent / "csv_content_manifest.csv"
        with content_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "video_id",
                    "csv_rows",
                    "schema_sha256",
                    "csv_size_bytes",
                    "csv_sha256",
                    "status",
                ),
            )
            writer.writeheader()
            writer.writerows(content_rows)
        content_manifest_sha256 = _sha(content_manifest)
        summary = {
            "audit": "OpenFace original aligned-image behavior feature extraction summary",
            "feature_profile": FROZEN_BEHAVIOR_PROFILE,
            "run_scope": "full_dataset",
            "video_count": len(csv_paths),
            "pass_count": len(csv_paths),
            "fail_count": 0,
            "total_images": total_rows,
            "total_csv_rows": total_rows,
            "total_successes": total_rows,
            "success_ratio": 1.0,
            "min_required_success_ratio": FROZEN_MIN_SUCCESS_RATIO,
            "schema_count": len(schema_hashes),
            "schema_consistent": len(schema_hashes) == 1,
            "hog_file_count": 0,
            "tracked_video_file_count": 0,
            "regenerated_aligned_image_count": 0,
            "status": "PASS",
            "run_manifest": str(path),
            "csv_content_manifest": str(content_manifest),
            "csv_content_manifest_sha256": content_manifest_sha256,
        }
        extraction_summary = path.parent / "extraction_summary.json"
        extraction_summary.write_text(json.dumps(summary), encoding="utf-8")
        payload.update(
            audit="OpenFace original aligned-image behavior feature extraction",
            script_sha256=_sha(
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "run_openface_aligned_behavior_features.ps1"
            ),
            git_commit="4" * 40,
            git_branch="dev",
            git_status_available=True,
            git_status_short="",
            openface_package_directory=FROZEN_OPENFACE_PACKAGE,
            feature_extraction_sha256=FROZEN_FEATURE_EXTRACTION_SHA256,
            expected_feature_extraction_sha256=FROZEN_FEATURE_EXTRACTION_SHA256,
            model_sha256=FROZEN_MODEL_SHA256,
            expected_model_sha256=FROZEN_MODEL_SHA256,
            openface_readme_sha256=FROZEN_OPENFACE_README_SHA256,
            expected_openface_readme_sha256=FROZEN_OPENFACE_README_SHA256,
            arguments=[
                "-fdir",
                "<video_dir>",
                "-out_dir",
                str(openface_root),
                "-2Dfp",
                "-pose",
                "-aus",
                "-mloc",
                "<model>",
            ],
            feature_profile=FROZEN_BEHAVIOR_PROFILE,
            behavior_source_columns=[
                "AU12_r",
                "AU14_r",
                "AU15_r",
                "pose_Rx",
                "pose_Ry",
                "pose_Rz",
            ],
            run_scope="full_dataset",
            min_success_ratio=FROZEN_MIN_SUCCESS_RATIO,
            input_video_count=len(csv_paths),
            selected_video_count=len(csv_paths),
            input_frame_count=total_rows,
            max_videos=0,
            source_video_contract_sha256=_sha(source_contract),
            integrity_comparison_summary=comparison_summary.relative_to(
                project_root
            ).as_posix(),
            integrity_comparison_summary_sha256=_sha(comparison_summary),
            integrity_reference_manifest_sha256=candidate_manifest_sha256,
            integrity_candidate_manifest_sha256=candidate_manifest_sha256,
            integrity_reference_count=total_rows,
            integrity_candidate_count=total_rows,
            csv_timestamp_authorized_for_head_velocity=False,
            gaze_requested=False,
            gaze_training_access_count=0,
            gaze_normalizer_access_count=0,
            gaze_loss_access_count=0,
            csv_content_manifest=str(content_manifest),
            csv_content_manifest_sha256=content_manifest_sha256,
            extraction_summary=str(extraction_summary),
            extraction_summary_sha256=_sha(extraction_summary),
        )
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_openface_csv(
    path,
    *,
    rich,
    au_scale=1.0,
    pose_scale=1.0,
    frame_count=3,
    failed_frames=(),
):
    if not rich:
        fields = ["frame", "timestamp", "confidence", "success", "x_0", "y_0"]
        rows = [
            {"frame": frame, "timestamp": 0, "confidence": 0.99, "success": 1, "x_0": 1, "y_0": 2}
            for frame in range(1, frame_count + 1)
        ]
    else:
        fields = list(REQUIRED_OPENFACE_COLUMNS)
        rows = [
            {
                "frame": 1,
                "timestamp": 0,
                "confidence": 0.99,
                "success": 1,
                "AU12_r": 1 * au_scale,
                "AU14_r": 2 * au_scale,
                "AU15_r": 3 * au_scale,
                "pose_Rx": 0,
                "pose_Ry": 0,
                "pose_Rz": 0,
            },
            {
                "frame": 2,
                "timestamp": 0,
                "confidence": 0.99,
                "success": 1,
                "AU12_r": 2 * au_scale,
                "AU14_r": 4 * au_scale,
                "AU15_r": 6 * au_scale,
                "pose_Rx": 0.1 * pose_scale,
                "pose_Ry": 0.2 * pose_scale,
                "pose_Rz": 0.3 * pose_scale,
            },
            {
                "frame": 3,
                "timestamp": 0,
                "confidence": 0.99,
                "success": 1,
                "AU12_r": 4 * au_scale,
                "AU14_r": 8 * au_scale,
                "AU15_r": 12 * au_scale,
                "pose_Rx": 0.3 * pose_scale,
                "pose_Ry": 0.5 * pose_scale,
                "pose_Rz": 0.9 * pose_scale,
            },
        ]
        extension_values = {
            4: (4.5, 4.8, 4.0, 0.6, 1.0, 1.8),
            5: (4.8, 4.9, 4.5, 1.0, 1.7, 3.0),
        }
        for frame in range(4, frame_count + 1):
            au12, au14, au15, pose_rx, pose_ry, pose_rz = extension_values.get(
                frame,
                (1.0, 2.0, 3.0, 0.1, 0.2, 0.3),
            )
            rows.append(
                {
                    "frame": frame,
                    "timestamp": 0,
                    "confidence": 0.99,
                    "success": 1,
                    "AU12_r": au12 * au_scale,
                    "AU14_r": au14 * au_scale,
                    "AU15_r": au15 * au_scale,
                    "pose_Rx": pose_rx * pose_scale,
                    "pose_Ry": pose_ry * pose_scale,
                    "pose_Rz": pose_rz * pose_scale,
                }
            )
        rows = rows[:frame_count]
    failed = set(failed_frames)
    for row in rows:
        if row["frame"] in failed:
            row["success"] = 0
            row["confidence"] = 0.2
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_inputs(tmp_path, *, rich):
    split = tmp_path / "dataset_split.json"
    image_root = tmp_path / "face_images"
    openface_root = tmp_path / "openface"
    source_contract = tmp_path / "source_video_contract.csv"
    source_manifest = openface_root / "_audit" / "run_manifest.json"
    _write_split(split)
    _write_images(image_root)
    openface_root.mkdir()
    _write_source_contract(source_contract)
    _write_openface_csv(openface_root / f"{VIDEO_ID}.csv", rich=rich)
    _write_source_manifest(
        source_manifest,
        image_root,
        source_contract,
        behavior=rich,
    )
    return split, image_root, openface_root, source_contract, source_manifest


def _build_legacy_fail_inputs(tmp_path):
    split = tmp_path / "dataset_split.json"
    image_root = tmp_path / "face_images"
    openface_root = tmp_path / "openface"
    source_contract = tmp_path / "source_video_contract.csv"
    source_manifest = openface_root / "_audit" / "run_manifest.json"
    _write_split(split)
    _write_images(image_root, count=5)
    openface_root.mkdir()
    _write_source_contract(source_contract, count=5)
    _write_openface_csv(
        openface_root / f"{VIDEO_ID}.csv",
        rich=True,
        frame_count=5,
        failed_frames=(5,),
    )
    _write_source_manifest(
        source_manifest,
        image_root,
        source_contract,
        behavior=True,
    )

    content_path = source_manifest.parent / "csv_content_manifest.csv"
    content_rows = _read_csv(content_path)
    content_rows[0]["status"] = "FAIL"
    _write_csv(content_path, content_rows[0].keys(), content_rows)

    summary_path = source_manifest.parent / "extraction_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "pass_count": 0,
            "fail_count": 1,
            "total_successes": 4,
            "success_ratio": 0.8,
            "status": "FAIL",
            "csv_content_manifest_sha256": _sha(content_path),
        }
    )
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest.update(
        {
            "csv_content_manifest_sha256": _sha(content_path),
            "extraction_summary_sha256": _sha(summary_path),
        }
    )
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    _write_csv(
        source_manifest.parent / "video_run_summary.csv",
        (
            "video_id",
            "image_count",
            "success_count",
            "success_ratio",
            "schema_column_count",
            "status",
            "issues",
        ),
        [
            {
                "video_id": VIDEO_ID,
                "image_count": 5,
                "success_count": 4,
                "success_ratio": 0.8,
                "schema_column_count": 182,
                "status": "FAIL",
                "issues": "success_ratio:0.8<0.995",
            }
        ],
    )
    _write_csv(
        source_manifest.parent / "input_frame_contract.csv",
        ("video_id", "image_count", "selected_for_run", "status", "issues"),
        [
            {
                "video_id": VIDEO_ID,
                "image_count": 5,
                "selected_for_run": True,
                "status": "PASS",
                "issues": "",
            }
        ],
    )
    return split, image_root, openface_root, source_contract, source_manifest


def _write_full_coverage_binding(
    tmp_path,
    *,
    split,
    openface_root,
    source_contract,
    source_manifest,
    decision_count_updates=None,
):
    policy = {
        "policy_id": "pb_behavior_source_coverage_v1",
        "policy_version": 1,
        "status": "FROZEN",
        "label_blind": True,
        "source_contract": {
            "feature_profile": FROZEN_BEHAVIOR_PROFILE,
            "confidence_threshold": 0.8,
            "legacy_strict_success_ratio": FROZEN_MIN_SUCCESS_RATIO,
            "legacy_strict_status_is_diagnostic_only": True,
            "expected_csv_column_count": 182,
            "required_au_columns": ["AU12_r", "AU14_r", "AU15_r"],
            "required_pose_columns": ["pose_Rx", "pose_Ry", "pose_Rz"],
            "required_source_video_contract_sha256": _sha(source_contract),
        },
        "pilot_evidence": {},
        "full_source_expectations": {},
        "pilot_feasibility_thresholds": {},
        "physical_train_full_thresholds": {},
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
    policy_path = tmp_path / "coverage_policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    counts = {
        "video_count": 1,
        "frame_count": 5,
        "quality_valid_count": 4,
        "au_valid_count": 4,
        "head_valid_pair_count": 3,
        "blocking_issue_count": 0,
        "warning_count": 0,
    }
    counts.update(decision_count_updates or {})
    decision = {
        "policy_id": policy["policy_id"],
        "policy_version": policy["policy_version"],
        "policy_sha256": _sha(policy_path),
        "scope": "full",
        "status": "PASS_FULL_SOURCE_COVERAGE",
        "source_structural_complete": True,
        "mask_aware_coverage_pass": True,
        "full_train_aggregate_thresholds_evaluated": True,
        "legacy_strict_status_observed": "FAIL",
        "legacy_strict_status_used_as_gate": False,
        "full_rich_authorized": False,
        "p0b_authorized": False,
        "training_authorized": False,
        "next_action": "REQUEST_SEPARATE_P0B_AUTHORIZATION",
        "nonblocking_future_full_warnings": [],
        "counts": counts,
        "label_access": {
            "bdi_label_access_count": 0,
            "prediction_access_count": 0,
            "checkpoint_access_count": 0,
            "validation_test_threshold_callback_count": 0,
        },
    }
    a2_root = tmp_path / "a2_full"
    a2_root.mkdir()
    decision_path = a2_root / "coverage_policy_decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    audit_root = source_manifest.parent
    coverage_implementation = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "diagnostics"
        / "privileged_behavior_coverage_policy.py"
    )
    a2_manifest = {
        "audit": "PB-P0A2 mask-aware source/coverage policy validation",
        "status": decision["status"],
        "inputs": {
            "policy": {"path": str(policy_path), "sha256": _sha(policy_path)},
            "dataset_split": {"path": str(split), "sha256": _sha(split)},
            "extraction_root": str(openface_root),
            "run_manifest": {
                "path": str(source_manifest),
                "sha256": _sha(source_manifest),
            },
            "extraction_summary": {
                "path": str(audit_root / "extraction_summary.json"),
                "sha256": _sha(audit_root / "extraction_summary.json"),
            },
            "video_run_summary": {
                "path": str(audit_root / "video_run_summary.csv"),
                "sha256": _sha(audit_root / "video_run_summary.csv"),
            },
            "input_frame_contract": {
                "path": str(audit_root / "input_frame_contract.csv"),
                "sha256": _sha(audit_root / "input_frame_contract.csv"),
            },
            "csv_content_manifest": {
                "path": str(audit_root / "csv_content_manifest.csv"),
                "sha256": _sha(audit_root / "csv_content_manifest.csv"),
            },
        },
        "outputs": {
            decision_path.name: {
                "path": str(decision_path),
                "sha256": _sha(decision_path),
            }
        },
        "implementation": {
            "path": str(coverage_implementation),
            "sha256": _sha(coverage_implementation),
        },
        "read_only_inputs": True,
        "source_data_modified": False,
        "model_modified": False,
        "training_started": False,
        "full_rich_authorized": False,
        "p0b_authorized": False,
        "training_authorized": False,
    }
    (a2_root / "run_manifest.json").write_text(
        json.dumps(a2_manifest), encoding="utf-8"
    )
    return policy_path, decision_path, _sha(decision_path)


def _build_multisplit_rich_inputs(tmp_path):
    videos = [
        ("train", "203_1_Freeform_video_aligned", "Freeform", 1.0),
        ("val", "205_1_Freeform_video_aligned", "Freeform", 1_000_000.0),
        ("test", "206_1_Northwind_video_aligned", "Northwind", 2_000_000.0),
    ]
    split = tmp_path / "dataset_split.json"
    split.write_text(
        json.dumps(
            {
                split_name: [video_id for row_split, video_id, _task, _scale in videos if row_split == split_name]
                for split_name in ("train", "val", "test")
            }
        ),
        encoding="utf-8",
    )
    image_root = tmp_path / "face_images"
    openface_root = tmp_path / "openface"
    openface_root.mkdir()
    source_contract = tmp_path / "source_video_contract.csv"
    fields = [
        "split",
        "source_split",
        "video_id",
        "raw_video_id",
        "task_name",
        "raw_video_path",
        "raw_openface_csv",
        "aligned_frame_count",
        "raw_frame_count",
        "frame_count_match",
        "raw_width",
        "raw_height",
        "raw_fps",
        "pure_black_run_count",
        "pure_black_frame_count",
        "contract_status",
        "issues",
    ]
    with source_contract.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split_name, video_id, task_name, scale in videos:
            _write_images(image_root, video_id=video_id)
            _write_openface_csv(
                openface_root / f"{video_id}.csv",
                rich=True,
                au_scale=scale,
                pose_scale=1.0 if split_name == "train" else 3.0,
            )
            writer.writerow(
                {
                    "split": split_name,
                    "source_split": "dev" if split_name == "val" else split_name,
                    "video_id": video_id,
                    "raw_video_id": video_id.removesuffix("_aligned"),
                    "task_name": task_name,
                    "raw_video_path": f"/must/not/be/opened/{video_id}.mp4",
                    "raw_openface_csv": f"/must/not/be/opened/{video_id}.csv",
                    "aligned_frame_count": 3,
                    "raw_frame_count": 3,
                    "frame_count_match": 1,
                    "raw_width": 640,
                    "raw_height": 480,
                    "raw_fps": 30,
                    "pure_black_run_count": 0,
                    "pure_black_frame_count": 0,
                    "contract_status": "PASS",
                    "issues": "",
                }
            )
    source_manifest = openface_root / "_audit" / "run_manifest.json"
    _write_source_manifest(
        source_manifest,
        image_root,
        source_contract,
        behavior=True,
    )
    return split, image_root, openface_root, source_contract, source_manifest


def _rewrite_json(path, transform):
    payload = json.loads(path.read_text(encoding="utf-8"))
    transform(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _issue_codes(output):
    return {
        row["issue_code"]
        for row in _read_csv(output / "tables" / "behavior_contract_issues.csv")
    }


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_cross_platform_path_equivalence_preserves_posix_case_and_supports_wsl_unc():
    assert _paths_cross_platform_equivalent(
        r"D:\Data\FaceImages", "/mnt/d/data/faceimages"
    )
    assert _paths_cross_platform_equivalent(
        r"\\wsl.localhost\Ubuntu\home\zhen\Data",
        "/home/zhen/Data",
    )
    assert _paths_cross_platform_equivalent(
        r"\\wsl$\Ubuntu\home\zhen\Data",
        "/home/zhen/Data",
    )
    assert not _paths_cross_platform_equivalent(
        "/home/zhen/Data", "/home/zhen/data"
    )


def test_frozen_coverage_policy_hash_pin_matches_reviewed_config():
    policy = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "behavior_alignment"
        / "behavior_source_coverage_policy_v1.json"
    )
    assert _sha(policy) == p0_module.FROZEN_COVERAGE_POLICY_SHA256


@pytest.mark.parametrize(
    ("owner", "field", "expected_issue"),
    [
        (
            "source_manifest",
            "extraction_summary",
            "rich_source_extraction_summary_path_mismatch",
        ),
        (
            "source_manifest",
            "csv_content_manifest",
            "rich_source_csv_content_manifest_path_mismatch",
        ),
        (
            "summary",
            "run_manifest",
            "rich_extraction_summary_run_manifest_path_mismatch",
        ),
        (
            "summary",
            "csv_content_manifest",
            "rich_extraction_summary_csv_content_manifest_path_mismatch",
        ),
    ],
)
def test_rich_provenance_paths_must_identify_adjacent_files(
    tmp_path, owner, field, expected_issue
):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    if owner == "source_manifest":
        _rewrite_json(
            source_manifest,
            lambda payload: payload.__setitem__(field, "/different/audit/file"),
        )
    else:
        summary_path = source_manifest.parent / "extraction_summary.json"
        _rewrite_json(
            summary_path,
            lambda payload: payload.__setitem__(field, "/different/audit/file"),
        )
        _rewrite_json(
            source_manifest,
            lambda payload: payload.__setitem__(
                "extraction_summary_sha256", _sha(summary_path)
            ),
        )

    output = tmp_path / f"bad_path_{owner}_{field}"
    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    assert expected_issue in _issue_codes(output)
    selected = json.loads(
        (output / "selected_target_manifest.json").read_text(encoding="utf-8")
    )
    assert selected["status"] == "BLOCKED"


def test_rich_success_threshold_is_frozen_and_low_claimed_coverage_blocks(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    summary_path = source_manifest.parent / "extraction_summary.json"
    _rewrite_json(
        summary_path,
        lambda payload: payload.update(
            {
                "total_successes": 2,
                "success_ratio": 2 / 3,
                "min_required_success_ratio": 0.0,
            }
        ),
    )
    _rewrite_json(
        source_manifest,
        lambda payload: payload.update(
            {
                "min_success_ratio": 0.0,
                "extraction_summary_sha256": _sha(summary_path),
            }
        ),
    )
    output = tmp_path / "low_success_threshold"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    issues = _issue_codes(output)
    assert "rich_source_min_success_ratio_mismatch" in issues
    assert "rich_extraction_summary_min_required_success_ratio_mismatch" in issues
    assert "rich_extraction_summary_success_counts_mismatch" in issues


def test_rich_current_image_bytes_must_match_candidate_manifest(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    image_path = next(image_root.rglob("*.jpg"))
    original = image_path.read_bytes()
    tampered = b"other-byte"
    assert len(tampered) == len(original)
    image_path.write_bytes(tampered)
    output = tmp_path / "image_byte_tamper"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    assert "rich_current_image_content_binding_mismatch" in _issue_codes(output)
    selected = json.loads(
        (output / "selected_target_manifest.json").read_text(encoding="utf-8")
    )
    binding = selected["rich_provenance"]["current_image_integrity"]
    assert selected["status"] == "BLOCKED"
    assert binding["status"] == "BLOCKED"
    assert binding["size_mismatch_count"] == 0
    assert binding["sha256_mismatch_count"] == 1
    assert binding["exact_match_count"] == 2


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        (
            "summary_path",
            "rich_source_integrity_comparison_summary_path_mismatch",
        ),
        (
            "summary_hash",
            "rich_source_integrity_comparison_summary_sha256_mismatch",
        ),
        (
            "candidate_path",
            "rich_integrity_candidate_manifest_path_mismatch",
        ),
        (
            "candidate_hash",
            "rich_integrity_candidate_manifest_sha256_mismatch",
        ),
    ],
)
def test_rich_image_integrity_provenance_path_and_hash_failures_block(
    tmp_path, mutation, expected_issue
):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    comparison_summary = (
        source_manifest.parent / "image_integrity_comparison_summary.json"
    )
    candidate_manifest = (
        source_manifest.parent / "image_integrity_candidate_manifest.csv"
    )
    if mutation == "summary_path":
        _rewrite_json(
            source_manifest,
            lambda payload: payload.__setitem__(
                "integrity_comparison_summary", "missing/summary.json"
            ),
        )
    elif mutation == "summary_hash":
        _rewrite_json(
            source_manifest,
            lambda payload: payload.__setitem__(
                "integrity_comparison_summary_sha256", "bad"
            ),
        )
    elif mutation == "candidate_path":
        _rewrite_json(
            comparison_summary,
            lambda payload: payload.__setitem__(
                "candidate_manifest", "missing/candidate.csv"
            ),
        )
        _rewrite_json(
            source_manifest,
            lambda payload: payload.__setitem__(
                "integrity_comparison_summary_sha256", _sha(comparison_summary)
            ),
        )
    else:
        candidate_manifest.write_bytes(candidate_manifest.read_bytes() + b"\n")
    output = tmp_path / f"image_integrity_{mutation}"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    assert expected_issue in _issue_codes(output)
    selected = json.loads(
        (output / "selected_target_manifest.json").read_text(encoding="utf-8")
    )
    assert selected["status"] == "BLOCKED"
    assert selected["rich_provenance"]["current_image_integrity"]["status"] == "BLOCKED"


@pytest.mark.parametrize(
    ("owner", "field", "value", "expected_issue"),
    [
        (
            "source_manifest",
            "input_video_count",
            1.5,
            "rich_source_input_video_count_mismatch",
        ),
        (
            "source_manifest",
            "max_videos",
            0.5,
            "rich_source_max_videos_mismatch",
        ),
        (
            "summary",
            "video_count",
            1.5,
            "rich_extraction_summary_video_count_mismatch",
        ),
        (
            "summary",
            "fail_count",
            0.5,
            "rich_extraction_summary_fail_count_mismatch",
        ),
        (
            "summary",
            "total_successes",
            2.5,
            "rich_extraction_summary_success_counts_mismatch",
        ),
    ],
)
def test_rich_integer_count_fields_reject_fractional_values(
    tmp_path, owner, field, value, expected_issue
):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    if owner == "source_manifest":
        _rewrite_json(
            source_manifest,
            lambda payload: payload.__setitem__(field, value),
        )
    else:
        summary_path = source_manifest.parent / "extraction_summary.json"
        _rewrite_json(
            summary_path,
            lambda payload: payload.__setitem__(field, value),
        )
        _rewrite_json(
            source_manifest,
            lambda payload: payload.__setitem__(
                "extraction_summary_sha256", _sha(summary_path)
            ),
        )
    output = tmp_path / f"fractional_{owner}_{field}"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    assert expected_issue in _issue_codes(output)


def test_partial_rich_schema_cannot_pass_when_train_statistics_are_available(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = (
        _build_multisplit_rich_inputs(tmp_path)
    )
    _write_openface_csv(
        openface_root / "205_1_Freeform_video_aligned.csv",
        rich=False,
    )
    output = tmp_path / "partial_rich"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    selected = json.loads(
        (output / "selected_target_manifest.json").read_text(encoding="utf-8")
    )
    issues = _read_csv(output / "tables" / "behavior_contract_issues.csv")
    run_manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert selected["status"] == "BLOCKED"
    assert selected["normalization"]["available"] is True
    assert selected["rich_provenance"]["status"] == "NOT_APPLICABLE"
    assert {row["issue_code"] for row in issues} == {"missing_behavior_columns"}
    assert run_manifest["counts"]["rich_schema_video_count"] == 2
    assert run_manifest["counts"]["blocking_issue_count"] == sum(
        row["severity"] == "BLOCKER" for row in issues
    )


def test_rich_extraction_summary_hash_binding_detects_unbound_change(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    summary_path = source_manifest.parent / "extraction_summary.json"
    _rewrite_json(summary_path, lambda payload: payload.__setitem__("note", "changed"))
    output = tmp_path / "summary_hash_mismatch"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    assert "rich_extraction_summary_sha256_mismatch" in _issue_codes(output)


def test_rich_csv_content_hash_detects_same_size_tamper(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    csv_path = openface_root / f"{VIDEO_ID}.csv"
    original = csv_path.read_bytes()
    tampered = original.replace(b"1.0", b"9.0", 1)
    assert tampered != original
    assert len(tampered) == len(original)
    csv_path.write_bytes(tampered)
    output = tmp_path / "same_size_csv_tamper"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    mismatch = next(
        row
        for row in _read_csv(output / "tables" / "behavior_contract_issues.csv")
        if row["issue_code"] == "rich_csv_content_mismatch"
    )
    assert "csv_sha256" in mismatch["detail"]
    assert "csv_size_bytes" not in mismatch["detail"]


@pytest.mark.parametrize(
    ("owner", "expected_issue"),
    [
        ("summary", "rich_csv_content_manifest_sha256_summary_mismatch"),
        (
            "source_manifest",
            "rich_csv_content_manifest_sha256_source_manifest_mismatch",
        ),
    ],
)
def test_rich_csv_content_manifest_hash_is_bound_by_both_final_documents(
    tmp_path, owner, expected_issue
):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    if owner == "summary":
        summary_path = source_manifest.parent / "extraction_summary.json"
        _rewrite_json(
            summary_path,
            lambda payload: payload.__setitem__("csv_content_manifest_sha256", "bad"),
        )
        _rewrite_json(
            source_manifest,
            lambda payload: payload.__setitem__(
                "extraction_summary_sha256", _sha(summary_path)
            ),
        )
    else:
        _rewrite_json(
            source_manifest,
            lambda payload: payload.__setitem__("csv_content_manifest_sha256", "bad"),
        )
    output = tmp_path / f"content_manifest_hash_{owner}"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    assert expected_issue in _issue_codes(output)


def test_landmark_only_schema_emits_complete_blocked_artifacts_without_raw_or_gaze_access(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=False
    )
    output = tmp_path / "blocked"

    generated = run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    assert len(generated) == 9
    for relative in (
        "tables/behavior_frame_contract.csv",
        "tables/behavior_group_coverage.csv",
        "tables/behavior_target_stats.csv",
        "tables/behavior_contract_issues.csv",
        "tables/behavior_source_fidelity.csv",
        "tables/behavior_identity_task_risk.csv",
        "selected_target_manifest.json",
        "reports/behavior_contract_report.md",
        "run_manifest.json",
    ):
        assert (output / relative).is_file()

    selected = json.loads((output / "selected_target_manifest.json").read_text(encoding="utf-8"))
    assert selected["status"] == "BLOCKED"
    assert selected["missing_behavior_columns"] == [
        "AU12_r",
        "AU14_r",
        "AU15_r",
        "pose_Rx",
        "pose_Ry",
        "pose_Rz",
    ]
    assert selected["gaze"]["target_count"] == 0
    assert selected["gaze"]["value_access_count"] == 0
    assert selected["raw_openface"]["feature_file_open_count"] == 0
    assert selected["openface_value_access_counts"] == {"frame": 3}
    assert selected["rich_provenance"] == {
        "mode": "not_applicable_landmark_or_partial_schema",
        "status": "NOT_APPLICABLE",
    }
    assert not any(code.startswith("rich_") for code in _issue_codes(output))

    frames = _read_csv(output / "tables/behavior_frame_contract.csv")
    assert [float(row["source_timestamp_seconds"]) for row in frames] == pytest.approx(
        [0.0, 1 / 30, 2 / 30]
    )
    assert all(row["exact_frame_join"] == "1" for row in frames)
    assert all(row["status"] == "BLOCKED" for row in frames)
    assert not any((output / name).exists() for name in ("gaze", "gaze_features", "gaze_targets"))

    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["status"] == "BLOCKED"
    assert run_manifest["access_contract"]["raw_openface_feature_file_open_count"] == 0
    assert run_manifest["access_contract"]["gaze_value_access_count"] == 0
    assert "raw_openface_csv" not in run_manifest["inputs"]["source_video_contract"]["accessed_columns"]


def test_rich_schema_calls_core_and_fits_physical_train_only_statistics(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    output = tmp_path / "pass"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    selected = json.loads((output / "selected_target_manifest.json").read_text(encoding="utf-8"))
    assert selected["status"] == "PASS"
    assert selected["training_authorized"] is False
    assert selected["rich_provenance"]["coverage_policy_binding"]["status"] == (
        "NOT_REQUIRED_LEGACY_PASS"
    )
    assert selected["normalization"] == {
        "physical_train_only": True,
        "available": True,
        "validation_or_test_access_count": 0,
    }

    stats = {row["target_name"]: row for row in _read_csv(output / "tables/behavior_target_stats.csv")}
    assert set(stats) == {
        "AU12_r",
        "AU14_r",
        "AU15_r",
        "d_pose_Rx_dt",
        "d_pose_Ry_dt",
        "d_pose_Rz_dt",
    }
    assert all(row["physical_split"] == "train" for row in stats.values())
    assert all(row["status"] == "PASS" for row in stats.values())
    assert stats["AU12_r"]["count"] == "3"
    assert stats["d_pose_Rx_dt"]["count"] == "2"

    frames = _read_csv(output / "tables/behavior_frame_contract.csv")
    assert frames[0]["pose_velocity_valid"] == "0"
    assert frames[1]["pose_velocity_valid"] == "1"
    assert float(frames[1]["d_pose_Rx_dt"]) == pytest.approx(3.0)
    assert float(frames[2]["d_pose_Rx_dt"]) == pytest.approx(6.0)
    risk = _read_csv(output / "tables/behavior_identity_task_risk.csv")
    assert len(risk) == 4
    assert all(row["status"] == "NOT_RUN_P0_PLACEHOLDER" for row in risk)
    assert all(row["probe_value_access_count"] == "0" for row in risk)

    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["status"] == "PASS"
    assert run_manifest["counts"]["blocking_issue_count"] == 0
    assert run_manifest["inputs"]["source_run_manifest"]["sha256"] == _sha(
        source_manifest
    )
    assert selected["provenance"]["source_run_manifest_sha256"] == _sha(
        source_manifest
    )
    assert set(run_manifest["implementation_files"]) == {
        "p0_orchestration",
        "behavior_contract_core",
        "cli",
    }
    for value in run_manifest["implementation_files"].values():
        assert value["sha256"] == _sha(value["path"])
    assert "run_manifest.json" not in run_manifest["outputs"]
    for value in run_manifest["outputs"].values():
        assert value["sha256"] == _sha(value["path"])
    assert run_manifest["git_status_available"] is False
    assert run_manifest["git_status_interpretation"] == (
        "unavailable; do not interpret empty as clean"
    )


def test_legacy_strict_failure_requires_bound_full_coverage_decision(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = (
        _build_legacy_fail_inputs(tmp_path)
    )
    output = tmp_path / "missing_a2"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    selected = json.loads(
        (output / "selected_target_manifest.json").read_text(encoding="utf-8")
    )
    assert selected["status"] == "BLOCKED"
    assert "rich_coverage_policy_decision_required" in _issue_codes(output)
    assert selected["rich_provenance"]["legacy_strict_status_observed"] == "FAIL"
    assert selected["rich_provenance"]["legacy_strict_status_used_as_gate"] is False


def test_bound_full_coverage_decision_replaces_only_legacy_coverage_gate(
    tmp_path, monkeypatch
):
    split, image_root, openface_root, source_contract, source_manifest = (
        _build_legacy_fail_inputs(tmp_path)
    )
    policy, decision, decision_sha256 = _write_full_coverage_binding(
        tmp_path,
        split=split,
        openface_root=openface_root,
        source_contract=source_contract,
        source_manifest=source_manifest,
    )
    monkeypatch.setattr(
        p0_module, "FROZEN_COVERAGE_POLICY_SHA256", _sha(policy)
    )
    output = tmp_path / "bound_a2"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        coverage_policy_decision=decision,
        coverage_policy_decision_sha256=decision_sha256,
        coverage_policy=policy,
        project_root=tmp_path,
    )

    selected = json.loads(
        (output / "selected_target_manifest.json").read_text(encoding="utf-8")
    )
    assert selected["status"] == "PASS"
    binding = selected["rich_provenance"]["coverage_policy_binding"]
    assert binding["status"] == "PASS"
    assert binding["required"] is True
    assert binding["decision"]["observed_sha256"] == decision_sha256
    assert binding["observed_counts"] == {
        "video_count": 1,
        "frame_count": 5,
        "quality_valid_count": 4,
        "au_valid_count": 4,
        "head_valid_pair_count": 3,
    }
    run_manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["inputs"]["coverage_policy_decision"]["sha256"] == (
        decision_sha256
    )
    assert run_manifest["inputs"]["coverage_policy_run_manifest"]["sha256"] == _sha(
        decision.parent / "run_manifest.json"
    )


def test_bound_coverage_count_mismatch_blocks_without_legacy_fallback(
    tmp_path, monkeypatch
):
    split, image_root, openface_root, source_contract, source_manifest = (
        _build_legacy_fail_inputs(tmp_path)
    )
    policy, decision, decision_sha256 = _write_full_coverage_binding(
        tmp_path,
        split=split,
        openface_root=openface_root,
        source_contract=source_contract,
        source_manifest=source_manifest,
        decision_count_updates={"quality_valid_count": 3},
    )
    monkeypatch.setattr(
        p0_module, "FROZEN_COVERAGE_POLICY_SHA256", _sha(policy)
    )
    output = tmp_path / "count_mismatch"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        coverage_policy_decision=decision,
        coverage_policy_decision_sha256=decision_sha256,
        coverage_policy=policy,
        project_root=tmp_path,
    )

    assert json.loads(
        (output / "selected_target_manifest.json").read_text(encoding="utf-8")
    )["status"] == "BLOCKED"
    assert (
        "rich_coverage_policy_decision_count_quality_valid_count_mismatch"
        in _issue_codes(output)
    )


def test_bound_coverage_manifest_detects_post_decision_source_evidence_change(
    tmp_path, monkeypatch
):
    split, image_root, openface_root, source_contract, source_manifest = (
        _build_legacy_fail_inputs(tmp_path)
    )
    policy, decision, decision_sha256 = _write_full_coverage_binding(
        tmp_path,
        split=split,
        openface_root=openface_root,
        source_contract=source_contract,
        source_manifest=source_manifest,
    )
    monkeypatch.setattr(
        p0_module, "FROZEN_COVERAGE_POLICY_SHA256", _sha(policy)
    )
    with (source_manifest.parent / "video_run_summary.csv").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("post-decision-change\n")
    output = tmp_path / "source_evidence_changed"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        coverage_policy_decision=decision,
        coverage_policy_decision_sha256=decision_sha256,
        coverage_policy=policy,
        project_root=tmp_path,
    )

    assert json.loads(
        (output / "selected_target_manifest.json").read_text(encoding="utf-8")
    )["status"] == "BLOCKED"
    assert (
        "rich_coverage_policy_run_manifest_input_video_run_summary_sha256_mismatch"
        in _issue_codes(output)
    )


@pytest.mark.parametrize(
    ("mutation", "expected_issue"),
    [
        (
            lambda payload: payload.__setitem__("feature_extraction_sha256", "bad"),
            "rich_source_feature_extraction_sha256_mismatch",
        ),
        (
            lambda payload: payload.__setitem__("run_scope", "debug_subset"),
            "rich_source_run_scope_mismatch",
        ),
        (
            lambda payload: payload.pop("source_video_contract_sha256"),
            "rich_source_video_contract_sha256_mismatch",
        ),
        (
            lambda payload: payload.__setitem__("script_sha256", "bad"),
            "rich_source_extractor_script_sha256_mismatch",
        ),
        (
            lambda payload: payload.__setitem__("git_status_available", False),
            "rich_source_git_provenance_mismatch",
        ),
        (
            lambda payload: payload.__setitem__("git_commit", "not-a-sha"),
            "rich_source_git_provenance_mismatch",
        ),
        (
            lambda payload: payload.__setitem__("git_branch", "bad..branch"),
            "rich_source_git_provenance_mismatch",
        ),
        (
            lambda payload: payload.__setitem__("image_root", "/different/image/root"),
            "rich_source_image_root_binding_mismatch",
        ),
    ],
)
def test_rich_source_manifest_provenance_failures_block_pass(
    tmp_path, mutation, expected_issue
):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    _rewrite_json(source_manifest, mutation)
    output = tmp_path / "blocked_provenance"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    selected = json.loads((output / "selected_target_manifest.json").read_text(encoding="utf-8"))
    assert selected["status"] == "BLOCKED"
    assert expected_issue in _issue_codes(output)


@pytest.mark.parametrize("summary_mode", ["failed", "missing"])
def test_rich_extraction_summary_must_exist_and_pass(tmp_path, summary_mode):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    summary_path = source_manifest.parent / "extraction_summary.json"
    if summary_mode == "failed":
        _rewrite_json(summary_path, lambda payload: payload.__setitem__("status", "FAIL"))
        _rewrite_json(
            source_manifest,
            lambda payload: payload.__setitem__(
                "extraction_summary_sha256", _sha(summary_path)
            ),
        )
    else:
        summary_path.unlink()
    output = tmp_path / f"summary_{summary_mode}"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    assert json.loads(
        (output / "selected_target_manifest.json").read_text(encoding="utf-8")
    )["status"] == "BLOCKED"
    expected = (
        "rich_extraction_summary_status_mismatch"
        if summary_mode == "failed"
        else "rich_extraction_summary_missing"
    )
    assert expected in _issue_codes(output)


def test_rich_csv_content_manifest_detects_post_extraction_tamper(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    csv_path = openface_root / f"{VIDEO_ID}.csv"
    rows = _read_csv(csv_path)
    rows[1]["AU12_r"] = "123.456"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "tampered"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    assert json.loads(
        (output / "selected_target_manifest.json").read_text(encoding="utf-8")
    )["status"] == "BLOCKED"
    assert "rich_csv_content_mismatch" in _issue_codes(output)


def test_rich_csv_content_manifest_is_mandatory(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    (source_manifest.parent / "csv_content_manifest.csv").unlink()
    output = tmp_path / "missing_content_manifest"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    assert json.loads(
        (output / "selected_target_manifest.json").read_text(encoding="utf-8")
    )["status"] == "BLOCKED"
    assert "rich_csv_content_manifest_missing" in _issue_codes(output)


def test_physical_train_statistics_ignore_extreme_val_and_test_targets(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = (
        _build_multisplit_rich_inputs(tmp_path)
    )
    output = tmp_path / "multisplit"

    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    selected = json.loads((output / "selected_target_manifest.json").read_text(encoding="utf-8"))
    assert selected["status"] == "PASS"
    stats = {
        row["target_name"]: row
        for row in _read_csv(output / "tables" / "behavior_target_stats.csv")
    }
    assert stats["AU12_r"]["count"] == "3"
    assert float(stats["AU12_r"]["mean"]) == pytest.approx(7 / 3)
    assert float(stats["AU12_r"]["maximum"]) == pytest.approx(4.0)
    assert stats["d_pose_Rx_dt"]["count"] == "2"
    assert float(stats["d_pose_Rx_dt"]["mean"]) == pytest.approx(4.5)
    assert float(stats["d_pose_Rx_dt"]["maximum"]) == pytest.approx(6.0)
    run_manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["access_contract"][
        "validation_test_normalization_access_count"
    ] == 0


def test_exact_join_failure_is_reported_without_partial_success(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=True
    )
    csv_path = openface_root / f"{VIDEO_ID}.csv"
    rows = _read_csv(csv_path)
    rows[-1]["frame"] = "4"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "join_blocked"
    run_privileged_behavior_p0(
        dataset_split_file=split,
        image_root=image_root,
        openface_root=openface_root,
        source_run_manifest=source_manifest,
        source_video_contract=source_contract,
        output_dir=output,
        project_root=tmp_path,
    )

    selected = json.loads((output / "selected_target_manifest.json").read_text(encoding="utf-8"))
    assert selected["status"] == "BLOCKED"
    issues = _read_csv(output / "tables/behavior_contract_issues.csv")
    assert "exact_frame_join_failed" in {row["issue_code"] for row in issues}
    coverage = _read_csv(output / "tables/behavior_group_coverage.csv")[0]
    assert coverage["joined_frame_count"] == "2"
    assert coverage["status"] == "BLOCKED"


def test_cli_defaults_and_exposes_no_raw_openface_or_gaze_inputs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset-split-file",
            "/split.json",
            "--image-root",
            "/images",
            "--openface-root",
            "/aligned-rich",
            "--source-video-contract",
            "/source.csv",
            "--output-dir",
            "/audit",
        ]
    )

    assert args.confidence_threshold == pytest.approx(0.8)
    assert args.max_pose_velocity_dt_seconds == pytest.approx(0.1)
    assert args.source_run_manifest is None
    assert args.coverage_policy_decision is None
    assert args.coverage_policy_decision_sha256 is None
    assert args.coverage_policy is None
    assert all("raw_openface" not in key and "gaze" not in key for key in vars(args))


def test_nonempty_output_directory_is_refused_before_writing(tmp_path):
    split, image_root, openface_root, source_contract, source_manifest = _build_inputs(
        tmp_path, rich=False
    )
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("user-owned", encoding="utf-8")

    with pytest.raises(FileExistsError, match="empty or absent"):
        run_privileged_behavior_p0(
            dataset_split_file=split,
            image_root=image_root,
            openface_root=openface_root,
            source_run_manifest=source_manifest,
            source_video_contract=source_contract,
            output_dir=output,
            project_root=tmp_path,
        )
    assert (output / "keep.txt").read_text(encoding="utf-8") == "user-owned"
