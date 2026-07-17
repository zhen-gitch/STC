import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.diagnostics.frame_recovery import (
    EXPOSURE_REVIEW_FIELDS,
    FrameRecoveryPolicy,
    apply_luma_gamma,
    apply_luma_inverse_log_curve,
    apply_luma_log_curve,
    classify_failed_frame,
    exposure_curve_value,
    fit_exposure_curve_parameter,
    inspect_frame,
    materialize_frame_repairs,
    run_frame_failure_audit,
    write_exposure_previews,
)
from src.diagnostics.source_presence import SOURCE_REVIEW_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "audit_frame_recovery",
    PROJECT_ROOT / "scripts" / "audit_frame_recovery.py",
)
audit_frame_recovery = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(audit_frame_recovery)


def _write_rgb(path, value, *, shifted_square=None):
    array = np.full((32, 32, 3), int(value), dtype=np.uint8)
    if shifted_square is not None:
        x = int(shifted_square)
        array[10:20, x : x + 8] = np.array([180, 120, 80], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="JPEG", quality=95, subsampling=0)


def _write_video(root, video_id, values):
    video_dir = root / video_id
    for frame, value in enumerate(values, start=1):
        _write_rgb(video_dir / f"frame_det_00_{frame:06d}.jpg", value)
    return video_dir


def _write_openface(path, success_values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "timestamp", "confidence", "success"],
        )
        writer.writeheader()
        for index, success in enumerate(success_values, start=1):
            writer.writerow(
                {
                    "frame": index,
                    "timestamp": (index - 1) / 30.0,
                    "confidence": 0.98 if success else 0.0,
                    "success": success,
                }
            )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_completed_exposure_review(audit_dir, path, video_id, decision="stable_log"):
    with (audit_dir / "tables" / "video_failure_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary = next(row for row in csv.DictReader(handle) if row["video_id"] == video_id)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPOSURE_REVIEW_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "split": summary["split"],
                "video_id": video_id,
                "segment_start_frame": 1,
                "segment_end_frame": summary["frame_count"],
                "exposure_status": summary["video_exposure_status"],
                "review_status": "REVIEWED",
                "review_decision": decision,
                "exposure_curve": summary["planned_exposure_curve"],
                "observed_luma_median": summary["video_luma_median"],
                "observed_luma_source": "full_segment_visible_luma_median",
                "target_luma": summary["exposure_target_luma"],
                "curve_parameter": summary["planned_exposure_parameter"],
                "reviewer": "tester",
                "review_notes": "synthetic stable exposure segment",
            }
        )


def test_cli_is_two_stage_and_does_not_expose_a_facelandmark_action():
    parser = audit_frame_recovery.build_parser()
    args = parser.parse_args(
        [
            "audit",
            "--image-root",
            "/images",
            "--openface-root",
            "/existing_csv",
            "--dataset-split-file",
            "/split.json",
            "--output-dir",
            "/audit",
        ]
    )

    assert args.command == "audit"
    assert args.max_optical_flow_gap == 3
    assert args.max_copy_gap == 1
    assert args.exposure_safe_low_quantile == 0.20
    assert args.exposure_target_low_quantile == 0.25
    assert args.exposure_target_high_quantile == 0.75
    assert args.exposure_safe_high_quantile == 0.80
    assert all("facelandmark" not in key.lower() for key in vars(args))

    materialize_args = parser.parse_args(
        [
            "materialize",
            "--audit-dir",
            "/audit",
            "--image-root",
            "/images",
            "--output-root",
            "/derived",
            "--source-review-manifest",
            "/source_review.csv",
            "--exposure-review-manifest",
            "/exposure_review.csv",
        ]
    )
    assert materialize_args.exposure_review_manifest == "/exposure_review.csv"


def test_pure_black_classification_and_padding_preserving_log_curves(tmp_path):
    policy = FrameRecoveryPolicy()
    black_path = tmp_path / "black.jpg"
    _write_rgb(black_path, 0)
    metrics = inspect_frame(black_path, policy)

    assert classify_failed_frame(metrics, policy) == "pure_black"

    image = np.full((16, 16, 3), 40, dtype=np.uint8)
    image[:2] = 0
    image[-2:] = 0
    image[:, :2] = 0
    image[:, -2:] = 0
    adjusted = apply_luma_log_curve(image, 2.0, policy)

    assert np.array_equal(adjusted[:2], image[:2])
    assert np.array_equal(adjusted[:, :2], image[:, :2])
    assert float(adjusted[4:-4, 4:-4].mean()) > float(image[4:-4, 4:-4].mean())

    bright = np.full((16, 16, 3), 230, dtype=np.uint8)
    bright[:2] = 0
    compressed = apply_luma_inverse_log_curve(bright, 3.0, policy)
    assert np.array_equal(compressed[:2], bright[:2])
    assert float(compressed[4:-4, 4:-4].mean()) < float(bright[4:-4, 4:-4].mean())

    # The historical gamma implementation remains available for old reports.
    assert apply_luma_gamma(image, 0.7, policy).dtype == np.uint8


def test_log_family_parameter_fit_maps_observed_median_to_frozen_target():
    for observed, target, curve in [(30.0, 53.740, "log"), (224.0, 135.769, "inverse_log")]:
        parameter = fit_exposure_curve_parameter(observed, target, curve)
        mapped = float(exposure_curve_value(observed / 255.0, curve, parameter)) * 255.0
        assert 0.0 < parameter <= 32.0
        assert abs(mapped - target) < 1e-6


def test_audit_builds_frame_blocks_and_train_only_exposure_plan(tmp_path):
    image_root = tmp_path / "images"
    openface_root = tmp_path / "openface"
    under_video = "203_1_Freeform_video_aligned"
    normal_video = "204_1_Freeform_video_aligned"
    _write_video(image_root, under_video, [20, 0, 20, 0, 0])
    _write_video(image_root, normal_video, [90, 90, 90, 90, 90])
    _write_openface(openface_root / f"{under_video}.csv", [1, 0, 1, 0, 0])
    _write_openface(openface_root / f"{normal_video}.csv", [1, 1, 1, 1, 1])
    split_file = tmp_path / "dataset_split.json"
    split_file.write_text(
        json.dumps({"train": [under_video, normal_video], "val": [], "test": []}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"

    generated = run_frame_failure_audit(
        image_root=image_root,
        openface_root=openface_root,
        dataset_split_file=split_file,
        output_dir=output_dir,
        policy=FrameRecoveryPolicy(exposure_sample_frames=5),
        workers=1,
        project_root=Path(__file__).resolve().parents[1],
    )

    assert len(generated) == 7
    with (output_dir / "tables" / "frame_failure_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        failures = list(csv.DictReader(handle))
    with (output_dir / "tables" / "failure_blocks.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        blocks = list(csv.DictReader(handle))
    with (output_dir / "tables" / "video_failure_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summaries = {row["video_id"]: row for row in csv.DictReader(handle)}

    assert len(failures) == 3
    assert {row["failure_type"] for row in failures} == {"pure_black"}
    assert len(blocks) == 2
    assert blocks[0]["proposed_action"] == "source_video_review"
    assert blocks[0]["repair_eligible"] == "0"
    assert blocks[1]["proposed_action"] == "keep_invalid"
    assert blocks[1]["repair_eligible"] == "0"
    assert summaries[under_video]["video_exposure_status"] == "underexposed"
    assert summaries[normal_video]["video_exposure_status"] == "normal"
    assert summaries[under_video]["exposure_reference_source"] == "train_normal_video_luma_quantiles"
    assert summaries[under_video]["planned_exposure_curve"] == "log"
    assert summaries[under_video]["planned_gamma"] == ""
    assert float(summaries[under_video]["exposure_safe_low_luma"]) <= float(
        summaries[under_video]["exposure_target_luma"]
    )
    assert float(summaries[under_video]["exposure_target_luma"]) <= float(
        summaries[under_video]["exposure_safe_high_luma"]
    )
    with (output_dir / "tables" / "exposure_review_template.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        exposure_review = list(csv.DictReader(handle))
    assert len(exposure_review) == 1
    assert exposure_review[0]["review_status"] == "PENDING"
    assert exposure_review[0]["review_decision"] == "segment_review_required"
    assert exposure_review[0]["observed_luma_source"] == "sampled_audit_median_replace_before_review"
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["face_landmark_rerun_performed"] is False


def test_preview_accepts_relocated_root_only_with_exact_integrity_evidence(tmp_path):
    audited_root = tmp_path / "audited_images"
    relocated_root = tmp_path / "relocated_images"
    openface_root = tmp_path / "openface"
    under_video = "203_1_Freeform_video_aligned"
    normal_video = "204_1_Freeform_video_aligned"
    for root in (audited_root, relocated_root):
        _write_video(root, under_video, [20, 20])
        _write_video(root, normal_video, [90, 90])
    _write_openface(openface_root / f"{under_video}.csv", [1, 1])
    _write_openface(openface_root / f"{normal_video}.csv", [1, 1])
    split_file = tmp_path / "dataset_split.json"
    split_file.write_text(
        json.dumps({"train": [under_video, normal_video], "val": [], "test": []}),
        encoding="utf-8",
    )
    audit_dir = tmp_path / "audit"
    run_frame_failure_audit(
        image_root=audited_root,
        openface_root=openface_root,
        dataset_split_file=split_file,
        output_dir=audit_dir,
        policy=FrameRecoveryPolicy(exposure_sample_frames=2),
        workers=1,
    )

    with pytest.raises(ValueError, match="image-integrity-comparison-summary"):
        write_exposure_previews(
            audit_dir=audit_dir,
            image_root=relocated_root,
            output_dir=tmp_path / "blocked_relocated_preview",
            video_ids=[under_video],
        )

    inventory_dir = tmp_path / "integrity" / "local_wsl"
    manifest_path = inventory_dir / "tables" / "image_manifest.csv"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("synthetic exact manifest\n", encoding="utf-8")
    inventory_path = inventory_dir / "inventory_summary.json"
    inventory_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "image_root": str(relocated_root.resolve()),
                "image_count": 4,
                "decode_ok_count": 4,
            }
        ),
        encoding="utf-8",
    )
    comparison_path = tmp_path / "integrity" / "comparison_summary.json"
    comparison_path.write_text(
        json.dumps(
            {
                "status": "EXACT_PASS",
                "reference_count": 4,
                "candidate_count": 4,
                "status_counts": {"EXACT_MATCH": 4},
                "candidate_manifest": str(manifest_path.resolve()),
                "candidate_manifest_sha256": _sha256(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "relocated_preview"
    write_exposure_previews(
        audit_dir=audit_dir,
        image_root=relocated_root,
        output_dir=output_dir,
        video_ids=[under_video],
        image_integrity_comparison_summary=comparison_path,
    )
    preview_manifest = json.loads(
        (output_dir / "preview_manifest.json").read_text(encoding="utf-8")
    )
    evidence = preview_manifest["image_root_relocation_evidence"]
    assert evidence["mode"] == "relocated_root_exact_pass"
    assert evidence["candidate_count"] == 4


def test_materialization_requires_source_review_and_keeps_black_frames_invalid(tmp_path):
    image_root = tmp_path / "images"
    openface_root = tmp_path / "openface"
    under_video = "203_1_Freeform_video_aligned"
    normal_video = "204_1_Freeform_video_aligned"
    _write_video(image_root, under_video, [20, 0, 20, 0, 0])
    _write_video(image_root, normal_video, [90, 90, 90, 90, 90])
    _write_openface(openface_root / f"{under_video}.csv", [1, 0, 1, 0, 0])
    _write_openface(openface_root / f"{normal_video}.csv", [1, 1, 1, 1, 1])
    split_file = tmp_path / "dataset_split.json"
    split_file.write_text(
        json.dumps({"train": [under_video, normal_video], "val": [], "test": []}),
        encoding="utf-8",
    )
    audit_dir = tmp_path / "audit"
    run_frame_failure_audit(
        image_root=image_root,
        openface_root=openface_root,
        dataset_split_file=split_file,
        output_dir=audit_dir,
        policy=FrameRecoveryPolicy(exposure_sample_frames=5),
        workers=1,
        project_root=Path(__file__).resolve().parents[1],
    )
    source_hashes = {
        path.name: _sha256(path)
        for path in (image_root / under_video).glob("*.jpg")
    }
    preview_dir = tmp_path / "exposure_preview"
    preview_outputs = write_exposure_previews(
        audit_dir=audit_dir,
        image_root=image_root,
        output_dir=preview_dir,
        video_ids=[under_video],
        frames_per_video=4,
        columns=2,
        thumb_width=96,
        project_root=Path(__file__).resolve().parents[1],
    )
    assert len(preview_outputs) == 3
    with (preview_dir / "exposure_preview_frames.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        preview_rows = list(csv.DictReader(handle))
    assert len(preview_rows) == 2
    assert all(
        float(row["preview_luma_median"]) > float(row["source_luma_median"])
        for row in preview_rows
    )
    assert (preview_dir / "contact_sheets" / f"{under_video}.jpg").exists()
    preview_manifest = json.loads(
        (preview_dir / "preview_manifest.json").read_text(encoding="utf-8")
    )
    assert preview_manifest["source_tree_modified"] is False
    assert preview_manifest["exposure_review_approval_implied"] is False
    assert {
        path.name: _sha256(path)
        for path in (image_root / under_video).glob("*.jpg")
    } == source_hashes
    derived_root = tmp_path / "derived"
    source_review = tmp_path / "source_review.csv"
    with source_review.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "split": "train",
                    "video_id": under_video,
                    "pure_black_run_id": f"{under_video}:PB0001",
                    "segment_start_frame": 2,
                    "segment_end_frame": 2,
                    "review_status": "REVIEWED",
                    "source_presence_status": "person_present_detection_failure",
                    "recovery_permission": "raw_frame_warp_only",
                    "reviewer": "tester",
                    "review_notes": "synthetic source frame contains a person",
                },
                {
                    "split": "train",
                    "video_id": under_video,
                    "pure_black_run_id": f"{under_video}:PB0002",
                    "segment_start_frame": 4,
                    "segment_end_frame": 5,
                    "review_status": "REVIEWED",
                    "source_presence_status": "person_absent",
                    "recovery_permission": "keep_invalid",
                    "reviewer": "tester",
                    "review_notes": "synthetic empty source frames",
                },
            ]
        )
    exposure_review = tmp_path / "exposure_review.csv"
    _write_completed_exposure_review(audit_dir, exposure_review, under_video)

    with pytest.raises(ValueError, match="exposure_review_manifest is required"):
        materialize_frame_repairs(
            audit_dir=audit_dir,
            image_root=image_root,
            output_root=tmp_path / "blocked_missing_review",
            source_review_manifest=source_review,
            max_videos=1,
        )
    with pytest.raises(ValueError, match="not REVIEWED"):
        materialize_frame_repairs(
            audit_dir=audit_dir,
            image_root=image_root,
            output_root=tmp_path / "blocked_pending_review",
            source_review_manifest=source_review,
            exposure_review_manifest=audit_dir / "tables" / "exposure_review_template.csv",
            max_videos=1,
        )
    assert not (tmp_path / "blocked_missing_review").exists()
    assert not (tmp_path / "blocked_pending_review").exists()

    generated = materialize_frame_repairs(
        audit_dir=audit_dir,
        image_root=image_root,
        output_root=derived_root,
        source_review_manifest=source_review,
        exposure_review_manifest=exposure_review,
        layout="sparse_overlay",
        max_videos=1,
        project_root=Path(__file__).resolve().parents[1],
    )

    assert len(generated) == 3
    derived_video = derived_root / under_video
    assert (derived_video / "frame_det_00_000001.jpg").exists()
    assert (derived_video / "frame_det_00_000003.jpg").exists()
    assert not (derived_video / "frame_det_00_000002.jpg").exists()
    assert not (derived_video / "frame_det_00_000004.jpg").exists()
    assert not (derived_video / "frame_det_00_000005.jpg").exists()
    assert {
        path.name: _sha256(path)
        for path in (image_root / under_video).glob("*.jpg")
    } == source_hashes
    with (derived_root / "_audit" / "repair_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        repairs = list(csv.DictReader(handle))
    assert len(repairs) == 2
    assert {row["repair_type"] for row in repairs} == {"underexposed_log_tone_normalization"}
    assert {row["exposure_curve"] for row in repairs} == {"log"}
    assert all(row["exposure_gamma"] == "" for row in repairs)
    with (derived_root / "_audit" / "repair_failures.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        skipped = list(csv.DictReader(handle))
    assert len(skipped) == 3
    assert {row["materialization_reason"] for row in skipped} == {
        "raw_frame_warp_required_but_not_implemented",
        "source_person_absent_keep_invalid",
    }
    materialization = json.loads(
        (derived_root / "_audit" / "materialization_manifest.json").read_text(encoding="utf-8")
    )
    assert materialization["source_tree_modified"] is False
    assert materialization["face_landmark_rerun_performed"] is False
    assert materialization["aligned_black_frame_synthesis_enabled"] is False
    assert materialization["exposure_curve_family"]["per_frame_parameter_fitting"] is False
