import csv

import pytest

from src.diagnostics.io import write_prediction_table
from src.diagnostics.openface_quality import summarize_openface_csv
from src.diagnostics.shortcut_audit import (
    _make_group_folds,
    evaluate_shortcut_predictors_grouped_cv,
    merge_predictions_with_quality,
    run_shortcut_audit,
)


def _write_openface_csv(path, confidence_values, pose_values):
    fieldnames = [
        "frame",
        "face_id",
        "timestamp",
        "confidence",
        "success",
        "gaze_angle_x",
        "gaze_angle_y",
        "pose_Rx",
        "pose_Ry",
        "pose_Rz",
        "x_0",
        "x_1",
        "y_0",
        "y_1",
        "AU01_r",
        "AU01_c",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (confidence, pose) in enumerate(zip(confidence_values, pose_values), start=1):
            writer.writerow(
                {
                    "frame": idx,
                    "face_id": 0,
                    "timestamp": f"{(idx - 1) * 0.04:.3f}",
                    "confidence": confidence,
                    "success": 1 if confidence >= 0.5 else 0,
                    "gaze_angle_x": 0.01 * idx,
                    "gaze_angle_y": -0.02 * idx,
                    "pose_Rx": pose,
                    "pose_Ry": pose * 0.5,
                    "pose_Rz": -pose,
                    "x_0": 10 + idx,
                    "x_1": 12 + idx,
                    "y_0": 20 + idx,
                    "y_1": 23 + idx,
                    "AU01_r": 0.1 * idx,
                    "AU01_c": 1 if idx % 2 == 0 else 0,
                }
            )


def test_summarize_openface_csv(tmp_path):
    csv_path = tmp_path / "203_1_Freeform_video.csv"
    _write_openface_csv(csv_path, confidence_values=[0.98, 0.70, 0.40], pose_values=[0.1, 0.2, 0.3])

    summary = summarize_openface_csv(csv_path, low_confidence_threshold=0.8)

    assert summary["video_id"] == "203_1_Freeform_video"
    assert summary["subject_id"] == "203_1"
    assert summary["task_name"] == "Freeform"
    assert summary["frame_count"] == 3
    assert summary["success_count"] == 2
    assert summary["low_confidence_ratio"] == pytest.approx(2 / 3)
    assert summary["pose_rx_abs_mean"] == pytest.approx(0.2)
    assert summary["landmark_motion_mean"] > 0
    assert "AU01_r_mean" in summary
    assert "AU01_c_mean" in summary


def test_run_shortcut_audit(tmp_path):
    pytest.importorskip("matplotlib")

    openface_root = tmp_path / "openface"
    openface_root.mkdir()
    _write_openface_csv(openface_root / "203_1_Freeform_video.csv", [0.98, 0.95, 0.94], [0.10, 0.11, 0.12])
    _write_openface_csv(openface_root / "204_1_Freeform_video.csv", [0.70, 0.65, 0.60], [0.40, 0.42, 0.44])
    _write_openface_csv(openface_root / "205_1_Freeform_video.csv", [0.45, 0.40, 0.35], [0.80, 0.82, 0.84])

    predictions_csv = tmp_path / "predictions.csv"
    write_prediction_table(
        predictions_csv,
        subject_ids=["203_1", "204_1", "205_1"],
        video_ids=["203_1_Freeform_video", "204_1_Freeform_video", "205_1_Freeform_video"],
        targets=[8, 20, 35],
        preds=[9, 18, 32],
    )

    generated = run_shortcut_audit(
        predictions_csv=predictions_csv,
        openface_root=openface_root,
        output_dir=tmp_path / "shortcut_audit",
    )

    assert generated
    assert (tmp_path / "shortcut_audit" / "tables" / "openface_quality_summary.csv").exists()
    assert (tmp_path / "shortcut_audit" / "tables" / "shortcut_correlation.csv").exists()
    assert (tmp_path / "shortcut_audit" / "tables" / "shortcut_predictor_results.csv").exists()
    assert (tmp_path / "shortcut_audit" / "tables" / "shortcut_predictor_grouped_cv.csv").exists()
    assert (tmp_path / "shortcut_audit" / "tables" / "case_study_manifest.csv").exists()
    assert (tmp_path / "shortcut_audit" / "reports" / "case_study_manifest.md").exists()
    assert (tmp_path / "shortcut_audit" / "reports" / "shortcut_audit_report.md").exists()


def test_shortcut_audit_skips_ambiguous_subject_only_predictions(tmp_path):
    openface_root = tmp_path / "openface"
    openface_root.mkdir()
    _write_openface_csv(openface_root / "203_1_Freeform_video.csv", [0.98, 0.95], [0.10, 0.11])
    _write_openface_csv(openface_root / "203_1_Northwind_video.csv", [0.70, 0.65], [0.40, 0.42])

    predictions_csv = tmp_path / "predictions.csv"
    write_prediction_table(
        predictions_csv,
        subject_ids=["203_1"],
        targets=[8],
        preds=[9],
    )

    generated = run_shortcut_audit(
        predictions_csv=predictions_csv,
        openface_root=openface_root,
        output_dir=tmp_path / "shortcut_audit",
    )

    assert generated
    merged_path = tmp_path / "shortcut_audit" / "tables" / "shortcut_merged.csv"
    with merged_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def test_shortcut_audit_matches_aligned_prediction_video_ids():
    prediction_records = [
        {
            "video_id": "203_2_Freeform_video_aligned",
            "subject_id": "203_2",
            "true_bdi": 8.0,
            "pred_bdi": 4.5,
            "residual": -3.5,
            "abs_error": 3.5,
            "severity_group": "minimal",
        }
    ]
    quality_rows = [
        {
            "video_id": "203_2_Freeform_video",
            "subject_id": "203_2",
            "task_name": "Freeform",
            "confidence_mean": "0.97",
        },
        {
            "video_id": "203_2_Northwind_video",
            "subject_id": "203_2",
            "task_name": "Northwind",
            "confidence_mean": "0.96",
        },
    ]

    merged = merge_predictions_with_quality(prediction_records, quality_rows)

    assert len(merged) == 1
    assert merged[0]["video_id"] == "203_2_Freeform_video"
    assert merged[0]["task_name"] == "Freeform"
    assert merged[0]["confidence_mean"] == "0.97"


def test_make_group_folds_keeps_subject_videos_together():
    groups = [
        "203_1",
        "203_1",
        "204_1",
        "204_1",
        "205_1",
        "205_1",
        "206_1",
        "206_1",
    ]

    folds = _make_group_folds(groups, num_folds=3, seed=42)

    assert len(folds) == 3
    for subject_id in set(groups):
        assert sum(subject_id in fold for fold in folds) == 1


def test_grouped_cv_predictor_outputs_expected_rows():
    rows = []
    for idx, subject_id in enumerate(["203_1", "204_1", "205_1", "206_1"], start=1):
        for task_offset, task_name in enumerate(["Freeform", "Northwind"]):
            true_bdi = float(idx * 5)
            rows.append(
                {
                    "video_id": f"{subject_id}_{task_name}_video",
                    "subject_id": subject_id,
                    "matched_on": "video_id",
                    "true_bdi": true_bdi,
                    "pred_bdi": true_bdi + 0.5 * task_offset,
                    "residual": 0.5 * task_offset,
                    "abs_error": 0.5 * task_offset,
                    "severity_group": "minimal",
                    "task_name": task_name,
                    "source_file": f"{subject_id}_{task_name}_video.csv",
                    "confidence_mean": 0.8 + 0.01 * idx,
                    "pose_rx_abs_mean": 0.1 * idx,
                    "AU01_r_mean": 0.2 * idx,
                }
            )

    predictor_rows = evaluate_shortcut_predictors_grouped_cv(
        rows,
        alpha_values=(10.0, 100.0),
        num_folds=2,
        seed=42,
    )

    by_model = {row["model"]: row for row in predictor_rows}
    assert {"rgb_mtl_lite", "mean", "ridge_alpha_10", "ridge_alpha_100"}.issubset(by_model)
    assert by_model["mean"]["evaluation"] == "grouped_cv_subject"
    assert by_model["ridge_alpha_10"]["num_groups"] == 4
    assert by_model["ridge_alpha_10"]["num_folds"] == 2
    assert by_model["ridge_alpha_10"]["num_features"] == 3
