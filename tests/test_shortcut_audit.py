import csv

import pytest

from src.diagnostics.io import write_prediction_table
from src.diagnostics.openface_quality import summarize_openface_csv
from src.diagnostics.shortcut_audit import run_shortcut_audit


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
