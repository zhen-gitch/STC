import csv

from src.diagnostics.alignment_geometry import (
    merge_predictions_with_geometry,
    run_alignment_geometry_audit,
    summarize_openface_geometry_csv,
)
from src.diagnostics.io import write_prediction_table


def _write_openface_csv(path, x_shift=0.0, y_shift=0.0, scale=1.0):
    fieldnames = ["frame", "timestamp", "confidence", "success", "x_0", "x_1", "x_2", "y_0", "y_1", "y_2"]
    base_points = [(40.0, 42.0), (60.0, 42.0), (50.0, 72.0)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(3):
            row = {
                "frame": idx + 1,
                "timestamp": f"{idx * 0.04:.3f}",
                "confidence": 0.95 - 0.01 * idx,
                "success": 1,
            }
            for point_idx, (x, y) in enumerate(base_points):
                row[f"x_{point_idx}"] = x_shift + scale * x + idx
                row[f"y_{point_idx}"] = y_shift + scale * y + idx
            writer.writerow(row)
    return path


def test_summarize_openface_geometry_csv_computes_bbox_and_offsets(tmp_path):
    csv_path = _write_openface_csv(tmp_path / "203_1_Freeform_video.csv")

    summary = summarize_openface_geometry_csv(csv_path, frame_width=112, frame_height=112)

    assert summary["video_id"] == "203_1_Freeform_video"
    assert summary["subject_id"] == "203_1"
    assert summary["task_name"] == "Freeform"
    assert summary["frame_count"] == 3
    assert summary["valid_landmark_frame_count"] == 3
    assert summary["landmark_count"] == 3
    assert summary["landmark_bbox_width_mean"] == 20
    assert summary["landmark_bbox_height_mean"] == 30
    assert summary["landmark_bbox_area_mean"] == 600
    assert summary["eye_distance_mean"] == 20
    assert summary["landmark_jitter_mean"] > 0


def test_merge_predictions_with_geometry_matches_aligned_video_ids(tmp_path):
    geometry = [summarize_openface_geometry_csv(_write_openface_csv(tmp_path / "203_1_Freeform_video.csv"))]
    prediction_rows = [
        {
            "video_id": "203_1_Freeform_video_aligned",
            "subject_id": "203_1",
            "task_name": "Freeform",
            "true_bdi": 8.0,
            "pred_bdi": 10.0,
            "residual": 2.0,
            "abs_error": 2.0,
            "severity_group": "minimal",
        }
    ]

    merged, missing = merge_predictions_with_geometry(prediction_rows, geometry)

    assert missing == []
    assert len(merged) == 1
    assert merged[0]["matched_on"] == "video_id"
    assert merged[0]["landmark_bbox_area_mean"] == 600


def test_run_alignment_geometry_audit_writes_expected_outputs(tmp_path):
    openface_root = tmp_path / "openface"
    openface_root.mkdir()
    _write_openface_csv(openface_root / "203_1_Freeform_video.csv", scale=1.0)
    _write_openface_csv(openface_root / "204_1_Freeform_video.csv", scale=1.5)
    _write_openface_csv(openface_root / "205_1_Freeform_video.csv", x_shift=8.0, scale=0.8)

    predictions_csv = tmp_path / "predictions.csv"
    write_prediction_table(
        predictions_csv,
        subject_ids=["203_1", "204_1", "205_1"],
        video_ids=[
            "203_1_Freeform_video_aligned",
            "204_1_Freeform_video_aligned",
            "205_1_Freeform_video_aligned",
        ],
        targets=[8, 18, 32],
        preds=[10, 16, 22],
    )

    generated = run_alignment_geometry_audit(
        predictions_csv=predictions_csv,
        openface_root=openface_root,
        output_dir=tmp_path / "alignment_geometry",
        frame_width=112,
        frame_height=112,
    )

    assert len(generated) == 5
    tables = tmp_path / "alignment_geometry" / "tables"
    reports = tmp_path / "alignment_geometry" / "reports"
    assert (tables / "alignment_geometry_summary.csv").exists()
    assert (tables / "alignment_geometry_merged.csv").exists()
    assert (tables / "alignment_geometry_correlation.csv").exists()
    assert (tables / "alignment_geometry_group_summary.csv").exists()
    assert (reports / "alignment_geometry_audit_report.md").exists()
