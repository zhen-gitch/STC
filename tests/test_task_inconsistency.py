import csv
import tempfile
from pathlib import Path

import pytest

from src.diagnostics.task_inconsistency import (
    compute_task_artifact_correlations,
    load_prediction_records,
    pair_predictions_by_subject,
    run_task_inconsistency_audit,
)


def _write_predictions_csv(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "video_id": "203_1_Freeform_video",
            "subject_id": "203_1",
            "task_name": "Freeform",
            "true_bdi": "10.0",
            "pred_bdi": "12.0",
            "residual": "2.0",
            "abs_error": "2.0",
            "severity_group": "minimal",
        },
        {
            "video_id": "203_1_Northwind_video",
            "subject_id": "203_1",
            "task_name": "Northwind",
            "true_bdi": "11.0",
            "pred_bdi": "18.0",
            "residual": "7.0",
            "abs_error": "7.0",
            "severity_group": "minimal",
        },
        {
            "video_id": "204_1_Freeform_video",
            "subject_id": "204_1",
            "task_name": "Freeform",
            "true_bdi": "25.0",
            "pred_bdi": "20.0",
            "residual": "-5.0",
            "abs_error": "5.0",
            "severity_group": "moderate",
        },
        {
            "video_id": "204_1_Northwind_video",
            "subject_id": "204_1",
            "task_name": "Northwind",
            "true_bdi": "26.0",
            "pred_bdi": "22.0",
            "residual": "-4.0",
            "abs_error": "4.0",
            "severity_group": "moderate",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_black_artifact_summary(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "video_id": "203_1_Freeform_video",
            "black_border_ratio_mean": "0.5",
            "frame_count": "100",
        },
        {
            "video_id": "203_1_Northwind_video",
            "black_border_ratio_mean": "0.1",
            "frame_count": "50",
        },
        {
            "video_id": "204_1_Freeform_video",
            "black_border_ratio_mean": "0.2",
            "frame_count": "80",
        },
        {
            "video_id": "204_1_Northwind_video",
            "black_border_ratio_mean": "0.25",
            "frame_count": "70",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_prediction_records(tmp_path):
    pred_path = tmp_path / "test_predictions.csv"
    _write_predictions_csv(pred_path)
    records = load_prediction_records(pred_path)
    assert len(records) == 4
    assert records[0]["task_name"] == "Freeform"
    assert records[1]["task_name"] == "Northwind"


def test_pair_predictions_by_subject(tmp_path):
    pred_path = tmp_path / "test_predictions.csv"
    _write_predictions_csv(pred_path)
    records = load_prediction_records(pred_path)
    pairs = pair_predictions_by_subject(records)
    assert len(pairs) == 2
    subject_203 = next(p for p in pairs if p["subject_id"] == "203_1")
    assert subject_203["freeform_pred_bdi"] == pytest.approx(12.0)
    assert subject_203["northwind_pred_bdi"] == pytest.approx(18.0)
    assert subject_203["abs_pred_bdi_diff"] == pytest.approx(6.0)


def test_compute_task_artifact_correlations():
    pairs = [
        {
            "true_bdi_diff": 1.0,
            "pred_bdi_diff": 6.0,
            "abs_pred_bdi_diff": 6.0,
            "mean_abs_error": 4.5,
            "black_border_ratio_mean_diff": 0.4,
            "black_border_ratio_mean_mean": 0.3,
        },
        {
            "true_bdi_diff": 2.0,
            "pred_bdi_diff": 2.0,
            "abs_pred_bdi_diff": 2.0,
            "mean_abs_error": 4.5,
            "black_border_ratio_mean_diff": 0.05,
            "black_border_ratio_mean_mean": 0.22,
        },
    ]
    rows = compute_task_artifact_correlations(pairs, ["black_border_ratio_mean"])
    # Should produce rows for _diff and _mean.
    assert len(rows) == 2
    diff_row = next(r for r in rows if r["variable"] == "black_border_ratio_mean_diff")
    assert diff_row["n_pairs"] == 2
    assert diff_row["with_abs_pred_bdi_diff"] is not None


def test_run_task_inconsistency_audit(tmp_path):
    pred_path = tmp_path / "test_predictions.csv"
    artifact_path = tmp_path / "black_artifact_summary.csv"
    _write_predictions_csv(pred_path)
    _write_black_artifact_summary(artifact_path)

    output_dir = tmp_path / "task_inconsistency"
    generated = run_task_inconsistency_audit(
        predictions_csv=pred_path,
        output_dir=output_dir,
        black_artifacts_summary=artifact_path,
        top_n=10,
    )

    manifest_csv = output_dir / "tables" / "task_inconsistency_manifest.csv"
    correlation_csv = output_dir / "tables" / "task_artifact_correlation.csv"
    report_md = output_dir / "reports" / "task_inconsistency_report.md"

    assert manifest_csv.exists()
    assert correlation_csv.exists()
    assert report_md.exists()
    assert manifest_csv in generated
    assert correlation_csv in generated
    assert report_md in generated

    with manifest_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["subject_id"] == "203_1"

    with correlation_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
