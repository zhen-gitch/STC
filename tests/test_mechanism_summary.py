import csv
from pathlib import Path

import pytest

from src.diagnostics.mechanism_summary import (
    build_mechanism_rows,
    build_mechanism_summary,
    write_mechanism_summary,
)


def _write_prediction_summary(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_name": "rgb",
            "count": "100",
            "mae": "8.5",
            "rmse": "10.5",
            "pearson": "0.45",
            "ccc": "0.36",
            "true_mean": "15.0",
            "true_std": "11.5",
            "pred_mean": "14.0",
            "pred_std": "6.0",
        },
        {
            "run_name": "center_mask",
            "count": "100",
            "mae": "9.0",
            "rmse": "11.0",
            "pearson": "0.40",
            "ccc": "0.30",
            "true_mean": "15.0",
            "true_std": "11.5",
            "pred_mean": "14.5",
            "pred_std": "5.5",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_severity_bias_summary(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"run_name": "rgb", "severity_group": "minimal", "mean_residual": "6.5", "mae": "7.0", "count": "20"},
        {"run_name": "rgb", "severity_group": "severe", "mean_residual": "-16.0", "mae": "16.0", "count": "15"},
        {"run_name": "center_mask", "severity_group": "minimal", "mean_residual": "7.0", "mae": "7.5", "count": "20"},
        {"run_name": "center_mask", "severity_group": "severe", "mean_residual": "-15.5", "mae": "15.5", "count": "15"},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_task_consistency_summary(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"run_name": "rgb", "task_name": "Freeform", "count": "50", "mae": "8.0", "rmse": "10.0", "pearson": "0.45", "ccc": "0.36", "mean_abs_pred_diff": "4.2"},
        {"run_name": "rgb", "task_name": "Northwind", "count": "50", "mae": "9.0", "rmse": "11.0", "pearson": "0.42", "ccc": "0.34", "mean_abs_pred_diff": "4.2"},
        {"run_name": "center_mask", "task_name": "Freeform", "count": "50", "mae": "8.5", "rmse": "10.5", "pearson": "0.40", "ccc": "0.30", "mean_abs_pred_diff": "5.1"},
        {"run_name": "center_mask", "task_name": "Northwind", "count": "50", "mae": "9.5", "rmse": "11.5", "pearson": "0.38", "ccc": "0.28", "mean_abs_pred_diff": "5.1"},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_identity_summary(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_name": "rgb",
            "same_subject_top1_rate": "0.85",
            "same_subject_top5_rate": "0.92",
            "severity_neighbor_agreement_top5_mean": "0.65",
            "task_neighbor_agreement_top5_mean": "0.40",
        },
        {
            "run_name": "center_mask",
            "same_subject_top1_rate": "0.80",
            "same_subject_top5_rate": "0.88",
            "severity_neighbor_agreement_top5_mean": "0.60",
            "task_neighbor_agreement_top5_mean": "0.45",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_calibration_summary(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"run_name": "rgb", "a": "0.87", "b": "2.69", "delta_mae": "-0.2", "delta_rmse": "-0.2", "delta_pearson": "0.00", "delta_ccc": "-0.02"},
        {"run_name": "center_mask", "a": "0.80", "b": "3.00", "delta_mae": "-0.1", "delta_rmse": "-0.1", "delta_pearson": "0.00", "delta_ccc": "-0.01"},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_mechanism_rows(tmp_path):
    pred_path = tmp_path / "prediction_run_summary.csv"
    bias_path = tmp_path / "severity_bias_summary.csv"
    task_path = tmp_path / "task_consistency_summary.csv"
    identity_path = tmp_path / "identity_retrieval_run_summary.csv"
    calibration_path = tmp_path / "severity_calibration_run_summary.csv"

    _write_prediction_summary(pred_path)
    _write_severity_bias_summary(bias_path)
    _write_task_consistency_summary(task_path)
    _write_identity_summary(identity_path)
    _write_calibration_summary(calibration_path)

    # Import loaders here to keep test focused.
    from src.diagnostics.mechanism_summary import (
        _load_calibration_summary,
        _load_identity_summary,
        _load_prediction_summary,
        _load_severity_bias_summary,
        _load_task_consistency_summary,
    )

    prediction_summary = _load_prediction_summary(pred_path)
    severity_bias_summary = _load_severity_bias_summary(bias_path)
    task_consistency_summary = _load_task_consistency_summary(task_path)
    identity_summary = _load_identity_summary(identity_path)
    calibration_summary = _load_calibration_summary(calibration_path)

    rows = build_mechanism_rows(
        prediction_summary,
        severity_bias_summary,
        task_consistency_summary,
        identity_summary,
        calibration_summary,
    )

    assert len(rows) == 2
    rgb_row = next(row for row in rows if row["run_name"] == "rgb")
    assert rgb_row["mae"] == pytest.approx(8.5)
    assert rgb_row["pred_compression_ratio"] == pytest.approx(6.0 / 11.5)
    assert rgb_row["minimal_residual_original"] == pytest.approx(6.5)
    assert rgb_row["severe_residual_original"] == pytest.approx(-16.0)
    assert rgb_row["task_diff_mean"] == pytest.approx(4.2)
    assert rgb_row["same_subject_top1_rate"] == pytest.approx(0.85)
    assert rgb_row["calibration_delta_mae"] == pytest.approx(-0.2)


def test_build_mechanism_summary_full_pipeline(tmp_path):
    pred_path = tmp_path / "prediction_run_summary.csv"
    bias_path = tmp_path / "severity_bias_summary.csv"
    task_path = tmp_path / "task_consistency_summary.csv"
    identity_path = tmp_path / "identity_retrieval_run_summary.csv"
    calibration_path = tmp_path / "severity_calibration_run_summary.csv"

    _write_prediction_summary(pred_path)
    _write_severity_bias_summary(bias_path)
    _write_task_consistency_summary(task_path)
    _write_identity_summary(identity_path)
    _write_calibration_summary(calibration_path)

    output_dir = tmp_path / "mechanism_summary"
    generated = build_mechanism_summary(
        prediction_summary_csv=pred_path,
        output_dir=output_dir,
        severity_bias_summary_csv=bias_path,
        task_consistency_summary_csv=task_path,
        identity_summary_csv=identity_path,
        calibration_summary_csv=calibration_path,
    )

    summary_csv = output_dir / "tables" / "mechanism_summary.csv"
    report_md = output_dir / "reports" / "mechanism_report.md"
    assert summary_csv.exists()
    assert report_md.exists()
    assert summary_csv in generated
    assert report_md in generated

    with summary_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["run_name"] for row in rows} == {"rgb", "center_mask"}


def test_write_mechanism_summary_missing_sources(tmp_path):
    pred_path = tmp_path / "prediction_run_summary.csv"
    _write_prediction_summary(pred_path)

    from src.diagnostics.mechanism_summary import _load_prediction_summary

    rows = build_mechanism_rows(
        _load_prediction_summary(pred_path),
        {},
        {},
        {},
        {},
    )
    assert len(rows) == 2
    assert rows[0]["minimal_residual_original"] is None
    assert rows[0]["same_subject_top1_rate"] is None
    assert rows[0]["calibration_delta_mae"] is None

    summary_path = tmp_path / "mechanism_summary.csv"
    write_mechanism_summary(summary_path, rows)
    assert summary_path.exists()
