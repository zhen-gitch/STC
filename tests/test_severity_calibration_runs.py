import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.diagnostics.severity_calibration_runs import (
    load_severity_calibration_fit,
    load_severity_calibration_group_bias,
    load_severity_calibration_test_summary,
    summarize_severity_calibration_runs,
)


def _write_fit_csv(path, a=0.8, b=3.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "a": f"{a:.6f}",
        "b": f"{b:.6f}",
        "val_count": "100",
        "val_mae": "8.5",
        "val_rmse": "10.5",
        "val_pearson": "0.45",
        "val_ccc": "0.36",
        "val_true_mean": "15.0",
        "val_true_std": "11.5",
        "val_pred_mean": "14.0",
        "val_pred_std": "6.0",
    }
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _write_test_summary_csv(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "version": "original",
            "count": "100",
            "mae": "9.0",
            "rmse": "11.0",
            "pearson": "0.35",
            "ccc": "0.29",
            "true_mean": "15.0",
            "true_std": "11.5",
            "pred_mean": "14.0",
            "pred_std": "6.0",
        },
        {
            "version": "calibrated",
            "count": "100",
            "mae": "8.8",
            "rmse": "10.8",
            "pearson": "0.35",
            "ccc": "0.27",
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


def _write_group_bias_csv(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"version": "original", "severity_group": "minimal", "count": "20", "mean_residual": "6.5", "mae": "7.0"},
        {"version": "original", "severity_group": "severe", "count": "15", "mean_residual": "-16.0", "mae": "16.0"},
        {"version": "calibrated", "severity_group": "minimal", "count": "20", "mean_residual": "7.0", "mae": "7.5"},
        {"version": "calibrated", "severity_group": "severe", "count": "15", "mean_residual": "-15.5", "mae": "15.5"},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_load_severity_calibration_fit(tmp_path):
    fit_path = tmp_path / "severity_calibration_fit.csv"
    _write_fit_csv(fit_path, a=0.87, b=2.69)
    fit = load_severity_calibration_fit(fit_path)
    assert fit["a"] == pytest.approx(0.87)
    assert fit["b"] == pytest.approx(2.69)
    assert fit["val_count"] == 100
    assert fit["val_mae"] == pytest.approx(8.5)


def test_load_severity_calibration_test_summary(tmp_path):
    summary_path = tmp_path / "severity_calibration_test_summary.csv"
    _write_test_summary_csv(summary_path)
    summary = load_severity_calibration_test_summary(summary_path)
    assert summary["original"]["mae"] == pytest.approx(9.0)
    assert summary["calibrated"]["mae"] == pytest.approx(8.8)
    assert summary["original"]["count"] == 100


def test_load_severity_calibration_group_bias(tmp_path):
    bias_path = tmp_path / "severity_calibration_group_bias.csv"
    _write_group_bias_csv(bias_path)
    records = load_severity_calibration_group_bias(bias_path)
    assert len(records) == 4
    assert records[0]["version"] == "original"
    assert records[0]["severity_group"] == "minimal"
    assert records[0]["mean_residual"] == pytest.approx(6.5)


def test_summarize_severity_calibration_runs(tmp_path):
    run_dir_a = tmp_path / "rgb"
    run_dir_b = tmp_path / "center_mask"
    _write_fit_csv(run_dir_a / "tables" / "severity_calibration_fit.csv", a=0.87, b=2.69)
    _write_fit_csv(run_dir_b / "tables" / "severity_calibration_fit.csv", a=0.80, b=3.00)
    _write_test_summary_csv(run_dir_a / "tables" / "severity_calibration_test_summary.csv")
    _write_test_summary_csv(run_dir_b / "tables" / "severity_calibration_test_summary.csv")
    _write_group_bias_csv(run_dir_a / "tables" / "severity_calibration_group_bias.csv")
    _write_group_bias_csv(run_dir_b / "tables" / "severity_calibration_group_bias.csv")

    output_dir = tmp_path / "summary"
    generated = summarize_severity_calibration_runs(
        run_specs=[("rgb", run_dir_a), ("center_mask", run_dir_b)],
        output_dir=output_dir,
    )

    summary_csv = output_dir / "tables" / "severity_calibration_run_summary.csv"
    bias_csv = output_dir / "tables" / "severity_calibration_group_bias_summary.csv"
    report_md = output_dir / "reports" / "severity_calibration_runs_report.md"

    assert summary_csv.exists()
    assert bias_csv.exists()
    assert report_md.exists()
    assert summary_csv in generated
    assert bias_csv in generated
    assert report_md in generated

    with summary_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["run_name"] for row in rows} == {"rgb", "center_mask"}
    rgb_row = next(row for row in rows if row["run_name"] == "rgb")
    assert float(rgb_row["a"]) == pytest.approx(0.87)
    assert float(rgb_row["delta_mae"]) == pytest.approx(-0.2)

    with bias_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8  # 2 runs x 2 groups x 2 versions
