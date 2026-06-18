import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.diagnostics.io import write_prediction_table
from src.diagnostics.severity_calibration import (
    apply_calibration,
    evaluate_metrics,
    fit_linear_calibration,
    run_severity_calibration_audit,
    severity_group_bias,
    write_severity_calibration_fit,
    write_severity_calibration_report,
    write_severity_calibration_test_summary,
)


def _make_prediction_row(video_id, true_bdi, pred_bdi, task_name="Freeform"):
    residual = pred_bdi - true_bdi
    return {
        "video_id": video_id,
        "subject_id": video_id.split("_")[0] + "_1",
        "task_name": task_name,
        "true_bdi": true_bdi,
        "pred_bdi": pred_bdi,
        "residual": residual,
        "abs_error": abs(residual),
        "severity_group": _severity_group(true_bdi),
    }


def _severity_group(bdi):
    if bdi < 14:
        return "minimal"
    if bdi < 20:
        return "mild"
    if bdi < 28:
        return "moderate"
    return "severe"


def _make_synthetic_predictions(rng, n=30, slope=0.6, intercept=8.0, noise=1.5):
    """Generate predictions where true = slope * pred + intercept + noise."""
    true_bdi = rng.uniform(5, 50, size=n)
    pred_bdi = (true_bdi - intercept) / slope + rng.normal(scale=noise, size=n)
    rows = []
    for i, (true, pred) in enumerate(zip(true_bdi, pred_bdi)):
        task = "Freeform" if i % 2 == 0 else "Northwind"
        rows.append(_make_prediction_row(f"subj_{i:03d}", true, pred, task))
    return rows


def test_fit_linear_calibration_recovers_mapping():
    rng = np.random.default_rng(0)
    # Build val set with known linear relationship.
    val_true = rng.uniform(5, 50, size=50)
    val_pred = (val_true - 10.0) / 0.5 + rng.normal(scale=0.5, size=50)
    val_records = [
        _make_prediction_row(f"v_{i}", t, p) for i, (t, p) in enumerate(zip(val_true, val_pred))
    ]
    a, b = fit_linear_calibration(val_records)
    assert a == pytest.approx(0.5, abs=0.05)
    assert b == pytest.approx(10.0, abs=0.8)


def test_fit_linear_calibration_rejects_low_variance():
    records = [
        _make_prediction_row("v1", 10.0, 5.0),
        _make_prediction_row("v2", 12.0, 5.0),
        _make_prediction_row("v3", 14.0, 5.0),
    ]
    with pytest.raises(ValueError, match="zero variance"):
        fit_linear_calibration(records)


def test_apply_calibration_updates_residuals():
    records = [
        _make_prediction_row("v1", 10.0, 8.0),
        _make_prediction_row("v2", 20.0, 18.0),
    ]
    calibrated = apply_calibration(records, a=1.0, b=2.0)
    assert calibrated[0]["pred_bdi"] == pytest.approx(10.0)
    assert calibrated[0]["residual"] == pytest.approx(0.0)
    assert calibrated[0]["abs_error"] == pytest.approx(0.0)
    assert calibrated[1]["pred_bdi"] == pytest.approx(20.0)
    assert calibrated[1]["residual"] == pytest.approx(0.0)


def test_evaluate_metrics_basic():
    records = [
        _make_prediction_row("v1", 10.0, 12.0),
        _make_prediction_row("v2", 20.0, 18.0),
    ]
    metrics = evaluate_metrics(records)
    assert metrics["count"] == 2
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(2.0)
    assert metrics["pearson"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["true_std"] > 0
    assert metrics["pred_std"] > 0


def test_severity_group_bias():
    records = [
        _make_prediction_row("v1", 8.0, 10.0),
        _make_prediction_row("v2", 10.0, 14.0),
        _make_prediction_row("v3", 25.0, 20.0),
    ]
    bias = severity_group_bias(records)
    assert "minimal" in bias
    assert "moderate" in bias
    assert bias["minimal"]["mean_residual"] == pytest.approx(3.0)
    assert bias["moderate"]["mean_residual"] == pytest.approx(-5.0)


def test_write_severity_calibration_fit(tmp_path):
    records = [
        _make_prediction_row("v1", 10.0, 12.0),
        _make_prediction_row("v2", 20.0, 18.0),
    ]
    metrics = evaluate_metrics(records)
    csv_path = tmp_path / "fit.csv"
    write_severity_calibration_fit(csv_path, a=0.5, b=10.0, val_metrics=metrics)
    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    assert float(row["a"]) == pytest.approx(0.5)
    assert float(row["b"]) == pytest.approx(10.0)


def test_write_severity_calibration_test_summary(tmp_path):
    original = {"count": 2, "mae": 2.0, "rmse": 2.0, "pearson": 0.5, "ccc": 0.4,
                "true_mean": 15.0, "true_std": 5.0, "pred_mean": 14.0, "pred_std": 4.0}
    calibrated = {"count": 2, "mae": 1.0, "rmse": 1.0, "pearson": 0.5, "ccc": 0.45,
                  "true_mean": 15.0, "true_std": 5.0, "pred_mean": 15.0, "pred_std": 5.0}
    csv_path = tmp_path / "summary.csv"
    write_severity_calibration_test_summary(csv_path, original, calibrated)
    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["version"] == "original"
    assert rows[1]["version"] == "calibrated"


def test_run_severity_calibration_audit(tmp_path):
    rng = np.random.default_rng(42)
    val_records = _make_synthetic_predictions(rng, n=40, slope=0.6, intercept=8.0, noise=1.5)
    test_records = _make_synthetic_predictions(rng, n=20, slope=0.6, intercept=8.0, noise=1.5)

    val_csv = tmp_path / "val_predictions.csv"
    test_csv = tmp_path / "test_predictions.csv"
    write_prediction_table(
        val_csv,
        subject_ids=[r["subject_id"] for r in val_records],
        targets=[r["true_bdi"] for r in val_records],
        preds=[r["pred_bdi"] for r in val_records],
        video_ids=[r["video_id"] for r in val_records],
        task_names=[r["task_name"] for r in val_records],
    )
    write_prediction_table(
        test_csv,
        subject_ids=[r["subject_id"] for r in test_records],
        targets=[r["true_bdi"] for r in test_records],
        preds=[r["pred_bdi"] for r in test_records],
        video_ids=[r["video_id"] for r in test_records],
        task_names=[r["task_name"] for r in test_records],
    )

    out = tmp_path / "severity_calibration"
    generated = run_severity_calibration_audit(
        val_predictions=val_csv,
        test_predictions=test_csv,
        output_dir=out,
        max_score=63,
    )

    assert (out / "tables" / "severity_calibration_fit.csv").exists()
    assert (out / "tables" / "severity_calibration_test_summary.csv").exists()
    assert (out / "figures" / "calibration_scatter.png").exists()
    assert (out / "figures" / "severity_group_residual.png").exists()
    assert (out / "figures" / "severity_group_mae.png").exists()
    assert (out / "reports" / "severity_calibration_report.md").exists()
    assert generated

    # Verify that calibration improves MAE on the synthetic test set.
    summary_csv = out / "tables" / "severity_calibration_test_summary.csv"
    with summary_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    original_mae = float(rows[0]["mae"])
    calibrated_mae = float(rows[1]["mae"])
    assert calibrated_mae < original_mae


def test_run_severity_calibration_audit_empty_val(tmp_path):
    val_csv = tmp_path / "empty_val.csv"
    test_csv = tmp_path / "empty_test.csv"
    write_prediction_table(val_csv, [], [], [], [], [])
    write_prediction_table(test_csv, [], [], [], [], [])

    with pytest.raises(ValueError, match="No validation records"):
        run_severity_calibration_audit(val_csv, test_csv, tmp_path / "out")
