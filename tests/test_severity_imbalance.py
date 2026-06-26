"""Tests for RPDF Stage A4: severity imbalance / prediction compression summary.

Pure-csv tests verifying the join of prediction / severity-bias / calibration
summaries, the recommendation heuristic, and output schema.
"""

import csv

import pytest

from src.diagnostics.io import write_csv_rows
from src.diagnostics.severity_imbalance import (
    SUMMARY_COLUMNS,
    _recommend_severity_balanced,
    build_severity_imbalance_rows,
    run_severity_imbalance_audit,
)


def _write_prediction_summary(tmp_path, runs):
    rows = [
        {
            "run": name,
            "count": data["count"],
            "mae": data.get("mae", 8.0),
            "rmse": data.get("rmse", 10.0),
            "pearson": data.get("pearson", 0.35),
            "ccc": data.get("ccc", 0.29),
            "true_std": data.get("true_std", 11.0),
            "pred_std": data.get("pred_std", 6.0),
        }
        for name, data in runs.items()
    ]
    path = tmp_path / "pred_summary.csv"
    write_csv_rows(
        path,
        rows,
        ["run", "count", "mae", "rmse", "pearson", "ccc", "true_std", "pred_std"],
    )
    return path


def _write_severity_bias(tmp_path, runs):
    rows = []
    for name, data in runs.items():
        for group in ("minimal", "mild", "moderate", "severe"):
            g = data.get(group, {"count": 0, "bias": None})
            rows.append(
                {
                    "run": name,
                    "severity_group": group,
                    "count": g["count"],
                    "bias": g.get("bias"),
                    "mae": g.get("mae", 8.0),
                }
            )
    path = tmp_path / "severity_bias.csv"
    write_csv_rows(path, rows, ["run", "severity_group", "count", "bias", "mae"])
    return path


def _write_calibration_summary(tmp_path, runs):
    rows = [
        {"run_name": name, "delta_ccc": data.get("delta_ccc", -0.02)}
        for name, data in runs.items()
    ]
    path = tmp_path / "calibration.csv"
    write_csv_rows(path, rows, ["run_name", "delta_ccc"])
    return path


def test_recommend_stage_b_baseline_when_imbalanced_and_systematic():
    rec = _recommend_severity_balanced(
        imbalance_ratio=4.0,
        minimal_residual=6.0,
        severe_residual=-16.0,
        compression=0.55,
        calibration_delta_ccc=-0.02,
    )
    assert rec.startswith("stage_b_baseline")


def test_recommend_stage_d_when_calibration_helps():
    rec = _recommend_severity_balanced(
        imbalance_ratio=4.0,
        minimal_residual=6.0,
        severe_residual=-16.0,
        compression=0.55,
        calibration_delta_ccc=0.05,
    )
    assert rec.startswith("stage_d_side_branch")


def test_recommend_stage_d_optional_when_balanced():
    # Truly balanced: no imbalance, no systematic bias (residuals small / same sign),
    # no compression. Any of these failing would push to side_branch.
    rec = _recommend_severity_balanced(
        imbalance_ratio=1.1,
        minimal_residual=0.2,
        severe_residual=0.1,  # same sign -> not systematic central-tendency bias
        compression=0.95,
        calibration_delta_ccc=0.0,
    )
    assert rec.startswith("stage_d_optional")


def test_build_rows_joins_three_sources(tmp_path):
    runs = {
        "rgb": {
            "count": 100,
            "minimal": {"count": 40, "bias": 6.89},
            "mild": {"count": 25, "bias": -1.0},
            "moderate": {"count": 20, "bias": -4.0},
            "severe": {"count": 15, "bias": -16.5},
            "true_std": 11.48,
            "pred_std": 6.13,
            "ccc": 0.29,
            "delta_ccc": -0.02,
        }
    }
    pred = _write_prediction_summary(tmp_path, runs)
    bias = _write_severity_bias(tmp_path, runs)
    calib = _write_calibration_summary(tmp_path, runs)

    rows = build_severity_imbalance_rows(
        prediction_summary_csv=pred,
        severity_bias_csv=bias,
        calibration_summary_csv=calib,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["run_name"] == "rgb"
    assert row["total_count"] == 100
    assert row["bin_count_minimal"] == 40
    assert row["bin_count_severe"] == 15
    assert row["imbalance_ratio"] == pytest.approx(40 / 15, abs=1e-6)
    assert row["minimal_residual"] == pytest.approx(6.89)
    assert row["severe_residual"] == pytest.approx(-16.5)
    assert row["pred_compression_ratio"] == pytest.approx(6.13 / 11.48, abs=1e-6)
    assert row["calibration_delta_ccc"] == pytest.approx(-0.02)
    assert row["severity_balanced_recommendation"].startswith("stage_b_baseline")


def test_run_audit_emits_outputs_and_schema(tmp_path):
    runs = {
        "rgb": {
            "count": 100,
            "minimal": {"count": 40, "bias": 6.89},
            "mild": {"count": 25, "bias": -1.0},
            "moderate": {"count": 20, "bias": -4.0},
            "severe": {"count": 15, "bias": -16.5},
            "true_std": 11.48, "pred_std": 6.13, "ccc": 0.29, "delta_ccc": -0.02,
        },
        "center_mask": {
            "count": 100,
            "minimal": {"count": 40, "bias": 3.0},
            "mild": {"count": 25, "bias": 0.5},
            "moderate": {"count": 20, "bias": -2.0},
            "severe": {"count": 15, "bias": -12.0},
            "true_std": 11.48, "pred_std": 8.13, "ccc": 0.48, "delta_ccc": -0.03,
        },
    }
    pred = _write_prediction_summary(tmp_path, runs)
    bias = _write_severity_bias(tmp_path, runs)
    calib = _write_calibration_summary(tmp_path, runs)

    generated = run_severity_imbalance_audit(
        output_dir=tmp_path / "out",
        prediction_summary_csv=pred,
        severity_bias_csv=bias,
        calibration_summary_csv=calib,
    )
    summary_path = tmp_path / "out" / "tables" / "severity_imbalance_summary.csv"
    report_path = tmp_path / "out" / "reports" / "severity_imbalance_report.md"
    assert summary_path in generated
    assert report_path in generated
    assert summary_path.exists() and report_path.exists()

    with open(summary_path) as f:
        header = next(csv.reader(f))
    assert set(SUMMARY_COLUMNS).issubset(set(header))

    from src.diagnostics.io import read_csv_rows

    rows = read_csv_rows(summary_path)
    assert len(rows) == 2
    assert {r["run_name"] for r in rows} == {"rgb", "center_mask"}


def test_run_audit_works_with_partial_sources(tmp_path):
    """Only prediction summary supplied still yields rows (no bias/calibration)."""
    runs = {"rgb": {"count": 100, "true_std": 11.0, "pred_std": 6.0, "ccc": 0.29}}
    pred = _write_prediction_summary(tmp_path, runs)
    generated = run_severity_imbalance_audit(
        output_dir=tmp_path / "out",
        prediction_summary_csv=pred,
    )
    from src.diagnostics.io import read_csv_rows

    rows = read_csv_rows(tmp_path / "out" / "tables" / "severity_imbalance_summary.csv")
    assert len(rows) == 1
    assert rows[0]["run_name"] == "rgb"
    # No severity bias -> imbalance/bins empty, recommendation defaults to optional.
    assert rows[0]["severity_balanced_recommendation"].startswith("stage_d")


def test_run_audit_rejects_empty_inputs(tmp_path):
    with pytest.raises(RuntimeError, match="No runs found"):
        run_severity_imbalance_audit(output_dir=tmp_path / "out")
