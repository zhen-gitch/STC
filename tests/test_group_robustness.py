import csv

import pytest

from src.diagnostics.group_robustness import (
    analyze_run_groups,
    compare_group_robustness,
    load_prediction_records_strict,
    resolve_axis_spec,
    run_group_robustness_analysis,
)
from src.diagnostics.io import write_csv_rows


PREDICTION_FIELDS = [
    "video_id",
    "subject_id",
    "task_name",
    "true_bdi",
    "pred_bdi",
    "residual",
    "abs_error",
    "severity_group",
]


def _prediction_rows(prefix, offset=0.0):
    rows = []
    targets = [8.0, 8.0, 16.0, 16.0, 24.0, 24.0, 35.0, 35.0]
    for index, target in enumerate(targets):
        subject = f"{prefix}{index // 2}"
        task = "Freeform" if index % 2 == 0 else "Northwind"
        residual = offset + (index % 4) - 1.5
        rows.append(
            {
                "video_id": f"{subject}_{task}_video",
                "subject_id": subject,
                "task_name": task,
                "true_bdi": target,
                "pred_bdi": target + residual,
                "residual": residual,
                "abs_error": abs(residual),
                "severity_group": ["minimal", "mild", "moderate", "severe"][index // 2],
            }
        )
    return rows


def _write_predictions(path, rows):
    write_csv_rows(path, rows, PREDICTION_FIELDS)
    return path


def _write_axis(path, rows, column="risk"):
    write_csv_rows(path, rows, ["video_id", column])
    return path


def test_axis_threshold_is_fit_from_train_only(tmp_path):
    train = _write_axis(
        tmp_path / "train_axis.csv",
        [{"video_id": f"T{i}", "risk": value} for i, value in enumerate([1, 2, 3, 4])],
    )
    val = _write_axis(
        tmp_path / "val_axis.csv",
        [{"video_id": f"V{i}", "risk": value} for i, value in enumerate([100, 200])],
    )

    axis = resolve_axis_spec("risk", train, val, "risk")

    assert axis["status"] == "available"
    assert axis["threshold"] == pytest.approx(2.5)
    assert axis["val_count"] == 2


def test_missing_train_axis_remains_unavailable(tmp_path):
    val = _write_axis(tmp_path / "val.csv", [{"video_id": "V1", "risk": 1.0}])

    axis = resolve_axis_spec("risk", tmp_path / "missing.csv", val, "risk")

    assert axis["status"] == "unavailable"
    assert axis["threshold"] is None
    assert axis["reason"] == "missing_train_or_val_csv"


def test_analyze_run_groups_reports_severity_task_and_axis(tmp_path):
    val_path = _write_predictions(tmp_path / "val_predictions.csv", _prediction_rows("V"))
    records = load_prediction_records_strict(val_path)
    train_axis = _write_axis(
        tmp_path / "train_axis.csv",
        [{"video_id": f"T{i}", "risk": i} for i in range(8)],
    )
    val_axis = _write_axis(
        tmp_path / "val_axis.csv",
        [
            {"video_id": row["video_id"], "risk": index}
            for index, row in enumerate(_prediction_rows("V"))
        ],
    )
    axis = resolve_axis_spec("risk", train_axis, val_axis, "risk")

    rows = analyze_run_groups(
        "C-REF",
        records,
        resolved_axes=[axis],
        bootstrap_samples=20,
    )

    assert {row["grouping"] for row in rows} == {"severity", "task", "risk"}
    risk_rows = [row for row in rows if row["grouping"] == "risk"]
    assert {row["group"] for row in risk_rows} == {"low", "high"}
    assert all(row["coverage"] == 1.0 for row in risk_rows)


def test_group_comparison_applies_frozen_failure_limits():
    rows = [
        {"run_name": "C-REF", "grouping": "task", "group": "A", "mae": 4.0},
        {"run_name": "C-REF", "grouping": "task", "group": "B", "mae": 5.0},
        {"run_name": "C-BN", "grouping": "task", "group": "A", "mae": 4.0},
        {"run_name": "C-BN", "grouping": "task", "group": "B", "mae": 5.4},
    ]

    comparisons = compare_group_robustness(rows)
    candidate = next(row for row in comparisons if row["run_name"] == "C-BN")

    assert candidate["delta_worst_group_mae"] == pytest.approx(0.4)
    assert candidate["delta_worst_group_gap"] == pytest.approx(0.4)
    assert candidate["group_utility_failure"] is True


def test_run_group_robustness_is_read_only_and_writes_outputs(tmp_path):
    ref_train = _write_predictions(tmp_path / "ref_train.csv", _prediction_rows("T"))
    ref_val = _write_predictions(tmp_path / "ref_val.csv", _prediction_rows("V"))
    bn_train = _write_predictions(tmp_path / "bn_train.csv", _prediction_rows("T"))
    bn_val = _write_predictions(tmp_path / "bn_val.csv", _prediction_rows("V", offset=1.0))
    inputs = [ref_train, ref_val, bn_train, bn_val]
    before = [path.read_bytes() for path in inputs]

    generated = run_group_robustness_analysis(
        [("C-REF", ref_train, ref_val), ("C-BN", bn_train, bn_val)],
        tmp_path / "output",
        bootstrap_samples=20,
    )

    assert [path.read_bytes() for path in inputs] == before
    assert len(generated) == 4
    comparison = tmp_path / "output" / "tables" / "group_comparison.csv"
    assert comparison.exists()
    with comparison.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["run_name"] for row in rows} == {"C-REF", "C-BN"}
    assert (tmp_path / "output" / "reports" / "group_robustness_report.md").exists()


def test_run_group_robustness_rejects_subject_overlap(tmp_path):
    rows = _prediction_rows("S")
    train = _write_predictions(tmp_path / "train.csv", rows)
    val = _write_predictions(tmp_path / "val.csv", rows)

    with pytest.raises(ValueError, match="subject overlap"):
        run_group_robustness_analysis(
            [("C-REF", train, val)],
            tmp_path / "output",
            bootstrap_samples=0,
        )


def test_multi_run_group_analysis_rejects_mismatched_metadata(tmp_path):
    ref_train = _write_predictions(tmp_path / "ref_train.csv", _prediction_rows("T"))
    ref_val = _write_predictions(tmp_path / "ref_val.csv", _prediction_rows("V"))
    bn_train = _write_predictions(tmp_path / "bn_train.csv", _prediction_rows("T"))
    bn_val = _write_predictions(tmp_path / "bn_val.csv", _prediction_rows("X"))

    with pytest.raises(ValueError, match="metadata differs"):
        run_group_robustness_analysis(
            [("C-REF", ref_train, ref_val), ("C-BN", bn_train, bn_val)],
            tmp_path / "output",
            bootstrap_samples=0,
        )
