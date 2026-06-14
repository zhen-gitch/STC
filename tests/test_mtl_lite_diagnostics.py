import csv

import pytest

from src.diagnostics.correlation import plot_predictions_correlation_heatmap
from src.diagnostics.io import read_prediction_table, write_prediction_table
from src.diagnostics.regression import plot_regression_diagnostics, regression_summary
from src.diagnostics.reports import write_diagnostic_report
from src.diagnostics.training_curves import plot_training_curves


pytest.importorskip("matplotlib")


def test_prediction_table_and_regression_plots(tmp_path):
    prediction_csv = tmp_path / "predictions.csv"
    records = write_prediction_table(
        prediction_csv,
        subject_ids=["001", "002", "003", "004"],
        task_names=["Freeform", "Northwind", "Freeform", "Northwind"],
        targets=[3, 15, 27, 42],
        preds=[4, 13, 30, 38],
    )

    loaded = read_prediction_table(prediction_csv)
    summary = regression_summary(loaded)
    output_dir = tmp_path / "regression"
    paths = plot_regression_diagnostics(prediction_csv, output_dir)

    assert len(records) == 4
    assert loaded[0]["task_name"] == "Freeform"
    assert summary["count"] == 4
    assert summary["mae"] > 0
    assert (output_dir / "prediction_target_scatter.png").exists()
    assert (output_dir / "residual_histogram.png").exists()
    assert (output_dir / "high_error_subjects.csv").exists()
    assert (output_dir / "case_study_manifest.csv").exists()
    assert (output_dir / "case_study_manifest.md").exists()
    assert paths


def test_training_curves_and_correlation_plots(tmp_path):
    metrics_csv = tmp_path / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "step",
                "train_loss",
                "val_loss",
                "val_RMSE_epoch",
                "val_MAE_epoch",
                "lr-AdamW",
            ],
        )
        writer.writeheader()
        for epoch in range(3):
            writer.writerow(
                {
                    "epoch": epoch,
                    "step": epoch,
                    "train_loss": 1.0 / (epoch + 1),
                    "val_loss": 1.2 / (epoch + 1),
                    "val_RMSE_epoch": 10 - epoch,
                    "val_MAE_epoch": 8 - epoch,
                    "lr-AdamW": 1e-4,
                }
            )

    prediction_csv = tmp_path / "predictions.csv"
    write_prediction_table(
        prediction_csv,
        subject_ids=["001", "002", "003", "004"],
        targets=[3, 15, 27, 42],
        preds=[4, 13, 30, 38],
    )

    curves_path = plot_training_curves(metrics_csv, tmp_path / "training")
    corr_path = plot_predictions_correlation_heatmap(prediction_csv, tmp_path / "correlation")

    assert curves_path.exists()
    assert corr_path.exists()


def test_diagnostic_report(tmp_path):
    prediction_csv = tmp_path / "predictions.csv"
    write_prediction_table(
        prediction_csv,
        subject_ids=["001", "002"],
        targets=[10, 20],
        preds=[12, 17],
    )
    records = read_prediction_table(prediction_csv)
    report_path = write_diagnostic_report(
        tmp_path / "reports" / "diagnostic_report.md",
        run_dir=tmp_path,
        checkpoint_path=tmp_path / "checkpoints" / "best.ckpt",
        records=records,
        generated_files=[prediction_csv],
    )

    assert report_path.exists()
    assert "MTL-Lite Diagnostic Report" in report_path.read_text(encoding="utf-8")
