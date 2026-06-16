import csv

from src.diagnostics.training_overfit import summarize_training_overfit, summarize_training_run


def _write_metrics(path, rows):
    fieldnames = [
        "epoch",
        "step",
        "train_RMSE_epoch",
        "train_MAE_epoch",
        "train_loss",
        "val_RMSE_epoch",
        "val_MAE_epoch",
        "val_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_summarize_training_run_detects_overfit_after_best_val(tmp_path):
    metrics_csv = _write_metrics(
        tmp_path / "metrics.csv",
        [
            {"epoch": 0, "train_RMSE_epoch": 12, "train_MAE_epoch": 9, "train_loss": 0.50},
            {"epoch": 0, "val_RMSE_epoch": 13, "val_MAE_epoch": 10, "val_loss": 0.60},
            {"epoch": 1, "train_RMSE_epoch": 8, "train_MAE_epoch": 6, "train_loss": 0.30},
            {"epoch": 1, "val_RMSE_epoch": 10, "val_MAE_epoch": 8, "val_loss": 0.45},
            {"epoch": 2, "train_RMSE_epoch": 5, "train_MAE_epoch": 4, "train_loss": 0.20},
            {"epoch": 2, "val_RMSE_epoch": 12, "val_MAE_epoch": 9, "val_loss": 0.55},
        ],
    )

    summary, curve_rows = summarize_training_run("rgb", metrics_csv)

    assert len(curve_rows) == 3
    assert summary["best_monitor"] == "rmse"
    assert summary["best_val_epoch"] == 1
    assert summary["best_val_rmse"] == 10
    assert summary["train_rmse_at_best_val"] == 8
    assert summary["train_val_rmse_gap"] == 2
    assert summary["last_epoch"] == 2
    assert summary["val_degradation_after_best"] == 2
    assert summary["train_improvement_after_best"] == 3
    assert summary["overfit_after_best_val"] is True


def test_summarize_training_overfit_writes_tables_and_report(tmp_path):
    rgb_csv = _write_metrics(
        tmp_path / "rgb_metrics.csv",
        [
            {"epoch": 0, "train_RMSE_epoch": 12, "val_RMSE_epoch": 13},
            {"epoch": 1, "train_RMSE_epoch": 8, "val_RMSE_epoch": 10},
        ],
    )
    center_csv = _write_metrics(
        tmp_path / "center_metrics.csv",
        [
            {"epoch": 0, "train_RMSE_epoch": 11, "val_RMSE_epoch": 12},
            {"epoch": 1, "train_RMSE_epoch": 9, "val_RMSE_epoch": 9},
        ],
    )

    generated = summarize_training_overfit(
        [("rgb", rgb_csv), ("center_mask", center_csv)],
        output_dir=tmp_path / "training_overfit",
    )

    assert len(generated) == 3
    for path in generated:
        assert path.exists()

    summary_path = tmp_path / "training_overfit" / "tables" / "training_overfit_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    assert summary_rows[0]["run"] == "center_mask"
    assert "train_val_gap_at_best" in summary_rows[0]

    curve_path = tmp_path / "training_overfit" / "tables" / "training_curve_gap_by_run.csv"
    with curve_path.open(newline="", encoding="utf-8") as f:
        curve_rows = list(csv.DictReader(f))
    assert len(curve_rows) == 4
    assert set(curve_rows[0].keys()) >= {"run", "epoch", "rmse_gap"}

    report = (tmp_path / "training_overfit" / "reports" / "training_overfit_report.md").read_text(
        encoding="utf-8"
    )
    assert "Training Overfit Summary" in report
