import math
from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, read_csv_rows, write_csv_rows


CURVE_FIELDS = [
    "run",
    "epoch",
    "train_rmse",
    "val_rmse",
    "rmse_gap",
    "train_mae",
    "val_mae",
    "mae_gap",
    "train_loss",
    "val_loss",
    "loss_gap",
]

SUMMARY_FIELDS = [
    "run",
    "metrics_csv",
    "epoch_count",
    "best_monitor",
    "best_val_epoch",
    "best_val_value",
    "train_value_at_best_val",
    "train_val_gap_at_best",
    "best_val_rmse",
    "train_rmse_at_best_val",
    "train_val_rmse_gap",
    "best_val_mae",
    "train_mae_at_best_val",
    "train_val_mae_gap",
    "last_epoch",
    "last_train_rmse",
    "last_val_rmse",
    "last_train_mae",
    "last_val_mae",
    "last_train_val_rmse_gap",
    "val_degradation_after_best",
    "train_improvement_after_best",
    "overfit_after_best_val",
    "status",
]

METRIC_COLUMNS = {
    "train_rmse": "train_RMSE_epoch",
    "val_rmse": "val_RMSE_epoch",
    "train_mae": "train_MAE_epoch",
    "val_mae": "val_MAE_epoch",
    "train_loss": "train_loss",
    "val_loss": "val_loss",
}


def _safe_float(value):
    if value in (None, ""):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _format_rows(rows, fields):
    return [{field: _format_value(row.get(field)) for field in fields} for row in rows]


def _epoch_metric_map(rows):
    values = {}
    for row in rows:
        epoch = _safe_float(row.get("epoch"))
        if epoch is None:
            continue
        epoch = int(epoch)
        epoch_values = values.setdefault(epoch, {name: [] for name in METRIC_COLUMNS})
        for name, column in METRIC_COLUMNS.items():
            value = _safe_float(row.get(column))
            if value is not None:
                epoch_values[name].append(value)

    epoch_rows = []
    for epoch in sorted(values):
        item = {"epoch": epoch}
        for name, name_values in values[epoch].items():
            item[name] = float(np.mean(name_values)) if name_values else None
        item["rmse_gap"] = _gap(item.get("val_rmse"), item.get("train_rmse"))
        item["mae_gap"] = _gap(item.get("val_mae"), item.get("train_mae"))
        item["loss_gap"] = _gap(item.get("val_loss"), item.get("train_loss"))
        epoch_rows.append(item)
    return epoch_rows


def _gap(val_value, train_value):
    if val_value is None or train_value is None:
        return None
    return float(val_value - train_value)


def _select_monitor(epoch_rows):
    candidates = [
        ("rmse", "val_rmse", "train_rmse"),
        ("mae", "val_mae", "train_mae"),
        ("loss", "val_loss", "train_loss"),
    ]
    for monitor_name, val_key, train_key in candidates:
        valid = [row for row in epoch_rows if row.get(val_key) is not None]
        if valid:
            return monitor_name, val_key, train_key
    return "", "", ""


def _best_epoch_row(epoch_rows, val_key):
    valid = [row for row in epoch_rows if row.get(val_key) is not None]
    if not valid:
        return None
    return min(valid, key=lambda row: (row[val_key], row["epoch"]))


def _last_epoch_row(epoch_rows, val_key):
    valid = [row for row in epoch_rows if row.get(val_key) is not None]
    if not valid:
        return epoch_rows[-1] if epoch_rows else None
    return max(valid, key=lambda row: row["epoch"])


def summarize_training_run(name, metrics_csv):
    metrics_csv = Path(metrics_csv)
    rows = read_csv_rows(metrics_csv)
    epoch_rows = _epoch_metric_map(rows)
    for row in epoch_rows:
        row["run"] = str(name)

    monitor_name, val_key, train_key = _select_monitor(epoch_rows)
    if not epoch_rows or not val_key:
        return _empty_summary(name, metrics_csv), epoch_rows

    best_row = _best_epoch_row(epoch_rows, val_key)
    last_row = _last_epoch_row(epoch_rows, val_key)
    train_at_best = best_row.get(train_key)
    best_val_value = best_row.get(val_key)
    last_train_value = last_row.get(train_key)
    last_val_value = last_row.get(val_key)
    val_degradation = _gap(last_val_value, best_val_value)
    train_improvement = _gap(train_at_best, last_train_value)

    overfit_after_best = False
    status = "ok"
    if best_row is None or last_row is None:
        status = "insufficient_metrics"
    elif best_row["epoch"] == last_row["epoch"]:
        status = "best_is_last_epoch"
    elif val_degradation is None:
        status = "missing_last_val"
    elif train_improvement is None:
        status = "missing_train_metric"
    else:
        overfit_after_best = val_degradation > 1e-8 and train_improvement > 1e-8
        status = "overfit_after_best_val" if overfit_after_best else "no_overfit_after_best_val"

    summary = {
        "run": str(name),
        "metrics_csv": str(metrics_csv),
        "epoch_count": len(epoch_rows),
        "best_monitor": monitor_name,
        "best_val_epoch": best_row["epoch"],
        "best_val_value": best_val_value,
        "train_value_at_best_val": train_at_best,
        "train_val_gap_at_best": _gap(best_val_value, train_at_best),
        "best_val_rmse": best_row.get("val_rmse"),
        "train_rmse_at_best_val": best_row.get("train_rmse"),
        "train_val_rmse_gap": best_row.get("rmse_gap"),
        "best_val_mae": best_row.get("val_mae"),
        "train_mae_at_best_val": best_row.get("train_mae"),
        "train_val_mae_gap": best_row.get("mae_gap"),
        "last_epoch": last_row["epoch"],
        "last_train_rmse": last_row.get("train_rmse"),
        "last_val_rmse": last_row.get("val_rmse"),
        "last_train_mae": last_row.get("train_mae"),
        "last_val_mae": last_row.get("val_mae"),
        "last_train_val_rmse_gap": last_row.get("rmse_gap"),
        "val_degradation_after_best": val_degradation,
        "train_improvement_after_best": train_improvement,
        "overfit_after_best_val": overfit_after_best,
        "status": status,
    }
    return summary, epoch_rows


def _empty_summary(name, metrics_csv):
    return {
        "run": str(name),
        "metrics_csv": str(metrics_csv),
        "epoch_count": 0,
        "best_monitor": "",
        "best_val_epoch": "",
        "best_val_value": None,
        "train_value_at_best_val": None,
        "train_val_gap_at_best": None,
        "best_val_rmse": None,
        "train_rmse_at_best_val": None,
        "train_val_rmse_gap": None,
        "best_val_mae": None,
        "train_mae_at_best_val": None,
        "train_val_mae_gap": None,
        "last_epoch": "",
        "last_train_rmse": None,
        "last_val_rmse": None,
        "last_train_mae": None,
        "last_val_mae": None,
        "last_train_val_rmse_gap": None,
        "val_degradation_after_best": None,
        "train_improvement_after_best": None,
        "overfit_after_best_val": False,
        "status": "insufficient_metrics",
    }


def summarize_training_overfit(run_specs, output_dir):
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")

    summary_rows = []
    curve_rows = []
    for name, metrics_csv in run_specs:
        summary, rows = summarize_training_run(name, metrics_csv)
        summary_rows.append(summary)
        curve_rows.extend(rows)

    summary_rows.sort(key=lambda row: (math.inf if row.get("best_val_value") is None else row["best_val_value"], row["run"]))
    curve_rows.sort(key=lambda row: (row["run"], row["epoch"]))

    summary_path = tables_dir / "training_overfit_summary.csv"
    curves_path = tables_dir / "training_curve_gap_by_run.csv"
    report_path = reports_dir / "training_overfit_report.md"

    write_csv_rows(summary_path, _format_rows(summary_rows, SUMMARY_FIELDS), SUMMARY_FIELDS)
    write_csv_rows(curves_path, _format_rows(curve_rows, CURVE_FIELDS), CURVE_FIELDS)
    write_training_overfit_report(report_path, summary_rows)
    return [summary_path, curves_path, report_path]


def write_training_overfit_report(report_path, summary_rows):
    ensure_dir(Path(report_path).parent)
    lines = [
        "# Training Overfit Summary",
        "",
        "| Run | Monitor | Best Epoch | Best Val | Train At Best | Gap At Best | Last Epoch | Val Degradation | Train Improvement | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['run']} | {row['best_monitor']} | {_format_value(row['best_val_epoch'])} | "
            f"{_format_value(row['best_val_value'])} | {_format_value(row['train_value_at_best_val'])} | "
            f"{_format_value(row['train_val_gap_at_best'])} | {_format_value(row['last_epoch'])} | "
            f"{_format_value(row['val_degradation_after_best'])} | {_format_value(row['train_improvement_after_best'])} | "
            f"{row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Best epoch is selected by `val_RMSE_epoch` when available, then `val_MAE_epoch`, then `val_loss`.",
            "- Gap is `validation - train`; larger positive values indicate a larger generalization gap.",
            "- `overfit_after_best_val` means validation worsened after the best epoch while the matching train metric kept improving.",
            "- This report is diagnostic only and should not be used to tune on test results.",
        ]
    )
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
