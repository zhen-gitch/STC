import math
from pathlib import Path

from src.diagnostics.io import ensure_dir, read_csv_rows, write_csv_rows


GRADIENT_COLUMNS = {
    "bdi_norm": (
        "train_grad_bdi_norm_epoch",
        "train_grad_bdi_norm_step",
        "train_grad_bdi_norm",
    ),
    "identity_reversed_norm": (
        "train_grad_identity_reversed_norm_epoch",
        "train_grad_identity_reversed_norm_step",
        "train_grad_identity_reversed_norm",
    ),
    "identity_to_bdi_ratio": (
        "train_grad_identity_to_bdi_ratio_epoch",
        "train_grad_identity_to_bdi_ratio_step",
        "train_grad_identity_to_bdi_ratio",
    ),
    "cosine": (
        "train_grad_cosine_epoch",
        "train_grad_cosine_step",
        "train_grad_cosine",
    ),
    "conflict_rate": (
        "train_grad_conflict_epoch",
        "train_grad_conflict_step",
        "train_grad_conflict",
    ),
    "cancellation": (
        "train_grad_cancellation_epoch",
        "train_grad_cancellation_step",
        "train_grad_cancellation",
    ),
    "identity_accuracy": (
        "train_identity_accuracy_epoch",
        "train_identity_accuracy_step",
        "train_identity_accuracy",
    ),
}

EPOCH_FIELDS = [
    "run",
    "epoch",
    "bdi_norm",
    "identity_reversed_norm",
    "identity_to_bdi_ratio",
    "cosine",
    "conflict_rate",
    "cancellation",
    "identity_accuracy",
    "val_rmse",
]

SUMMARY_FIELDS = [
    "run",
    "metrics_csv",
    "gradient_epoch_count",
    "mean_bdi_norm",
    "mean_identity_reversed_norm",
    "mean_identity_to_bdi_ratio",
    "mean_cosine",
    "min_cosine",
    "max_cosine",
    "mean_conflict_rate",
    "mean_cancellation",
    "mean_identity_accuracy",
    "corr_cosine_vs_val_rmse",
    "corr_conflict_vs_val_rmse",
    "corr_ratio_vs_val_rmse",
    "status",
]


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


def _mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _pearson(x_values, y_values):
    pairs = [
        (x, y)
        for x, y in zip(x_values, y_values)
        if x is not None and y is not None
    ]
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denominator <= 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / denominator


def _preferred_values(rows, columns):
    for column in columns:
        values = [_safe_float(row.get(column)) for row in rows]
        values = [value for value in values if value is not None]
        if values:
            return values
    return []


def summarize_identity_gradient_run(name, metrics_csv):
    metrics_csv = Path(metrics_csv)
    rows = read_csv_rows(metrics_csv)
    epoch_buckets = {}
    for row in rows:
        epoch_value = _safe_float(row.get("epoch"))
        if epoch_value is None:
            continue
        epoch_buckets.setdefault(int(epoch_value), []).append(row)

    epoch_rows = []
    for epoch in sorted(epoch_buckets):
        bucket = epoch_buckets[epoch]
        item = {"run": str(name), "epoch": epoch}
        for field, columns in GRADIENT_COLUMNS.items():
            item[field] = _mean(_preferred_values(bucket, columns))
        item["val_rmse"] = _mean(
            _preferred_values(bucket, ("val_RMSE_epoch",))
        )
        if any(item[field] is not None for field in GRADIENT_COLUMNS):
            epoch_rows.append(item)

    status = "ok" if epoch_rows else "missing_gradient_metrics"
    summary = {
        "run": str(name),
        "metrics_csv": str(metrics_csv),
        "gradient_epoch_count": len(epoch_rows),
        "mean_bdi_norm": _mean([row["bdi_norm"] for row in epoch_rows]),
        "mean_identity_reversed_norm": _mean(
            [row["identity_reversed_norm"] for row in epoch_rows]
        ),
        "mean_identity_to_bdi_ratio": _mean(
            [row["identity_to_bdi_ratio"] for row in epoch_rows]
        ),
        "mean_cosine": _mean([row["cosine"] for row in epoch_rows]),
        "min_cosine": min(
            (row["cosine"] for row in epoch_rows if row["cosine"] is not None),
            default=None,
        ),
        "max_cosine": max(
            (row["cosine"] for row in epoch_rows if row["cosine"] is not None),
            default=None,
        ),
        "mean_conflict_rate": _mean(
            [row["conflict_rate"] for row in epoch_rows]
        ),
        "mean_cancellation": _mean(
            [row["cancellation"] for row in epoch_rows]
        ),
        "mean_identity_accuracy": _mean(
            [row["identity_accuracy"] for row in epoch_rows]
        ),
        "corr_cosine_vs_val_rmse": _pearson(
            [row["cosine"] for row in epoch_rows],
            [row["val_rmse"] for row in epoch_rows],
        ),
        "corr_conflict_vs_val_rmse": _pearson(
            [row["conflict_rate"] for row in epoch_rows],
            [row["val_rmse"] for row in epoch_rows],
        ),
        "corr_ratio_vs_val_rmse": _pearson(
            [row["identity_to_bdi_ratio"] for row in epoch_rows],
            [row["val_rmse"] for row in epoch_rows],
        ),
        "status": status,
    }
    return summary, epoch_rows


def _format(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def summarize_identity_gradient_audit(run_specs, output_dir):
    output_dir = Path(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    summaries = []
    epoch_rows = []
    for name, metrics_csv in run_specs:
        summary, rows = summarize_identity_gradient_run(name, metrics_csv)
        summaries.append(summary)
        epoch_rows.extend(rows)

    summary_path = tables_dir / "identity_gradient_summary.csv"
    curve_path = tables_dir / "identity_gradient_by_epoch.csv"
    write_csv_rows(
        summary_path,
        [
            {field: _format(row.get(field)) for field in SUMMARY_FIELDS}
            for row in summaries
        ],
        fieldnames=SUMMARY_FIELDS,
    )
    write_csv_rows(
        curve_path,
        [{field: _format(row.get(field)) for field in EPOCH_FIELDS} for row in epoch_rows],
        fieldnames=EPOCH_FIELDS,
    )

    report_lines = [
        "# Identity-Adversarial Gradient Audit",
        "",
        "The identity gradient is measured after GRL reversal at the shared "
        "representation.",
        "Negative cosine means the BDI and reversed-identity gradients conflict.",
        "Conflict/gradient correlations with validation RMSE are descriptive, "
        "not causal.",
        "",
        "| Run | Epochs | ID/BDI norm ratio | Mean cosine | Conflict rate | "
        "Cancellation | Identity accuracy | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        report_lines.append(
            "| {run} | {epochs} | {ratio} | {cosine} | {conflict} | "
            "{cancellation} | {accuracy} | {status} |".format(
                run=row["run"],
                epochs=row["gradient_epoch_count"],
                ratio=_format(row["mean_identity_to_bdi_ratio"]),
                cosine=_format(row["mean_cosine"]),
                conflict=_format(row["mean_conflict_rate"]),
                cancellation=_format(row["mean_cancellation"]),
                accuracy=_format(row["mean_identity_accuracy"]),
                status=row["status"],
            )
        )
    report_path = reports_dir / "identity_gradient_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return [summary_path, curve_path, report_path]
