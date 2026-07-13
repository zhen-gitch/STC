"""Read-only Stage C train-thresholded validation group robustness audit."""

import math
from pathlib import Path

import numpy as np

from src.diagnostics.black_artifacts import normalize_video_id
from src.diagnostics.io import ensure_dir, read_csv_rows, severity_group, write_csv_rows
from src.diagnostics.representation_leakage import concordance_ccc


GROUP_COLUMNS = [
    "run_name",
    "grouping",
    "group",
    "threshold",
    "count",
    "num_subjects",
    "coverage",
    "mae",
    "mae_ci_low",
    "mae_ci_high",
    "rmse",
    "bias",
    "ccc",
]

COMPARISON_COLUMNS = [
    "run_name",
    "grouping",
    "worst_group",
    "worst_group_mae",
    "best_group_mae",
    "worst_group_gap",
    "reference_worst_group_mae",
    "reference_worst_group_gap",
    "delta_worst_group_mae",
    "delta_worst_group_gap",
    "group_utility_failure",
]

AXIS_COLUMNS = [
    "axis_name",
    "value_column",
    "status",
    "threshold",
    "train_count",
    "val_count",
    "reason",
]


def load_prediction_records_strict(csv_path):
    """Load a prediction table and reject malformed/non-finite rows."""
    csv_path = Path(csv_path)
    rows = read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"Prediction table is empty or missing: {csv_path}")
    records = []
    for index, row in enumerate(rows, start=2):
        try:
            target = float(row["true_bdi"])
            pred = float(row["pred_bdi"])
            residual = float(row.get("residual", pred - target))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed prediction row {index} in {csv_path}") from exc
        if not all(math.isfinite(value) for value in (target, pred, residual)):
            raise ValueError(f"Non-finite prediction row {index} in {csv_path}")
        video_id = str(row.get("video_id") or row.get("subject_id") or "")
        subject_id = str(row.get("subject_id") or video_id)
        records.append(
            {
                "video_id": video_id,
                "normalized_video_id": normalize_video_id(video_id),
                "subject_id": subject_id,
                "task_name": str(row.get("task_name", "")),
                "true_bdi": target,
                "pred_bdi": pred,
                "residual": residual,
                "abs_error": abs(residual),
                "severity_group": str(row.get("severity_group") or severity_group(target)),
            }
        )
    return records


def load_axis_values(csv_path, value_column, key_column="video_id"):
    """Load one numeric audit axis indexed by normalized video id."""
    rows = read_csv_rows(csv_path)
    values = {}
    for row in rows:
        video_id = row.get(key_column) or row.get("video_id") or ""
        raw = row.get(value_column, "")
        if not video_id or raw in ("", None):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values[normalize_video_id(video_id)] = value
    return values


def resolve_axis_spec(axis_name, train_csv, val_csv, value_column):
    """Freeze an axis threshold from train and load validation values."""
    train_csv = Path(train_csv)
    val_csv = Path(val_csv)
    if not train_csv.exists() or not val_csv.exists():
        return {
            "axis_name": axis_name,
            "value_column": value_column,
            "status": "unavailable",
            "threshold": None,
            "train_count": 0,
            "val_count": 0,
            "reason": "missing_train_or_val_csv",
            "val_values": {},
        }
    train_values = load_axis_values(train_csv, value_column)
    val_values = load_axis_values(val_csv, value_column)
    if not train_values or not val_values:
        return {
            "axis_name": axis_name,
            "value_column": value_column,
            "status": "unavailable",
            "threshold": None,
            "train_count": len(train_values),
            "val_count": len(val_values),
            "reason": "no_numeric_train_or_val_values",
            "val_values": val_values,
        }
    return {
        "axis_name": axis_name,
        "value_column": value_column,
        "status": "available",
        "threshold": float(np.median(list(train_values.values()))),
        "train_count": len(train_values),
        "val_count": len(val_values),
        "reason": "",
        "val_values": val_values,
    }


def _metrics(records):
    if not records:
        return {"mae": None, "rmse": None, "bias": None, "ccc": None}
    targets = np.asarray([row["true_bdi"] for row in records], dtype=float)
    preds = np.asarray([row["pred_bdi"] for row in records], dtype=float)
    residuals = preds - targets
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "bias": float(np.mean(residuals)),
        "ccc": concordance_ccc(targets, preds),
    }


def _subject_bootstrap_mae_ci(records, samples=1000, seed=42):
    if samples <= 0 or not records:
        return None, None
    by_subject = {}
    for record in records:
        by_subject.setdefault(record["subject_id"], []).append(record)
    subjects = sorted(by_subject)
    if len(subjects) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selected = rng.choice(subjects, size=len(subjects), replace=True)
        sample_records = []
        for subject in selected:
            sample_records.extend(by_subject[str(subject)])
        estimates.append(_metrics(sample_records)["mae"])
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def _group_rows(
    run_name,
    grouping,
    records,
    labels,
    threshold=None,
    coverage=1.0,
    bootstrap_samples=1000,
    seed=42,
):
    grouped = {}
    for record, label in zip(records, labels):
        if label is not None and label != "":
            grouped.setdefault(str(label), []).append(record)
    rows = []
    for group_name in sorted(grouped):
        group_records = grouped[group_name]
        metrics = _metrics(group_records)
        ci_low, ci_high = _subject_bootstrap_mae_ci(
            group_records,
            samples=bootstrap_samples,
            seed=seed,
        )
        rows.append(
            {
                "run_name": run_name,
                "grouping": grouping,
                "group": group_name,
                "threshold": threshold,
                "count": len(group_records),
                "num_subjects": len({row["subject_id"] for row in group_records}),
                "coverage": coverage,
                "mae": metrics["mae"],
                "mae_ci_low": ci_low,
                "mae_ci_high": ci_high,
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "ccc": metrics["ccc"],
            }
        )
    return rows


def analyze_run_groups(
    run_name,
    val_records,
    resolved_axes=(),
    bootstrap_samples=1000,
    seed=42,
):
    """Compute severity/task and available train-thresholded axis groups."""
    rows = []
    rows.extend(
        _group_rows(
            run_name,
            "severity",
            val_records,
            [record["severity_group"] for record in val_records],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    )
    rows.extend(
        _group_rows(
            run_name,
            "task",
            val_records,
            [record["task_name"] for record in val_records],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    )
    for axis in resolved_axes:
        if axis["status"] != "available":
            continue
        values = axis["val_values"]
        labels = []
        matched = 0
        for record in val_records:
            value = values.get(record["normalized_video_id"])
            if value is None:
                labels.append(None)
            else:
                matched += 1
                labels.append("low" if value <= axis["threshold"] else "high")
        rows.extend(
            _group_rows(
                run_name,
                axis["axis_name"],
                val_records,
                labels,
                threshold=axis["threshold"],
                coverage=matched / len(val_records) if val_records else 0.0,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )
    return rows


def compare_group_robustness(group_rows, reference_run="C-REF"):
    """Compare worst-group MAE and gap against the reference run."""
    grouped = {}
    for row in group_rows:
        if row.get("mae") is not None:
            grouped.setdefault((row["run_name"], row["grouping"]), []).append(row)
    summaries = {}
    for key, rows in grouped.items():
        worst = max(rows, key=lambda row: row["mae"])
        best = min(rows, key=lambda row: row["mae"])
        summaries[key] = {
            "worst_group": worst["group"],
            "worst_group_mae": worst["mae"],
            "best_group_mae": best["mae"],
            "worst_group_gap": worst["mae"] - best["mae"],
        }

    comparisons = []
    for (run_name, grouping), summary in sorted(summaries.items()):
        reference = summaries.get((reference_run, grouping))
        delta_mae = None
        delta_gap = None
        failure = None
        if reference is not None:
            delta_mae = summary["worst_group_mae"] - reference["worst_group_mae"]
            delta_gap = summary["worst_group_gap"] - reference["worst_group_gap"]
            failure = delta_mae > 0.25 or delta_gap > 0.50
        comparisons.append(
            {
                "run_name": run_name,
                "grouping": grouping,
                **summary,
                "reference_worst_group_mae": (
                    reference["worst_group_mae"] if reference else None
                ),
                "reference_worst_group_gap": (
                    reference["worst_group_gap"] if reference else None
                ),
                "delta_worst_group_mae": delta_mae,
                "delta_worst_group_gap": delta_gap,
                "group_utility_failure": failure,
            }
        )
    return comparisons


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.6f}"
    return value


def _write_rows(path, rows, columns):
    formatted = [
        {column: _format_value(row.get(column)) for column in columns}
        for row in rows
    ]
    write_csv_rows(path, formatted, columns)
    return Path(path)


def _write_report(report_path, group_rows, comparisons, axes, generated):
    lines = [
        "# Stage C Group Robustness Report",
        "",
        "All continuous thresholds are medians estimated from train-only audit data",
        "and then frozen for validation. Missing train axes remain unavailable.",
        "",
        "## Axis Availability",
        "",
    ]
    if axes:
        for axis in axes:
            lines.append(
                f"- {axis['axis_name']}: {axis['status']} threshold="
                f"{_format_value(axis.get('threshold'))} reason={axis.get('reason', '')}"
            )
    else:
        lines.append("- No continuous audit axes supplied; severity/task only.")
    lines.extend(
        [
            "",
            "## Worst-Group Comparison",
            "",
            "| Run | Grouping | Worst group | Worst MAE | Gap | Delta worst MAE | Delta gap | Failure |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['run_name']} | {row['grouping']} | {row['worst_group']} | "
            f"{_format_value(row.get('worst_group_mae'))} | "
            f"{_format_value(row.get('worst_group_gap'))} | "
            f"{_format_value(row.get('delta_worst_group_mae'))} | "
            f"{_format_value(row.get('delta_worst_group_gap'))} | "
            f"{row.get('group_utility_failure')} |"
        )
    lines.extend(["", "## Generated Files", ""])
    lines.extend(f"- `{path}`" for path in generated)
    report_path = Path(report_path)
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_group_robustness_analysis(
    run_specs,
    output_dir,
    axis_specs=(),
    reference_run="C-REF",
    bootstrap_samples=1000,
    seed=42,
):
    """Run the multi-run validation group audit without changing inputs."""
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    axes = [resolve_axis_spec(*spec) for spec in axis_specs]
    group_rows = []
    reference_signatures = None
    for run_name, train_predictions, val_predictions in run_specs:
        train_records = load_prediction_records_strict(train_predictions)
        val_records = load_prediction_records_strict(val_predictions)
        train_subjects = {record["subject_id"] for record in train_records}
        val_subjects = {record["subject_id"] for record in val_records}
        if train_subjects & val_subjects:
            raise ValueError(f"{run_name} train/val subject overlap detected")
        train_videos = {record["normalized_video_id"] for record in train_records}
        val_videos = {record["normalized_video_id"] for record in val_records}
        if train_videos & val_videos:
            raise ValueError(f"{run_name} train/val video overlap detected")
        signatures = (
            {
                record["normalized_video_id"]: (
                    record["subject_id"],
                    record["task_name"],
                    record["true_bdi"],
                )
                for record in train_records
            },
            {
                record["normalized_video_id"]: (
                    record["subject_id"],
                    record["task_name"],
                    record["true_bdi"],
                )
                for record in val_records
            },
        )
        if reference_signatures is None:
            reference_signatures = signatures
        elif signatures != reference_signatures:
            raise ValueError(
                f"{run_name} train/val metadata differs from the reference run"
            )
        group_rows.extend(
            analyze_run_groups(
                run_name,
                val_records,
                resolved_axes=axes,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )
    comparisons = compare_group_robustness(group_rows, reference_run=reference_run)

    generated = []
    generated.append(
        _write_rows(tables_dir / "group_metrics.csv", group_rows, GROUP_COLUMNS)
    )
    generated.append(
        _write_rows(
            tables_dir / "group_comparison.csv",
            comparisons,
            COMPARISON_COLUMNS,
        )
    )
    axis_rows = [{key: value for key, value in axis.items() if key != "val_values"} for axis in axes]
    generated.append(
        _write_rows(tables_dir / "axis_thresholds.csv", axis_rows, AXIS_COLUMNS)
    )
    report = _write_report(
        reports_dir / "group_robustness_report.md",
        group_rows,
        comparisons,
        axes,
        generated,
    )
    generated.append(report)
    return generated
