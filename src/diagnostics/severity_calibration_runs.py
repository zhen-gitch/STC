"""Multi-run summary for severity calibration audits.

This module aggregates per-run severity calibration outputs into a unified
comparison table.  It is the multi-run counterpart of
``src/diagnostics/severity_calibration.py`` and produces the outputs required by
``docs/RGB_OVERFITTING_AUDIT_PLAN.md`` (P0-F follow-up).

Outputs:

- ``severity_calibration_run_summary.csv``: one row per run with calibration
  fit parameters, val metrics, and original/calibrated test metrics.
- ``severity_calibration_group_bias_summary.csv``: per-run severity group bias
  for original and calibrated predictions.
- ``severity_calibration_runs_report.md``: a markdown comparison report.
"""

import math
from pathlib import Path

from src.diagnostics.io import ensure_dir, read_csv_rows, write_csv_rows


RUN_SUMMARY_COLUMNS = [
    "run_name",
    "a",
    "b",
    "val_count",
    "val_mae",
    "val_rmse",
    "val_pearson",
    "val_ccc",
    "val_true_std",
    "val_pred_std",
    "test_original_count",
    "test_original_mae",
    "test_original_rmse",
    "test_original_pearson",
    "test_original_ccc",
    "test_original_true_std",
    "test_original_pred_std",
    "test_calibrated_count",
    "test_calibrated_mae",
    "test_calibrated_rmse",
    "test_calibrated_pearson",
    "test_calibrated_ccc",
    "test_calibrated_true_std",
    "test_calibrated_pred_std",
    "delta_mae",
    "delta_rmse",
    "delta_pearson",
    "delta_ccc",
]

GROUP_BIAS_SUMMARY_COLUMNS = [
    "run_name",
    "version",
    "severity_group",
    "count",
    "mean_residual",
    "mae",
]


def _format_scalar(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_int(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _resolve_fit_csv(path):
    path = Path(path)
    if path.is_file() and path.name.endswith("_fit.csv"):
        return path
    candidate = path / "tables" / "severity_calibration_fit.csv"
    if candidate.exists():
        return candidate
    candidate = path / "severity_calibration_fit.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find severity_calibration_fit.csv under {path}")


def _resolve_test_summary_csv(path):
    path = Path(path)
    if path.is_file() and path.name.endswith("_summary.csv"):
        return path
    candidate = path / "tables" / "severity_calibration_test_summary.csv"
    if candidate.exists():
        return candidate
    candidate = path / "severity_calibration_test_summary.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find severity_calibration_test_summary.csv under {path}")


def _resolve_group_bias_csv(path):
    path = Path(path)
    if path.is_file():
        candidate = path.parent / "severity_calibration_group_bias.csv"
        if candidate.exists():
            return candidate
        return None
    candidate = path / "tables" / "severity_calibration_group_bias.csv"
    if candidate.exists():
        return candidate
    candidate = path / "severity_calibration_group_bias.csv"
    if candidate.exists():
        return candidate
    return None


def _read_single_row_csv(csv_path):
    rows = read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    return rows[0]


def _read_version_rows(csv_path):
    rows = read_csv_rows(csv_path)
    if len(rows) < 2:
        raise ValueError(f"Expected at least two rows in summary CSV: {csv_path}")
    return {row["version"]: row for row in rows}


def load_severity_calibration_fit(csv_path):
    row = _read_single_row_csv(csv_path)
    result = {}
    for key, value in row.items():
        if key == "val_count":
            result[key] = _safe_int(value)
        else:
            result[key] = _safe_float(value)
    return result


def load_severity_calibration_test_summary(csv_path):
    versions = _read_version_rows(csv_path)
    result = {}
    for version in ("original", "calibrated"):
        row = versions.get(version)
        if row is None:
            raise ValueError(f"Missing '{version}' row in {csv_path}")
        parsed = {}
        for key, value in row.items():
            if key == "count":
                parsed[key] = _safe_int(value)
            elif key == "version":
                parsed[key] = value
            else:
                parsed[key] = _safe_float(value)
        result[version] = parsed
    return result


def load_severity_calibration_group_bias(csv_path):
    rows = read_csv_rows(csv_path)
    records = []
    for row in rows:
        records.append(
            {
                "version": row.get("version", ""),
                "severity_group": row.get("severity_group", ""),
                "count": _safe_int(row.get("count", "")),
                "mean_residual": _safe_float(row.get("mean_residual", "")),
                "mae": _safe_float(row.get("mae", "")),
            }
        )
    return records


def build_run_summary_row(run_name, fit, test_summary):
    original = test_summary["original"]
    calibrated = test_summary["calibrated"]
    return {
        "run_name": run_name,
        "a": fit.get("a"),
        "b": fit.get("b"),
        "val_count": fit.get("val_count"),
        "val_mae": fit.get("val_mae"),
        "val_rmse": fit.get("val_rmse"),
        "val_pearson": fit.get("val_pearson"),
        "val_ccc": fit.get("val_ccc"),
        "val_true_std": fit.get("val_true_std"),
        "val_pred_std": fit.get("val_pred_std"),
        "test_original_count": original.get("count"),
        "test_original_mae": original.get("mae"),
        "test_original_rmse": original.get("rmse"),
        "test_original_pearson": original.get("pearson"),
        "test_original_ccc": original.get("ccc"),
        "test_original_true_std": original.get("true_std"),
        "test_original_pred_std": original.get("pred_std"),
        "test_calibrated_count": calibrated.get("count"),
        "test_calibrated_mae": calibrated.get("mae"),
        "test_calibrated_rmse": calibrated.get("rmse"),
        "test_calibrated_pearson": calibrated.get("pearson"),
        "test_calibrated_ccc": calibrated.get("ccc"),
        "test_calibrated_true_std": calibrated.get("true_std"),
        "test_calibrated_pred_std": calibrated.get("pred_std"),
        "delta_mae": _delta(calibrated.get("mae"), original.get("mae")),
        "delta_rmse": _delta(calibrated.get("rmse"), original.get("rmse")),
        "delta_pearson": _delta(calibrated.get("pearson"), original.get("pearson")),
        "delta_ccc": _delta(calibrated.get("ccc"), original.get("ccc")),
    }


def _delta(after, before):
    if after is None or before is None:
        return None
    return after - before


def write_severity_calibration_run_summary(csv_path, rows):
    """Write the multi-run severity calibration summary CSV."""
    formatted = [
        {col: _format_scalar(row.get(col)) for col in RUN_SUMMARY_COLUMNS}
        for row in rows
    ]
    write_csv_rows(csv_path, formatted, RUN_SUMMARY_COLUMNS)
    return Path(csv_path)


def write_severity_calibration_group_bias_summary(csv_path, rows):
    """Write the multi-run severity group bias summary CSV."""
    formatted = [
        {col: _format_scalar(row.get(col)) for col in GROUP_BIAS_SUMMARY_COLUMNS}
        for row in rows
    ]
    write_csv_rows(csv_path, formatted, GROUP_BIAS_SUMMARY_COLUMNS)
    return Path(csv_path)


def write_severity_calibration_runs_report(report_path, run_rows, group_bias_rows, generated_files):
    """Write the markdown multi-run severity calibration report."""
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    lines = [
        "# Severity Calibration Multi-Run Summary Report",
        "",
        "This report compares post-hoc linear calibration across multiple runs.",
        "",
        "## Run-Level Summary",
        "",
        "| Run | a | b | val_MAE | val_Pearson | orig_MAE | cal_MAE | delta_MAE | orig_CCC | cal_CCC | delta_CCC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append(
            f"| {row['run_name']} | "
            f"{_format_scalar(row.get('a'))} | "
            f"{_format_scalar(row.get('b'))} | "
            f"{_format_scalar(row.get('val_mae'))} | "
            f"{_format_scalar(row.get('val_pearson'))} | "
            f"{_format_scalar(row.get('test_original_mae'))} | "
            f"{_format_scalar(row.get('test_calibrated_mae'))} | "
            f"{_format_scalar(row.get('delta_mae'))} | "
            f"{_format_scalar(row.get('test_original_ccc'))} | "
            f"{_format_scalar(row.get('test_calibrated_ccc'))} | "
            f"{_format_scalar(row.get('delta_ccc'))} |"
        )

    if group_bias_rows:
        lines.extend(
            [
                "",
                "## Severity Group Bias Summary",
                "",
                "| Run | Version | Group | Count | Mean Residual | MAE |",
                "|---|---|---|---|---:|---:|",
            ]
        )
        for row in group_bias_rows:
            lines.append(
                f"| {row['run_name']} | {row['version']} | {row['severity_group']} | "
                f"{row.get('count', '')} | "
                f"{_format_scalar(row.get('mean_residual'))} | "
                f"{_format_scalar(row.get('mae'))} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- ``delta_MAE < 0`` means calibration reduced average error.",
            "- ``delta_CCC > 0`` means calibration improved concordance; ``delta_CCC < 0`` "
            "means calibration merely shifted the prediction mean without improving ranking.",
            "- A run where calibration improves minimal/mild bias but leaves severe bias "
            "unchanged is still subject to prediction compression at the high-severity end.",
            "",
            "## Generated Files",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in generated_files)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def summarize_severity_calibration_runs(run_specs, output_dir):
    """Aggregate severity calibration audits across multiple runs.

    Args:
        run_specs: List of ``(run_name, path)`` tuples.  ``path`` may be the
            output directory of ``scripts/audit_severity_calibration.py`` or the
            fit CSV itself.
        output_dir: Directory for summary outputs.

    Returns:
        List of generated file paths.
    """
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    run_rows = []
    group_bias_rows = []

    for run_name, path in run_specs:
        fit = load_severity_calibration_fit(_resolve_fit_csv(path))
        test_summary = load_severity_calibration_test_summary(_resolve_test_summary_csv(path))
        run_rows.append(build_run_summary_row(run_name, fit, test_summary))

        group_bias_csv = _resolve_group_bias_csv(path)
        if group_bias_csv is not None:
            for record in load_severity_calibration_group_bias(group_bias_csv):
                group_bias_rows.append(
                    {
                        "run_name": run_name,
                        "version": record["version"],
                        "severity_group": record["severity_group"],
                        "count": record["count"],
                        "mean_residual": record["mean_residual"],
                        "mae": record["mae"],
                    }
                )

    summary_path = write_severity_calibration_run_summary(
        tables_dir / "severity_calibration_run_summary.csv", run_rows
    )
    generated.append(summary_path)

    if group_bias_rows:
        bias_path = write_severity_calibration_group_bias_summary(
            tables_dir / "severity_calibration_group_bias_summary.csv", group_bias_rows
        )
        generated.append(bias_path)

    report_path = write_severity_calibration_runs_report(
        reports_dir / "severity_calibration_runs_report.md",
        run_rows=run_rows,
        group_bias_rows=group_bias_rows,
        generated_files=generated,
    )
    generated.append(report_path)
    return generated
