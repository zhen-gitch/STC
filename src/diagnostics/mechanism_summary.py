"""Merge prediction, identity retrieval, and severity calibration summaries.

This module builds the unified mechanism table required by
``docs/RGB_OVERFITTING_AUDIT_PLAN.md`` (P0-G.1).  It joins:

- ``prediction_run_summary.csv`` (overall metrics + distribution)
- ``severity_bias_summary.csv`` (minimal / severe residuals)
- ``task_consistency_summary.csv`` (Freeform/Northwind prediction diff)
- ``identity_retrieval_run_summary.csv`` (same-subject retrieval)
- ``severity_calibration_run_summary.csv`` (calibration deltas)

into one table so that the relationships between prediction compression,
severity bias, task inconsistency, identity shortcut, and calibration can be
read in a single view.

Outputs:

- ``mechanism_summary.csv``
- ``mechanism_report.md``
"""

import math
from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, read_csv_rows, write_csv_rows


MECHANISM_COLUMNS = [
    "run_name",
    "count",
    "mae",
    "rmse",
    "pearson",
    "ccc",
    "true_std",
    "pred_std",
    "pred_compression_ratio",
    "minimal_residual_original",
    "minimal_residual_calibrated",
    "severe_residual_original",
    "severe_residual_calibrated",
    "task_diff_mean",
    "same_subject_top1_rate",
    "same_subject_top5_rate",
    "severity_neighbor_agreement_top5_mean",
    "task_neighbor_agreement_top5_mean",
    "calibration_delta_mae",
    "calibration_delta_rmse",
    "calibration_delta_ccc",
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


def _read_csv_dict(csv_path, key_column="run_name"):
    """Read a CSV and return a dict keyed by ``key_column``."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}
    rows = read_csv_rows(csv_path)
    result = {}
    for row in rows:
        key = row.get(key_column)
        if not key:
            continue
        result[key] = row
    return result


def _index_by_run_name(csv_path):
    return _read_csv_dict(csv_path, key_column="run_name")


def _load_prediction_summary(csv_path):
    rows = _index_by_run_name(csv_path)
    result = {}
    for run_name, row in rows.items():
        result[run_name] = {
            "count": _safe_int(row.get("count")),
            "mae": _safe_float(row.get("mae")),
            "rmse": _safe_float(row.get("rmse")),
            "pearson": _safe_float(row.get("pearson")),
            "ccc": _safe_float(row.get("ccc")),
            "true_std": _safe_float(row.get("true_std")),
            "pred_std": _safe_float(row.get("pred_std")),
        }
    return result


def _load_severity_bias_summary(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}
    rows = read_csv_rows(csv_path)
    result = {}
    for row in rows:
        run_name = row.get("run_name")
        group = row.get("severity_group")
        if not run_name or not group:
            continue
        mean_residual = _safe_float(row.get("mean_residual"))
        result.setdefault(run_name, {})[group] = mean_residual
    return result


def _load_task_consistency_summary(csv_path):
    rows = _read_csv_dict(csv_path, key_column="run_name")
    result = {}
    for run_name, row in rows.items():
        diff = _safe_float(row.get("mean_abs_pred_diff"))
        if diff is None:
            continue
        result.setdefault(run_name, []).append(diff)
    return {run: float(np.mean(diffs)) for run, diffs in result.items()}


def _load_identity_summary(csv_path):
    rows = _index_by_run_name(csv_path)
    result = {}
    for run_name, row in rows.items():
        result[run_name] = {
            "same_subject_top1_rate": _safe_float(row.get("same_subject_top1_rate")),
            "same_subject_top5_rate": _safe_float(row.get("same_subject_top5_rate")),
            "severity_neighbor_agreement_top5_mean": _safe_float(row.get("severity_neighbor_agreement_top5_mean")),
            "task_neighbor_agreement_top5_mean": _safe_float(row.get("task_neighbor_agreement_top5_mean")),
        }
    return result


def _load_calibration_summary(csv_path):
    rows = _index_by_run_name(csv_path)
    result = {}
    for run_name, row in rows.items():
        result[run_name] = {
            "calibration_delta_mae": _safe_float(row.get("delta_mae")),
            "calibration_delta_rmse": _safe_float(row.get("delta_rmse")),
            "calibration_delta_ccc": _safe_float(row.get("delta_ccc")),
        }
    return result


def _compression_ratio(pred_std, true_std):
    if pred_std is None or true_std is None or true_std == 0:
        return None
    return pred_std / true_std


def build_mechanism_rows(
    prediction_summary,
    severity_bias_summary,
    task_consistency_summary,
    identity_summary,
    calibration_summary,
):
    """Build the unified mechanism table rows."""
    all_runs = set(prediction_summary.keys())
    all_runs.update(severity_bias_summary.keys())
    all_runs.update(task_consistency_summary.keys())
    all_runs.update(identity_summary.keys())
    all_runs.update(calibration_summary.keys())

    rows = []
    for run_name in sorted(all_runs):
        pred = prediction_summary.get(run_name, {})
        bias = severity_bias_summary.get(run_name, {})
        task_diff = task_consistency_summary.get(run_name)
        identity = identity_summary.get(run_name, {})
        calibration = calibration_summary.get(run_name, {})

        true_std = pred.get("true_std")
        pred_std = pred.get("pred_std")

        rows.append(
            {
                "run_name": run_name,
                "count": pred.get("count"),
                "mae": pred.get("mae"),
                "rmse": pred.get("rmse"),
                "pearson": pred.get("pearson"),
                "ccc": pred.get("ccc"),
                "true_std": true_std,
                "pred_std": pred_std,
                "pred_compression_ratio": _compression_ratio(pred_std, true_std),
                "minimal_residual_original": bias.get("minimal"),
                "minimal_residual_calibrated": None,
                "severe_residual_original": bias.get("severe"),
                "severe_residual_calibrated": None,
                "task_diff_mean": task_diff,
                "same_subject_top1_rate": identity.get("same_subject_top1_rate"),
                "same_subject_top5_rate": identity.get("same_subject_top5_rate"),
                "severity_neighbor_agreement_top5_mean": identity.get("severity_neighbor_agreement_top5_mean"),
                "task_neighbor_agreement_top5_mean": identity.get("task_neighbor_agreement_top5_mean"),
                "calibration_delta_mae": calibration.get("calibration_delta_mae"),
                "calibration_delta_rmse": calibration.get("calibration_delta_rmse"),
                "calibration_delta_ccc": calibration.get("calibration_delta_ccc"),
            }
        )
    return rows


def write_mechanism_summary(csv_path, rows):
    """Write the unified mechanism summary CSV."""
    formatted = [
        {col: _format_scalar(row.get(col)) for col in MECHANISM_COLUMNS}
        for row in rows
    ]
    write_csv_rows(csv_path, formatted, MECHANISM_COLUMNS)
    return Path(csv_path)


def write_mechanism_report(report_path, rows, generated_files):
    """Write the markdown mechanism report."""
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    lines = [
        "# RGB Overfitting Mechanism Summary Report",
        "",
        "This table merges prediction metrics, severity bias, task inconsistency, "
        "embedding identity retrieval, and severity calibration into a single view.",
        "",
        "## Mechanism Table",
        "",
        "| Run | MAE | CCC | pred_std/true_std | minimal_res | severe_res | task_diff | same_top1 | same_top5 | delta_CCC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | "
            f"{_format_scalar(row.get('mae'))} | "
            f"{_format_scalar(row.get('ccc'))} | "
            f"{_format_scalar(row.get('pred_compression_ratio'))} | "
            f"{_format_scalar(row.get('minimal_residual_original'))} | "
            f"{_format_scalar(row.get('severe_residual_original'))} | "
            f"{_format_scalar(row.get('task_diff_mean'))} | "
            f"{_format_scalar(row.get('same_subject_top1_rate'))} | "
            f"{_format_scalar(row.get('same_subject_top5_rate'))} | "
            f"{_format_scalar(row.get('calibration_delta_ccc'))} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- **pred_compression_ratio**: ``pred_std / true_std``.  Values well below 1.0 "
            "indicate prediction range compression.",
            "- **minimal_res / severe_res**: Mean residual (pred - true) for minimal and severe "
            "groups.  Positive minimal + negative severe = central-tendency bias.",
            "- **task_diff**: Mean absolute Freeform/Northwind prediction difference per subject. "
            "Large values suggest task-context confound.",
            "- **same_top1 / same_top5**: Same-subject embedding retrieval rate.  High values "
            "suggest identity/static-appearance shortcut.",
            "- **delta_CCC**: CCC(calibrated) - CCC(original).  Positive means post-hoc calibration "
            "improved concordance; negative means calibration only shifted the mean.",
            "- A run that improves MAE while pred_compression_ratio stays low, same_top1 stays high, "
            "and task_diff increases should be interpreted as a bias trade-off rather than clean "
            "generalization improvement.",
            "",
            "## Generated Files",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in generated_files)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def build_mechanism_summary(
    prediction_summary_csv,
    output_dir,
    severity_bias_summary_csv=None,
    task_consistency_summary_csv=None,
    identity_summary_csv=None,
    calibration_summary_csv=None,
):
    """Build the unified mechanism summary from multiple diagnostic summaries.

    Args:
        prediction_summary_csv: Path to prediction_run_summary.csv.
        output_dir: Directory for outputs.
        severity_bias_summary_csv: Optional severity_bias_summary.csv.
        task_consistency_summary_csv: Optional task_consistency_summary.csv.
        identity_summary_csv: Optional identity_retrieval_run_summary.csv.
        calibration_summary_csv: Optional severity_calibration_run_summary.csv.

    Returns:
        List of generated file paths.
    """
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    prediction_summary = _load_prediction_summary(prediction_summary_csv)
    severity_bias_summary = _load_severity_bias_summary(severity_bias_summary_csv) if severity_bias_summary_csv else {}
    task_consistency_summary = _load_task_consistency_summary(task_consistency_summary_csv) if task_consistency_summary_csv else {}
    identity_summary = _load_identity_summary(identity_summary_csv) if identity_summary_csv else {}
    calibration_summary = _load_calibration_summary(calibration_summary_csv) if calibration_summary_csv else {}

    rows = build_mechanism_rows(
        prediction_summary,
        severity_bias_summary,
        task_consistency_summary,
        identity_summary,
        calibration_summary,
    )

    summary_path = write_mechanism_summary(
        tables_dir / "mechanism_summary.csv", rows
    )
    generated.append(summary_path)

    report_path = write_mechanism_report(
        reports_dir / "mechanism_report.md",
        rows=rows,
        generated_files=generated,
    )
    generated.append(report_path)
    return generated
