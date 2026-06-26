"""RPDF Stage A4: severity imbalance / prediction compression summary.

This module decides whether ``severity-balanced regression`` should be a
mandatory Stage B baseline or a Stage D side-branch.  It joins three existing
P0 summaries:

- ``prediction_run_summary.csv``: overall count / mae / ccc / true_std / pred_std
- ``severity_bias_summary.csv``: per-severity-group count / bias / mae
- ``severity_calibration_run_summary.csv``: post-hoc calibration delta_ccc

and reports:

- severity-bin sample imbalance (ratio of largest to smallest bin),
- prediction compression ratio (pred_std / true_std),
- systematic minimal-overestimation / severe-underestimation bias,
- whether post-hoc calibration closes the gap (it never does, per P0-Evidence),
- a Stage B placement recommendation for severity-balanced regression.

Outputs:

- ``severity_imbalance_summary.csv``: one row per run.
- ``severity_imbalance_report.md``.
"""

import math
from pathlib import Path

from src.diagnostics.io import ensure_dir, read_csv_rows, write_csv_rows


SUMMARY_COLUMNS = [
    "run_name",
    "total_count",
    "bin_count_minimal",
    "bin_count_mild",
    "bin_count_moderate",
    "bin_count_severe",
    "bin_ratio_minimal",
    "bin_ratio_mild",
    "bin_ratio_moderate",
    "bin_ratio_severe",
    "imbalance_ratio",
    "minimal_residual",
    "severe_residual",
    "pred_compression_ratio",
    "true_std",
    "pred_std",
    "ccc",
    "calibration_delta_ccc",
    "severity_balanced_recommendation",
]


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


def _format_scalar(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _load_prediction_summary(csv_path):
    """Load prediction_run_summary.csv keyed by run name."""
    if csv_path is None or not Path(csv_path).exists():
        return {}
    rows = read_csv_rows(csv_path)
    result = {}
    for row in rows:
        key = row.get("run") or row.get("run_name")
        if key:
            result[str(key)] = row
    return result


def _load_severity_bias(csv_path):
    """Load severity_bias_summary.csv into {run_name: {severity_group: row}}."""
    if csv_path is None or not Path(csv_path).exists():
        return {}
    rows = read_csv_rows(csv_path)
    result = {}
    for row in rows:
        run = row.get("run") or row.get("run_name")
        group = row.get("severity_group")
        if run and group:
            result.setdefault(str(run), {})[str(group)] = row
    return result


def _load_calibration_summary(csv_path):
    """Load severity_calibration_run_summary.csv keyed by run name."""
    if csv_path is None or not Path(csv_path).exists():
        return {}
    rows = read_csv_rows(csv_path)
    result = {}
    for row in rows:
        key = row.get("run_name") or row.get("run")
        if key:
            result[str(key)] = row
    return result


def build_severity_imbalance_rows(
    prediction_summary_csv=None,
    severity_bias_csv=None,
    calibration_summary_csv=None,
    runs=None,
):
    """Build per-run severity imbalance summary rows.

    Args:
        prediction_summary_csv: Optional ``prediction_run_summary.csv``.
        severity_bias_csv: Optional ``severity_bias_summary.csv``.
        calibration_summary_csv: Optional ``severity_calibration_run_summary.csv``.
        runs: Optional explicit iterable of run names to include (defaults to
            the union of runs found across the supplied summaries).

    Returns:
        List of per-run summary dicts aligned with ``SUMMARY_COLUMNS``.
    """
    pred_summary = _load_prediction_summary(prediction_summary_csv)
    severity_bias = _load_severity_bias(severity_bias_csv)
    calibration = _load_calibration_summary(calibration_summary_csv)

    if runs is None:
        run_set = set(pred_summary.keys())
        run_set.update(severity_bias.keys())
        run_set.update(calibration.keys())
        runs = sorted(run_set)

    rows = []
    for run_name in runs:
        pred = pred_summary.get(run_name, {})
        bias_by_group = severity_bias.get(run_name, {})
        calib = calibration.get(run_name, {})

        bin_counts = {
            group: _safe_int(bias_by_group.get(group, {}).get("count"))
            for group in ("minimal", "mild", "moderate", "severe")
        }
        bin_residuals = {
            group: _safe_float(bias_by_group.get(group, {}).get("bias"))
            for group in ("minimal", "mild", "moderate", "severe")
        }

        known_counts = [c for c in bin_counts.values() if c is not None and c > 0]
        total_count = sum(known_counts) if known_counts else _safe_int(pred.get("count"))

        bin_ratios = {}
        for group, count in bin_counts.items():
            if total_count and count is not None:
                bin_ratios[group] = count / total_count
            else:
                bin_ratios[group] = None

        imbalance_ratio = None
        if known_counts:
            imbalance_ratio = max(known_counts) / min(known_counts)

        true_std = _safe_float(pred.get("true_std"))
        pred_std = _safe_float(pred.get("pred_std"))
        compression = None
        if pred_std is not None and true_std is not None and true_std > 0:
            compression = pred_std / true_std

        recommendation = _recommend_severity_balanced(
            imbalance_ratio=imbalance_ratio,
            minimal_residual=bin_residuals["minimal"],
            severe_residual=bin_residuals["severe"],
            compression=compression,
            calibration_delta_ccc=_safe_float(calib.get("delta_ccc")),
        )

        rows.append(
            {
                "run_name": run_name,
                "total_count": total_count,
                "bin_count_minimal": bin_counts["minimal"],
                "bin_count_mild": bin_counts["mild"],
                "bin_count_moderate": bin_counts["moderate"],
                "bin_count_severe": bin_counts["severe"],
                "bin_ratio_minimal": bin_ratios["minimal"],
                "bin_ratio_mild": bin_ratios["mild"],
                "bin_ratio_moderate": bin_ratios["moderate"],
                "bin_ratio_severe": bin_ratios["severe"],
                "imbalance_ratio": imbalance_ratio,
                "minimal_residual": bin_residuals["minimal"],
                "severe_residual": bin_residuals["severe"],
                "pred_compression_ratio": compression,
                "true_std": true_std,
                "pred_std": pred_std,
                "ccc": _safe_float(pred.get("ccc")),
                "calibration_delta_ccc": _safe_float(calib.get("delta_ccc")),
                "severity_balanced_recommendation": recommendation,
            }
        )
    return rows


def _recommend_severity_balanced(
    imbalance_ratio,
    minimal_residual,
    severe_residual,
    compression,
    calibration_delta_ccc,
):
    """Decide whether severity-balanced regression is a Stage B baseline or side-branch.

    Heuristics (per docs/SHORTCUT_AUDIT_DESIGN.md section 13.4):

    - Imbalanced bins (imbalance_ratio >= 2.0) AND systematic bias
      (minimal overestimates while severe underestimates) -> Stage B baseline.
    - If post-hoc calibration improved CCC (delta_ccc > 0), the compression is
      partly linear and severity-balanced regression may be a Stage D side-branch.
    - If bins are balanced or bias is not systematic, downgrade to Stage D.
    """
    imbalanced = imbalance_ratio is not None and imbalance_ratio >= 2.0
    systematic_bias = (
        minimal_residual is not None
        and severe_residual is not None
        and minimal_residual > 0
        and severe_residual < 0
    )
    calibration_helps = calibration_delta_ccc is not None and calibration_delta_ccc > 0

    if imbalanced and systematic_bias:
        if calibration_helps:
            return "stage_d_side_branch (calibration partially closes gap; imbalance confirmed)"
        return "stage_b_baseline (imbalance + systematic bias; calibration does not help)"
    if imbalanced or systematic_bias or (compression is not None and compression < 0.8):
        return "stage_d_side_branch (mild imbalance or compression; not strongly systematic)"
    return "stage_d_optional (bins balanced; no systematic severity bias)"


def _write_report(report_path, rows, generated_files):
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    lines = [
        "# Severity Imbalance / Prediction Compression Report (RPDF Stage A4)",
        "",
        "This report decides whether severity-balanced regression should be a "
        "mandatory Stage B baseline or a Stage D side-branch, by joining the "
        "prediction, severity-bias and calibration summaries.",
        "",
        "## Per-Run Summary",
        "",
        "| Run | total | imbalance | minimal_res | severe_res | pred_std/true_std | CCC | calib_delta_CCC | recommendation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_name']} | {row['total_count']} | "
            f"{_format_scalar(row.get('imbalance_ratio'))} | "
            f"{_format_scalar(row.get('minimal_residual'))} | "
            f"{_format_scalar(row.get('severe_residual'))} | "
            f"{_format_scalar(row.get('pred_compression_ratio'))} | "
            f"{_format_scalar(row.get('ccc'))} | "
            f"{_format_scalar(row.get('calibration_delta_ccc'))} | "
            f"{row.get('severity_balanced_recommendation', '')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **imbalance_ratio**: largest severity bin / smallest bin. >= 2.0 means "
            "the majority bin dominates the gradient.",
            "- **minimal_res / severe_res**: mean residual per group. Positive minimal "
            "+ negative severe = systematic central-tendency bias.",
            "- **pred_std/true_std**: prediction range compression. < 1.0 means the "
            "model collapses predictions toward the mean.",
            "- **calib_delta_CCC**: CCC(calibrated) - CCC(original). Positive means "
            "post-hoc linear calibration improved concordance; negative means it only "
            "shifted the mean (per P0-Evidence, calibration is diagnostic, not a fix).",
            "",
            "### Recommendation Logic",
            "",
            "- `stage_b_baseline`: imbalanced bins + systematic minimal/severe bias + "
            "calibration does not help -> severity-balanced regression is a mandatory "
            "Stage B baseline (E2).",
            "- `stage_d_side_branch`: imbalance or compression present but not strongly "
            "systematic, or calibration partially closes the gap -> severity-balanced "
            "regression tested as a Stage D side-branch, not a core baseline.",
            "- `stage_d_optional`: bins balanced, no systematic bias -> severity-balanced "
            "regression not needed for this run.",
            "",
            "## Generated Files",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in generated_files)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_severity_imbalance_audit(
    output_dir,
    prediction_summary_csv=None,
    severity_bias_csv=None,
    calibration_summary_csv=None,
    runs=None,
):
    """Run the A4 severity imbalance audit end-to-end.

    Args:
        output_dir: Directory for outputs.
        prediction_summary_csv: Optional ``prediction_run_summary.csv``.
        severity_bias_csv: Optional ``severity_bias_summary.csv``.
        calibration_summary_csv: Optional ``severity_calibration_run_summary.csv``.
        runs: Optional explicit run names to include.

    Returns:
        List of generated file paths.
    """
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    rows = build_severity_imbalance_rows(
        prediction_summary_csv=prediction_summary_csv,
        severity_bias_csv=severity_bias_csv,
        calibration_summary_csv=calibration_summary_csv,
        runs=runs,
    )

    if not rows:
        raise RuntimeError(
            "No runs found across the supplied summaries. Provide at least one of "
            "prediction_summary_csv / severity_bias_csv / calibration_summary_csv."
        )

    summary_path = tables_dir / "severity_imbalance_summary.csv"
    write_csv_rows(summary_path, rows, SUMMARY_COLUMNS)
    generated.append(summary_path)

    report_path = _write_report(
        reports_dir / "severity_imbalance_report.md",
        rows=rows,
        generated_files=generated,
    )
    generated.append(report_path)
    return generated
