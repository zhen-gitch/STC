"""Severity calibration verification for MTL-Lite depression predictions.

This module implements a post-hoc linear calibration audit:

1. Fit ``pred_calibrated = a * pred + b`` on validation predictions only.
2. Apply the fixed ``(a, b)`` to test predictions.
3. Compare original and calibrated test metrics, severity group bias, and
   prediction distribution.

The calibration is intentionally simple and diagnostic-only.  It must not use
any test labels during fitting, and it must not be used to retroactively tune
the model after observing the test set.

See ``docs/RGB_OVERFITTING_AUDIT_PLAN.md`` (P0-F) for the research context.
"""

import math
from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, read_prediction_table, write_csv_rows


# Columns written to the val-fit table.
FIT_COLUMNS = [
    "a",
    "b",
    "val_count",
    "val_mae",
    "val_rmse",
    "val_pearson",
    "val_ccc",
    "val_true_mean",
    "val_true_std",
    "val_pred_mean",
    "val_pred_std",
]

# Columns written to the test summary table.
SUMMARY_COLUMNS = [
    "version",
    "count",
    "mae",
    "rmse",
    "pearson",
    "ccc",
    "true_mean",
    "true_std",
    "pred_mean",
    "pred_std",
]

SEVERITY_GROUPS = ["minimal", "mild", "moderate", "severe"]


def _extract_arrays(records):
    """Return (targets, preds, residuals, abs_errors) as numpy arrays."""
    targets = np.asarray([float(row["true_bdi"]) for row in records], dtype=float)
    preds = np.asarray([float(row["pred_bdi"]) for row in records], dtype=float)
    residuals = preds - targets
    abs_errors = np.abs(residuals)
    return targets, preds, residuals, abs_errors


def _ccc_numpy(preds, targets, eps=1e-8):
    """Concordance Correlation Coefficient computed in numpy.

    Args:
        preds: Predicted BDI scores.
        targets: Ground-truth BDI scores.
        eps: Small constant for numerical stability.

    Returns:
        CCC as a float, or ``nan`` if fewer than two finite samples remain.
    """
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    valid = np.isfinite(preds) & np.isfinite(targets)
    preds = preds[valid]
    targets = targets[valid]
    if preds.size < 2:
        return float("nan")

    pred_mean = float(preds.mean())
    target_mean = float(targets.mean())
    pred_var = float(np.mean((preds - pred_mean) ** 2))
    target_var = float(np.mean((targets - target_mean) ** 2))
    covariance = float(np.mean((preds - pred_mean) * (targets - target_mean)))

    denominator = pred_var + target_var + (pred_mean - target_mean) ** 2 + eps
    return float((2.0 * covariance) / denominator)


def fit_linear_calibration(val_records):
    """Fit ``true = a * pred + b`` on validation records.

    Args:
        val_records: List of prediction dicts from the validation split.

    Returns:
        Tuple ``(a, b)`` of slope and intercept.

    Raises:
        ValueError: If there are fewer than two records or zero variance in
            validation predictions.
    """
    if len(val_records) < 2:
        raise ValueError(
            f"Need at least 2 validation records to fit calibration, got {len(val_records)}."
        )

    targets, preds, _, _ = _extract_arrays(val_records)
    if np.std(preds) < 1e-8:
        raise ValueError(
            "Validation predictions have near-zero variance; cannot fit linear calibration."
        )

    # np.polyfit returns coefficients for: true = a * pred + b
    a, b = np.polyfit(preds, targets, 1)
    return float(a), float(b)


def apply_calibration(records, a, b):
    """Apply a fixed linear calibration to a set of prediction records.

    Args:
        records: List of prediction dicts.
        a: Slope.
        b: Intercept.

    Returns:
        New list of dicts with updated ``pred_bdi``, ``residual`` and
        ``abs_error``.  Other fields (``video_id``, ``subject_id``,
        ``task_name``, ``true_bdi``, ``severity_group``) are preserved.
    """
    calibrated = []
    for row in records:
        true_bdi = float(row["true_bdi"])
        calibrated_pred = a * float(row["pred_bdi"]) + b
        residual = calibrated_pred - true_bdi
        calibrated.append(
            {
                "video_id": row.get("video_id", ""),
                "subject_id": row.get("subject_id", ""),
                "task_name": row.get("task_name", ""),
                "true_bdi": true_bdi,
                "pred_bdi": calibrated_pred,
                "residual": residual,
                "abs_error": abs(residual),
                "severity_group": row.get("severity_group", ""),
            }
        )
    return calibrated


def evaluate_metrics(records):
    """Compute overall regression metrics for a set of prediction records.

    Returns:
        Dict with ``count``, ``mae``, ``rmse``, ``pearson``, ``ccc``,
        ``true_mean``, ``true_std``, ``pred_mean``, ``pred_std``.
    """
    if not records:
        return {
            "count": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson": float("nan"),
            "ccc": float("nan"),
            "true_mean": float("nan"),
            "true_std": float("nan"),
            "pred_mean": float("nan"),
            "pred_std": float("nan"),
        }

    targets, preds, residuals, abs_errors = _extract_arrays(records)
    pearson = float("nan")
    if len(records) >= 2 and np.std(targets) > 1e-8 and np.std(preds) > 1e-8:
        pearson = float(np.corrcoef(targets, preds)[0, 1])

    return {
        "count": len(records),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "pearson": pearson,
        "ccc": _ccc_numpy(preds, targets),
        "true_mean": float(np.mean(targets)),
        "true_std": float(np.std(targets)),
        "pred_mean": float(np.mean(preds)),
        "pred_std": float(np.std(preds)),
    }


def severity_group_bias(records):
    """Compute mean residual and MAE per severity group.

    Returns:
        Dict mapping severity group name to ``{"count", "mean_residual", "mae"}``.
        Groups with no samples are omitted.
    """
    bias = {}
    for group in SEVERITY_GROUPS:
        subset = [r for r in records if r.get("severity_group") == group]
        if not subset:
            continue
        residuals = np.asarray([r["residual"] for r in subset], dtype=float)
        abs_errors = np.asarray([r["abs_error"] for r in subset], dtype=float)
        bias[group] = {
            "count": len(subset),
            "mean_residual": float(np.mean(residuals)),
            "mae": float(np.mean(abs_errors)),
        }
    return bias


def _format_scalar(value):
    """Format a scalar for CSV output; empty string for missing values."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_severity_calibration_fit(csv_path, a, b, val_metrics):
    """Write the val-fit parameters and val metrics to CSV."""
    row = {
        "a": _format_scalar(a),
        "b": _format_scalar(b),
        "val_count": val_metrics["count"],
        "val_mae": _format_scalar(val_metrics["mae"]),
        "val_rmse": _format_scalar(val_metrics["rmse"]),
        "val_pearson": _format_scalar(val_metrics["pearson"]),
        "val_ccc": _format_scalar(val_metrics["ccc"]),
        "val_true_mean": _format_scalar(val_metrics["true_mean"]),
        "val_true_std": _format_scalar(val_metrics["true_std"]),
        "val_pred_mean": _format_scalar(val_metrics["pred_mean"]),
        "val_pred_std": _format_scalar(val_metrics["pred_std"]),
    }
    write_csv_rows(csv_path, [row], FIT_COLUMNS)
    return Path(csv_path)


def write_severity_calibration_test_summary(csv_path, original_metrics, calibrated_metrics):
    """Write original vs calibrated test metrics to CSV."""
    rows = [
        {"version": "original", **{k: _format_scalar(original_metrics[k]) for k in SUMMARY_COLUMNS if k != "version"}},
        {"version": "calibrated", **{k: _format_scalar(calibrated_metrics[k]) for k in SUMMARY_COLUMNS if k != "version"}},
    ]
    write_csv_rows(csv_path, rows, SUMMARY_COLUMNS)
    return Path(csv_path)


def _require_matplotlib():
    """Import matplotlib on demand so the module can be used without plotting."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for severity calibration figures.") from exc
    return plt


def plot_calibration_scatter(
    original_records, calibrated_records, save_path, max_score=63
):
    """Render original vs calibrated prediction-target scatter plots side by side."""
    plt = _require_matplotlib()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2), sharex=True, sharey=True)
    versions = [
        ("original", original_records, axes[0]),
        ("calibrated", calibrated_records, axes[1]),
    ]

    for label, records, ax in versions:
        targets = np.asarray([r["true_bdi"] for r in records], dtype=float)
        preds = np.asarray([r["pred_bdi"] for r in records], dtype=float)
        metrics = evaluate_metrics(records)

        ax.scatter(targets, preds, s=46, alpha=0.75, edgecolors="none")
        ax.plot([0, max_score], [0, max_score], linestyle="--", color="black", linewidth=1.1)
        ax.set_xlim(0, max_score)
        ax.set_ylim(0, max_score)
        ax.set_xlabel("True BDI")
        ax.set_ylabel("Predicted BDI")
        ax.set_title(f"{label.capitalize()} Predictions")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.text(
            0.04,
            0.96,
            f"MAE={metrics['mae']:.2f}\nRMSE={metrics['rmse']:.2f}\n"
            f"Pearson={metrics['pearson']:.3f}\nCCC={metrics['ccc']:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.84, edgecolor="none"),
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_severity_group_comparison(
    original_bias,
    calibrated_bias,
    save_path,
    metric="mean_residual",
    ylabel="Mean Residual (Pred - True)",
    title="Mean Residual by Severity Group",
):
    """Render a grouped bar chart comparing original vs calibrated severity bias."""
    plt = _require_matplotlib()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    groups = [g for g in SEVERITY_GROUPS if g in original_bias or g in calibrated_bias]
    x = np.arange(len(groups))
    width = 0.35

    original_values = [original_bias.get(g, {}).get(metric, 0.0) for g in groups]
    calibrated_values = [calibrated_bias.get(g, {}).get(metric, 0.0) for g in groups]

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.bar(x - width / 2, original_values, width, label="original", alpha=0.82)
    ax.bar(x + width / 2, calibrated_values, width, label="calibrated", alpha=0.82)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Severity Group")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def write_severity_calibration_report(
    report_path,
    a,
    b,
    val_metrics,
    original_test_metrics,
    calibrated_test_metrics,
    original_bias,
    calibrated_bias,
    generated_files,
):
    """Write the markdown interpretation report."""
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    def _metrics_table(metrics):
        return (
            f"| Count | {metrics['count']} |\n"
            f"| MAE | {metrics['mae']:.4f} |\n"
            f"| RMSE | {metrics['rmse']:.4f} |\n"
            f"| Pearson | {metrics['pearson']:.4f} |\n"
            f"| CCC | {metrics['ccc']:.4f} |\n"
            f"| True mean / std | {metrics['true_mean']:.2f} / {metrics['true_std']:.2f} |\n"
            f"| Pred mean / std | {metrics['pred_mean']:.2f} / {metrics['pred_std']:.2f} |"
        )

    def _bias_table(bias):
        lines = ["| Severity | Count | Mean Residual | MAE |", "|---|---:|---:|---:|"]
        for group in SEVERITY_GROUPS:
            if group not in bias:
                continue
            entry = bias[group]
            lines.append(
                f"| {group} | {entry['count']} | {entry['mean_residual']:.4f} | {entry['mae']:.4f} |"
            )
        return "\n".join(lines)

    lines = [
        "# Severity Calibration Verification Report",
        "",
        "## Calibration Fit (Validation Only)",
        "",
        f"- Linear mapping: ``pred_calibrated = {a:.6f} * pred + {b:.6f}``",
        f"- Fit on {val_metrics['count']} validation records.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Val MAE | {val_metrics['mae']:.4f} |",
        f"| Val RMSE | {val_metrics['rmse']:.4f} |",
        f"| Val Pearson | {val_metrics['pearson']:.4f} |",
        f"| Val CCC | {val_metrics['ccc']:.4f} |",
        f"| Val pred std | {val_metrics['pred_std']:.2f} |",
        f"| Val true std | {val_metrics['true_std']:.2f} |",
        "",
        "## Test Performance",
        "",
        "### Original",
        "",
        "| Metric | Value |",
        "|---|---:|",
        _metrics_table(original_test_metrics),
        "",
        "### Calibrated",
        "",
        "| Metric | Value |",
        "|---|---:|",
        _metrics_table(calibrated_test_metrics),
        "",
        "## Severity Group Bias",
        "",
        "### Original",
        "",
        _bias_table(original_bias),
        "",
        "### Calibrated",
        "",
        _bias_table(calibrated_bias),
        "",
        "## Interpretation",
        "",
        "- If calibration reduces severe underestimation and minimal overestimation "
        "but Pearson/CCC change little, the model retains severity ranking information "
        "and the failure is mainly prediction compression / scale shrinkage.",
        "- If calibration barely improves severity bias, the model lacks reliable "
        "severity signals in its input or representation; input shortcut and "
        "behavior-audit investigations should continue.",
        "- This audit is diagnostic only.  The fitted ``(a, b)`` must not be used to "
        "retroactively tune the model after observing test results.",
        "",
        "## Generated Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in generated_files)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_severity_calibration_audit(
    val_predictions,
    test_predictions,
    output_dir,
    max_score=63,
):
    """Run the full severity calibration verification pipeline.

    Args:
        val_predictions: Path to ``val_predictions.csv``.
        test_predictions: Path to ``test_predictions.csv``.
        output_dir: Directory for tables, figures and reports.
        max_score: Maximum BDI score used for scatter plot axis limits.

    Returns:
        List of generated file paths.
    """
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    val_records = read_prediction_table(val_predictions)
    test_records = read_prediction_table(test_predictions)
    if not val_records:
        raise ValueError(f"No validation records found in {val_predictions}")
    if not test_records:
        raise ValueError(f"No test records found in {test_predictions}")

    a, b = fit_linear_calibration(val_records)
    calibrated_test_records = apply_calibration(test_records, a, b)

    val_metrics = evaluate_metrics(val_records)
    original_test_metrics = evaluate_metrics(test_records)
    calibrated_test_metrics = evaluate_metrics(calibrated_test_records)
    original_bias = severity_group_bias(test_records)
    calibrated_bias = severity_group_bias(calibrated_test_records)

    fit_path = write_severity_calibration_fit(
        tables_dir / "severity_calibration_fit.csv", a, b, val_metrics
    )
    generated.append(fit_path)

    summary_path = write_severity_calibration_test_summary(
        tables_dir / "severity_calibration_test_summary.csv",
        original_test_metrics,
        calibrated_test_metrics,
    )
    generated.append(summary_path)

    scatter_path = plot_calibration_scatter(
        test_records,
        calibrated_test_records,
        figures_dir / "calibration_scatter.png",
        max_score=max_score,
    )
    generated.append(scatter_path)

    residual_path = plot_severity_group_comparison(
        original_bias,
        calibrated_bias,
        figures_dir / "severity_group_residual.png",
        metric="mean_residual",
        ylabel="Mean Residual (Pred - True)",
        title="Mean Residual by Severity Group",
    )
    generated.append(residual_path)

    mae_path = plot_severity_group_comparison(
        original_bias,
        calibrated_bias,
        figures_dir / "severity_group_mae.png",
        metric="mae",
        ylabel="MAE",
        title="MAE by Severity Group",
    )
    generated.append(mae_path)

    report_path = write_severity_calibration_report(
        reports_dir / "severity_calibration_report.md",
        a=a,
        b=b,
        val_metrics=val_metrics,
        original_test_metrics=original_test_metrics,
        calibrated_test_metrics=calibrated_test_metrics,
        original_bias=original_bias,
        calibrated_bias=calibrated_bias,
        generated_files=generated,
    )
    generated.append(report_path)
    return generated
