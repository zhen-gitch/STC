import math
from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, read_prediction_table, write_csv_rows


OVERALL_FIELDS = [
    "run",
    "count",
    "mae",
    "rmse",
    "pearson",
    "ccc",
    "true_mean",
    "true_std",
    "pred_mean",
    "pred_std",
    "residual_mean",
    "abs_error_std",
]
SEVERITY_FIELDS = [
    "run",
    "severity_group",
    "count",
    "true_mean",
    "pred_mean",
    "bias",
    "mae",
    "rmse",
]
TASK_FIELDS = [
    "run",
    "pair_count",
    "task_diff_mean",
    "task_diff_std",
    "task_diff_max",
]
PAIRWISE_FIELDS = [
    "run",
    "baseline_run",
    "matched_count",
    "mean_abs_error_improvement",
    "better_than_baseline_count",
    "worse_than_baseline_count",
    "tie_count",
]


def normalize_video_id(video_id):
    video_id = str(video_id or "")
    if video_id.endswith(".csv"):
        video_id = Path(video_id).stem
    if video_id.endswith("_aligned"):
        video_id = video_id[: -len("_aligned")]
    return video_id


def _task_name(row):
    task_name = str(row.get("task_name") or "")
    if task_name:
        return task_name
    video_id = str(row.get("video_id") or "")
    if "Freeform" in video_id:
        return "Freeform"
    if "Northwind" in video_id:
        return "Northwind"
    return ""


def _format_float(value):
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def _format_row(row, fields):
    formatted = {}
    for key in fields:
        value = row.get(key)
        formatted[key] = _format_float(value) if isinstance(value, float) else str(value)
    return formatted


def _pearson(targets, preds):
    targets = np.asarray(targets, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if targets.size < 2 or np.std(targets) <= 1e-8 or np.std(preds) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(targets, preds)[0, 1])


def _ccc(targets, preds):
    targets = np.asarray(targets, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if targets.size < 2:
        return float("nan")
    mean_true = float(np.mean(targets))
    mean_pred = float(np.mean(preds))
    var_true = float(np.var(targets))
    var_pred = float(np.var(preds))
    covariance = float(np.mean((targets - mean_true) * (preds - mean_pred)))
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator <= 1e-8:
        return float("nan")
    return float((2.0 * covariance) / denominator)


def _metrics(rows):
    if not rows:
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
            "residual_mean": float("nan"),
            "abs_error_std": float("nan"),
        }
    targets = np.asarray([row["true_bdi"] for row in rows], dtype=float)
    preds = np.asarray([row["pred_bdi"] for row in rows], dtype=float)
    residuals = preds - targets
    abs_errors = np.abs(residuals)
    return {
        "count": int(targets.size),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "pearson": _pearson(targets, preds),
        "ccc": _ccc(targets, preds),
        "true_mean": float(np.mean(targets)),
        "true_std": float(np.std(targets, ddof=1)) if targets.size > 1 else 0.0,
        "pred_mean": float(np.mean(preds)),
        "pred_std": float(np.std(preds, ddof=1)) if preds.size > 1 else 0.0,
        "residual_mean": float(np.mean(residuals)),
        "abs_error_std": float(np.std(abs_errors, ddof=1)) if abs_errors.size > 1 else 0.0,
    }


def load_prediction_run(name, csv_path):
    rows = []
    for row in read_prediction_table(csv_path):
        item = dict(row)
        item["run"] = str(name)
        item["video_key"] = normalize_video_id(item.get("video_id"))
        item["task_name"] = _task_name(item)
        rows.append(item)
    return rows


def summarize_prediction_runs(run_specs, output_dir, baseline_name=None):
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    runs = {name: load_prediction_run(name, csv_path) for name, csv_path in run_specs}

    overall_rows = _overall_rows(runs)
    severity_rows = _severity_rows(runs)
    task_rows = _task_consistency_rows(runs)
    pairwise_rows = _pairwise_rows(runs, baseline_name=baseline_name) if baseline_name else []

    overall_path = tables_dir / "prediction_run_summary.csv"
    severity_path = tables_dir / "severity_bias_summary.csv"
    task_path = tables_dir / "task_consistency_summary.csv"
    pairwise_path = tables_dir / "pairwise_baseline_improvement.csv"
    report_path = reports_dir / "prediction_runs_report.md"

    write_csv_rows(overall_path, [_format_row(row, OVERALL_FIELDS) for row in overall_rows], OVERALL_FIELDS)
    write_csv_rows(severity_path, [_format_row(row, SEVERITY_FIELDS) for row in severity_rows], SEVERITY_FIELDS)
    write_csv_rows(task_path, [_format_row(row, TASK_FIELDS) for row in task_rows], TASK_FIELDS)
    write_csv_rows(pairwise_path, [_format_row(row, PAIRWISE_FIELDS) for row in pairwise_rows], PAIRWISE_FIELDS)
    _write_report(report_path, overall_rows, severity_rows, task_rows, pairwise_rows, baseline_name)
    return [overall_path, severity_path, task_path, pairwise_path, report_path]


def _overall_rows(runs):
    rows = []
    for name, run_rows in runs.items():
        metrics = _metrics(run_rows)
        rows.append({"run": name, **metrics})
    rows.sort(key=lambda row: (math.inf if not math.isfinite(row["mae"]) else row["mae"]))
    return rows


def _severity_rows(runs):
    rows = []
    for name, run_rows in runs.items():
        groups = {}
        for row in run_rows:
            groups.setdefault(row.get("severity_group") or "unknown", []).append(row)
        for severity in sorted(groups):
            metrics = _metrics(groups[severity])
            rows.append(
                {
                    "run": name,
                    "severity_group": severity,
                    "count": metrics["count"],
                    "true_mean": metrics["true_mean"],
                    "pred_mean": metrics["pred_mean"],
                    "bias": metrics["residual_mean"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                }
            )
    return rows


def _task_consistency_rows(runs):
    rows = []
    for name, run_rows in runs.items():
        by_subject = {}
        for row in run_rows:
            by_subject.setdefault(row.get("subject_id"), []).append(row)
        diffs = []
        for subject_rows in by_subject.values():
            freeform = [row for row in subject_rows if row.get("task_name") == "Freeform"]
            northwind = [row for row in subject_rows if row.get("task_name") == "Northwind"]
            if freeform and northwind:
                diffs.append(abs(float(freeform[0]["pred_bdi"]) - float(northwind[0]["pred_bdi"])))
        diffs_array = np.asarray(diffs, dtype=float)
        rows.append(
            {
                "run": name,
                "pair_count": int(diffs_array.size),
                "task_diff_mean": float(np.mean(diffs_array)) if diffs_array.size else float("nan"),
                "task_diff_std": float(np.std(diffs_array, ddof=1)) if diffs_array.size > 1 else 0.0,
                "task_diff_max": float(np.max(diffs_array)) if diffs_array.size else float("nan"),
            }
        )
    return rows


def _pairwise_rows(runs, baseline_name):
    if baseline_name not in runs:
        raise ValueError(f"Baseline run '{baseline_name}' was not provided.")
    baseline_by_video = {row["video_key"]: row for row in runs[baseline_name]}
    rows = []
    for name, run_rows in runs.items():
        if name == baseline_name:
            continue
        improvements = []
        better = worse = tie = 0
        for row in run_rows:
            baseline = baseline_by_video.get(row["video_key"])
            if baseline is None:
                continue
            improvement = float(baseline["abs_error"]) - float(row["abs_error"])
            improvements.append(improvement)
            if math.isclose(improvement, 0.0, abs_tol=1e-8):
                tie += 1
            elif improvement > 0:
                better += 1
            else:
                worse += 1
        rows.append(
            {
                "run": name,
                "baseline_run": baseline_name,
                "matched_count": len(improvements),
                "mean_abs_error_improvement": float(np.mean(improvements)) if improvements else float("nan"),
                "better_than_baseline_count": better,
                "worse_than_baseline_count": worse,
                "tie_count": tie,
            }
        )
    rows.sort(key=lambda row: row["mean_abs_error_improvement"], reverse=True)
    return rows


def _write_report(report_path, overall_rows, severity_rows, task_rows, pairwise_rows, baseline_name):
    ensure_dir(Path(report_path).parent)
    lines = [
        "# Prediction Runs Summary",
        "",
        "## Overall",
        "",
        "| Run | Count | MAE | RMSE | Pearson | CCC | True Std | Pred Std | Residual Mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['run']} | {row['count']} | {row['mae']:.4f} | {row['rmse']:.4f} | "
            f"{row['pearson']:.4f} | {row['ccc']:.4f} | {row['true_std']:.4f} | "
            f"{row['pred_std']:.4f} | {row['residual_mean']:.4f} |"
        )

    lines.extend(["", "## Severity Bias", ""])
    lines.append("| Run | Severity | Count | True Mean | Pred Mean | Bias | MAE |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in severity_rows:
        lines.append(
            f"| {row['run']} | {row['severity_group']} | {row['count']} | "
            f"{row['true_mean']:.4f} | {row['pred_mean']:.4f} | {row['bias']:.4f} | {row['mae']:.4f} |"
        )

    lines.extend(["", "## Task Consistency", ""])
    lines.append("| Run | Pair Count | Mean Diff | Max Diff |")
    lines.append("|---|---:|---:|---:|")
    for row in task_rows:
        lines.append(f"| {row['run']} | {row['pair_count']} | {row['task_diff_mean']:.4f} | {row['task_diff_max']:.4f} |")

    if baseline_name:
        lines.extend(["", f"## Pairwise Improvement vs `{baseline_name}`", ""])
        lines.append("| Run | Matched | Mean Abs Error Improvement | Better | Worse | Tie |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in pairwise_rows:
            lines.append(
                f"| {row['run']} | {row['matched_count']} | {row['mean_abs_error_improvement']:.4f} | "
                f"{row['better_than_baseline_count']} | {row['worse_than_baseline_count']} | {row['tie_count']} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Positive residual/bias means overprediction.",
            "- Pairwise improvement is `baseline_abs_error - run_abs_error`; positive values mean the run improved over the baseline.",
        ]
    )
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
