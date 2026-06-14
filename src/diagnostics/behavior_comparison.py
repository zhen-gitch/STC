import math
from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, read_prediction_table, write_csv_rows


COMPARISON_FIELDS = [
    "video_id",
    "subject_id",
    "task_name",
    "true_bdi",
    "rgb_pred_bdi",
    "behavior_pred_bdi",
    "rgb_residual",
    "behavior_residual",
    "rgb_abs_error",
    "behavior_abs_error",
    "error_delta_behavior_minus_rgb",
    "better_model",
    "severity_group",
]

SUMMARY_FIELDS = [
    "group",
    "count",
    "rgb_mae",
    "behavior_mae",
    "rgb_rmse",
    "behavior_rmse",
    "rgb_pearson",
    "behavior_pearson",
    "rgb_ccc",
    "behavior_ccc",
    "behavior_better_count",
    "rgb_better_count",
    "tie_count",
]


def _normalize_video_id(video_id):
    video_id = str(video_id or "")
    if video_id.endswith(".csv"):
        video_id = Path(video_id).stem
    for suffix in ("_aligned",):
        if video_id.endswith(suffix):
            video_id = video_id[: -len(suffix)]
    return video_id


def _prediction_key(row):
    video_id = _normalize_video_id(row.get("video_id"))
    if video_id:
        return ("video", video_id)
    return ("subject", str(row.get("subject_id", "")))


def _format_float(value):
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def _pearson(targets, preds):
    targets = np.asarray(targets, dtype=float)
    preds = np.asarray(preds, dtype=float)
    if targets.size < 2 or np.std(targets) < 1e-8 or np.std(preds) < 1e-8:
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
            "rgb_mae": float("nan"),
            "behavior_mae": float("nan"),
            "rgb_rmse": float("nan"),
            "behavior_rmse": float("nan"),
            "rgb_pearson": float("nan"),
            "behavior_pearson": float("nan"),
            "rgb_ccc": float("nan"),
            "behavior_ccc": float("nan"),
            "behavior_better_count": 0,
            "rgb_better_count": 0,
            "tie_count": 0,
        }

    targets = np.asarray([row["true_bdi"] for row in rows], dtype=float)
    rgb_preds = np.asarray([row["rgb_pred_bdi"] for row in rows], dtype=float)
    behavior_preds = np.asarray([row["behavior_pred_bdi"] for row in rows], dtype=float)
    rgb_errors = np.abs(rgb_preds - targets)
    behavior_errors = np.abs(behavior_preds - targets)
    return {
        "count": int(targets.size),
        "rgb_mae": float(np.mean(rgb_errors)),
        "behavior_mae": float(np.mean(behavior_errors)),
        "rgb_rmse": float(np.sqrt(np.mean((rgb_preds - targets) ** 2))),
        "behavior_rmse": float(np.sqrt(np.mean((behavior_preds - targets) ** 2))),
        "rgb_pearson": _pearson(targets, rgb_preds),
        "behavior_pearson": _pearson(targets, behavior_preds),
        "rgb_ccc": _ccc(targets, rgb_preds),
        "behavior_ccc": _ccc(targets, behavior_preds),
        "behavior_better_count": int(np.sum(behavior_errors < rgb_errors)),
        "rgb_better_count": int(np.sum(rgb_errors < behavior_errors)),
        "tie_count": int(np.sum(np.isclose(rgb_errors, behavior_errors))),
    }


def align_prediction_tables(rgb_rows, behavior_rows):
    behavior_by_key = {_prediction_key(row): row for row in behavior_rows}
    aligned = []
    for rgb in rgb_rows:
        behavior = behavior_by_key.get(_prediction_key(rgb))
        if behavior is None:
            continue

        true_bdi = float(rgb["true_bdi"])
        rgb_pred = float(rgb["pred_bdi"])
        behavior_pred = float(behavior["pred_bdi"])
        rgb_abs_error = abs(rgb_pred - true_bdi)
        behavior_abs_error = abs(behavior_pred - true_bdi)
        if math.isclose(rgb_abs_error, behavior_abs_error):
            better_model = "tie"
        elif behavior_abs_error < rgb_abs_error:
            better_model = "behavior"
        else:
            better_model = "rgb"

        aligned.append(
            {
                "video_id": _normalize_video_id(rgb.get("video_id") or behavior.get("video_id")),
                "subject_id": str(rgb.get("subject_id") or behavior.get("subject_id")),
                "task_name": str(rgb.get("task_name") or behavior.get("task_name") or ""),
                "true_bdi": true_bdi,
                "rgb_pred_bdi": rgb_pred,
                "behavior_pred_bdi": behavior_pred,
                "rgb_residual": rgb_pred - true_bdi,
                "behavior_residual": behavior_pred - true_bdi,
                "rgb_abs_error": rgb_abs_error,
                "behavior_abs_error": behavior_abs_error,
                "error_delta_behavior_minus_rgb": behavior_abs_error - rgb_abs_error,
                "better_model": better_model,
                "severity_group": str(rgb.get("severity_group") or behavior.get("severity_group") or ""),
            }
        )
    return aligned


def _format_comparison_rows(rows):
    formatted = []
    for row in rows:
        formatted.append(
            {
                key: _format_float(value) if isinstance(value, float) else value
                for key, value in row.items()
            }
        )
    return formatted


def _summary_rows(rows):
    groups = [("all", rows)]
    severity_groups = sorted({row["severity_group"] for row in rows if row.get("severity_group")})
    for severity in severity_groups:
        groups.append((f"severity:{severity}", [row for row in rows if row.get("severity_group") == severity]))

    summary = []
    for group_name, group_rows in groups:
        metrics = _metrics(group_rows)
        summary.append(
            {
                "group": group_name,
                "count": str(metrics["count"]),
                "rgb_mae": _format_float(metrics["rgb_mae"]),
                "behavior_mae": _format_float(metrics["behavior_mae"]),
                "rgb_rmse": _format_float(metrics["rgb_rmse"]),
                "behavior_rmse": _format_float(metrics["behavior_rmse"]),
                "rgb_pearson": _format_float(metrics["rgb_pearson"]),
                "behavior_pearson": _format_float(metrics["behavior_pearson"]),
                "rgb_ccc": _format_float(metrics["rgb_ccc"]),
                "behavior_ccc": _format_float(metrics["behavior_ccc"]),
                "behavior_better_count": str(metrics["behavior_better_count"]),
                "rgb_better_count": str(metrics["rgb_better_count"]),
                "tie_count": str(metrics["tie_count"]),
            }
        )
    return summary


def compare_behavior_predictions(rgb_prediction_csv, behavior_prediction_csv, output_dir):
    output_dir = ensure_dir(output_dir)
    rgb_rows = read_prediction_table(rgb_prediction_csv)
    behavior_rows = read_prediction_table(behavior_prediction_csv)
    aligned = align_prediction_tables(rgb_rows, behavior_rows)

    comparison_path = output_dir / "rgb_behavior_prediction_comparison.csv"
    summary_path = output_dir / "rgb_behavior_prediction_summary.csv"
    write_csv_rows(comparison_path, _format_comparison_rows(aligned), COMPARISON_FIELDS)
    write_csv_rows(summary_path, _summary_rows(aligned), SUMMARY_FIELDS)
    return comparison_path, summary_path
