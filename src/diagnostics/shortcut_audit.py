import math
from pathlib import Path

import numpy as np

from src.diagnostics.case_studies import write_case_study_manifest
from src.diagnostics.io import ensure_dir, read_csv_rows, read_prediction_table, write_csv_rows
from src.diagnostics.openface_quality import (
    DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    summarize_openface_root,
    write_openface_quality_summary,
)


TARGET_COLUMNS = ("true_bdi", "pred_bdi", "residual", "abs_error")
PREDICTOR_EXCLUDED_COLUMNS = set(TARGET_COLUMNS) | {
    "video_id",
    "subject_id",
    "matched_on",
    "severity_group",
    "source_file",
    "task_name",
}


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _numeric_feature_names(rows, excluded=()):
    if not rows:
        return []
    excluded = set(excluded)
    names = []
    for key in rows[0]:
        if key in excluded:
            continue
        values = [_safe_float(row.get(key)) for row in rows]
        if values and all(value is not None for value in values):
            names.append(key)
    return names


def _normalize_video_id(video_id):
    video_id = str(video_id or "")
    if video_id.endswith(".csv"):
        video_id = Path(video_id).stem
    processing_suffixes = ("_aligned",)
    for suffix in processing_suffixes:
        if video_id.endswith(suffix):
            video_id = video_id[: -len(suffix)]
    return video_id


def _compatible_video_ids(left, right):
    left = _normalize_video_id(left)
    right = _normalize_video_id(right)
    if not left or not right:
        return False
    return left == right or f"{left}_video" == right or left == f"{right}_video"


def merge_predictions_with_quality(prediction_records, quality_rows):
    quality_by_video = {}
    quality_by_subject = {}
    for row in quality_rows:
        video_id = _normalize_video_id(row.get("video_id"))
        subject_id = str(row.get("subject_id", ""))
        if video_id:
            quality_by_video[video_id] = row
        if subject_id:
            quality_by_subject.setdefault(subject_id, []).append(row)

    merged = []
    for record in prediction_records:
        video_id = _normalize_video_id(record.get("video_id", record["subject_id"]))
        subject_id = str(record["subject_id"])
        quality = quality_by_video.get(video_id)
        matched_on = "video_id"

        if quality is None:
            for quality_video_id, quality_row in quality_by_video.items():
                if _compatible_video_ids(video_id, quality_video_id):
                    quality = quality_row
                    break

        if quality is None:
            subject_matches = quality_by_subject.get(subject_id, [])
            if len(subject_matches) == 1:
                quality = subject_matches[0]
                matched_on = "subject_id"
            elif len(subject_matches) > 1:
                continue

        if quality is None:
            continue
        row = {
            "video_id": video_id,
            "subject_id": subject_id,
            "matched_on": matched_on,
            "true_bdi": record["true_bdi"],
            "pred_bdi": record["pred_bdi"],
            "residual": record["residual"],
            "abs_error": record["abs_error"],
            "severity_group": record.get("severity_group", ""),
        }
        for key, value in quality.items():
            if key not in row:
                row[key] = value
        merged.append(row)
    return merged


def write_merged_shortcut_table(rows, csv_path):
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    if not rows:
        write_csv_rows(csv_path, [], ["subject_id"])
        return csv_path

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    formatted_rows = [{key: _format_value(row.get(key)) for key in fieldnames} for row in rows]
    write_csv_rows(csv_path, formatted_rows, fieldnames)
    return csv_path


def compute_shortcut_correlations(rows, target_columns=TARGET_COLUMNS):
    feature_names = _numeric_feature_names(
        rows,
        excluded=set(target_columns) | {"video_id", "subject_id", "matched_on", "severity_group", "source_file"},
    )
    correlations = []
    for feature in feature_names:
        feature_values = np.asarray([_safe_float(row.get(feature)) for row in rows], dtype=float)
        if feature_values.size < 2 or feature_values.std() <= 1e-8:
            continue
        for target in target_columns:
            target_values = np.asarray([_safe_float(row.get(target)) for row in rows], dtype=float)
            if target_values.size < 2 or target_values.std() <= 1e-8:
                continue
            corr = float(np.corrcoef(feature_values, target_values)[0, 1])
            correlations.append(
                {
                    "feature": feature,
                    "target": target,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                    "count": int(feature_values.size),
                }
            )
    correlations.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return correlations


def write_shortcut_correlations(correlations, csv_path):
    fields = ["feature", "target", "correlation", "abs_correlation", "count"]
    rows = [{key: _format_value(row.get(key)) for key in fields} for row in correlations]
    write_csv_rows(csv_path, rows, fields)
    return Path(csv_path)


def _correlation_matrix_rows(correlations, max_features=40):
    if not correlations:
        return [], []
    features = []
    for row in correlations:
        feature = row["feature"]
        if feature not in features:
            features.append(feature)
        if len(features) >= max_features:
            break

    by_pair = {(row["feature"], row["target"]): row["correlation"] for row in correlations}
    matrix_rows = []
    for feature in features:
        matrix_row = {"feature_index": str(len(matrix_rows))}
        for target in TARGET_COLUMNS:
            matrix_row[target] = _format_value(by_pair.get((feature, target), 0.0))
        matrix_rows.append(matrix_row)
    return matrix_rows, features


def plot_shortcut_correlation_heatmap(correlations, save_path, max_features=40):
    matrix_rows, features = _correlation_matrix_rows(correlations, max_features=max_features)
    if not matrix_rows:
        return None

    plt = _require_matplotlib()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    values = np.asarray(
        [[_safe_float(row.get(target)) or 0.0 for target in TARGET_COLUMNS] for row in matrix_rows],
        dtype=float,
    )

    fig_height = max(4.8, min(16.0, len(features) * 0.34))
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    im = ax.imshow(values, vmin=-1.0, vmax=1.0, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(TARGET_COLUMNS)))
    ax.set_xticklabels(TARGET_COLUMNS, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features, fontsize=7)
    ax.set_title("Shortcut Feature Correlation Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.86)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    labels_path = save_path.with_name(save_path.stem + "_features.csv")
    write_csv_rows(
        labels_path,
        [{"feature_index": str(idx), "feature": feature} for idx, feature in enumerate(features)],
        ["feature_index", "feature"],
    )
    return save_path


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for shortcut audit plots.") from exc
    return plt


def plot_residual_dependency(rows, save_dir, features=None):
    plt = _require_matplotlib()
    save_dir = ensure_dir(save_dir)
    if features is None:
        features = [
            "confidence_mean",
            "success_ratio",
            "failed_frame_ratio",
            "pose_rx_abs_mean",
            "pose_ry_abs_mean",
            "pose_rz_abs_mean",
            "gaze_angle_x_abs_mean",
            "gaze_angle_y_abs_mean",
            "landmark_motion_mean",
            "tracking_quality_score",
        ]

    paths = []
    for feature in features:
        values = [_safe_float(row.get(feature)) for row in rows]
        errors = [_safe_float(row.get("abs_error")) for row in rows]
        pairs = [(x, y) for x, y in zip(values, errors) if x is not None and y is not None]
        if len(pairs) < 2:
            continue
        x_values = np.asarray([pair[0] for pair in pairs], dtype=float)
        y_values = np.asarray([pair[1] for pair in pairs], dtype=float)
        corr = np.nan
        if x_values.std() > 1e-8 and y_values.std() > 1e-8:
            corr = float(np.corrcoef(x_values, y_values)[0, 1])

        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        ax.scatter(x_values, y_values, s=38, alpha=0.75, edgecolors="none")
        ax.set_xlabel(feature)
        ax.set_ylabel("Absolute Error")
        ax.set_title(f"Residual Dependency: {feature}")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.text(
            0.04,
            0.96,
            f"Pearson={corr:.3f}" if math.isfinite(corr) else "Pearson=nan",
            transform=ax.transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.84, edgecolor="none"),
        )
        save_path = save_dir / f"residual_vs_{feature}.png"
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(save_path)
    return paths


def _predictor_metrics(targets, preds):
    residuals = preds - targets
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    pearson = np.nan
    if targets.size >= 2 and targets.std() > 1e-8 and preds.std() > 1e-8:
        pearson = float(np.corrcoef(targets, preds)[0, 1])
    return mae, rmse, pearson


def _predictor_feature_names(rows):
    return _numeric_feature_names(rows, excluded=PREDICTOR_EXCLUDED_COLUMNS)


def _predictor_arrays(rows, target_column="true_bdi"):
    feature_names = _predictor_feature_names(rows)
    if len(rows) < 2 or not feature_names:
        return None, None, []

    target_values = [_safe_float(row.get(target_column)) for row in rows]
    if any(value is None for value in target_values):
        return None, None, []

    x = np.asarray([[float(_safe_float(row.get(name))) for name in feature_names] for row in rows], dtype=float)
    y = np.asarray([float(value) for value in target_values], dtype=float)
    keep = x.std(axis=0) > 1e-8
    x = x[:, keep]
    feature_names = [name for name, keep_feature in zip(feature_names, keep) if keep_feature]
    if x.shape[1] == 0:
        return None, None, []
    return x, y, feature_names


def _make_group_folds(groups, num_folds=5, seed=42):
    unique_groups = sorted({str(group) for group in groups})
    if len(unique_groups) < 2:
        return []

    num_folds = min(max(2, int(num_folds)), len(unique_groups))
    rng = np.random.default_rng(seed)
    shuffled = list(unique_groups)
    rng.shuffle(shuffled)

    folds = [set() for _ in range(num_folds)]
    for idx, group in enumerate(shuffled):
        folds[idx % num_folds].add(group)
    return [fold for fold in folds if fold]


def evaluate_shortcut_predictors(rows):
    x, y, feature_names = _predictor_arrays(rows)
    if x is None:
        return []

    x_mean = x.mean(axis=0)
    x_std = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    x_scaled = (x - x_mean) / x_std
    rows_out = []

    mean_preds = np.full_like(y, y.mean(), dtype=float)
    mae, rmse, pearson = _predictor_metrics(y, mean_preds)
    rows_out.append(
        {
            "model": "mean",
            "evaluation": "in_sample_diagnostic",
            "mae": mae,
            "rmse": rmse,
            "pearson": pearson,
            "num_samples": int(y.size),
            "num_features": 0,
        }
    )

    design = np.column_stack([np.ones(y.size), x_scaled])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    linear_preds = design @ coef
    mae, rmse, pearson = _predictor_metrics(y, linear_preds)
    rows_out.append(
        {
            "model": "linear_regression",
            "evaluation": "in_sample_diagnostic",
            "mae": mae,
            "rmse": rmse,
            "pearson": pearson,
            "num_samples": int(y.size),
            "num_features": int(x.shape[1]),
        }
    )

    alpha = 1.0
    xtx = x_scaled.T @ x_scaled
    ridge_weights = np.linalg.solve(xtx + alpha * np.eye(xtx.shape[0]), x_scaled.T @ (y - y.mean()))
    ridge_preds = y.mean() + x_scaled @ ridge_weights
    mae, rmse, pearson = _predictor_metrics(y, ridge_preds)
    rows_out.append(
        {
            "model": "ridge_alpha_1",
            "evaluation": "in_sample_diagnostic",
            "mae": mae,
            "rmse": rmse,
            "pearson": pearson,
            "num_samples": int(y.size),
            "num_features": int(x.shape[1]),
        }
    )
    return rows_out


def evaluate_shortcut_predictors_grouped_cv(
    rows,
    group_column="subject_id",
    target_column="true_bdi",
    alpha_values=(10.0, 100.0, 1000.0, 10000.0),
    num_folds=5,
    seed=42,
):
    x, y, feature_names = _predictor_arrays(rows, target_column=target_column)
    if x is None:
        return []

    groups = np.asarray([str(row.get(group_column, "")) for row in rows], dtype=object)
    folds = _make_group_folds(groups, num_folds=num_folds, seed=seed)
    if len(folds) < 2:
        return []

    num_groups = len(set(groups.tolist()))
    rows_out = []

    model_preds = np.asarray([_safe_float(row.get("pred_bdi")) for row in rows], dtype=object)
    if all(value is not None for value in model_preds):
        model_preds = model_preds.astype(float)
        mae, rmse, pearson = _predictor_metrics(y, model_preds)
        rows_out.append(
            {
                "model": "rgb_mtl_lite",
                "evaluation": "existing_predictions",
                "mae": mae,
                "rmse": rmse,
                "pearson": pearson,
                "num_samples": int(y.size),
                "num_features": 0,
                "num_groups": int(num_groups),
                "num_folds": int(len(folds)),
            }
        )

    mean_preds = np.full_like(y, np.nan, dtype=float)
    for fold in folds:
        test_mask = np.asarray([group in fold for group in groups], dtype=bool)
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            continue
        mean_preds[test_mask] = y[train_mask].mean()
    if np.isfinite(mean_preds).all():
        mae, rmse, pearson = _predictor_metrics(y, mean_preds)
        rows_out.append(
            {
                "model": "mean",
                "evaluation": "grouped_cv_subject",
                "mae": mae,
                "rmse": rmse,
                "pearson": pearson,
                "num_samples": int(y.size),
                "num_features": 0,
                "num_groups": int(num_groups),
                "num_folds": int(len(folds)),
            }
        )

    for alpha in alpha_values:
        alpha = float(alpha)
        preds = np.full_like(y, np.nan, dtype=float)
        for fold in folds:
            test_mask = np.asarray([group in fold for group in groups], dtype=bool)
            train_mask = ~test_mask
            if not train_mask.any() or not test_mask.any():
                continue

            x_train = x[train_mask]
            x_test = x[test_mask]
            y_train = y[train_mask]
            x_mean = x_train.mean(axis=0)
            x_std = np.where(x_train.std(axis=0) > 1e-8, x_train.std(axis=0), 1.0)
            x_train_scaled = (x_train - x_mean) / x_std
            x_test_scaled = (x_test - x_mean) / x_std
            y_mean = y_train.mean()
            xtx = x_train_scaled.T @ x_train_scaled
            weights = np.linalg.solve(
                xtx + alpha * np.eye(xtx.shape[0]),
                x_train_scaled.T @ (y_train - y_mean),
            )
            preds[test_mask] = y_mean + x_test_scaled @ weights

        if np.isfinite(preds).all():
            mae, rmse, pearson = _predictor_metrics(y, preds)
            alpha_text = f"{alpha:g}"
            rows_out.append(
                {
                    "model": f"ridge_alpha_{alpha_text}",
                    "evaluation": "grouped_cv_subject",
                    "mae": mae,
                    "rmse": rmse,
                    "pearson": pearson,
                    "num_samples": int(y.size),
                    "num_features": int(len(feature_names)),
                    "num_groups": int(num_groups),
                    "num_folds": int(len(folds)),
                }
            )
    return rows_out


def write_predictor_results(rows, csv_path):
    fields = [
        "model",
        "evaluation",
        "mae",
        "rmse",
        "pearson",
        "num_samples",
        "num_features",
        "num_groups",
        "num_folds",
    ]
    formatted = [{key: _format_value(row.get(key)) for key in fields} for row in rows]
    write_csv_rows(csv_path, formatted, fields)
    return Path(csv_path)


def write_shortcut_audit_report(
    report_path,
    generated_files,
    merged_rows,
    correlations,
    predictor_rows,
    grouped_predictor_rows=None,
):
    report_path = Path(report_path)
    ensure_dir(report_path.parent)
    grouped_predictor_rows = grouped_predictor_rows or []
    top_correlations = correlations[:15]
    max_abs = max((row["abs_correlation"] for row in correlations), default=0.0)
    if max_abs >= 0.5:
        risk = "high"
    elif max_abs >= 0.3:
        risk = "medium"
    else:
        risk = "low"

    lines = [
        "# Shortcut Audit Report",
        "",
        f"- Matched samples: {len(merged_rows)}",
        f"- Heuristic shortcut risk: {risk}",
        f"- Max absolute correlation: {max_abs:.4f}",
        "",
        "## Generated Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in generated_files)
    lines.extend(["", "## Top Shortcut Correlations", ""])
    if top_correlations:
        lines.append("| Feature | Target | Correlation |")
        lines.append("|---|---:|---:|")
        for row in top_correlations:
            lines.append(f"| {row['feature']} | {row['target']} | {row['correlation']:.4f} |")
    else:
        lines.append("No valid shortcut correlations were computed.")

    lines.extend(["", "## Shortcut-only Predictor Diagnostics", ""])
    if predictor_rows:
        lines.append("| Model | Evaluation | MAE | RMSE | Pearson |")
        lines.append("|---|---|---:|---:|---:|")
        for row in predictor_rows:
            pearson = row["pearson"]
            pearson_text = f"{pearson:.4f}" if math.isfinite(pearson) else "nan"
            lines.append(
                f"| {row['model']} | {row['evaluation']} | "
                f"{row['mae']:.4f} | {row['rmse']:.4f} | {pearson_text} |"
            )
    else:
        lines.append("Shortcut-only predictor diagnostics were skipped because there were too few valid samples.")

    lines.extend(["", "## Grouped-CV Shortcut-only Predictor Diagnostics", ""])
    if grouped_predictor_rows:
        lines.append("| Model | Evaluation | MAE | RMSE | Pearson | Groups | Folds |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in grouped_predictor_rows:
            pearson = row["pearson"]
            pearson_text = f"{pearson:.4f}" if math.isfinite(pearson) else "nan"
            lines.append(
                f"| {row['model']} | {row['evaluation']} | "
                f"{row['mae']:.4f} | {row['rmse']:.4f} | {pearson_text} | "
                f"{row.get('num_groups', '')} | {row.get('num_folds', '')} |"
            )
    else:
        lines.append(
            "Grouped-CV shortcut-only predictor diagnostics were skipped because there were too few valid groups."
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- In-sample predictor diagnostics are overfitting-sensitive and should only be used as shortcut-risk signals.",
            "- Grouped-CV predictor diagnostics group by subject_id to reduce Freeform/Northwind leakage.",
            "- This report is offline diagnostics only; it must not be used to tune on validation/test labels.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_shortcut_audit(
    predictions_csv,
    openface_root,
    output_dir,
    low_confidence_threshold=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    max_heatmap_features=40,
):
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")
    reports_dir = ensure_dir(output_dir / "reports")

    generated_files = []

    summaries = summarize_openface_root(
        openface_root,
        low_confidence_threshold=low_confidence_threshold,
    )
    quality_path = write_openface_quality_summary(summaries, tables_dir / "openface_quality_summary.csv")
    generated_files.append(quality_path)

    prediction_records = read_prediction_table(predictions_csv)
    merged_rows = merge_predictions_with_quality(prediction_records, read_csv_rows(quality_path))
    merged_path = write_merged_shortcut_table(merged_rows, tables_dir / "shortcut_merged.csv")
    generated_files.append(merged_path)

    case_table_path, case_report_path = write_case_study_manifest(
        merged_rows,
        table_path=tables_dir / "case_study_manifest.csv",
        report_path=reports_dir / "case_study_manifest.md",
    )
    generated_files.extend([case_table_path, case_report_path])

    correlations = compute_shortcut_correlations(merged_rows)
    correlation_path = write_shortcut_correlations(correlations, tables_dir / "shortcut_correlation.csv")
    generated_files.append(correlation_path)

    try:
        heatmap_path = plot_shortcut_correlation_heatmap(
            correlations,
            figures_dir / "shortcut_correlation_heatmap.png",
            max_features=max_heatmap_features,
        )
        if heatmap_path is not None:
            generated_files.append(heatmap_path)
    except ImportError:
        heatmap_path = None

    try:
        generated_files.extend(plot_residual_dependency(merged_rows, figures_dir))
    except ImportError:
        pass

    predictor_rows = evaluate_shortcut_predictors(merged_rows)
    grouped_predictor_rows = evaluate_shortcut_predictors_grouped_cv(merged_rows)
    predictor_path = write_predictor_results(
        predictor_rows + grouped_predictor_rows,
        tables_dir / "shortcut_predictor_results.csv",
    )
    generated_files.append(predictor_path)
    grouped_predictor_path = write_predictor_results(
        grouped_predictor_rows,
        tables_dir / "shortcut_predictor_grouped_cv.csv",
    )
    generated_files.append(grouped_predictor_path)

    report_path = write_shortcut_audit_report(
        reports_dir / "shortcut_audit_report.md",
        generated_files=generated_files,
        merged_rows=merged_rows,
        correlations=correlations,
        predictor_rows=predictor_rows,
        grouped_predictor_rows=grouped_predictor_rows,
    )
    generated_files.append(report_path)

    return generated_files
