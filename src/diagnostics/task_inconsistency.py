"""Task inconsistency mixed-factor audit.

This module investigates why the same subject receives different BDI
predictions on the Freeform and Northwind videos.  It pairs the two task
videos for each subject, computes the prediction gap, and correlates that gap
with frame-level and subject-level artifact / quality / geometry / temporal
variables.

Outputs (see ``docs/RGB_OVERFITTING_AUDIT_PLAN.md`` P0-G):

- ``task_inconsistency_manifest.csv``: paired Freeform/Northwind records sorted
  by prediction difference.
- ``task_artifact_correlation.csv``: correlation of prediction/task gaps with
  merged artifact / quality / geometry / temporal variables.
- ``task_inconsistency_report.md``: interpretation report.
"""

import math
from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, read_csv_rows, write_csv_rows


MANIFEST_COLUMNS = [
    "rank",
    "subject_id",
    "freeform_video_id",
    "northwind_video_id",
    "freeform_true_bdi",
    "northwind_true_bdi",
    "freeform_pred_bdi",
    "northwind_pred_bdi",
    "true_bdi_diff",
    "pred_bdi_diff",
    "abs_pred_bdi_diff",
    "mean_abs_error",
    "freeform_severity_group",
    "northwind_severity_group",
    "recommended_diagnostics",
]

CORRELATION_COLUMNS = [
    "variable",
    "n_pairs",
    "with_true_bdi_diff",
    "with_pred_bdi_diff",
    "with_abs_pred_bdi_diff",
    "with_mean_abs_error",
]

SEVERITY_GROUPS = ["minimal", "mild", "moderate", "severe"]


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


def _normalize_video_id(video_id):
    """Strip processing suffixes so aligned and raw IDs can be matched."""
    video_id = str(video_id or "")
    for suffix in ("_aligned",):
        if video_id.endswith(suffix):
            video_id = video_id[: -len(suffix)]
    return video_id


def load_prediction_records(predictions_csv):
    """Load prediction rows and normalize video IDs."""
    rows = read_csv_rows(predictions_csv)
    records = []
    for row in rows:
        try:
            records.append(
                {
                    "video_id": str(row.get("video_id", row["subject_id"])),
                    "normalized_video_id": _normalize_video_id(row.get("video_id", row["subject_id"])),
                    "subject_id": str(row["subject_id"]),
                    "task_name": str(row.get("task_name", "")),
                    "true_bdi": float(row["true_bdi"]),
                    "pred_bdi": float(row["pred_bdi"]),
                    "residual": float(row["residual"]),
                    "abs_error": float(row["abs_error"]),
                    "severity_group": row.get("severity_group", ""),
                }
            )
        except (KeyError, ValueError):
            continue
    return records


def pair_predictions_by_subject(records):
    """Pair Freeform and Northwind predictions for the same subject.

    Returns:
        List of paired dicts.  Subjects without exactly one Freeform and one
        Northwind record are skipped.
    """
    by_subject = {}
    for record in records:
        by_subject.setdefault(record["subject_id"], []).append(record)

    pairs = []
    for subject_id, subject_records in by_subject.items():
        freeform = [r for r in subject_records if r["task_name"] == "Freeform"]
        northwind = [r for r in subject_records if r["task_name"] == "Northwind"]
        if len(freeform) != 1 or len(northwind) != 1:
            continue
        ff = freeform[0]
        nw = northwind[0]
        pred_diff = ff["pred_bdi"] - nw["pred_bdi"]
        pairs.append(
            {
                "subject_id": subject_id,
                "freeform_video_id": ff["video_id"],
                "northwind_video_id": nw["video_id"],
                "freeform_true_bdi": ff["true_bdi"],
                "northwind_true_bdi": nw["true_bdi"],
                "freeform_pred_bdi": ff["pred_bdi"],
                "northwind_pred_bdi": nw["pred_bdi"],
                "true_bdi_diff": ff["true_bdi"] - nw["true_bdi"],
                "pred_bdi_diff": pred_diff,
                "abs_pred_bdi_diff": abs(pred_diff),
                "mean_abs_error": (ff["abs_error"] + nw["abs_error"]) / 2.0,
                "freeform_severity_group": ff["severity_group"],
                "northwind_severity_group": nw["severity_group"],
            }
        )
    return pairs


def _load_summary_csv(csv_path, key_column="video_id"):
    """Load a summary CSV and index it by normalized video_id.

    Numeric columns are auto-detected.  The returned dict maps video_id to a
    dict of ``{column: float_value_or_string}``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}

    rows = read_csv_rows(csv_path)
    if not rows:
        return {}

    numeric_columns = set()
    for column in rows[0].keys():
        if column == key_column:
            continue
        values = [row.get(column, "") for row in rows if row.get(column, "") != ""]
        if values and all(_safe_float(v) is not None for v in values[:10]):
            numeric_columns.add(column)

    result = {}
    for row in rows:
        video_id = _normalize_video_id(row.get(key_column, row.get("video_id", "")))
        if not video_id:
            continue
        parsed = {}
        for column, value in row.items():
            if column == key_column:
                continue
            if column in numeric_columns:
                parsed[column] = _safe_float(value)
            else:
                parsed[column] = value
        result[video_id] = parsed
    return result


def _pair_numeric_variables(pairs, summaries_by_source):
    """Merge external numeric summaries into paired records.

    For each numeric variable ``X`` present in any summary source, the paired
    record receives:

    - ``ff_X`` and ``nw_X``: values for Freeform and Northwind videos.
    - ``X_mean``: mean of the two values.
    - ``X_diff``: Freeform minus Northwind.

    Variables with missing values for a pair are left as ``None``.
    """
    numeric_variables = set()
    for source in summaries_by_source.values():
        for video_data in source.values():
            for key, value in video_data.items():
                if isinstance(value, (int, float)) and key not in ("video_id", "subject_id"):
                    numeric_variables.add(key)

    variable_names = sorted(numeric_variables)

    for pair in pairs:
        ff_id = _normalize_video_id(pair["freeform_video_id"])
        nw_id = _normalize_video_id(pair["northwind_video_id"])
        for var in variable_names:
            ff_values = [source.get(ff_id, {}).get(var) for source in summaries_by_source.values()]
            nw_values = [source.get(nw_id, {}).get(var) for source in summaries_by_source.values()]
            ff_value = next((v for v in ff_values if v is not None), None)
            nw_value = next((v for v in nw_values if v is not None), None)

            pair[f"ff_{var}"] = ff_value
            pair[f"nw_{var}"] = nw_value
            if ff_value is not None and nw_value is not None:
                pair[f"{var}_mean"] = (ff_value + nw_value) / 2.0
                pair[f"{var}_diff"] = ff_value - nw_value
            else:
                pair[f"{var}_mean"] = None
                pair[f"{var}_diff"] = None

    return variable_names


def _correlation(x, y):
    """Pearson correlation between two equal-length sequences."""
    x = np.asarray([v for v in x if v is not None], dtype=float)
    y = np.asarray([v for v in y if v is not None], dtype=float)
    if x.size < 2 or y.size < 2 or x.size != y.size:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def compute_task_artifact_correlations(pairs, variable_names):
    """Compute correlations between task inconsistency and merged variables.

    For each numeric variable ``X`` we correlate four target variables with
    ``X_diff`` and ``X_mean``:

    - ``true_bdi_diff``
    - ``pred_bdi_diff``
    - ``abs_pred_bdi_diff``
    - ``mean_abs_error``

    Returns a list of correlation rows.
    """
    target_variables = [
        ("true_bdi_diff", "true_bdi_diff"),
        ("pred_bdi_diff", "pred_bdi_diff"),
        ("abs_pred_bdi_diff", "abs_pred_bdi_diff"),
        ("mean_abs_error", "mean_abs_error"),
    ]
    suffixes = ["_diff", "_mean"]

    rows = []
    for var in variable_names:
        for suffix in suffixes:
            col = f"{var}{suffix}"
            values = [pair.get(col) for pair in pairs]
            n_valid = sum(1 for v in values if v is not None)
            if n_valid < 2:
                continue

            row = {"variable": col, "n_pairs": n_valid}
            for target_key, target_name in target_variables:
                target_values = [pair[target_key] for pair in pairs]
                clean = [(v, t) for v, t in zip(values, target_values) if v is not None and t is not None]
                if len(clean) < 2:
                    row[f"with_{target_name}"] = None
                else:
                    vals, targets = zip(*clean)
                    row[f"with_{target_name}"] = _correlation(vals, targets)
            rows.append(row)
    return rows


def _recommended_diagnostics(pair):
    """Return a short diagnostic recommendation string for a paired subject."""
    recommendations = []
    if pair["abs_pred_bdi_diff"] > 5:
        recommendations.append("high_task_diff")
    if pair["mean_abs_error"] > 8:
        recommendations.append("high_error")
    severity_groups = {pair["freeform_severity_group"], pair["northwind_severity_group"]}
    if len(severity_groups) > 1:
        recommendations.append("severity_mismatch")
    if not recommendations:
        recommendations.append("reference")
    return ";".join(recommendations)


def write_task_inconsistency_manifest(csv_path, pairs, top_n=None):
    """Write the paired-task inconsistency manifest."""
    sorted_pairs = sorted(pairs, key=lambda p: p["abs_pred_bdi_diff"], reverse=True)
    if top_n is not None:
        sorted_pairs = sorted_pairs[:top_n]

    rows = []
    for rank, pair in enumerate(sorted_pairs, start=1):
        rows.append(
            {
                "rank": rank,
                "subject_id": pair["subject_id"],
                "freeform_video_id": pair["freeform_video_id"],
                "northwind_video_id": pair["northwind_video_id"],
                "freeform_true_bdi": _format_scalar(pair["freeform_true_bdi"]),
                "northwind_true_bdi": _format_scalar(pair["northwind_true_bdi"]),
                "freeform_pred_bdi": _format_scalar(pair["freeform_pred_bdi"]),
                "northwind_pred_bdi": _format_scalar(pair["northwind_pred_bdi"]),
                "true_bdi_diff": _format_scalar(pair["true_bdi_diff"]),
                "pred_bdi_diff": _format_scalar(pair["pred_bdi_diff"]),
                "abs_pred_bdi_diff": _format_scalar(pair["abs_pred_bdi_diff"]),
                "mean_abs_error": _format_scalar(pair["mean_abs_error"]),
                "freeform_severity_group": pair["freeform_severity_group"],
                "northwind_severity_group": pair["northwind_severity_group"],
                "recommended_diagnostics": _recommended_diagnostics(pair),
            }
        )
    write_csv_rows(csv_path, rows, MANIFEST_COLUMNS)
    return Path(csv_path)


def write_task_artifact_correlation(csv_path, correlation_rows):
    """Write the variable-correlation table."""
    formatted = [
        {col: _format_scalar(row.get(col)) for col in CORRELATION_COLUMNS}
        for row in correlation_rows
    ]
    write_csv_rows(csv_path, formatted, CORRELATION_COLUMNS)
    return Path(csv_path)


def write_task_inconsistency_report(report_path, pairs, correlation_rows, generated_files, top_n=10):
    """Write the markdown interpretation report."""
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    pred_diffs = [p["abs_pred_bdi_diff"] for p in pairs]
    mean_abs_errors = [p["mean_abs_error"] for p in pairs]

    lines = [
        "# Task Inconsistency Mixed-Factor Audit Report",
        "",
        "## Summary",
        "",
        f"- Paired subjects: {len(pairs)}",
        f"- Mean absolute prediction diff: {float(np.mean(pred_diffs)):.4f}" if pred_diffs else "- Mean absolute prediction diff: N/A",
        f"- Median absolute prediction diff: {float(np.median(pred_diffs)):.4f}" if pred_diffs else "- Median absolute prediction diff: N/A",
        f"- Mean paired MAE: {float(np.mean(mean_abs_errors)):.4f}" if mean_abs_errors else "- Mean paired MAE: N/A",
        "",
        "## Artifact / Quality / Geometry / Temporal Correlations",
        "",
        "| Variable | N | true_diff | pred_diff | abs_pred_diff | mean_abs_error |",
        "|---|---|---:|---:|---:|---:|",
    ]

    sorted_rows = sorted(
        correlation_rows,
        key=lambda r: abs(r.get("with_abs_pred_bdi_diff") or 0.0),
        reverse=True,
    )
    for row in sorted_rows:
        lines.append(
            f"| {row['variable']} | {row['n_pairs']} | "
            f"{_format_scalar(row.get('with_true_bdi_diff'))} | "
            f"{_format_scalar(row.get('with_pred_bdi_diff'))} | "
            f"{_format_scalar(row.get('with_abs_pred_bdi_diff'))} | "
            f"{_format_scalar(row.get('with_mean_abs_error'))} |"
        )

    lines.extend(
        [
            "",
            f"## Top {top_n} Most Inconsistent Subjects",
            "",
            "| Rank | Subject | FF Video | NW Video | FF Pred | NW Pred | Abs Diff | Mean Error | Recommendation |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    sorted_pairs = sorted(pairs, key=lambda p: p["abs_pred_bdi_diff"], reverse=True)[:top_n]
    for rank, pair in enumerate(sorted_pairs, start=1):
        lines.append(
            f"| {rank} | {pair['subject_id']} | {pair['freeform_video_id']} | {pair['northwind_video_id']} | "
            f"{_format_scalar(pair['freeform_pred_bdi'])} | "
            f"{_format_scalar(pair['northwind_pred_bdi'])} | "
            f"{_format_scalar(pair['abs_pred_bdi_diff'])} | "
            f"{_format_scalar(pair['mean_abs_error'])} | "
            f"{_recommended_diagnostics(pair)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A high correlation between ``abs_pred_bdi_diff`` and a variable "
            "suggests that Freeform/Northwind prediction inconsistency is driven "
            "by that factor.",
            "- Variables ending in ``_diff`` compare the two task videos of the same "
            "subject; strong correlations here point to task-specific confounds.",
            "- Variables ending in ``_mean`` reflect the average level of a factor; "
            "correlations here suggest subject-level confounds.",
            "- Inspect the top inconsistent subjects with attention, occlusion, "
            "keyframe and aligned-frame diagnostics to confirm visual mechanisms.",
            "",
            "## Generated Files",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in generated_files)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_task_inconsistency_audit(
    predictions_csv,
    output_dir,
    black_artifacts_summary=None,
    openface_quality_summary=None,
    alignment_geometry_summary=None,
    temporal_sampling_summary=None,
    top_n=20,
):
    """Run the full task inconsistency mixed-factor audit.

    Args:
        predictions_csv: Path to a prediction CSV with task_name column.
        output_dir: Directory for outputs.
        black_artifacts_summary: Optional path to black_artifact_summary.csv.
        openface_quality_summary: Optional path to openface_quality_summary.csv.
        alignment_geometry_summary: Optional path to alignment_geometry_summary.csv.
        temporal_sampling_summary: Optional path to temporal_sampling_summary.csv.
        top_n: Number of top inconsistent subjects to highlight in the report.

    Returns:
        List of generated file paths.
    """
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    records = load_prediction_records(predictions_csv)
    pairs = pair_predictions_by_subject(records)
    if not pairs:
        raise ValueError(f"No Freeform/Northwind paired subjects found in {predictions_csv}")

    summaries_by_source = {}
    if black_artifacts_summary:
        summaries_by_source["black_artifacts"] = _load_summary_csv(black_artifacts_summary)
    if openface_quality_summary:
        summaries_by_source["openface_quality"] = _load_summary_csv(openface_quality_summary)
    if alignment_geometry_summary:
        summaries_by_source["alignment_geometry"] = _load_summary_csv(alignment_geometry_summary)
    if temporal_sampling_summary:
        summaries_by_source["temporal_sampling"] = _load_summary_csv(temporal_sampling_summary)

    variable_names = _pair_numeric_variables(pairs, summaries_by_source)
    correlation_rows = compute_task_artifact_correlations(pairs, variable_names)

    manifest_path = write_task_inconsistency_manifest(
        tables_dir / "task_inconsistency_manifest.csv", pairs, top_n=top_n
    )
    generated.append(manifest_path)

    correlation_path = write_task_artifact_correlation(
        tables_dir / "task_artifact_correlation.csv", correlation_rows
    )
    generated.append(correlation_path)

    report_path = write_task_inconsistency_report(
        reports_dir / "task_inconsistency_report.md",
        pairs=pairs,
        correlation_rows=correlation_rows,
        generated_files=generated,
        top_n=top_n,
    )
    generated.append(report_path)
    return generated
