import math
from pathlib import Path

import numpy as np

from src.diagnostics.black_artifacts import _find_video_dir, normalize_video_id
from src.datasets.temporal_sampling import normalize_temporal_sampling_strategy, select_temporal_indices
from src.diagnostics.io import ensure_dir, read_prediction_table, write_csv_rows


TARGET_COLUMNS = ("true_bdi", "pred_bdi", "residual", "abs_error")
TEMPORAL_SAMPLING_COLUMNS = [
    "video_id",
    "source_dir",
    "frame_count",
    "sample_step",
    "max_seq_len",
    "sampling_strategy",
    "model_max_len",
    "sampled_frame_count",
    "selected_frame_count",
    "truncated_frame_count",
    "padding_frame_count",
    "valid_ratio",
    "padding_ratio",
    "truncated_ratio",
    "raw_to_selected_ratio",
]


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


def _task_name_from_video_id(video_id):
    text = str(video_id or "")
    if "Freeform" in text:
        return "Freeform"
    if "Northwind" in text:
        return "Northwind"
    return ""


def summarize_video_temporal_sampling(video_dir, video_id=None, sample_step=10, max_seq_len=2000, sampling_strategy="stride_head"):
    video_dir = Path(video_dir)
    video_id = video_id or video_dir.name
    sample_step = max(1, int(sample_step))
    max_seq_len = max(1, int(max_seq_len))
    sampling_strategy = normalize_temporal_sampling_strategy(sampling_strategy)
    model_max_len = max(1, int(max_seq_len // sample_step))

    frames = sorted(video_dir.glob("*.jpg"))
    sampled_count = len(frames[::sample_step])
    selected_indices = select_temporal_indices(
        len(frames),
        sample_step=sample_step,
        max_seq_len=max_seq_len,
        strategy=sampling_strategy,
        seed=str(video_dir),
    )
    selected_count = len(selected_indices)
    candidate_count = sampled_count if sampling_strategy == "stride_head" else len(frames)
    truncated_count = max(0, candidate_count - selected_count)
    padding_count = max(0, model_max_len - selected_count)

    return {
        "video_id": str(video_id),
        "source_dir": str(video_dir),
        "frame_count": len(frames),
        "sample_step": sample_step,
        "max_seq_len": max_seq_len,
        "sampling_strategy": sampling_strategy,
        "model_max_len": model_max_len,
        "sampled_frame_count": sampled_count,
        "selected_frame_count": selected_count,
        "truncated_frame_count": truncated_count,
        "padding_frame_count": padding_count,
        "valid_ratio": selected_count / model_max_len if model_max_len else None,
        "padding_ratio": padding_count / model_max_len if model_max_len else None,
        "truncated_ratio": truncated_count / candidate_count if candidate_count else None,
        "raw_to_selected_ratio": selected_count / len(frames) if frames else None,
    }


def summarize_prediction_videos_temporal_sampling(
    predictions_csv,
    image_root,
    sample_step=10,
    max_seq_len=2000,
    sampling_strategy="stride_head",
):
    prediction_rows = read_prediction_table(predictions_csv)
    summaries = []
    missing = []
    seen = set()
    for row in prediction_rows:
        video_id = row.get("video_id") or row.get("subject_id")
        normalized = normalize_video_id(video_id)
        if normalized in seen:
            continue
        seen.add(normalized)
        video_dir = _find_video_dir(image_root, video_id)
        if video_dir is None:
            missing.append(video_id)
            continue
        summaries.append(
            summarize_video_temporal_sampling(
                video_dir,
                video_id=normalized,
                sample_step=sample_step,
                max_seq_len=max_seq_len,
                sampling_strategy=sampling_strategy,
            )
        )
    return summaries, missing


def write_temporal_sampling_summary(rows, csv_path):
    formatted = [{key: _format_value(row.get(key)) for key in TEMPORAL_SAMPLING_COLUMNS} for row in rows]
    write_csv_rows(csv_path, formatted, TEMPORAL_SAMPLING_COLUMNS)
    return Path(csv_path)


def merge_predictions_with_temporal_sampling(prediction_rows, sampling_rows):
    by_video = {normalize_video_id(row.get("video_id")): row for row in sampling_rows}
    merged = []
    for prediction in prediction_rows:
        sampling = by_video.get(normalize_video_id(prediction.get("video_id")))
        if sampling is None:
            continue
        row = dict(prediction)
        if not row.get("task_name"):
            row["task_name"] = _task_name_from_video_id(row.get("video_id"))
        for key, value in sampling.items():
            if key not in row:
                row[key] = value
        merged.append(row)
    return merged


def _numeric_feature_names(rows, excluded):
    if not rows:
        return []
    names = []
    for key in rows[0]:
        if key in excluded:
            continue
        values = [_safe_float(row.get(key)) for row in rows]
        if values and all(value is not None for value in values):
            names.append(key)
    return names


def compute_temporal_sampling_correlations(rows):
    excluded = set(TARGET_COLUMNS) | {
        "video_id",
        "subject_id",
        "task_name",
        "severity_group",
        "source_dir",
    }
    features = _numeric_feature_names(rows, excluded=excluded)
    correlations = []
    for feature in features:
        x = np.asarray([_safe_float(row.get(feature)) for row in rows], dtype=float)
        if x.size < 2 or x.std() <= 1e-8:
            continue
        for target in TARGET_COLUMNS:
            y = np.asarray([_safe_float(row.get(target)) for row in rows], dtype=float)
            if y.size < 2 or y.std() <= 1e-8:
                continue
            corr = float(np.corrcoef(x, y)[0, 1])
            correlations.append(
                {
                    "feature": feature,
                    "target": target,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                    "count": int(x.size),
                }
            )
    correlations.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return correlations


def write_temporal_sampling_correlations(rows, csv_path):
    fields = ["feature", "target", "correlation", "abs_correlation", "count"]
    formatted = [{key: _format_value(row.get(key)) for key in fields} for row in rows]
    write_csv_rows(csv_path, formatted, fields)
    return Path(csv_path)


def _mean(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(values)) if values else None


def _group_summary(group_name, rows):
    numeric = ["true_bdi", "pred_bdi", "residual", "abs_error", "frame_count", "selected_frame_count", "valid_ratio", "padding_ratio", "truncated_ratio"]
    result = {"group": group_name, "count": len(rows)}
    for key in numeric:
        result[f"{key}_mean"] = _mean([_safe_float(row.get(key)) for row in rows])
    return result


def build_temporal_sampling_group_summary(rows):
    summaries = []
    for key in ("severity_group", "task_name"):
        groups = {}
        for row in rows:
            value = row.get(key) or "unknown"
            groups.setdefault(value, []).append(row)
        for value in sorted(groups):
            summaries.append(_group_summary(f"{key}:{value}", groups[value]))

    frame_values = sorted(_safe_float(row.get("frame_count")) for row in rows)
    frame_values = [value for value in frame_values if value is not None]
    if len(frame_values) >= 4:
        q1_max = frame_values[max(0, int(math.floor((len(frame_values) - 1) * 0.25)))]
        q3_min = frame_values[int(math.floor((len(frame_values) - 1) * 0.75))]
        low = [row for row in rows if _safe_float(row.get("frame_count")) is not None and _safe_float(row.get("frame_count")) <= q1_max]
        high = [row for row in rows if _safe_float(row.get("frame_count")) is not None and _safe_float(row.get("frame_count")) >= q3_min]
        summaries.append(_group_summary("frame_count:low_q1", low))
        summaries.append(_group_summary("frame_count:high_q4", high))
    return summaries


def write_temporal_sampling_group_summary(rows, csv_path):
    fields = [
        "group",
        "count",
        "true_bdi_mean",
        "pred_bdi_mean",
        "residual_mean",
        "abs_error_mean",
        "frame_count_mean",
        "selected_frame_count_mean",
        "valid_ratio_mean",
        "padding_ratio_mean",
        "truncated_ratio_mean",
    ]
    formatted = [{key: _format_value(row.get(key)) for key in fields} for row in rows]
    write_csv_rows(csv_path, formatted, fields)
    return Path(csv_path)


def write_temporal_sampling_report(report_path, generated_files, summaries, missing, merged_rows, correlations, group_rows):
    report_path = Path(report_path)
    ensure_dir(report_path.parent)
    top = correlations[:12]
    max_abs = max((row["abs_correlation"] for row in correlations), default=0.0)

    lines = [
        "# Temporal Sampling Audit Report",
        "",
        f"- Videos summarized: {len(summaries)}",
        f"- Missing videos: {len(missing)}",
        f"- Matched prediction rows: {len(merged_rows)}",
        f"- Max absolute correlation: {max_abs:.4f}",
        "",
        "## Generated Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in generated_files)
    if missing:
        lines.extend(["", "## Missing Video Directories", ""])
        for item in missing[:20]:
            lines.append(f"- `{item}`")

    lines.extend(["", "## Top Correlations", ""])
    if top:
        lines.append("| Feature | Target | Correlation |")
        lines.append("|---|---:|---:|")
        for row in top:
            lines.append(f"| {row['feature']} | {row['target']} | {row['correlation']:.4f} |")
    else:
        lines.append("No valid correlations were computed.")

    lines.extend(["", "## Group Summary", ""])
    if group_rows:
        lines.append("| Group | Count | Pred Mean | Residual Mean | Abs Error Mean | Valid Ratio | Padding Ratio |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in group_rows[:20]:
            lines.append(
                "| {group} | {count} | {pred:.4f} | {residual:.4f} | {abs_error:.4f} | {valid:.4f} | {padding:.4f} |".format(
                    group=row["group"],
                    count=row["count"],
                    pred=row["pred_bdi_mean"] or 0.0,
                    residual=row["residual_mean"] or 0.0,
                    abs_error=row["abs_error_mean"] or 0.0,
                    valid=row["valid_ratio_mean"] or 0.0,
                    padding=row["padding_ratio_mean"] or 0.0,
                )
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `sampled_frame_count` is computed after `SAMPLE_STEP` subsampling for compatibility with the historical stride rule.",
            "- `selected_frame_count` follows the configured sampling strategy and the model-visible length `MAX_SEQ_LEN // SAMPLE_STEP`.",
            "- `padding_ratio` and `valid_ratio` describe the model-visible temporal tensor length, not raw video duration.",
            "- Use `--sampling-strategy` to match `PROCESS_TEMPORAL.SAMPLING_STRATEGY` from the experiment config.",
            "- This audit is offline diagnostics only and must not alter training split, labels, or test-time decisions.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_temporal_sampling_audit(
    predictions_csv,
    image_root,
    output_dir,
    sample_step=10,
    max_seq_len=2000,
    sampling_strategy="stride_head",
):
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    summaries, missing = summarize_prediction_videos_temporal_sampling(
        predictions_csv=predictions_csv,
        image_root=image_root,
        sample_step=sample_step,
        max_seq_len=max_seq_len,
        sampling_strategy=sampling_strategy,
    )
    summary_path = write_temporal_sampling_summary(summaries, tables_dir / "temporal_sampling_summary.csv")
    generated.append(summary_path)

    prediction_rows = read_prediction_table(predictions_csv)
    merged_rows = merge_predictions_with_temporal_sampling(prediction_rows, summaries)
    merged_fields = []
    for row in merged_rows:
        for key in row:
            if key not in merged_fields:
                merged_fields.append(key)
    if not merged_fields:
        merged_fields = ["video_id", "subject_id"]
    merged_path = tables_dir / "temporal_sampling_merged.csv"
    write_csv_rows(
        merged_path,
        [{key: _format_value(row.get(key)) for key in merged_fields} for row in merged_rows],
        merged_fields,
    )
    generated.append(merged_path)

    correlations = compute_temporal_sampling_correlations(merged_rows)
    correlation_path = write_temporal_sampling_correlations(
        correlations,
        tables_dir / "temporal_sampling_correlation.csv",
    )
    generated.append(correlation_path)

    group_rows = build_temporal_sampling_group_summary(merged_rows)
    group_path = write_temporal_sampling_group_summary(group_rows, tables_dir / "temporal_sampling_group_summary.csv")
    generated.append(group_path)

    report_path = write_temporal_sampling_report(
        reports_dir / "temporal_sampling_audit_report.md",
        generated_files=generated,
        summaries=summaries,
        missing=missing,
        merged_rows=merged_rows,
        correlations=correlations,
        group_rows=group_rows,
    )
    generated.append(report_path)
    return generated
