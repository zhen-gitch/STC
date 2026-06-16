import csv
import math
import re
from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, read_prediction_table, write_csv_rows
from src.diagnostics.openface_quality import (
    find_openface_csv_files,
    infer_subject_id_from_openface_path,
    infer_task_name_from_openface_path,
    infer_video_id_from_openface_path,
)


TARGET_COLUMNS = ("true_bdi", "pred_bdi", "residual", "abs_error")
SUMMARY_FIELDS = [
    "video_id",
    "subject_id",
    "task_name",
    "source_file",
    "frame_count",
    "valid_landmark_frame_count",
    "landmark_count",
    "confidence_mean",
    "success_ratio",
    "landmark_bbox_width_mean",
    "landmark_bbox_width_std",
    "landmark_bbox_height_mean",
    "landmark_bbox_height_std",
    "landmark_bbox_area_mean",
    "landmark_bbox_area_std",
    "landmark_bbox_aspect_mean",
    "face_center_x_mean",
    "face_center_y_mean",
    "face_center_offset_x_mean",
    "face_center_offset_y_mean",
    "eye_distance_mean",
    "normalized_face_scale_mean",
    "landmark_jitter_mean",
    "landmark_jitter_std",
]
CORRELATION_FIELDS = ["feature", "target", "correlation", "abs_correlation", "count"]
GROUP_FIELDS = [
    "severity_group",
    "count",
    "abs_error_mean",
    "residual_mean",
    "landmark_bbox_area_mean",
    "normalized_face_scale_mean",
    "face_center_offset_x_mean",
    "face_center_offset_y_mean",
    "landmark_jitter_mean",
]


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


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


def _column_index(column_name, prefix):
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", column_name)
    return int(match.group(1)) if match else None


def _numbered_columns(fieldnames, prefix):
    indexed = []
    for name in fieldnames:
        idx = _column_index(name, prefix)
        if idx is not None:
            indexed.append((idx, name))
    return [name for _idx, name in sorted(indexed)]


def normalize_video_id(video_id):
    video_id = str(video_id or "")
    if video_id.endswith(".csv"):
        video_id = Path(video_id).stem
    if video_id.endswith("_aligned"):
        video_id = video_id[: -len("_aligned")]
    return video_id


def _compatible_video_ids(left, right):
    left = normalize_video_id(left)
    right = normalize_video_id(right)
    if not left or not right:
        return False
    return left == right or f"{left}_video" == right or left == f"{right}_video"


def _landmark_array(row, x_columns, y_columns):
    if not x_columns or len(x_columns) != len(y_columns):
        return None
    points = []
    for x_col, y_col in zip(x_columns, y_columns):
        x = _safe_float(row.get(x_col))
        y = _safe_float(row.get(y_col))
        if x is None or y is None:
            return None
        points.append((x, y))
    return np.asarray(points, dtype=float)


def _eye_distance(points):
    if points is None or points.shape[0] < 2:
        return None
    if points.shape[0] >= 68:
        left_eye = points[36:42].mean(axis=0)
        right_eye = points[42:48].mean(axis=0)
        return float(np.linalg.norm(left_eye - right_eye))
    return float(np.linalg.norm(points[0] - points[1]))


def _mean(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(values)) if values else None


def _std(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.std(values)) if values else None


def summarize_openface_geometry_csv(
    csv_path,
    frame_width=112.0,
    frame_height=112.0,
    sample_step=1,
    max_frames=None,
):
    csv_path = Path(csv_path)
    video_id = infer_video_id_from_openface_path(csv_path)
    subject_id = infer_subject_id_from_openface_path(csv_path)
    task_name = infer_task_name_from_openface_path(csv_path)

    widths = []
    heights = []
    areas = []
    aspects = []
    centers_x = []
    centers_y = []
    offsets_x = []
    offsets_y = []
    eye_distances = []
    scales = []
    jitters = []
    confidences = []
    successes = []
    previous_points = None
    frame_count = 0
    valid_count = 0
    landmark_count = 0

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        x_columns = _numbered_columns(fieldnames, "x")
        y_columns = _numbered_columns(fieldnames, "y")
        landmark_count = min(len(x_columns), len(y_columns))

        for row_idx, row in enumerate(reader):
            frame_count += 1
            if sample_step > 1 and row_idx % int(sample_step) != 0:
                continue
            if max_frames is not None and valid_count >= int(max_frames):
                break

            confidence = _safe_float(row.get("confidence"))
            success = _safe_float(row.get("success"))
            if confidence is not None:
                confidences.append(confidence)
            if success is not None:
                successes.append(success)

            points = _landmark_array(row, x_columns, y_columns)
            if points is None or points.size == 0:
                continue
            valid_count += 1

            min_xy = points.min(axis=0)
            max_xy = points.max(axis=0)
            width = float(max_xy[0] - min_xy[0])
            height = float(max_xy[1] - min_xy[1])
            area = float(width * height)
            center = (min_xy + max_xy) / 2.0
            eye_distance = _eye_distance(points)

            widths.append(width)
            heights.append(height)
            areas.append(area)
            aspects.append(float(width / height) if height > 1e-8 else None)
            centers_x.append(float(center[0]))
            centers_y.append(float(center[1]))
            offsets_x.append(float((center[0] - frame_width / 2.0) / frame_width))
            offsets_y.append(float((center[1] - frame_height / 2.0) / frame_height))
            eye_distances.append(eye_distance)
            scales.append(float(math.sqrt(max(area, 0.0)) / math.sqrt(frame_width * frame_height)))
            if previous_points is not None and previous_points.shape == points.shape:
                delta = points - previous_points
                jitters.append(float(np.sqrt(np.sum(delta * delta, axis=1)).mean()))
            previous_points = points

    return {
        "video_id": video_id,
        "subject_id": subject_id,
        "task_name": task_name,
        "source_file": str(csv_path),
        "frame_count": frame_count,
        "valid_landmark_frame_count": valid_count,
        "landmark_count": landmark_count,
        "confidence_mean": _mean(confidences),
        "success_ratio": _mean([1.0 if value >= 0.5 else 0.0 for value in successes]),
        "landmark_bbox_width_mean": _mean(widths),
        "landmark_bbox_width_std": _std(widths),
        "landmark_bbox_height_mean": _mean(heights),
        "landmark_bbox_height_std": _std(heights),
        "landmark_bbox_area_mean": _mean(areas),
        "landmark_bbox_area_std": _std(areas),
        "landmark_bbox_aspect_mean": _mean(aspects),
        "face_center_x_mean": _mean(centers_x),
        "face_center_y_mean": _mean(centers_y),
        "face_center_offset_x_mean": _mean(offsets_x),
        "face_center_offset_y_mean": _mean(offsets_y),
        "eye_distance_mean": _mean(eye_distances),
        "normalized_face_scale_mean": _mean(scales),
        "landmark_jitter_mean": _mean(jitters),
        "landmark_jitter_std": _std(jitters),
    }


def summarize_openface_geometry_root(openface_root, frame_width=112.0, frame_height=112.0, sample_step=1, max_frames=None):
    rows = []
    for csv_path in find_openface_csv_files(openface_root):
        rows.append(
            summarize_openface_geometry_csv(
                csv_path,
                frame_width=frame_width,
                frame_height=frame_height,
                sample_step=sample_step,
                max_frames=max_frames,
            )
        )
    return rows


def write_alignment_geometry_summary(rows, csv_path):
    formatted = [{field: _format_value(row.get(field)) for field in SUMMARY_FIELDS} for row in rows]
    write_csv_rows(csv_path, formatted, SUMMARY_FIELDS)
    return Path(csv_path)


def merge_predictions_with_geometry(prediction_rows, geometry_rows):
    by_video = {normalize_video_id(row["video_id"]): row for row in geometry_rows}
    by_subject = {}
    for row in geometry_rows:
        by_subject.setdefault(str(row.get("subject_id", "")), []).append(row)

    merged = []
    missing = []
    for pred in prediction_rows:
        pred_video = normalize_video_id(pred.get("video_id") or pred.get("subject_id"))
        geometry = by_video.get(pred_video)
        matched_on = "video_id"
        if geometry is None:
            for geometry_video, geometry_row in by_video.items():
                if _compatible_video_ids(pred_video, geometry_video):
                    geometry = geometry_row
                    break
        if geometry is None:
            subject_matches = by_subject.get(str(pred.get("subject_id", "")), [])
            if len(subject_matches) == 1:
                geometry = subject_matches[0]
                matched_on = "subject_id"
        if geometry is None:
            missing.append(pred.get("video_id") or pred.get("subject_id"))
            continue

        row = dict(pred)
        row["matched_on"] = matched_on
        for key, value in geometry.items():
            if key not in row:
                row[key] = value
        merged.append(row)
    return merged, missing


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


def compute_alignment_geometry_correlations(rows):
    excluded = set(TARGET_COLUMNS) | {
        "video_id",
        "subject_id",
        "task_name",
        "severity_group",
        "source_file",
        "matched_on",
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


def write_alignment_geometry_correlations(rows, csv_path):
    formatted = [{field: _format_value(row.get(field)) for field in CORRELATION_FIELDS} for row in rows]
    write_csv_rows(csv_path, formatted, CORRELATION_FIELDS)
    return Path(csv_path)


def summarize_alignment_geometry_groups(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row.get("severity_group") or "unknown", []).append(row)

    summaries = []
    for group_name, group_rows in sorted(grouped.items()):
        summary = {"severity_group": group_name, "count": len(group_rows)}
        for key in GROUP_FIELDS:
            if key in {"severity_group", "count"}:
                continue
            summary[key] = _mean([_safe_float(row.get(key)) for row in group_rows])
        summaries.append(summary)
    return summaries


def write_alignment_geometry_groups(rows, csv_path):
    formatted = [{field: _format_value(row.get(field)) for field in GROUP_FIELDS} for row in rows]
    write_csv_rows(csv_path, formatted, GROUP_FIELDS)
    return Path(csv_path)


def _write_merged(rows, csv_path):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["video_id", "subject_id"]
    formatted = [{field: _format_value(row.get(field)) for field in fieldnames} for row in rows]
    write_csv_rows(csv_path, formatted, fieldnames)
    return Path(csv_path)


def write_alignment_geometry_report(report_path, generated_files, summaries, merged_rows, missing, correlations):
    report_path = Path(report_path)
    ensure_dir(report_path.parent)
    top = correlations[:12]
    max_abs = max((row["abs_correlation"] for row in correlations), default=0.0)
    lines = [
        "# Alignment Geometry Audit Report",
        "",
        f"- OpenFace videos summarized: {len(summaries)}",
        f"- Matched prediction rows: {len(merged_rows)}",
        f"- Missing prediction videos: {len(missing)}",
        f"- Max absolute correlation: {max_abs:.4f}",
        "",
        "## Generated Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in generated_files)
    if missing:
        lines.extend(["", "## Missing Prediction Videos", ""])
        lines.extend(f"- `{item}`" for item in missing[:20])

    lines.extend(["", "## Top Geometry Correlations", ""])
    if top:
        lines.append("| Feature | Target | Correlation |")
        lines.append("|---|---:|---:|")
        for row in top:
            lines.append(f"| {row['feature']} | {row['target']} | {row['correlation']:.4f} |")
    else:
        lines.append("No valid correlations were computed.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Geometry is computed from OpenFace landmark columns `x_*` and `y_*`.",
            "- `face_center_offset_*` is normalized by the configured frame size.",
            "- This audit is offline diagnostics only and must not alter training split, labels, or test-time decisions.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_alignment_geometry_audit(
    predictions_csv,
    openface_root,
    output_dir,
    frame_width=112.0,
    frame_height=112.0,
    sample_step=1,
    max_frames=None,
):
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    summaries = summarize_openface_geometry_root(
        openface_root,
        frame_width=frame_width,
        frame_height=frame_height,
        sample_step=sample_step,
        max_frames=max_frames,
    )
    summary_path = write_alignment_geometry_summary(summaries, tables_dir / "alignment_geometry_summary.csv")
    generated.append(summary_path)

    prediction_rows = read_prediction_table(predictions_csv)
    merged_rows, missing = merge_predictions_with_geometry(prediction_rows, summaries)
    merged_path = _write_merged(merged_rows, tables_dir / "alignment_geometry_merged.csv")
    generated.append(merged_path)

    correlations = compute_alignment_geometry_correlations(merged_rows)
    correlation_path = write_alignment_geometry_correlations(
        correlations,
        tables_dir / "alignment_geometry_correlation.csv",
    )
    generated.append(correlation_path)

    group_rows = summarize_alignment_geometry_groups(merged_rows)
    group_path = write_alignment_geometry_groups(group_rows, tables_dir / "alignment_geometry_group_summary.csv")
    generated.append(group_path)

    report_path = write_alignment_geometry_report(
        reports_dir / "alignment_geometry_audit_report.md",
        generated_files=generated,
        summaries=summaries,
        merged_rows=merged_rows,
        missing=missing,
        correlations=correlations,
    )
    generated.append(report_path)
    return generated
