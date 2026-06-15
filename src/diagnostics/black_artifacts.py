import math
from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, read_prediction_table, write_csv_rows


TARGET_COLUMNS = ("true_bdi", "pred_bdi", "residual", "abs_error")
BLACK_ARTIFACT_COLUMNS = [
    "video_id",
    "source_dir",
    "frame_count",
    "sampled_frame_count",
    "black_ratio_mean",
    "black_ratio_std",
    "black_border_ratio_mean",
    "black_center_ratio_mean",
    "black_boundary_edge_ratio_mean",
    "black_ratio_delta_mean",
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
    return left == right or f"{left}_aligned" == right or left == f"{right}_aligned"


def _find_video_dir(image_root, video_id):
    image_root = Path(image_root).expanduser()
    candidates = [
        image_root / str(video_id),
        image_root / normalize_video_id(video_id),
        image_root / f"{normalize_video_id(video_id)}_aligned",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    normalized = normalize_video_id(video_id)
    if image_root.exists():
        for candidate in image_root.iterdir():
            if candidate.is_dir() and _compatible_video_ids(candidate.name, normalized):
                return candidate
    return None


def _frame_paths(video_dir, sample_step=10, max_frames=None):
    paths = sorted(Path(video_dir).glob("*.jpg"))
    if sample_step > 1:
        paths = paths[:: int(sample_step)]
    if max_frames is not None:
        paths = paths[: int(max_frames)]
    return paths


def _border_mask(height, width, border_fraction=0.15):
    border_y = max(1, int(round(height * border_fraction)))
    border_x = max(1, int(round(width * border_fraction)))
    mask = np.zeros((height, width), dtype=bool)
    mask[:border_y, :] = True
    mask[-border_y:, :] = True
    mask[:, :border_x] = True
    mask[:, -border_x:] = True
    return mask


def _black_boundary_edge_ratio(black_mask):
    horizontal = black_mask[:, 1:] != black_mask[:, :-1]
    vertical = black_mask[1:, :] != black_mask[:-1, :]
    edge_count = horizontal.size + vertical.size
    if edge_count == 0:
        return 0.0
    return float((horizontal.sum() + vertical.sum()) / edge_count)


def _read_black_mask(frame_path, black_threshold=8):
    from torchvision.io import ImageReadMode, read_image

    image = read_image(str(frame_path), mode=ImageReadMode.RGB)
    values = image.detach().cpu().numpy()
    return np.all(values <= int(black_threshold), axis=0)


def summarize_video_black_artifacts(
    video_dir,
    video_id=None,
    black_threshold=8,
    border_fraction=0.15,
    sample_step=10,
    max_frames=None,
):
    video_dir = Path(video_dir)
    video_id = video_id or video_dir.name
    all_frames = sorted(video_dir.glob("*.jpg"))
    sampled_frames = _frame_paths(video_dir, sample_step=sample_step, max_frames=max_frames)

    black_ratios = []
    border_ratios = []
    center_ratios = []
    boundary_edges = []
    previous_black_ratio = None
    black_ratio_deltas = []

    for frame_path in sampled_frames:
        black_mask = _read_black_mask(frame_path, black_threshold=black_threshold)
        height, width = black_mask.shape
        border = _border_mask(height, width, border_fraction=border_fraction)
        center = ~border

        black_ratio = float(black_mask.mean())
        black_ratios.append(black_ratio)
        border_ratios.append(float(black_mask[border].mean()) if border.any() else 0.0)
        center_ratios.append(float(black_mask[center].mean()) if center.any() else 0.0)
        boundary_edges.append(_black_boundary_edge_ratio(black_mask))
        if previous_black_ratio is not None:
            black_ratio_deltas.append(abs(black_ratio - previous_black_ratio))
        previous_black_ratio = black_ratio

    def _mean(values):
        return float(np.mean(values)) if values else None

    def _std(values):
        return float(np.std(values)) if values else None

    return {
        "video_id": str(video_id),
        "source_dir": str(video_dir),
        "frame_count": len(all_frames),
        "sampled_frame_count": len(sampled_frames),
        "black_ratio_mean": _mean(black_ratios),
        "black_ratio_std": _std(black_ratios),
        "black_border_ratio_mean": _mean(border_ratios),
        "black_center_ratio_mean": _mean(center_ratios),
        "black_boundary_edge_ratio_mean": _mean(boundary_edges),
        "black_ratio_delta_mean": _mean(black_ratio_deltas),
    }


def summarize_prediction_videos(
    predictions_csv,
    image_root,
    black_threshold=8,
    border_fraction=0.15,
    sample_step=10,
    max_frames=None,
):
    prediction_rows = read_prediction_table(predictions_csv)
    summaries = []
    missing = []
    seen = set()
    for row in prediction_rows:
        video_id = row.get("video_id") or row["subject_id"]
        normalized = normalize_video_id(video_id)
        if normalized in seen:
            continue
        seen.add(normalized)
        video_dir = _find_video_dir(image_root, video_id)
        if video_dir is None:
            missing.append(video_id)
            continue
        summaries.append(
            summarize_video_black_artifacts(
                video_dir,
                video_id=normalized,
                black_threshold=black_threshold,
                border_fraction=border_fraction,
                sample_step=sample_step,
                max_frames=max_frames,
            )
        )
    return summaries, missing


def write_black_artifact_summary(rows, csv_path):
    formatted = [{key: _format_value(row.get(key)) for key in BLACK_ARTIFACT_COLUMNS} for row in rows]
    write_csv_rows(csv_path, formatted, BLACK_ARTIFACT_COLUMNS)
    return Path(csv_path)


def merge_predictions_with_black_artifacts(prediction_rows, artifact_rows):
    by_video = {normalize_video_id(row.get("video_id")): row for row in artifact_rows}
    merged = []
    for prediction in prediction_rows:
        artifact = by_video.get(normalize_video_id(prediction.get("video_id")))
        if artifact is None:
            continue
        row = dict(prediction)
        for key, value in artifact.items():
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


def compute_black_artifact_correlations(rows):
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


def write_black_artifact_correlations(rows, csv_path):
    fields = ["feature", "target", "correlation", "abs_correlation", "count"]
    formatted = [{key: _format_value(row.get(key)) for key in fields} for row in rows]
    write_csv_rows(csv_path, formatted, fields)
    return Path(csv_path)


def write_black_artifact_report(report_path, generated_files, summaries, missing, merged_rows, correlations):
    report_path = Path(report_path)
    ensure_dir(report_path.parent)
    top = correlations[:12]
    max_abs = max((row["abs_correlation"] for row in correlations), default=0.0)

    lines = [
        "# Black Artifact Audit Report",
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

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Black pixels are detected when all RGB channels are below the configured threshold.",
            "- This audit is offline diagnostics only and must not alter training split, labels, or test-time decisions.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_black_artifact_audit(
    predictions_csv,
    image_root,
    output_dir,
    black_threshold=8,
    border_fraction=0.15,
    sample_step=10,
    max_frames=None,
):
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    summaries, missing = summarize_prediction_videos(
        predictions_csv=predictions_csv,
        image_root=image_root,
        black_threshold=black_threshold,
        border_fraction=border_fraction,
        sample_step=sample_step,
        max_frames=max_frames,
    )
    summary_path = write_black_artifact_summary(summaries, tables_dir / "black_artifact_summary.csv")
    generated.append(summary_path)

    prediction_rows = read_prediction_table(predictions_csv)
    merged_rows = merge_predictions_with_black_artifacts(prediction_rows, summaries)
    merged_fields = []
    for row in merged_rows:
        for key in row:
            if key not in merged_fields:
                merged_fields.append(key)
    if not merged_fields:
        merged_fields = ["video_id", "subject_id"]
    merged_path = tables_dir / "black_artifact_merged.csv"
    write_csv_rows(
        merged_path,
        [{key: _format_value(row.get(key)) for key in merged_fields} for row in merged_rows],
        merged_fields,
    )
    generated.append(merged_path)

    correlations = compute_black_artifact_correlations(merged_rows)
    correlation_path = write_black_artifact_correlations(
        correlations,
        tables_dir / "black_artifact_correlation.csv",
    )
    generated.append(correlation_path)

    report_path = write_black_artifact_report(
        reports_dir / "black_artifact_audit_report.md",
        generated_files=generated,
        summaries=summaries,
        missing=missing,
        merged_rows=merged_rows,
        correlations=correlations,
    )
    generated.append(report_path)
    return generated
