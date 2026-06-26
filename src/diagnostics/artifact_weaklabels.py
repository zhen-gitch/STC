"""RPDF Stage A3: artifact weak-label audit for the ``z_art`` factor.

This module aggregates the artifact / quality / geometry / temporal weak labels
already produced by the P0 diagnostics into a single view, then correlates each
weak label with ``true_bdi``, ``pred_bdi``, ``residual`` and ``abs_error``.  The
result decides whether ``z_art`` should enter the first RPDF-lite version as a
supervised outlet, or remain an attack / evaluation signal only.

It does *not* recompute any weak labels -- it consumes the existing summary CSVs:

- ``black_artifacts``: black_ratio, black_border_ratio, black_center_ratio, ...
- ``alignment_geometry``: landmark_bbox, face_center_offset, eye_distance, ...
- ``openface_quality``: confidence, success_ratio, pose, gaze, ...
- ``temporal_sampling``: frame_count, valid_ratio, padding_ratio, truncated_ratio

Outputs:

- ``artifact_weaklabel_summary.csv``: per-video weak labels joined with
  prediction error.
- ``artifact_weaklabel_correlation.csv``: per-weak-label correlation with the
  four prediction targets.
- ``artifact_weaklabel_report.md``.
"""

import math
from pathlib import Path

from src.diagnostics.io import ensure_dir, read_csv_rows, write_csv_rows


# Weak-label fields to extract from each source summary CSV.  These are the
# numeric per-video statistics produced by the P0 diagnostics; metadata columns
# (video_id / subject_id / task_name / source_*) are joined separately.
WEAKLABEL_SOURCES = {
    "black_artifacts": {
        "fields": [
            "black_ratio_mean",
            "black_ratio_std",
            "black_border_ratio_mean",
            "black_center_ratio_mean",
            "black_boundary_edge_ratio_mean",
            "black_ratio_delta_mean",
        ],
    },
    "alignment_geometry": {
        "fields": [
            "confidence_mean",
            "success_ratio",
            "landmark_bbox_width_mean",
            "landmark_bbox_height_mean",
            "landmark_bbox_area_mean",
            "landmark_bbox_aspect_mean",
            "face_center_offset_x_mean",
            "face_center_offset_y_mean",
            "eye_distance_mean",
            "normalized_face_scale_mean",
            "landmark_jitter_mean",
            "landmark_jitter_std",
        ],
    },
    "openface_quality": {
        # openface_quality fieldnames are dynamic; we collect any numeric field
        # not in the metadata set below at runtime, so this list is a fallback.
        "fields": [
            "confidence_mean",
            "confidence_std",
            "confidence_min",
            "success_ratio",
        ],
    },
    "temporal_sampling": {
        "fields": [
            "frame_count",
            "sampled_frame_count",
            "selected_frame_count",
            "truncated_frame_count",
            "padding_frame_count",
            "valid_ratio",
            "padding_ratio",
            "truncated_ratio",
        ],
    },
}

METADATA_FIELDS = {"video_id", "subject_id", "task_name", "source_dir", "source_file"}

TARGET_FIELDS = ("true_bdi", "pred_bdi", "residual", "abs_error")

SUMMARY_METADATA_COLUMNS = [
    "video_id",
    "subject_id",
    "task_name",
    "true_bdi",
    "pred_bdi",
    "residual",
    "abs_error",
    "severity_group",
]

CORRELATION_COLUMNS = [
    "weaklabel_name",
    "source",
    "n",
    "corr_with_true_bdi",
    "corr_with_pred_bdi",
    "corr_with_residual",
    "corr_with_abs_error",
    "abs_corr_with_abs_error",
]


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _format_scalar(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _normalize_video_id(video_id):
    """Normalize a video id for cross-source joining.

    Mirrors :func:`alignment_geometry.normalize_video_id` so that ``_aligned``
    suffixes and ``.csv`` extensions do not break the join.  Implemented here
    to avoid importing a private helper from another diagnostic module.
    """
    video_id = str(video_id or "")
    if video_id.endswith(".csv"):
        video_id = Path(video_id).stem
    if video_id.endswith("_aligned"):
        video_id = video_id[: -len("_aligned")]
    return video_id


def _pearson(a, b):
    """Pearson correlation between two equal-length lists of floats.

    Returns ``None`` when fewer than 2 paired values or zero variance.
    """
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    am = [x - mx for x in xs]
    bm = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in am) * sum(y * y for y in bm))
    if denom < 1e-12:
        return None
    return sum(x * y for x, y in zip(am, bm)) / denom


def _load_source_summary(csv_path):
    """Load a source summary CSV into a dict keyed by normalized video_id."""
    if csv_path is None:
        return {}
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}
    rows = read_csv_rows(csv_path)
    by_video = {}
    for row in rows:
        raw_id = row.get("video_id") or row.get("source_file") or row.get("source_dir")
        if not raw_id:
            continue
        by_video[_normalize_video_id(raw_id)] = row
    return by_video


def _load_predictions(predictions_csv):
    """Load prediction rows keyed by normalized video_id."""
    rows = read_csv_rows(predictions_csv)
    by_video = {}
    for row in rows:
        raw_id = row.get("video_id") or row.get("subject_id")
        if raw_id:
            by_video[_normalize_video_id(raw_id)] = row
    return by_video


def _collect_weaklabel_fields(source_name, source_rows):
    """Return the weak-label fields actually present *with data* in a source.

    A source with no matched rows returns an empty list, so it contributes no
    correlation rows -- this avoids emitting zero-count weak labels for sources
    the caller did not supply.
    """
    if not source_rows:
        return []
    declared = WEAKLABEL_SOURCES.get(source_name, {}).get("fields", [])
    if source_name == "openface_quality":
        # openface_quality has dynamic fieldnames; gather any non-metadata field.
        seen = []
        for row in source_rows.values():
            for key in row:
                if key in METADATA_FIELDS or key in TARGET_FIELDS:
                    continue
                if key not in seen:
                    seen.append(key)
        # Keep declared first, then extras, deduped.
        merged = []
        for field in declared + seen:
            if field not in merged:
                merged.append(field)
        return merged
    # For static-schema sources, only keep declared fields that actually appear
    # in at least one row (handles partial summary CSVs).
    present = set()
    for row in source_rows.values():
        present.update(row.keys())
    return [field for field in declared if field in present]


def build_weaklabel_summary(
    predictions_csv,
    black_artifacts_csv=None,
    alignment_geometry_csv=None,
    openface_quality_csv=None,
    temporal_sampling_csv=None,
):
    """Join prediction targets with all available weak-label sources.

    Returns:
        Tuple ``(summary_rows, weaklabel_fields_by_source)`` where
        ``summary_rows`` is a list of per-video dicts and
        ``weaklabel_fields_by_source`` maps source name -> list of weak-label
        field names actually present.
    """
    sources = {
        "black_artifacts": _load_source_summary(black_artifacts_csv),
        "alignment_geometry": _load_source_summary(alignment_geometry_csv),
        "openface_quality": _load_source_summary(openface_quality_csv),
        "temporal_sampling": _load_source_summary(temporal_sampling_csv),
    }
    predictions = _load_predictions(predictions_csv)

    weaklabel_fields_by_source = {
        name: _collect_weaklabel_fields(name, rows) for name, rows in sources.items()
    }

    # Union of all normalized video ids across prediction + sources.
    all_ids = set(predictions.keys())
    for rows in sources.values():
        all_ids.update(rows.keys())

    summary_rows = []
    for video_id in sorted(all_ids):
        pred = predictions.get(video_id, {})
        row = {
            "video_id": video_id,
            "subject_id": pred.get("subject_id", ""),
            "task_name": pred.get("task_name", ""),
            "true_bdi": _safe_float(pred.get("true_bdi")),
            "pred_bdi": _safe_float(pred.get("pred_bdi")),
            "residual": _safe_float(pred.get("residual")),
            "abs_error": _safe_float(pred.get("abs_error")),
            "severity_group": pred.get("severity_group", ""),
        }
        # Fill residual/abs_error when missing but pred+true are available.
        if row["residual"] is None and row["pred_bdi"] is not None and row["true_bdi"] is not None:
            row["residual"] = row["pred_bdi"] - row["true_bdi"]
        if row["abs_error"] is None and row["residual"] is not None:
            row["abs_error"] = abs(row["residual"])

        for source_name, source_rows in sources.items():
            source_row = source_rows.get(video_id, {})
            for field in weaklabel_fields_by_source[source_name]:
                # Prefix with source name to avoid collisions (e.g. confidence_mean
                # appears in both alignment_geometry and openface_quality).
                key = f"{source_name}:{field}"
                row[key] = _safe_float(source_row.get(field))
        summary_rows.append(row)

    return summary_rows, weaklabel_fields_by_source


def compute_weaklabel_correlations(summary_rows, weaklabel_fields_by_source):
    """Correlate every weak label with the four prediction targets.

    Returns a list of correlation rows (one per weak label).
    """
    targets = {target: [r.get(target) for r in summary_rows] for target in TARGET_FIELDS}
    rows = []
    for source_name, fields in weaklabel_fields_by_source.items():
        for field in fields:
            key = f"{source_name}:{field}"
            values = [r.get(key) for r in summary_rows]
            n = sum(1 for v in values if v is not None)
            corr_abs_error = _pearson(values, targets["abs_error"])
            row = {
                "weaklabel_name": key,
                "source": source_name,
                "n": n,
                "corr_with_true_bdi": _pearson(values, targets["true_bdi"]),
                "corr_with_pred_bdi": _pearson(values, targets["pred_bdi"]),
                "corr_with_residual": _pearson(values, targets["residual"]),
                "corr_with_abs_error": corr_abs_error,
                "abs_corr_with_abs_error": abs(corr_abs_error) if corr_abs_error is not None else None,
            }
            rows.append(row)
    # Sort by absolute correlation with abs_error (strongest error coupling first).
    rows.sort(
        key=lambda r: (r["abs_corr_with_abs_error"] is None, -(r["abs_corr_with_abs_error"] or 0))
    )
    return rows


def _write_weaklabel_report(
    report_path,
    correlation_rows,
    weaklabel_fields_by_source,
    matched_video_count,
    generated_files,
    coupling_threshold=0.2,
):
    """Write the markdown report with z_art entry-decision interpretation."""
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    error_coupled = [
        r for r in correlation_rows
        if r["abs_corr_with_abs_error"] is not None
        and r["abs_corr_with_abs_error"] >= coupling_threshold
    ]
    label_only = [
        r for r in correlation_rows
        if r["abs_corr_with_abs_error"] is not None
        and r["abs_corr_with_abs_error"] < coupling_threshold
        and r["corr_with_true_bdi"] is not None
        and abs(r["corr_with_true_bdi"]) >= coupling_threshold
    ]

    lines = [
        "# Artifact Weak-label Audit Report (RPDF Stage A3)",
        "",
        f"Matched videos: {matched_video_count}",
        "",
        "This audit joins the existing black-artifact, alignment-geometry, "
        "OpenFace-quality and temporal-sampling summaries with prediction error, "
        "then correlates each weak label with true_bdi, pred_bdi, residual and "
        "abs_error. The result decides whether `z_art` enters the first RPDF-lite "
        "version as a supervised outlet.",
        "",
        "## Weak-label Correlation (sorted by |corr with abs_error|)",
        "",
        "| weaklabel | source | n | corr(true_bdi) | corr(pred_bdi) | corr(residual) | corr(abs_error) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in correlation_rows:
        lines.append(
            f"| {row['weaklabel_name']} | {row['source']} | {row['n']} | "
            f"{_format_scalar(row.get('corr_with_true_bdi'))} | "
            f"{_format_scalar(row.get('corr_with_pred_bdi'))} | "
            f"{_format_scalar(row.get('corr_with_residual'))} | "
            f"{_format_scalar(row.get('corr_with_abs_error'))} |"
        )

    lines.extend(
        [
            "",
            "## z_art Entry Decision",
            "",
            f"Coupling threshold (|corr with abs_error|): {coupling_threshold}",
            "",
            f"- Weak labels coupled with prediction error (|corr(abs_error)| >= {coupling_threshold}): "
            f"{len(error_coupled)}",
        ]
    )
    for row in error_coupled:
        lines.append(
            f"  - `{row['weaklabel_name']}`: corr(abs_error)={_format_scalar(row.get('corr_with_abs_error'))}"
        )
    lines.append(
        f"- Weak labels coupled with true_bdi only (acquisition/label confound, not error): "
        f"{len(label_only)}"
    )
    for row in label_only:
        lines.append(
            f"  - `{row['weaklabel_name']}`: corr(true_bdi)={_format_scalar(row.get('corr_with_true_bdi'))}"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A weak label with |corr(abs_error)| >= threshold enters the `z_art` "
            "supervised outlet in RPDF-lite v1: the artifact participates in the "
            "error pattern and should be absorbed by `z_art`.",
            "- A weak label correlated with true_bdi but not abs_error indicates an "
            "acquisition / collection bias confound (artifact co-varies with depression "
            "label itself). Such labels should be `z_art` attack / evaluation signals, "
            "not training supervision, to avoid leaking label information.",
            "- If no weak label clears the threshold, `z_art` v1 stays an audit / "
            "evaluation outlet only and does not enter the training loss.",
            "",
            "## Sources Covered",
            "",
        ]
    )
    for source, fields in weaklabel_fields_by_source.items():
        lines.append(f"- `{source}`: {len(fields)} weak-label fields")
    lines.extend(["", "## Generated Files", ""])
    lines.extend(f"- `{path}`" for path in generated_files)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_artifact_weaklabel_audit(
    predictions_csv,
    output_dir,
    black_artifacts_csv=None,
    alignment_geometry_csv=None,
    openface_quality_csv=None,
    temporal_sampling_csv=None,
    coupling_threshold=0.2,
):
    """Run the A3 artifact weak-label audit end-to-end.

    Args:
        predictions_csv: Prediction CSV with residual/abs_error (required).
        output_dir: Directory for outputs.
        black_artifacts_csv: Optional black_artifacts summary CSV.
        alignment_geometry_csv: Optional alignment_geometry summary CSV.
        openface_quality_csv: Optional openface_quality summary CSV.
        temporal_sampling_csv: Optional temporal_sampling summary CSV.
        coupling_threshold: |corr with abs_error| above which a weak label is
            considered coupled with prediction error (default 0.2).

    Returns:
        List of generated file paths.
    """
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    summary_rows, weaklabel_fields_by_source = build_weaklabel_summary(
        predictions_csv=predictions_csv,
        black_artifacts_csv=black_artifacts_csv,
        alignment_geometry_csv=alignment_geometry_csv,
        openface_quality_csv=openface_quality_csv,
        temporal_sampling_csv=temporal_sampling_csv,
    )

    if not summary_rows:
        raise RuntimeError(
            "No videos matched across the prediction CSV and weak-label sources. "
            "Check that video_id normalization is consistent."
        )

    # Summary CSV: metadata + all weak-label columns (dynamic).
    summary_columns = list(SUMMARY_METADATA_COLUMNS)
    for source_name, fields in weaklabel_fields_by_source.items():
        for field in fields:
            key = f"{source_name}:{field}"
            if key not in summary_columns:
                summary_columns.append(key)

    summary_path = tables_dir / "artifact_weaklabel_summary.csv"
    write_csv_rows(summary_path, summary_rows, summary_columns)
    generated.append(summary_path)

    correlation_rows = compute_weaklabel_correlations(
        summary_rows, weaklabel_fields_by_source
    )
    corr_path = tables_dir / "artifact_weaklabel_correlation.csv"
    write_csv_rows(corr_path, correlation_rows, CORRELATION_COLUMNS)
    generated.append(corr_path)

    matched = sum(
        1 for r in summary_rows if r.get("true_bdi") is not None
    )
    report_path = _write_weaklabel_report(
        reports_dir / "artifact_weaklabel_report.md",
        correlation_rows=correlation_rows,
        weaklabel_fields_by_source=weaklabel_fields_by_source,
        matched_video_count=matched,
        generated_files=generated,
        coupling_threshold=coupling_threshold,
    )
    generated.append(report_path)
    return generated
