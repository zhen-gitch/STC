"""Layer-wise identity retrieval aggregation for RPDF Stage A1.

This module is the multi-layer counterpart of
:mod:`src.diagnostics.identity_retrieval`.  It consumes a multi-key embedding
NPZ (one ``features_<layer_name>`` array per probed layer, plus shared
metadata) and runs the existing paired-task retrieval audit independently on
each layer, then aggregates the per-layer summaries into a single comparison
table.

The retrieval logic itself is reused verbatim via
:func:`src.diagnostics.identity_retrieval.compute_identity_retrieval_metrics`;
this module only adds the layer loop and aggregation.

Outputs:

- ``layerwise_identity_summary.csv``: one row per layer with the same aggregate
  metrics as the single-run identity retrieval summary.
- ``layerwise_identity_per_query.csv``: per-query, per-layer retrieval rows for
  case-level analysis (A2 consumes this).
- ``layerwise_identity_report.md``: markdown comparison report.
"""

import math
from pathlib import Path

import numpy as np

from src.diagnostics.identity_retrieval import (
    compute_identity_retrieval_metrics,
    load_embeddings_and_predictions,
)
from src.diagnostics.io import ensure_dir, write_csv_rows


SUMMARY_COLUMNS = [
    "layer_name",
    "num_queries",
    "num_subjects",
    "num_paired_subjects",
    "same_subject_top1_rate",
    "same_subject_top3_rate",
    "same_subject_top5_rate",
    "paired_task_rank_mean",
    "paired_task_rank_median",
    "paired_task_rank_std",
    "paired_task_in_top1_rate",
    "paired_task_in_top3_rate",
    "paired_task_in_top5_rate",
    "severity_neighbor_agreement_top5_mean",
    "severity_neighbor_agreement_top5_std",
    "task_neighbor_agreement_top5_mean",
    "task_neighbor_agreement_top5_std",
]

PER_QUERY_COLUMNS = [
    "layer_name",
    "query_video_id",
    "query_subject_id",
    "query_task_name",
    "query_true_bdi",
    "query_pred_bdi",
    "query_severity_group",
    "top1_neighbor_video_id",
    "top1_neighbor_subject_id",
    "top1_neighbor_task_name",
    "top1_neighbor_same_subject",
    "top3_contains_same_subject",
    "top5_contains_same_subject",
    "paired_task_video_id",
    "paired_task_rank",
    "paired_task_in_top1",
    "paired_task_in_top3",
    "paired_task_in_top5",
    "severity_neighbor_agreement_top5",
    "task_neighbor_agreement_top5",
]


def save_layerwise_features_npz(
    save_path, layer_features, subject_ids, targets, preds, video_ids=None
):
    """Save per-layer embeddings and shared metadata to a compressed NPZ.

    Args:
        save_path: Destination ``.npz`` path.
        layer_features: Dict mapping ``layer_name -> (N, D)`` array.  All layers
            must share the same row count ``N`` and the same row order, which
            matches the dataloader order used by the diagnostic script.
        subject_ids: Subject identifier per video (length N).
        targets: Ground-truth BDI scores (length N).
        preds: Model predictions (length N).
        video_ids: Optional full video identifiers (length N).
    """
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    archive = {
        "subject_ids": np.asarray([str(item) for item in subject_ids]),
        "true_bdi": np.asarray(targets, dtype=float),
        "pred_bdi": np.asarray(preds, dtype=float),
    }
    if video_ids is not None:
        archive["video_ids"] = np.asarray([str(item) for item in video_ids])
    for layer_name, features in layer_features.items():
        archive[f"features_{layer_name}"] = np.asarray(features, dtype=float)
    np.savez_compressed(save_path, **archive)
    return save_path


def load_layerwise_features_npz(features_npz, predictions_csv=None):
    """Load a multi-key layer-wise NPZ.

    Returns:
        Tuple ``(layer_features, records)`` where ``layer_features`` maps
        ``layer_name -> (N, D)`` array and ``records`` is the metadata list
        aligned with row order.  Metadata is enriched from ``predictions_csv``
        when provided (same convention as the single-layer audit).
    """
    data = np.load(features_npz, allow_pickle=True)
    layer_names = [
        key[len("features_"):]
        for key in data.files
        if key.startswith("features_")
    ]

    if not layer_names:
        raise ValueError(
            f"No per-layer feature arrays (keys starting with 'features_') "
            f"found in {features_npz}."
        )

    # Reuse the single-layer loader to build canonical records from the first
    # layer's metadata, then enrich via the predictions CSV if supplied.  We
    # synthesize a transient in-memory NPZ view is unnecessary because
    # load_embeddings_and_predictions reads subject_ids/true_bdi/pred_bdi/video_ids
    # directly; those keys are present in the layer-wise archive too.
    features_first = np.asarray(data[f"features_{layer_names[0]}"], dtype=float)
    records = _build_records_from_npz(data, features_first.shape[0], predictions_csv)

    layer_features = {
        name: np.asarray(data[f"features_{name}"], dtype=float)
        for name in layer_names
    }
    return layer_features, records


def _build_records_from_npz(data, n_rows, predictions_csv):
    """Build metadata records aligned with NPZ row order.

    Mirrors :func:`identity_retrieval.load_embeddings_and_predictions` but
    without requiring a single ``features`` key.
    """
    from src.diagnostics.identity_retrieval import _task_name_from_video_id
    from src.diagnostics.io import read_prediction_table, severity_group

    subject_ids = np.asarray(data["subject_ids"]).astype(str)
    true_bdi = np.asarray(data["true_bdi"], dtype=float)
    pred_bdi = np.asarray(data["pred_bdi"], dtype=float)
    video_ids = (
        np.asarray(data["video_ids"]).astype(str)
        if "video_ids" in data.files
        else subject_ids
    )

    records = []
    for i in range(n_rows):
        video_id = str(video_ids[i])
        true_bdi_i = float(true_bdi[i])
        records.append(
            {
                "video_id": video_id,
                "subject_id": str(subject_ids[i]),
                "task_name": _task_name_from_video_id(video_id),
                "true_bdi": true_bdi_i,
                "pred_bdi": float(pred_bdi[i]),
                "severity_group": severity_group(true_bdi_i),
            }
        )

    if predictions_csv is not None:
        prediction_rows = read_prediction_table(predictions_csv)
        if len(prediction_rows) != n_rows:
            raise ValueError(
                f"Prediction row count ({len(prediction_rows)}) does not match "
                f"feature count ({n_rows})."
            )
        for record, prediction in zip(records, prediction_rows):
            pred_video_id = prediction.get("video_id") or prediction["subject_id"]
            record["video_id"] = str(pred_video_id)
            record["subject_id"] = str(prediction["subject_id"])
            record["task_name"] = prediction.get("task_name") or _task_name_from_video_id(pred_video_id)
            record["true_bdi"] = float(prediction["true_bdi"])
            record["pred_bdi"] = float(prediction["pred_bdi"])
            record["severity_group"] = prediction.get(
                "severity_group", severity_group(prediction["true_bdi"])
            )

    return records


def _format_scalar(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def run_layerwise_identity_audit(
    features_npz,
    output_dir,
    predictions_csv=None,
    top_k=5,
    layer_order=None,
):
    """Run paired-task retrieval on every layer in a layer-wise NPZ.

    Args:
        features_npz: Path to a multi-key NPZ produced by
            :func:`save_layerwise_features_npz`.
        output_dir: Directory where tables and reports are written.
        predictions_csv: Optional prediction CSV for metadata enrichment.
        top_k: Neighborhood size for agreement metrics (default 5).
        layer_order: Optional explicit layer ordering.  When ``None``, layers
            are sorted by their appearance in ``LAYERWISE_PROBE_LAYERS`` and any
            extra layers are appended alphabetically.

    Returns:
        List of generated file paths.
    """
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    layer_features, records = load_layerwise_features_npz(
        features_npz, predictions_csv=predictions_csv
    )

    layer_names = _order_layers(layer_features.keys(), layer_order)

    summary_rows = []
    per_query_rows = []
    for layer_name in layer_names:
        features = layer_features[layer_name]
        if features.shape[0] != len(records):
            raise ValueError(
                f"Layer '{layer_name}' has {features.shape[0]} rows but "
                f"{len(records)} records; row order must match."
            )
        per_query, summary = compute_identity_retrieval_metrics(
            records, features, top_k=top_k
        )

        summary_row = {"layer_name": layer_name}
        summary_row.update(summary)
        summary_rows.append(summary_row)

        for row in per_query:
            row_with_layer = {"layer_name": layer_name}
            row_with_layer.update(row)
            per_query_rows.append(row_with_layer)

    summary_path = tables_dir / "layerwise_identity_summary.csv"
    write_csv_rows(summary_path, summary_rows, SUMMARY_COLUMNS)
    generated.append(summary_path)

    per_query_path = tables_dir / "layerwise_identity_per_query.csv"
    write_csv_rows(per_query_path, per_query_rows, PER_QUERY_COLUMNS)
    generated.append(per_query_path)

    report_path = _write_layerwise_report(
        reports_dir / "layerwise_identity_report.md",
        summary_rows=summary_rows,
        generated_files=generated,
        top_k=top_k,
    )
    generated.append(report_path)
    return generated


def _order_layers(available_layers, layer_order=None):
    """Return layers in canonical probe order, extras appended alphabetically."""
    from src.models.mtl_lite import LAYERWISE_PROBE_LAYERS

    available = set(available_layers)
    ordered = [name for name in LAYERWISE_PROBE_LAYERS if name in available]
    extras = sorted(available - set(ordered))
    if layer_order:
        custom = [name for name in layer_order if name in available and name not in ordered]
        ordered.extend(custom)
    ordered.extend(extras)
    return ordered


def _write_layerwise_report(report_path, summary_rows, generated_files, top_k=5):
    """Write a markdown report comparing identity retrieval across layers."""
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    lines = [
        "# Layer-wise Identity Retrieval Report",
        "",
        "This report compares same-subject retrieval, paired-task rank, and "
        "neighborhood agreement across model layers. It supports RPDF Stage A1: "
        "locating where subject identity becomes separable in the backbone / "
        "temporal / shared representation stack.",
        "",
        f"Neighborhood size (top-k): {top_k}",
        "",
        "## Per-Layer Summary",
        "",
        "| Layer | same_top1 | same_top5 | paired_rank_mean | paired_in_top5 | severity_agree | task_agree |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['layer_name']} | "
            f"{_format_scalar(row.get('same_subject_top1_rate'))} | "
            f"{_format_scalar(row.get('same_subject_top5_rate'))} | "
            f"{_format_scalar(row.get('paired_task_rank_mean'))} | "
            f"{_format_scalar(row.get('paired_task_in_top5_rate'))} | "
            f"{_format_scalar(row.get('severity_neighbor_agreement_top5_mean'))} | "
            f"{_format_scalar(row.get('task_neighbor_agreement_top5_mean'))} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- **same_top1 / same_top5**: Same-subject retrieval rate. High values "
            "at a layer mean the embedding at that layer encodes subject identity / "
            "static appearance.",
            "- **paired_rank_mean**: Mean rank of the paired Freeform/Northwind "
            "video among all neighbors (1 = best). Lower is stronger identity memory.",
            "- **severity_agree**: Fraction of top-5 neighbors sharing the query's "
            "severity group. Low values indicate the embedding is identity-driven "
            "rather than severity-driven.",
            "- If identity retrieval is already strong at middle/lower backbone "
            "blocks, the `z_id` outlet should attach early; if it grows toward the "
            "shared representation, identity is exploited by the upper head.",
            "- This audit only proves identity *exists* in the embedding; A2 "
            "(error-identity coupling) is required to prove the prediction *uses* it.",
            "",
            "## Generated Files",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in generated_files)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
