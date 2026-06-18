"""Embedding identity retrieval audit for RGB/MTL-Lite models.

This module implements paired-task retrieval diagnostics to check whether the
video-level embeddings learned by the RGB model mainly encode subject identity
and static appearance rather than depression severity.  For each query video we
rank all other test videos by embedding similarity and report:

- same-subject top-k retrieval rates
- rank of the paired Freeform/Northwind video for the same subject
- severity / task neighbor agreement in the local embedding neighborhood

The audit is offline and read-only: it consumes a `.npz` file produced by
`scripts/diagnose_mtl_lite.py` (or an equivalent prediction CSV) and writes
 tables, figures, and a markdown report.
"""

import math
from pathlib import Path

import numpy as np

from src.diagnostics.black_artifacts import normalize_video_id
from src.diagnostics.io import ensure_dir, read_prediction_table, severity_group, write_csv_rows


# Columns written to the per-query retrieval table.
RETRIEVAL_COLUMNS = [
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

# Columns written to the summary table.
SUMMARY_COLUMNS = [
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


def _task_name_from_video_id(video_id):
    """Infer task name from a video identifier.

    AVEC2014 video IDs conventionally contain ``Freeform`` or ``Northwind``.
    If neither token is present, return an empty string.
    """
    text = str(video_id or "")
    if "Freeform" in text:
        return "Freeform"
    if "Northwind" in text:
        return "Northwind"
    return ""


def _safe_float(value):
    """Convert a scalar to float, returning None for invalid / non-finite inputs."""
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _format_value(value):
    """Format a scalar for CSV output; empty string represents missing values."""
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


def _load_npz_data(features_npz):
    """Load embeddings and metadata from a feature archive.

    Returns:
        A dictionary with ``features``, ``subject_ids``, ``true_bdi``,
        ``pred_bdi`` and optionally ``video_ids``.  If the archive does not
        contain ``video_ids``, they are reconstructed from the prediction CSV
        fallback in :func:`load_embeddings_and_predictions`.
    """
    data = np.load(features_npz, allow_pickle=True)
    result = {
        "features": np.asarray(data["features"], dtype=float),
        "subject_ids": np.asarray(data["subject_ids"]).astype(str),
        "true_bdi": np.asarray(data["true_bdi"], dtype=float),
        "pred_bdi": np.asarray(data["pred_bdi"], dtype=float),
    }
    if "video_ids" in data.files:
        result["video_ids"] = np.asarray(data["video_ids"]).astype(str)
    return result


def load_embeddings_and_predictions(features_npz, predictions_csv=None):
    """Load embeddings and align them with prediction metadata.

    If ``features_npz`` already stores ``video_ids`` they are used directly.
    Otherwise, when ``predictions_csv`` is provided, its rows are aligned by
    row order (both the diagnostic script and the prediction table are written
    in dataloader order).  The predictions CSV is also the fallback source for
    ``task_name`` when the NPZ does not contain it.

    Args:
        features_npz: Path to the compressed embedding archive.
        predictions_csv: Optional path to a ``test_predictions.csv`` style file.

    Returns:
        Tuple ``(features, records)`` where ``records`` is a list of dicts with
        keys ``video_id``, ``subject_id``, ``task_name``, ``true_bdi``,
        ``pred_bdi`` and ``severity_group``.
    """
    npz_data = _load_npz_data(features_npz)
    features = npz_data["features"]
    n = features.shape[0]

    # Build base records from NPZ metadata.
    records = []
    for i in range(n):
        video_id = (
            npz_data["video_ids"][i]
            if "video_ids" in npz_data
            else npz_data["subject_ids"][i]
        )
        true_bdi = float(npz_data["true_bdi"][i])
        records.append(
            {
                "video_id": str(video_id),
                "subject_id": str(npz_data["subject_ids"][i]),
                "task_name": _task_name_from_video_id(video_id),
                "true_bdi": true_bdi,
                "pred_bdi": float(npz_data["pred_bdi"][i]),
                "severity_group": severity_group(true_bdi),
            }
        )

    # If a predictions CSV is supplied, enrich / verify metadata by video ID.
    if predictions_csv is not None:
        prediction_rows = read_prediction_table(predictions_csv)
        if len(prediction_rows) != n:
            raise ValueError(
                f"Prediction row count ({len(prediction_rows)}) does not match "
                f"feature count ({n})."
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

    return features, records


def _cosine_similarity_matrix(features):
    """Compute pairwise cosine similarities for L2-normalized embeddings.

    Self-similarities are set to -1 so a query is never its own nearest
    neighbor.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    zero_mask = norms.ravel() < 1e-12
    normalized = np.divide(
        features,
        norms,
        out=np.zeros_like(features),
        where=norms > 1e-12,
    )
    similarities = normalized @ normalized.T
    # Ensure exact symmetry and mask self-similarity.
    similarities = (similarities + similarities.T) / 2.0
    np.fill_diagonal(similarities, -1.0)
    # Treat zero-norm rows as dissimilar to everything.
    similarities[zero_mask, :] = -1.0
    similarities[:, zero_mask] = -1.0
    return similarities


def _subject_index_groups(records):
    """Group record indices by ``subject_id``.

    Returns a dict mapping subject ID to the list of record indices belonging
    to that subject.  For AVEC2014 each subject should have two videos
    (Freeform and Northwind).
    """
    groups = {}
    for idx, record in enumerate(records):
        groups.setdefault(record["subject_id"], []).append(idx)
    return groups


def _paired_task_index(query_idx, records, subject_groups):
    """Return the index of the paired Freeform/Northwind video for a query.

    If the subject has exactly two records and the other record has a different
    task name, return its index.  Otherwise return ``None``.
    """
    subject_id = records[query_idx]["subject_id"]
    group = subject_groups.get(subject_id, [])
    if len(group) != 2:
        return None
    other_idx = group[0] if group[1] == query_idx else group[1]
    if records[other_idx]["task_name"] == records[query_idx]["task_name"]:
        return None
    return other_idx


def _rank_of_target(similarities_row, target_idx):
    """Return the 1-based rank of ``target_idx`` when sorting by descending similarity.

    In the unlikely event of ties the NumPy ``argsort`` order is stable enough
    for diagnostic purposes; we do not implement random tie-breaking.
    """
    order = np.argsort(-similarities_row)
    position = np.where(order == target_idx)[0]
    if position.size == 0:
        return None
    return int(position[0]) + 1


def _topk_contains(topk_indices, target_idx):
    """Return ``True`` if ``target_idx`` is in the provided top-k index array."""
    return int(target_idx in topk_indices)


def compute_identity_retrieval_metrics(records, features, top_k=5):
    """Compute paired-task retrieval and neighborhood-agreement metrics.

    Args:
        records: List of metadata records aligned with ``features`` rows.
        features: Array of shape ``(N, D)`` with video-level embeddings.
        top_k: Size of the local neighborhood used for agreement metrics.

    Returns:
        Tuple ``(per_query_rows, summary)``.
    """
    features = np.asarray(features, dtype=float)
    n = features.shape[0]
    if n != len(records):
        raise ValueError("Feature row count must match record count.")

    similarities = _cosine_similarity_matrix(features)
    subject_groups = _subject_index_groups(records)

    per_query_rows = []
    paired_ranks = []
    paired_in_top1 = []
    paired_in_top3 = []
    paired_in_top5 = []

    for query_idx in range(n):
        query_record = records[query_idx]
        sim_row = similarities[query_idx]

        # Exclude the query itself; nearest neighbors are the highest similarities.
        neighbor_order = np.argsort(-sim_row)
        top1_idx = int(neighbor_order[0])
        top3_indices = set(neighbor_order[:3].tolist())
        top5_indices = set(neighbor_order[:5].tolist())
        top5_records = [records[idx] for idx in neighbor_order[:5]]

        top1_record = records[top1_idx]
        same_subject_top1 = int(top1_record["subject_id"] == query_record["subject_id"])
        same_subject_top3 = int(
            any(
                records[idx]["subject_id"] == query_record["subject_id"]
                for idx in neighbor_order[:3]
            )
        )
        same_subject_top5 = int(
            any(
                records[idx]["subject_id"] == query_record["subject_id"]
                for idx in neighbor_order[:5]
            )
        )

        paired_idx = _paired_task_index(query_idx, records, subject_groups)
        paired_video_id = ""
        paired_rank = None
        paired_in_top1_val = 0
        paired_in_top3_val = 0
        paired_in_top5_val = 0
        if paired_idx is not None:
            paired_video_id = records[paired_idx]["video_id"]
            paired_rank = _rank_of_target(sim_row, paired_idx)
            paired_in_top1_val = int(top1_idx == paired_idx)
            paired_in_top3_val = int(paired_idx in top3_indices)
            paired_in_top5_val = int(paired_idx in top5_indices)
            if paired_rank is not None:
                paired_ranks.append(paired_rank)
                paired_in_top1.append(paired_in_top1_val)
                paired_in_top3.append(paired_in_top3_val)
                paired_in_top5.append(paired_in_top5_val)

        # Local neighborhood agreement: fraction of top-5 neighbors sharing the
        # query's severity group or task name.
        query_severity = query_record["severity_group"]
        query_task = query_record["task_name"]
        severity_agreements = [
            int(neighbor["severity_group"] == query_severity) for neighbor in top5_records
        ]
        task_agreements = [
            int(neighbor["task_name"] == query_task) for neighbor in top5_records
            if query_task
        ]
        severity_agreement_mean = float(np.mean(severity_agreements)) if severity_agreements else None
        task_agreement_mean = float(np.mean(task_agreements)) if task_agreements else None

        per_query_rows.append(
            {
                "query_video_id": query_record["video_id"],
                "query_subject_id": query_record["subject_id"],
                "query_task_name": query_record["task_name"],
                "query_true_bdi": query_record["true_bdi"],
                "query_pred_bdi": query_record["pred_bdi"],
                "query_severity_group": query_record["severity_group"],
                "top1_neighbor_video_id": top1_record["video_id"],
                "top1_neighbor_subject_id": top1_record["subject_id"],
                "top1_neighbor_task_name": top1_record["task_name"],
                "top1_neighbor_same_subject": same_subject_top1,
                "top3_contains_same_subject": same_subject_top3,
                "top5_contains_same_subject": same_subject_top5,
                "paired_task_video_id": paired_video_id,
                "paired_task_rank": paired_rank,
                "paired_task_in_top1": paired_in_top1_val,
                "paired_task_in_top3": paired_in_top3_val,
                "paired_task_in_top5": paired_in_top5_val,
                "severity_neighbor_agreement_top5": severity_agreement_mean,
                "task_neighbor_agreement_top5": task_agreement_mean,
            }
        )

    # Aggregate summary statistics over all queries.
    same_subject_top1_rates = [row["top1_neighbor_same_subject"] for row in per_query_rows]
    same_subject_top3_rates = [row["top3_contains_same_subject"] for row in per_query_rows]
    same_subject_top5_rates = [row["top5_contains_same_subject"] for row in per_query_rows]
    severity_agreements = [
        row["severity_neighbor_agreement_top5"]
        for row in per_query_rows
        if row["severity_neighbor_agreement_top5"] is not None
    ]
    task_agreements = [
        row["task_neighbor_agreement_top5"]
        for row in per_query_rows
        if row["task_neighbor_agreement_top5"] is not None
    ]

    num_paired_subjects = sum(1 for group in subject_groups.values() if len(group) == 2)

    summary = {
        "num_queries": n,
        "num_subjects": len(subject_groups),
        "num_paired_subjects": num_paired_subjects,
        "same_subject_top1_rate": float(np.mean(same_subject_top1_rates)),
        "same_subject_top3_rate": float(np.mean(same_subject_top3_rates)),
        "same_subject_top5_rate": float(np.mean(same_subject_top5_rates)),
        "paired_task_rank_mean": float(np.mean(paired_ranks)) if paired_ranks else None,
        "paired_task_rank_median": float(np.median(paired_ranks)) if paired_ranks else None,
        "paired_task_rank_std": float(np.std(paired_ranks)) if paired_ranks else None,
        "paired_task_in_top1_rate": float(np.mean(paired_in_top1)) if paired_in_top1 else None,
        "paired_task_in_top3_rate": float(np.mean(paired_in_top3)) if paired_in_top3 else None,
        "paired_task_in_top5_rate": float(np.mean(paired_in_top5)) if paired_in_top5 else None,
        "severity_neighbor_agreement_top5_mean": float(np.mean(severity_agreements)) if severity_agreements else None,
        "severity_neighbor_agreement_top5_std": float(np.std(severity_agreements)) if severity_agreements else None,
        "task_neighbor_agreement_top5_mean": float(np.mean(task_agreements)) if task_agreements else None,
        "task_neighbor_agreement_top5_std": float(np.std(task_agreements)) if task_agreements else None,
    }

    return per_query_rows, summary


def write_embedding_identity_retrieval(rows, csv_path):
    """Write the per-query retrieval table to CSV."""
    formatted = [{key: _format_value(row.get(key)) for key in RETRIEVAL_COLUMNS} for row in rows]
    write_csv_rows(csv_path, formatted, RETRIEVAL_COLUMNS)
    return Path(csv_path)


def write_embedding_identity_summary(summary, csv_path):
    """Write the aggregate summary table to CSV."""
    formatted = [{key: _format_value(summary.get(key)) for key in SUMMARY_COLUMNS}]
    write_csv_rows(csv_path, formatted, SUMMARY_COLUMNS)
    return Path(csv_path)


def _require_matplotlib():
    """Import matplotlib on demand so the module can be used without plotting."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for the embedding similarity heatmap.") from exc
    return plt


def plot_embedding_similarity_matrix(similarities, records, save_path):
    """Render a sorted cosine-similarity heatmap and save it to disk.

    Rows and columns are grouped by subject and then by task name so that
    same-subject and same-task blocks are visually apparent.
    """
    plt = _require_matplotlib()
    n = similarities.shape[0]
    if n < 2:
        return None

    # Build a stable sort key: subject, then task, then video_id.
    order = sorted(
        range(n),
        key=lambda idx: (
            records[idx]["subject_id"],
            records[idx]["task_name"],
            records[idx]["video_id"],
        ),
    )
    order = np.asarray(order, dtype=int)
    sorted_sim = similarities[np.ix_(order, order)]

    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    fig_size = max(6.0, min(20.0, n * 0.25))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(sorted_sim, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title("Embedding Cosine Similarity Matrix")
    ax.set_xlabel("Query Index (sorted by subject / task)")
    ax.set_ylabel("Neighbor Index (sorted by subject / task)")
    fig.colorbar(im, ax=ax, shrink=0.82, label="Cosine similarity")

    # Tick labels: short subject ID to keep the figure readable.
    tick_labels = [records[idx]["subject_id"] for idx in order]
    stride = max(1, n // 40)
    ticks = np.arange(0, n, stride)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([tick_labels[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([tick_labels[i] for i in ticks], fontsize=6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def write_embedding_identity_report(report_path, generated_files, summary, per_query_rows, top_k=5):
    """Write a markdown report interpreting the retrieval diagnostics."""
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    lines = [
        "# Embedding Identity Retrieval Audit Report",
        "",
        "## Summary",
        "",
        f"- Queries: {summary['num_queries']}",
        f"- Subjects: {summary['num_subjects']}",
        f"- Paired-task subjects: {summary['num_paired_subjects']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Same-subject top-1 rate | {summary['same_subject_top1_rate']:.4f} |",
        f"| Same-subject top-3 rate | {summary['same_subject_top3_rate']:.4f} |",
        f"| Same-subject top-5 rate | {summary['same_subject_top5_rate']:.4f} |",
    ]

    if summary["paired_task_rank_mean"] is not None:
        lines.extend(
            [
                f"| Paired-task rank mean | {summary['paired_task_rank_mean']:.2f} |",
                f"| Paired-task rank median | {summary['paired_task_rank_median']:.2f} |",
                f"| Paired-task in top-1 rate | {summary['paired_task_in_top1_rate']:.4f} |",
                f"| Paired-task in top-3 rate | {summary['paired_task_in_top3_rate']:.4f} |",
                f"| Paired-task in top-5 rate | {summary['paired_task_in_top5_rate']:.4f} |",
            ]
        )
    else:
        lines.append("| Paired-task retrieval | No paired tasks detected |")

    if summary["severity_neighbor_agreement_top5_mean"] is not None:
        lines.extend(
            [
                f"| Severity neighbor agreement (top-{top_k}) | {summary['severity_neighbor_agreement_top5_mean']:.4f} |",
                f"| Task neighbor agreement (top-{top_k}) | {summary['task_neighbor_agreement_top5_mean']:.4f} |",
            ]
        )

    lines.extend(["", "## Generated Files", ""])
    lines.extend(f"- `{path}`" for path in generated_files)

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Same-subject retrieval**: A high top-1/top-3/top-5 same-subject rate "
            "suggests the embedding space is dominated by subject identity / static "
            "appearance rather than BDI severity.",
            "- **Paired-task retrieval**: For AVEC2014 each subject has a Freeform and a "
            "Northwind video.  If the paired video is consistently among the nearest "
            "neighbors, the embedding is task-insensitive and mainly identity-driven.",
            "- **Severity neighbor agreement**: Low agreement means videos with similar "
            "severity are not close in embedding space, indicating the model does not "
            "encode a stable severity representation.",
            "- **Task neighbor agreement**: High agreement can reflect task-specific "
            "recording conditions (pose, gaze, length, background) rather than "
            "depression-related behavior.",
            "",
            "## Notes",
            "",
            "- Similarity is cosine similarity on L2-normalized embeddings.",
            "- The query itself is excluded from the neighbor ranking.",
            "- This audit is offline diagnostics only and must not alter training split, "
            "labels, or test-time decisions.",
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_embedding_identity_retrieval_audit(
    features_npz,
    output_dir,
    predictions_csv=None,
    top_k=5,
):
    """Run the full embedding identity retrieval audit pipeline.

    Args:
        features_npz: Path to a ``*_features.npz`` archive produced by the
            MTL-Lite diagnostic script.
        output_dir: Directory where tables, figures, and reports are written.
        predictions_csv: Optional prediction CSV for metadata fallback/enrichment.
        top_k: Neighborhood size for agreement metrics (default 5).

    Returns:
        List of generated file paths.
    """
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    figures_dir = ensure_dir(output_dir / "figures")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    features, records = load_embeddings_and_predictions(features_npz, predictions_csv=predictions_csv)
    per_query_rows, summary = compute_identity_retrieval_metrics(records, features, top_k=top_k)

    retrieval_path = write_embedding_identity_retrieval(
        per_query_rows, tables_dir / "embedding_identity_retrieval.csv"
    )
    generated.append(retrieval_path)

    summary_path = write_embedding_identity_summary(
        summary, tables_dir / "embedding_identity_summary.csv"
    )
    generated.append(summary_path)

    similarities = _cosine_similarity_matrix(features)
    heatmap_path = plot_embedding_similarity_matrix(
        similarities, records, figures_dir / "embedding_similarity_matrix.png"
    )
    if heatmap_path is not None:
        generated.append(heatmap_path)

    report_path = write_embedding_identity_report(
        reports_dir / "embedding_identity_report.md",
        generated_files=generated,
        summary=summary,
        per_query_rows=per_query_rows,
        top_k=top_k,
    )
    generated.append(report_path)
    return generated
