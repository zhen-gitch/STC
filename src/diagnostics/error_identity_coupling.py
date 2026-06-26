"""RPDF Stage A2: prediction error x identity similarity coupling.

This module proves whether the prediction *uses* subject identity, not just
whether the embedding *contains* it (which A1 establishes).  It joins:

- per-layer embeddings from a layer-wise NPZ (A1 output)
- prediction metadata (residual / abs_error / severity_group)

and reports how prediction error co-varies with identity similarity, paired-task
rank, and neighborhood agreement, both continuously (Pearson / Spearman
correlation) and per severity bin.

Outputs:

- ``error_identity_correlation.csv``: per-layer correlation of error with
  identity signals.
- ``severity_bin_identity_error_summary.csv``: per-severity, per-layer means.
- ``high_error_high_identity_cases.csv``: worst coupling cases for paper
  case-study anchors.
- ``identity_error_coupling_report.md``.
"""

import math
from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, read_csv_rows, write_csv_rows


CORRELATION_COLUMNS = [
    "layer_name",
    "n",
    "corr_abs_error_vs_paired_identity_sim",
    "corr_residual_vs_paired_identity_sim",
    "corr_abs_error_vs_paired_rank",
    "corr_residual_vs_paired_rank",
    "corr_abs_error_vs_severity_agree",
    "corr_residual_vs_severity_agree",
    "mean_paired_identity_sim",
    "mean_abs_error",
    "mean_residual",
]

SEVERITY_BIN_COLUMNS = [
    "layer_name",
    "severity_group",
    "n",
    "mean_abs_error",
    "mean_residual",
    "mean_paired_identity_sim",
    "mean_paired_rank",
    "mean_severity_agree",
]

HIGH_COUPLING_CASE_COLUMNS = [
    "layer_name",
    "video_id",
    "subject_id",
    "task_name",
    "true_bdi",
    "pred_bdi",
    "residual",
    "abs_error",
    "severity_group",
    "paired_identity_sim",
    "paired_rank",
    "paired_in_top5",
    "severity_agree",
    "coupling_score",
]


def _cosine_similarity_matrix(features):
    """Pairwise cosine similarities with self-similarity masked to -1."""
    features = np.asarray(features, dtype=float)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    zero_mask = norms.ravel() < 1e-12
    normalized = np.divide(
        features,
        norms,
        out=np.zeros_like(features),
        where=norms > 1e-12,
    )
    similarities = normalized @ normalized.T
    similarities = (similarities + similarities.T) / 2.0
    np.fill_diagonal(similarities, -1.0)
    similarities[zero_mask, :] = -1.0
    similarities[:, zero_mask] = -1.0
    return similarities


def _subject_index_groups(records):
    groups = {}
    for idx, record in enumerate(records):
        groups.setdefault(record["subject_id"], []).append(idx)
    return groups


def _paired_task_index(query_idx, records, subject_groups):
    """Return the index of the paired Freeform/Northwind video, or None."""
    subject_id = records[query_idx]["subject_id"]
    group = subject_groups.get(subject_id, [])
    if len(group) != 2:
        return None
    other_idx = group[0] if group[1] == query_idx else group[1]
    if records[other_idx]["task_name"] == records[query_idx]["task_name"]:
        return None
    return other_idx


def _pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return None
    am = a - a.mean()
    bm = b - b.mean()
    denom = np.sqrt((am ** 2).sum() * (bm ** 2).sum())
    if denom < 1e-12:
        return None
    return float((am * bm).sum() / denom)


def _spearman(a, b):
    """Spearman rank correlation (Pearson on ranks)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return None
    ra = _rank(a)
    rb = _rank(b)
    return _pearson(ra, rb)


def _rank(values):
    """Average-rank of values (ties share the mean rank)."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based ranks
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


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


def _load_predictions(predictions_csv):
    """Load prediction rows keyed by video_id."""
    rows = read_csv_rows(predictions_csv)
    by_video = {}
    for row in rows:
        key = row.get("video_id") or row.get("subject_id")
        if key:
            by_video[str(key)] = row
    return by_video


def _prediction_error(row, fallback_record):
    """Return (residual, abs_error) from a prediction row, falling back to record."""
    residual = _safe_float(row.get("residual"))
    abs_error = _safe_float(row.get("abs_error"))
    if residual is None:
        pred = _safe_float(row.get("pred_bdi"))
        true = _safe_float(row.get("true_bdi"))
        if pred is None:
            pred = _safe_float(fallback_record.get("pred_bdi"))
        if true is None:
            true = _safe_float(fallback_record.get("true_bdi"))
        if pred is not None and true is not None:
            residual = pred - true
    if abs_error is None and residual is not None:
        abs_error = abs(residual)
    return residual, abs_error


def compute_error_identity_coupling(records, features, per_query_rows, top_k=5):
    """Compute per-query error/identity coupling rows for one layer.

    Args:
        records: Metadata records aligned with ``features`` rows.
        features: ``(N, D)`` array of video-level embeddings for this layer.
        per_query_rows: A1 per-query rows for this layer (must be aligned with
            ``records`` by video_id); supplies paired_rank, paired_in_top5,
            severity_agree.  May be ``None`` if A1 output is unavailable, in
            which case only continuous identity similarity is reported.
        top_k: Unused here (agreement metrics come from per_query_rows); kept
            for API symmetry.

    Returns:
        List of per-query coupling dicts with keys matching
        ``HIGH_COUPLING_CASE_COLUMNS`` (minus ``coupling_score``).
    """
    features = np.asarray(features, dtype=float)
    n = features.shape[0]
    if n != len(records):
        raise ValueError(
            f"Feature count ({n}) does not match record count ({len(records)})."
        )

    similarities = _cosine_similarity_matrix(features)
    subject_groups = _subject_index_groups(records)

    # Index A1 per-query rows by video_id for aligned lookup.
    per_query_by_video = {}
    if per_query_rows is not None:
        for row in per_query_rows:
            key = row.get("query_video_id") or row.get("query_subject_id")
            if key:
                per_query_by_video[str(key)] = row

    coupling_rows = []
    for query_idx in range(n):
        record = records[query_idx]
        video_id = record["video_id"]

        paired_idx = _paired_task_index(query_idx, records, subject_groups)
        paired_sim = None
        paired_rank = None
        paired_in_top5 = None
        severity_agree = None
        if paired_idx is not None:
            paired_sim = float(similarities[query_idx, paired_idx])

        pq = per_query_by_video.get(video_id)
        if pq is not None:
            paired_rank = _safe_float(pq.get("paired_task_rank"))
            paired_in_top5 = _safe_float(pq.get("paired_task_in_top5"))
            severity_agree = _safe_float(pq.get("severity_neighbor_agreement_top5"))

        coupling_rows.append(
            {
                "video_id": video_id,
                "subject_id": record["subject_id"],
                "task_name": record["task_name"],
                "true_bdi": record["true_bdi"],
                "pred_bdi": record["pred_bdi"],
                "residual": None,  # filled by caller from predictions
                "abs_error": None,
                "severity_group": record["severity_group"],
                "paired_identity_sim": paired_sim,
                "paired_rank": paired_rank,
                "paired_in_top5": paired_in_top5,
                "severity_agree": severity_agree,
            }
        )
    return coupling_rows


def _enrich_with_predictions(coupling_rows, predictions_by_video):
    """Fill residual/abs_error on coupling rows from the prediction table."""
    for row in coupling_rows:
        pred_row = predictions_by_video.get(row["video_id"])
        if pred_row is None:
            continue
        residual, abs_error = _prediction_error(pred_row, row)
        row["residual"] = residual
        row["abs_error"] = abs_error


def _correlation_summary(layer_name, coupling_rows):
    """Aggregate per-query coupling into a correlation summary row."""
    abs_errors = []
    residuals = []
    sims = []
    ranks = []
    severity_agrees = []
    for row in coupling_rows:
        if row["abs_error"] is None or row["residual"] is None:
            continue
        abs_errors.append(row["abs_error"])
        residuals.append(row["residual"])
        if row["paired_identity_sim"] is not None:
            sims.append(row["paired_identity_sim"])
        if row["paired_rank"] is not None:
            ranks.append(row["paired_rank"])
        if row["severity_agree"] is not None:
            severity_agrees.append(row["severity_agree"])

    n = len(abs_errors)
    summary = {
        "layer_name": layer_name,
        "n": n,
        "corr_abs_error_vs_paired_identity_sim": _pearson(abs_errors, sims) if len(sims) == n else None,
        "corr_residual_vs_paired_identity_sim": _pearson(residuals, sims) if len(sims) == n else None,
        "corr_abs_error_vs_paired_rank": _spearman(abs_errors, ranks) if len(ranks) == n else None,
        "corr_residual_vs_paired_rank": _spearman(residuals, ranks) if len(ranks) == n else None,
        "corr_abs_error_vs_severity_agree": _pearson(abs_errors, severity_agrees) if len(severity_agrees) == n else None,
        "corr_residual_vs_severity_agree": _pearson(residuals, severity_agrees) if len(severity_agrees) == n else None,
        "mean_paired_identity_sim": float(np.mean(sims)) if sims else None,
        "mean_abs_error": float(np.mean(abs_errors)) if abs_errors else None,
        "mean_residual": float(np.mean(residuals)) if residuals else None,
    }
    return summary


def _severity_bin_summary(layer_name, coupling_rows):
    """Aggregate per-query coupling by severity group."""
    bins = {}
    for row in coupling_rows:
        if row["abs_error"] is None:
            continue
        bins.setdefault(row["severity_group"], []).append(row)

    rows = []
    for group in ("minimal", "mild", "moderate", "severe"):
        group_rows = bins.get(group, [])
        if not group_rows:
            continue
        sims = [r["paired_identity_sim"] for r in group_rows if r["paired_identity_sim"] is not None]
        ranks = [r["paired_rank"] for r in group_rows if r["paired_rank"] is not None]
        agrees = [r["severity_agree"] for r in group_rows if r["severity_agree"] is not None]
        rows.append(
            {
                "layer_name": layer_name,
                "severity_group": group,
                "n": len(group_rows),
                "mean_abs_error": float(np.mean([r["abs_error"] for r in group_rows])),
                "mean_residual": float(np.mean([r["residual"] for r in group_rows])),
                "mean_paired_identity_sim": float(np.mean(sims)) if sims else None,
                "mean_paired_rank": float(np.mean(ranks)) if ranks else None,
                "mean_severity_agree": float(np.mean(agrees)) if agrees else None,
            }
        )
    return rows


def _high_coupling_cases(coupling_rows, max_cases=20):
    """Select worst error-x-identity coupling cases.

    ``coupling_score = normalized(abs_error) * normalized(paired_identity_sim)``.
    High abs_error and high identity similarity to the paired task video
    indicate the model "recognized" the subject but still mispredicted BDI.
    """
    valid = [r for r in coupling_rows if r["abs_error"] is not None and r["paired_identity_sim"] is not None]
    if not valid:
        return []
    abs_errors = np.array([r["abs_error"] for r in valid], dtype=float)
    sims = np.array([r["paired_identity_sim"] for r in valid], dtype=float)

    def _norm(x):
        lo, hi = x.min(), x.max()
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    scores = _norm(abs_errors) * _norm(sims)
    order = np.argsort(-scores)
    selected = []
    for idx in order[:max_cases]:
        row = dict(valid[idx])
        row["coupling_score"] = float(scores[idx])
        selected.append(row)
    return selected


def run_error_identity_coupling_audit(
    features_npz,
    predictions_csv,
    output_dir,
    per_query_csv=None,
    layer_order=None,
    max_cases=20,
):
    """Run the A2 error x identity coupling audit across all layers.

    Args:
        features_npz: Layer-wise NPZ from A1
            (:func:`layerwise_identity.save_layerwise_features_npz`).
        predictions_csv: Prediction CSV with residual/abs_error (required).
        output_dir: Directory for outputs.
        per_query_csv: Optional A1 ``layerwise_identity_per_query.csv`` for
            paired_rank / paired_in_top5 / severity_agree.  When omitted, only
            continuous paired identity similarity is reported.
        layer_order: Optional explicit layer ordering.
        max_cases: Max high-coupling cases to retain per layer.

    Returns:
        List of generated file paths.
    """
    from src.diagnostics.layerwise_identity import (
        _order_layers,
        load_layerwise_features_npz,
    )

    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    layer_features, records = load_layerwise_features_npz(
        features_npz, predictions_csv=predictions_csv
    )
    predictions_by_video = _load_predictions(predictions_csv)

    per_query_rows_by_layer = {}
    if per_query_csv is not None:
        all_pq = read_csv_rows(per_query_csv)
        for row in all_pq:
            layer = row.get("layer_name")
            if layer:
                per_query_rows_by_layer.setdefault(layer, []).append(row)

    layer_names = _order_layers(layer_features.keys(), layer_order)

    correlation_rows = []
    severity_rows = []
    high_case_rows = []
    for layer_name in layer_names:
        features = layer_features[layer_name]
        pq = per_query_rows_by_layer.get(layer_name)
        coupling_rows = compute_error_identity_coupling(records, features, pq)
        _enrich_with_predictions(coupling_rows, predictions_by_video)

        correlation_rows.append(_correlation_summary(layer_name, coupling_rows))
        severity_rows.extend(_severity_bin_summary(layer_name, coupling_rows))
        for case in _high_coupling_cases(coupling_rows, max_cases=max_cases):
            case_row = {"layer_name": layer_name}
            case_row.update(case)
            high_case_rows.append(case_row)

    corr_path = tables_dir / "error_identity_correlation.csv"
    write_csv_rows(corr_path, correlation_rows, CORRELATION_COLUMNS)
    generated.append(corr_path)

    sev_path = tables_dir / "severity_bin_identity_error_summary.csv"
    write_csv_rows(sev_path, severity_rows, SEVERITY_BIN_COLUMNS)
    generated.append(sev_path)

    cases_path = tables_dir / "high_error_high_identity_cases.csv"
    write_csv_rows(cases_path, high_case_rows, HIGH_COUPLING_CASE_COLUMNS)
    generated.append(cases_path)

    report_path = _write_coupling_report(
        reports_dir / "identity_error_coupling_report.md",
        correlation_rows=correlation_rows,
        severity_rows=severity_rows,
        generated_files=generated,
    )
    generated.append(report_path)
    return generated


def _write_coupling_report(report_path, correlation_rows, severity_rows, generated_files):
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    lines = [
        "# Prediction Error x Identity Similarity Coupling Report",
        "",
        "RPDF Stage A2. This report tests whether prediction error co-varies "
        "with subject identity signals. A1 only proves identity *exists* in the "
        "embedding; A2 is required to prove the prediction *uses* it.",
        "",
        "## Per-Layer Correlation",
        "",
        "| Layer | n | corr(|err|,id_sim) | corr(res,id_sim) | corr(|err|,rank) | corr(res,sev_agree) | mean_id_sim |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in correlation_rows:
        lines.append(
            f"| {row['layer_name']} | {row['n']} | "
            f"{_format_scalar(row.get('corr_abs_error_vs_paired_identity_sim'))} | "
            f"{_format_scalar(row.get('corr_residual_vs_paired_identity_sim'))} | "
            f"{_format_scalar(row.get('corr_abs_error_vs_paired_rank'))} | "
            f"{_format_scalar(row.get('corr_residual_vs_severity_agree'))} | "
            f"{_format_scalar(row.get('mean_paired_identity_sim'))} |"
        )

    lines.extend(
        [
            "",
            "## Severity Bin Summary",
            "",
            "| Layer | Group | n | mean_|err| | mean_res | mean_id_sim | mean_rank | mean_sev_agree |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in severity_rows:
        lines.append(
            f"| {row['layer_name']} | {row['severity_group']} | {row['n']} | "
            f"{_format_scalar(row.get('mean_abs_error'))} | "
            f"{_format_scalar(row.get('mean_residual'))} | "
            f"{_format_scalar(row.get('mean_paired_identity_sim'))} | "
            f"{_format_scalar(row.get('mean_paired_rank'))} | "
            f"{_format_scalar(row.get('mean_severity_agree'))} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **corr(|err|, id_sim) > 0**: higher identity similarity to the paired "
            "task video coincides with larger absolute error -- the model 'recognized' "
            "the subject but still mispredicted BDI, evidence the prediction uses identity.",
            "- **corr(|err|, rank) > 0** (rank = paired-task neighbor rank, 1=best): "
            "stronger identity memory coincides with larger error.",
            "- If coupling is concentrated in the severe bin (negative residual, high "
            "id_sim), it supports 'severe underestimation is coupled with identity memory'.",
            "- If correlations are ~0 across all layers, identity exists in the embedding "
            "but is not used by the prediction head; Stage B identity suppression should "
            "become risk monitoring rather than a strong adversarial branch.",
            "",
            "## Generated Files",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in generated_files)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
