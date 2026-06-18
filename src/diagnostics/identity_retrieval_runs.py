"""Multi-run summary for embedding identity retrieval audits.

This module aggregates per-run identity retrieval summaries into a unified
comparison table.  It is the multi-run counterpart of
``src/diagnostics/identity_retrieval.py`` and produces the outputs required by
``docs/RGB_OVERFITTING_AUDIT_PLAN.md`` (P0-E follow-up).

Outputs:

- ``identity_retrieval_run_summary.csv``: one row per run with the same
  aggregate metrics as the individual run summary.
- ``identity_retrieval_severity_summary.csv``: per-run and per-severity-group
  retrieval metrics, when per-query retrieval tables are available.
- ``identity_retrieval_runs_report.md``: a markdown comparison report.
"""

import math
from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, read_csv_rows, write_csv_rows


RUN_SUMMARY_COLUMNS = [
    "run_name",
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

SEVERITY_SUMMARY_COLUMNS = [
    "run_name",
    "severity_group",
    "count",
    "same_subject_top1_rate",
    "same_subject_top3_rate",
    "same_subject_top5_rate",
    "paired_task_in_top1_rate",
    "paired_task_in_top3_rate",
    "paired_task_in_top5_rate",
    "severity_neighbor_agreement_top5_mean",
    "severity_neighbor_agreement_top5_std",
    "task_neighbor_agreement_top5_mean",
    "task_neighbor_agreement_top5_std",
]

SEVERITY_GROUPS = ["minimal", "mild", "moderate", "severe"]


def _format_scalar(value):
    """Format a scalar for CSV output; empty string for missing values."""
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


def _safe_int(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _resolve_summary_csv(path):
    """Resolve an identity retrieval summary CSV from a directory or explicit path."""
    path = Path(path)
    if path.is_file():
        return path
    candidate = path / "tables" / "embedding_identity_summary.csv"
    if candidate.exists():
        return candidate
    candidate = path / "embedding_identity_summary.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find embedding_identity_summary.csv under {path}")


def _resolve_per_query_csv(path):
    """Resolve the per-query retrieval CSV if it exists."""
    path = Path(path)
    if path.is_file():
        candidate = path.parent / "embedding_identity_retrieval.csv"
        if candidate.exists():
            return candidate
        return None
    candidate = path / "tables" / "embedding_identity_retrieval.csv"
    if candidate.exists():
        return candidate
    candidate = path / "embedding_identity_retrieval.csv"
    if candidate.exists():
        return candidate
    return None


def load_identity_retrieval_summary(csv_path):
    """Load a single-run aggregate summary CSV.

    Returns:
        Dict mapping summary column names to scalar values.
    """
    rows = read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"Empty summary CSV: {csv_path}")
    row = rows[0]
    summary = {}
    for key, value in row.items():
        if key in ("num_queries", "num_subjects", "num_paired_subjects"):
            summary[key] = _safe_int(value)
        else:
            summary[key] = _safe_float(value)
    return summary


def load_identity_retrieval_per_query(csv_path):
    """Load the per-query retrieval table.

    Returns:
        List of dict records with numeric fields converted where possible.
    """
    rows = read_csv_rows(csv_path)
    records = []
    for row in rows:
        record = {}
        for key, value in row.items():
            if key in (
                "top1_neighbor_same_subject",
                "top3_contains_same_subject",
                "top5_contains_same_subject",
                "paired_task_in_top1",
                "paired_task_in_top3",
                "paired_task_in_top5",
            ):
                record[key] = _safe_int(value) or 0
            elif key in (
                "query_true_bdi",
                "query_pred_bdi",
                "severity_neighbor_agreement_top5",
                "task_neighbor_agreement_top5",
                "paired_task_rank",
            ):
                record[key] = _safe_float(value)
            else:
                record[key] = value
        records.append(record)
    return records


def _mean_or_none(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(np.mean(values))


def _std_or_none(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    return float(np.std(values))


def _rate_or_none(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(np.mean(values))


def summarize_identity_retrieval_by_severity(run_name, per_query_records):
    """Aggregate per-query retrieval records by query severity group.

    Returns:
        List of dicts, one per severity group present in the records.
    """
    groups = {}
    for row in per_query_records:
        group = row.get("query_severity_group")
        if not group:
            continue
        groups.setdefault(group, []).append(row)

    summary_rows = []
    for group in SEVERITY_GROUPS:
        subset = groups.get(group)
        if not subset:
            continue
        task_agreements = [r["task_neighbor_agreement_top5"] for r in subset]
        severity_agreements = [r["severity_neighbor_agreement_top5"] for r in subset]
        paired_in_top1 = [r["paired_task_in_top1"] for r in subset if r.get("paired_task_in_top1") is not None]
        paired_in_top3 = [r["paired_task_in_top3"] for r in subset if r.get("paired_task_in_top3") is not None]
        paired_in_top5 = [r["paired_task_in_top5"] for r in subset if r.get("paired_task_in_top5") is not None]

        summary_rows.append(
            {
                "run_name": run_name,
                "severity_group": group,
                "count": len(subset),
                "same_subject_top1_rate": _rate_or_none([r["top1_neighbor_same_subject"] for r in subset]),
                "same_subject_top3_rate": _rate_or_none([r["top3_contains_same_subject"] for r in subset]),
                "same_subject_top5_rate": _rate_or_none([r["top5_contains_same_subject"] for r in subset]),
                "paired_task_in_top1_rate": _rate_or_none(paired_in_top1) if paired_in_top1 else None,
                "paired_task_in_top3_rate": _rate_or_none(paired_in_top3) if paired_in_top3 else None,
                "paired_task_in_top5_rate": _rate_or_none(paired_in_top5) if paired_in_top5 else None,
                "severity_neighbor_agreement_top5_mean": _mean_or_none(severity_agreements),
                "severity_neighbor_agreement_top5_std": _std_or_none(severity_agreements),
                "task_neighbor_agreement_top5_mean": _mean_or_none(task_agreements),
                "task_neighbor_agreement_top5_std": _std_or_none(task_agreements),
            }
        )
    return summary_rows


def write_identity_retrieval_run_summary(csv_path, rows):
    """Write the multi-run aggregate summary CSV."""
    formatted = [
        {col: _format_scalar(row.get(col)) for col in RUN_SUMMARY_COLUMNS}
        for row in rows
    ]
    write_csv_rows(csv_path, formatted, RUN_SUMMARY_COLUMNS)
    return Path(csv_path)


def write_identity_retrieval_severity_summary(csv_path, rows):
    """Write the per-run per-severity-group summary CSV."""
    formatted = [
        {col: _format_scalar(row.get(col)) for col in SEVERITY_SUMMARY_COLUMNS}
        for row in rows
    ]
    write_csv_rows(csv_path, formatted, SEVERITY_SUMMARY_COLUMNS)
    return Path(csv_path)


def write_identity_retrieval_runs_report(report_path, run_rows, severity_rows, generated_files):
    """Write the markdown multi-run comparison report."""
    report_path = Path(report_path)
    ensure_dir(report_path.parent)

    lines = [
        "# Identity Retrieval Multi-Run Summary Report",
        "",
        "This report compares embedding identity-retrieval diagnostics across multiple runs.",
        "",
        "## Run-Level Summary",
        "",
        "| Run | N | same_top1 | same_top3 | same_top5 | paired_rank_mean | paired_in_top5 | severity_agree | task_agree |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append(
            f"| {row['run_name']} | {row.get('num_queries', '')} | "
            f"{_format_scalar(row.get('same_subject_top1_rate'))} | "
            f"{_format_scalar(row.get('same_subject_top3_rate'))} | "
            f"{_format_scalar(row.get('same_subject_top5_rate'))} | "
            f"{_format_scalar(row.get('paired_task_rank_mean'))} | "
            f"{_format_scalar(row.get('paired_task_in_top5_rate'))} | "
            f"{_format_scalar(row.get('severity_neighbor_agreement_top5_mean'))} | "
            f"{_format_scalar(row.get('task_neighbor_agreement_top5_mean'))} |"
        )

    if severity_rows:
        lines.extend(
            [
                "",
                "## Severity-Group Summary",
                "",
                "| Run | Severity | Count | same_top1 | same_top5 | paired_in_top5 | severity_agree | task_agree |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in severity_rows:
            lines.append(
                f"| {row['run_name']} | {row['severity_group']} | {row['count']} | "
                f"{_format_scalar(row.get('same_subject_top1_rate'))} | "
                f"{_format_scalar(row.get('same_subject_top5_rate'))} | "
                f"{_format_scalar(row.get('paired_task_in_top5_rate'))} | "
                f"{_format_scalar(row.get('severity_neighbor_agreement_top5_mean'))} | "
                f"{_format_scalar(row.get('task_neighbor_agreement_top5_mean'))} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A run with high same-subject retrieval but low severity agreement is likely encoding "
            "subject identity / static appearance rather than depression severity.",
            "- Compare paired-task rank across runs: a lower mean rank means the Freeform/Northwind "
            "pair is closer in embedding space.",
            "- Use the severity-group table to check whether severe/minimal samples drive the identity shortcut.",
            "",
            "## Generated Files",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in generated_files)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def summarize_identity_retrieval_runs(run_specs, output_dir):
    """Aggregate identity retrieval summaries across multiple runs.

    Args:
        run_specs: List of ``(run_name, path_or_summary_csv)`` tuples.  If a
            directory is provided, both ``tables/embedding_identity_summary.csv``
            and ``tables/embedding_identity_retrieval.csv`` are used when
            available.
        output_dir: Directory for the summary outputs.

    Returns:
        List of generated file paths.
    """
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    generated = []

    run_rows = []
    severity_rows = []

    for run_name, path in run_specs:
        summary_csv = _resolve_summary_csv(path)
        summary = load_identity_retrieval_summary(summary_csv)
        summary["run_name"] = run_name
        run_rows.append(summary)

        per_query_csv = _resolve_per_query_csv(path)
        if per_query_csv is not None:
            per_query_records = load_identity_retrieval_per_query(per_query_csv)
            severity_rows.extend(summarize_identity_retrieval_by_severity(run_name, per_query_records))

    summary_path = write_identity_retrieval_run_summary(
        tables_dir / "identity_retrieval_run_summary.csv", run_rows
    )
    generated.append(summary_path)

    severity_path = None
    if severity_rows:
        severity_path = write_identity_retrieval_severity_summary(
            tables_dir / "identity_retrieval_severity_summary.csv", severity_rows
        )
        generated.append(severity_path)

    report_path = write_identity_retrieval_runs_report(
        reports_dir / "identity_retrieval_runs_report.md",
        run_rows=run_rows,
        severity_rows=severity_rows,
        generated_files=generated,
    )
    generated.append(report_path)
    return generated
