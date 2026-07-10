"""Offline subject-attacker accuracy audit for Stage B identity risk.

The identity-adversarial branch (E1/E3) trains a ``subject_id_head`` on the
GRL-reversed shared features to predict the train subject id.  The strength of
that attacker is a B5 identity-risk signal distinct from A1 retrieval (cosine
neighbor same-subject rate): retrieval measures whether the embedding *geometry*
separates subjects; attacker accuracy measures whether a trained *classifier*
recovers subject id from ``z_dep``.  A successful adversarial defense drives
attacker accuracy toward chance while preserving BDI utility.

This module is the offline counterpart of the training-time identity CE.  It
loads a checkpoint, runs the test split through ``subject_id_head``, and reports
top-1 / top-3 subject-classification accuracy **on the seen-subject subset**
(test videos whose subject appears in the train-only subject table).  Test
subjects absent from training (unseen) cannot be classified by the head (their
class index is undefined) and are reported as ``coverage = seen / total`` --
a low coverage itself indicates the head was trained on a narrow subject pool.

Outputs (single-run schema, mirroring :mod:`src.diagnostics.identity_retrieval`):

- ``tables/subject_attacker_summary.csv``: one row, aggregate metrics.
- ``tables/subject_attacker_per_query.csv``: per-video seen/unseen + prediction.
- ``reports/subject_attacker_report.md``: human-readable summary.

The summary is a single row with the same shape the identity-retrieval summary
uses, so :func:`src.diagnostics.identity_retrieval_runs._resolve_summary_csv`
can consume it via the multi-run aggregator below.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.diagnostics.io import ensure_dir, read_csv_rows, write_csv_rows

# Aggregate summary columns (one row per run).  ``skipped`` marks runs without
# an identity head (E0/E2) so the aggregator can show them as not-applicable.
SUMMARY_COLUMNS = [
    "num_evaluated",
    "num_seen",
    "num_unseen_excluded",
    "coverage",
    "top1_accuracy",
    "top3_accuracy",
    "num_subject_classes",
    "chance_top1",
    "skipped",
]

# Per-query (per-video) columns.
PER_QUERY_COLUMNS = [
    "video_id",
    "subject_id",
    "seen",
    "true_subject_index",
    "pred_subject_index",
    "pred_top3_subject_indices",
    "correct_top1",
    "correct_top3",
]


def compute_attacker_metrics(
    per_query_rows: List[Dict],
    num_subject_classes: int,
) -> Dict:
    """Aggregate per-query attacker rows into a single-run summary.

    Args:
        per_query_rows: One dict per evaluated video (see PER_QUERY_COLUMNS).
            Unseen videos carry ``seen=False`` / ``correct_top1=None``.
        num_subject_classes: Size of the train subject table (head output dim).

    Returns:
        Dict with SUMMARY_COLUMNS values.  ``top1_accuracy`` / ``top3_accuracy``
        are computed over the seen subset only; ``coverage`` = seen / total.
        ``chance_top1`` = 1 / num_subject_classes (reference floor).
    """
    num_evaluated = len(per_query_rows)
    seen_rows = [r for r in per_query_rows if r.get("seen")]
    num_seen = len(seen_rows)
    num_unseen = num_evaluated - num_seen

    top1_correct = sum(1 for r in seen_rows if r.get("correct_top1"))
    top3_correct = sum(1 for r in seen_rows if r.get("correct_top3"))

    top1 = (top1_correct / num_seen) if num_seen else None
    top3 = (top3_correct / num_seen) if num_seen else None
    coverage = (num_seen / num_evaluated) if num_evaluated else None
    chance = (1.0 / num_subject_classes) if num_subject_classes else None

    return {
        "num_evaluated": num_evaluated,
        "num_seen": num_seen,
        "num_unseen_excluded": num_unseen,
        "coverage": coverage,
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "num_subject_classes": num_subject_classes,
        "chance_top1": chance,
        "skipped": False,
    }


def skipped_summary(reason: str = "no identity head") -> Dict:
    """Summary row for a run that cannot be attacker-evaluated (E0/E2)."""
    return {
        "num_evaluated": 0,
        "num_seen": 0,
        "num_unseen_excluded": 0,
        "coverage": None,
        "top1_accuracy": None,
        "top3_accuracy": None,
        "num_subject_classes": 0,
        "chance_top1": None,
        "skipped": True,
    }


def _fmt_indices(indices: Optional[List[int]]) -> str:
    if not indices:
        return ""
    return ";".join(str(int(i)) for i in indices)


def write_subject_attacker_summary(summary: Dict, csv_path) -> Path:
    """Write the single-run aggregate summary CSV (one row)."""
    formatted = [{col: _format_value(summary.get(col)) for col in SUMMARY_COLUMNS}]
    write_csv_rows(csv_path, formatted, SUMMARY_COLUMNS)
    return Path(csv_path)


def write_subject_attacker_per_query(rows: List[Dict], csv_path) -> Path:
    """Write the per-video attacker prediction table."""
    formatted = [
        {
            "video_id": r.get("video_id", ""),
            "subject_id": r.get("subject_id", ""),
            "seen": r.get("seen", False),
            "true_subject_index": _format_value(r.get("true_subject_index")),
            "pred_subject_index": _format_value(r.get("pred_subject_index")),
            "pred_top3_subject_indices": _fmt_indices(r.get("pred_top3_subject_indices")),
            "correct_top1": r.get("correct_top1", False),
            "correct_top3": r.get("correct_top3", False),
        }
        for r in rows
    ]
    write_csv_rows(csv_path, formatted, PER_QUERY_COLUMNS)
    return Path(csv_path)


def _format_value(value):
    """Format a scalar for CSV: None -> '', bool -> int, float -> rounded str."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_subject_attacker_report(report_path, summary: Dict, per_query_rows: List[Dict]) -> Path:
    """Write a human-readable single-run attacker report."""
    report_path = Path(report_path)
    ensure_dir(report_path.parent)
    lines = [
        "# Subject Attacker Accuracy Report",
        "",
        "Offline evaluation of the identity-adversarial subject_id_head on the test split.",
        "Accuracy is computed over the **seen-subject subset** (test subjects present in",
        "the train-only subject table); unseen subjects are excluded (reported as coverage).",
        "",
        "## Summary",
        "",
        f"- evaluated: {summary.get('num_evaluated', 0)}",
        f"- seen (classified): {summary.get('num_seen', 0)}",
        f"- unseen (excluded): {summary.get('num_unseen_excluded', 0)}",
        f"- coverage: {_format_value(summary.get('coverage'))}",
        f"- top-1 accuracy: {_format_value(summary.get('top1_accuracy'))}",
        f"- top-3 accuracy: {_format_value(summary.get('top3_accuracy'))}",
        f"- num subject classes: {summary.get('num_subject_classes', 0)}",
        f"- chance top-1: {_format_value(summary.get('chance_top1'))}",
        f"- skipped: {summary.get('skipped', False)}",
        "",
    ]
    if summary.get("skipped"):
        lines.append("Run has no identity head (identity_adversarial disabled) -- skipped.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Multi-run aggregation (consumed by aggregate_stage_b.sh).
# ---------------------------------------------------------------------------

RUN_SUMMARY_COLUMNS = ["run_name"] + SUMMARY_COLUMNS


def _resolve_summary_csv(path) -> Path:
    """Resolve a subject-attacker summary CSV from a dir or explicit path."""
    path = Path(path)
    if path.is_file():
        return path
    candidate = path / "tables" / "subject_attacker_summary.csv"
    if candidate.exists():
        return candidate
    candidate = path / "subject_attacker_summary.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find subject_attacker_summary.csv under {path}")


def load_subject_attacker_summary(csv_path) -> Dict:
    """Load a single-run summary CSV into a dict (first row)."""
    rows = read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"Empty summary CSV: {csv_path}")
    row = rows[0]
    summary: Dict = {}
    for key, value in row.items():
        if key in ("num_evaluated", "num_seen", "num_unseen_excluded", "num_subject_classes"):
            summary[key] = _safe_int(value)
        elif key in ("coverage", "top1_accuracy", "top3_accuracy", "chance_top1"):
            summary[key] = _safe_float(value)
        elif key == "skipped":
            summary[key] = str(value).strip().lower() in ("1", "true", "yes")
        else:
            summary[key] = value
    return summary


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _format_scalar(value):
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def write_subject_attacker_run_summary(csv_path, rows: List[Dict]) -> Path:
    """Write the multi-run aggregate summary CSV (one row per run)."""
    formatted = [
        {col: _format_scalar(row.get(col)) for col in RUN_SUMMARY_COLUMNS} for row in rows
    ]
    write_csv_rows(csv_path, formatted, RUN_SUMMARY_COLUMNS)
    return Path(csv_path)


def write_subject_attacker_runs_report(report_path, run_rows: List[Dict], generated_files: List) -> Path:
    """Write the markdown multi-run comparison report."""
    report_path = Path(report_path)
    ensure_dir(report_path.parent)
    lines = [
        "# Subject Attacker Multi-Run Summary Report",
        "",
        "Compares the offline subject_id_head classification accuracy across runs.",
        "Accuracy is over the seen-subject subset; coverage = seen / total test videos.",
        "``skipped`` runs have no identity head (E0/E2).",
        "",
        "## Run-Level Summary",
        "",
        "| Run | evaluated | seen | coverage | top1 | top3 | num_classes | chance | skipped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in run_rows:
        lines.append(
            f"| {row.get('run_name', '')} | {row.get('num_evaluated', '')} | "
            f"{row.get('num_seen', '')} | {_format_scalar(row.get('coverage'))} | "
            f"{_format_scalar(row.get('top1_accuracy'))} | {_format_scalar(row.get('top3_accuracy'))} | "
            f"{row.get('num_subject_classes', '')} | {_format_scalar(row.get('chance_top1'))} | "
            f"{row.get('skipped', '')} |"
        )
    lines.extend(["", "## Generated Files", ""])
    for path in generated_files:
        lines.append(f"- {path}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def summarize_subject_attacker_runs(run_specs: List[Tuple[str, str]], output_dir) -> List[Path]:
    """Aggregate per-run subject-attacker summaries into one comparison table.

    Args:
        run_specs: List of ``(run_name, path_or_summary_csv)`` tuples.  A
            directory is resolved to ``tables/subject_attacker_summary.csv``.
        output_dir: Directory for the multi-run summary + report.

    Returns:
        List of generated file paths.  Runs whose summary is missing are
        skipped with a warning (not an error) so partial diagnoses aggregate.
    """
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    generated: List[Path] = []

    run_rows: List[Dict] = []
    for run_name, path in run_specs:
        try:
            summary_csv = _resolve_summary_csv(path)
        except FileNotFoundError:
            print(f"[SUBJECT_ATTACKER_RUNS] no summary for {run_name} under {path}, skipping")
            continue
        summary = load_subject_attacker_summary(summary_csv)
        summary["run_name"] = run_name
        run_rows.append(summary)

    if not run_rows:
        print("[SUBJECT_ATTACKER_RUNS] no runs to aggregate")
        return generated

    summary_path = write_subject_attacker_run_summary(
        tables_dir / "subject_attacker_run_summary.csv", run_rows
    )
    generated.append(summary_path)

    report_path = write_subject_attacker_runs_report(
        reports_dir / "subject_attacker_runs_report.md", run_rows, generated
    )
    generated.append(report_path)
    return generated
