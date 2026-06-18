import csv
import tempfile
from pathlib import Path

import pytest

from src.diagnostics.identity_retrieval_runs import (
    SEVERITY_GROUPS,
    load_identity_retrieval_per_query,
    load_identity_retrieval_summary,
    summarize_identity_retrieval_by_severity,
    summarize_identity_retrieval_runs,
)


def _make_summary_row(run_name="rgb"):
    return {
        "num_queries": "100",
        "num_subjects": "50",
        "num_paired_subjects": "50",
        "same_subject_top1_rate": "0.66",
        "same_subject_top3_rate": "0.85",
        "same_subject_top5_rate": "0.90",
        "paired_task_rank_mean": "5.09",
        "paired_task_rank_median": "1.00",
        "paired_task_rank_std": "8.12",
        "paired_task_in_top1_rate": "0.35",
        "paired_task_in_top3_rate": "0.55",
        "paired_task_in_top5_rate": "0.75",
        "severity_neighbor_agreement_top5_mean": "0.492",
        "severity_neighbor_agreement_top5_std": "0.20",
        "task_neighbor_agreement_top5_mean": "0.55",
        "task_neighbor_agreement_top5_std": "0.18",
    }


def _write_summary_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(row.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)


def _make_per_query_rows():
    rows = []
    for i, group in enumerate(["minimal", "mild", "moderate", "severe"] * 5):
        rows.append(
            {
                "query_video_id": f"subj_{i:03d}_Freeform_video",
                "query_subject_id": f"subj_{i:03d}",
                "query_task_name": "Freeform",
                "query_true_bdi": "10.0",
                "query_pred_bdi": "12.0",
                "query_severity_group": group,
                "top1_neighbor_video_id": f"subj_{i:03d}_Northwind_video",
                "top1_neighbor_subject_id": f"subj_{i:03d}",
                "top1_neighbor_task_name": "Northwind",
                "top1_neighbor_same_subject": "1",
                "top3_contains_same_subject": "1",
                "top5_contains_same_subject": "1",
                "paired_task_video_id": f"subj_{i:03d}_Northwind_video",
                "paired_task_rank": "1",
                "paired_task_in_top1": "1",
                "paired_task_in_top3": "1",
                "paired_task_in_top5": "1",
                "severity_neighbor_agreement_top5": "0.5",
                "task_neighbor_agreement_top5": "0.6",
            }
        )
    return rows


def _write_per_query_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_load_identity_retrieval_summary(tmp_path):
    summary_path = tmp_path / "embedding_identity_summary.csv"
    _write_summary_csv(summary_path, _make_summary_row())
    summary = load_identity_retrieval_summary(summary_path)
    assert summary["num_queries"] == 100
    assert summary["same_subject_top1_rate"] == pytest.approx(0.66)
    assert summary["paired_task_rank_mean"] == pytest.approx(5.09)


def test_load_identity_retrieval_per_query(tmp_path):
    query_path = tmp_path / "embedding_identity_retrieval.csv"
    rows = _make_per_query_rows()
    _write_per_query_csv(query_path, rows)
    records = load_identity_retrieval_per_query(query_path)
    assert len(records) == len(rows)
    assert records[0]["top1_neighbor_same_subject"] == 1
    assert records[0]["severity_neighbor_agreement_top5"] == pytest.approx(0.5)


def test_summarize_identity_retrieval_by_severity():
    records = _make_per_query_rows()
    for record in records:
        record["top1_neighbor_same_subject"] = int(record["top1_neighbor_same_subject"])
        record["top3_contains_same_subject"] = int(record["top3_contains_same_subject"])
        record["top5_contains_same_subject"] = int(record["top5_contains_same_subject"])
        record["paired_task_in_top1"] = int(record["paired_task_in_top1"])
        record["paired_task_in_top3"] = int(record["paired_task_in_top3"])
        record["paired_task_in_top5"] = int(record["paired_task_in_top5"])
        record["severity_neighbor_agreement_top5"] = float(record["severity_neighbor_agreement_top5"])
        record["task_neighbor_agreement_top5"] = float(record["task_neighbor_agreement_top5"])
    summary = summarize_identity_retrieval_by_severity("rgb", records)
    groups = {row["severity_group"]: row for row in summary}
    for group in SEVERITY_GROUPS:
        assert group in groups
        assert groups[group]["count"] == 5
        assert groups[group]["same_subject_top1_rate"] == pytest.approx(1.0)


def test_summarize_identity_retrieval_runs(tmp_path):
    run_dir_a = tmp_path / "run_a"
    run_dir_b = tmp_path / "run_b"
    _write_summary_csv(run_dir_a / "tables" / "embedding_identity_summary.csv", _make_summary_row("rgb"))
    _write_summary_csv(run_dir_b / "tables" / "embedding_identity_summary.csv", _make_summary_row("center_mask"))
    _write_per_query_csv(run_dir_a / "tables" / "embedding_identity_retrieval.csv", _make_per_query_rows())
    _write_per_query_csv(run_dir_b / "tables" / "embedding_identity_retrieval.csv", _make_per_query_rows())

    output_dir = tmp_path / "summary"
    generated = summarize_identity_retrieval_runs(
        run_specs=[("rgb", run_dir_a), ("center_mask", run_dir_b)],
        output_dir=output_dir,
    )

    summary_csv = output_dir / "tables" / "identity_retrieval_run_summary.csv"
    severity_csv = output_dir / "tables" / "identity_retrieval_severity_summary.csv"
    report_md = output_dir / "reports" / "identity_retrieval_runs_report.md"

    assert summary_csv.exists()
    assert severity_csv.exists()
    assert report_md.exists()
    assert summary_csv in generated
    assert severity_csv in generated
    assert report_md in generated

    with summary_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["run_name"] for row in rows} == {"rgb", "center_mask"}

    with severity_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8  # 2 runs x 4 severity groups
