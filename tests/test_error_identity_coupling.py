"""Tests for RPDF Stage A2: error x identity similarity coupling.

Pure-numpy tests that verify correlation computation, severity-bin aggregation,
and high-coupling case selection, without requiring torch.
"""

import csv
import math

import numpy as np
import pytest

from src.diagnostics.error_identity_coupling import (
    CORRELATION_COLUMNS,
    HIGH_COUPLING_CASE_COLUMNS,
    SEVERITY_BIN_COLUMNS,
    _pearson,
    _rank,
    _spearman,
    compute_error_identity_coupling,
    run_error_identity_coupling_audit,
)
from src.diagnostics.io import read_csv_rows, write_csv_rows
from src.diagnostics.layerwise_identity import save_layerwise_features_npz


def _records():
    """4 videos: 2 subjects x (Freeform, Northwind). Paired tasks share subject."""
    return [
        {"video_id": "101_1_Freeform_video", "subject_id": "101", "task_name": "Freeform", "true_bdi": 10.0, "pred_bdi": 18.0, "severity_group": "minimal"},
        {"video_id": "101_1_Northwind_video", "subject_id": "101", "task_name": "Northwind", "true_bdi": 10.0, "pred_bdi": 17.0, "severity_group": "minimal"},
        {"video_id": "102_1_Freeform_video", "subject_id": "102", "task_name": "Freeform", "true_bdi": 30.0, "pred_bdi": 22.0, "severity_group": "severe"},
        {"video_id": "102_1_Northwind_video", "subject_id": "102", "task_name": "Northwind", "true_bdi": 30.0, "pred_bdi": 23.0, "severity_group": "severe"},
    ]


def _prediction_csv(tmp_path, records):
    """Write a prediction CSV with residual/abs_error derived from records."""
    rows = []
    for r in records:
        residual = r["pred_bdi"] - r["true_bdi"]
        rows.append(
            {
                "video_id": r["video_id"],
                "subject_id": r["subject_id"],
                "task_name": r["task_name"],
                "true_bdi": r["true_bdi"],
                "pred_bdi": r["pred_bdi"],
                "residual": residual,
                "abs_error": abs(residual),
                "severity_group": r["severity_group"],
            }
        )
    path = tmp_path / "predictions.csv"
    write_csv_rows(
        path,
        rows,
        ["video_id", "subject_id", "task_name", "true_bdi", "pred_bdi", "residual", "abs_error", "severity_group"],
    )
    return path


def _layer_npz(tmp_path, features, layer_name="layer_shared"):
    records = _records()
    save_layerwise_features_npz(
        tmp_path / "lw.npz",
        {layer_name: features},
        [r["subject_id"] for r in records],
        [r["true_bdi"] for r in records],
        [r["pred_bdi"] for r in records],
        video_ids=[r["video_id"] for r in records],
    )
    return tmp_path / "lw.npz"


def test_pearson_and_spearman_basics():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 4.0, 6.0, 8.0]
    assert abs(_pearson(a, b) - 1.0) < 1e-9
    assert abs(_spearman(a, b) - 1.0) < 1e-9
    # Inverse correlation.
    assert _pearson(a, [-x for x in b]) == pytest.approx(-1.0, abs=1e-9)


def test_rank_handles_ties_with_average_ranks():
    ranks = _rank([10.0, 20.0, 20.0, 30.0])
    # Two tied at value 20.0 share ranks 2 and 3 -> average 2.5.
    assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_compute_coupling_paired_identity_similarity():
    """Same-subject paired videos get high identity similarity."""
    records = _records()
    # Subject 101 features near-identical; subject 102 near-identical; subjects differ.
    feats = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.01],
        ],
        dtype=float,
    )
    rows = compute_error_identity_coupling(records, feats, per_query_rows=None)
    assert len(rows) == 4
    # Each video's paired task partner is the other video of the same subject.
    sims = [r["paired_identity_sim"] for r in rows]
    assert all(s is not None and s > 0.9 for s in sims), sims


def test_run_audit_emits_all_outputs(tmp_path):
    records = _records()
    feats = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.01],
        ],
        dtype=float,
    )
    npz = _layer_npz(tmp_path, feats, "layer_shared")
    pred_csv = _prediction_csv(tmp_path, records)

    generated = run_error_identity_coupling_audit(
        features_npz=npz,
        predictions_csv=pred_csv,
        output_dir=tmp_path / "out",
        per_query_csv=None,
    )

    corr_path = tmp_path / "out" / "tables" / "error_identity_correlation.csv"
    sev_path = tmp_path / "out" / "tables" / "severity_bin_identity_error_summary.csv"
    cases_path = tmp_path / "out" / "tables" / "high_error_high_identity_cases.csv"
    report_path = tmp_path / "out" / "reports" / "identity_error_coupling_report.md"
    assert corr_path in generated
    assert sev_path in generated
    assert cases_path in generated
    assert report_path in generated
    for p in (corr_path, sev_path, cases_path, report_path):
        assert p.exists()

    corr_rows = read_csv_rows(corr_path)
    assert len(corr_rows) == 1
    assert corr_rows[0]["layer_name"] == "layer_shared"
    assert int(corr_rows[0]["n"]) == 4

    # Headers contain all declared columns.
    with open(corr_path) as f:
        header = next(csv.reader(f))
    assert set(CORRELATION_COLUMNS).issubset(set(header))


def test_severity_bin_summary_covers_groups(tmp_path):
    records = _records()
    feats = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.01],
        ],
        dtype=float,
    )
    npz = _layer_npz(tmp_path, feats, "layer_shared")
    pred_csv = _prediction_csv(tmp_path, records)

    run_error_identity_coupling_audit(
        features_npz=npz,
        predictions_csv=pred_csv,
        output_dir=tmp_path / "out",
    )
    sev_rows = read_csv_rows(tmp_path / "out" / "tables" / "severity_bin_identity_error_summary.csv")
    groups = {row["severity_group"] for row in sev_rows}
    assert groups == {"minimal", "severe"}
    with open(tmp_path / "out" / "tables" / "severity_bin_identity_error_summary.csv") as f:
        header = next(csv.reader(f))
    assert set(SEVERITY_BIN_COLUMNS).issubset(set(header))


def test_high_coupling_cases_selected(tmp_path):
    """The video with the largest abs_error and highest id_sim ranks first."""
    records = _records()
    # Subject 101 paired videos are very close (high id_sim); subject 102 far.
    # Video 0 has the largest abs_error (18-10=8).
    feats = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.001, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    npz = _layer_npz(tmp_path, feats, "layer_shared")
    pred_csv = _prediction_csv(tmp_path, records)

    run_error_identity_coupling_audit(
        features_npz=npz,
        predictions_csv=pred_csv,
        output_dir=tmp_path / "out",
        max_cases=2,
    )
    case_rows = read_csv_rows(tmp_path / "out" / "tables" / "high_error_high_identity_cases.csv")
    assert len(case_rows) <= 2
    # The top case should be a subject-101 video (high id_sim) with high abs_error.
    assert case_rows[0]["subject_id"] == "101"
    assert float(case_rows[0]["coupling_score"]) > 0.0
    with open(tmp_path / "out" / "tables" / "high_error_high_identity_cases.csv") as f:
        header = next(csv.reader(f))
    assert set(HIGH_COUPLING_CASE_COLUMNS).issubset(set(header))


def test_coupling_with_per_query_csv_enriches_rank(tmp_path):
    """When A1 per_query CSV is supplied, paired_rank / severity_agree are used."""
    records = _records()
    feats = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.01],
        ],
        dtype=float,
    )
    npz = _layer_npz(tmp_path, feats, "layer_shared")
    pred_csv = _prediction_csv(tmp_path, records)

    # Minimal A1 per_query CSV with rank/agreement signals. Ranks vary across
    # videos so Spearman correlation is computable (constant ranks yield None).
    pq_rows = [
        {"layer_name": "layer_shared", "query_video_id": "101_1_Freeform_video", "paired_task_rank": 1, "paired_task_in_top5": 1, "severity_neighbor_agreement_top5": 0.4},
        {"layer_name": "layer_shared", "query_video_id": "101_1_Northwind_video", "paired_task_rank": 2, "paired_task_in_top5": 1, "severity_neighbor_agreement_top5": 0.4},
        {"layer_name": "layer_shared", "query_video_id": "102_1_Freeform_video", "paired_task_rank": 1, "paired_task_in_top5": 1, "severity_neighbor_agreement_top5": 0.6},
        {"layer_name": "layer_shared", "query_video_id": "102_1_Northwind_video", "paired_task_rank": 2, "paired_task_in_top5": 1, "severity_neighbor_agreement_top5": 0.6},
    ]
    pq_path = tmp_path / "per_query.csv"
    write_csv_rows(
        pq_path,
        pq_rows,
        ["layer_name", "query_video_id", "paired_task_rank", "paired_task_in_top5", "severity_neighbor_agreement_top5"],
    )

    run_error_identity_coupling_audit(
        features_npz=npz,
        predictions_csv=pred_csv,
        output_dir=tmp_path / "out",
        per_query_csv=pq_path,
    )
    corr_rows = read_csv_rows(tmp_path / "out" / "tables" / "error_identity_correlation.csv")
    # With per_query supplied and varying ranks, rank/severity columns populate.
    assert corr_rows[0]["corr_abs_error_vs_paired_rank"] != ""
    assert corr_rows[0]["corr_residual_vs_severity_agree"] != ""
