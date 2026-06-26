"""Tests for RPDF Stage A3: artifact weak-label audit.

Pure-numpy/csv tests verifying cross-source joining by normalized video_id,
weak-label correlation with prediction targets, and the z_art entry-decision
classification (error-coupled vs label-only confound).
"""

import csv

import pytest

from src.diagnostics.artifact_weaklabels import (
    CORRELATION_COLUMNS,
    _normalize_video_id,
    _pearson,
    build_weaklabel_summary,
    compute_weaklabel_correlations,
    run_artifact_weaklabel_audit,
)
from src.diagnostics.io import write_csv_rows


def _write_csv(path, rows, fieldnames):
    write_csv_rows(path, rows, fieldnames)
    return path


def test_normalize_video_id_strips_aligned_and_csv():
    assert _normalize_video_id("301_1_Freeform_video_aligned") == "301_1_Freeform_video"
    assert _normalize_video_id("301_1_Freeform_video_aligned.csv") == "301_1_Freeform_video"
    assert _normalize_video_id("301_1_Freeform_video") == "301_1_Freeform_video"


def test_pearson_basic():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 4.0, 6.0, 8.0]
    assert _pearson(a, b) == pytest.approx(1.0, abs=1e-9)
    assert _pearson(a, [-x for x in b]) == pytest.approx(-1.0, abs=1e-9)
    assert _pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None  # zero variance


def _base_predictions(tmp_path, n=8):
    """Predictions where abs_error is monotonic in the index."""
    rows = []
    for i in range(n):
        vid = f"{300 + i:03d}_1_Freeform_video"
        true = 20.0
        # pred drifts with i so abs_error grows with i.
        pred = 20.0 + (i - 3) * 1.5
        rows.append(
            {
                "video_id": vid,
                "subject_id": f"{300 + i:03d}",
                "task_name": "Freeform",
                "true_bdi": true,
                "pred_bdi": pred,
                "residual": pred - true,
                "abs_error": abs(pred - true),
                "severity_group": "minimal" if i % 2 == 0 else "severe",
            }
        )
    path = tmp_path / "predictions.csv"
    _write_csv(
        path,
        rows,
        ["video_id", "subject_id", "task_name", "true_bdi", "pred_bdi", "residual", "abs_error", "severity_group"],
    )
    return path, rows


def test_build_summary_joins_by_normalized_video_id(tmp_path):
    pred_path, pred_rows = _base_predictions(tmp_path, n=4)
    # black_artifacts uses _aligned suffix on video_id to test normalization.
    black_rows = [
        {"video_id": f"{300 + i:03d}_1_Freeform_video_aligned", "black_border_ratio_mean": float(i) * 0.1}
        for i in range(4)
    ]
    black_path = _write_csv(
        tmp_path / "black.csv",
        black_rows,
        ["video_id", "black_border_ratio_mean"],
    )

    summary_rows, fields_by_source = build_weaklabel_summary(
        predictions_csv=pred_path,
        black_artifacts_csv=black_path,
    )
    # All 4 prediction videos matched via normalization.
    assert len(summary_rows) == 4
    assert "black_border_ratio_mean" in fields_by_source["black_artifacts"]
    # The joined weak label is prefixed with its source.
    assert all("black_artifacts:black_border_ratio_mean" in r for r in summary_rows)
    # Values carried through correctly.
    for i, row in enumerate(summary_rows):
        assert row["black_artifacts:black_border_ratio_mean"] == pytest.approx(i * 0.1)


def test_compute_correlations_flags_error_coupled_label(tmp_path):
    """A weak label that co-varies with abs_error is flagged error-coupled."""
    pred_path, pred_rows = _base_predictions(tmp_path, n=8)
    # weak label = abs_error exactly -> corr should be ~1.0.
    black_rows = [
        {"video_id": r["video_id"], "black_border_ratio_mean": r["abs_error"]}
        for r in pred_rows
    ]
    black_path = _write_csv(
        tmp_path / "black.csv", black_rows, ["video_id", "black_border_ratio_mean"]
    )

    summary_rows, fields_by_source = build_weaklabel_summary(
        predictions_csv=pred_path, black_artifacts_csv=black_path
    )
    corr_rows = compute_weaklabel_correlations(summary_rows, fields_by_source)
    border_corr = next(
        r for r in corr_rows if r["weaklabel_name"] == "black_artifacts:black_border_ratio_mean"
    )
    assert border_corr["corr_with_abs_error"] == pytest.approx(1.0, abs=1e-6)
    assert border_corr["abs_corr_with_abs_error"] >= 0.2


def test_compute_correlations_flags_label_only_confound(tmp_path):
    """A weak label correlated with true_bdi but not abs_error is label-only."""
    # Build predictions where true_bdi varies but abs_error is constant.
    rows = []
    for i in range(6):
        vid = f"{300 + i:03d}_1_Freeform_video"
        true = 10.0 + i * 5.0
        pred = true + 2.0  # constant error -> abs_error constant
        rows.append(
            {
                "video_id": vid, "subject_id": f"{300 + i:03d}", "task_name": "Freeform",
                "true_bdi": true, "pred_bdi": pred, "residual": 2.0, "abs_error": 2.0,
                "severity_group": "minimal",
            }
        )
    pred_path = tmp_path / "predictions.csv"
    _write_csv(pred_path, rows, ["video_id", "subject_id", "task_name", "true_bdi", "pred_bdi", "residual", "abs_error", "severity_group"])
    # weak label co-varies with true_bdi, not abs_error.
    black_rows = [{"video_id": r["video_id"], "black_ratio_mean": r["true_bdi"]} for r in rows]
    black_path = _write_csv(tmp_path / "black.csv", black_rows, ["video_id", "black_ratio_mean"])

    summary_rows, fields_by_source = build_weaklabel_summary(
        predictions_csv=pred_path, black_artifacts_csv=black_path
    )
    corr_rows = compute_weaklabel_correlations(summary_rows, fields_by_source)
    ratio_corr = next(
        r for r in corr_rows if r["weaklabel_name"] == "black_artifacts:black_ratio_mean"
    )
    assert ratio_corr["corr_with_true_bdi"] == pytest.approx(1.0, abs=1e-6)
    # abs_error is constant -> correlation is None (zero variance).
    assert ratio_corr["corr_with_abs_error"] is None


def test_run_audit_emits_all_outputs(tmp_path):
    pred_path, pred_rows = _base_predictions(tmp_path, n=6)
    black_rows = [
        {"video_id": r["video_id"], "black_border_ratio_mean": r["abs_error"]}
        for r in pred_rows
    ]
    black_path = _write_csv(tmp_path / "black.csv", black_rows, ["video_id", "black_border_ratio_mean"])

    generated = run_artifact_weaklabel_audit(
        predictions_csv=pred_path,
        output_dir=tmp_path / "out",
        black_artifacts_csv=black_path,
    )
    summary_path = tmp_path / "out" / "tables" / "artifact_weaklabel_summary.csv"
    corr_path = tmp_path / "out" / "tables" / "artifact_weaklabel_correlation.csv"
    report_path = tmp_path / "out" / "reports" / "artifact_weaklabel_report.md"
    assert summary_path in generated
    assert corr_path in generated
    assert report_path in generated
    for p in (summary_path, corr_path, report_path):
        assert p.exists()

    with open(corr_path) as f:
        header = next(csv.reader(f))
    assert set(CORRELATION_COLUMNS).issubset(set(header))


def test_run_audit_handles_missing_sources(tmp_path):
    """Only predictions + one source still produces a valid audit."""
    pred_path, _ = _base_predictions(tmp_path, n=4)
    black_rows = [{"video_id": f"{300 + i:03d}_1_Freeform_video", "black_ratio_mean": 0.1 * i} for i in range(4)]
    black_path = _write_csv(tmp_path / "black.csv", black_rows, ["video_id", "black_ratio_mean"])

    generated = run_artifact_weaklabel_audit(
        predictions_csv=pred_path,
        output_dir=tmp_path / "out",
        black_artifacts_csv=black_path,
        # alignment_geometry / openface_quality / temporal_sampling omitted.
    )
    assert any("artifact_weaklabel_summary.csv" in str(p) for p in generated)
    from src.diagnostics.io import read_csv_rows

    corr_rows = read_csv_rows(tmp_path / "out" / "tables" / "artifact_weaklabel_correlation.csv")
    # Only black_artifacts source contributed.
    sources = {r["source"] for r in corr_rows}
    assert sources == {"black_artifacts"}


def test_run_audit_report_documents_z_art_decision(tmp_path):
    pred_path, pred_rows = _base_predictions(tmp_path, n=6)
    black_rows = [{"video_id": r["video_id"], "black_border_ratio_mean": r["abs_error"]} for r in pred_rows]
    black_path = _write_csv(tmp_path / "black.csv", black_rows, ["video_id", "black_border_ratio_mean"])

    run_artifact_weaklabel_audit(
        predictions_csv=pred_path,
        output_dir=tmp_path / "out",
        black_artifacts_csv=black_path,
    )
    report = (tmp_path / "out" / "reports" / "artifact_weaklabel_report.md").read_text(encoding="utf-8")
    assert "z_art Entry Decision" in report
    assert "black_artifacts:black_border_ratio_mean" in report
