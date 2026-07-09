"""Tests for the RPDF Stage A1 layer-wise identity retrieval aggregation.

These tests exercise the pure-numpy aggregation in
``src.diagnostics.layerwise_identity``: multi-key NPZ round-trip, per-layer
retrieval via the existing ``compute_identity_retrieval_metrics``, layer
ordering, and report generation.  No torch is required.
"""

import numpy as np
import pytest

from src.diagnostics.layerwise_identity import (
    PER_QUERY_COLUMNS,
    SUMMARY_COLUMNS,
    load_layerwise_features_npz,
    run_layerwise_identity_audit,
    save_layerwise_features_npz,
)


def _make_subject_separable_features(n=4, dim=6, seed=0):
    """Build features where same-subject videos are near-identical.

    Subjects 101 and 102 each have a Freeform and a Northwind video.  Within a
    subject the two videos share a base vector (so same-subject retrieval is
    strong), while subjects differ.
    """
    rng = np.random.RandomState(seed)
    base_101 = rng.randn(dim)
    base_102 = rng.randn(dim)
    # Small perturbation so paired-task videos are not identical but close.
    feats = np.stack(
        [
            base_101 + 0.01 * rng.randn(dim),  # 101 Freeform
            base_101 + 0.01 * rng.randn(dim),  # 101 Northwind
            base_102 + 0.01 * rng.randn(dim),  # 102 Freeform
            base_102 + 0.01 * rng.randn(dim),  # 102 Northwind
        ]
    )
    return feats.astype(float)


def _make_records():
    return [
        {"video_id": "101_1_Freeform_video", "subject_id": "101", "task_name": "Freeform", "true_bdi": 10.0, "pred_bdi": 12.0, "severity_group": "minimal"},
        {"video_id": "101_1_Northwind_video", "subject_id": "101", "task_name": "Northwind", "true_bdi": 10.0, "pred_bdi": 11.0, "severity_group": "minimal"},
        {"video_id": "102_1_Freeform_video", "subject_id": "102", "task_name": "Freeform", "true_bdi": 30.0, "pred_bdi": 28.0, "severity_group": "severe"},
        {"video_id": "102_1_Northwind_video", "subject_id": "102", "task_name": "Northwind", "true_bdi": 30.0, "pred_bdi": 27.0, "severity_group": "severe"},
    ]


def _metadata_lists(records):
    video_ids = [r["video_id"] for r in records]
    subject_ids = [r["subject_id"] for r in records]
    targets = [r["true_bdi"] for r in records]
    preds = [r["pred_bdi"] for r in records]
    return video_ids, subject_ids, targets, preds


def test_save_and_load_layerwise_npz_roundtrip(tmp_path):
    records = _make_records()
    video_ids, subject_ids, targets, preds = _metadata_lists(records)
    layer_features = {
        "layer_stem": _make_subject_separable_features(dim=6, seed=1),
        "layer_block_3": _make_subject_separable_features(dim=8, seed=2),
        "layer_shared": _make_subject_separable_features(dim=4, seed=3),
    }

    npz_path = tmp_path / "layerwise.npz"
    save_layerwise_features_npz(
        npz_path, layer_features, subject_ids, targets, preds, video_ids=video_ids
    )

    loaded_features, loaded_records = load_layerwise_features_npz(npz_path)
    assert set(loaded_features.keys()) == set(layer_features.keys())
    for name in layer_features:
        assert loaded_features[name].shape == layer_features[name].shape
        np.testing.assert_allclose(loaded_features[name], layer_features[name])
    assert len(loaded_records) == len(records)
    assert [r["subject_id"] for r in loaded_records] == subject_ids
    assert [r["video_id"] for r in loaded_records] == video_ids


def test_layerwise_audit_runs_retrieval_per_layer(tmp_path):
    records = _make_records()
    video_ids, subject_ids, targets, preds = _metadata_lists(records)
    # All layers use separable features so retrieval should be strong.
    layer_features = {
        "layer_stem": _make_subject_separable_features(dim=6, seed=1),
        "layer_block_3": _make_subject_separable_features(dim=8, seed=2),
        "layer_shared": _make_subject_separable_features(dim=4, seed=3),
    }
    npz_path = tmp_path / "layerwise.npz"
    save_layerwise_features_npz(
        npz_path, layer_features, subject_ids, targets, preds, video_ids=video_ids
    )

    generated = run_layerwise_identity_audit(npz_path, tmp_path / "out", top_k=3)

    summary_path = tmp_path / "out" / "tables" / "layerwise_identity_summary.csv"
    per_query_path = tmp_path / "out" / "tables" / "layerwise_identity_per_query.csv"
    report_path = tmp_path / "out" / "reports" / "layerwise_identity_report.md"
    assert summary_path in generated
    assert per_query_path in generated
    assert report_path in generated
    assert summary_path.exists() and per_query_path.exists() and report_path.exists()

    from src.diagnostics.io import read_csv_rows

    summary_rows = read_csv_rows(summary_path)
    assert len(summary_rows) == 3
    assert {row["layer_name"] for row in summary_rows} == set(layer_features.keys())
    # Same-subject features -> top1 retrieval should be perfect across layers.
    for row in summary_rows:
        top1 = float(row["same_subject_top1_rate"])
        assert top1 == 1.0, f"{row['layer_name']} expected top1=1.0, got {top1}"


def test_layer_ordering_follows_probe_constant(tmp_path):
    """Output layers appear in LAYERWISE_PROBE_LAYERS order, not NPZ order."""
    records = _make_records()
    video_ids, subject_ids, targets, preds = _metadata_lists(records)
    # Insert in reversed order to verify reordering.
    layer_features = {
        "layer_shared": _make_subject_separable_features(dim=4, seed=3),
        "layer_block_3": _make_subject_separable_features(dim=8, seed=2),
        "layer_stem": _make_subject_separable_features(dim=6, seed=1),
    }
    npz_path = tmp_path / "layerwise.npz"
    save_layerwise_features_npz(
        npz_path, layer_features, subject_ids, targets, preds, video_ids=video_ids
    )

    run_layerwise_identity_audit(npz_path, tmp_path / "out", top_k=3)
    from src.diagnostics.io import read_csv_rows

    summary_rows = read_csv_rows(tmp_path / "out" / "tables" / "layerwise_identity_summary.csv")
    ordered_names = [row["layer_name"] for row in summary_rows]
    # layer_stem < layer_block_3 < layer_shared in LAYERWISE_PROBE_LAYERS.
    assert ordered_names == ["layer_stem", "layer_block_3", "layer_shared"]


def test_per_query_csv_has_layer_column_and_expected_fields(tmp_path):
    records = _make_records()
    video_ids, subject_ids, targets, preds = _metadata_lists(records)
    layer_features = {"layer_shared": _make_subject_separable_features(dim=4, seed=3)}
    npz_path = tmp_path / "layerwise.npz"
    save_layerwise_features_npz(
        npz_path, layer_features, subject_ids, targets, preds, video_ids=video_ids
    )

    run_layerwise_identity_audit(npz_path, tmp_path / "out", top_k=3)
    from src.diagnostics.io import read_csv_rows

    rows = read_csv_rows(tmp_path / "out" / "tables" / "layerwise_identity_per_query.csv")
    assert len(rows) == 4  # one per query video
    assert all(row["layer_name"] == "layer_shared" for row in rows)
    assert "paired_task_rank" in rows[0]
    assert "top1_neighbor_same_subject" in rows[0]
    # All required per-query columns are present in the header.
    import csv

    with open(tmp_path / "out" / "tables" / "layerwise_identity_per_query.csv") as f:
        header = next(csv.reader(f))
    assert set(PER_QUERY_COLUMNS).issubset(set(header))


def test_summary_csv_has_expected_columns(tmp_path):
    records = _make_records()
    video_ids, subject_ids, targets, preds = _metadata_lists(records)
    layer_features = {"layer_shared": _make_subject_separable_features(dim=4, seed=3)}
    npz_path = tmp_path / "layerwise.npz"
    save_layerwise_features_npz(
        npz_path, layer_features, subject_ids, targets, preds, video_ids=video_ids
    )

    run_layerwise_identity_audit(npz_path, tmp_path / "out", top_k=3)
    import csv

    with open(tmp_path / "out" / "tables" / "layerwise_identity_summary.csv") as f:
        header = next(csv.reader(f))
    assert set(SUMMARY_COLUMNS).issubset(set(header))


def test_load_rejects_npz_without_layer_features(tmp_path):
    # A single-layer NPZ (no features_* keys) should raise.
    npz_path = tmp_path / "single.npz"
    np.savez_compressed(
        npz_path,
        features=_make_subject_separable_features(dim=4),
        subject_ids=np.asarray(["101", "101", "102", "102"]),
        true_bdi=np.asarray([10.0, 10.0, 30.0, 30.0]),
        pred_bdi=np.asarray([12.0, 11.0, 28.0, 27.0]),
    )
    with pytest.raises(ValueError, match="No per-layer feature arrays"):
        load_layerwise_features_npz(npz_path)


def test_convenience_single_run_files_for_shared_layer(tmp_path):
    """The audit also emits single-run-schema files for the layer_shared row.

    summarize_identity_retrieval_runs reads tables/embedding_identity_summary.csv
    (one row, single-run schema).  The layerwise probe must produce it from the
    layer_shared row so a separate single-run audit pass is not needed.
    """
    from src.diagnostics.identity_retrieval import (
        RETRIEVAL_COLUMNS,
        SUMMARY_COLUMNS as SINGLE_SUMMARY_COLUMNS,
    )
    from src.diagnostics.io import read_csv_rows

    records = _make_records()
    video_ids, subject_ids, targets, preds = _metadata_lists(records)
    layer_features = {
        "layer_stem": _make_subject_separable_features(dim=6, seed=1),
        "layer_shared": _make_subject_separable_features(dim=4, seed=3),
    }
    npz_path = tmp_path / "layerwise.npz"
    save_layerwise_features_npz(
        npz_path, layer_features, subject_ids, targets, preds, video_ids=video_ids
    )

    generated = run_layerwise_identity_audit(npz_path, tmp_path / "out", top_k=3)

    emb_summary = tmp_path / "out" / "tables" / "embedding_identity_summary.csv"
    emb_retrieval = tmp_path / "out" / "tables" / "embedding_identity_retrieval.csv"
    assert emb_summary in generated
    assert emb_retrieval in generated
    assert emb_summary.exists() and emb_retrieval.exists()

    # Summary is a single row, single-run schema (no layer_name column).
    summary_rows = read_csv_rows(emb_summary)
    assert len(summary_rows) == 1
    assert "layer_name" not in summary_rows[0]
    assert set(SINGLE_SUMMARY_COLUMNS).issubset(set(summary_rows[0].keys()))

    # Per-query carries only the layer_shared rows (one per query), single-run
    # schema, and its metrics match the layer_shared row of the layerwise table.
    per_query_rows = read_csv_rows(emb_retrieval)
    assert len(per_query_rows) == len(records)
    assert "layer_name" not in per_query_rows[0]
    assert set(RETRIEVAL_COLUMNS).issubset(set(per_query_rows[0].keys()))

    layerwise_summary = read_csv_rows(
        tmp_path / "out" / "tables" / "layerwise_identity_summary.csv"
    )
    shared_row = next(r for r in layerwise_summary if r["layer_name"] == "layer_shared")
    assert (
        float(summary_rows[0]["same_subject_top1_rate"])
        == float(shared_row["same_subject_top1_rate"])
    )


def test_no_convenience_files_when_shared_layer_absent(tmp_path):
    """No embedding_identity_* files when layer_shared is not probed."""
    records = _make_records()
    video_ids, subject_ids, targets, preds = _metadata_lists(records)
    layer_features = {
        "layer_stem": _make_subject_separable_features(dim=6, seed=1),
        "layer_block_3": _make_subject_separable_features(dim=8, seed=2),
    }
    npz_path = tmp_path / "layerwise.npz"
    save_layerwise_features_npz(
        npz_path, layer_features, subject_ids, targets, preds, video_ids=video_ids
    )

    generated = run_layerwise_identity_audit(npz_path, tmp_path / "out", top_k=3)

    emb_summary = tmp_path / "out" / "tables" / "embedding_identity_summary.csv"
    emb_retrieval = tmp_path / "out" / "tables" / "embedding_identity_retrieval.csv"
    assert emb_summary not in generated
    assert emb_retrieval not in generated
    assert not emb_summary.exists() and not emb_retrieval.exists()
