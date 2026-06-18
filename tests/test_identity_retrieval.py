import csv
import numpy as np
import pytest

from src.diagnostics.identity_retrieval import (
    compute_identity_retrieval_metrics,
    load_embeddings_and_predictions,
    plot_embedding_similarity_matrix,
    run_embedding_identity_retrieval_audit,
    write_embedding_identity_report,
    write_embedding_identity_retrieval,
    write_embedding_identity_summary,
    _cosine_similarity_matrix,
    _paired_task_index,
    _rank_of_target,
    _subject_index_groups,
    _task_name_from_video_id,
)
from src.diagnostics.io import save_features_npz, write_prediction_table


def _make_records():
    """Return six synthetic records representing three paired subjects."""
    return [
        {
            "video_id": "203_1_Freeform_video",
            "subject_id": "203_1",
            "task_name": "Freeform",
            "true_bdi": 8.0,
            "pred_bdi": 10.0,
            "severity_group": "minimal",
        },
        {
            "video_id": "203_1_Northwind_video",
            "subject_id": "203_1",
            "task_name": "Northwind",
            "true_bdi": 9.0,
            "pred_bdi": 11.0,
            "severity_group": "minimal",
        },
        {
            "video_id": "204_1_Freeform_video",
            "subject_id": "204_1",
            "task_name": "Freeform",
            "true_bdi": 22.0,
            "pred_bdi": 20.0,
            "severity_group": "moderate",
        },
        {
            "video_id": "204_1_Northwind_video",
            "subject_id": "204_1",
            "task_name": "Northwind",
            "true_bdi": 23.0,
            "pred_bdi": 21.0,
            "severity_group": "moderate",
        },
        {
            "video_id": "205_1_Freeform_video",
            "subject_id": "205_1",
            "task_name": "Freeform",
            "true_bdi": 35.0,
            "pred_bdi": 30.0,
            "severity_group": "severe",
        },
        {
            "video_id": "205_1_Northwind_video",
            "subject_id": "205_1",
            "task_name": "Northwind",
            "true_bdi": 36.0,
            "pred_bdi": 31.0,
            "severity_group": "severe",
        },
    ]


def _make_identity_features(records, noise_scale=0.01, seed=42):
    """Build embeddings where same-subject videos are close, others are far."""
    rng = np.random.default_rng(seed)
    subject_vectors = {}
    for subject_id in sorted({record["subject_id"] for record in records}):
        subject_vectors[subject_id] = rng.standard_normal(16)

    features = []
    for record in records:
        vec = subject_vectors[record["subject_id"]] + rng.normal(scale=noise_scale, size=16)
        # Make task-specific direction small but nonzero.
        if record["task_name"] == "Northwind":
            vec += rng.normal(scale=noise_scale * 0.5, size=16)
        features.append(vec)
    return np.asarray(features, dtype=float)


def test_task_name_from_video_id():
    assert _task_name_from_video_id("203_1_Freeform_video") == "Freeform"
    assert _task_name_from_video_id("203_1_Northwind_video") == "Northwind"
    assert _task_name_from_video_id("203_1_video") == ""


def test_subject_index_groups():
    records = _make_records()
    groups = _subject_index_groups(records)
    assert set(groups.keys()) == {"203_1", "204_1", "205_1"}
    assert len(groups["203_1"]) == 2


def test_paired_task_index_finds_opposite_task():
    records = _make_records()
    groups = _subject_index_groups(records)
    assert _paired_task_index(0, records, groups) == 1
    assert _paired_task_index(1, records, groups) == 0
    assert _paired_task_index(2, records, groups) == 3


def test_rank_of_target_excludes_self():
    similarities = np.asarray(
        [
            [-1.0, 0.9, 0.5, 0.1],
            [0.9, -1.0, 0.3, 0.2],
            [0.5, 0.3, -1.0, 0.8],
            [0.1, 0.2, 0.8, -1.0],
        ],
        dtype=float,
    )
    # For query 0, sorted similarities (excluding self) are idx1(0.9), idx2(0.5), idx3(0.1).
    assert _rank_of_target(similarities[0], 1) == 1
    assert _rank_of_target(similarities[0], 2) == 2
    assert _rank_of_target(similarities[0], 3) == 3


def test_cosine_similarity_matrix_masks_self():
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    sim = _cosine_similarity_matrix(features)
    assert sim.shape == (3, 3)
    np.testing.assert_allclose(np.diag(sim), -1.0)
    # Orthogonal vectors should have near-zero similarity.
    assert sim[0, 1] == pytest.approx(0.0, abs=1e-8)
    assert sim[0, 2] == pytest.approx(np.sqrt(2) / 2, abs=1e-8)


def test_compute_identity_retrieval_metrics_finds_identity_shortcut():
    records = _make_records()
    features = _make_identity_features(records, noise_scale=0.01)
    rows, summary = compute_identity_retrieval_metrics(records, features, top_k=5)

    assert len(rows) == len(records)
    assert summary["num_queries"] == len(records)
    assert summary["num_subjects"] == 3
    assert summary["num_paired_subjects"] == 3

    # Same-subject retrieval should be near-perfect with strong identity features.
    assert summary["same_subject_top1_rate"] == pytest.approx(1.0, abs=0.1)
    assert summary["same_subject_top3_rate"] == pytest.approx(1.0, abs=0.05)
    assert summary["same_subject_top5_rate"] == pytest.approx(1.0, abs=0.05)

    # Paired-task retrieval should also be strong.
    assert summary["paired_task_rank_mean"] <= 1.5
    assert summary["paired_task_in_top1_rate"] == pytest.approx(1.0, abs=0.1)


def test_compute_identity_retrieval_metrics_with_random_features():
    rng = np.random.default_rng(0)
    records = _make_records()
    features = rng.standard_normal((len(records), 16))
    rows, summary = compute_identity_retrieval_metrics(records, features, top_k=5)

    assert len(rows) == len(records)
    # With random features same-subject top-1 should be around chance (~1/5 = 0.2).
    assert 0.0 <= summary["same_subject_top1_rate"] <= 0.6


def test_load_embeddings_and_predictions_from_npz(tmp_path):
    records = _make_records()
    features = _make_identity_features(records)
    npz_path = tmp_path / "test_features.npz"
    save_features_npz(
        npz_path,
        features=features,
        subject_ids=[r["subject_id"] for r in records],
        targets=[r["true_bdi"] for r in records],
        preds=[r["pred_bdi"] for r in records],
        video_ids=[r["video_id"] for r in records],
    )

    loaded_features, loaded_records = load_embeddings_and_predictions(npz_path)
    np.testing.assert_allclose(loaded_features, features)
    assert len(loaded_records) == len(records)
    assert loaded_records[0]["video_id"] == "203_1_Freeform_video"
    assert loaded_records[0]["task_name"] == "Freeform"


def test_load_embeddings_and_predictions_fallback_to_predictions_csv(tmp_path):
    records = _make_records()
    features = _make_identity_features(records)
    npz_path = tmp_path / "test_features.npz"
    save_features_npz(
        npz_path,
        features=features,
        subject_ids=[r["subject_id"] for r in records],
        targets=[r["true_bdi"] for r in records],
        preds=[r["pred_bdi"] for r in records],
        # Intentionally omit video_ids to exercise the CSV fallback path.
    )
    predictions_csv = tmp_path / "test_predictions.csv"
    write_prediction_table(
        predictions_csv,
        subject_ids=[r["subject_id"] for r in records],
        targets=[r["true_bdi"] for r in records],
        preds=[r["pred_bdi"] for r in records],
        video_ids=[r["video_id"] for r in records],
        task_names=[r["task_name"] for r in records],
    )

    loaded_features, loaded_records = load_embeddings_and_predictions(
        npz_path, predictions_csv=predictions_csv
    )
    np.testing.assert_allclose(loaded_features, features)
    assert loaded_records[0]["video_id"] == "203_1_Freeform_video"
    assert loaded_records[0]["task_name"] == "Freeform"


def test_load_embeddings_and_predictions_detects_length_mismatch(tmp_path):
    records = _make_records()
    features = _make_identity_features(records)
    npz_path = tmp_path / "test_features.npz"
    save_features_npz(
        npz_path,
        features=features,
        subject_ids=[r["subject_id"] for r in records],
        targets=[r["true_bdi"] for r in records],
        preds=[r["pred_bdi"] for r in records],
    )
    predictions_csv = tmp_path / "bad_predictions.csv"
    write_prediction_table(
        predictions_csv,
        subject_ids=[records[0]["subject_id"]],
        targets=[records[0]["true_bdi"]],
        preds=[records[0]["pred_bdi"]],
        video_ids=[records[0]["video_id"]],
    )

    with pytest.raises(ValueError, match="does not match"):
        load_embeddings_and_predictions(npz_path, predictions_csv=predictions_csv)


def test_write_embedding_identity_retrieval(tmp_path):
    records = _make_records()
    features = _make_identity_features(records)
    rows, _ = compute_identity_retrieval_metrics(records, features)
    csv_path = tmp_path / "retrieval.csv"
    write_embedding_identity_retrieval(rows, csv_path)
    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        loaded = list(reader)
    assert len(loaded) == len(records)
    assert "query_video_id" in loaded[0]
    assert "top1_neighbor_same_subject" in loaded[0]


def test_write_embedding_identity_summary(tmp_path):
    records = _make_records()
    features = _make_identity_features(records)
    _, summary = compute_identity_retrieval_metrics(records, features)
    csv_path = tmp_path / "summary.csv"
    write_embedding_identity_summary(summary, csv_path)
    assert csv_path.exists()


def test_plot_embedding_similarity_matrix(tmp_path):
    records = _make_records()
    features = _make_identity_features(records)
    sim = _cosine_similarity_matrix(features)
    save_path = tmp_path / "figures" / "sim.png"
    result = plot_embedding_similarity_matrix(sim, records, save_path)
    assert result is not None
    assert result.exists()


def test_write_embedding_identity_report(tmp_path):
    records = _make_records()
    features = _make_identity_features(records)
    rows, summary = compute_identity_retrieval_metrics(records, features)
    report_path = tmp_path / "reports" / "report.md"
    write_embedding_identity_report(
        report_path,
        generated_files=[],
        summary=summary,
        per_query_rows=rows,
        top_k=5,
    )
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "Embedding Identity Retrieval Audit Report" in text
    assert f"Queries: {summary['num_queries']}" in text


def test_run_embedding_identity_retrieval_audit(tmp_path):
    records = _make_records()
    features = _make_identity_features(records)
    npz_path = tmp_path / "test_features.npz"
    save_features_npz(
        npz_path,
        features=features,
        subject_ids=[r["subject_id"] for r in records],
        targets=[r["true_bdi"] for r in records],
        preds=[r["pred_bdi"] for r in records],
        video_ids=[r["video_id"] for r in records],
    )

    generated = run_embedding_identity_retrieval_audit(
        features_npz=npz_path,
        output_dir=tmp_path / "identity_retrieval",
        top_k=5,
    )
    assert generated
    output = tmp_path / "identity_retrieval"
    assert (output / "tables" / "embedding_identity_retrieval.csv").exists()
    assert (output / "tables" / "embedding_identity_summary.csv").exists()
    assert (output / "figures" / "embedding_similarity_matrix.png").exists()
    assert (output / "reports" / "embedding_identity_report.md").exists()
