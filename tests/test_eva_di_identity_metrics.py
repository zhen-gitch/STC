"""CPU tests for the external identity metrics adapter (EVA-DI-CODE-v2)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.eva_di.contracts import EvaDiError
from src.eva_di.export_embeddings import EXPORT_SCHEMA
from src.eva_di.identity_metrics import (
    a1_retrieval, assert_common_run, evaluate_export_npz, lovo_ridge_attacker,
    main, pair_identity_auroc, subject_from_video_id)


def _write_npz(tmp_path, name, ids, emb, *, split="val", ckpt_sha="sha-A"):
    path = tmp_path / name
    np.savez(path, schema=EXPORT_SCHEMA, split=split,
             sample_ids=np.asarray(ids), embeddings=emb,
             run_id="r-unit", config_name="di-full",
             config_source_sha256="cfg-sha", checkpoint_sha256=ckpt_sha,
             input_mode="image", git_commit="abc123")
    return path.with_suffix("") if False else path  # npz keeps its suffix


def _clustered(seed=7, persons=3, per_person=4, dim=16):
    rng = np.random.default_rng(seed)
    ids, emb = [], []
    for p in range(persons):
        centroid = rng.normal(0, 1, dim) * 3
        for k in range(per_person):
            ids.append(f"20{p}_{k % 2 + 1}_Freeform_video")
            emb.append(centroid + rng.normal(0, 0.05, dim))
    return ids, np.asarray(emb, dtype=np.float64)


def test_subject_is_person_prefix():
    assert subject_from_video_id("203_1_Freeform_video") == "203"


def test_clustered_embeddings_are_perfectly_decodable(tmp_path):
    ids, emb = _clustered()
    p = _write_npz(tmp_path, "p_mean_val.npz", ids, emb)
    res = evaluate_export_npz(p, per_query_out_dir=tmp_path / "out")
    assert res["exit"] == "p_mean" and res["n_persons"] == 3
    assert res["attacker"]["top1_accuracy"] == 1.0
    assert res["attacker"]["coverage"] == 1.0
    assert res["pair_auroc"]["auc"] > 0.999
    assert res["a1_retrieval"]["top1_same_subject"] == 1.0
    assert (tmp_path / "out" / "p_mean_val_per_query.csv").is_file()
    assert res["provenance"]["checkpoint_sha256"] == "sha-A"


def test_noise_embeddings_near_chance(tmp_path):
    rng = np.random.default_rng(0)
    ids, emb = _clustered(seed=1)
    emb = rng.normal(0, 5, emb.shape)  # identity destroyed
    p = _write_npz(tmp_path, "v_h0_val.npz", ids, emb)
    res = evaluate_export_npz(p)
    assert res["attacker"]["top1_accuracy"] <= 0.5  # chance = 1/3
    assert 0.25 < res["pair_auroc"]["auc"] < 0.75


def test_singleton_person_is_reported_unseen_not_guessed(tmp_path):
    ids, emb = _clustered(persons=3, per_person=2)
    ids.append("999_1_Freeform_video")           # only recording of person 999
    emb = np.vstack([emb, np.zeros((1, emb.shape[1]))])
    rows, n_classes = lovo_ridge_attacker(emb, [subject_from_video_id(i) for i in ids])
    singleton = [r for r in rows if r["subject_id"] == "999"][0]
    assert singleton["seen"] is False and singleton["correct_top1"] is None
    assert n_classes == 4


def test_guards(tmp_path):
    ids, emb = _clustered()
    bad = tmp_path / "bad.npz"
    np.savez(bad, schema="other_v0", sample_ids=np.asarray(ids),
             embeddings=emb, split="val")
    with pytest.raises(EvaDiError, match="not an"):
        evaluate_export_npz(bad)
    test_split = _write_npz(tmp_path, "t.npz", ids, emb, split="test")
    with pytest.raises(EvaDiError, match="locked"):
        evaluate_export_npz(test_split)
    mixed_a = _write_npz(tmp_path, "a.npz", ids, emb, ckpt_sha="x")
    mixed_b = _write_npz(tmp_path, "b.npz", ids, emb, ckpt_sha="y")
    with pytest.raises(EvaDiError, match="mixed checkpoints"):
        assert_common_run([mixed_a, mixed_b])


def test_ridge_matches_sklearn_decision(tmp_path):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import RidgeClassifier

    from src.eva_di.identity_metrics import ridge_onehot_scores
    ids, emb = _clustered(seed=3)
    y = [subject_from_video_id(i) for i in ids]
    classes = sorted(set(y))
    scores = ridge_onehot_scores(emb, y, classes, emb[:5], alpha=1.0)
    clf = RidgeClassifier(alpha=1.0).fit(emb, [classes.index(c) for c in y])
    ref = clf.decision_function(emb[:5])
    ref = np.atleast_2d(ref)
    if ref.shape[1] == 1:  # sklearn collapses binary to signed distance
        ref = np.stack([-ref.ravel(), ref.ravel()], axis=1)
    # sklearn's RidgeClassifier regresses on {-1,+1} indicators, so its scores
    # equal ours scaled by 2 and shifted by the SAME constant per column; the
    # attacker only ever uses the ranking, so assert argmax equality plus the
    # exact affine relation.
    assert np.array_equal(scores.argmax(axis=1), ref.argmax(axis=1))
    assert np.allclose(scores, (ref + 1.0) / 2.0, atol=1e-6)


def test_main_roundtrip(tmp_path):
    ids, emb = _clustered()
    a = _write_npz(tmp_path, "p_mean_val.npz", ids, emb)
    b = _write_npz(tmp_path, "v_h0_val.npz", ids, emb + 0.01)
    rc = main(["--npz", str(a), str(b), "--output-dir", str(tmp_path / "id")])
    assert rc == 0
    report = json.loads((tmp_path / "id" / "identity_metrics.json").read_text())
    assert [r["exit"] for r in report] == ["p_mean", "v_h0"]
    assert all(r["attacker"]["num_seen"] == len(ids) for r in report)
