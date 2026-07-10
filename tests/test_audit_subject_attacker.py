"""Tests for the fresh linear-probe attacker in ``scripts/audit_subject_attacker.py``.

The pure aggregation logic (``compute_attacker_metrics`` etc.) is covered by
``tests/test_subject_attacker.py``.  These tests cover the only NEW logic: the
LOVO Ridge attacker (``lovo_attack``) and the NPZ-loading CLI path -- no torch
or checkpoint is involved (features come from the A1 layerwise NPZ).
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the wrapper as a module (it lives under scripts/, not a package).
_SPEC = importlib.util.spec_from_file_location(
    "audit_subject_attacker", PROJECT_ROOT / "scripts" / "audit_subject_attacker.py"
)
audit_subject_attacker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit_subject_attacker)
lovo_attack = audit_subject_attacker.lovo_attack


def _make_features(n_subjects=10, videos_per_subject=2, separable=True, dim=None):
    """Build (X, y, video_ids, subject_ids) for a synthetic subject set."""
    dim = dim or n_subjects
    rng = np.random.default_rng(0)
    X, sids, vids = [], [], []
    for s in range(n_subjects):
        for v in range(videos_per_subject):
            if separable:
                # one-hot-ish subject signal + small noise -> linearly separable
                vec = np.zeros(dim)
                vec[s % dim] = 1.0
                vec = vec + rng.normal(0, 0.01, dim)
            else:
                # identical feature for every video -> no separability
                vec = np.ones(dim)
            X.append(vec)
            sids.append(f"S{s}")
            vids.append(f"S{s}_v{v}_video")
    X = np.array(X)
    y = np.array([s for s in range(n_subjects) for _ in range(videos_per_subject)])
    return X, y, np.array(vids), np.array(sids)


def test_lovo_attack_separable_features_near_perfect():
    X, y, vids, sids = _make_features(separable=True)
    pq = lovo_attack(X, y, vids, sids, alpha=1.0)
    assert len(pq) == len(y)
    # All rows are "seen" (fresh attacker trains on test subjects).
    assert all(r["seen"] is True for r in pq)
    top1 = sum(r["correct_top1"] for r in pq) / len(pq)
    assert top1 >= 0.9, f"separable features should yield high top1, got {top1}"


def test_lovo_attack_identical_features_near_chance():
    X, y, vids, sids = _make_features(separable=False)
    pq = lovo_attack(X, y, vids, sids, alpha=1.0)
    top1 = sum(r["correct_top1"] for r in pq) / len(pq)
    n_classes = len(set(y.tolist()))
    # No signal -> top1 should be well below separable case, near chance (1/n).
    assert top1 <= 0.3, f"identical features should not beat chance much, got {top1}"
    assert 1.0 / n_classes == pytest.approx(0.1, abs=1e-9)


def test_lovo_attack_per_query_schema():
    X, y, vids, sids = _make_features()
    pq = lovo_attack(X, y, vids, sids)
    from src.diagnostics.subject_attacker import PER_QUERY_COLUMNS
    assert set(pq[0].keys()) == set(PER_QUERY_COLUMNS)
    # pred_top3 has 3 entries (or fewer if fewer classes).
    assert len(pq[0]["pred_top3_subject_indices"]) == min(3, len(set(y.tolist())))


def test_lovo_attack_coverage_full():
    X, y, vids, sids = _make_features()
    pq = lovo_attack(X, y, vids, sids)
    from src.diagnostics.subject_attacker import compute_attacker_metrics
    summary = compute_attacker_metrics(pq, num_subject_classes=len(set(y.tolist())))
    assert summary["num_seen"] == len(pq)
    assert summary["coverage"] == 1.0
    assert summary["skipped"] is False
    assert summary["chance_top1"] == pytest.approx(1.0 / len(set(y.tolist())))


def test_cli_resolves_npz_and_runs(tmp_path, monkeypatch):
    """End-to-end: synthesize an A1-style NPZ, run main(), check the summary."""
    from src.diagnostics.io import ensure_dir

    # Build a run dir with the A1 NPZ at the expected path.
    run_dir = tmp_path / "run"
    a1_dir = run_dir / "diagnostics" / "test" / "a1_layerwise"
    ensure_dir(a1_dir)
    X, y, vids, sids = _make_features(n_subjects=8, separable=True)
    np.savez(a1_dir / "test_layerwise_features.npz",
             features_layer_shared=X, subject_ids=sids, video_ids=vids)

    out_dir = run_dir / "diagnostics" / "test" / "subject_attacker"
    monkeypatch.setattr(sys, "argv", [
        "audit_subject_attacker.py",
        "--run-dir", str(run_dir),
        "--split", "test",
        "--output-dir", str(out_dir),
    ])
    audit_subject_attacker.main()

    summary_csv = out_dir / "tables" / "subject_attacker_summary.csv"
    assert summary_csv.exists()
    import csv
    rows = list(csv.DictReader(open(summary_csv)))
    assert len(rows) == 1
    r = rows[0]
    assert r["skipped"] == "0"
    assert float(r["coverage"]) == 1.0
    assert float(r["num_subject_classes"]) == 8
    assert float(r["chance_top1"]) == pytest.approx(1.0 / 8)
    assert float(r["top1_accuracy"]) >= 0.9  # separable

    # per_query has one row per video.
    pq_csv = out_dir / "tables" / "subject_attacker_per_query.csv"
    assert len(list(csv.DictReader(open(pq_csv)))) == len(y)


def test_cli_errors_when_npz_missing(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "audit_subject_attacker.py",
        "--run-dir", str(run_dir),
        "--split", "test",
        "--output-dir", str(out_dir),
    ])
    with pytest.raises(FileNotFoundError, match="A1 layerwise NPZ"):
        audit_subject_attacker.main()
