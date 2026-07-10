#!/usr/bin/env python
"""Offline subject-attacker accuracy audit for Stage B identity risk.

**Fresh linear-probe attacker**: trains a Ridge classifier on the frozen shared
features (``z_dep``) to recover test-subject identity, leave-one-video-out.
This measures whether subject identity is linearly decodable from the learned
representation -- a parametric complement to A1's nearest-neighbor retrieval.

Why a FRESH attacker (not the trained ``subject_id_head``): AVEC2014 is
subject-disjoint -- train and test subjects do not overlap -- so the jointly-
trained identity head has never seen any test subject. Evaluating it on test
gives ``coverage=0`` (every test subject is "unseen") and no accuracy can be
computed. The fresh attacker trains on test subjects themselves (LOVO: each
held-out video's subject is still in the training set via its other video),
so ``coverage=100%`` and every run -- including the E0/E2 baseline -- gets a
real number. A successful identity defense drives the fresh attacker's top1
toward chance (``1 / num_subject_classes``); if it stays high, ``z_dep``
still leaks identity.

Features are read from the A1 layerwise NPZ
(``<run_dir>/diagnostics/test/a1_layerwise/test_layerwise_features.npz``,
array ``features_layer_shared``), so NO checkpoint forward pass or GPU is
needed.  Requires the A1 layerwise probe (``diagnose_run`` Stage 2 /
``audit_layerwise_identity_probe.py``) to have run first.

This is a convergent signal with A1 retrieval (both measure identity
separability in ``z_dep``); its value is (a) making the B5 "subject attacker"
gate criterion live for all runs, and (b) a stricter parametric (linear) view
that can exceed non-parametric NN when identity is linearly separable but not
nearest-neighbor-separable.

Example::

    python scripts/audit_subject_attacker.py \\
        --run-dir <LOG_DIR>/stage_b/e1_identity_adversarial/version_0 \\
        --split test \\
        --output-dir <RUN_DIR>/diagnostics/test/subject_attacker
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Offline subject-attacker accuracy audit (Stage B identity risk)."
    )
    parser.add_argument("--run-dir", required=True, help="MTL-Lite CSVLogger version directory.")
    parser.add_argument("--split", default="test", help="Split whose A1 NPZ to attack (default test).")
    parser.add_argument("--output-dir", required=True, help="Directory for tables and report.")
    parser.add_argument(
        "--features",
        default=None,
        help="Override path to the layerwise features NPZ (default "
        "<run-dir>/diagnostics/<split>/a1_layerwise/*_layerwise_features.npz).",
    )
    parser.add_argument(
        "--layer",
        default="features_layer_shared",
        help="NPZ array to attack (default features_layer_shared = z_dep).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength (default 1.0).",
    )
    parser.add_argument(
        "--cv",
        default="lovo",
        choices=["lovo"],
        help="Cross-validation protocol (default lovo = leave-one-video-out).",
    )
    return parser


def _resolve_features_npz(args):
    """Locate the A1 layerwise NPZ for the split, or use --features override."""
    if args.features:
        p = Path(args.features)
        if not p.exists():
            raise FileNotFoundError(f"--features NPZ not found: {p}")
        return p
    base = Path(args.run_dir) / "diagnostics" / args.split / "a1_layerwise"
    cands = sorted(base.glob("*_layerwise_features.npz")) if base.exists() else []
    if not cands:
        raise FileNotFoundError(
            f"A1 layerwise NPZ not found under {base}/ -- run "
            "audit_layerwise_identity_probe.py (diagnose_run Stage 2) first."
        )
    return cands[0]


def lovo_attack(X, y, video_ids, subject_ids, alpha=1.0):
    """Leave-one-video-out Ridge attacker over frozen features.

    For each held-out video, fits a Ridge classifier on the other videos and
    ranks all classes by decision score.  Each subject is represented in the
    training fold via its other video (AVEC2014 has 2 videos/subject), so the
    held-out subject is a candidate label -- this is the identification test.

    Args:
        X: (n, d) feature matrix (z_dep).
        y: (n,) integer class labels (subject index in the test subject space).
        video_ids: (n,) video id strings (aligned with X).
        subject_ids: (n,) raw subject id strings (aligned with X).
        alpha: Ridge regularization strength.

    Returns:
        List of per-query dicts (PER_QUERY_COLUMNS schema), all with seen=True.
    """
    import numpy as np
    from sklearn.linear_model import RidgeClassifier

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n = len(y)
    per_query = []
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        clf = RidgeClassifier(alpha=alpha)
        clf.fit(X[train_mask], y[train_mask])
        # decision_function: (1, n_classes_trained) signed scores; higher = more likely.
        scores = clf.decision_function(X[i : i + 1])[0]
        ranked = clf.classes_[np.argsort(scores)[::-1]]
        true_idx = int(y[i])
        pred_top3 = [int(c) for c in ranked[:3]]
        per_query.append({
            "video_id": str(video_ids[i]),
            "subject_id": str(subject_ids[i]),
            "seen": True,  # fresh attacker: every test subject is in-scope
            "true_subject_index": true_idx,
            "pred_subject_index": int(ranked[0]),
            "pred_top3_subject_indices": pred_top3,
            "correct_top1": bool(int(ranked[0]) == true_idx),
            "correct_top3": bool(true_idx in pred_top3),
        })
    return per_query


def main():
    args = build_parser().parse_args()
    import numpy as np

    from src.diagnostics.io import ensure_dir
    from src.diagnostics.subject_attacker import (
        compute_attacker_metrics,
        write_subject_attacker_per_query,
        write_subject_attacker_report,
        write_subject_attacker_summary,
    )

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir))
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")

    npz_path = _resolve_features_npz(args)
    npz = np.load(npz_path, allow_pickle=True)
    if args.layer not in npz:
        raise KeyError(
            f"NPZ {npz_path} has no array '{args.layer}'; available: {list(npz.keys())}"
        )
    X = npz[args.layer]
    # subject_ids/video_ids are stored alongside features in the A1 NPZ.
    if "subject_ids" not in npz or "video_ids" not in npz:
        raise KeyError(
            f"NPZ {npz_path} must carry 'subject_ids' and 'video_ids' arrays "
            f"(A1 layerwise probe output); available: {list(npz.keys())}"
        )
    subject_ids = npz["subject_ids"]
    video_ids = npz["video_ids"]

    # Build a compact test-subject class index (0..num_subject_classes-1).
    classes = sorted(set(str(s) for s in subject_ids))
    cls_idx = {s: i for i, s in enumerate(classes)}
    y = np.array([cls_idx[str(s)] for s in subject_ids])
    num_classes = len(classes)

    print(
        f"[SUBJECT-ATTACKER] {npz_path.name} layer={args.layer} "
        f"n={len(y)} subjects={num_classes} chance={1.0/num_classes:.4f} "
        f"feat_dim={X.shape[1]}"
    )
    print(f"[SUBJECT-ATTACKER] running LOVO Ridge attacker (alpha={args.alpha}) ...")

    per_query = lovo_attack(X, y, video_ids, subject_ids, alpha=args.alpha)
    summary = compute_attacker_metrics(per_query, num_subject_classes=num_classes)

    write_subject_attacker_summary(summary, tables_dir / "subject_attacker_summary.csv")
    write_subject_attacker_per_query(per_query, tables_dir / "subject_attacker_per_query.csv")
    write_subject_attacker_report(reports_dir / "subject_attacker_report.md", summary, per_query)

    print(
        f"[SUBJECT-ATTACKER] evaluated={summary['num_evaluated']} "
        f"seen={summary['num_seen']} coverage={summary['coverage']} "
        f"top1={summary['top1_accuracy']} top3={summary['top3_accuracy']} "
        f"chance={summary['chance_top1']}"
    )
    print(f"[SUBJECT-ATTACKER] output: {output_dir}")


if __name__ == "__main__":
    main()
