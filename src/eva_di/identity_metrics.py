"""External identity metrics on the exported exit-P / exit-V embeddings.

The model itself carries NO identity metric (design section 10 / mechanism
section 6): identity risk is judged OFFLINE on the npz files written by
``src.eva_di.export_embeddings`` --

* fresh LOVO Ridge attacker top1/top3 (regression-on-indicators; the
  leave-one-VIDEO-out protocol keeps every held-out recording's PERSON present
  through its other recordings, so the attacker never sees the sample it
  judges; mirrors the frozen legacy ``scripts/audit_subject_attacker.py``
  semantics and reuses its row schema);
* identity pair-AUROC (A2: cosine of same-person vs different-person pairs);
* A1 same-person retrieval top1/top3 (cosine, self always excluded).

Subjects are PERSONS (``video_id[:IDENTITY_PREFIX_LEN]``, user decision
2026-09-05).  Pure numpy: the ridge and the AUROC are implemented inline
(exact ridge-on-indicators / rank statistic), so local and server numbers are
bit-reproducible and no new dependency enters.  Test-split embeddings are
refused (locked benchmark).  CPU only; no checkpoint forward.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.eva_di.contracts import IDENTITY_PREFIX_LEN, EvaDiError
from src.eva_di.export_embeddings import EXPORT_SCHEMA


def subject_from_video_id(video_id: str) -> str:
    return str(video_id)[:IDENTITY_PREFIX_LEN]


def load_export_npz(path: Path) -> tuple[np.ndarray, list[str], dict]:
    """Return (embeddings [n,d], sample_ids, provenance); schema-guarded."""
    path = Path(path)
    if not path.is_file():
        raise EvaDiError(f"embedding npz missing: {path}")
    with np.load(path, allow_pickle=True) as data:
        if "schema" not in data or str(data["schema"]) != EXPORT_SCHEMA:
            raise EvaDiError(
                f"{path.name}: not an {EXPORT_SCHEMA} export (keys: "
                f"{sorted(data.files)})")
        embeddings = np.asarray(data["embeddings"], dtype=np.float64)
        sample_ids = [str(s) for s in data["sample_ids"]]
        split = str(data["split"]) if "split" in data else ""
        prov = {k: str(data[k]) for k in data.files
                if k not in ("schema", "split", "embeddings", "sample_ids")}
    if embeddings.ndim != 2 or embeddings.shape[0] != len(sample_ids):
        raise EvaDiError(
            f"{path.name}: embeddings {embeddings.shape} vs {len(sample_ids)} ids")
    if not np.isfinite(embeddings).all():
        raise EvaDiError(f"{path.name}: non-finite embeddings")
    if split == "test":
        raise EvaDiError(f"{path.name}: test-split identity evaluation is locked")
    prov.setdefault("split", split)
    return embeddings, sample_ids, prov


def _l2(units: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(units, axis=1, keepdims=True)
    return np.divide(units, norms, out=np.zeros_like(units), where=norms > 1e-12)


def cosine_matrix(units: np.ndarray) -> np.ndarray:
    """Pairwise cosine with self-similarity masked to -1 (legacy A1 dialect)."""
    sim = _l2(units) @ _l2(units).T
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, -1.0)
    zero_rows = np.linalg.norm(units, axis=1) < 1e-12
    sim[zero_rows, :] = -1.0
    sim[:, zero_rows] = -1.0
    return sim


def ridge_onehot_scores(x_fit: np.ndarray, y_fit: np.ndarray,
                        classes: list[str], x_query: np.ndarray,
                        alpha: float) -> np.ndarray:
    """Exact ridge regression on one-hot indicators -> scores [nq, K].

    Closed form identical to ``sklearn.RidgeClassifier`` (centered X and Y,
    intercept recovered), written inline for version-stable reproduction.
    """
    index = {c: i for i, c in enumerate(classes)}
    y = np.asarray([[index[c] for c in y_fit]], dtype=np.float64).ravel()
    onehot = np.zeros((len(y_fit), len(classes)))
    onehot[np.arange(len(y_fit)), y.astype(int)] = 1.0
    xbar, ybar = x_fit.mean(axis=0), onehot.mean(axis=0)
    xc, yc = x_fit - xbar, onehot - ybar
    gram = xc.T @ xc + alpha * np.eye(x_fit.shape[1])
    weights = np.linalg.solve(gram, xc.T @ yc)
    return x_query @ weights + (ybar - xbar @ weights)


def lovo_ridge_attacker(embeddings: np.ndarray, subjects: list[str], *,
                        alpha: float = 1.0) -> tuple[list[dict], int]:
    """Leave-one-video-out Ridge attacker -> (per_query_rows, n_classes).

    ``per_query_rows`` follows the frozen PER_QUERY_COLUMNS schema of
    ``src.diagnostics.subject_attacker``.  A person with no OTHER recording in
    the set is ``seen=False`` (its class cannot exist in that fold --
    reported, never guessed).
    """
    classes = sorted(set(subjects))
    class_index = {c: i for i, c in enumerate(classes)}
    counts = Counter(subjects)
    rows: list[dict] = []
    for hold in range(len(subjects)):
        fit = [i for i in range(len(subjects)) if i != hold]
        fit_classes = sorted({subjects[i] for i in fit})
        seen = counts[subjects[hold]] >= 2 and subjects[hold] in fit_classes
        truth = class_index[subjects[hold]]
        row = {"video_id": f"idx{hold}", "subject_id": subjects[hold],
               "seen": bool(seen), "true_subject_index": truth,
               "pred_subject_index": None, "pred_top3_subject_indices": "",
               "correct_top1": None, "correct_top3": None}
        if seen:
            scores = ridge_onehot_scores(
                embeddings[fit], [subjects[i] for i in fit], fit_classes,
                embeddings[hold:hold + 1], alpha)[0]
            # fold-local ranking mapped back to global class indices
            order = [fit_classes[i] for i in np.argsort(-scores)]
            top3 = [class_index[c] for c in order[:3]]
            row["pred_subject_index"] = class_index[order[0]]
            row["pred_top3_subject_indices"] = ",".join(map(str, top3))
            row["correct_top1"] = bool(truth == row["pred_subject_index"])
            row["correct_top3"] = bool(truth in top3)
        rows.append(row)
    return rows, len(classes)


def _auc_rank_statistic(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUROC with mid-rank tie handling (rank based, exact)."""
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1.0, len(scores) + 1.0)
    starts = np.r_[0, np.nonzero(np.diff(sorted_scores))[0] + 1, len(scores)]
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a > 1:
            ranks[a:b] = ranks[a:b].mean()
    rank_of = np.empty_like(ranks)
    rank_of[order] = ranks
    n_pos = int(positive.sum())
    n_neg = len(scores) - n_pos
    return float((rank_of[positive].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def pair_identity_auroc(embeddings: np.ndarray, subjects: list[str]) -> dict:
    """AUROC of cosine similarity for same-person (pos) vs different (neg)."""
    sim = cosine_matrix(embeddings)
    iu = np.triu_indices(len(subjects), k=1)
    same = np.asarray([subjects[i] == subjects[j]
                       for i, j in zip(iu[0], iu[1])])
    scores = sim[iu]
    if same.sum() == 0 or (~same).sum() == 0:
        return {"auc": None, "n_same_pairs": int(same.sum()),
                "n_diff_pairs": int((~same).sum()),
                "note": "single-class pair set; AUROC undefined"}
    return {"auc": _auc_rank_statistic(scores, same),
            "n_same_pairs": int(same.sum()), "n_diff_pairs": int((~same).sum())}


def a1_retrieval(embeddings: np.ndarray, subjects: list[str]) -> dict:
    """A1 same-person top1/top3 rates (self always excluded via diag=-1)."""
    sim = cosine_matrix(embeddings)
    n = len(subjects)
    top1_hits = top3_hits = scored = 0
    for q in range(n):
        others = [i for i in range(n) if i != q]
        if not others:
            return {"top1_same_subject": None, "top3_same_subject": None,
                    "n_queries": 0, "note": "single-sample set"}
        if not any(subjects[i] == subjects[q] for i in others):
            continue  # no same-person candidate: query excluded, never guessed
        order = sorted(others, key=lambda i: -sim[q, i])
        scored += 1
        top1_hits += int(subjects[order[0]] == subjects[q])
        top3_hits += int(any(subjects[i] == subjects[q] for i in order[:3]))
    if scored == 0:
        return {"top1_same_subject": None, "top3_same_subject": None,
                "n_queries": 0, "note": "no query had a same-person candidate"}
    return {"top1_same_subject": top1_hits / scored,
            "top3_same_subject": top3_hits / scored, "n_queries": scored}


_PROVENANCE_KEYS = ("run_id", "config_name", "config_source_sha256",
                    "checkpoint_sha256", "input_mode", "git_commit")


def evaluate_export_npz(npz_path: Path, *, per_query_out_dir: Path | None = None,
                        alpha: float = 1.0) -> dict:
    """All external identity metrics for one exported exit-embedding npz."""
    from src.diagnostics.subject_attacker import compute_attacker_metrics

    embeddings, sample_ids, prov = load_export_npz(npz_path)
    subjects = [subject_from_video_id(s) for s in sample_ids]
    rows, n_classes = lovo_ridge_attacker(embeddings, subjects, alpha=alpha)
    for row, sid in zip(rows, sample_ids):
        row["video_id"] = sid
    result = {
        "npz": str(npz_path),
        "split": prov.get("split", ""),
        "exit": ("p_mean" if "p_mean" in Path(npz_path).name else
                 "v_h0" if "v_h0" in Path(npz_path).name else ""),
        "n_recordings": len(sample_ids),
        "n_persons": len(set(subjects)),
        "attacker": compute_attacker_metrics(rows, n_classes),
        "pair_auroc": pair_identity_auroc(embeddings, subjects),
        "a1_retrieval": a1_retrieval(embeddings, subjects),
        "provenance": {k: prov[k] for k in _PROVENANCE_KEYS if k in prov},
    }
    if per_query_out_dir is not None:
        out = Path(per_query_out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dest = out / (Path(npz_path).stem + "_per_query.csv")
        header = ["video_id", "subject_id", "seen", "true_subject_index",
                  "pred_subject_index", "pred_top3_subject_indices",
                  "correct_top1", "correct_top3"]
        lines = [",".join(header)]
        for row in rows:
            lines.append(",".join("" if row[k] is None else str(row[k])
                                  for k in header))
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result["per_query_csv"] = str(dest)
    return result


def assert_common_run(npz_paths: list[Path]) -> None:
    """Every npz of one evaluation must come from the SAME checkpoint."""
    shas: dict[str, str] = {}
    for path in npz_paths:
        with np.load(path, allow_pickle=True) as data:
            if "checkpoint_sha256" not in data:
                continue  # pre-provenance export; nothing to cross-check
            shas[Path(path).name] = str(data["checkpoint_sha256"])
    if len(set(shas.values())) > 1:
        raise EvaDiError(
            "mixed checkpoints across exported npz files: " + json.dumps(shas))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="External identity metrics (LOVO Ridge / pair-AUROC / A1) "
                    "on exported exit embeddings (CPU)")
    parser.add_argument("--npz", required=True, nargs="+",
                        help="p_mean_*.npz / v_h0_*.npz files from export_embeddings")
    parser.add_argument("--output-dir", required=True,
                        help="where identity_metrics.json + per-query CSVs land")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args(argv)
    paths = [Path(p) for p in args.npz]
    if not paths:
        raise EvaDiError("no npz files given")
    assert_common_run(paths)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = [evaluate_export_npz(p, per_query_out_dir=out, alpha=args.ridge_alpha)
              for p in paths]
    (out / "identity_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
