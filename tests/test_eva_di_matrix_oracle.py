"""CPU tests for the oracle equivalence harness and the matrix planner/report
(EVA-DI-CODE-v2).  No trans import, no GPU: the oracle is exercised through
its injectable encoder callables; the reference subprocess is covered by the
failure guard test only."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.eva_di.contracts import EvaDiError
from src.eva_di.extract.oracle_equivalence import (
    COSINE_ACCEPT, compare_and_record, equivalence_report_skeleton)
from src.eva_di.matrix import build_plan, launch, load_matrix_spec, plan_table
from src.eva_di.matrix_report import collect_rows, write_reports


# ---------------------------------------------------------------- oracle
def _frames(n=4):
    return [{"sample_id": "203_1_Freeform_video", "frame_index": i,
             "path": f"/nonexistent/{i}.jpg", "sha256": "ab" * 32} for i in range(n)]


def test_identity_cosine_passes(tmp_path):
    frames = _frames()
    base = np.random.default_rng(0).normal(0, 1, (4, 8))
    rep = compare_and_record(frames, (base, base), (base.copy(), base.copy()),
                             evidence_dir=tmp_path / "ev")
    assert rep["status"] == "PASSED" and rep["min_cos_cls"] == pytest.approx(1.0)
    files = list((tmp_path / "ev").glob("oracle_equivalence_*.json"))
    assert len(files) == 1
    on_disk = json.loads(files[0].read_text())
    assert on_disk["status"] == "PASSED" and on_disk["n_frames"] == 4


def test_perturbation_fails_and_still_writes_evidence(tmp_path):
    frames = _frames()
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, (4, 8))
    off = rng.normal(0, 1, (4, 8)) * 0.5
    rep = compare_and_record(frames, (base, base), (off, off),
                             evidence_dir=tmp_path / "ev")
    assert rep["status"] == "FAILED" and rep["min_cos_cls"] < COSINE_ACCEPT
    assert list((tmp_path / "ev").glob("oracle_equivalence_*.json"))


def test_shape_mismatch_refused(tmp_path):
    with pytest.raises(EvaDiError, match="shape mismatch"):
        compare_and_record(_frames(4), (np.ones((4, 8)), np.ones((4, 8))),
                           (np.ones((3, 8)), np.ones((3, 8))),
                           evidence_dir=tmp_path / "ev")


def test_skeleton_still_available():
    sk = equivalence_report_skeleton(cache_root="/tmp/c")
    assert sk["status"] == "NOT_RUN" and sk["evidence_dir"].endswith("_equivalence")


# ---------------------------------------------------------------- matrix
def _mk_cfg(root, rel, *, run_id, seed=None):
    cfg = root / rel
    cfg.parent.mkdir(parents=True, exist_ok=True)
    lines = ["schema_version: eva_di_config_v1", "run:", f"  run_id: {run_id}"]
    if seed is not None:
        lines.append(f"  seed: {seed}")
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _spec(tmp_path):
    return {"schema_version": "eva_di_matrix_v1",
            "stage1": {"seed": 42, "arms": [
                {"name": "A", "config": "configs/a.yaml"},
                {"name": "B", "config": "configs/b.yaml"},
                {"name": "C", "config": "configs/missing.yaml"}]},
            "ablations": [{"name": "AB", "config": "configs/ab.yaml"}],
            "stage2": {"seeds": [43], "arms": ["A"]}}


def _mk_repo(tmp_path):
    repo = tmp_path / "repo"
    _mk_cfg(repo, "configs/a.yaml", run_id="run-a")
    _mk_cfg(repo, "configs/b.yaml", run_id="run-b", seed=7)   # seed mismatch
    _mk_cfg(repo, "configs/ab.yaml", run_id="run-ab")
    runs = tmp_path / "runs"
    (runs / "run-a").mkdir(parents=True)
    (runs / "run-a" / "summary.json").write_text('{"best_epoch": 1}',
                                                 encoding="utf-8")
    return repo, runs


def test_plan_statuses(tmp_path):
    repo, runs = _mk_repo(tmp_path)
    plan = build_plan(_spec(tmp_path), repo_root=repo, runs_root=runs)
    by_arm = {e["arm"]: e for e in plan}
    assert by_arm["A"]["status"] == "done"
    assert by_arm["B"]["status"].startswith("blocked: ")
    assert "run.seed=7" in by_arm["B"]["status"]
    assert by_arm["C"]["status"].startswith("blocked: config missing")
    assert by_arm["AB"]["status"] == "ready"
    assert by_arm["AB"]["commands"]["export_val"].endswith("--checkpoint RUN_ROOT/run-ab/checkpoints/best.pt")
    assert "p_mean_val.npz" in by_arm["A"]["commands"]["identity"]
    assert "DI-REF" not in plan_table(plan)


def test_stage2_override_and_unknown_arm(tmp_path):
    repo, runs = _mk_repo(tmp_path)
    spec = _spec(tmp_path)
    plan = build_plan(spec, repo_root=repo, runs_root=runs, seeds=[43], arms=["A"],
                      include_ablations=False)
    assert len(plan) == 1
    assert plan[0]["seed"] == 43
    assert plan[0]["status"].startswith("blocked: ")  # a.yaml is seed-42
    with pytest.raises(EvaDiError, match="not in stage1"):
        build_plan(spec, repo_root=repo, runs_root=runs, seeds=[43], arms=["Z"])


def test_spec_validation_and_dry_launch(tmp_path, monkeypatch):
    with pytest.raises(EvaDiError, match="schema_version"):
        load_matrix_spec(_write(tmp_path, {"schema_version": "other"}))
    with pytest.raises(EvaDiError, match="non-empty"):
        load_matrix_spec(_write(tmp_path, {"schema_version": "eva_di_matrix_v1",
                                           "stage1": {"seed": 42, "arms": []}}))

    def _boom(*a, **k):  # dry-run must never spawn
        raise AssertionError("subprocess spawned in dry run")

    monkeypatch.setattr("src.eva_di.matrix.subprocess.run", _boom)
    repo, runs = _mk_repo(tmp_path)
    plan = build_plan(_spec(tmp_path), repo_root=repo, runs_root=runs)
    ready_only = [e for e in plan if e["status"] in ("ready", "done")]
    assert launch(ready_only, repo_root=repo, python="python", device="cpu",
                  dry_run=True) == 0


def _write(tmp_path, payload):
    p = tmp_path / "spec.yaml"
    import yaml
    p.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return p


# ------------------------------------------------------------ matrix report
def test_report_blanks_missing_identity(tmp_path):
    runs = tmp_path / "runs"
    (runs / "r1" / "identity").mkdir(parents=True)
    (runs / "r1" / "summary.json").write_text(json.dumps(
        {"best_epoch": 2, "best_val_ccc": 0.5, "best_defined": True,
         "epochs_run": 4, "stopped_early": False, "val_n": 100,
         "n_subjects": 41, "max_recordings": None, "checkpoint": "x/best.pt"}),
        encoding="utf-8")
    (runs / "r1" / "identity" / "identity_metrics.json").write_text(json.dumps([
        {"exit": "p_mean", "attacker": {"top1_accuracy": 0.9, "top3_accuracy": 1.0},
         "pair_auroc": {"auc": 0.95},
         "a1_retrieval": {"top1_same_subject": 0.8, "top3_same_subject": 0.95}},
    ]), encoding="utf-8")
    (runs / "r2").mkdir()
    (runs / "r2" / "summary.json").write_text('{"best_epoch": 0}', encoding="utf-8")
    (runs / "notarun").mkdir()

    rows = collect_rows(runs)
    assert [r["run_id"] for r in rows] == ["r1", "r2"]
    r1, r2 = rows
    assert r1["p_attacker_top1"] == 0.9 and r1["p_pair_auroc"] == 0.95
    assert r1["v_attacker_top1"] is None          # v_h0 evidence missing
    assert all(r2.get(k) is None for k in ("p_attacker_top1", "best_val_ccc"))
    csv_path, md_path = write_reports(rows, tmp_path / "out")
    text = csv_path.read_text()
    assert text.count(",") and ",," in text       # blank cells stay blank
    assert "| 0.9000 |" in md_path.read_text()
