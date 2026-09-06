"""CPU tests for the in-training gradient-conflict observer (GRADAUDIT/OBSV2).

Pins: flat_stats math incl. the A4 negative-tangent fraction; the ROBUST GRL
reversal-scale check (== 1 at any lambda, violation only on magnitude identity
failure -- the OBSV2 replacement for the cross-forward bit-exact residual);
permanent replay bit-exactness evidence; observer READ-ONLYness (bit-identical
trained parameters with audit on/off, same seed); gate policies report|fail;
config validation and defaults.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import replace as dreplace
from pathlib import Path

import pytest
import torch
import yaml

from src.eva_di.config import GradAuditCfg, load_config
from src.eva_di.contracts import EvaDiConfigError, EvaDiError
from src.eva_di.grad_audit import flat_stats
from src.eva_di.train import run_training

from tests.test_eva_di_r2_runtime import CONFIG_DIR, _cfg, tree  # noqa: F401


def test_flat_stats_exact():
    a = torch.tensor([3.0, 0.0])
    head_on = flat_stats(a, torch.tensor([-4.0, 0.0]))
    assert head_on["cos"] == pytest.approx(-1.0) and head_on["conflict"] is True
    assert head_on["neg_tangent_frac"] == pytest.approx(1.0)   # full drag-back
    orth = flat_stats(a, torch.tensor([0.0, 5.0]))
    assert orth["cos"] == pytest.approx(0.0) and orth["conflict"] is False
    assert orth["neg_tangent_frac"] == pytest.approx(0.0)
    deg = flat_stats(a, torch.zeros(2))
    assert deg["degenerate"] and deg["cos"] is None


def _audit_cfg(fresh_tree, tmp, run_id, *, every=1, epochs=2, policy="report"):
    cfg = _cfg(fresh_tree, tmp, epochs=epochs, run_id=run_id)
    return dreplace(cfg, identity=dreplace(
        cfg.identity, grad_audit=GradAuditCfg(True, every, 500, policy)))


def test_grl_hard_check_and_records_on_real_run(tmp_path, tree):
    summary = run_training(_audit_cfg(tree, tmp_path, "ga-main"), device="cpu")
    ga = summary["grad_audit"]
    assert ga["n_steps"] >= 1 and ga["grl_checked"] >= 1
    assert ga["grl_violations"] == 0                    # mechanism verified live
    assert ga["eff"]["n_scored"] >= 1 and ga["modules"]  # per-module decomposition
    run_dir = tmp_path / "outputs" / "ga-main"
    assert (run_dir / "grad_audit" / "step_metrics.jsonl").is_file()
    assert (run_dir / "grad_audit" / "grad_audit_report.md").is_file()


def test_robust_reversal_scale_and_replay_evidence(tmp_path, tree):
    summary = run_training(_audit_cfg(tree, tmp_path, "ga-robust"), device="cpu")
    ga = summary["grad_audit"]
    assert ga["grl_violations"] == 0
    assert ga["replay_bitexact_all"] is True       # replay reproduces the forward
    first = json.loads((tmp_path / "outputs" / "ga-robust" / "grad_audit"
                        / "step_metrics.jsonl").read_text(encoding="utf-8")
                       .splitlines()[0])
    rows = [r for r in first["grl_rows"] if not r["vacuous"]]
    assert rows
    for row in rows:
        assert row["reversal_scale"] == pytest.approx(1.0, abs=1e-3)
        assert row["norm_ratio"] > 0.0


def test_observer_is_read_only(tmp_path, tree):
    """Same seed, audit off vs on: trained parameters must be bit-identical
    (autograd.grad writes nothing: no params, no .grad buffers, no RNG use)."""
    plain = run_training(_cfg(tree, tmp_path, epochs=2, run_id="ga-off"),
                         device="cpu")
    audited = run_training(_audit_cfg(tree, tmp_path, "ga-on"), device="cpu")
    assert plain["best_epoch"] == audited["best_epoch"]
    a = torch.load(tmp_path / "outputs/ga-off/checkpoints/last.pt",
                   map_location="cpu", weights_only=False)["model"]
    b = torch.load(tmp_path / "outputs/ga-on/checkpoints/last.pt",
                   map_location="cpu", weights_only=False)["model"]
    assert set(a) == set(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_gate_policy_fail_raises_report_continues(tmp_path, tree, monkeypatch):
    """Force deviations (tol 0.0 -> float noise counts) and verify the two
    policies: 'fail' hard-stops with the reversal message, 'report' completes
    and flags in summary['grad_audit']."""
    import src.eva_di.grad_audit as ga_mod
    monkeypatch.setattr(ga_mod, "_GRL_SCALE_TOL", 0.0)
    with pytest.raises(EvaDiError, match="reversal-scale"):
        run_training(_audit_cfg(tree, tmp_path, "ga-fail", policy="fail"),
                     device="cpu")
    summary = run_training(_audit_cfg(tree, tmp_path, "ga-report", policy="report"),
                           device="cpu")
    assert summary["grad_audit"]["grl_violations"] >= 1
    assert "on_grl_violation=report" in summary["grad_audit"]["violation_note"]


def _derived(base: str, mutate) -> Path:
    raw = yaml.safe_load((CONFIG_DIR / base).read_text(encoding="utf-8"))
    raw["extends"] = str(CONFIG_DIR / raw["extends"])   # abs: loadable from tmp
    mutate(raw)
    fd, name = tempfile.mkstemp(suffix=".yaml")
    import os
    os.close(fd)
    Path(name).write_text(yaml.safe_dump(raw), encoding="utf-8")
    return Path(name)


def test_grad_audit_defaults_off_everywhere():
    for name in ("eva_di_base.yaml", "di_full.yaml", "di_ref.yaml"):
        cfg = load_config(CONFIG_DIR / name)
        assert cfg.identity.grad_audit == GradAuditCfg(False, 10, 2000, "report"), name


def test_grad_audit_requires_an_enabled_head():
    path = _derived("di_ref.yaml", lambda raw: raw.setdefault(
        "identity", {}).__setitem__("grad_audit", {"enable": True}))
    with pytest.raises(EvaDiConfigError, match="grad_audit"):
        load_config(path)


def test_grad_audit_parses_overrides_and_policy():
    path = _derived("di_full.yaml", lambda raw: raw["identity"].__setitem__(
        "grad_audit", {"enable": True, "every_n_steps": 5, "max_records": 50,
                       "on_grl_violation": "fail"}))
    assert load_config(path).identity.grad_audit == GradAuditCfg(True, 5, 50, "fail")
    bad = _derived("di_full.yaml", lambda raw: raw["identity"].__setitem__(
        "grad_audit", {"enable": True, "every_n_steps": 0}))
    with pytest.raises(EvaDiConfigError, match="every_n_steps"):
        load_config(bad)
    bad2 = _derived("di_full.yaml", lambda raw: raw["identity"].__setitem__(
        "grad_audit", {"enable": True, "on_grl_violation": "ignore"}))
    with pytest.raises(EvaDiConfigError, match="on_grl_violation"):
        load_config(bad2)
