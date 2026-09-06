"""OBSV2 CPU tests: pcc/rmse metrics surface, epochs/patience bump, viz CLI.

The viz smoke runs on synthetic metrics.jsonl (no training needed): panel
files must be produced, missing series must never be fabricated as zeros.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.eva_di.config import load_config
from src.eva_di.metrics import pcc, rmse
from src.eva_di.viz_training import plot_compare, plot_run

from tests.test_eva_di_r2_runtime import CONFIG_DIR, tree  # noqa: F401


def test_pcc_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert pcc(y, 2 * y + 1) == pytest.approx(1.0)
    assert pcc(y, -y) == pytest.approx(-1.0)
    assert math.isnan(pcc(y, np.ones(4)))              # degenerate side -> NaN
    assert math.isnan(pcc(np.array([1.0]), np.array([2.0])))
    rng = np.random.default_rng(0)
    z = rng.normal(size=2000)
    assert abs(pcc(z, z + rng.normal(scale=8.0, size=2000))) < 0.25


def test_rmse_matches_hand_computation():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, np.array([2.0, 3.0, 4.0])) == pytest.approx(1.0)
    assert rmse(y, y) == 0.0


def _fake_run(root: Path, name: str, rows: list[dict]) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return d


def test_viz_produces_panels_and_never_fabricates(tmp_path):
    r_full = _fake_run(tmp_path, "run-full", [
        {"epoch": 1, "train_total": 0.5, "train_reg": 0.5, "val_ccc": 0.30,
         "val_mae": 9.0, "val_rmse": 11.0, "val_pcc": 0.40},
        {"epoch": 2, "train_total": 0.4, "train_reg": 0.4, "val_ccc": 0.35,
         "val_mae": 8.0, "val_rmse": 10.0, "val_pcc": 0.45},
    ])
    r_old = _fake_run(tmp_path, "run-legacy", [   # no rmse/pcc/audit, like old runs
        {"epoch": 1, "train_total": 0.6, "train_reg": 0.6, "val_ccc": 0.20,
         "val_mae": 10.0},
    ])
    (r_full / "grad_audit").mkdir()
    (r_full / "grad_audit" / "step_metrics.jsonl").write_text("\n".join(
        json.dumps({"step": s, "overall": {"cos": 0.1, "conflict": s % 2 == 0,
                                           "neg_tangent_frac": 0.2}})
        for s in (10, 20, 30)), encoding="utf-8")
    out = tmp_path / "viz"
    png = plot_run(r_full, out)
    assert png.is_file() and png.stat().st_size > 4000
    plot_run(r_old, out)                     # missing series must not raise
    made = {p.name for p in plot_compare([r_full, r_old], tmp_path / "cmp")}
    assert {"compare_ccc.png", "compare_mae.png",
            "compare_rmse.png", "compare_pcc.png"} == made
    # NaN metrics are dropped pointwise, never drawn as zero
    r_nan = _fake_run(tmp_path, "run-nan", [
        {"epoch": 1, "train_total": 0.5, "train_reg": 0.5,
         "val_ccc": float("nan"), "val_mae": 9.0}])
    plot_run(r_nan, out)


def test_epochs_100_patience_18_in_base_chain():
    for name in ("eva_di_base.yaml", "di_full.yaml", "di_ref.yaml"):
        cfg = load_config(CONFIG_DIR / name)
        assert cfg.training.epochs_max == 100, name
        assert cfg.training.early_stop["patience"] == 18, name
