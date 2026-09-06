"""REGNORM tests: plain-MSE L_reg on the normalized target scale.

Pins: mse_norm exact math + shape guard; plain == severity-with-unit-weights
(semantics of "de-weighting"); config default (legacy severity when the key is
absent) vs base-yaml protocol (plain); the 1-epoch training path runs and
records reg_weighting/target stats (the provenance gap the loss report
flagged) in summary.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.eva_di.config import load_config
from src.eva_di.contracts import EvaDiConfigError, EvaDiSchemaError
from src.eva_di.losses import mse_norm, severity_weighted_l2
from src.eva_di.train import run_training

from tests.test_eva_di_r2_runtime import CONFIG_DIR, _cfg, tree  # noqa: F401


def test_mse_norm_exact_and_guarded():
    p = torch.tensor([1.0, 2.5])
    t = torch.tensor([0.0, 0.5])
    assert float(mse_norm(p, t)) == pytest.approx((1.0 + 4.0) / 2)
    assert float(mse_norm(p.reshape(2, 1), t)) == pytest.approx(2.5)  # flatten
    with pytest.raises(EvaDiSchemaError, match="mse_norm"):
        mse_norm(p, torch.zeros(3))


def test_plain_equals_severity_with_unit_weights():
    g = torch.Generator().manual_seed(0)
    p = torch.rand(64, generator=g)
    t = torch.rand(64, generator=g)
    ones = torch.ones(64)
    assert float(mse_norm(p, t)) == pytest.approx(
        float(severity_weighted_l2(p, t, ones)), rel=1e-6)


def test_weighting_config_surface():
    assert load_config(CONFIG_DIR / "eva_di_base.yaml").training.reg_weighting == "plain"
    assert load_config(CONFIG_DIR / "di_full.yaml").training.reg_weighting == "plain"
    # dataclass default stays legacy (in-code construction without the key)
    from src.eva_di.config import TrainingCfg
    legacy = TrainingCfg("bf16-mixed", {}, 8, 100, {"metric": "val_ccc",
                                                    "patience": 18}, 1.0)
    assert legacy.reg_weighting == "severity"

    raw = yaml.safe_load((CONFIG_DIR / "di_full.yaml").read_text(encoding="utf-8"))
    raw["extends"] = str(CONFIG_DIR / raw["extends"])
    fd, name = tempfile.mkstemp(suffix=".yaml")
    import os
    os.close(fd)
    path = Path(name)
    raw.setdefault("training", {})["reg_weighting"] = "severity"   # legacy opt-in
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert load_config(path).training.reg_weighting == "severity"
    raw["training"]["reg_weighting"] = "bogus"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(EvaDiConfigError, match="reg_weighting"):
        load_config(path)
    path.unlink(missing_ok=True)


def test_plain_path_trains_and_records_provenance(tmp_path, tree):
    summary = run_training(_cfg(tree, tmp_path, epochs=1, run_id="rg-e1"), device="cpu")
    assert summary["reg_weighting"] == "plain"
    assert isinstance(summary["severity_weights"], dict) and summary["severity_weights"]
    stats = summary["reg_target_stats"]
    assert stats["std"] > 0.0 and np.isfinite(stats["mean"])
    line = json.loads((tmp_path / "outputs/rg-e1/metrics.jsonl")
                      .read_text(encoding="utf-8").splitlines()[-1])
    assert {"train_reg", "val_ccc", "val_rmse", "val_pcc"} <= set(line)
