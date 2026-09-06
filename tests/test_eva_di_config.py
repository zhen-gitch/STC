"""config: strict schema, extends merge, pinned values, test-split refusal."""

import copy
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.eva_di.contracts import (
    EVA_WEIGHT_SHA256,
    PROJ_DIM,
    SCHEMA_CONFIG,
    EvaDiConfigError,
    EvaDiPathError,
)
from src.eva_di.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "eva_di"
MATRIX = ["eva_di_base.yaml", "di_ref.yaml", "di_t1.yaml", "di_t3.yaml",
          "di_full.yaml", "di_derm.yaml", "ab_eva_only.yaml", "ab_of3_only.yaml",
          "extract_uniform512.yaml"]


def _base_payload() -> dict:
    return yaml.safe_load((CONFIG_DIR / "eva_di_base.yaml").read_text(encoding="utf-8"))


def _mktmp(payload: dict) -> Path:
    fd, raw = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    path = Path(raw)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("name", MATRIX)
def test_shipped_matrix_loads(name):
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.run.mode in ("train", "extract")
    assert cfg.data.splits == ("train", "val")  # no config may open test
    assert cfg.model.proj_dim == PROJ_DIM
    assert cfg.extraction.weight_sha256 == EVA_WEIGHT_SHA256
    assert cfg.source_sha256 and cfg.resolved_sha256


def test_matrix_semantics_via_extends():
    load = lambda name: load_config(CONFIG_DIR / name)
    base = load("eva_di_base.yaml")
    assert (base.identity.t1.enable, base.identity.t3.enable) == (False, False)
    assert (load("di_t1.yaml").identity.t1.enable,
            load("di_t1.yaml").identity.t3.enable) == (True, False)
    assert (load("di_t3.yaml").identity.t1.enable,
            load("di_t3.yaml").identity.t3.enable) == (False, True)
    full = load("di_full.yaml")
    assert (full.identity.t1.enable, full.identity.t3.enable, full.identity.derange) \
        == (True, True, False)
    derm = load("di_derm.yaml")  # two-level extends chain through di_full
    assert (derm.identity.t1.enable, derm.identity.t3.enable, derm.identity.derange) \
        == (True, True, True)
    assert derm.extraction.chunk == base.extraction.chunk  # inherited
    assert load("ab_eva_only.yaml").ablation.input_mode == "eva_only"
    assert load("ab_of3_only.yaml").ablation.input_mode == "of3_only"
    assert load("extract_uniform512.yaml").run.mode == "extract"
    assert load("di_t1.yaml").run.run_id == "di-t1"


def test_missing_file_is_path_error(tmp_path):
    with pytest.raises(EvaDiPathError):
        load_config(tmp_path / "nope.yaml")


def test_schema_version_and_sections():
    payload = _base_payload()
    payload["schema_version"] = "eva_di_config_v0"
    with pytest.raises(EvaDiConfigError, match="schema_version"):
        load_config(_mktmp(payload))


def test_unknown_keys_anywhere_are_fatal():
    payload = _base_payload()
    payload["learning_rate"] = 0.1
    with pytest.raises(EvaDiConfigError, match="unknown key"):
        load_config(_mktmp(payload))
    payload = _base_payload()
    payload["training"]["momentum"] = 0.9
    with pytest.raises(EvaDiConfigError, match="momentum"):
        load_config(_mktmp(payload))
    payload = _base_payload()
    del payload["model"]
    with pytest.raises(EvaDiConfigError, match="missing required section"):
        load_config(_mktmp(payload))


def test_test_split_is_refused_at_parse_time():
    payload = _base_payload()
    payload["data"]["splits"] = ["train", "val", "test"]
    with pytest.raises(EvaDiConfigError, match="test"):
        load_config(_mktmp(payload))
    payload = _base_payload()
    payload["data"]["splits"] = ["val"]  # train required for train-only stats
    with pytest.raises(EvaDiConfigError, match="train"):
        load_config(_mktmp(payload))


@pytest.mark.parametrize("section,key,value", [
    ("model", "proj_dim", 96),
    ("model", "proj_dropout", 1.0),
    ("extraction", "backbone", "eva02_base_patch14_224"),
    ("extraction", "weight_sha256", "0" * 64),
    ("extraction", "frame_budget", 256),
    ("extraction", "input_size", 336),
    ("training", "precision", "fp16-mixed"),
    ("training", "grad_clip_norm", 0),
])
def test_pinned_values_cannot_drift(section, key, value):
    payload = _base_payload()
    payload[section][key] = value
    with pytest.raises(EvaDiConfigError):
        load_config(_mktmp(payload))


def test_identity_head_ranges_and_seed():
    payload = _base_payload()
    payload["identity"]["t1"]["lam"] = 0.0
    with pytest.raises(EvaDiConfigError, match="lam"):
        load_config(_mktmp(payload))
    payload = _base_payload()
    payload["identity"]["t3"]["enable"] = "yes"
    with pytest.raises(EvaDiConfigError, match="enable"):
        load_config(_mktmp(payload))
    payload = _base_payload()
    payload["run"]["seed"] = True  # bool is not a seed
    with pytest.raises(EvaDiConfigError, match="seed"):
        load_config(_mktmp(payload))
    payload = _base_payload()
    payload["run"]["mode"] = "finetune"
    with pytest.raises(EvaDiConfigError, match="mode"):
        load_config(_mktmp(payload))


def test_behavior_norm_policy_and_provenance_fields():
    cfg = load_config(CONFIG_DIR / "eva_di_base.yaml")
    assert cfg.data.behavior_norm == "reuse"  # v1 pinned policy
    assert cfg.config_name == "eva_di_base"
    assert cfg.resolved_payload.get("schema_version") == SCHEMA_CONFIG
    payload = _base_payload()
    payload["data"]["behavior_norm"] = "recompute"
    assert load_config(_mktmp(payload)).data.behavior_norm == "recompute"
    payload["data"]["behavior_norm"] = "online"
    with pytest.raises(EvaDiConfigError, match="behavior_norm"):
        load_config(_mktmp(payload))


def test_extends_cycle_is_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(EvaDiConfigError, match="cycle"):
        load_config(tmp_path / "a.yaml")


def test_extends_deep_merge_override_precedence(tmp_path):
    base = copy.deepcopy(_base_payload())
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base), encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text(
        yaml.safe_dump({"extends": "base.yaml",
                        "identity": {"t1": {"enable": True}},
                        "run": {"run_id": "child"}}), encoding="utf-8")
    cfg = load_config(child)
    assert cfg.identity.t1.enable is True
    assert cfg.identity.t1.weight == base["identity"]["t1"]["weight"]  # merged, not replaced
    assert cfg.identity.t3.enable is False
    assert cfg.run.run_id == "child"
    assert cfg.run.seed == base["run"]["seed"]
    assert SCHEMA_CONFIG == "eva_di_config_v1"
