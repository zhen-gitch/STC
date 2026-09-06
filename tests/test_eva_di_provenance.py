"""provenance + entry-point guards: schema fields, test-split refusal, offline export guard.

These pin the *contract* of runtime entry points (train.main / export_embeddings)
without executing training: the GPU smoke belongs to EVA-DI-SMOKE-v1.
"""

from pathlib import Path

import pytest

from src.eva_di.contracts import PLAN_ID, SCHEMA_PROVENANCE, EvaDiError
from src.eva_di.config import load_config
from src.eva_di.paths import PathSet
from src.eva_di.paths import sha256_file
from src.eva_di.train import build_provenance

from tests._eva_di_synth import build_tree

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "eva_di"


@pytest.fixture()
def env(tmp_path):
    tree = build_tree(tmp_path / "synth", frames_per_video=4)
    paths = PathSet.build(overrides={
        "of3_root": tree.of3_root, "avec_root": tree.avec_root,
        "split_file": tree.split_file, "label_dir": tree.label_dir,
        "cache_root": tree.cache_root, "weight_path": tree.weight_path,
        "behavior_norm_root": tree.norm_root,
        "log_root": tmp_path / "logs", "output_root": tmp_path / "outputs",
    })
    cfg = load_config(CONFIG_DIR / "di_full.yaml")
    return cfg, paths, tree


def test_provenance_schema_and_provenance_fields(env):
    cfg, paths, tree = env
    manifest_sha = sha256_file(tree.cache_root / "cache_manifest.json")
    split_sha = sha256_file(tree.split_file)
    prov = build_provenance(cfg, paths, cache_manifest_sha=manifest_sha,
                            split_file_sha=split_sha)
    assert prov["schema"] == SCHEMA_PROVENANCE
    assert prov["plan_id"] == PLAN_ID
    assert prov["seed"] == cfg.run.seed
    assert prov["splits"] == ["train", "val"] and "test" not in prov["splits"]
    assert prov["ablation"] == "dual"
    assert prov["identity"] == {"t1": True, "t3": True, "derange": False,
                                "warmup_epochs": 5}
    assert prov["rerun_overwrite"] is False and prov["run_stats"] == {}
    assert prov["precision"] == "bf16-mixed" and prov["autocast_enabled"] is False
    assert prov["git_toplevel"] == str(prov["git_toplevel"])
    assert prov["config_source_sha256"] == cfg.source_sha256
    assert prov["config_resolved_sha256"] == cfg.resolved_sha256
    assert prov["cache_manifest_sha256"] == manifest_sha
    assert prov["split_file_sha256"] == split_sha
    assert isinstance(prov["git_dirty"], bool)
    for key in ("git_commit", "git_branch", "python", "torch_version", "argv",
                "device", "config_name", "behavior_norm_sha256"):
        assert key in prov
    assert prov["config_name"] == cfg.config_name == "di_full"
    assert set(prov["paths"]) == {
        "of3_root", "avec_root", "split_file", "label_dir", "cache_root",
        "weight_path", "behavior_norm_root", "log_root", "output_root"}


def test_run_training_refuses_test_split_before_any_io(env):
    from dataclasses import replace as dreplace

    cfg, _paths, _tree = env
    from src.eva_di.config import DataCfg

    poisoned = dreplace(cfg, data=DataCfg(splits=("train", "val", "test"),
                                          max_recordings=None))
    with pytest.raises(EvaDiError, match="test split is locked"):
        from src.eva_di.train import run_training

        run_training(poisoned, device="cpu")


def test_export_embeddings_refuses_test_split(env):
    from src.eva_di.export_embeddings import export_embeddings

    cfg, _paths, _tree = env
    with pytest.raises(EvaDiError, match="split must be one of"):
        export_embeddings(cfg, split="test", checkpoint=Path("unused.pt"))
