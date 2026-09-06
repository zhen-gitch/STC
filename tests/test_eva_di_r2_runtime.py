"""R2 runtime guards (CPU, synthetic): P0-A optimizer inclusion, positive
learning smoke, export guards, config/cache/dataset/metrics hardening.

The P0-A differential (1-epoch vs 2-epoch last.pt identity-head weights) is
the regression lock for "identity heads never reached the optimizer": a run
whose heads never update leaves them bit-identical at init across horizons.
"""

import csv
import json
from dataclasses import replace as dreplace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.eva_di.behavior import load_pinned_stats
from src.eva_di.cache_reader import FrozenFrameCache
from src.eva_di.config import AblationCfg, PathsCfg, load_config
from src.eva_di.contracts import (
    EvaDiConfigError,
    EvaDiError,
    EvaDiFingerprintError,
    EvaDiSchemaError,
)
from src.eva_di.dataset import build_recording_index
from src.eva_di.export_embeddings import export_embeddings
from src.eva_di.metrics import severity_group, severity_worst_group_mae
from src.eva_di.model import DualStreamDI
from src.eva_di.paths import PathSet
from src.eva_di.train import run_training

from tests._eva_di_synth import build_tree

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "eva_di"


def _paths(tree, tmp_path):
    return PathSet.build(overrides={
        "of3_root": tree.of3_root, "avec_root": tree.avec_root,
        "split_file": tree.split_file, "label_dir": tree.label_dir,
        "cache_root": tree.cache_root, "weight_path": tree.weight_path,
        "behavior_norm_root": tree.norm_root,
        "log_root": tmp_path / "logs", "output_root": tmp_path / "outputs",
    })


def _cfg(tree, tmp_path, *, epochs: int, run_id: str, base: str = "di_full.yaml"):
    cfg = load_config(CONFIG_DIR / base)
    training = dreplace(cfg.training, epochs_max=epochs, batch_recordings=2,
                        early_stop={"metric": "val_ccc", "patience": 2})
    identity = dreplace(
        cfg.identity,
        t1=dreplace(cfg.identity.t1, warmup_epochs=1),
        t3=dreplace(cfg.identity.t3, warmup_epochs=1),
    )
    return dreplace(cfg,
                    run=dreplace(cfg.run, run_id=run_id),
                    paths=PathsCfg(overrides={
                        k: str(v) for k, v in {
                            "of3_root": tree.of3_root, "avec_root": tree.avec_root,
                            "split_file": tree.split_file, "label_dir": tree.label_dir,
                            "cache_root": tree.cache_root, "weight_path": tree.weight_path,
                            "behavior_norm_root": tree.norm_root,
                            "log_root": tmp_path / "logs",
                            "output_root": tmp_path / "outputs"}.items()}),
                    training=training, identity=identity)


def _run_dir(cfg, tmp_path) -> Path:
    return tmp_path / "outputs" / cfg.run.run_id


@pytest.fixture()
def tree(tmp_path):
    return build_tree(tmp_path / "synth", frames_per_video=5)


def _reject(constant):  # strict JSON: NaN/Infinity must never serialize
    raise AssertionError(f"non-finite JSON constant in metrics: {constant}")


def test_run_training_updates_identity_heads(tmp_path, tree):
    cfg1 = _cfg(tree, tmp_path, epochs=1, run_id="r2-1ep")
    cfg2 = _cfg(tree, tmp_path, epochs=2, run_id="r2-2ep")
    summary1 = run_training(cfg1, device="cpu")
    summary2 = run_training(cfg2, device="cpu")
    assert summary1["epochs_run"] == 1 and summary2["epochs_run"] == 2
    for summary in (summary1, summary2):
        assert Path(summary["checkpoint"]).is_file()
        assert {"best_defined", "val_n", "n_subjects", "max_recordings",
                "stopped_early"} <= set(summary)

    last1 = torch.load(_run_dir(cfg1, tmp_path) / "checkpoints" / "last.pt",
                       map_location="cpu", weights_only=False)
    last2 = torch.load(_run_dir(cfg2, tmp_path) / "checkpoints" / "last.pt",
                       map_location="cpu", weights_only=False)
    s1, s2 = last1["model"], last2["model"]
    # heads EXIST in the checkpoint (lazy materialization before optimizer) ...
    for key in ("frame_head.weight", "video_head.weight"):
        assert key in s1 and key in s2
        # ... AND moved: identical head weights across horizons would mean the
        # heads were never in the optimizer (the P0-A failure mode).
        assert not torch.equal(s1[key], s2[key]), f"{key} never updated"
    # GRL lambdas were scheduled (full strength: warmup_epochs=1, epoch 1+)
    assert last2["provenance"]["identity"]["warmup_epochs"] == 1

    lines = (_run_dir(cfg2, tmp_path) / "metrics.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0], parse_constant=_reject)
    assert first["warmup_scale"] == 1.0 and first["lambda1_eff"] > 0


def test_rerun_resets_run_dir(tmp_path, tree):
    cfg = _cfg(tree, tmp_path, epochs=1, run_id="r2-reset")
    run_training(cfg, device="cpu")
    run_training(cfg, device="cpu")  # same run_id must RESET, not append
    lines = (_run_dir(cfg, tmp_path) / "metrics.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_export_roundtrip_and_guards(tmp_path, tree):
    cfg = _cfg(tree, tmp_path, epochs=1, run_id="r2-export")
    summary = run_training(cfg, device="cpu")
    dests = export_embeddings(cfg, split="val",
                              checkpoint=Path(summary["checkpoint"]), device="cpu")
    assert len(dests) == 2
    with np.load(dests[0], allow_pickle=True) as data:
        assert str(data["schema"]) == "eva_di_embeddings_v1"
        assert str(data["run_id"]) == "r2-export"
        assert len(str(data["checkpoint_sha256"])) == 64
        assert data["embeddings"].shape == (len(tree.splits["val"]), cfg.model.proj_dim)
        assert sorted(str(s) for s in data["sample_ids"]) == sorted(tree.splits["val"])
        assert str(data["input_mode"]) == cfg.ablation.input_mode

    with pytest.raises(EvaDiError, match="train-mode"):
        export_embeddings(dreplace(cfg, run=dreplace(cfg.run, mode="extract")),
                          split="val", checkpoint=Path(summary["checkpoint"]))
    wrong = dreplace(cfg, ablation=AblationCfg(input_mode="eva_only"))
    with pytest.raises(EvaDiError, match="ablation"):
        export_embeddings(wrong, split="val",
                          checkpoint=Path(summary["checkpoint"]), device="cpu")


# ---------------------------------------------------------------------------
# config guards (audit R2 config hardening)


def _mutate_base(tmp_path, old: str, new: str) -> Path:
    text = (CONFIG_DIR / "eva_di_base.yaml").read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "mut.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_config_type_and_consistency_guards(tmp_path):
    probes = [
        ("chunk: 128", "chunk: true"),                      # bool-as-int
        ("seed: 42", "seed: 4294967296"),                   # seed < 2**32
        ("run_id: null", "run_id: a/b"),                    # path-safe run_id
        ("splits: [train, val]", "splits: [train, train]"),  # dup split
        ("weight_decay: 0.01", "weight_decay: -1.0"),       # negative wd
        ("gru_hidden: 192", "gru_hidden: 96"),              # pinned proj==gru
        ("derange: false", "derange: true"),                # derange w/o head
        ("warmup_epochs: 5", "warmup_epochs: 101"),         # warmup > epochs_max(100)
    ]
    for original, mutated in probes:
        with pytest.raises(EvaDiConfigError):
            load_config(_mutate_base(tmp_path, original, mutated))


def test_config_shared_warmup_equality_enforced(tmp_path):
    # di_t3 enables T3 only; silently driving it from T1's warmup key (the old
    # dead-key trap) is now a load-time refusal.
    text = (CONFIG_DIR / "di_t3.yaml").read_text(encoding="utf-8")
    child = tmp_path / "di_t3.yaml"
    child.write_text(text, encoding="utf-8")  # same directory: extends resolves
    base = tmp_path / "eva_di_base.yaml"
    base.write_text((CONFIG_DIR / "eva_di_base.yaml").read_text(encoding="utf-8"),
                    encoding="utf-8")
    load_config(child)  # control: unpinned copy loads
    base.write_text((CONFIG_DIR / "eva_di_base.yaml").read_text(encoding="utf-8")
                    .replace("warmup_epochs: 5", "warmup_epochs: 3", 1),
                    encoding="utf-8")
    # t1 warmup=3, t3 warmup=5 -> schedule split -> refused
    with pytest.raises(EvaDiConfigError, match="warmup"):
        load_config(child)


def test_config_base_drift_visible_in_resolved_sha(tmp_path):
    child_dir = tmp_path / "cfg"
    child_dir.mkdir()
    base_text = (CONFIG_DIR / "eva_di_base.yaml").read_text(encoding="utf-8")
    child_text = (CONFIG_DIR / "di_full.yaml").read_text(encoding="utf-8")
    (child_dir / "eva_di_base.yaml").write_text(base_text, encoding="utf-8")
    child = child_dir / "di_full.yaml"
    child.write_text(child_text, encoding="utf-8")
    before = load_config(child)
    (child_dir / "eva_di_base.yaml").write_text(
        base_text.replace("weight: 0.05", "weight: 0.06"), encoding="utf-8")
    after = load_config(child)
    assert before.source_sha256 == after.source_sha256      # leaf untouched
    assert before.resolved_sha256 != after.resolved_sha256  # merge moved


# ---------------------------------------------------------------------------
# cache reader + dataset cross-source guards


def test_cache_reader_pins(tree):
    with pytest.raises(EvaDiSchemaError, match="unchecked"):
        FrozenFrameCache(tree.cache_root, expected={"bogus_key": 1})
    with pytest.raises(EvaDiFingerprintError, match="feature_dim"):
        FrozenFrameCache(tree.cache_root, expected={"feature_dim": 128})
    with pytest.raises(EvaDiFingerprintError, match="different OF3 root"):
        FrozenFrameCache(tree.cache_root, expected_of3_root=Path("/nonexistent-root"))
    ok = FrozenFrameCache(tree.cache_root, expected_of3_root=tree.of3_root)
    sid = sorted(ok.sample_ids())[0]
    meta_path = tree.cache_root / sid / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["n_selected"] += 1  # tamper AFTER the entry sha was pinned
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(EvaDiFingerprintError):
        FrozenFrameCache(tree.cache_root).load_sample(sid)


def test_dataset_split_dialect_and_sha_drift(tmp_path, tree):
    paths = _paths(tree, tmp_path)
    stats = load_pinned_stats(tree.norm_root)
    index = build_recording_index(paths, splits=("train", "val"),
                                  behavior_stats=stats, max_recordings=2)
    assert len(index.splits["train"]) == 2       # train-only truncation
    assert len(index.splits["val"]) == len(tree.splits["val"])  # val untouched
    assert all(r.subject == -1 for r in index.by_split("val"))

    victim = sorted(tree.splits["train"])[0]
    csv_path = tree.of3_root / "samples" / victim / "features.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    col = header.index("frame_index_zero")
    split_col = header.index("split")

    def _rewrite(new_rows: list[list[str]]) -> None:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([header] + new_rows)

    # 1) append one LEGAL extra frame row: csv stays parseable, sha changes,
    #    so the cross-source manifest-pin refusal (not a schema error) fires.
    bumped = list(body[-1])
    bumped[col] = str(int(body[-1][col]) + 1)
    _rewrite(body + [bumped])
    with pytest.raises(EvaDiFingerprintError, match="manifest pin"):
        build_recording_index(paths, splits=("train", "val"),
                              behavior_stats=stats)
    # 2) relabel the split column (csv-vs-json disagreement): schema-valid but
    #    split-inconsistent -> refused even with hash verification disabled.
    relabeled = [list(r) for r in body]
    for r in relabeled:
        r[split_col] = "test"
    _rewrite(relabeled)
    with pytest.raises(EvaDiSchemaError, match="does not match"):
        build_recording_index(paths, splits=("train", "val"),
                              behavior_stats=stats, verify_hashes=False)


# ---------------------------------------------------------------------------
# model per-flag heads + metrics hardening


def test_model_heads_mirror_flags(tree):
    cfg = load_config(CONFIG_DIR / "di_full.yaml")
    model = DualStreamDI(cfg.model, input_mode="dual", n_subjects=3,
                         t1_enable=True, t3_enable=False)
    model.materialize_identity_heads()
    state = model.state_dict()
    assert "frame_head.weight" in state and "video_head.weight" not in state
    both = DualStreamDI(cfg.model, input_mode="dual", n_subjects=3,
                        t1_enable=True, t3_enable=True)
    both.materialize_identity_heads()
    assert {"frame_head.weight", "video_head.weight"} <= set(both.state_dict())
    with pytest.raises(EvaDiSchemaError, match="gru_hidden"):
        dmodel = dreplace(cfg.model, gru_hidden=96)
        DualStreamDI(dmodel, input_mode="dual", n_subjects=3)


def test_metrics_finite_and_shape_discipline():
    assert severity_group(np.float32("nan")) == "unknown"
    assert severity_group("nonsense") == "unknown"
    assert severity_group(np.float32(30.0)) == "very_severe"
    with pytest.raises(ValueError, match="shape mismatch"):
        severity_worst_group_mae(np.zeros((3, 1)), np.zeros(3))
    worst, value, counts = severity_worst_group_mae(
        np.array([1.0, 15.0]), np.array([2.0, 16.0]))
    assert worst == "mild" and counts == {"mild": 1, "moderate": 1}  # tie -> order
    worst, _, counts = severity_worst_group_mae(
        np.array([float("nan")]), np.array([1.0]))
    assert worst == "unknown" and counts == {"unknown": 1}


def test_reuse_mode_skips_full_recompute_audit(tmp_path, tree):
    """User decision 2026-09-05: production ``reuse`` must NOT pay a full
    train-split recompute on the run path; the audit runs only under
    ``behavior_norm: recompute``.  Both modes still feed from pinned stats."""
    from src.eva_di.config import DataCfg

    reuse = _cfg(tree, tmp_path, epochs=1, run_id="r2-norm-reuse")
    assert reuse.data.behavior_norm == "reuse"
    s_reuse = run_training(reuse, device="cpu")
    assert "skipped" in s_reuse["behavior_stats_consistency"]

    audit = dreplace(reuse, data=DataCfg(splits=("train", "val"),
                                         max_recordings=None,
                                         behavior_norm="recompute"))
    audit = dreplace(audit, run=dreplace(audit.run, run_id="r2-norm-audit"))
    s_audit = run_training(audit, device="cpu")
    assert "audit PASSED" in s_audit["behavior_stats_consistency"]
