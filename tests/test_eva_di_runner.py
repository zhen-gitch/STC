"""runner: worklist/jobs_sha256, manifest provenance fields, kill-safe resume.

The encoder is injected (pure-CPU fake); no torch model, no GPU, no real data.
"""

import csv
import hashlib
import json
import sys

import numpy as np
import pytest

from src.eva_di.cache_reader import FrozenFrameCache
from src.eva_di.contracts import (
    CACHE_MANIFEST_CSV_SHA_KEY,
    CACHE_MANIFEST_FILE,
    CACHE_MANIFEST_OF3_ROOT_KEY,
    SCHEMA_RUNNER_LOG,
    EvaDiError,
    EvaDiFingerprintError,
    EvaDiSchemaError,
)
from src.eva_di.extract import runner as runner_module
from src.eva_di.extract.cache_writer import SelectionArrays, write_manifest
from src.eva_di.extract.runner import run_extraction
from src.eva_di.paths import PathSet, sha256_file

from tests._eva_di_synth import build_tree

RNG = np.random.default_rng(99)


def _csv_sha(of3_root, sample_id: str) -> str:
    return sha256_file(of3_root / "samples" / sample_id / "features.csv")


@pytest.fixture()
def env(tmp_path):
    tree = build_tree(tmp_path / "synth", frames_per_video=6)
    cache_root = tmp_path / "fresh_cache"
    paths = PathSet.build(overrides={
        "of3_root": tree.of3_root, "avec_root": tree.avec_root,
        "split_file": tree.split_file, "label_dir": tree.label_dir,
        "cache_root": cache_root, "weight_path": tree.weight_path,
        "behavior_norm_root": tree.norm_root,
        "log_root": tmp_path / "logs", "output_root": tmp_path / "outputs",
    })

    def encode_video(path_set, sample_id):
        csv_path = path_set.of3_root / "samples" / sample_id / "features.csv"
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        indices = [int(r["frame_index_zero"]) for r in rows if r["image_valid"] == "True"]
        n = len(indices)
        sel = SelectionArrays(
            cls=RNG.normal(0, 1, (n, 384)).astype(np.float32),
            gap=RNG.normal(0, 1, (n, 384)).astype(np.float32),
            frame_index_zero=np.asarray(indices, dtype=np.int64),
            frame_valid=np.ones(n, dtype=bool),
            meta={"sample_id": sample_id, "n_rows": len(rows),
                  "n_image_valid": n, "n_selected": n,
                  "n_exact_zero_masked": 0, "selection_policy": "uniform512_v1",
                  # REAL csv sha (R2-P2-4: fake shas made the resume/drift
                  # paths untestable); re-read live so csv drift changes it.
                  "features_sha256": sha256_file(csv_path), "image_decode": "PIL"},
        )
        return sel, {"n_black": 0}

    meta = {"timm_version": "test", "torch_version": "test",
            "cuda_version": None, "device": "cpu"}
    return tree, paths, encode_video, meta


def _all_ids(tree):
    return sorted({*tree.splits["train"], *tree.splits["val"], *tree.splits["test"]})


def test_full_run_manifest_and_log(env):
    tree, paths, encode_video, meta = env
    log = run_extraction(paths, meta, encode_video, log_root=paths.log_root)
    assert log["schema"] == SCHEMA_RUNNER_LOG and log["accepted"] is True
    assert log["verify_image_sha"] is True
    assert log["jobs_sha256"] == hashlib.sha256(
        "\n".join(_all_ids(tree)).encode("utf-8")).hexdigest()
    assert [r["status"] for r in log["results"]] == ["completed"] * 6
    expected_map = {sid: _csv_sha(tree.of3_root, sid) for sid in _all_ids(tree)}
    assert all(r["features_sha256"] == expected_map[r["sample_id"]] for r in log["results"])
    assert all("seconds" in r for r in log["results"])
    assert log["throughput_frames_per_s"] and log["throughput_frames_per_s"] > 0
    logs = list(paths.log_root.glob("runner_*.json"))
    assert len(logs) == 1 and json.loads(logs[0].read_text())["jobs_sha256"] == log["jobs_sha256"]

    manifest = json.loads((paths.cache_root / CACHE_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest[CACHE_MANIFEST_OF3_ROOT_KEY] == str(tree.of3_root)
    assert manifest[CACHE_MANIFEST_CSV_SHA_KEY] == expected_map
    assert manifest["sample_count"] == 6
    assert "verify_image_sha" in manifest and manifest["timm_version"] == "test"
    # the produced cache satisfies the strict reader end-to-end
    assert FrozenFrameCache(paths.cache_root,
                            expected_of3_root=tree.of3_root).counts()["sample_count"] == 6


def test_resume_reuses_verified_entries_without_reencode(env):
    tree, paths, encode_video, meta = env
    calls = []

    def counting_encode(ps, sid):
        calls.append(sid)
        return encode_video(ps, sid)

    run_extraction(paths, meta, counting_encode)
    assert len(calls) == 6
    second = run_extraction(paths, meta, counting_encode)
    assert calls == [sid for sid in _all_ids(tree)]  # no re-encode on resume
    assert [r["status"] for r in second["results"]] == ["reused"] * 6


def test_resume_detects_csv_drift_and_reextracts(env):
    tree, paths, encode_video, meta = env
    run_extraction(paths, meta, encode_video)
    victim = _all_ids(tree)[0]
    csv_path = tree.of3_root / "samples" / victim / "features.csv"
    trailing = csv_path.read_bytes()[-1:]  # sha-changing byte drift
    with csv_path.open("ab") as handle:
        handle.write(b" " + trailing)
    calls = []

    def counting_encode(ps, sid):
        calls.append(sid)
        return encode_video(ps, sid)

    second = run_extraction(paths, meta, counting_encode)
    statuses = {r["sample_id"]: r["status"] for r in second["results"]}
    assert statuses[victim] == "completed"          # stale entry was replaced
    assert calls == [victim]
    assert all(s == "reused" for sid, s in statuses.items() if sid != victim)


def test_stale_manifest_environment_drift_is_refused(env):
    tree, paths, encode_video, meta = env
    run_extraction(paths, meta, encode_video)
    manifest_path = paths.cache_root / CACHE_MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["timm_version"] = "older-than-current"
    write_manifest(paths.cache_root, manifest)
    with pytest.raises(EvaDiFingerprintError, match="different provenance"):
        run_extraction(paths, meta, encode_video)


def test_failed_video_is_logged_before_abort(env, tmp_path):
    tree, paths, encode_video, meta = env
    boom = _all_ids(tree)[1]

    def flaky(ps, sid):
        if sid == boom:
            raise RuntimeError("synthetic encoder crash")
        return encode_video(ps, sid)

    with pytest.raises(RuntimeError, match="synthetic encoder crash"):
        run_extraction(paths, meta, flaky, log_root=paths.log_root)
    logs = list(paths.log_root.glob("runner_*.json"))
    assert len(logs) == 1
    log = json.loads(logs[0].read_text())
    assert log["accepted"] is False
    failed = [r for r in log["results"] if r["status"] == "failed"]
    assert len(failed) == 1 and failed[0]["sample_id"] == boom
    assert "synthetic encoder crash" in failed[0]["error"]


def test_explicit_worklist_and_rejects_unknown(env):
    tree, paths, encode_video, meta = env
    wanted = sorted({"203_1_Freeform_video", "209_1_Freeform_video"})
    log = run_extraction(paths, meta, encode_video, samples=wanted)
    assert [r["sample_id"] for r in log["results"]] == wanted
    with pytest.raises(EvaDiSchemaError, match="subset"):
        run_extraction(paths, meta, encode_video, samples=["ghost_video"])


def test_smoke_worklist_is_capped(env, monkeypatch):
    # monkeypatched cap: the old assertion was a tautology at 6<=10 fixtures
    monkeypatch.setattr(runner_module, "SMOKE_SAMPLE_COUNT", 2)
    tree, paths, encode_video, meta = env
    log = run_extraction(paths, meta, encode_video, smoke=True)
    assert [r["sample_id"] for r in log["results"]] == _all_ids(tree)[:2]


def test_extract_mode_guard_in_main(env, tmp_path):
    from src.eva_di.extract.runner import main
    from pathlib import Path

    cfg = tmp_path / "train_mode.yaml"
    cfg.write_text(
        (Path(__file__).resolve().parents[1] / "configs" / "eva_di" / "eva_di_base.yaml")
        .read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(EvaDiError, match="extract"):
        main(["--config", str(cfg)])


def test_light_env_guard_actually_fires(monkeypatch, env, tmp_path):
    # R2-P0(tests): the guard used to be vacuous when the suite itself ran in
    # 'light'; point sys.executable elsewhere and pin the refusal.
    from src.eva_di.extract.runner import main

    monkeypatch.setattr(sys, "executable", "/usr/bin/python3")
    with pytest.raises(EvaDiError, match="light"):
        main(["--config", str(tmp_path / "anything.yaml")])
