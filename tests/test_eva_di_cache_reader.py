"""cache_reader: manifest contract, fingerprint diff, per-file SHA verification."""

import json

import numpy as np
import pytest

from src.eva_di.cache_reader import FrozenFrameCache
from src.eva_di.contracts import (
    CACHE_CLS_FILE,
    CACHE_MANIFEST_FILE,
    SCHEMA_FRAME_CACHE,
    EvaDiFingerprintError,
    EvaDiPathError,
    EvaDiSchemaError,
)

from tests._eva_di_synth import build_tree

SAMPLE = "203_1_Freeform_video"


@pytest.fixture()
def tree(tmp_path):
    return build_tree(tmp_path / "synth", frames_per_video=6)


def _manifest(tree):
    path = tree.cache_root / CACHE_MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8")), path


def _rewrite(tree, payload):
    (tree.cache_root / CACHE_MANIFEST_FILE).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_load_round_trip_and_counts(tree):
    cache = FrozenFrameCache(tree.cache_root, verify_hashes=True)
    sample = cache.load_sample(SAMPLE)
    manifest, _ = _manifest(tree)
    n = manifest["samples"][SAMPLE]["n_frames"]
    assert sample.cls.shape == (n, 384) and sample.cls.dtype == np.float32
    assert sample.gap.shape == (n, 384)
    assert sample.frame_index_zero.dtype == np.int64 and sample.frame_valid.dtype == np.bool_
    assert np.all(np.diff(sample.frame_index_zero) > 0)
    assert cache.counts()["sample_count"] == 6
    assert SAMPLE in cache.sample_ids()
    assert cache.load_sample(SAMPLE) is sample  # cached object reuse


def test_missing_manifest_raises_path_error(tmp_path):
    with pytest.raises(EvaDiPathError):
        FrozenFrameCache(tmp_path / "nothing")


def test_schema_version_drift_is_fingerprint_error(tree):
    payload, _ = _manifest(tree)
    payload["schema_version"] = "eva_di_frame_feature_cache_v0"
    _rewrite(tree, payload)
    with pytest.raises(EvaDiFingerprintError):
        FrozenFrameCache(tree.cache_root)


def test_sample_count_inconsistency_rejected(tree):
    payload, _ = _manifest(tree)
    payload["sample_count"] = 5
    _rewrite(tree, payload)
    with pytest.raises(EvaDiSchemaError):
        FrozenFrameCache(tree.cache_root)


def test_required_field_missing_rejected(tree):
    payload, _ = _manifest(tree)
    del payload["frame_budget"]
    _rewrite(tree, payload)
    with pytest.raises(EvaDiSchemaError, match="frame_budget"):
        FrozenFrameCache(tree.cache_root)


def test_expected_fingerprint_reports_field_level_diff(tree):
    with pytest.raises(EvaDiFingerprintError, match="backbone:"):
        FrozenFrameCache(tree.cache_root, expected={"backbone": "some_other_backbone"})
    # matching fingerprint passes
    FrozenFrameCache(tree.cache_root, expected={
        "schema_version": SCHEMA_FRAME_CACHE, "backbone": payload_backbone(tree)})


def payload_backbone(tree) -> str:
    return _manifest(tree)[0]["backbone"]


def test_silent_sample_skip_is_impossible(tree):
    cache = FrozenFrameCache(tree.cache_root)
    with pytest.raises(EvaDiPathError, match="absent from cache manifest"):
        cache.load_sample("999_9_Freeform_video")


def test_tampered_array_caught_by_sha(tree):
    path = tree.cache_root / SAMPLE / CACHE_CLS_FILE
    array = np.load(path, allow_pickle=False)
    array[0, 0] += 1.0
    np.save(path, array, allow_pickle=False)
    with pytest.raises(EvaDiFingerprintError, match=CACHE_CLS_FILE):
        FrozenFrameCache(tree.cache_root, verify_hashes=True).load_sample(SAMPLE)
    # with verification off the tampered file still loads (documented behaviour)
    loose = FrozenFrameCache(tree.cache_root, verify_hashes=False).load_sample(SAMPLE)
    assert loose.cls.shape[1] == 384
