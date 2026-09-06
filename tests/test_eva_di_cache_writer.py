"""cache_writer: validation, atomic writes, manifest-last, resume semantics."""

import json

import numpy as np
import pytest

from src.eva_di.contracts import (
    CACHE_CLS_FILE,
    CACHE_MANIFEST_CSV_SHA_KEY,
    CACHE_MANIFEST_FILE,
    CACHE_MANIFEST_OF3_ROOT_KEY,
    CACHE_META_FILE,
    SELECTION_POLICY,
    EvaDiSchemaError,
)
from src.eva_di.extract.cache_writer import (
    SelectionArrays,
    entry_matches,
    read_manifest,
    validate_selection,
    write_manifest,
    write_video_cache,
)
from src.eva_di.paths import sha256_file

RNG = np.random.default_rng(11)


def _meta(n=5, features_sha256=""):
    return {"sample_id": "s", "selection_policy": SELECTION_POLICY,
            "n_rows": n, "n_image_valid": n, "n_selected": n,
            "n_exact_zero_masked": 0, "features_sha256": features_sha256}


def _sel(n=5, *, dtype=np.float32, indices=None, valid=None, meta=None):
    return SelectionArrays(
        cls=RNG.normal(0, 1, (n, 384)).astype(dtype),
        gap=RNG.normal(0, 1, (n, 384)).astype(dtype),
        frame_index_zero=np.asarray(indices if indices is not None else range(n),
                                     dtype=np.int64),
        frame_valid=np.asarray(valid if valid is not None else [True] * n, dtype=bool),
        meta=meta if meta is not None else _meta(n),
    )


def test_validate_rejects_bad_schema():
    with pytest.raises(EvaDiSchemaError):
        validate_selection(_sel(dtype=np.float64))
    bad = _sel()
    bad.frame_index_zero[1] = bad.frame_index_zero[0]  # not strictly increasing
    with pytest.raises(EvaDiSchemaError):
        validate_selection(bad)
    with pytest.raises(EvaDiSchemaError):
        validate_selection(_sel(valid=[False] * 5))  # zero valid frames
    badfin = _sel()
    badfin.cls[0, 0] = np.nan
    with pytest.raises(EvaDiSchemaError):
        validate_selection(badfin)


def test_validate_requires_full_meta_and_hex_csv_sha():
    # R2-P2: provenance fields used to fail open through .get() defaults.
    short = _meta()
    del short["n_image_valid"]
    with pytest.raises(EvaDiSchemaError, match="n_image_valid"):
        validate_selection(_sel(meta=short))
    with pytest.raises(EvaDiSchemaError, match="64-hex"):
        validate_selection(_sel(meta=_meta(features_sha256="nothex")))
    validate_selection(_sel(meta=_meta(features_sha256="a" * 64)))  # ok


def test_write_round_trip_and_atomicity(tmp_path):
    entry, reused = write_video_cache(tmp_path, "203_1_Freeform_video", _sel())
    assert reused is False
    directory = tmp_path / "203_1_Freeform_video"
    assert entry["n_frames"] == 5 and entry["n_valid"] == 5
    assert entry["cls_sha256"] == sha256_file(directory / CACHE_CLS_FILE)
    assert entry["meta_sha256"] == sha256_file(directory / CACHE_META_FILE)
    assert list(directory.glob(".*.tmp")) == []  # no temp-file litter
    loaded = np.load(directory / CACHE_CLS_FILE, allow_pickle=False)
    assert loaded.shape == (5, 384) and loaded.dtype == np.float32


def test_resume_short_circuits_on_sha_agreement(tmp_path):
    sel = _sel()
    entry, _ = write_video_cache(tmp_path, "vid", sel)
    again, reused = write_video_cache(tmp_path, "vid", sel, resume_entry=entry)
    assert reused is True
    assert again == entry


def test_resume_mismatch_rewrites_and_matches_after(tmp_path):
    sel = _sel()
    entry, _ = write_video_cache(tmp_path, "vid", sel)
    cls_path = tmp_path / "vid" / CACHE_CLS_FILE
    corrupted = np.load(cls_path, allow_pickle=False)
    corrupted[0, 0] += 1.0
    np.save(cls_path, corrupted, allow_pickle=False)
    assert entry_matches(tmp_path / "vid", entry) is False
    fresh, reused = write_video_cache(tmp_path, "vid", sel, resume_entry=entry)
    assert reused is False
    assert fresh["cls_sha256"] == sha256_file(cls_path)
    assert entry_matches(tmp_path / "vid", fresh) is True


def _good_manifest(entry):
    return {
        "schema_version": "eva_di_frame_feature_cache_v1",
        "selection_policy": SELECTION_POLICY, "frame_budget": 512,
        "backbone": "eva02_small_patch14_224.mim_in22k",
        "weight_sha256": "d" * 64, "weight_revision": "e" * 40,
        "timm_version": "t", "input_size": 224,
        "feature_dim": 384, "feature_dtype": "float32", "valid_dtype": "bool",
        CACHE_MANIFEST_OF3_ROOT_KEY: "/synthetic/root",
        CACHE_MANIFEST_CSV_SHA_KEY: {},
        "sample_count": 1, "n_rows_total": 5, "n_selected_total": 5,
        "n_valid_total": 5, "n_exact_zero_masked": 0,
        "samples": {"a": entry},
    }


def test_manifest_structure_guards(tmp_path):
    # count mismatch on an otherwise valid manifest
    entry = {"cls_sha256": "a" * 64, "gap_sha256": "b" * 64,
             "index_sha256": "c" * 64, "valid_sha256": "d" * 64,
             "n_frames": 5, "n_valid": 5}
    bad_count = _good_manifest(entry) | {"sample_count": 2}
    with pytest.raises(EvaDiSchemaError, match="sample_count"):
        write_manifest(tmp_path, bad_count)
    # missing top-level fields now REFUSE (the old writer produced manifests
    # its own reader would reject; R2)
    with pytest.raises(EvaDiSchemaError, match="missing fields"):
        write_manifest(tmp_path, {"samples": {"a": {}}, "sample_count": 1})
    # entries missing required sha/count keys refuse
    with pytest.raises(EvaDiSchemaError, match="missing"):
        write_manifest(tmp_path, _good_manifest({"n_frames": 5}))
    # entry-vs-csv-map disagreement refuses
    crossed = _good_manifest(entry | {"features_sha256": "a" * 64})
    crossed[CACHE_MANIFEST_CSV_SHA_KEY] = {"a": "b" * 64}
    with pytest.raises(EvaDiSchemaError, match="disagrees"):
        write_manifest(tmp_path, crossed)
    sha = write_manifest(tmp_path, _good_manifest(entry))
    assert read_manifest(tmp_path)["sample_count"] == 1
    assert sha == sha256_file(tmp_path / CACHE_MANIFEST_FILE)
    assert read_manifest(tmp_path / "nowhere") is None


def test_read_manifest_rejects_corrupt_json(tmp_path):
    (tmp_path / CACHE_MANIFEST_FILE).write_text("{ not json", encoding="utf-8")
    with pytest.raises(EvaDiSchemaError, match="corrupt"):
        read_manifest(tmp_path)
