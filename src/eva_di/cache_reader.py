"""Strict reader for ``eva_di_frame_feature_cache_v1``.

Manifest fingerprints are compared field-by-field against the caller-provided
expected fingerprint (encoder metadata) and every per-file SHA is verified on
load (default).  Design doc section 5: missing samples and fingerprint drift
raise with field-level detail; nothing is silently skipped or fixed.

Since audit R2 the reader additionally: rejects unknown keys in ``expected``
(a typo there used to silently disable one check), verifies ``of3_root``
against the caller's real path when provided, validates manifest entries
(sha keys, n_frames/n_valid), verifies ``meta.json`` against its manifest
entry sha, and cross-checks per-sample csv provenance through
:meth:`FrozenFrameCache.csv_sha256`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.eva_di.contracts import (
    CACHE_CLS_FILE,
    CACHE_FEATURE_DTYPE,
    CACHE_GAP_FILE,
    CACHE_INDEX_FILE,
    CACHE_MANIFEST_CSV_SHA_KEY,
    CACHE_MANIFEST_FILE,
    CACHE_MANIFEST_OF3_ROOT_KEY,
    CACHE_META_FILE,
    CACHE_VALID_DTYPE,
    CACHE_VALID_FILE,
    REQUIRED_ENTRY_FIELDS,
    REQUIRED_MANIFEST_FIELDS,
    SCHEMA_FRAME_CACHE,
    EvaDiFingerprintError,
    EvaDiPathError,
    EvaDiSchemaError,
)
from src.eva_di.paths import sha256_file

# REQUIRED_MANIFEST_FIELDS / REQUIRED_ENTRY_FIELDS live in contracts (single
# source; the writer enforces the same sets) and are re-exported here for
# existing importers.

FINGERPRINT_FIELDS = (
    "schema_version", "selection_policy", "frame_budget", "backbone",
    "weight_sha256", "weight_revision", "timm_version", "input_size",
    "feature_dim", "feature_dtype",
)


@dataclass(frozen=True)
class CachedSample:
    sample_id: str
    cls: np.ndarray  # [n,384] float32
    gap: np.ndarray  # [n,384] float32
    frame_index_zero: np.ndarray  # [n] int64
    frame_valid: np.ndarray  # [n] bool
    meta: dict


def _load_json_object(path: Path, what: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaDiSchemaError(f"corrupt {what}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaDiSchemaError(f"{what} must be a JSON object: {path}")
    return payload


class FrozenFrameCache:
    def __init__(
        self,
        cache_root: Path,
        *,
        expected: dict | None = None,
        verify_hashes: bool = True,
        expected_of3_root: Path | str | None = None,
    ) -> None:
        self.root = Path(cache_root)
        manifest_path = self.root / CACHE_MANIFEST_FILE
        if not manifest_path.is_file():
            raise EvaDiPathError(f"cache manifest missing: {manifest_path}")
        self.manifest = _load_json_object(manifest_path, "cache manifest")
        missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in self.manifest]
        if missing:
            raise EvaDiSchemaError(f"cache manifest missing fields: {missing}")
        samples = self.manifest["samples"]
        if not isinstance(samples, dict) or not samples:
            raise EvaDiSchemaError("cache manifest samples must be a non-empty mapping")
        for sid, entry in samples.items():
            if not isinstance(entry, dict):
                raise EvaDiSchemaError(f"manifest sample {sid!r} is not a mapping")
            absent = [f for f in REQUIRED_ENTRY_FIELDS if f not in entry]
            if absent:
                raise EvaDiSchemaError(f"manifest sample {sid!r} missing {absent}")
        if int(self.manifest["sample_count"]) != len(samples):
            raise EvaDiSchemaError(
                f"sample_count {self.manifest['sample_count']} != len(samples) {len(samples)}"
            )
        if self.manifest["schema_version"] != SCHEMA_FRAME_CACHE:
            raise EvaDiFingerprintError(
                f"cache schema {self.manifest['schema_version']!r} != {SCHEMA_FRAME_CACHE!r}"
            )
        if expected is not None:
            unknown = sorted(set(expected) - set(FINGERPRINT_FIELDS))
            if unknown:
                raise EvaDiSchemaError(
                    f"expected fingerprint contains unchecked key(s) {unknown}; "
                    "extend FINGERPRINT_FIELDS instead of silently ignoring them")
            diffs = []
            for field in FINGERPRINT_FIELDS:
                want = expected.get(field)
                if want is not None and self.manifest.get(field) != want:
                    diffs.append(f"{field}: manifest={self.manifest.get(field)!r} expected={want!r}")
            if diffs:
                raise EvaDiFingerprintError("cache fingerprint mismatch: " + "; ".join(diffs))
        if expected_of3_root is not None:
            want_root = os.path.realpath(str(Path(expected_of3_root).expanduser()))
            got_root = os.path.realpath(str(self.manifest[CACHE_MANIFEST_OF3_ROOT_KEY]))
            if want_root != got_root:
                raise EvaDiFingerprintError(
                    "cache was extracted against a different OF3 root: "
                    f"manifest={got_root!r} expected={want_root!r}")
        self.verify_hashes = bool(verify_hashes)
        self._manifest_sha = sha256_file(manifest_path)
        self._loaded: dict[str, CachedSample] = {}

    @property
    def manifest_sha256(self) -> str:
        return self._manifest_sha

    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.manifest["samples"]))

    def csv_sha256(self, sample_id: str) -> str | None:
        """Pinned features.csv sha for one sample (None when unrecorded)."""
        mapping = self.manifest.get(CACHE_MANIFEST_CSV_SHA_KEY) or {}
        value = str(mapping.get(sample_id, "") or "")
        return value or None

    def load_sample(self, sample_id: str) -> CachedSample:
        if sample_id in self._loaded:
            return self._loaded[sample_id]
        entry = self.manifest["samples"].get(sample_id)
        if entry is None:
            raise EvaDiPathError(f"sample {sample_id!r} absent from cache manifest")
        directory = self.root / sample_id
        arrays: dict[str, np.ndarray] = {}
        for key, filename, expected_dtype, sha_key in (
            ("cls", CACHE_CLS_FILE, CACHE_FEATURE_DTYPE, "cls_sha256"),
            ("gap", CACHE_GAP_FILE, CACHE_FEATURE_DTYPE, "gap_sha256"),
            ("frame_index_zero", CACHE_INDEX_FILE, "int64", "index_sha256"),
            ("frame_valid", CACHE_VALID_FILE, CACHE_VALID_DTYPE, "valid_sha256"),
        ):
            path = directory / filename
            if not path.is_file():
                raise EvaDiPathError(f"cache file missing: {path}")
            if self.verify_hashes:
                want = entry.get(sha_key)
                got = sha256_file(path)
                if want != got:
                    raise EvaDiFingerprintError(
                        f"{sample_id}/{filename}: sha256 {got} != manifest {want}"
                    )
            array = np.load(path, allow_pickle=False)
            if np.dtype(expected_dtype) == np.dtype("bool"):
                if array.dtype != np.bool_:
                    raise EvaDiSchemaError(f"{sample_id}/{filename}: expected bool, got {array.dtype}")
            elif array.dtype != np.dtype(expected_dtype):
                raise EvaDiSchemaError(
                    f"{sample_id}/{filename}: expected {expected_dtype}, got {array.dtype}"
                )
            arrays[key] = array
        n = arrays["cls"].shape[0]
        if arrays["cls"].shape != (n, int(self.manifest["feature_dim"])):
            raise EvaDiSchemaError(f"{sample_id}: cls shape {arrays['cls'].shape}")
        if arrays["gap"].shape != (n, int(self.manifest["feature_dim"])):
            raise EvaDiSchemaError(f"{sample_id}: gap shape {arrays['gap'].shape}")
        if arrays["frame_index_zero"].shape != (n,):
            raise EvaDiSchemaError(f"{sample_id}: index shape {arrays['frame_index_zero'].shape}")
        if arrays["frame_valid"].shape != (n,):
            raise EvaDiSchemaError(f"{sample_id}: valid shape {arrays['frame_valid'].shape}")
        if int(entry["n_frames"]) != n:
            raise EvaDiSchemaError(
                f"{sample_id}: manifest n_frames {entry['n_frames']} != {n}"
            )
        if int(entry["n_valid"]) != int(arrays["frame_valid"].sum()):
            raise EvaDiSchemaError(
                f"{sample_id}: manifest n_valid {entry['n_valid']} != "
                f"{int(arrays['frame_valid'].sum())} valid rows on disk"
            )
        meta_path = directory / CACHE_META_FILE
        meta: dict = {}
        if meta_path.is_file():
            meta = _load_json_object(meta_path, f"{sample_id} meta")
            if self.verify_hashes and entry.get("meta_sha256"):
                want_meta = entry["meta_sha256"]
                got_meta = sha256_file(meta_path)
                if got_meta != want_meta:
                    raise EvaDiFingerprintError(
                        f"{sample_id}/{CACHE_META_FILE}: sha256 {got_meta} "
                        f"!= manifest {want_meta}")
            if meta.get("sample_id", sample_id) != sample_id:
                raise EvaDiSchemaError(
                    f"{sample_id}: meta sample_id {meta['sample_id']!r} mismatch")
        elif self.verify_hashes and entry.get("meta_sha256"):
            raise EvaDiPathError(f"cache file missing: {meta_path}")
        if not np.any(arrays["frame_valid"]):
            raise EvaDiSchemaError(f"{sample_id}: cache sample carries zero valid frames")
        sample = CachedSample(
            sample_id=sample_id,
            cls=arrays["cls"],
            gap=arrays["gap"],
            frame_index_zero=arrays["frame_index_zero"],
            frame_valid=arrays["frame_valid"],
            meta=meta,
        )
        self._loaded[sample_id] = sample
        return sample

    def counts(self) -> dict[str, int]:
        return {
            "sample_count": int(self.manifest["sample_count"]),
            "n_rows_total": int(self.manifest["n_rows_total"]),
            "n_selected_total": int(self.manifest["n_selected_total"]),
            "n_valid_total": int(self.manifest["n_valid_total"]),
            "n_exact_zero_masked": int(self.manifest["n_exact_zero_masked"]),
        }
