"""Atomic writer for ``eva_di_frame_feature_cache_v1`` (trans-pinned discipline).

Mirrors the proven trans builder contract (build_frozen_region_feature_cache.py
:40-58): unique ``.{name}.{pid}.tmp`` + fsync + rename, per-array SHA256, JSON
manifest written last, resumable via manifest-vs-file SHA agreement.  The
writer enforces the SAME required-field sets as the reader (contracts).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.eva_di.contracts import (
    CACHE_CLS_FILE,
    CACHE_GAP_FILE,
    CACHE_INDEX_FILE,
    CACHE_MANIFEST_CSV_SHA_KEY,
    CACHE_MANIFEST_FILE,
    CACHE_META_FILE,
    CACHE_VALID_FILE,
    EVA_FEATURE_DIM,
    REQUIRED_ENTRY_FIELDS,
    REQUIRED_MANIFEST_FIELDS,
    REQUIRED_META_FIELDS,
    EvaDiSchemaError,
)
from src.eva_di.paths import sha256_file

_HEX64 = set("0123456789abcdef")


@dataclass(frozen=True)
class SelectionArrays:
    cls: np.ndarray  # [n,384] float32
    gap: np.ndarray  # [n,384] float32
    frame_index_zero: np.ndarray  # [n] int64
    frame_valid: np.ndarray  # [n] bool
    meta: dict


def _fsync_dir(directory: Path) -> None:
    try:  # best effort: make the rename itself durable where the FS allows
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _tmp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _atomic_npy(path: Path, array: np.ndarray) -> str:
    tmp = _tmp_path(path)
    try:
        with tmp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)  # no-op after a successful rename
    _fsync_dir(path.parent)
    return sha256_file(path)


def _atomic_json(path: Path, payload: dict) -> str:
    tmp = _tmp_path(path)
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    _fsync_dir(path.parent)
    return sha256_file(path)


def _check_csv_sha(value: str, where: str) -> str:
    if value and (len(value) != 64 or not set(value) <= _HEX64):
        raise EvaDiSchemaError(f"{where} is not a 64-hex sha256 (or empty): {value!r}")
    return value


def validate_selection(sel: SelectionArrays) -> None:
    n = int(sel.cls.shape[0])
    if sel.cls.ndim != 2 or sel.cls.shape[1] != EVA_FEATURE_DIM \
            or sel.gap.shape != (n, EVA_FEATURE_DIM):
        raise EvaDiSchemaError(
            f"cls/gap must be [n,{EVA_FEATURE_DIM}]; got {sel.cls.shape}/{sel.gap.shape}")
    if sel.cls.dtype != np.float32 or sel.gap.dtype != np.float32:
        raise EvaDiSchemaError("cls/gap must be float32")
    if sel.frame_index_zero.shape != (n,) or sel.frame_index_zero.dtype != np.int64:
        raise EvaDiSchemaError("frame_index_zero must be int64 [n]")
    if sel.frame_valid.shape != (n,) or sel.frame_valid.dtype != np.bool_:
        raise EvaDiSchemaError("frame_valid must be bool [n]")
    if n and np.any(np.diff(sel.frame_index_zero) <= 0):
        raise EvaDiSchemaError("frame_index_zero must be strictly increasing")
    if n and not bool(sel.frame_valid.any()):
        raise EvaDiSchemaError("cache sample must carry at least one valid frame")
    if not np.isfinite(sel.cls).all() or not np.isfinite(sel.gap).all():
        raise EvaDiSchemaError("features contain non-finite values")
    if not isinstance(sel.meta, dict):
        raise EvaDiSchemaError("selection meta must be a dict")
    missing = [f for f in REQUIRED_META_FIELDS if f not in sel.meta]
    if missing:
        # audit R2: provenance fields used to fail open with .get() defaults
        raise EvaDiSchemaError(f"selection meta missing required field(s) {missing}")
    _check_csv_sha(str(sel.meta["features_sha256"]), "meta features_sha256")


def entry_matches(directory: Path, entry: dict) -> bool:
    """Resume probe: full SHA agreement between manifest entry and files.

    Any IO error (unreadable/corrupt file) means "not reusable", never a
    crash mid-resume (audit R2 runner P3).
    """
    try:
        for key, filename in (
            ("cls", CACHE_CLS_FILE), ("gap", CACHE_GAP_FILE),
            ("index", CACHE_INDEX_FILE), ("valid", CACHE_VALID_FILE),
        ):
            path = directory / filename
            want = entry.get(f"{key}_sha256")
            if not path.is_file() or want != sha256_file(path):
                return False
        meta_path = directory / CACHE_META_FILE
        if not meta_path.is_file():
            return False
        return entry.get("meta_sha256") == sha256_file(meta_path)
    except OSError:
        return False


def write_video_cache(cache_root: Path, sample_id: str, sel: SelectionArrays,
                      *, resume_entry: dict | None = None) -> tuple[dict, bool]:
    """Write one sample; returns (manifest_entry, reused).

    ``resume_entry`` (from an existing manifest) short-circuits on full SHA
    agreement — that is the resume path, never a silent overwrite.
    """
    validate_selection(sel)
    directory = Path(cache_root) / sample_id
    if resume_entry is not None and directory.is_dir() and entry_matches(directory, resume_entry):
        return dict(resume_entry), True
    directory.mkdir(parents=True, exist_ok=True)
    entry = {
        "n_frames": int(sel.cls.shape[0]),
        "n_valid": int(sel.frame_valid.sum()),
        "cls_sha256": _atomic_npy(directory / CACHE_CLS_FILE, sel.cls),
        "gap_sha256": _atomic_npy(directory / CACHE_GAP_FILE, sel.gap),
        "index_sha256": _atomic_npy(directory / CACHE_INDEX_FILE, sel.frame_index_zero),
        "valid_sha256": _atomic_npy(directory / CACHE_VALID_FILE, sel.frame_valid),
    }
    entry["meta_sha256"] = _atomic_json(directory / CACHE_META_FILE, dict(sel.meta))
    entry["features_sha256"] = _check_csv_sha(
        str(sel.meta["features_sha256"]), "entry features_sha256")
    return entry, False


def write_manifest(cache_root: Path, manifest: dict) -> str:
    """Validate the full manifest against the reader's required sets BEFORE
    the atomic rename (audit R2: the writer used to accept manifests its own
    reader would refuse)."""
    missing = [f for f in REQUIRED_MANIFEST_FIELDS if f not in manifest]
    if missing:
        raise EvaDiSchemaError(f"cache manifest missing fields: {missing}")
    samples = manifest["samples"]
    if not isinstance(samples, dict):
        raise EvaDiSchemaError("manifest samples must be a mapping")
    if int(manifest["sample_count"]) != len(samples):
        raise EvaDiSchemaError("sample_count must match len(samples)")
    csv_map = manifest.get(CACHE_MANIFEST_CSV_SHA_KEY) or {}
    for sid, entry in samples.items():
        if not isinstance(entry, dict):
            raise EvaDiSchemaError(f"manifest sample {sid!r} is not a mapping")
        absent = [f for f in REQUIRED_ENTRY_FIELDS if f not in entry]
        if absent:
            raise EvaDiSchemaError(f"manifest sample {sid!r} missing {absent}")
        entry_sha = _check_csv_sha(str(entry.get("features_sha256", "")),
                                   f"sample {sid} features_sha256")
        map_sha = _check_csv_sha(str(csv_map.get(sid, "")),
                                 f"csv sha map for {sid}")
        if entry_sha and map_sha and entry_sha != map_sha:
            raise EvaDiSchemaError(
                f"sample {sid}: entry features_sha256 {entry_sha} disagrees "
                f"with features_csv_sha256_by_sample {map_sha}")
    return _atomic_json(Path(cache_root) / CACHE_MANIFEST_FILE, manifest)


def read_manifest(cache_root: Path) -> dict | None:
    path = Path(cache_root) / CACHE_MANIFEST_FILE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaDiSchemaError(f"corrupt cache manifest: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaDiSchemaError(f"cache manifest must be a JSON object: {path}")
    return payload
