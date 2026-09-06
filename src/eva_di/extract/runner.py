"""Extraction driver (EVA-DI-EXTRACT-v1 boundary: code here, runs NOT authorized).

Pipeline per video (design doc sections 2-4): registry -> uniform512_v1 over
image-valid rows -> decode jpgs (+mandatory per-frame SHA semantics) ->
exact-zero black guard -> frozen EVA02 encode (chunked, bf16) -> atomic cache
write.  Resumable; runner log mirrors the trans ``feature_runner`` format;
10-video smoke worklist precedes the full 300.  Frozen-backbone extraction is
label-free, so the cache may cover all splits without leakage (validation plan
section 5.2); every downstream statistic stays train-only.

Audit R2 hardening: an EXISTING manifest must agree with the current
environment (of3_root / backbone / weight / timm / torch ...) before anything
is appended -- otherwise a "resume" would relabel stale-features provenance;
reused entries re-verify their features.csv sha on disk; every per-video
failure is recorded in the runner log before the run aborts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np

from src.eva_di.contracts import (
    CACHE_EQUIV_DIR,
    CACHE_MANIFEST_CSV_SHA_KEY,
    CACHE_MANIFEST_OF3_ROOT_KEY,
    EVA_BACKBONE,
    EVA_FEATURE_DIM,
    EVA_WEIGHT_REVISION,
    EVA_WEIGHT_SHA256,
    FEATURES_CSV_FILE,
    FEATURES_SUBDIR,
    FRAME_BUDGET,
    MAX_INVALID_IMAGE_RATIO,
    SCHEMA_FRAME_CACHE,
    SCHEMA_RUNNER_LOG,
    SELECTION_POLICY,
    EvaDiError,
    EvaDiFingerprintError,
    EvaDiPathError,
    EvaDiSchemaError,
)
from src.eva_di.dataset import load_split_ids
from src.eva_di.extract.cache_writer import (
    SelectionArrays,
    entry_matches,
    read_manifest,
    write_manifest,
    write_video_cache,
)
from src.eva_di.frame_selection import uniform_select
from src.eva_di.of3_registry import load_video_record
from src.eva_di.paths import PathSet, sha256_file

SMOKE_SAMPLE_COUNT = 10

# fields whose drift invalidates an existing cache for further appends
_STALE_GUARD_FIELDS = (
    "schema_version", "backbone", "weight_sha256", "weight_revision",
    "selection_policy", "frame_budget", "timm_version", "torch_version",
)


def _load_image_tensor(paths: list[str], root: Path, sample_id: str = "") -> "np.ndarray":  # noqa: F821
    """Decode jpgs to uint8 [N,3,112,112] via Pillow (provenance: cv2 not pinned).

    PIL stays the pinned decoder for bit-level reproducibility; cv2/
    torchvision/imageio ARE installed but libjpeg build differences would
    change decoded pixels and break the oracle-equivalence gate.  Swap =
    separate, disclosed package (audit R2 user decision: keep PIL).
    """
    from PIL import Image

    frames = []
    for rel in paths:
        path = Path(rel)
        if not path.is_absolute():
            path = root / path
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                if rgb.size != (112, 112):
                    raise EvaDiSchemaError(
                        f"{sample_id}: aligned face not 112x112: {path} size={rgb.size}")
                frames.append(np.asarray(rgb, dtype=np.uint8))
        except EvaDiError:
            raise
        except Exception as exc:
            raise EvaDiPathError(
                f"{sample_id}: cannot decode aligned face {path}: {exc}") from exc
    array = np.stack(frames)
    return array.transpose(0, 3, 1, 2)  # NHWC -> NCHW


def extract_video(paths: PathSet, sample_id: str, encoder, *,
                  verify_image_sha: bool = True) -> tuple[SelectionArrays, dict]:
    record = load_video_record(paths.of3_root, sample_id)
    valid_rows = np.flatnonzero(record.image_valid)
    if valid_rows.size == 0:
        raise EvaDiSchemaError(
            f"{sample_id}: no image_valid rows in features.csv; refusing an "
            "empty cache sample (np.stack would die without context)")
    selection = uniform_select(int(valid_rows.size))
    chosen_rows = valid_rows[selection]
    paths_of_chosen = [record.aligned_image_path[i] for i in chosen_rows]
    # Pinned OF3 v2 semantics (verified against real csv): relative
    # aligned_image_path values are of3_root-relative (``samples/<sid>/aligned/...``).
    if verify_image_sha:
        for row, rel in zip(chosen_rows, paths_of_chosen):
            path = Path(rel) if Path(rel).is_absolute() else paths.of3_root / rel
            if not path.is_file():
                raise EvaDiError(f"{sample_id}: aligned image missing: {path}")
            want = record.aligned_image_sha256[int(row)]
            if not want:
                # an image_valid row without a pinned sha silently bypasses the
                # byte-level verification the cache provenance claims (R2).
                raise EvaDiError(
                    f"{sample_id}: empty aligned_image_sha256 for image_valid "
                    f"row {int(row)} ({path})")
            got = sha256_file(path)
            if got != want:
                raise EvaDiError(f"{sample_id}: image sha mismatch for {path}")
    array = _load_image_tensor(paths_of_chosen, paths.of3_root, sample_id)
    frames_uint8 = _torch().from_numpy(array)
    n_before = int(frames_uint8.shape[0])
    # NOTE: deliberately NOT Tensor.any(): this pinned env (torch 2.8.0+cu129)
    # measures ``uint8.any(dim=1)`` -> uint8, and a uint8 "mask" silently
    # degrades every boolean index below into integer fancy-indexing (features
    # collapse onto one row -- caught by the E2E run).  A comparison is bool on
    # every build, so amax(...)>0 is the version-safe exact-zero guard.
    nonblack = frames_uint8.flatten(1).amax(dim=1) > 0  # [N] bool
    keep = nonblack.numpy()
    if keep.dtype != np.dtype(bool):  # belt-and-braces, never trust mask dtype
        keep = keep.astype(bool)
    invalid_ratio = 1.0 - float(keep.mean()) if n_before else 1.0
    if invalid_ratio > MAX_INVALID_IMAGE_RATIO:
        # guard BEFORE encoding: no GPU work is wasted on a refused video (R2).
        raise EvaDiSchemaError(
            f"{sample_id}: invalid/black frame ratio {invalid_ratio:.3f} exceeds "
            f"{MAX_INVALID_IMAGE_RATIO}"
        )
    cls, gap = encoder.encode(frames_uint8[nonblack]) if bool(nonblack.any()) else (
        _torch().zeros((0, EVA_FEATURE_DIM)), _torch().zeros((0, EVA_FEATURE_DIM))
    )
    # re-map kept (nonblack) features back onto the selection grid
    full_cls = np.zeros((n_before, EVA_FEATURE_DIM), dtype=np.float32)
    full_gap = np.zeros((n_before, EVA_FEATURE_DIM), dtype=np.float32)
    full_cls[keep] = cls.numpy()
    full_gap[keep] = gap.numpy()
    sel = SelectionArrays(
        cls=full_cls,
        gap=full_gap,
        frame_index_zero=record.frame_index_zero[chosen_rows].astype(np.int64),
        frame_valid=keep.astype(bool),
        meta={
            "sample_id": sample_id,
            "n_rows": record.n_rows,
            "n_image_valid": int(record.image_valid.sum()),
            "n_selected": int(chosen_rows.size),
            "n_exact_zero_masked": int((~keep).sum()),
            "selection_policy": SELECTION_POLICY,
            "features_sha256": record.features_sha256,
            "image_decode": "PIL",
        },
    )
    return sel, {"n_black": int((~keep).sum())}


def _torch():
    import torch

    return torch


def _assert_no_stale_drift(existing: dict, paths: PathSet, encoder_metadata: dict) -> None:
    """An existing manifest with samples must match the CURRENT environment,
    or appending would merge two feature generations under one provenance
    (audit R2 runner P1-1)."""
    if not existing.get("samples"):
        return
    if "schema_version" not in existing:
        raise EvaDiSchemaError(
            "existing cache manifest is not a valid v1 manifest (no "
            "schema_version); refusing to merge unknown-provenance samples")
    reference = {
        "schema_version": SCHEMA_FRAME_CACHE,
        "backbone": EVA_BACKBONE,
        "weight_sha256": EVA_WEIGHT_SHA256,
        "weight_revision": EVA_WEIGHT_REVISION,
        "selection_policy": SELECTION_POLICY,
        "frame_budget": FRAME_BUDGET,
        "timm_version": encoder_metadata.get("timm_version"),
        "torch_version": encoder_metadata.get("torch_version"),
    }
    diffs = [f"{f}: existing={existing.get(f)!r} current={reference[f]!r}"
             for f in _STALE_GUARD_FIELDS if existing.get(f) != reference[f]]
    import os

    want_root = os.path.realpath(str(paths.of3_root))
    got_root = os.path.realpath(str(existing.get(CACHE_MANIFEST_OF3_ROOT_KEY) or ""))
    if want_root != got_root:
        diffs.append(f"of3_root: existing={got_root!r} current={want_root!r}")
    if diffs:
        raise EvaDiFingerprintError(
            "existing cache was extracted under different provenance; refuse "
            "to mix generations: " + "; ".join(diffs))


def run_extraction(paths: PathSet, encoder_metadata: dict, encode_video,
                   *, smoke: bool = False, verify_image_sha: bool = True,
                   log_root: Path | None = None,
                   samples: list[str] | None = None) -> dict:
    """encode_video(paths, sample_id) -> (SelectionArrays, info): injected for testability.

    ``samples`` pins an explicit worklist (E2E/debug); it must be a subset of
    the split universe and overrides ``smoke``.  ``verify_image_sha`` is
    recorded in the manifest/log; the actual enforcement lives in
    ``extract_video`` via the caller's closure (main wires both from one flag).
    """
    split_ids = load_split_ids(paths.split_file)
    universe = set(split_ids["train"]) | set(split_ids["val"]) | set(split_ids["test"])
    if samples is not None:
        unknown = sorted(set(samples) - universe)
        if unknown or not samples:
            raise EvaDiSchemaError(
                f"explicit worklist must be a non-empty subset of the split universe; "
                f"unknown={unknown[:5]}")
        worklist = sorted(set(samples))
    else:
        worklist = sorted(universe)[:SMOKE_SAMPLE_COUNT] if smoke else sorted(universe)
    for sid in worklist:  # path hygiene: ids become directory names (R2-P3)
        if sid != Path(sid).name or sid in ("", ".", ".."):
            raise EvaDiSchemaError(f"sample id {sid!r} is not a safe directory name")
    jobs_sha256 = hashlib.sha256("\n".join(worklist).encode("utf-8")).hexdigest()
    paths.cache_root.mkdir(parents=True, exist_ok=True)
    (paths.cache_root / CACHE_EQUIV_DIR).mkdir(exist_ok=True)
    existing = read_manifest(paths.cache_root) or {"samples": {}}
    _assert_no_stale_drift(existing, paths, encoder_metadata)
    manifest_samples: dict = dict(existing.get("samples", {}))
    csv_sha_by_sample: dict[str, str] = dict(existing.get(CACHE_MANIFEST_CSV_SHA_KEY, {}))
    started = time.time()
    results = []
    failure: BaseException | None = None

    def _finalize_log() -> dict:
        duration = time.time() - started
        n_new = sum(int(r.get("n_frames", 0)) for r in results if r["status"] == "completed")
        log = {
            "schema": SCHEMA_RUNNER_LOG,
            "accepted": failure is None,
            "smoke": smoke,
            "verify_image_sha": bool(verify_image_sha),
            "jobs_sha256": jobs_sha256,
            "worklist": worklist,
            "device": encoder_metadata.get("device"),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "duration_seconds": round(duration, 3),
            "throughput_frames_per_s": round(n_new / duration, 3) if duration > 0 else None,
            "results": results,
        }
        if log_root is not None:
            root = Path(log_root)
            root.mkdir(parents=True, exist_ok=True)
            # unique name: two runs in the same second used to overwrite each
            # other, and a non-atomic write left truncated JSON (R2-P2).
            name = f"runner_{time.strftime('%Y%m%dT%H%M%S', time.gmtime(started))}_{uuid.uuid4().hex[:8]}.json"
            tmp = root / (name + ".tmp")
            tmp.write_text(json.dumps(log, ensure_ascii=False, indent=2, sort_keys=True),
                           encoding="utf-8")
            tmp.replace(root / name)
        return log

    for sample_id in worklist:
        try:
            resume_entry = manifest_samples.get(sample_id)
            directory = paths.cache_root / sample_id
            if (
                resume_entry is not None
                and directory.is_dir()
                and entry_matches(directory, resume_entry)
            ):
                # reuse ALSO requires the live features.csv sha to still match
                # the entry (audit R2 runner P1-2: csv drift under unchanged
                # npy hashes would keep stale-but-"reused" provenance).
                csv_path = paths.of3_root / FEATURES_SUBDIR / sample_id / FEATURES_CSV_FILE
                live_sha = sha256_file(csv_path) if csv_path.is_file() else ""
                entry_sha = str(resume_entry.get("features_sha256", "") or "")
                if entry_sha and live_sha and entry_sha != live_sha:
                    results.append({"sample_id": sample_id, "status": "stale_csv",
                                    "features_sha256": live_sha})
                else:
                    manifest_samples[sample_id] = dict(resume_entry)
                    if entry_sha:
                        csv_sha_by_sample[sample_id] = entry_sha
                    result = {"sample_id": sample_id, "status": "reused",
                              "n_frames": resume_entry.get("n_frames"),
                              "features_sha256": entry_sha}
                    if not entry_sha or not live_sha:
                        result["csv_sha_unverified"] = True
                    results.append(result)
                    continue
            video_started = time.time()
            sel, info = encode_video(paths, sample_id)
            entry, reused = write_video_cache(
                paths.cache_root, sample_id, sel, resume_entry=None
            )
            entry["n_rows"] = int(sel.meta["n_rows"])
            manifest_samples[sample_id] = entry
            csv_sha_by_sample[sample_id] = str(sel.meta["features_sha256"])
            results.append({"sample_id": sample_id,
                            "status": "reused" if reused else "completed",
                            "n_frames": entry["n_frames"],
                            "features_sha256": csv_sha_by_sample[sample_id],
                            "seconds": round(time.time() - video_started, 3),
                            **info})  # runner-owned keys precede, info cannot override
            # manifest refreshed every video: a kill at any point leaves a
            # self-consistent, resumable state.
            manifest = _build_manifest(paths, encoder_metadata, manifest_samples,
                                       csv_sha_by_sample)
            write_manifest(paths.cache_root, manifest)
        except Exception as exc:  # fail fast, but leave the evidence (R2-P2)
            results.append({"sample_id": sample_id, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}"})
            failure = exc
            break
    log = _finalize_log()
    if failure is not None:
        raise failure
    return log


def _entry_int(entry: dict, key: str) -> int:
    try:
        return int(entry[key])
    except KeyError as exc:
        raise EvaDiSchemaError(f"cache manifest entry missing {key!r}: {sorted(entry)}") from exc


def _build_manifest(paths: PathSet, encoder_metadata: dict, samples: dict,
                    csv_sha_by_sample: dict[str, str]) -> dict:
    from src.eva_di.contracts import (
        EVA_INPUT_SIZE,
    )

    n_selected = sum(_entry_int(e, "n_frames") for e in samples.values())
    n_valid = sum(_entry_int(e, "n_valid") for e in samples.values())
    return {
        "schema_version": SCHEMA_FRAME_CACHE,
        "selection_policy": SELECTION_POLICY,
        "frame_budget": FRAME_BUDGET,
        "backbone": EVA_BACKBONE,
        "weight_sha256": EVA_WEIGHT_SHA256,
        "weight_revision": EVA_WEIGHT_REVISION,
        "timm_version": encoder_metadata.get("timm_version"),
        "torch_version": encoder_metadata.get("torch_version"),
        "cuda_version": encoder_metadata.get("cuda_version"),
        "input_size": EVA_INPUT_SIZE,
        "feature_dim": EVA_FEATURE_DIM,
        "feature_dtype": "float32",
        "valid_dtype": "bool",
        # runtime provenance of THIS extraction pass (audit R2 runner P2-5):
        "autocast": encoder_metadata.get("autocast"),
        "chunk": encoder_metadata.get("chunk"),
        "device": encoder_metadata.get("device"),
        "verify_image_sha": encoder_metadata.get("verify_image_sha"),
        "config_source_sha256": encoder_metadata.get("config_source_sha256"),
        CACHE_MANIFEST_OF3_ROOT_KEY: str(paths.of3_root),
        CACHE_MANIFEST_CSV_SHA_KEY: {sid: csv_sha_by_sample.get(sid, "") for sid in samples},
        "sample_count": len(samples),
        "n_rows_total": sum(_entry_int(e, "n_rows") for e in samples.values()),
        "n_selected_total": n_selected,
        "n_valid_total": n_valid,
        "n_exact_zero_masked": n_selected - n_valid,
        "samples": samples,
    }


def main(argv=None) -> int:  # pragma: no cover - extraction runtime (EXTRACT package)
    from src.eva_di.extract.encoder import Eva02FrozenEncoder

    parser = argparse.ArgumentParser(description="Build the eva_di frozen frame feature cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="first 10 videos only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-verify-image-sha", dest="verify_image_sha", action="store_false")
    args = parser.parse_args(argv)
    _require_light_env()
    from src.eva_di.config import load_config

    cfg = load_config(args.config)
    if cfg.run.mode != "extract":
        raise EvaDiError(f"extraction requires run.mode='extract', got {cfg.run.mode!r}")
    paths = PathSet.build(overrides=cfg.paths.overrides).resolve_read_only(mode="extract")
    encoder = Eva02FrozenEncoder(
        paths.weight_path, device=args.device,
        chunk=cfg.extraction.chunk, autocast=cfg.extraction.autocast,
    )
    meta = encoder.metadata()
    meta["device"] = args.device
    meta["verify_image_sha"] = bool(args.verify_image_sha)
    meta["config_source_sha256"] = cfg.source_sha256

    def encode_video(path_set: PathSet, sample_id: str):
        sel, info = extract_video(path_set, sample_id, encoder, verify_image_sha=args.verify_image_sha)
        return sel, info

    log = run_extraction(paths, meta, encode_video, smoke=args.smoke,
                         verify_image_sha=args.verify_image_sha, log_root=paths.log_root)
    print(json.dumps({k: v for k, v in log.items() if k != "results"}, indent=2))
    return 0


def _require_light_env() -> None:
    import sys

    if "envs/light/" not in sys.executable.replace("\\", "/"):
        raise EvaDiError(
            f"extraction must run in conda env 'light'; got interpreter {sys.executable}"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
