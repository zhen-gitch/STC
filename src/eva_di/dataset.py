"""Four-source recording assembly with cross-source assertions.

Sources: ``dataset_split.json`` (the only split truth), ``depression_labels``
(mirrors the existing AVEC contract at ``src/datasets/dataset.py:276-280``),
the frozen EVA02 frame cache, and per-video OpenFace3 ``features.csv``.

Prefix-run rule: cached frames carry the image-valid mask; behavior validity
can punch interior holes.  The GRU-without-pack precondition (mechanism doc
section 2) demands a True-prefix frame mask, so a Recording keeps the leading
contiguous run of frames valid in *both* streams (legacy STC adapter precedent:
longest-contiguous-run).  Dropped tails are counted as diagnostics, never
silently re-used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.eva_di.behavior import BehaviorStats
from src.eva_di.cache_reader import FrozenFrameCache
from src.eva_di.contracts import (
    CSV_SPLIT_ALIASES,
    LABEL_SUFFIX,
    SPLIT_ID_SUFFIX,
    SPLIT_KEYS,
    EvaDiFingerprintError,
    EvaDiPathError,
    EvaDiSchemaError,
)
from src.eva_di.of3_registry import VideoRecord, load_video_record
from src.eva_di.paths import PathSet
from src.eva_di.subject_table import SubjectTable, build_subject_table, subject_of


@dataclass(frozen=True)
class Recording:
    sample_id: str
    split: str
    subject: int  # train-only table id; -1 outside train mapping
    bdi_score: float  # raw 0..63
    bdi_norm: float  # (bdi - target_mean)/target_std (train-only)
    cls: np.ndarray  # [n, 384] float32
    gap: np.ndarray  # [n, 384] float32
    behavior_norm: np.ndarray  # [n, 206] float32 (masked-invalid rows are zero)
    frame_mask: np.ndarray  # [n] bool, all True by construction (prefix run)
    n_selected: int  # frames in cache for this video
    n_dropped_tail: int
    n_behavior_invalid: int


@dataclass
class TargetStats:
    mean: float
    std: float


@dataclass
class RecordingIndex:
    recordings: dict[str, Recording]
    splits: dict[str, list[str]]  # split -> sample_ids (sorted)
    subject_table: SubjectTable
    target_stats: TargetStats
    diagnostics: dict = field(default_factory=dict)

    def by_split(self, split: str) -> list[Recording]:
        if split not in SPLIT_KEYS:
            raise EvaDiSchemaError(f"unknown split {split!r}")
        if split not in self.splits:
            raise EvaDiSchemaError(
                f"split {split!r} was never loaded; recordings only exist for the "
                "decision splits and the test split stays locked (design section 3)"
            )
        return [self.recordings[s] for s in self.splits[split]]


def load_split_ids(split_file: Path) -> dict[str, list[str]]:
    payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not set(SPLIT_KEYS) <= set(payload):
        raise EvaDiSchemaError(f"split file must contain keys {SPLIT_KEYS}: {split_file}")
    out: dict[str, list[str]] = {}
    for split in SPLIT_KEYS:
        raw = payload[split]
        if not isinstance(raw, list) or not raw:
            raise EvaDiSchemaError(f"split {split!r} must be a non-empty list")
        ids = []
        for entry in raw:
            text = str(entry)
            if not text.endswith(SPLIT_ID_SUFFIX):
                raise EvaDiSchemaError(
                    f"split id {text!r} lacks {SPLIT_ID_SUFFIX!r} suffix"
                )
            ids.append(text[: -len(SPLIT_ID_SUFFIX)])
        if len(set(ids)) != len(ids):
            raise EvaDiSchemaError(f"duplicate ids in split {split!r}")
        out[split] = sorted(ids)
    seen: set[str] = set()
    for split in SPLIT_KEYS:
        overlap = seen & set(out[split])
        if overlap:
            raise EvaDiSchemaError(f"cross-split overlap: {sorted(overlap)[:5]}")
        seen |= set(out[split])
    return out


def load_bdi_score(label_dir: Path, sample_id: str) -> float:
    """Contract mirror of src/datasets/dataset.py:276-280 (single-int file)."""
    path = Path(label_dir) / f"{subject_of(sample_id)}{LABEL_SUFFIX}"
    if not path.is_file():
        raise EvaDiPathError(f"BDI label missing for {sample_id}: {path}")
    try:
        return float(int(path.read_text(encoding="utf-8").strip()))
    except ValueError as exc:
        raise EvaDiSchemaError(f"BDI label not a single int: {path}") from exc


def _prefix_run(valid: np.ndarray) -> int:
    index = np.flatnonzero(~valid)
    return int(index[0]) if index.size else int(valid.size)


def build_recording_index(
    paths: PathSet,
    *,
    splits: tuple[str, ...],
    behavior_stats: BehaviorStats,
    max_recordings: int | None = None,
    cache: FrozenFrameCache | None = None,
    verify_hashes: bool = True,
) -> RecordingIndex:
    split_ids = load_split_ids(paths.split_file)
    wanted = {s: list(split_ids[s]) for s in splits}
    if max_recordings is not None:
        # train-ONLY subsampling: the val decision set stays complete so
        # val_ccc / best-model selection remain comparable across runs
        # (audit R2-P1-3; train.py's leak guard always reads the FULL train
        # split through the registry, independent of this truncation).
        wanted = {s: (ids[:max_recordings] if s == "train" else ids)
                  for s, ids in wanted.items()}
    all_ids = sorted({sample_id for ids in wanted.values() for sample_id in ids})
    if "train" not in splits:
        raise EvaDiSchemaError("recording index always needs the train split "
                               "(subject table + target/behavior stats are train-only)")

    # --- cross-source video set equality (train ∪ val ∪ test) ---------------
    from src.eva_di.of3_registry import list_sample_ids

    registry_ids = set(list_sample_ids(paths.of3_root))
    split_universe = set(split_ids["train"]) | set(split_ids["val"]) | set(split_ids["test"])
    missing_in_registry = sorted(split_universe - registry_ids)
    extra_in_registry = sorted(registry_ids - split_universe)
    if missing_in_registry or extra_in_registry:
        raise EvaDiSchemaError(
            "split vs OpenFace3 registry mismatch: "
            f"missing_in_registry={missing_in_registry[:5]} "
            f"extra_in_registry={extra_in_registry[:5]}"
        )

    cache_obj = cache or FrozenFrameCache(paths.cache_root, verify_hashes=verify_hashes,
                                          expected_of3_root=paths.of3_root)
    cache_ids = set(cache_obj.sample_ids())
    wanted_missing_cache = sorted(set(all_ids) - cache_ids)
    if wanted_missing_cache:
        raise EvaDiSchemaError(
            f"cache lacks {len(wanted_missing_cache)} requested videos, e.g. "
            f"{wanted_missing_cache[:5]}"
        )

    subject_table = build_subject_table(split_ids["train"])
    diagnostics = {"behavior_invalid_frames": 0, "dropped_tail_frames": 0}

    # --- target stats from train split only ---------------------------------
    train_raw = [load_bdi_score(paths.label_dir, s) for s in split_ids["train"]]
    target_stats = TargetStats(mean=float(np.mean(train_raw)), std=float(np.std(train_raw)))
    if target_stats.std <= 0:
        raise EvaDiSchemaError("degenerate train BDI std")

    split_of = {sample_id: s for s, ids in wanted.items() for sample_id in ids}
    recordings: dict[str, Recording] = {}
    for sample_id in all_ids:
        record: VideoRecord = load_video_record(
            paths.of3_root, sample_id, verify_sha=verify_hashes
        )
        split_name = split_of[sample_id]
        # the per-video csv split column must agree with the json split
        # (modulo the pinned OF3 "dev" dialect, contracts.CSV_SPLIT_ALIASES);
        # this catches a train/val mislabel no other guard sees (audit R2-P1).
        if record.split not in CSV_SPLIT_ALIASES[split_name]:
            raise EvaDiSchemaError(
                f"{sample_id}: features.csv split column {record.split!r} does "
                f"not match dataset_split.json split {split_name!r}")
        cached = cache_obj.load_sample(sample_id)
        if verify_hashes:
            # cache-pinned features.csv sha vs the csv we just parsed: two
            # files, one truth (audit R2 data-layer cross-file finding).
            want_sha = cache_obj.csv_sha256(sample_id)
            if want_sha and record.features_sha256 and want_sha != record.features_sha256:
                raise EvaDiFingerprintError(
                    f"{sample_id}: features.csv sha {record.features_sha256} "
                    f"!= cache manifest pin {want_sha}; the OF3 source moved "
                    "under the frozen cache (re-extract is a separate package)")
        # cached frames come from image-valid rows; map behavior by frame index
        row_of_index = {int(f): i for i, f in enumerate(record.frame_index_zero)}
        missing = [int(f) for f in cached.frame_index_zero if int(f) not in row_of_index]
        if missing:
            raise EvaDiSchemaError(
                f"{sample_id}: cache frame indices absent from features.csv: {missing[:5]}"
            )
        rows = np.asarray([row_of_index[int(f)] for f in cached.frame_index_zero])
        behavior_valid = record.behavior_valid[rows]
        combined_valid = cached.frame_valid & behavior_valid
        n_keep = _prefix_run(combined_valid)
        if n_keep == 0:
            raise EvaDiSchemaError(f"{sample_id}: zero frames valid in both streams")
        diagnostics["behavior_invalid_frames"] += int((~behavior_valid).sum())
        diagnostics["dropped_tail_frames"] += int(combined_valid.size - n_keep)
        behavior_norm = behavior_stats.normalize(record.behavior[rows], combined_valid)
        bdi_score = load_bdi_score(paths.label_dir, sample_id)
        recordings[sample_id] = Recording(
            sample_id=sample_id,
            split=split_name,
            subject=subject_table(sample_id) if split_name == "train" else -1,
            bdi_score=bdi_score,
            bdi_norm=(bdi_score - target_stats.mean) / target_stats.std,
            cls=cached.cls[:n_keep],
            gap=cached.gap[:n_keep],
            behavior_norm=behavior_norm[:n_keep],
            frame_mask=np.ones(n_keep, dtype=bool),
            n_selected=int(cached.frame_index_zero.size),
            n_dropped_tail=int(combined_valid.size - n_keep),
            n_behavior_invalid=int((~behavior_valid).sum()),
        )
    return RecordingIndex(
        recordings=recordings,
        splits={s: sorted(ids) for s, ids in wanted.items()},
        subject_table=subject_table,
        target_stats=target_stats,
        diagnostics=diagnostics,
    )
