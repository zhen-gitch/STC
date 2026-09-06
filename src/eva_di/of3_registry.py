"""Strict per-video reader for OpenFace3 formal ``features.csv`` files.

Only-read asset root.  Violations of the pinned 42-column schema, axis parsing,
or monotone frame indices raise :class:`EvaDiSchemaError` with column name and
first offending row.  Non-finite behavior values are *masked*, never imputed:
they become ``behavior_valid=False`` rows with zeros in the array, and masked
rows must never enter normalization statistics, CE targets, or regression
inputs (enforced by consumers via the returned mask).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.eva_di.contracts import (
    BEHAVIOR_DIM,
    BEHAVIOR_LANDMARKS,
    BEHAVIOR_LANDMARK_CANVAS,
    FEATURES_CSV_COLUMNS,
    FEATURES_CSV_FILE,
    FEATURES_SUBDIR,
    SCHEMA_ALIGNED_MASK,
    SPLIT_KEYS,
    EvaDiPathError,
    EvaDiSchemaError,
    parse_bool,
)
from src.eva_di.paths import sha256_file

_SPLIT_VALUES = SPLIT_KEYS + ("dev",)  # csv split column is assertion-only
_FRAME_INDEX_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class VideoRecord:
    sample_id: str
    split: str  # from csv, assertion-only; dataset_split.json is the source of truth
    n_rows: int
    frame_index_zero: np.ndarray  # [N] int64, strictly increasing (all rows)
    image_valid: np.ndarray  # [N] bool
    behavior: np.ndarray  # [N, 206] float32; non-finite entries zero-filled,
    #                       whole-row masking via behavior_valid (never imputed)
    behavior_valid: np.ndarray  # [N] bool
    pair_valid: np.ndarray  # [N] bool
    aligned_image_path: tuple[str, ...]  # raw column (relative or absolute per row)
    aligned_image_sha256: tuple[str, ...]
    features_sha256: str  # sha256 of features.csv itself


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_frame_index(value: str, sample_id: str, row_index: int) -> int:
    """Strict non-negative integer parse.

    ``int()`` alone accepts ``"+5"``, ``"5_0"`` and surrounding whitespace --
    a silent frame-index mutation corrupts the cache->csv mapping, so the
    pinned dialect is plain digits only (audit R2 registry P2).
    """
    text = str(value)
    if not _FRAME_INDEX_RE.fullmatch(text):
        raise EvaDiSchemaError(
            f"{sample_id}: frame_index_zero {value!r} is not plain digits at row {row_index}")
    return int(text)


def _parse_behavior_row(row: dict[str, str], row_index: int, sample_id: str) -> np.ndarray:
    """Return the 206-d vector or raise; caller masks non-finite outcomes.

    Landmark axes come back in the pinned ``*_norm`` space
    (``2*v/111 - 1`` over the 111-px aligned canvas -- float32, same op order
    as the generator); gaze/au stay raw.  Mixing spaces breaks every behavior
    feature (caught by the E2E run: raw pixels vs pinned normalized stats).
    """
    landmarks_raw = row["landmarks_98_xy_aligned"]
    try:
        landmarks = json.loads(landmarks_raw)
    except json.JSONDecodeError as exc:
        raise EvaDiSchemaError(
            f"{sample_id}: landmarks_98_xy_aligned not valid JSON at row {row_index}"
        ) from exc
    if not isinstance(landmarks, list) or len(landmarks) != 98:
        raise EvaDiSchemaError(
            f"{sample_id}: landmarks_98_xy_aligned must hold 98 points "
            f"(got {len(landmarks) if isinstance(landmarks, list) else type(landmarks).__name__}) "
            f"at row {row_index}"
        )
    vector = np.full(BEHAVIOR_DIM, np.nan, dtype=np.float32)
    cursor = 0
    for point in landmarks:
        if not isinstance(point, list) or len(point) != 2:
            raise EvaDiSchemaError(
                f"{sample_id}: landmark point must be [x, y] at row {row_index}"
            )
        vector[cursor] = _parse_float(point[0])
        vector[cursor + 1] = _parse_float(point[1])
        cursor += 2
    for column in ("gaze_angle_x", "gaze_angle_y", *(f"au_score_{i}" for i in range(8))):
        vector[cursor] = _parse_float(row[column])
        cursor += 1
    if cursor != BEHAVIOR_DIM:
        raise EvaDiSchemaError(f"{sample_id}: axis drift: cursor={cursor} at row {row_index}")
    vector[:BEHAVIOR_LANDMARKS] = 2.0 * vector[:BEHAVIOR_LANDMARKS] / BEHAVIOR_LANDMARK_CANVAS - 1.0
    return vector


def load_video_record(
    of3_root: Path, sample_id: str, *, verify_sha: bool = True
) -> VideoRecord:
    csv_path = Path(of3_root) / FEATURES_SUBDIR / sample_id / FEATURES_CSV_FILE
    if not csv_path.is_file():
        raise EvaDiPathError(f"features.csv missing for {sample_id}: {csv_path}")

    frame_indices: list[int] = []
    image_valid: list[bool] = []
    behavior_valid: list[bool] = []
    pair_valid: list[bool] = []
    behaviors: list[np.ndarray] = []
    image_paths: list[str] = []
    image_shas: list[str] = []
    split_values: set[str] = set()
    previous_index = -1

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != FEATURES_CSV_COLUMNS:
            missing = [c for c in FEATURES_CSV_COLUMNS if c not in fieldnames]
            extra = [c for c in fieldnames if c not in FEATURES_CSV_COLUMNS]
            reordered = (not missing) and (not extra) and fieldnames != FEATURES_CSV_COLUMNS
            kind = "column order changed" if reordered else f"missing={missing} extra={extra}"
            raise EvaDiSchemaError(f"{sample_id}: features.csv schema violation: {kind}")
        for row_index, row in enumerate(reader):
            # column-count drift: long rows spill into the restkey (None)
            # slot, short rows leave None cells -- both would otherwise be
            # read as data downstream (audit R2 registry P2).
            if row.get(None) is not None or any(v is None for v in row.values()):
                raise EvaDiSchemaError(
                    f"{sample_id}: row {row_index} does not match the 42-column "
                    f"header width (restkey={row.get(None)!r})")
            if row["sample_id"] != sample_id:
                raise EvaDiSchemaError(
                    f"{sample_id}: row sample_id {row['sample_id']!r} at row {row_index}"
                )
            if row["split"] not in _SPLIT_VALUES:
                raise EvaDiSchemaError(
                    f"{sample_id}: unknown split {row['split']!r} at row {row_index}"
                )
            split_values.add(row["split"])
            frame_index = _parse_frame_index(row["frame_index_zero"], sample_id, row_index)
            if frame_index <= previous_index:
                raise EvaDiSchemaError(
                    f"{sample_id}: frame_index_zero {frame_index} not strictly increasing "
                    f"(previous {previous_index}) at row {row_index}"
                )
            previous_index = frame_index
            image_ok = parse_bool(row["image_valid"], field="image_valid", row=row_index)
            if not image_ok and row["aligned_mask_schema"] == "":
                # OF3 v2 reality (measured): alignment-failed / quality-rejected
                # rows carry NO aligned artifacts (empty schema, path, sha; the
                # behavior columns are empty or untrustworthy).  They are masked
                # rows, not schema violations; any *partial* artifact state still
                # fails below.
                frame_indices.append(frame_index)
                image_valid.append(False)
                behavior_valid.append(False)
                pair_valid.append(False)
                behaviors.append(np.zeros(BEHAVIOR_DIM, dtype=np.float32))
                image_paths.append(row["aligned_image_path"])
                image_shas.append(row["aligned_image_sha256"])
                continue
            if row["aligned_mask_schema"] != SCHEMA_ALIGNED_MASK:
                raise EvaDiSchemaError(
                    f"{sample_id}: aligned_mask_schema {row['aligned_mask_schema']!r} "
                    f"!= {SCHEMA_ALIGNED_MASK!r} at row {row_index}"
                )
            if image_ok and not row["aligned_image_path"].strip():
                # an image_valid row must carry a decodable artifact; empty
                # paths crash the extractor with a context-free OSError (R2).
                raise EvaDiSchemaError(
                    f"{sample_id}: image_valid=True row {row_index} has an empty "
                    "aligned_image_path")
            vector = _parse_behavior_row(row, row_index, sample_id)
            finite = bool(np.isfinite(vector).all())
            declared = parse_bool(row["behavior_valid"], field="behavior_valid", row=row_index)
            pair_ok = parse_bool(row["pair_valid"], field="pair_valid", row=row_index)
            frame_indices.append(frame_index)
            image_valid.append(image_ok)
            behavior_valid.append(declared and finite)
            pair_valid.append(pair_ok)
            behaviors.append(np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0))
            image_paths.append(row["aligned_image_path"])
            image_shas.append(row["aligned_image_sha256"])

    if not frame_indices:
        raise EvaDiSchemaError(f"{sample_id}: features.csv has no data rows")
    if len(split_values) != 1:
        raise EvaDiSchemaError(f"{sample_id}: csv split column not constant: {split_values}")
    behavior_matrix = np.stack(behaviors).astype(np.float32)
    record = VideoRecord(
        sample_id=sample_id,
        split=split_values.pop(),
        n_rows=len(frame_indices),
        frame_index_zero=np.asarray(frame_indices, dtype=np.int64),
        image_valid=np.asarray(image_valid, dtype=bool),
        behavior=behavior_matrix,
        behavior_valid=np.asarray(behavior_valid, dtype=bool),
        pair_valid=np.asarray(pair_valid, dtype=bool),
        aligned_image_path=tuple(image_paths),
        aligned_image_sha256=tuple(image_shas),
        features_sha256=sha256_file(csv_path) if verify_sha else "",
    )
    return record


def list_sample_ids(of3_root: Path) -> tuple[str, ...]:
    root = Path(of3_root) / FEATURES_SUBDIR
    if not root.is_dir():
        raise EvaDiPathError(f"OpenFace3 samples root missing: {root}")
    ids = sorted(p.name for p in root.iterdir() if (p / FEATURES_CSV_FILE).is_file())
    if not ids:
        raise EvaDiSchemaError(f"no sample directories with features.csv under {root}")
    return tuple(ids)
