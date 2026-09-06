"""Synthetic fixtures for eva_di unit tests -- NO real data, NO GPU.

Builds a miniature but schema-faithful tree:
  of3_root/samples/<id>/features.csv   (pinned 42 columns)
  avec_root/dataset_split.json         (train/val/test with _aligned suffix)
  avec_root/depression_labels/*.csv    (single int)
  norm_root/technical.json             (pinned stats shape, train-only values)
  cache_root/<id>/*.npy + manifest     (via the real writer)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.eva_di.contracts import (
    BEHAVIOR_NORM_POLICY,
    BEHAVIOR_NORM_SCHEMA,
    CACHE_MANIFEST_FILE,
    EVA_BACKBONE,
    EVA_FEATURE_DIM,
    EVA_INPUT_SIZE,
    EVA_WEIGHT_REVISION,
    EVA_WEIGHT_SHA256,
    FEATURES_CSV_COLUMNS,
    FRAME_BUDGET,
    SCHEMA_ALIGNED_MASK,
    SCHEMA_BEHAVIOR,
    SCHEMA_FRAME_CACHE,
    SELECTION_POLICY,
    SPLIT_ID_SUFFIX,
    expected_behavior_axis_names,
)
from src.eva_di.extract.cache_writer import SelectionArrays, write_video_cache
from src.eva_di.paths import sha256_file

RNG = np.random.default_rng(20260904)


def _behavior_vector(rng, scale=1.0) -> tuple[str, float, float, list[float]]:
    landmarks = [[round(float(rng.normal(56, 20 * scale)), 6),
                  round(float(rng.normal(56, 20 * scale)), 6)] for _ in range(98)]
    gaze = (round(float(rng.normal(0, 0.3)), 6), round(float(rng.normal(0, 0.3)), 6))
    aus = [round(float(rng.uniform(0, 1)), 6) for _ in range(8)]
    return json.dumps(landmarks), gaze[0], gaze[1], aus


def write_features_csv(video_dir: Path, sample_id: str, split: str, *, n_frames: int,
                       rng, bad_landmark_rows: frozenset = frozenset(),
                       bad_au_rows: frozenset = frozenset(),
                       invalid_image_rows: frozenset = frozenset(),
                       blank_schema_rows: frozenset = frozenset(),
                       nonmonotone: bool = False) -> Path:
    import csv as _csv

    video_dir.mkdir(parents=True, exist_ok=True)
    path = video_dir / "features.csv"
    lines: list[list[str]] = [list(FEATURES_CSV_COLUMNS)]
    frame_index = 0
    for row in range(n_frames):
        landmarks, gx, gy, aus = _behavior_vector(rng)
        if row in bad_landmark_rows:
            # structurally valid 98 points with NaN values -> masking path (not schema error)
            landmarks = json.dumps([[float("nan"), float("nan")]] +
                                   [[0.5 * i, 0.25 * i] for i in range(97)])
        au_values = list(aus)
        if row in bad_au_rows:
            au_values[3] = "nan"
        blank = row in blank_schema_rows  # OF3 v2 alignment-failed row: no aligned artifacts
        image_valid = (row not in invalid_image_rows) and not blank
        behavior_ok = (row not in bad_landmark_rows) and (row not in bad_au_rows) and not blank
        index = frame_index - 2 if (nonmonotone and row == n_frames - 1 and n_frames >= 3) else frame_index
        values = {
            "sample_id": sample_id, "split": split, "task_id": "freeform",
            "frame_index_zero": str(index), "frame_index_one": str(index + 1),
            "timestamp": f"{index / 25:.3f}", "source_video": f"{split}/x.mp4",
            "source_video_sha256": "a" * 64, "frame_quality_sha256": "b" * 64,
            "decoded": "True", "detection_count": "1", "confidence": "0.9",
            "quality_status": "ok", "quality_geometry_status": "ok",
            "bbox_xyxy": "[0,0,100,100]", "retinaface_5pt_xy_original": "[]",
            "technical_valid": "True", "alignment_matrix_2x3": "[]",
            "landmarks_98_xy_original": "[]",
            "landmarks_98_xy_aligned": "" if blank else landmarks,
            "aligned_image_path": "" if blank else f"samples/{sample_id}/aligned/frame_{index:06d}.jpg",
            "aligned_image_sha256": "" if blank else f"{row:064x}",
            "aligned_mask_schema": "" if blank else SCHEMA_ALIGNED_MASK,
            "gaze_angle_x": str(gx), "gaze_angle_y": str(gy),
            **{f"au_score_{i}": str(v) for i, v in enumerate(au_values)},
            "emotion_index": "2",
            "alignment_status": "failed" if blank else "ok",
            "landmark_status": "not_run_quality_invalid" if blank else "ok",
            "behavior_status": "not_run_alignment_failed" if blank else ("ok" if behavior_ok else "failed"),
            "image_valid": str(image_valid), "behavior_valid": str(behavior_ok),
            "pair_valid": str(image_valid and behavior_ok),
            "inference_seconds": "0.1", "issues": "",
        }
        lines.append([values[column] for column in FEATURES_CSV_COLUMNS])
        frame_index += 2
    with path.open("w", newline="", encoding="utf-8") as handle:
        _csv.writer(handle).writerows(lines)
    return path


def _behavior_matrix_from_csv(video_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(behavior[N,206] with masked rows zero, behavior_valid[N], frame_index[N]) from fixture."""
    import csv as _csv

    rows = list(_csv.DictReader(video_csv.open(encoding="utf-8")))
    behaviors, valids, indices = [], [], []
    for row in rows:
        try:
            landmarks = json.loads(row["landmarks_98_xy_aligned"])
        except json.JSONDecodeError:
            landmarks = None
        if not isinstance(landmarks, list) or len(landmarks) != 98:
            behaviors.append(np.zeros(206, dtype=np.float32))
            valids.append(False)
            indices.append(int(row["frame_index_zero"]))
            continue
        vector = np.asarray([v for point in landmarks for v in point] +
                            [float(row["gaze_angle_x"]), float(row["gaze_angle_y"])] +
                            [float(row[f"au_score_{i}"]) for i in range(8)], dtype=np.float32)
        # same pinned landmark space as src.eva_di.of3_registry (2*v/111 - 1)
        vector[:196] = 2.0 * vector[:196] / 111.0 - 1.0
        finite = bool(np.isfinite(vector).all())
        behaviors.append(np.nan_to_num(vector, nan=0.0) if not finite else vector)
        valids.append(bool(row["behavior_valid"] == "True" and finite))
        indices.append(int(row["frame_index_zero"]))
    return np.stack(behaviors), np.asarray(valids, dtype=bool), np.asarray(indices, dtype=np.int64)


def write_norm_json(norm_root: Path, train_csvs: list[Path]) -> Path:
    mats, valids = [], []
    for csv_path in train_csvs:
        matrix, valid, _ = _behavior_matrix_from_csv(csv_path)
        mats.append(matrix)
        valids.append(valid)
    stacked = np.concatenate([m[v] for m, v in zip(mats, valids)], axis=0)
    mean = stacked.mean(axis=0)
    std = np.maximum(stacked.std(axis=0), 1e-6)
    payload = {
        "behavior_schema": SCHEMA_BEHAVIOR,
        "normalization_schema": BEHAVIOR_NORM_SCHEMA,
        "policy": BEHAVIOR_NORM_POLICY,
        "split": "train",
        "feature_names": list(expected_behavior_axis_names()),
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "count": [int(stacked.shape[0])] * 206,
        "valid_frame_count": int(stacked.shape[0]),
        "std_floor": 1e-06,
    }
    norm_root.mkdir(parents=True, exist_ok=True)
    path = norm_root / "technical.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_split_json(path: Path, splits: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {split: [f"{sample}{SPLIT_ID_SUFFIX}" for sample in ids]
               for split, ids in splits.items()}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def write_labels(label_dir: Path, subject_scores: dict[str, int]) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    for subject, score in subject_scores.items():
        (label_dir / f"{subject}_Depression.csv").write_text(f"{score}\n", encoding="utf-8")


def manifest_skeleton() -> dict:
    return {
        "schema_version": SCHEMA_FRAME_CACHE,
        "selection_policy": SELECTION_POLICY,
        "frame_budget": FRAME_BUDGET,
        "backbone": EVA_BACKBONE,
        "weight_sha256": EVA_WEIGHT_SHA256,
        "weight_revision": EVA_WEIGHT_REVISION,
        "timm_version": "test",
        "torch_version": "test",
        "cuda_version": None,
        "input_size": EVA_INPUT_SIZE,
        "feature_dim": EVA_FEATURE_DIM,
        "feature_dtype": "float32",
        "valid_dtype": "bool",
        "of3_root": "synthetic-of3-root",
        "features_csv_sha256_by_sample": {},
        "sample_count": 0,
        "n_rows_total": 0,
        "n_selected_total": 0,
        "n_valid_total": 0,
        "n_exact_zero_masked": 0,
        "samples": {},
    }


def build_cache_for_tree(cache_root: Path, of3_root: Path, sample_ids: list[str], *,
                         rng, manifest_path: Path | None = None) -> dict:
    """Fake frozen cache aligned with the fixture csvs (all image-valid rows)."""
    import csv as _csv

    manifest = manifest_skeleton()
    cache_root.mkdir(parents=True, exist_ok=True)
    for sample_id in sample_ids:
        csv_path = of3_root / "samples" / sample_id / "features.csv"
        rows = list(_csv.DictReader(csv_path.open(encoding="utf-8")))
        indices = [int(r["frame_index_zero"]) for r in rows if r["image_valid"] == "True"]
        n = len(indices)
        sel = SelectionArrays(
            cls=rng.normal(0, 1, (n, EVA_FEATURE_DIM)).astype(np.float32),
            gap=rng.normal(0, 1, (n, EVA_FEATURE_DIM)).astype(np.float32),
            frame_index_zero=np.asarray(indices, dtype=np.int64),
            frame_valid=np.ones(n, dtype=bool),
            meta={"sample_id": sample_id, "n_rows": len(rows),
                  "n_image_valid": n, "n_selected": n, "n_exact_zero_masked": 0,
                  "selection_policy": SELECTION_POLICY,
                  "features_sha256": sha256_file(csv_path),
                  "image_decode": "synthetic"},
        )
        entry, _ = write_video_cache(cache_root, sample_id, sel)
        entry["n_rows"] = len(rows)
        manifest["samples"][sample_id] = entry
    manifest["sample_count"] = len(manifest["samples"])
    manifest["of3_root"] = str(of3_root)  # real root so reader pins are exercised
    manifest["features_csv_sha256_by_sample"] = {
        sid: str(e.get("features_sha256", "")) for sid, e in manifest["samples"].items()
    }
    manifest["n_rows_total"] = sum(int(e["n_rows"]) for e in manifest["samples"].values())
    manifest["n_selected_total"] = sum(int(e["n_frames"]) for e in manifest["samples"].values())
    manifest["n_valid_total"] = sum(int(e["n_valid"]) for e in manifest["samples"].values())
    (cache_root / CACHE_MANIFEST_FILE).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


@dataclass(frozen=True)
class SynthTree:
    root: Path
    of3_root: Path
    avec_root: Path
    split_file: Path
    label_dir: Path
    norm_root: Path
    cache_root: Path
    weight_path: Path
    splits: dict[str, list[str]]
    subject_scores: dict[str, int]


def build_tree(root: Path, *, frames_per_video: int = 7,
               video_frame_counts: dict[str, int] | None = None,
               bad_landmark_rows: dict[str, set[int]] | None = None,
               bad_au_rows: dict[str, set[int]] | None = None,
               invalid_image_rows: dict[str, set[int]] | None = None,
               blank_schema_rows: dict[str, set[int]] | None = None,
               seed: int = 7) -> SynthTree:
    rng = np.random.default_rng(seed)
    splits = {
        "train": ["203_1_Freeform_video", "205_2_Freeform_video", "207_2_Northwind_video"],
        "val": ["209_1_Freeform_video", "211_1_Northwind_video"],
        "test": ["213_1_Freeform_video"],
    }
    counts = video_frame_counts or {}
    scores = {"203_1": 12, "205_2": 25, "207_2": 33, "209_1": 5, "211_1": 19, "213_1": 40}
    of3_root = root / "of3"
    for split, ids in splits.items():
        for sample_id in ids:
            write_features_csv(
                of3_root / "samples" / sample_id, sample_id, split,
                n_frames=counts.get(sample_id, frames_per_video), rng=rng,
                bad_landmark_rows=(bad_landmark_rows or {}).get(sample_id, frozenset()),
                bad_au_rows=(bad_au_rows or {}).get(sample_id, frozenset()),
                invalid_image_rows=(invalid_image_rows or {}).get(sample_id, frozenset()),
                blank_schema_rows=(blank_schema_rows or {}).get(sample_id, frozenset()),
            )
    avec_root = root / "avec"
    split_file = avec_root / "dataset_split.json"
    write_split_json(split_file, splits)
    label_dir = avec_root / "depression_labels"
    write_labels(label_dir, scores)
    norm_root = root / "norm"
    train_csvs = [of3_root / "samples" / s / "features.csv" for s in splits["train"]]
    write_norm_json(norm_root, train_csvs)
    cache_root = root / "cache"
    build_cache_for_tree(cache_root, of3_root, sorted({*splits["train"], *splits["val"], *splits["test"]}), rng=rng)
    weight_path = root / "model.safetensors"
    weight_path.write_bytes(b"not-a-real-weight-file")  # SHA tests only need presence
    return SynthTree(root=root, of3_root=of3_root, avec_root=avec_root, split_file=split_file,
                     label_dir=label_dir, norm_root=norm_root, cache_root=cache_root,
                     weight_path=weight_path, splits=splits, subject_scores=scores)
