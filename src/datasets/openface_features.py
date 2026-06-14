import bisect
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.diagnostics.openface_quality import (
    find_openface_csv_files,
    infer_subject_id_from_openface_path,
    infer_task_name_from_openface_path,
)


def _get_config_value(configs, section_name, key, default):
    section = getattr(configs, section_name, None)
    if section is None:
        return default
    return getattr(section, key, default)


def _normalize_video_id(video_id):
    video_id = str(video_id or "")
    if video_id.endswith(".csv"):
        video_id = Path(video_id).stem
    for suffix in ("_aligned",):
        if video_id.endswith(suffix):
            video_id = video_id[: -len(suffix)]
    return video_id


def _compatible_video_ids(left, right):
    left = _normalize_video_id(left)
    right = _normalize_video_id(right)
    if not left or not right:
        return False
    return left == right or f"{left}_video" == right or left == f"{right}_video"


def _safe_float(value, default=0.0):
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _column_index(column_name, prefix):
    match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)", column_name)
    return int(match.group(1)) if match else None


def _numbered_columns(fieldnames, prefix):
    indexed = []
    for name in fieldnames:
        idx = _column_index(name, prefix)
        if idx is not None:
            indexed.append((idx, name))
    return [name for _idx, name in sorted(indexed)]


def _read_header(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return next(reader, [])


def _feature_flag(configs, key, default):
    return bool(_get_config_value(configs, "BEHAVIOR_FEATURES", key, default))


def select_openface_feature_columns(fieldnames, configs):
    columns = []
    fieldnames = list(fieldnames)

    if _feature_flag(configs, "INCLUDE_QUALITY", True):
        columns.extend([name for name in ("confidence", "success") if name in fieldnames])

    if _feature_flag(configs, "INCLUDE_POSE", True):
        columns.extend([name for name in fieldnames if name.startswith("pose_R") or name.startswith("pose_T")])

    if _feature_flag(configs, "INCLUDE_GAZE", True):
        columns.extend([name for name in fieldnames if name.startswith("gaze_")])

    if _feature_flag(configs, "INCLUDE_AU", True):
        columns.extend([name for name in fieldnames if name.startswith("AU") and name.endswith(("_r", "_c"))])

    if _feature_flag(configs, "INCLUDE_LANDMARKS", True):
        columns.extend(_numbered_columns(fieldnames, "x"))
        columns.extend(_numbered_columns(fieldnames, "y"))

    deduped = []
    for column in columns:
        if column not in deduped:
            deduped.append(column)
    return deduped


def load_split_video_ids(dataset_split_file, split):
    split_path = Path(dataset_split_file).expanduser().resolve()
    if not split_path.exists():
        raise FileNotFoundError(f"Dataset split file not found: {split_path}")
    with split_path.open("r", encoding="utf-8") as f:
        split_data = json.load(f)
    return [Path(item).name for item in split_data[split]]


def _openface_record(csv_path):
    csv_path = Path(csv_path)
    video_id = csv_path.stem
    return {
        "csv_path": csv_path,
        "video_id": video_id,
        "subject_id": infer_subject_id_from_openface_path(csv_path),
        "task_name": infer_task_name_from_openface_path(csv_path),
    }


def build_openface_records(openface_root, dataset_split_file, split, allow_missing=False):
    csv_records = [_openface_record(path) for path in find_openface_csv_files(openface_root)]
    by_video = {_normalize_video_id(row["video_id"]): row for row in csv_records}
    by_subject = {}
    for row in csv_records:
        by_subject.setdefault(row["subject_id"], []).append(row)

    records = []
    missing = []
    for split_video_id in load_split_video_ids(dataset_split_file, split):
        normalized = _normalize_video_id(split_video_id)
        record = by_video.get(normalized)
        if record is None:
            for candidate_video_id, candidate in by_video.items():
                if _compatible_video_ids(normalized, candidate_video_id):
                    record = candidate
                    break
        if record is None:
            subject_id = normalized[:5]
            subject_matches = by_subject.get(subject_id, [])
            if len(subject_matches) == 1:
                record = subject_matches[0]
        if record is None:
            missing.append(split_video_id)
            continue
        records.append(record)

    if missing and not allow_missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing OpenFace CSV files for {len(missing)} split videos. First missing: {preview}"
        )
    if not records:
        raise FileNotFoundError(f"No OpenFace CSV files matched split '{split}'.")
    return records


def derive_temporal_features(features, include_delta=True, include_acceleration=False):
    outputs = [features]
    if include_delta:
        delta = np.zeros_like(features)
        if features.shape[0] > 1:
            delta[1:] = features[1:] - features[:-1]
        outputs.append(delta)
    if include_acceleration:
        acceleration = np.zeros_like(features)
        if features.shape[0] > 2:
            acceleration[2:] = features[2:] - 2.0 * features[1:-1] + features[:-2]
        outputs.append(acceleration)
    return np.concatenate(outputs, axis=1)


class OpenFaceFeatureDataset(Dataset):
    """Sequence dataset built from OpenFace CSV behavior features."""

    def __init__(self, configs, split, records, feature_columns, feature_mean=None, feature_std=None):
        super().__init__()
        self.cfgs = configs
        self.split = split
        self.records = list(records)
        self.feature_columns = list(feature_columns)
        self.label_dir = Path(configs.LABEL_DIR)
        self.class_step = int(configs.PROCESS_TEMPORAL.CLASS_STEP)
        self.sample_step = int(configs.PROCESS_TEMPORAL.SAMPLE_STEP)
        self.max_len = int(configs.PROCESS_TEMPORAL.MAX_SEQ_LEN // self.sample_step)
        self.include_delta = bool(_get_config_value(configs, "BEHAVIOR_FEATURES", "INCLUDE_DELTA", True))
        self.include_acceleration = bool(
            _get_config_value(configs, "BEHAVIOR_FEATURES", "INCLUDE_ACCELERATION", False)
        )
        self.feature_mean = feature_mean
        self.feature_std = feature_std

    @property
    def feature_dim(self):
        multiplier = 1 + int(self.include_delta) + int(self.include_acceleration)
        return len(self.feature_columns) * multiplier

    def _label_value_for_subject(self, subject_id):
        label_path = self.label_dir.joinpath(f"{subject_id}_Depression.csv").expanduser()
        with label_path.open("r", encoding="utf-8") as f:
            return int(f.read().strip())

    def _load_feature_sequence(self, csv_path):
        rows = []
        with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append([_safe_float(row.get(column), default=0.0) for column in self.feature_columns])
        if not rows:
            return np.zeros((0, len(self.feature_columns)), dtype=np.float32)
        features = np.asarray(rows, dtype=np.float32)
        features = features[:: self.sample_step]
        features = features[: self.max_len]
        return derive_temporal_features(
            features,
            include_delta=self.include_delta,
            include_acceleration=self.include_acceleration,
        ).astype(np.float32)

    def _pad_and_normalize(self, features):
        actual_len = min(features.shape[0], self.max_len)
        features = features[:actual_len]
        if self.feature_mean is not None and self.feature_std is not None and actual_len > 0:
            features = (features - self.feature_mean) / self.feature_std

        padded = np.zeros((self.max_len, self.feature_dim), dtype=np.float32)
        if actual_len > 0:
            padded[:actual_len] = features
        mask = np.concatenate(
            [
                np.ones(actual_len, dtype=bool),
                np.zeros(self.max_len - actual_len, dtype=bool),
            ]
        )
        return torch.from_numpy(padded), torch.from_numpy(mask)

    def _load_labels(self, record):
        label = self._label_value_for_subject(record["subject_id"])
        breakpoints = [score for score in range(self.class_step, 64, self.class_step)]
        class_label = bisect.bisect_right(breakpoints, label)
        return {
            "class_label": torch.tensor(class_label, dtype=torch.long),
            "bdi_score": torch.tensor(label, dtype=torch.float32),
            "subject_id": record["subject_id"],
            "video_id": record["video_id"],
            "task_name": record.get("task_name", ""),
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        features = self._load_feature_sequence(record["csv_path"])
        features, mask = self._pad_and_normalize(features)
        return features, mask, self._load_labels(record)


def compute_feature_stats(dataset):
    sequences = []
    for record in dataset.records:
        features = dataset._load_feature_sequence(record["csv_path"])
        if features.size > 0:
            sequences.append(features)
    if not sequences:
        mean = np.zeros((dataset.feature_dim,), dtype=np.float32)
        std = np.ones((dataset.feature_dim,), dtype=np.float32)
        return mean, std
    stacked = np.concatenate(sequences, axis=0).astype(np.float32)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean.astype(np.float32), std


class OpenFaceFeatureDataModule(pl.LightningDataModule):
    """Lightning data module for OpenFace behavior-only baselines."""

    def __init__(self, configs):
        super().__init__()
        self.cfgs = configs
        self.batch_size = int(configs.EXTRACT_FEATURE.BATCH_SIZE)
        self.num_workers = int(configs.EXTRACT_FEATURE.NUM_WORKERS)
        self.openface_root = _get_config_value(configs, "DATASET", "OPENFACE_ROOT", None)
        if self.openface_root is None:
            raise ValueError("DATASET.OPENFACE_ROOT is required for behavior baseline training.")
        self.allow_missing = bool(_get_config_value(configs, "DATASET", "ALLOW_MISSING_OPENFACE", False))
        self.feature_columns = None
        self.feature_mean = None
        self.feature_std = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def feature_dim(self):
        if self.train_dataset is None:
            return None
        return self.train_dataset.feature_dim

    def _records(self, split):
        return build_openface_records(
            self.openface_root,
            self.cfgs.DATASET_SPLIT_FILE,
            split,
            allow_missing=self.allow_missing,
        )

    def _build_feature_columns(self, records):
        for record in records:
            header = _read_header(record["csv_path"])
            columns = select_openface_feature_columns(header, self.cfgs)
            if columns:
                return columns
        raise ValueError("No OpenFace behavior feature columns were selected.")

    def setup(self, stage=None):
        train_records = self._records("train")
        val_records = self._records("val")
        test_records = self._records("test")
        self.feature_columns = self._build_feature_columns(train_records)

        raw_train = OpenFaceFeatureDataset(self.cfgs, "train", train_records, self.feature_columns)
        self.feature_mean, self.feature_std = compute_feature_stats(raw_train)

        self.train_dataset = OpenFaceFeatureDataset(
            self.cfgs, "train", train_records, self.feature_columns, self.feature_mean, self.feature_std
        )
        self.val_dataset = OpenFaceFeatureDataset(
            self.cfgs, "val", val_records, self.feature_columns, self.feature_mean, self.feature_std
        )
        self.test_dataset = OpenFaceFeatureDataset(
            self.cfgs, "test", test_records, self.feature_columns, self.feature_mean, self.feature_std
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
