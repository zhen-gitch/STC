import csv
import math
import re
from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, write_csv_rows


OPENFACE_REQUIRED_COLUMNS = ("frame", "confidence", "success")
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.8


class RunningStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value):
        if value is None or not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.total_sq += value * value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def mean(self):
        if self.count == 0:
            return None
        return self.total / self.count

    @property
    def std(self):
        if self.count == 0:
            return None
        mean = self.mean
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return math.sqrt(variance)


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


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


def _mean_landmark_motion(current, previous):
    if current is None or previous is None:
        return None
    if current.shape != previous.shape or current.size == 0:
        return None
    deltas = current - previous
    return float(np.sqrt(np.sum(deltas * deltas, axis=1)).mean())


def _landmark_array(row, x_columns, y_columns):
    if not x_columns or len(x_columns) != len(y_columns):
        return None
    points = []
    for x_col, y_col in zip(x_columns, y_columns):
        x_value = _safe_float(row.get(x_col))
        y_value = _safe_float(row.get(y_col))
        if x_value is None or y_value is None:
            return None
        points.append((x_value, y_value))
    return np.asarray(points, dtype=float)


def infer_subject_id_from_openface_path(csv_path):
    stem = Path(csv_path).stem
    match = re.search(r"(\d{3}_\d)", stem)
    if match:
        return match.group(1)
    return stem


def looks_like_openface_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.suffix.lower() != ".csv":
        return False
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except (OSError, UnicodeDecodeError):
        return False
    return all(column in header for column in OPENFACE_REQUIRED_COLUMNS)


def summarize_openface_csv(
    csv_path,
    subject_id=None,
    low_confidence_threshold=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
):
    csv_path = Path(csv_path)
    subject_id = subject_id or infer_subject_id_from_openface_path(csv_path)

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not all(column in fieldnames for column in OPENFACE_REQUIRED_COLUMNS):
            raise ValueError(f"Not an OpenFace CSV or missing required columns: {csv_path}")

        x_columns = _numbered_columns(fieldnames, "x")
        y_columns = _numbered_columns(fieldnames, "y")
        pose_columns = [col for col in ("pose_Rx", "pose_Ry", "pose_Rz") if col in fieldnames]
        gaze_columns = [col for col in fieldnames if col.startswith("gaze_")]
        gaze_angle_columns = [col for col in ("gaze_angle_x", "gaze_angle_y") if col in fieldnames]
        au_r_columns = [col for col in fieldnames if col.startswith("AU") and col.endswith("_r")]
        au_c_columns = [col for col in fieldnames if col.startswith("AU") and col.endswith("_c")]

        confidence_stats = RunningStats()
        timestamp_stats = RunningStats()
        landmark_motion_stats = RunningStats()
        pose_stats = {col: RunningStats() for col in pose_columns}
        pose_abs_stats = {col: RunningStats() for col in pose_columns}
        gaze_stats = {col: RunningStats() for col in gaze_columns}
        gaze_angle_abs_stats = {col: RunningStats() for col in gaze_angle_columns}
        au_r_stats = {col: RunningStats() for col in au_r_columns}
        au_c_stats = {col: RunningStats() for col in au_c_columns}

        frame_count = 0
        low_confidence_count = 0
        success_count = 0
        previous_landmarks = None

        for row in reader:
            frame_count += 1
            confidence = _safe_float(row.get("confidence"))
            success = _safe_float(row.get("success"))
            timestamp = _safe_float(row.get("timestamp"))

            confidence_stats.add(confidence)
            timestamp_stats.add(timestamp)
            if confidence is not None and confidence < low_confidence_threshold:
                low_confidence_count += 1
            if success is not None and success >= 0.5:
                success_count += 1

            for column in pose_columns:
                value = _safe_float(row.get(column))
                pose_stats[column].add(value)
                if value is not None:
                    pose_abs_stats[column].add(abs(value))

            for column in gaze_columns:
                value = _safe_float(row.get(column))
                gaze_stats[column].add(value)
                if column in gaze_angle_abs_stats and value is not None:
                    gaze_angle_abs_stats[column].add(abs(value))

            for column in au_r_columns:
                au_r_stats[column].add(_safe_float(row.get(column)))
            for column in au_c_columns:
                au_c_stats[column].add(_safe_float(row.get(column)))

            landmarks = _landmark_array(row, x_columns, y_columns)
            landmark_motion_stats.add(_mean_landmark_motion(landmarks, previous_landmarks))
            if landmarks is not None:
                previous_landmarks = landmarks

    failed_frame_count = max(0, frame_count - success_count)
    confidence_mean = confidence_stats.mean
    success_ratio = success_count / frame_count if frame_count else None
    summary = {
        "subject_id": subject_id,
        "source_file": str(csv_path),
        "frame_count": frame_count,
        "valid_frame_count": frame_count,
        "success_count": success_count,
        "failed_frame_count": failed_frame_count,
        "success_ratio": success_ratio,
        "failed_frame_ratio": failed_frame_count / frame_count if frame_count else None,
        "confidence_mean": confidence_mean,
        "confidence_std": confidence_stats.std,
        "confidence_min": None if confidence_stats.count == 0 else confidence_stats.minimum,
        "confidence_max": None if confidence_stats.count == 0 else confidence_stats.maximum,
        "low_confidence_ratio": low_confidence_count / frame_count if frame_count else None,
        "timestamp_min": None if timestamp_stats.count == 0 else timestamp_stats.minimum,
        "timestamp_max": None if timestamp_stats.count == 0 else timestamp_stats.maximum,
        "duration_seconds": (
            timestamp_stats.maximum - timestamp_stats.minimum if timestamp_stats.count > 0 else None
        ),
        "landmark_motion_mean": landmark_motion_stats.mean,
        "landmark_motion_std": landmark_motion_stats.std,
        "tracking_quality_score": (
            confidence_mean * success_ratio if confidence_mean is not None and success_ratio is not None else None
        ),
    }

    for column, stats in pose_stats.items():
        name = column.lower()
        summary[f"{name}_mean"] = stats.mean
        summary[f"{name}_std"] = stats.std
        summary[f"{name}_abs_mean"] = pose_abs_stats[column].mean

    for column, stats in gaze_stats.items():
        summary[f"{column}_mean"] = stats.mean
        summary[f"{column}_std"] = stats.std
        if column in gaze_angle_abs_stats:
            summary[f"{column}_abs_mean"] = gaze_angle_abs_stats[column].mean

    for column, stats in au_r_stats.items():
        summary[f"{column}_mean"] = stats.mean
        summary[f"{column}_std"] = stats.std

    for column, stats in au_c_stats.items():
        summary[f"{column}_mean"] = stats.mean

    return summary


def find_openface_csv_files(openface_root):
    openface_root = Path(openface_root)
    if openface_root.is_file():
        return [openface_root] if looks_like_openface_csv(openface_root) else []
    if not openface_root.exists():
        raise FileNotFoundError(f"OpenFace root not found: {openface_root}")
    return sorted(path for path in openface_root.rglob("*.csv") if looks_like_openface_csv(path))


def summarize_openface_root(
    openface_root,
    low_confidence_threshold=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
):
    summaries = []
    for csv_path in find_openface_csv_files(openface_root):
        summaries.append(
            summarize_openface_csv(
                csv_path,
                low_confidence_threshold=low_confidence_threshold,
            )
        )
    return summaries


def write_openface_quality_summary(
    summaries,
    csv_path,
):
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    if not summaries:
        write_csv_rows(csv_path, [], ["subject_id", "source_file"])
        return csv_path

    fieldnames = []
    for row in summaries:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    formatted_rows = []
    for row in summaries:
        formatted_rows.append({key: _format_value(row.get(key)) for key in fieldnames})
    write_csv_rows(csv_path, formatted_rows, fieldnames)
    return csv_path
