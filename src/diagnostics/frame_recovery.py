"""Auditable frame-failure inventory and non-destructive RGB recovery.

The audit consumes existing aligned-image OpenFace CSV files.  It never runs
OpenFace/FaceLandmark itself.  Materialization writes only to a separate
derived root and records every changed frame; the source image tree is opened
read-only and is never overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.diagnostics.source_presence import validate_source_review_segments


IMAGE_SUFFIXES = {".jpg", ".jpeg"}

FAILURE_FIELDS = [
    "split",
    "video_id",
    "subject_id",
    "task_name",
    "frame_id",
    "frame_index",
    "image_path",
    "relative_path",
    "original_sha256",
    "openface_csv",
    "openface_timestamp",
    "openface_confidence",
    "openface_success",
    "decode_status",
    "global_mean_luma",
    "global_std_luma",
    "nonblack_ratio",
    "visible_ratio",
    "luma_q05",
    "luma_q10",
    "luma_median",
    "luma_q90",
    "luma_q95",
    "low_saturation_ratio",
    "high_saturation_ratio",
    "failure_type",
    "failure_block_id",
    "failure_block_start",
    "failure_block_end",
    "failure_block_length",
    "previous_valid_frame",
    "previous_valid_path",
    "next_valid_frame",
    "next_valid_path",
    "repair_eligible",
    "proposed_action",
    "fallback_action",
    "eligibility_reason",
]

BLOCK_FIELDS = [
    "split",
    "video_id",
    "failure_block_id",
    "start_frame",
    "end_frame",
    "length",
    "failure_type",
    "previous_valid_frame",
    "previous_valid_path",
    "next_valid_frame",
    "next_valid_path",
    "repair_eligible",
    "proposed_action",
    "fallback_action",
    "eligibility_reason",
]

EXPOSURE_SAMPLE_FIELDS = [
    "split",
    "video_id",
    "frame_id",
    "frame_index",
    "image_path",
    "openface_success",
    "decode_status",
    "pure_black",
    "visible_ratio",
    "luma_q10",
    "luma_median",
    "luma_q90",
    "low_saturation_ratio",
    "high_saturation_ratio",
]

VIDEO_FIELDS = [
    "split",
    "video_id",
    "image_dir",
    "openface_csv",
    "frame_count",
    "openface_row_count",
    "openface_success_count",
    "openface_failure_count",
    "openface_success_ratio",
    "pure_black_failure_count",
    "underexposed_failure_count",
    "overexposed_failure_count",
    "visible_detection_failure_count",
    "unreadable_failure_count",
    "failure_block_count",
    "longest_failure_block",
    "eligible_black_frame_count",
    "exposure_sample_count",
    "exposure_sample_valid_count",
    "video_luma_median",
    "dark_sample_ratio",
    "bright_sample_ratio",
    "video_exposure_status",
    "exposure_reference_source",
    "exposure_safe_low_luma",
    "exposure_safe_high_luma",
    "exposure_target_luma",
    "planned_exposure_curve",
    "planned_exposure_parameter",
    "planned_gamma",
    "recommended_action",
    "issues",
]

EXPOSURE_REVIEW_FIELDS = [
    "split",
    "video_id",
    "segment_start_frame",
    "segment_end_frame",
    "exposure_status",
    "review_status",
    "review_decision",
    "exposure_curve",
    "observed_luma_median",
    "observed_luma_source",
    "target_luma",
    "curve_parameter",
    "reviewer",
    "review_notes",
]

REPAIR_FIELDS = [
    "split",
    "video_id",
    "frame_id",
    "relative_path",
    "source_path",
    "derived_path",
    "source_sha256",
    "derived_sha256",
    "repair_type",
    "source_previous_frame",
    "source_next_frame",
    "interpolation_alpha",
    "exposure_curve",
    "exposure_parameter",
    "exposure_gamma",
    "exposure_segment_start",
    "exposure_segment_end",
    "exposure_review_decision",
    "flow_median_magnitude",
    "flow_median_fb_error",
    "source_luma_median",
    "derived_luma_median",
    "status",
    "notes",
]

EXPOSURE_PREVIEW_FRAME_FIELDS = [
    "split",
    "video_id",
    "frame_id",
    "source_path",
    "exposure_status",
    "exposure_curve",
    "curve_parameter",
    "target_luma",
    "source_luma_median",
    "preview_luma_median",
    "source_luma_span_q90_q10",
    "preview_luma_span_q90_q10",
    "luma_span_retention_ratio",
    "source_low_saturation_ratio",
    "preview_low_saturation_ratio",
    "source_high_saturation_ratio",
    "preview_high_saturation_ratio",
]

EXPOSURE_PREVIEW_VIDEO_FIELDS = [
    "split",
    "video_id",
    "exposure_status",
    "exposure_curve",
    "curve_parameter",
    "observed_sample_luma_median",
    "target_luma",
    "safe_low_luma",
    "safe_high_luma",
    "preview_frame_count",
    "evaluated_frame_count",
    "source_frame_median_luma",
    "preview_frame_median_luma",
    "preview_inside_safe_band",
    "source_low_saturation_ratio_mean",
    "preview_low_saturation_ratio_mean",
    "source_high_saturation_ratio_mean",
    "preview_high_saturation_ratio_mean",
    "luma_span_retention_ratio_median",
    "contact_sheet",
]


@dataclass(frozen=True)
class FrameRecoveryPolicy:
    """Frozen audit and recovery parameters recorded in ``run_manifest.json``."""

    black_threshold: int = 8
    pure_black_max_nonblack_ratio: float = 0.01
    pure_black_max_visible_ratio: float = 0.01
    pure_black_max_mean_luma: float = 1.0
    underexposed_frame_median: float = 35.0
    underexposed_frame_q90: float = 90.0
    overexposed_frame_median: float = 220.0
    overexposed_frame_q90: float = 250.0
    exposure_sample_frames: int = 32
    exposure_video_ratio: float = 0.50
    exposure_safe_low_quantile: float = 0.20
    exposure_reference_low_quantile: float = 0.25
    exposure_reference_high_quantile: float = 0.75
    exposure_safe_high_quantile: float = 0.80
    max_exposure_curve_parameter: float = 32.0
    # Retained only so historical run manifests/configurations still load.
    # New audits and materialization use the log/inverse-log curve family.
    min_gamma: float = 0.50
    max_gamma: float = 4.00
    max_optical_flow_gap: int = 3
    max_copy_gap: int = 1
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 3
    farneback_winsize: int = 15
    farneback_iterations: int = 3
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.2
    max_flow_median_magnitude: float = 12.0
    max_flow_median_fb_error: float = 2.0
    jpeg_quality: int = 95
    jpeg_subsampling: int = 0

    def validate(self):
        if not 0 <= self.black_threshold <= 255:
            raise ValueError("black_threshold must be in [0, 255]")
        ratio_fields = (
            "pure_black_max_nonblack_ratio",
            "pure_black_max_visible_ratio",
            "exposure_video_ratio",
            "exposure_safe_low_quantile",
            "exposure_reference_low_quantile",
            "exposure_reference_high_quantile",
            "exposure_safe_high_quantile",
        )
        for field in ratio_fields:
            value = float(getattr(self, field))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
        if not (
            0.0
            <= self.exposure_reference_low_quantile
            < self.exposure_reference_high_quantile
            <= 1.0
        ):
            raise ValueError("exposure reference quantiles must be increasing")
        positive_ints = (
            "exposure_sample_frames",
            "max_optical_flow_gap",
            "max_copy_gap",
            "farneback_levels",
            "farneback_winsize",
            "farneback_iterations",
            "farneback_poly_n",
            "jpeg_quality",
        )
        for field in positive_ints:
            if int(getattr(self, field)) <= 0:
                raise ValueError(f"{field} must be positive")
        if not 0.0 < self.min_gamma <= 1.0 <= self.max_gamma:
            raise ValueError("gamma bounds must satisfy 0 < min_gamma <= 1 <= max_gamma")
        if not (
            math.isfinite(self.max_exposure_curve_parameter)
            and 0.0 < self.max_exposure_curve_parameter <= 64.0
        ):
            raise ValueError("max_exposure_curve_parameter must lie in (0, 64]")
        if self.jpeg_quality > 100:
            raise ValueError("jpeg_quality must not exceed 100")
        if self.jpeg_subsampling not in {0, 1, 2}:
            raise ValueError("jpeg_subsampling must be 0, 1, or 2")
        return self


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _write_csv(path, rows, fields):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field)) for field in fields})
    return path


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(project_root, arguments):
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _safe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_int(value):
    number = _safe_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _frame_id(path):
    groups = re.findall(r"\d+", Path(path).stem)
    return int(groups[-1]) if groups else None


def _subject_and_task(video_id):
    match = re.match(r"^(\d+_\d+)_(Freeform|Northwind)", str(video_id))
    if match is None:
        return "", ""
    return match.group(1), match.group(2)


def load_split_map(dataset_split_file):
    payload = json.loads(Path(dataset_split_file).read_text(encoding="utf-8"))
    result = {}
    duplicates = set()
    for split, video_ids in payload.items():
        for video_id in video_ids:
            key = str(video_id)
            if key in result and result[key] != split:
                duplicates.add(key)
            result[key] = str(split)
    if duplicates:
        raise ValueError(f"videos occur in multiple splits: {sorted(duplicates)[:10]}")
    return result


def _axis_connected_padding_mask(rgb, threshold):
    black = np.all(rgb <= int(threshold), axis=2)
    top = np.logical_and.accumulate(black, axis=0)
    bottom = np.logical_and.accumulate(black[::-1], axis=0)[::-1]
    left = np.logical_and.accumulate(black, axis=1)
    right = np.logical_and.accumulate(black[:, ::-1], axis=1)[:, ::-1]
    return top | bottom | left | right


def _metrics_from_rgb(rgb, policy):
    rgb = np.asarray(rgb, dtype=np.uint8)
    luma = (
        0.2989 * rgb[:, :, 0].astype(np.float32)
        + 0.5870 * rgb[:, :, 1].astype(np.float32)
        + 0.1140 * rgb[:, :, 2].astype(np.float32)
    )
    padding = _axis_connected_padding_mask(rgb, policy.black_threshold)
    visible = ~padding
    values = luma[visible]
    metrics = {
        "global_mean_luma": float(luma.mean()),
        "global_std_luma": float(luma.std()),
        "nonblack_ratio": float(np.mean(np.any(rgb > policy.black_threshold, axis=2))),
        "visible_ratio": float(visible.mean()),
        "luma_q05": None,
        "luma_q10": None,
        "luma_median": None,
        "luma_q90": None,
        "luma_q95": None,
        "low_saturation_ratio": None,
        "high_saturation_ratio": None,
    }
    if values.size:
        quantiles = np.quantile(values, [0.05, 0.10, 0.50, 0.90, 0.95])
        for name, value in zip(
            ("luma_q05", "luma_q10", "luma_median", "luma_q90", "luma_q95"),
            quantiles,
        ):
            metrics[name] = float(value)
        metrics["low_saturation_ratio"] = float(np.mean(values <= 16.0))
        metrics["high_saturation_ratio"] = float(np.mean(values >= 250.0))
    return metrics


def inspect_frame(path, policy, compute_hash=False):
    path = Path(path)
    result = {
        "decode_status": "ERROR",
        "original_sha256": "",
        "error": "",
    }
    try:
        encoded = path.read_bytes()
        if compute_hash:
            result["original_sha256"] = _sha256_bytes(encoded)
        with Image.open(BytesIO(encoded)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        result.update(_metrics_from_rgb(rgb, policy))
        result["decode_status"] = "OK"
    except Exception as exc:  # Pillow exposes decoder-specific exceptions.
        result["error"] = f"{type(exc).__name__}:{exc}"
    return result


def classify_failed_frame(metrics, policy):
    if metrics.get("decode_status") != "OK":
        return "image_unreadable"
    if (
        float(metrics["nonblack_ratio"]) <= policy.pure_black_max_nonblack_ratio
        or float(metrics["visible_ratio"]) <= policy.pure_black_max_visible_ratio
        or float(metrics["global_mean_luma"]) <= policy.pure_black_max_mean_luma
    ):
        return "pure_black"
    median = metrics.get("luma_median")
    q90 = metrics.get("luma_q90")
    if median is not None and q90 is not None:
        if median <= policy.underexposed_frame_median and q90 <= policy.underexposed_frame_q90:
            return "underexposed_detection_failed"
        if median >= policy.overexposed_frame_median and q90 >= policy.overexposed_frame_q90:
            return "overexposed_detection_failed"
    return "visible_detection_failed"


def _even_indices(length, count):
    if length <= 0 or count <= 0:
        return []
    return sorted(set(np.linspace(0, length - 1, min(length, count)).round().astype(int).tolist()))


def _find_openface_csv_map(openface_root):
    result = {}
    for path in sorted(Path(openface_root).rglob("*.csv")):
        if path.parent.name == "_audit":
            continue
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                header = next(csv.reader(handle), [])
        except (OSError, UnicodeDecodeError, StopIteration):
            continue
        if "frame" in header and "success" in header:
            result[path.stem] = path
    return result


def _read_openface_rows(path):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            rows.append(
                {
                    "row_index": index,
                    "frame": _safe_int(row.get("frame")),
                    "timestamp": _safe_float(row.get("timestamp")),
                    "confidence": _safe_float(row.get("confidence")),
                    "success": _safe_int(row.get("success")),
                }
            )
    return rows


def _block_action(block_length, previous_row, next_row, failure_type, policy):
    if failure_type != "pure_black":
        return False, "no_pixel_repair", "", "visible failure is not synthesized"
    previous_valid = previous_row is not None and previous_row.get("success") == 1
    next_valid = next_row is not None and next_row.get("success") == 1
    if block_length <= policy.max_optical_flow_gap and previous_valid and next_valid:
        return (
            False,
            "source_video_review",
            "",
            "short aligned placeholder; raw-source presence review and raw-frame warp are required",
        )
    return (
        False,
        "keep_invalid",
        "",
        "raw-source presence is unknown or the aligned gap is too long",
    )


def _audit_video(task):
    video_dir, openface_csv, split, policy = task
    video_dir = Path(video_dir)
    openface_csv = Path(openface_csv)
    video_id = video_dir.name
    subject_id, task_name = _subject_and_task(video_id)
    image_paths = sorted(
        path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    image_by_frame = {}
    duplicate_frames = set()
    for path in image_paths:
        frame = _frame_id(path)
        if frame is None:
            continue
        if frame in image_by_frame:
            duplicate_frames.add(frame)
        else:
            image_by_frame[frame] = path
    openface_rows = _read_openface_rows(openface_csv)

    sample_rows = []
    for row_index in _even_indices(len(openface_rows), policy.exposure_sample_frames):
        row = openface_rows[row_index]
        path = image_by_frame.get(row["frame"])
        value = inspect_frame(path, policy) if path is not None else {"decode_status": "MISSING"}
        pure_black = (
            value.get("decode_status") == "OK"
            and (
                value["nonblack_ratio"] <= policy.pure_black_max_nonblack_ratio
                or value["visible_ratio"] <= policy.pure_black_max_visible_ratio
                or value["global_mean_luma"] <= policy.pure_black_max_mean_luma
            )
        )
        sample_rows.append(
            {
                "split": split,
                "video_id": video_id,
                "frame_id": row["frame"],
                "frame_index": row_index,
                "image_path": str(path or ""),
                "openface_success": row["success"],
                "decode_status": value.get("decode_status", "ERROR"),
                "pure_black": pure_black,
                "visible_ratio": value.get("visible_ratio"),
                "luma_q10": value.get("luma_q10"),
                "luma_median": value.get("luma_median"),
                "luma_q90": value.get("luma_q90"),
                "low_saturation_ratio": value.get("low_saturation_ratio"),
                "high_saturation_ratio": value.get("high_saturation_ratio"),
            }
        )

    failure_rows = []
    for row in openface_rows:
        if row.get("success") == 1:
            continue
        path = image_by_frame.get(row["frame"])
        value = inspect_frame(path, policy, compute_hash=True) if path is not None else {
            "decode_status": "MISSING",
            "original_sha256": "",
            "error": "image path missing",
        }
        failure_type = classify_failed_frame(value, policy)
        failure_rows.append(
            {
                "split": split,
                "video_id": video_id,
                "subject_id": subject_id,
                "task_name": task_name,
                "frame_id": row["frame"],
                "frame_index": row["row_index"],
                "image_path": str(path or ""),
                "relative_path": (
                    Path(video_id, path.name).as_posix() if path is not None else ""
                ),
                "original_sha256": value.get("original_sha256", ""),
                "openface_csv": str(openface_csv),
                "openface_timestamp": row["timestamp"],
                "openface_confidence": row["confidence"],
                "openface_success": row["success"],
                **{
                    key: value.get(key)
                    for key in (
                        "decode_status",
                        "global_mean_luma",
                        "global_std_luma",
                        "nonblack_ratio",
                        "visible_ratio",
                        "luma_q05",
                        "luma_q10",
                        "luma_median",
                        "luma_q90",
                        "luma_q95",
                        "low_saturation_ratio",
                        "high_saturation_ratio",
                    )
                },
                "failure_type": failure_type,
            }
        )

    failure_by_index = {row["frame_index"]: row for row in failure_rows}
    blocks = []
    index = 0
    block_number = 0
    while index < len(openface_rows):
        if openface_rows[index].get("success") == 1:
            index += 1
            continue
        start = index
        while index + 1 < len(openface_rows) and openface_rows[index + 1].get("success") != 1:
            index += 1
        end = index
        block_number += 1
        block_failures = [failure_by_index[item] for item in range(start, end + 1)]
        types = {row["failure_type"] for row in block_failures}
        block_type = next(iter(types)) if len(types) == 1 else "mixed"
        previous_row = openface_rows[start - 1] if start > 0 else None
        next_row = openface_rows[end + 1] if end + 1 < len(openface_rows) else None
        previous_path = image_by_frame.get(previous_row["frame"]) if previous_row else None
        next_path = image_by_frame.get(next_row["frame"]) if next_row else None
        eligible, action, fallback, reason = _block_action(
            end - start + 1,
            previous_row,
            next_row,
            block_type,
            policy,
        )
        if previous_row is not None and previous_path is None:
            eligible, action, fallback, reason = False, "keep_invalid", "", "previous image missing"
        if action == "bidirectional_optical_flow" and next_path is None:
            eligible, action, fallback, reason = False, "keep_invalid", "", "next image missing"
        block_id = f"{video_id}:F{block_number:04d}"
        block = {
            "split": split,
            "video_id": video_id,
            "failure_block_id": block_id,
            "start_frame": openface_rows[start]["frame"],
            "end_frame": openface_rows[end]["frame"],
            "length": end - start + 1,
            "failure_type": block_type,
            "previous_valid_frame": previous_row["frame"] if previous_row else None,
            "previous_valid_path": str(previous_path or ""),
            "next_valid_frame": next_row["frame"] if next_row else None,
            "next_valid_path": str(next_path or ""),
            "repair_eligible": eligible,
            "proposed_action": action,
            "fallback_action": fallback,
            "eligibility_reason": reason,
        }
        blocks.append(block)
        for row in block_failures:
            row.update(
                {
                    "failure_block_id": block_id,
                    "failure_block_start": block["start_frame"],
                    "failure_block_end": block["end_frame"],
                    "failure_block_length": block["length"],
                    "previous_valid_frame": block["previous_valid_frame"],
                    "previous_valid_path": block["previous_valid_path"],
                    "next_valid_frame": block["next_valid_frame"],
                    "next_valid_path": block["next_valid_path"],
                    "repair_eligible": eligible,
                    "proposed_action": action,
                    "fallback_action": fallback,
                    "eligibility_reason": reason,
                }
            )
        index += 1

    valid_sample_luma = [
        row["luma_median"]
        for row in sample_rows
        if row["decode_status"] == "OK" and not row["pure_black"] and row["luma_median"] is not None
    ]
    dark_ratio = (
        sum(value <= policy.underexposed_frame_median for value in valid_sample_luma)
        / len(valid_sample_luma)
        if valid_sample_luma
        else 0.0
    )
    bright_ratio = (
        sum(value >= policy.overexposed_frame_median for value in valid_sample_luma)
        / len(valid_sample_luma)
        if valid_sample_luma
        else 0.0
    )
    type_counts = Counter(row["failure_type"] for row in failure_rows)
    issues = []
    if len(image_paths) != len(openface_rows):
        issues.append("image_csv_count_mismatch")
    if duplicate_frames:
        issues.append("duplicate_image_frame_ids")
    missing_failure_images = sum(not row["image_path"] for row in failure_rows)
    if missing_failure_images:
        issues.append("missing_failure_images")
    summary = {
        "split": split,
        "video_id": video_id,
        "image_dir": str(video_dir),
        "openface_csv": str(openface_csv),
        "frame_count": len(image_paths),
        "openface_row_count": len(openface_rows),
        "openface_success_count": sum(row.get("success") == 1 for row in openface_rows),
        "openface_failure_count": len(failure_rows),
        "openface_success_ratio": (
            sum(row.get("success") == 1 for row in openface_rows) / len(openface_rows)
            if openface_rows
            else 0.0
        ),
        "pure_black_failure_count": type_counts["pure_black"],
        "underexposed_failure_count": type_counts["underexposed_detection_failed"],
        "overexposed_failure_count": type_counts["overexposed_detection_failed"],
        "visible_detection_failure_count": type_counts["visible_detection_failed"],
        "unreadable_failure_count": type_counts["image_unreadable"],
        "failure_block_count": len(blocks),
        "longest_failure_block": max((row["length"] for row in blocks), default=0),
        "eligible_black_frame_count": sum(
            row["length"] for row in blocks if row["repair_eligible"]
        ),
        "exposure_sample_count": len(sample_rows),
        "exposure_sample_valid_count": len(valid_sample_luma),
        "video_luma_median": float(np.median(valid_sample_luma)) if valid_sample_luma else None,
        "dark_sample_ratio": dark_ratio,
        "bright_sample_ratio": bright_ratio,
        "issues": ";".join(issues),
    }
    return summary, failure_rows, blocks, sample_rows


def exposure_curve_value(value, curve, parameter):
    """Map normalized luma through a frozen endpoint-preserving curve."""

    value = np.asarray(value, dtype=np.float64)
    parameter = float(parameter)
    if not math.isfinite(parameter) or not 0.0 <= parameter <= 64.0:
        raise ValueError("exposure curve parameter must lie in [0, 64]")
    if curve not in {"identity", "log", "inverse_log"}:
        raise ValueError("exposure curve must be identity, log, or inverse_log")
    clipped = np.clip(value, 0.0, 1.0)
    if parameter == 0.0 or curve == "identity":
        return clipped
    if curve == "log":
        return np.log1p(parameter * clipped) / math.log1p(parameter)
    if curve == "inverse_log":
        return np.expm1(parameter * clipped) / math.expm1(parameter)
    raise AssertionError("unreachable exposure curve")


def fit_exposure_curve_parameter(observed_luma, target_luma, curve, max_parameter=32.0):
    """Fit one video/segment-level log-family parameter by monotone bisection."""

    observed = float(observed_luma) / 255.0
    target = float(target_luma) / 255.0
    max_parameter = float(max_parameter)
    if not all(math.isfinite(value) for value in (observed, target, max_parameter)):
        raise ValueError("observed_luma, target_luma, and max_parameter must be finite")
    if not 0.0 < observed < 1.0 or not 0.0 < target < 1.0:
        raise ValueError("observed_luma and target_luma must lie strictly inside (0, 255)")
    if not 0.0 < max_parameter <= 64.0:
        raise ValueError("max_parameter must lie in (0, 64]")
    if math.isclose(observed, target, rel_tol=0.0, abs_tol=1e-12):
        return 0.0
    if curve == "log" and target <= observed:
        raise ValueError("log exposure recovery requires target_luma > observed_luma")
    if curve == "inverse_log" and target >= observed:
        raise ValueError("inverse_log exposure recovery requires target_luma < observed_luma")
    if curve not in {"log", "inverse_log"}:
        raise ValueError("curve must be log or inverse_log")

    endpoint = float(exposure_curve_value(observed, curve, max_parameter))
    if (curve == "log" and endpoint < target) or (curve == "inverse_log" and endpoint > target):
        raise ValueError(
            f"target_luma requires a {curve} parameter above the frozen maximum {max_parameter:g}"
        )
    low, high = 0.0, max_parameter
    for _ in range(80):
        middle = (low + high) / 2.0
        mapped = float(exposure_curve_value(observed, curve, middle))
        if (curve == "log" and mapped < target) or (curve == "inverse_log" and mapped > target):
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _apply_exposure_plan(summaries, policy):
    if not (
        policy.exposure_safe_low_quantile
        < policy.exposure_reference_low_quantile
        < policy.exposure_reference_high_quantile
        < policy.exposure_safe_high_quantile
    ):
        raise ValueError(
            "new exposure audits require safe_low < target_low < target_high < safe_high"
        )
    for row in summaries:
        median = row.get("video_luma_median")
        if median is None:
            status = "unavailable"
        elif (
            median <= policy.underexposed_frame_median
            and row["dark_sample_ratio"] >= policy.exposure_video_ratio
        ):
            status = "underexposed"
        elif (
            median >= policy.overexposed_frame_median
            and row["bright_sample_ratio"] >= policy.exposure_video_ratio
        ):
            status = "overexposed"
        else:
            status = "normal"
        row["video_exposure_status"] = status

    train_reference = [
        row["video_luma_median"]
        for row in summaries
        if row["split"] == "train"
        and row["video_exposure_status"] == "normal"
        and row["video_luma_median"] is not None
    ]
    if train_reference:
        safe_low, low_target, high_target, safe_high = np.quantile(
            train_reference,
            [
                policy.exposure_safe_low_quantile,
                policy.exposure_reference_low_quantile,
                policy.exposure_reference_high_quantile,
                policy.exposure_safe_high_quantile,
            ],
        )
        reference_source = "train_normal_video_luma_quantiles"
    else:
        safe_low, low_target, high_target, safe_high = 48.0, 54.0, 136.0, 142.0
        reference_source = "fixed_fallback_luma_targets"

    for row in summaries:
        status = row["video_exposure_status"]
        observed = row.get("video_luma_median")
        target = None
        curve = "identity"
        parameter = 0.0
        if status == "underexposed":
            target = float(low_target)
            curve = "log"
        elif status == "overexposed":
            target = float(high_target)
            curve = "inverse_log"
        if target is not None and observed is not None:
            parameter = fit_exposure_curve_parameter(
                observed,
                target,
                curve,
                max_parameter=policy.max_exposure_curve_parameter,
            )
        row["exposure_reference_source"] = reference_source
        row["exposure_safe_low_luma"] = float(safe_low)
        row["exposure_safe_high_luma"] = float(safe_high)
        row["exposure_target_luma"] = target
        row["planned_exposure_curve"] = curve
        row["planned_exposure_parameter"] = parameter
        row["planned_gamma"] = None
        actions = []
        if row["eligible_black_frame_count"]:
            actions.append("repair_short_pure_black_blocks")
        if status in {"underexposed", "overexposed"}:
            actions.append("complete_exposure_review_before_log_curve")
        if row["openface_failure_count"] - row["eligible_black_frame_count"] > 0:
            actions.append("keep_remaining_failures_invalid")
        row["recommended_action"] = ";".join(actions) if actions else "no_pixel_change"
    return {
        "source": reference_source,
        "train_reference_video_count": len(train_reference),
        "safe_low_quantile": policy.exposure_safe_low_quantile,
        "low_quantile": policy.exposure_reference_low_quantile,
        "high_quantile": policy.exposure_reference_high_quantile,
        "safe_high_quantile": policy.exposure_safe_high_quantile,
        "safe_low_luma": float(safe_low),
        "underexposed_target_luma": float(low_target),
        "overexposed_target_luma": float(high_target),
        "safe_high_luma": float(safe_high),
    }


def _build_exposure_review_template(summaries):
    rows = []
    for summary in summaries:
        status = summary["video_exposure_status"]
        if status not in {"underexposed", "overexposed"}:
            continue
        rows.append(
            {
                "split": summary["split"],
                "video_id": summary["video_id"],
                "segment_start_frame": 1,
                "segment_end_frame": summary["frame_count"],
                "exposure_status": status,
                "review_status": "PENDING",
                "review_decision": "segment_review_required",
                "exposure_curve": summary["planned_exposure_curve"],
                "observed_luma_median": summary["video_luma_median"],
                "observed_luma_source": "sampled_audit_median_replace_before_review",
                "target_luma": summary["exposure_target_luma"],
                "curve_parameter": summary["planned_exposure_parameter"],
                "reviewer": "",
                "review_notes": "Review full-frame temporal exposure before approving this whole-video curve or splitting the interval.",
            }
        )
    return rows


def _write_audit_report(path, summaries, failures, blocks, exposure_reference, policy):
    status_counts = Counter(row["video_exposure_status"] for row in summaries)
    type_counts = Counter(row["failure_type"] for row in failures)
    action_counts = Counter(row["proposed_action"] for row in blocks)
    longest = sorted(blocks, key=lambda row: row["length"], reverse=True)[:15]
    lines = [
        "# Frame Failure and Exposure Recovery Audit",
        "",
        "- This command used existing OpenFace CSV outputs; it did not run FaceLandmark/OpenFace.",
        f"- Videos: {len(summaries)}",
        f"- OpenFace failed frames: {len(failures)}",
        f"- Pure-black failed frames: {type_counts['pure_black']}",
        f"- Visible failed frames: {len(failures) - type_counts['pure_black']}",
        f"- Directly repair-eligible black frames: {sum(row['eligible_black_frame_count'] for row in summaries)}",
        f"- Underexposed videos: {status_counts['underexposed']}",
        f"- Overexposed videos: {status_counts['overexposed']}",
        f"- Normal-exposure videos: {status_counts['normal']}",
        "",
        "## Frozen Policy",
        "",
        f"- Pure black: nonblack ratio <= {policy.pure_black_max_nonblack_ratio:.3f}, visible ratio <= {policy.pure_black_max_visible_ratio:.3f}, or mean luma <= {policy.pure_black_max_mean_luma:.1f}.",
        f"- Underexposed video: sampled median <= {policy.underexposed_frame_median:.1f} and dark-sample ratio >= {policy.exposure_video_ratio:.2f}.",
        f"- Overexposed video: sampled median >= {policy.overexposed_frame_median:.1f} and bright-sample ratio >= {policy.exposure_video_ratio:.2f}.",
        f"- The train-normal acceptance band is q{policy.exposure_safe_low_quantile:.2f}/q{policy.exposure_safe_high_quantile:.2f}: {exposure_reference['safe_low_luma']:.3f}/{exposure_reference['safe_high_luma']:.3f}.",
        f"- Exposure targets sit inside that band at q{policy.exposure_reference_low_quantile:.2f}/q{policy.exposure_reference_high_quantile:.2f}: {exposure_reference['underexposed_target_luma']:.3f}/{exposure_reference['overexposed_target_luma']:.3f}.",
        "- Pure-black aligned placeholders are not synthesized from aligned neighbors. Even short blocks require original-video presence review and a future raw-frame warp implementation.",
        "",
        "## Failure Type Counts",
        "",
        "| Failure type | Frames |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in sorted(type_counts.items()))
    lines.extend(["", "## Proposed Block Actions", "", "| Action | Blocks |", "|---|---:|"])
    lines.extend(f"| {name} | {count} |" for name, count in sorted(action_counts.items()))
    lines.extend(["", "## Longest Failure Blocks", "", "| Video | Frames | Type | Action |", "|---|---:|---|---|"])
    lines.extend(
        f"| {row['video_id']} | {row['length']} | {row['failure_type']} | {row['proposed_action']} |"
        for row in longest
    )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- A pure-black aligned frame is a preprocessing placeholder, not proof that the source frame is damaged or that the person is absent.",
            "- All black intervals remain invalid until a source-video review labels person_absent, person_present_detection_failure, mixed, or ambiguous.",
            "- Aligned-neighbor optical-flow/copy synthesis is disabled because it can invent a face and erase real occlusion, pose, or out-of-frame evidence.",
            "- Underexposure uses T_log(x;a)=log(1+a*x)/log(1+a); overexposure uses the endpoint-preserving inverse-log T_invlog(x;b)=(exp(b*x)-1)/(exp(b)-1).",
            "- One fixed parameter is fitted per reviewed video or stable exposure segment. Per-frame fitting is forbidden because it can create temporal flicker.",
            "- The generated exposure review template is intentionally PENDING. Materialization requires complete, non-overlapping reviewed coverage for every selected exposure candidate.",
            "- Curves change only visible-region luma. Axis-connected black padding is preserved and RGB channel differences are retained where gamut permits.",
            "- Overexposed clipped pixels cannot be reconstructed; the method only restores luminance to the train normal-video envelope and reports saturation ratios.",
            "- Materialized frames remain unvalidated for landmarks until a later, explicitly authorized OpenFace rerun. AU-T1/training stays blocked.",
            "- Paper comparison must pair raw and repaired inputs with identical split, seed, model, optimizer, epoch budget, precision, checkpoint rule, and diagnostics.",
        ]
    )
    ensure_dir(Path(path).parent)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Path(path)


def run_frame_failure_audit(
    image_root,
    openface_root,
    dataset_split_file,
    output_dir,
    policy=None,
    workers=None,
    max_videos=None,
    project_root=None,
):
    policy = (policy or FrameRecoveryPolicy()).validate()
    image_root = Path(image_root).expanduser().resolve()
    openface_root = Path(openface_root).expanduser().resolve()
    dataset_split_file = Path(dataset_split_file).expanduser().resolve()
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    split_map = load_split_map(dataset_split_file)
    csv_map = _find_openface_csv_map(openface_root)
    video_dirs = sorted(path for path in image_root.iterdir() if path.is_dir())
    if max_videos is not None:
        video_dirs = video_dirs[: max(0, int(max_videos))]
    tasks = []
    missing_csvs = []
    for video_dir in video_dirs:
        csv_path = csv_map.get(video_dir.name)
        if csv_path is None:
            missing_csvs.append(video_dir.name)
            continue
        tasks.append((video_dir, csv_path, split_map.get(video_dir.name, "unknown"), policy))

    worker_count = max(1, int(workers or min(8, os.cpu_count() or 1)))
    if worker_count == 1:
        results = map(_audit_video, tasks)
    else:
        executor = ThreadPoolExecutor(max_workers=worker_count)
        results = executor.map(_audit_video, tasks)
    summaries, failures, blocks, samples = [], [], [], []
    try:
        for index, (summary, video_failures, video_blocks, video_samples) in enumerate(results, 1):
            summaries.append(summary)
            failures.extend(video_failures)
            blocks.extend(video_blocks)
            samples.extend(video_samples)
            if index % 25 == 0 or index == len(tasks):
                print(f"[FRAME_RECOVERY_AUDIT] processed {index}/{len(tasks)} videos", flush=True)
    finally:
        if worker_count != 1:
            executor.shutdown()

    summaries.sort(key=lambda row: row["video_id"])
    failures.sort(key=lambda row: (row["video_id"], row["frame_index"]))
    blocks.sort(key=lambda row: (row["video_id"], row["start_frame"]))
    samples.sort(key=lambda row: (row["video_id"], row["frame_index"]))
    exposure_reference = _apply_exposure_plan(summaries, policy)

    failure_path = _write_csv(tables_dir / "frame_failure_manifest.csv", failures, FAILURE_FIELDS)
    block_path = _write_csv(tables_dir / "failure_blocks.csv", blocks, BLOCK_FIELDS)
    sample_path = _write_csv(tables_dir / "exposure_sample_manifest.csv", samples, EXPOSURE_SAMPLE_FIELDS)
    video_path = _write_csv(tables_dir / "video_failure_summary.csv", summaries, VIDEO_FIELDS)
    exposure_review_path = _write_csv(
        tables_dir / "exposure_review_template.csv",
        _build_exposure_review_template(summaries),
        EXPOSURE_REVIEW_FIELDS,
    )
    report_path = _write_audit_report(
        reports_dir / "frame_failure_recovery_report.md",
        summaries,
        failures,
        blocks,
        exposure_reference,
        policy,
    )

    project_root = Path(project_root or Path(__file__).resolve().parents[2])
    openface_manifest = openface_root / "_audit" / "run_manifest.json"
    payload = {
        "audit": "aligned frame failure and exposure recovery planning",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(shlex.quote(item) for item in sys.argv),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "image_root": str(image_root),
        "openface_root": str(openface_root),
        "dataset_split_file": str(dataset_split_file),
        "dataset_split_sha256": _sha256_file(dataset_split_file),
        "openface_run_manifest": str(openface_manifest) if openface_manifest.exists() else "",
        "openface_run_manifest_sha256": _sha256_file(openface_manifest) if openface_manifest.exists() else "",
        "face_landmark_rerun_performed": False,
        "policy": asdict(policy),
        "exposure_reference": exposure_reference,
        "exposure_curve_family": {
            "underexposed": {
                "name": "log",
                "formula": "log(1+a*x)/log(1+a)",
            },
            "overexposed": {
                "name": "inverse_log",
                "formula": "(exp(b*x)-1)/(exp(b)-1)",
                "clipped_detail_recovery_claimed": False,
            },
            "parameter_fit": "one fixed parameter per reviewed video or stable segment",
            "per_frame_parameter_fitting": False,
        },
        "workers": worker_count,
        "max_videos": max_videos,
        "video_count": len(summaries),
        "missing_openface_csv_videos": missing_csvs,
        "failed_frame_count": len(failures),
        "failure_block_count": len(blocks),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "implementation_files": {
            "core": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "cli": {
                "path": str(project_root / "scripts" / "audit_frame_recovery.py"),
                "sha256": _sha256_file(project_root / "scripts" / "audit_frame_recovery.py"),
            },
        },
        "outputs": {
            "frame_failure_manifest": {"path": str(failure_path), "sha256": _sha256_file(failure_path)},
            "failure_blocks": {"path": str(block_path), "sha256": _sha256_file(block_path)},
            "exposure_sample_manifest": {"path": str(sample_path), "sha256": _sha256_file(sample_path)},
            "video_failure_summary": {"path": str(video_path), "sha256": _sha256_file(video_path)},
            "exposure_review_template": {
                "path": str(exposure_review_path),
                "sha256": _sha256_file(exposure_review_path),
            },
            "report": {"path": str(report_path), "sha256": _sha256_file(report_path)},
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [
        failure_path,
        block_path,
        sample_path,
        video_path,
        exposure_review_path,
        report_path,
        manifest_path,
    ]


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve_recorded_path(value, *bases):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in bases:
        candidate = (Path(base) / path).resolve()
        if candidate.exists():
            return candidate
    return (Path(bases[0]) / path).resolve()


def validate_relocated_image_root(
    image_root,
    audited_root,
    comparison_summary,
    project_root,
):
    """Authorize a relocated byte-identical image root from an EXACT_PASS audit."""

    image_root = Path(image_root).resolve()
    audited_root = Path(audited_root).resolve()
    if image_root == audited_root:
        return {
            "mode": "audited_root_exact_path",
            "comparison_summary": "",
            "comparison_summary_sha256": "",
        }
    if comparison_summary is None:
        raise ValueError(
            "image_root differs from the audited source root; provide an EXACT_PASS "
            "--image-integrity-comparison-summary for this relocated root"
        )
    comparison_path = Path(comparison_summary).expanduser().resolve()
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    if payload.get("status") != "EXACT_PASS":
        raise ValueError("relocated image-root comparison status must be EXACT_PASS")
    reference_count = int(payload.get("reference_count", -1))
    candidate_count = int(payload.get("candidate_count", -1))
    exact_count = int(payload.get("status_counts", {}).get("EXACT_MATCH", -1))
    if not reference_count == candidate_count == exact_count or candidate_count <= 0:
        raise ValueError("relocated image-root comparison does not prove complete exact coverage")
    candidate_manifest = _resolve_recorded_path(
        payload["candidate_manifest"],
        project_root,
        comparison_path.parent,
    )
    manifest_sha256 = _sha256_file(candidate_manifest)
    if manifest_sha256 != payload.get("candidate_manifest_sha256"):
        raise ValueError("relocated candidate manifest SHA-256 differs from comparison summary")
    inventory_path = candidate_manifest.parents[1] / "inventory_summary.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("status") != "PASS":
        raise ValueError("relocated image-root inventory status must be PASS")
    if Path(inventory["image_root"]).expanduser().resolve() != image_root:
        raise ValueError("integrity inventory image_root does not match the requested relocated root")
    if int(inventory.get("image_count", -1)) != candidate_count:
        raise ValueError("integrity inventory count differs from the exact comparison")
    if int(inventory.get("decode_ok_count", -1)) != candidate_count:
        raise ValueError("not every relocated image passed decode validation")
    return {
        "mode": "relocated_root_exact_pass",
        "comparison_summary": str(comparison_path),
        "comparison_summary_sha256": _sha256_file(comparison_path),
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": manifest_sha256,
        "candidate_count": candidate_count,
        "inventory_summary": str(inventory_path),
        "inventory_summary_sha256": _sha256_file(inventory_path),
    }


def validate_exposure_review_segments(
    review_manifest,
    selected_summaries,
    policy,
    audited_summaries=None,
):
    """Validate complete reviewed exposure coverage without label/metric access."""

    rows = _read_csv(review_manifest)
    if rows and not set(EXPOSURE_REVIEW_FIELDS).issubset(rows[0]):
        missing = sorted(set(EXPOSURE_REVIEW_FIELDS) - set(rows[0]))
        raise ValueError(f"exposure review manifest is missing columns: {missing}")
    audited_summaries = list(audited_summaries or selected_summaries)
    audited_candidates = {
        row["video_id"]: row
        for row in audited_summaries
        if row["video_exposure_status"] in {"underexposed", "overexposed"}
    }
    selected_candidates = {
        row["video_id"]: row
        for row in selected_summaries
        if row["video_exposure_status"] in {"underexposed", "overexposed"}
    }
    unknown = sorted({row["video_id"] for row in rows} - set(audited_candidates))
    if unknown:
        raise ValueError(f"exposure review contains non-candidate or unknown videos: {unknown}")

    rows_by_video = {}
    for row in rows:
        if row["video_id"] in selected_candidates:
            rows_by_video.setdefault(row["video_id"], []).append(dict(row))

    validated = []
    for video_id, summary in selected_candidates.items():
        video_rows = rows_by_video.get(video_id, [])
        if not video_rows:
            raise ValueError(f"missing exposure review coverage for selected candidate {video_id}")
        try:
            video_rows.sort(key=lambda row: int(row["segment_start_frame"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid exposure segment frame for {video_id}") from exc
        expected_start = 1
        frame_count = int(summary["frame_count"])
        expected_status = summary["video_exposure_status"]
        expected_curve = "log" if expected_status == "underexposed" else "inverse_log"
        allowed_decision = "stable_log" if expected_status == "underexposed" else "tone_only_overexposed"
        for row in video_rows:
            start = int(row["segment_start_frame"])
            end = int(row["segment_end_frame"])
            if start != expected_start or end < start or end > frame_count:
                raise ValueError(
                    f"exposure review for {video_id} must provide gap-free, non-overlapping coverage; "
                    f"expected segment start {expected_start}, got {start}-{end}"
                )
            if row["split"] != summary["split"] or row["exposure_status"] != expected_status:
                raise ValueError(f"exposure review metadata differs from audit for {video_id}:{start}-{end}")
            if row["review_status"] != "REVIEWED":
                raise ValueError(f"exposure review is not REVIEWED for {video_id}:{start}-{end}")
            decision = row["review_decision"]
            if decision not in {allowed_decision, "keep_raw"}:
                raise ValueError(
                    f"invalid exposure review decision {decision!r} for {expected_status} segment "
                    f"{video_id}:{start}-{end}"
                )
            if not row["reviewer"].strip():
                raise ValueError(f"exposure reviewer is required for {video_id}:{start}-{end}")
            row["segment_start_frame"] = start
            row["segment_end_frame"] = end
            if decision == "keep_raw":
                row["exposure_curve"] = "identity"
                row["curve_parameter"] = 0.0
            else:
                if row["exposure_curve"] != expected_curve:
                    raise ValueError(
                        f"{video_id}:{start}-{end} requires exposure_curve={expected_curve}"
                    )
                if row["observed_luma_source"] != "full_segment_visible_luma_median":
                    raise ValueError(
                        f"{video_id}:{start}-{end} must recompute observed_luma_median from every "
                        "visible frame in the reviewed segment"
                    )
                try:
                    observed = float(row["observed_luma_median"])
                    target = float(row["target_luma"])
                    parameter = float(row["curve_parameter"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"non-numeric reviewed exposure parameter for {video_id}:{start}-{end}"
                    ) from exc
                if not 0.0 < parameter <= policy.max_exposure_curve_parameter:
                    raise ValueError(
                        f"curve parameter is outside (0, {policy.max_exposure_curve_parameter:g}] "
                        f"for {video_id}:{start}-{end}"
                    )
                audited_target = float(summary["exposure_target_luma"])
                if not math.isclose(target, audited_target, rel_tol=0.0, abs_tol=1e-3):
                    raise ValueError(
                        f"target_luma must retain the train-only audited target for {video_id}:{start}-{end}"
                    )
                safe_low = _safe_float(summary.get("exposure_safe_low_luma"))
                safe_high = _safe_float(summary.get("exposure_safe_high_luma"))
                if (
                    safe_low is not None
                    and safe_high is not None
                    and safe_high - safe_low > 1e-6
                    and not safe_low < target < safe_high
                ):
                    raise ValueError(
                        f"target_luma must lie strictly inside the audited safety band for "
                        f"{video_id}:{start}-{end}"
                    )
                mapped = float(exposure_curve_value(observed / 255.0, expected_curve, parameter)) * 255.0
                if not math.isclose(mapped, target, rel_tol=0.0, abs_tol=0.25):
                    raise ValueError(
                        f"curve parameter does not map reviewed median to target for {video_id}:{start}-{end}: "
                        f"mapped={mapped:.3f}, target={target:.3f}"
                    )
                row["observed_luma_median"] = observed
                row["target_luma"] = target
                row["curve_parameter"] = parameter
            validated.append(row)
            expected_start = end + 1
        if expected_start != frame_count + 1:
            raise ValueError(
                f"exposure review for {video_id} ends at {expected_start - 1}, expected {frame_count}"
            )
    return validated


def _bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _save_rgb(path, rgb, policy):
    path = Path(path)
    ensure_dir(path.parent)
    Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
        path,
        format="JPEG",
        quality=policy.jpeg_quality,
        subsampling=policy.jpeg_subsampling,
        optimize=False,
    )


def _apply_luma_mapping(rgb, adjusted_luma, policy):
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    luma = 0.2989 * values[:, :, 0] + 0.5870 * values[:, :, 1] + 0.1140 * values[:, :, 2]
    padding = _axis_connected_padding_mask(np.asarray(rgb, dtype=np.uint8), policy.black_threshold)
    adjusted_luma = np.asarray(adjusted_luma(luma), dtype=np.float32)
    if adjusted_luma.shape != luma.shape or not np.isfinite(adjusted_luma).all():
        raise ValueError("luma mapping must return a finite array with the input luma shape")
    adjusted_luma = np.clip(adjusted_luma, 0.0, 1.0)
    delta = adjusted_luma - luma
    min_delta = -values.min(axis=2)
    max_delta = 1.0 - values.max(axis=2)
    safe_delta = np.maximum(np.minimum(delta, max_delta), min_delta)
    adjusted = values + safe_delta[:, :, None]
    adjusted[padding] = values[padding]
    return np.clip(np.rint(adjusted * 255.0), 0.0, 255.0).astype(np.uint8)


def apply_luma_log_curve(rgb, parameter, policy):
    """Brighten underexposed visible pixels with one frozen logarithmic curve."""

    if not 0.0 <= float(parameter) <= policy.max_exposure_curve_parameter:
        raise ValueError("log parameter is outside the frozen policy bound")
    return _apply_luma_mapping(
        rgb,
        lambda luma: exposure_curve_value(luma, "log", parameter),
        policy,
    )


def apply_luma_inverse_log_curve(rgb, parameter, policy):
    """Compress overexposed visible pixels without claiming clipped-detail recovery."""

    if not 0.0 <= float(parameter) <= policy.max_exposure_curve_parameter:
        raise ValueError("inverse-log parameter is outside the frozen policy bound")
    return _apply_luma_mapping(
        rgb,
        lambda luma: exposure_curve_value(luma, "inverse_log", parameter),
        policy,
    )


def apply_luma_gamma(rgb, gamma, policy):
    """Legacy gamma mapper retained for historical audit reproducibility only."""

    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be positive and finite")
    return _apply_luma_mapping(
        rgb,
        lambda luma: np.power(np.clip(luma, 0.0, 1.0), gamma),
        policy,
    )


def _preview_curve_plan(summary, policy, target_override=None):
    status = summary["video_exposure_status"]
    if status not in {"underexposed", "overexposed"}:
        raise ValueError(f"preview requested for non-candidate exposure status: {status}")
    curve = summary.get("planned_exposure_curve") or (
        "log" if status == "underexposed" else "inverse_log"
    )
    parameter = _safe_float(summary.get("planned_exposure_parameter"))
    observed = _safe_float(summary.get("video_luma_median"))
    target = _safe_float(summary.get("exposure_target_luma"))
    if observed is None or target is None:
        raise ValueError(f"missing audited exposure median/target for {summary['video_id']}")
    if target_override is not None:
        target = float(target_override)
        parameter = None
    if parameter is None:
        parameter = fit_exposure_curve_parameter(
            observed,
            target,
            curve,
            max_parameter=policy.max_exposure_curve_parameter,
        )
    return curve, parameter, observed, target


def _select_exposure_preview_samples(sample_rows, count):
    count = max(1, int(count))
    rows = [
        row
        for row in sample_rows
        if row.get("decode_status") == "OK"
        and not _bool(row.get("pure_black"))
        and _safe_int(row.get("frame_id")) is not None
        and _safe_float(row.get("luma_median")) is not None
    ]
    rows.sort(key=lambda row: _safe_int(row["frame_id"]))
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]

    selected_indices = set(_even_indices(len(rows), max(1, count - 2)))
    selected_indices.add(min(range(len(rows)), key=lambda index: float(rows[index]["luma_median"])))
    selected_indices.add(max(range(len(rows)), key=lambda index: float(rows[index]["luma_median"])))
    if len(selected_indices) > count:
        extremes = {
            min(range(len(rows)), key=lambda index: float(rows[index]["luma_median"])),
            max(range(len(rows)), key=lambda index: float(rows[index]["luma_median"])),
        }
        remaining = [index for index in sorted(selected_indices) if index not in extremes]
        keep = set(sorted(extremes))
        keep.update(remaining[: max(0, count - len(keep))])
        selected_indices = keep
    return [rows[index] for index in sorted(selected_indices)]


def _contact_thumbnail(rgb, width, label):
    rgb = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    image_height = width
    canvas = Image.new("RGB", (width, image_height + 24), "white")
    resampling = getattr(Image, "Resampling", Image)
    contained = ImageOps.contain(rgb, (width, image_height), method=resampling.BILINEAR)
    offset = ((width - contained.width) // 2, (image_height - contained.height) // 2)
    canvas.paste(contained, offset)
    ImageDraw.Draw(canvas).text((4, image_height + 5), label, fill="black")
    return canvas


def _write_exposure_contact_sheet(path, pairs, columns=4, thumb_width=224, title=None):
    columns = max(1, int(columns))
    thumb_width = max(64, int(thumb_width))
    label_width = 72
    header_height = 30
    cell_height = thumb_width + 24
    page_count = math.ceil(len(pairs) / columns)
    canvas = Image.new(
        "RGB",
        (label_width + columns * thumb_width, header_height + page_count * 2 * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (6, 8),
        title or "Exposure preview (source pixels are not modified)",
        fill="black",
    )
    for index, pair in enumerate(pairs):
        page = index // columns
        column = index % columns
        x = label_width + column * thumb_width
        before_y = header_height + page * 2 * cell_height
        after_y = before_y + cell_height
        if column == 0:
            draw.text((6, before_y + 8), "BEFORE", fill="black")
            draw.text((6, after_y + 8), "AFTER", fill="black")
        before_label = f"f{pair['frame_id']} L={pair['source_metrics']['luma_median']:.1f}"
        after_label = f"f{pair['frame_id']} L={pair['preview_metrics']['luma_median']:.1f}"
        canvas.paste(_contact_thumbnail(pair["source_rgb"], thumb_width, before_label), (x, before_y))
        canvas.paste(_contact_thumbnail(pair["preview_rgb"], thumb_width, after_label), (x, after_y))
    path = Path(path)
    ensure_dir(path.parent)
    canvas.save(path, format="JPEG", quality=92, subsampling=0, optimize=False)
    return path


def write_exposure_previews(
    audit_dir,
    image_root,
    output_dir,
    video_ids=None,
    frames_per_video=12,
    columns=4,
    thumb_width=224,
    underexposed_target_luma=None,
    overexposed_target_luma=None,
    safe_low_luma=None,
    safe_high_luma=None,
    scan_all_frames=False,
    image_integrity_comparison_summary=None,
    project_root=None,
):
    """Write non-authorizing before/after contact sheets from audit samples."""

    audit_dir = Path(audit_dir).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"preview output_dir must be empty or absent: {output_dir}")
    run_manifest_path = audit_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    audited_root = Path(run_manifest["image_root"]).expanduser().resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    relocation_evidence = validate_relocated_image_root(
        image_root,
        audited_root,
        image_integrity_comparison_summary,
        project_root,
    )
    policy = FrameRecoveryPolicy(**run_manifest["policy"]).validate()
    summaries = _read_csv(audit_dir / "tables" / "video_failure_summary.csv")
    samples = _read_csv(audit_dir / "tables" / "exposure_sample_manifest.csv")
    candidates = [
        row
        for row in summaries
        if row["video_exposure_status"] in {"underexposed", "overexposed"}
    ]
    if video_ids:
        requested = {str(video_id) for video_id in video_ids}
        known = {row["video_id"] for row in candidates}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"requested video_ids are not audited exposure candidates: {unknown}")
        candidates = [row for row in candidates if row["video_id"] in requested]
    if not candidates:
        raise ValueError("no exposure candidates selected for preview")

    ensure_dir(output_dir)
    sheet_dir = ensure_dir(output_dir / "contact_sheets")
    samples_by_video = {}
    for row in samples:
        samples_by_video.setdefault(row["video_id"], []).append(row)
    frame_rows = []
    video_rows = []
    contact_paths = []
    for summary in candidates:
        video_id = summary["video_id"]
        target_override = (
            underexposed_target_luma
            if summary["video_exposure_status"] == "underexposed"
            else overexposed_target_luma
        )
        curve, parameter, observed, target = _preview_curve_plan(
            summary,
            policy,
            target_override=target_override,
        )
        selected_samples = _select_exposure_preview_samples(
            samples_by_video.get(video_id, []),
            frames_per_video,
        )
        image_by_frame = {
            _frame_id(path): path
            for path in (image_root / video_id).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        contact_frame_ids = {int(row["frame_id"]) for row in selected_samples}
        if scan_all_frames:
            evaluation_paths = [image_by_frame[key] for key in sorted(image_by_frame)]
        else:
            evaluation_paths = [
                image_by_frame[frame_id]
                for frame_id in sorted(contact_frame_ids)
                if frame_id in image_by_frame
            ]
        pairs = []
        video_frame_rows = []
        for source_path in evaluation_paths:
            frame_id = _frame_id(source_path)
            source_rgb = _load_rgb(source_path)
            source_metrics = _metrics_from_rgb(source_rgb, policy)
            if classify_failed_frame({"decode_status": "OK", **source_metrics}, policy) == "pure_black":
                continue
            if curve == "log":
                preview_rgb = apply_luma_log_curve(source_rgb, parameter, policy)
            else:
                preview_rgb = apply_luma_inverse_log_curve(source_rgb, parameter, policy)
            preview_metrics = _metrics_from_rgb(preview_rgb, policy)
            if frame_id in contact_frame_ids:
                pairs.append(
                    {
                        "frame_id": frame_id,
                        "source_rgb": source_rgb,
                        "preview_rgb": preview_rgb,
                        "source_metrics": source_metrics,
                        "preview_metrics": preview_metrics,
                    }
                )
            video_frame_rows.append(
                {
                    "split": summary["split"],
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "source_path": str(source_path),
                    "exposure_status": summary["video_exposure_status"],
                    "exposure_curve": curve,
                    "curve_parameter": parameter,
                    "target_luma": target,
                    "source_luma_median": source_metrics["luma_median"],
                    "preview_luma_median": preview_metrics["luma_median"],
                    "source_luma_span_q90_q10": (
                        source_metrics["luma_q90"] - source_metrics["luma_q10"]
                    ),
                    "preview_luma_span_q90_q10": (
                        preview_metrics["luma_q90"] - preview_metrics["luma_q10"]
                    ),
                    "luma_span_retention_ratio": (
                        (preview_metrics["luma_q90"] - preview_metrics["luma_q10"])
                        / max(source_metrics["luma_q90"] - source_metrics["luma_q10"], 1e-6)
                    ),
                    "source_low_saturation_ratio": source_metrics["low_saturation_ratio"],
                    "preview_low_saturation_ratio": preview_metrics["low_saturation_ratio"],
                    "source_high_saturation_ratio": source_metrics["high_saturation_ratio"],
                    "preview_high_saturation_ratio": preview_metrics["high_saturation_ratio"],
                }
            )
        frame_rows.extend(video_frame_rows)
        if not pairs:
            raise ValueError(f"no valid audited exposure samples available for {video_id}")
        contact_path = _write_exposure_contact_sheet(
            sheet_dir / f"{video_id}.jpg",
            pairs,
            columns=columns,
            thumb_width=thumb_width,
            title=(
                f"{summary['video_exposure_status']} | {curve}={parameter:.3f} | "
                f"target L={target:.3f} | preview only"
            ),
        )
        contact_paths.append(contact_path)
        effective_safe_low = (
            float(safe_low_luma)
            if safe_low_luma is not None
            else _safe_float(summary.get("exposure_safe_low_luma"))
        )
        effective_safe_high = (
            float(safe_high_luma)
            if safe_high_luma is not None
            else _safe_float(summary.get("exposure_safe_high_luma"))
        )
        source_frame_median = float(
            np.median([row["source_luma_median"] for row in video_frame_rows])
        )
        preview_frame_median = float(
            np.median([row["preview_luma_median"] for row in video_frame_rows])
        )
        inside_safe_band = (
            effective_safe_low is not None
            and effective_safe_high is not None
            and effective_safe_low <= preview_frame_median <= effective_safe_high
        )
        video_rows.append(
            {
                "split": summary["split"],
                "video_id": video_id,
                "exposure_status": summary["video_exposure_status"],
                "exposure_curve": curve,
                "curve_parameter": parameter,
                "observed_sample_luma_median": observed,
                "target_luma": target,
                "preview_frame_count": len(pairs),
                "safe_low_luma": effective_safe_low,
                "safe_high_luma": effective_safe_high,
                "evaluated_frame_count": len(video_frame_rows),
                "source_frame_median_luma": source_frame_median,
                "preview_frame_median_luma": preview_frame_median,
                "preview_inside_safe_band": inside_safe_band,
                "source_low_saturation_ratio_mean": float(
                    np.mean([row["source_low_saturation_ratio"] for row in video_frame_rows])
                ),
                "preview_low_saturation_ratio_mean": float(
                    np.mean([row["preview_low_saturation_ratio"] for row in video_frame_rows])
                ),
                "source_high_saturation_ratio_mean": float(
                    np.mean([row["source_high_saturation_ratio"] for row in video_frame_rows])
                ),
                "preview_high_saturation_ratio_mean": float(
                    np.mean([row["preview_high_saturation_ratio"] for row in video_frame_rows])
                ),
                "luma_span_retention_ratio_median": float(
                    np.median([row["luma_span_retention_ratio"] for row in video_frame_rows])
                ),
                "contact_sheet": str(contact_path),
            }
        )

    frame_path = _write_csv(
        output_dir / "exposure_preview_frames.csv",
        frame_rows,
        EXPOSURE_PREVIEW_FRAME_FIELDS,
    )
    video_path = _write_csv(
        output_dir / "exposure_preview_videos.csv",
        video_rows,
        EXPOSURE_PREVIEW_VIDEO_FIELDS,
    )
    payload = {
        "preview": "non-authorizing exposure before/after contact sheets",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(shlex.quote(item) for item in sys.argv),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "audit_dir": str(audit_dir),
        "audit_manifest_sha256": _sha256_file(run_manifest_path),
        "source_image_root": str(image_root),
        "audited_source_image_root": str(audited_root),
        "image_root_relocation_evidence": relocation_evidence,
        "source_tree_modified": False,
        "formal_materialization_performed": False,
        "exposure_review_approval_implied": False,
        "selection": "audited temporal samples plus sampled luma extrema",
        "frames_per_video": int(frames_per_video),
        "columns": int(columns),
        "thumb_width": int(thumb_width),
        "preview_target_overrides": {
            "underexposed_target_luma": underexposed_target_luma,
            "overexposed_target_luma": overexposed_target_luma,
        },
        "preview_safe_band_overrides": {
            "safe_low_luma": safe_low_luma,
            "safe_high_luma": safe_high_luma,
        },
        "scan_all_frames": bool(scan_all_frames),
        "video_count": len(video_rows),
        "frame_count": len(frame_rows),
        "contact_sheets": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in contact_paths
        ],
        "outputs": {
            "frames": {"path": str(frame_path), "sha256": _sha256_file(frame_path)},
            "videos": {"path": str(video_path), "sha256": _sha256_file(video_path)},
        },
    }
    manifest_path = output_dir / "preview_manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [frame_path, video_path, manifest_path]


def _sample_flow(flow, x, y, cv2):
    return cv2.remap(
        flow,
        x.astype(np.float32),
        y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _optical_flow_pair(previous_rgb, next_rgb, policy):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for optical-flow materialization") from exc
    previous_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(next_rgb, cv2.COLOR_RGB2GRAY)
    kwargs = dict(
        pyr_scale=policy.farneback_pyr_scale,
        levels=policy.farneback_levels,
        winsize=policy.farneback_winsize,
        iterations=policy.farneback_iterations,
        poly_n=policy.farneback_poly_n,
        poly_sigma=policy.farneback_poly_sigma,
        flags=0,
    )
    forward = cv2.calcOpticalFlowFarneback(previous_gray, next_gray, None, **kwargs)
    backward = cv2.calcOpticalFlowFarneback(next_gray, previous_gray, None, **kwargs)
    height, width = previous_gray.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    next_x = grid_x + forward[:, :, 0]
    next_y = grid_y + forward[:, :, 1]
    backward_at_next = _sample_flow(backward, next_x, next_y, cv2)
    fb_error = np.linalg.norm(forward + backward_at_next, axis=2)
    magnitude = np.linalg.norm(forward, axis=2)
    metrics = {
        "flow_median_magnitude": float(np.median(magnitude)),
        "flow_median_fb_error": float(np.median(fb_error)),
    }
    reliable = (
        metrics["flow_median_magnitude"] <= policy.max_flow_median_magnitude
        and metrics["flow_median_fb_error"] <= policy.max_flow_median_fb_error
    )
    return forward, backward, metrics, reliable


def synthesize_bidirectional(previous_rgb, next_rgb, alpha, forward, backward):
    import cv2

    height, width = previous_rgb.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    previous_x = grid_x - float(alpha) * forward[:, :, 0]
    previous_y = grid_y - float(alpha) * forward[:, :, 1]
    next_x = grid_x - (1.0 - float(alpha)) * backward[:, :, 0]
    next_y = grid_y - (1.0 - float(alpha)) * backward[:, :, 1]
    previous_warp = cv2.remap(
        previous_rgb,
        previous_x.astype(np.float32),
        previous_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    next_warp = cv2.remap(
        next_rgb,
        next_x.astype(np.float32),
        next_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    blended = (1.0 - float(alpha)) * previous_warp.astype(np.float32) + float(alpha) * next_warp.astype(np.float32)
    return np.clip(np.rint(blended), 0.0, 255.0).astype(np.uint8)


def _repair_row(
    *,
    split,
    video_id,
    frame_id,
    relative_path,
    source_path,
    derived_path,
    repair_type,
    policy,
    previous_frame=None,
    next_frame=None,
    alpha=None,
    exposure_curve=None,
    exposure_parameter=None,
    gamma=None,
    exposure_segment_start=None,
    exposure_segment_end=None,
    exposure_review_decision=None,
    flow_metrics=None,
    status="REPAIRED",
    notes="",
):
    source_metrics = inspect_frame(source_path, policy, compute_hash=True)
    derived_metrics = inspect_frame(derived_path, policy, compute_hash=True)
    flow_metrics = flow_metrics or {}
    return {
        "split": split,
        "video_id": video_id,
        "frame_id": frame_id,
        "relative_path": relative_path,
        "source_path": str(source_path),
        "derived_path": str(derived_path),
        "source_sha256": source_metrics.get("original_sha256", ""),
        "derived_sha256": derived_metrics.get("original_sha256", ""),
        "repair_type": repair_type,
        "source_previous_frame": previous_frame,
        "source_next_frame": next_frame,
        "interpolation_alpha": alpha,
        "exposure_curve": exposure_curve,
        "exposure_parameter": exposure_parameter,
        "exposure_gamma": gamma,
        "exposure_segment_start": exposure_segment_start,
        "exposure_segment_end": exposure_segment_end,
        "exposure_review_decision": exposure_review_decision,
        "flow_median_magnitude": flow_metrics.get("flow_median_magnitude"),
        "flow_median_fb_error": flow_metrics.get("flow_median_fb_error"),
        "source_luma_median": source_metrics.get("luma_median"),
        "derived_luma_median": derived_metrics.get("luma_median"),
        "status": status,
        "notes": notes,
    }


def _ensure_safe_output_root(image_root, output_root):
    image_root = Path(image_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == image_root or image_root in output_root.parents:
        raise ValueError("output_root must be separate from and outside the source image_root")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output_root must be empty or absent: {output_root}")
    return ensure_dir(output_root)


def _materialize_mirror(image_root, output_root, link_mode):
    for source_video in sorted(path for path in Path(image_root).iterdir() if path.is_dir()):
        derived_video = Path(output_root) / source_video.name
        if not derived_video.exists() and link_mode == "symlink":
            derived_video.symlink_to(source_video, target_is_directory=True)
            continue
        ensure_dir(derived_video)
        for source_path in sorted(
            path for path in source_video.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ):
            derived_path = derived_video / source_path.name
            if derived_path.exists() or derived_path.is_symlink():
                continue
            if link_mode == "symlink":
                derived_path.symlink_to(source_path)
            elif link_mode == "hardlink":
                os.link(source_path, derived_path)
            elif link_mode == "copy":
                shutil.copy2(source_path, derived_path)
            else:
                raise ValueError("link_mode must be symlink, hardlink, or copy")


def materialize_frame_repairs(
    audit_dir,
    image_root,
    output_root,
    source_review_manifest=None,
    exposure_review_manifest=None,
    layout="sparse_overlay",
    link_mode="hardlink",
    max_videos=None,
    video_ids=None,
    project_root=None,
):
    audit_dir = Path(audit_dir).resolve()
    image_root = Path(image_root).resolve()
    run_manifest = json.loads((audit_dir / "run_manifest.json").read_text(encoding="utf-8"))
    policy = FrameRecoveryPolicy(**run_manifest["policy"]).validate()
    all_summaries = _read_csv(audit_dir / "tables" / "video_failure_summary.csv")
    summaries = list(all_summaries)
    failures = _read_csv(audit_dir / "tables" / "frame_failure_manifest.csv")
    blocks = _read_csv(audit_dir / "tables" / "failure_blocks.csv")
    if Path(run_manifest["image_root"]).resolve() != image_root:
        raise ValueError("image_root differs from the audited source root")
    selected_videos = [row["video_id"] for row in summaries]
    if video_ids:
        requested = {str(video_id) for video_id in video_ids}
        unknown = sorted(requested - set(selected_videos))
        if unknown:
            raise ValueError(f"requested video_ids are absent from the audit: {unknown}")
        selected_videos = [video_id for video_id in selected_videos if video_id in requested]
    if max_videos is not None:
        selected_videos = selected_videos[: max(0, int(max_videos))]
    selected = set(selected_videos)
    summaries = [row for row in summaries if row["video_id"] in selected]
    failures = [row for row in failures if row["video_id"] in selected]
    blocks = [row for row in blocks if row["video_id"] in selected]
    if source_review_manifest is None:
        raise ValueError(
            "source_review_manifest is required before materialization; "
            "run audit_source_presence.py and complete every selected pure-black segment"
        )
    source_review_manifest = Path(source_review_manifest).expanduser().resolve()
    source_review_by_frame = validate_source_review_segments(
        source_review_manifest,
        failures,
    )
    if exposure_review_manifest is None:
        raise ValueError(
            "exposure_review_manifest is required before materialization; complete the generated "
            "exposure_review_template.csv without using BDI labels or validation/test metrics"
        )
    exposure_review_manifest = Path(exposure_review_manifest).expanduser().resolve()
    exposure_segments = validate_exposure_review_segments(
        exposure_review_manifest,
        summaries,
        policy,
        audited_summaries=all_summaries,
    )
    output_root = _ensure_safe_output_root(image_root, output_root)
    failure_by_video_frame = {
        (row["video_id"], int(row["frame_id"])): row for row in failures
    }
    pure_black_frames = {
        key for key, row in failure_by_video_frame.items() if row["failure_type"] == "pure_black"
    }
    repaired_rows = []
    failed_rows = []

    # Exposure changes are permitted only by the completed review manifest.
    # Each approved video/segment uses one frozen log-family parameter; there
    # is no frame-adaptive fitting and no inference from BDI labels/metrics.
    summary_by_video = {row["video_id"]: row for row in summaries}
    for segment in exposure_segments:
        decision = segment["review_decision"]
        if decision == "keep_raw":
            continue
        curve = segment["exposure_curve"]
        parameter = float(segment["curve_parameter"])
        start = int(segment["segment_start_frame"])
        end = int(segment["segment_end_frame"])
        video_id = segment["video_id"]
        summary = summary_by_video[video_id]
        source_video = image_root / video_id
        for source_path in sorted(
            path for path in source_video.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ):
            frame_id = _frame_id(source_path)
            if not start <= frame_id <= end:
                continue
            if (video_id, frame_id) in pure_black_frames:
                continue
            rgb = _load_rgb(source_path)
            if curve == "log":
                derived_rgb = apply_luma_log_curve(rgb, parameter, policy)
            elif curve == "inverse_log":
                derived_rgb = apply_luma_inverse_log_curve(rgb, parameter, policy)
            else:
                raise ValueError(f"unsupported reviewed exposure curve: {curve}")
            relative_path = Path(video_id, source_path.name)
            derived_path = output_root / relative_path
            _save_rgb(derived_path, derived_rgb, policy)
            repaired_rows.append(
                _repair_row(
                    split=summary["split"],
                    video_id=video_id,
                    frame_id=frame_id,
                    relative_path=relative_path.as_posix(),
                    source_path=source_path,
                    derived_path=derived_path,
                    repair_type=f"{summary['video_exposure_status']}_{curve}_tone_normalization",
                    policy=policy,
                    exposure_curve=curve,
                    exposure_parameter=parameter,
                    exposure_segment_start=start,
                    exposure_segment_end=end,
                    exposure_review_decision=decision,
                    notes=segment["review_notes"],
                )
            )

    # Black aligned placeholders are explicitly retained as invalid.  A person
    # may be absent, partly visible, occluded, or outside the face crop in the
    # source frame.  Aligned-neighbor synthesis would erase that distinction.
    for (video_id, frame_id), failure in sorted(failure_by_video_frame.items()):
        if failure["failure_type"] != "pure_black":
            continue
        review = source_review_by_frame[(video_id, frame_id)]
        presence = review["source_presence_status"]
        permission = review["recovery_permission"]
        if presence == "person_present_detection_failure" and permission == "raw_frame_warp_only":
            reason = "raw_frame_warp_required_but_not_implemented"
        elif presence == "person_absent":
            reason = "source_person_absent_keep_invalid"
        else:
            reason = f"source_{presence}_keep_invalid"
        failed_rows.append(
            {
                **failure,
                "materialization_status": "SKIPPED",
                "materialization_reason": reason,
                "flow_median_magnitude": None,
                "flow_median_fb_error": None,
            }
        )

    if layout == "mirror":
        _materialize_mirror(image_root, output_root, link_mode)
    elif layout != "sparse_overlay":
        raise ValueError("layout must be sparse_overlay or mirror")

    repaired_rows.sort(key=lambda row: (row["video_id"], int(row["frame_id"])))
    repair_path = _write_csv(output_root / "_audit" / "repair_manifest.csv", repaired_rows, REPAIR_FIELDS)
    failure_fields = FAILURE_FIELDS + [
        "materialization_status",
        "materialization_reason",
        "flow_median_magnitude",
        "flow_median_fb_error",
    ]
    failure_path = _write_csv(output_root / "_audit" / "repair_failures.csv", failed_rows, failure_fields)
    project_root = Path(project_root or Path(__file__).resolve().parents[2])
    payload = {
        "materialization": "non-destructive aligned frame recovery",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(shlex.quote(item) for item in sys.argv),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "source_image_root": str(image_root),
        "derived_image_root": str(output_root),
        "audit_dir": str(audit_dir),
        "audit_manifest_sha256": _sha256_file(audit_dir / "run_manifest.json"),
        "source_review_manifest": str(source_review_manifest),
        "source_review_manifest_sha256": _sha256_file(source_review_manifest),
        "exposure_review_manifest": str(exposure_review_manifest),
        "exposure_review_manifest_sha256": _sha256_file(exposure_review_manifest),
        "source_tree_modified": False,
        "face_landmark_rerun_performed": False,
        "aligned_black_frame_synthesis_enabled": False,
        "black_frame_recovery_status": "raw_frame_warp_not_implemented",
        "exposure_curve_family": {
            "underexposed": "log",
            "overexposed": "inverse_log",
            "per_frame_parameter_fitting": False,
            "clipped_detail_recovery_claimed": False,
        },
        "reviewed_exposure_segment_count": len(exposure_segments),
        "layout": layout,
        "link_mode": link_mode if layout == "mirror" else "not_applicable",
        "policy": asdict(policy),
        "selected_video_count": len(selected),
        "selected_video_ids": selected_videos,
        "repaired_frame_count": len(repaired_rows),
        "skipped_repair_count": len(failed_rows),
        "repair_type_counts": dict(sorted(Counter(row["repair_type"] for row in repaired_rows).items())),
        "repair_manifest": str(repair_path),
        "repair_failures": str(failure_path),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "implementation_files": {
            "core": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "cli": {
                "path": str(project_root / "scripts" / "audit_frame_recovery.py"),
                "sha256": _sha256_file(project_root / "scripts" / "audit_frame_recovery.py"),
            },
        },
    }
    try:
        import cv2

        payload["opencv"] = cv2.__version__
    except ImportError:
        payload["opencv"] = "unavailable"
    manifest_path = output_root / "_audit" / "materialization_manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [repair_path, failure_path, manifest_path]
