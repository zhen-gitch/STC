"""Read-only contract audit for landmark-localized semantic RGB regions.

AU/FACS is used only to name the three coarse semantic regions. The audit
reads aligned-space 68-point landmarks, never AU values, labels, predictions,
or checkpoints. It emits candidate geometry for later policy review; it does
not approve a training input policy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.diagnostics.au_coordinate_contract import read_landmark_rows
from src.diagnostics.au_region_tracking import (
    choose_frame_offset,
    extract_frame_id,
    normalize_video_id,
)


REGION_ORDER = ("eye_brow", "nose_cheek", "mouth_lower_face")
CANDIDATE_MODES = ("raw_dynamic", "static_canonical", "stabilized_dynamic")
POLICY_STATUS = "POLICY_UNFROZEN"
EXPECTED_IMAGE_SIZE = (112, 112)
SELECTION_SCHEMA_VERSION = "region_p0b_pilot_selection_v1"

# Zero-based standard 68-point indexing. The nose bridge is diagnostic rather
# than strict because it vertically overlaps the eye/brow band in a strict
# three-rectangle, zero-pixel-overlap partition.
REGION_REQUIRED_LANDMARKS = {
    "eye_brow": tuple(range(17, 27)) + tuple(range(36, 48)),
    "nose_cheek": tuple(range(30, 36)),
    "mouth_lower_face": tuple(range(48, 68)) + tuple(range(5, 12)),
}
REGION_X_EXTENT_LANDMARKS = {
    "eye_brow": REGION_REQUIRED_LANDMARKS["eye_brow"],
    "nose_cheek": (2, 3, 4, 12, 13, 14) + tuple(range(27, 36)),
    "mouth_lower_face": REGION_REQUIRED_LANDMARKS["mouth_lower_face"],
}
NOSE_BRIDGE_LANDMARKS = (27, 28, 29)

SOURCE_FIELDS = [
    "split",
    "subject_id",
    "task_name",
    "video_id",
    "image_dir",
    "landmark_csv",
    "image_frame_count",
    "landmark_row_count",
    "best_frame_offset",
    "joined_frame_count",
    "frame_join_rate",
    "image_width",
    "image_height",
    "image_tree_sha256",
    "landmark_csv_sha256",
    "status",
    "issues",
]

FRAME_FIELDS = [
    "split",
    "subject_id",
    "task_name",
    "video_id",
    "frame_id",
    "image_index",
    "image_path",
    "landmark_csv",
    "landmark_row_index",
    "landmark_frame_id",
    "success",
    "confidence",
    "landmark_count",
    "source_valid",
    "image_width",
    "image_height",
    "candidate_mode",
    "region",
    "x0",
    "y0",
    "x1",
    "y1",
    "box_width",
    "box_height",
    "area_ratio",
    "in_frame_area_ratio",
    "required_landmark_count",
    "required_contained_count",
    "required_containment_ratio",
    "nose_bridge_count",
    "nose_bridge_contained_count",
    "nose_bridge_containment_ratio",
    "adjacent_box_iou",
    "center_shift_ratio",
    "valid",
    "policy_status",
    "failure_reasons",
]

VIDEO_FIELDS = [
    "split",
    "subject_id",
    "task_name",
    "video_id",
    "candidate_mode",
    "region",
    "frame_count",
    "source_valid_count",
    "candidate_valid_count",
    "candidate_valid_ratio",
    "median_area_ratio",
    "median_adjacent_box_iou",
    "p10_adjacent_box_iou",
    "median_center_shift_ratio",
    "p95_center_shift_ratio",
    "policy_status",
]

ISSUE_FIELDS = [
    "split",
    "video_id",
    "frame_id",
    "candidate_mode",
    "region",
    "issue_type",
    "detail",
]

OVERLAY_FIELDS = [
    "split",
    "video_id",
    "frame_id",
    "candidate_mode",
    "image_path",
    "overlay_path",
    "selection_reason",
    "review_status",
]


def _safe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_int(value):
    number = _safe_float(value)
    return int(number) if number is not None and number.is_integer() else None


def _format(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}" if math.isfinite(float(value)) else ""
    return str(value)


def _write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field)) for field in fields})
    return path


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(paths, root):
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(Path(value).resolve() for value in paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
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


def _subject_and_task(video_id):
    subject_match = re.match(r"(\d+_\d+)", str(video_id))
    task_match = re.search(r"_(Freeform|Northwind)(?:_|$)", str(video_id))
    return (
        subject_match.group(1) if subject_match else "",
        task_match.group(1) if task_match else "",
    )


def load_split_map(dataset_split_file):
    payload = json.loads(Path(dataset_split_file).read_text(encoding="utf-8"))
    split_map = {}
    for split in ("train", "val", "test"):
        values = payload.get(split)
        if not isinstance(values, list):
            raise ValueError(f"dataset split JSON has no list for {split!r}")
        for raw_video_id in values:
            video_id = normalize_video_id(raw_video_id)
            if video_id in split_map:
                raise ValueError(f"video appears in multiple split entries: {video_id}")
            split_map[video_id] = split
    return split_map


def _resolve_project_path(value, project_root):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(project_root) / path).resolve()


def load_pilot_selection_manifest(
    selection_manifest,
    dataset_split_file,
    split_map,
    project_root,
    confidence_threshold,
    canonical_sample_step,
    stabilization_window,
    candidate_margin_ratio,
    candidate_modes,
    overlay_frames_per_mode,
):
    """Validate a label-blind, train-only pilot selection and parameter lock."""

    selection_manifest = Path(selection_manifest).expanduser().resolve()
    payload = json.loads(selection_manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("unsupported REGION-P0B selection schema")
    if payload.get("split") != "train":
        raise ValueError("REGION-P0B selection manifest must be physical-train only")
    expected_split_sha = str(payload.get("dataset_split_sha256", ""))
    if expected_split_sha != _sha256_file(dataset_split_file):
        raise ValueError("REGION-P0B dataset split SHA-256 mismatch")

    for key in ("source_evidence", "source_run_manifest"):
        source = payload.get(key)
        if not isinstance(source, dict):
            raise ValueError(f"REGION-P0B selection has no {key}")
        source_path = _resolve_project_path(source.get("path", ""), project_root)
        if not source_path.is_file():
            raise ValueError(f"REGION-P0B selection source is missing: {source_path}")
        if _sha256_file(source_path) != str(source.get("sha256", "")):
            raise ValueError(f"REGION-P0B selection source SHA-256 mismatch: {key}")

    raw_video_ids = payload.get("selected_video_ids")
    raw_subject_ids = payload.get("selected_subject_ids")
    if not isinstance(raw_video_ids, list) or not isinstance(raw_subject_ids, list):
        raise ValueError("REGION-P0B selection ids must be lists")
    video_ids = [normalize_video_id(value) for value in raw_video_ids]
    subject_ids = [str(value) for value in raw_subject_ids]
    if len(video_ids) != len(set(video_ids)) or len(subject_ids) != len(set(subject_ids)):
        raise ValueError("REGION-P0B selection ids must be unique")
    if len(video_ids) != int(payload.get("expected_video_count", -1)):
        raise ValueError("REGION-P0B selected video count mismatch")
    if len(subject_ids) != int(payload.get("expected_subject_count", -1)):
        raise ValueError("REGION-P0B selected subject count mismatch")
    if any(split_map.get(video_id) != "train" for video_id in video_ids):
        raise ValueError("REGION-P0B selection contains a non-train or unknown video")

    tasks_by_subject = defaultdict(set)
    for video_id in video_ids:
        subject_id, task_name = _subject_and_task(video_id)
        tasks_by_subject[subject_id].add(task_name)
    if set(tasks_by_subject) != set(subject_ids):
        raise ValueError("REGION-P0B selected subject ids do not match selected videos")
    if any(tasks != {"Freeform", "Northwind"} for tasks in tasks_by_subject.values()):
        raise ValueError("REGION-P0B must preserve paired Freeform/Northwind tasks")

    expected_counts = {}
    for key in (
        "expected_frame_count",
        "expected_candidate_row_count",
        "expected_overlay_count",
    ):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"REGION-P0B selection has invalid {key}")
        expected_counts[key] = value

    parameters = payload.get("candidate_parameters")
    if not isinstance(parameters, dict):
        raise ValueError("REGION-P0B selection has no candidate_parameters")
    expected_parameters = {
        "confidence_threshold": float(confidence_threshold),
        "canonical_sample_step": int(canonical_sample_step),
        "stabilization_window": int(stabilization_window),
        "candidate_margin_ratio": float(candidate_margin_ratio),
        "candidate_modes": list(candidate_modes),
        "overlay_frames_per_mode": int(overlay_frames_per_mode),
    }
    for key, expected in expected_parameters.items():
        actual = parameters.get(key)
        if isinstance(expected, float):
            try:
                matches = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(
                f"REGION-P0B runtime parameter mismatch for {key}: {actual!r} != {expected!r}"
            )
    return {
        "path": selection_manifest,
        "sha256": _sha256_file(selection_manifest),
        "payload": payload,
        "video_ids": video_ids,
        "expected_counts": expected_counts,
    }


def _landmark_csv_inventory(path):
    try:
        with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            header = {str(value).strip() for value in (reader.fieldnames or [])}
            frame_ids = []
            for raw_row in reader:
                row = {str(key).strip(): value for key, value in raw_row.items() if key is not None}
                frame_ids.append(_safe_int(row.get("frame")))
    except (OSError, UnicodeDecodeError):
        return set(), []
    return header, frame_ids


def _looks_like_landmark_csv(path):
    try:
        with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
            header = {str(value).strip() for value in next(csv.reader(handle), [])}
    except (OSError, UnicodeDecodeError):
        return False
    return "frame" in header and any(value.startswith("x_") for value in header)


def _required_landmark_schema():
    required = {"frame", "success", "confidence"}
    required.update(f"x_{index}" for index in range(68))
    required.update(f"y_{index}" for index in range(68))
    return required


def _find_image_dirs(image_root):
    root = Path(image_root).expanduser()
    if not root.exists():
        return []
    return sorted({path.parent for path in root.rglob("*.jpg")})


def _find_landmark_csvs(landmark_root):
    root = Path(landmark_root).expanduser()
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.csv") if _looks_like_landmark_csv(path))


def _unique_source_map(paths, id_getter):
    result = {}
    duplicates = defaultdict(list)
    for path in paths:
        video_id = normalize_video_id(id_getter(path))
        if video_id in result:
            if not duplicates[video_id]:
                duplicates[video_id].append(result[video_id])
            duplicates[video_id].append(path)
        else:
            result[video_id] = path
    return result, dict(duplicates)


def _image_size(path):
    with Image.open(path) as image:
        image.load()
        return int(image.width), int(image.height)


def _points_from_row(row):
    landmarks = row.get("landmarks", {}) if row else {}
    if len(landmarks) != 68 or any(index not in landmarks for index in range(68)):
        return None
    points = np.stack([landmarks[index] for index in range(68)]).astype(np.float64)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        return None
    return points


def source_validity(row, confidence_threshold=0.8):
    reasons = []
    if row is None:
        reasons.append("missing_landmark_row")
        return False, reasons
    if row.get("success") != 1:
        reasons.append("landmark_success_not_one")
    confidence = _safe_float(row.get("confidence"))
    if confidence is None:
        reasons.append("confidence_non_finite")
    elif confidence < float(confidence_threshold):
        reasons.append("confidence_below_threshold")
    if _points_from_row(row) is None:
        reasons.append("incomplete_or_non_finite_68_landmarks")
    return not reasons, reasons


def _integer_extent(points, landmark_ids):
    selected = points[list(landmark_ids)]
    return (
        int(math.floor(float(selected[:, 0].min()))),
        int(math.floor(float(selected[:, 1].min()))),
        int(math.ceil(float(selected[:, 0].max()))) + 1,
        int(math.ceil(float(selected[:, 1].max()))) + 1,
    )


def _box_area(box):
    if box is None:
        return 0
    x0, y0, x1, y1 = box
    return max(0, int(x1) - int(x0)) * max(0, int(y1) - int(y0))


def _intersection_area(left, right):
    if left is None or right is None:
        return 0
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def box_iou(left, right):
    intersection = _intersection_area(left, right)
    union = _box_area(left) + _box_area(right) - intersection
    return float(intersection / union) if union else None


def _in_frame_area_ratio(box, width, height):
    area = _box_area(box)
    if not area:
        return 0.0
    clipped = (
        max(0, box[0]),
        max(0, box[1]),
        min(int(width), box[2]),
        min(int(height), box[3]),
    )
    return float(_box_area(clipped) / area)


def _containment(points, landmark_ids, box):
    if box is None:
        return 0, len(landmark_ids), 0.0
    selected = points[list(landmark_ids)]
    contained = (
        (selected[:, 0] >= box[0])
        & (selected[:, 0] < box[2])
        & (selected[:, 1] >= box[1])
        & (selected[:, 1] < box[3])
    )
    count = int(contained.sum())
    total = len(landmark_ids)
    return count, total, float(count / total) if total else 1.0


def _candidate_boxes_from_points(points, width, height, margin_ratio=0.0):
    """Build a zero-overlap horizontal partition from semantic anchors."""

    eye_required = points[list(REGION_REQUIRED_LANDMARKS["eye_brow"])]
    nose_required = points[list(REGION_REQUIRED_LANDMARKS["nose_cheek"])]
    mouth_required = points[list(REGION_REQUIRED_LANDMARKS["mouth_lower_face"])]
    eye_bottom = float(eye_required[:, 1].max())
    nose_top = float(nose_required[:, 1].min())
    nose_bottom = float(nose_required[:, 1].max())
    mouth_top = float(mouth_required[:, 1].min())

    issues = []
    if eye_bottom >= nose_top:
        issues.append("eye_nose_semantic_order_conflict")
    if nose_bottom >= mouth_top:
        issues.append("nose_mouth_semantic_order_conflict")
    if issues:
        return {region: None for region in REGION_ORDER}, issues

    eye_nose_boundary = int(math.ceil((eye_bottom + nose_top) / 2.0))
    nose_mouth_boundary = int(math.ceil((nose_bottom + mouth_top) / 2.0))
    extents = {
        region: _integer_extent(points, REGION_X_EXTENT_LANDMARKS[region])
        for region in REGION_ORDER
    }
    margin_x = int(math.ceil(float(width) * float(margin_ratio)))
    margin_y = int(math.ceil(float(height) * float(margin_ratio)))
    boxes = {
        "eye_brow": (
            extents["eye_brow"][0] - margin_x,
            extents["eye_brow"][1] - margin_y,
            extents["eye_brow"][2] + margin_x,
            eye_nose_boundary,
        ),
        "nose_cheek": (
            extents["nose_cheek"][0] - margin_x,
            eye_nose_boundary,
            extents["nose_cheek"][2] + margin_x,
            nose_mouth_boundary,
        ),
        "mouth_lower_face": (
            extents["mouth_lower_face"][0] - margin_x,
            nose_mouth_boundary,
            extents["mouth_lower_face"][2] + margin_x,
            extents["mouth_lower_face"][3] + margin_y,
        ),
    }
    return boxes, []


def evaluate_region_boxes(points, boxes, width, height):
    group_issues = []
    for index, left_region in enumerate(REGION_ORDER):
        for right_region in REGION_ORDER[index + 1 :]:
            if _intersection_area(boxes.get(left_region), boxes.get(right_region)):
                group_issues.append("region_pixel_overlap")
                break

    results = {}
    for region in REGION_ORDER:
        box = boxes.get(region)
        reasons = list(group_issues)
        if box is None or _box_area(box) <= 0:
            reasons.append("missing_or_empty_box")
        in_frame_ratio = _in_frame_area_ratio(box, width, height) if box else 0.0
        if in_frame_ratio < 1.0:
            reasons.append("box_out_of_frame")
        contained, required_count, containment_ratio = _containment(
            points, REGION_REQUIRED_LANDMARKS[region], box
        )
        if containment_ratio < 1.0:
            reasons.append("semantic_containment_conflict")
        bridge_contained, bridge_count, bridge_ratio = (0, 0, None)
        if region == "nose_cheek":
            bridge_contained, bridge_count, bridge_ratio = _containment(
                points, NOSE_BRIDGE_LANDMARKS, box
            )
        results[region] = {
            "box": box,
            "in_frame_area_ratio": in_frame_ratio,
            "required_contained_count": contained,
            "required_landmark_count": required_count,
            "required_containment_ratio": containment_ratio,
            "nose_bridge_contained_count": bridge_contained,
            "nose_bridge_count": bridge_count,
            "nose_bridge_containment_ratio": bridge_ratio,
            "valid": not reasons,
            "failure_reasons": sorted(set(reasons)),
        }
    return results


def build_raw_region_geometry(points, width, height, margin_ratio=0.0):
    boxes, issues = _candidate_boxes_from_points(
        points, width, height, margin_ratio=margin_ratio
    )
    evaluated = evaluate_region_boxes(points, boxes, width, height)
    if issues:
        for region in REGION_ORDER:
            evaluated[region]["valid"] = False
            evaluated[region]["failure_reasons"] = sorted(
                set(evaluated[region]["failure_reasons"] + issues)
            )
    return evaluated


def fit_train_canonical(samples):
    """Fit a static aligned-space template from train-only normalized points."""

    arrays = [np.asarray(sample, dtype=np.float64) for sample in samples]
    if not arrays:
        return None
    if any(array.shape != (68, 2) or not np.isfinite(array).all() for array in arrays):
        raise ValueError("canonical samples must all be finite [68,2] arrays")
    return np.median(np.stack(arrays), axis=0)


def static_region_geometry(points, canonical_points, width, height, margin_ratio=0.0):
    if canonical_points is None:
        empty = {region: None for region in REGION_ORDER}
        evaluated = evaluate_region_boxes(points, empty, width, height)
        for region in REGION_ORDER:
            evaluated[region]["valid"] = False
            evaluated[region]["failure_reasons"] = sorted(
                set(evaluated[region]["failure_reasons"] + ["train_canonical_unavailable"])
            )
        return evaluated
    scaled = np.asarray(canonical_points, dtype=np.float64) * np.asarray([width, height])
    boxes, issues = _candidate_boxes_from_points(
        scaled, width, height, margin_ratio=margin_ratio
    )
    evaluated = evaluate_region_boxes(points, boxes, width, height)
    if issues:
        for region in REGION_ORDER:
            evaluated[region]["valid"] = False
            evaluated[region]["failure_reasons"] = sorted(
                set(evaluated[region]["failure_reasons"] + issues)
            )
    return evaluated


def stabilize_region_sequence(frames, window_size=5):
    """Median-smooth raw boxes within consecutive all-region-valid runs."""

    window_size = int(window_size)
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("stabilization window must be a positive odd integer")
    result = [None] * len(frames)
    segments = []
    start = None
    previous_frame = None
    for index, frame in enumerate(frames):
        all_valid = all(frame["raw"][region]["valid"] for region in REGION_ORDER)
        consecutive = previous_frame is not None and frame["frame_id"] == previous_frame + 1
        if all_valid and (start is None or consecutive):
            if start is None:
                start = index
        else:
            if start is not None:
                segments.append((start, index))
            start = index if all_valid else None
        previous_frame = frame["frame_id"]
    if start is not None:
        segments.append((start, len(frames)))

    radius = window_size // 2
    for segment_start, segment_end in segments:
        for index in range(segment_start, segment_end):
            left = max(segment_start, index - radius)
            right = min(segment_end, index + radius + 1)
            boxes = {}
            for region in REGION_ORDER:
                coordinates = [
                    frames[item]["raw"][region]["box"] for item in range(left, right)
                ]
                boxes[region] = tuple(
                    int(round(value))
                    for value in np.median(np.asarray(coordinates, dtype=np.float64), axis=0)
                )
            result[index] = evaluate_region_boxes(
                frames[index]["points"],
                boxes,
                frames[index]["width"],
                frames[index]["height"],
            )

    for index, frame in enumerate(frames):
        if result[index] is not None:
            continue
        result[index] = {}
        for region in REGION_ORDER:
            raw = frame["raw"][region]
            result[index][region] = {
                **raw,
                "valid": False,
                "failure_reasons": sorted(
                    set(raw["failure_reasons"] + ["stabilization_source_invalid"])
                ),
            }
    return result


def _blank_geometry(reasons):
    return {
        region: {
            "box": None,
            "in_frame_area_ratio": 0.0,
            "required_contained_count": 0,
            "required_landmark_count": len(REGION_REQUIRED_LANDMARKS[region]),
            "required_containment_ratio": 0.0,
            "nose_bridge_contained_count": 0,
            "nose_bridge_count": len(NOSE_BRIDGE_LANDMARKS) if region == "nose_cheek" else 0,
            "nose_bridge_containment_ratio": 0.0 if region == "nose_cheek" else None,
            "valid": False,
            "failure_reasons": list(reasons),
        }
        for region in REGION_ORDER
    }


def _frame_candidate_rows(frame, mode, geometry):
    rows = []
    for region in REGION_ORDER:
        item = geometry[region]
        box = item["box"]
        area_ratio = (
            _box_area(box) / max(frame["width"] * frame["height"], 1) if box else 0.0
        )
        rows.append(
            {
                "split": frame["split"],
                "subject_id": frame["subject_id"],
                "task_name": frame["task_name"],
                "video_id": frame["video_id"],
                "frame_id": frame["frame_id"],
                "image_index": frame["image_index"],
                "image_path": str(frame["image_path"]),
                "landmark_csv": str(frame["landmark_csv"]),
                "landmark_row_index": frame["landmark_row_index"],
                "landmark_frame_id": frame["landmark_frame_id"],
                "success": frame["success"],
                "confidence": frame["confidence"],
                "landmark_count": frame["landmark_count"],
                "source_valid": frame["source_valid"],
                "image_width": frame["width"],
                "image_height": frame["height"],
                "candidate_mode": mode,
                "region": region,
                "x0": box[0] if box else None,
                "y0": box[1] if box else None,
                "x1": box[2] if box else None,
                "y1": box[3] if box else None,
                "box_width": box[2] - box[0] if box else None,
                "box_height": box[3] - box[1] if box else None,
                "area_ratio": area_ratio,
                "in_frame_area_ratio": item["in_frame_area_ratio"],
                "required_landmark_count": item["required_landmark_count"],
                "required_contained_count": item["required_contained_count"],
                "required_containment_ratio": item["required_containment_ratio"],
                "nose_bridge_count": item["nose_bridge_count"],
                "nose_bridge_contained_count": item["nose_bridge_contained_count"],
                "nose_bridge_containment_ratio": item["nose_bridge_containment_ratio"],
                "adjacent_box_iou": None,
                "center_shift_ratio": None,
                "valid": item["valid"],
                "policy_status": POLICY_STATUS,
                "failure_reasons": ";".join(item["failure_reasons"]),
            }
        )
    return rows


def _add_temporal_metrics(rows):
    previous = {}
    for row in rows:
        key = (row["video_id"], row["candidate_mode"], row["region"])
        box = None
        if row["x0"] is not None:
            box = (row["x0"], row["y0"], row["x1"], row["y1"])
        prior = previous.get(key)
        if row["valid"] and prior and prior["frame_id"] + 1 == row["frame_id"]:
            row["adjacent_box_iou"] = box_iou(prior["box"], box)
            current_center = np.asarray(
                [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]
            )
            prior_box = prior["box"]
            prior_center = np.asarray(
                [(prior_box[0] + prior_box[2]) / 2.0, (prior_box[1] + prior_box[3]) / 2.0]
            )
            diagonal = math.hypot(row["image_width"], row["image_height"])
            row["center_shift_ratio"] = float(
                np.linalg.norm(current_center - prior_center) / max(diagonal, 1.0)
            )
        previous[key] = (
            {"frame_id": row["frame_id"], "box": box} if row["valid"] else None
        )


def _summarize_candidates(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["video_id"], row["candidate_mode"], row["region"])].append(row)
    summaries = []
    for (split, video_id, mode, region), values in sorted(grouped.items()):
        subject_id, task_name = _subject_and_task(video_id)
        valid = [row for row in values if row["valid"]]
        area = [row["area_ratio"] for row in valid]
        ious = [row["adjacent_box_iou"] for row in valid if row["adjacent_box_iou"] is not None]
        shifts = [row["center_shift_ratio"] for row in valid if row["center_shift_ratio"] is not None]
        source_valid_count = sum(bool(row["source_valid"]) for row in values)
        summaries.append(
            {
                "split": split,
                "subject_id": subject_id,
                "task_name": task_name,
                "video_id": video_id,
                "candidate_mode": mode,
                "region": region,
                "frame_count": len(values),
                "source_valid_count": source_valid_count,
                "candidate_valid_count": len(valid),
                "candidate_valid_ratio": len(valid) / len(values) if values else 0.0,
                "median_area_ratio": float(np.median(area)) if area else None,
                "median_adjacent_box_iou": float(np.median(ious)) if ious else None,
                "p10_adjacent_box_iou": float(np.quantile(ious, 0.10)) if ious else None,
                "median_center_shift_ratio": float(np.median(shifts)) if shifts else None,
                "p95_center_shift_ratio": float(np.quantile(shifts, 0.95)) if shifts else None,
                "policy_status": POLICY_STATUS,
            }
        )
    return summaries


def _issue_rows(candidate_rows):
    rows = []
    for candidate in candidate_rows:
        for issue in filter(None, candidate["failure_reasons"].split(";")):
            rows.append(
                {
                    "split": candidate["split"],
                    "video_id": candidate["video_id"],
                    "frame_id": candidate["frame_id"],
                    "candidate_mode": candidate["candidate_mode"],
                    "region": candidate["region"],
                    "issue_type": issue,
                    "detail": "",
                }
            )
    return rows


def _select_overlay_groups(candidate_rows, limit_per_mode):
    grouped = defaultdict(list)
    for row in candidate_rows:
        if row["split"] == "train":
            grouped[(row["candidate_mode"], row["video_id"], row["frame_id"])].append(row)

    def risk_key(item):
        key, rows = item
        invalid = not all(row["valid"] for row in rows)
        shifts = [
            float(row["center_shift_ratio"])
            for row in rows
            if row["center_shift_ratio"] is not None
        ]
        ious = [
            float(row["adjacent_box_iou"])
            for row in rows
            if row["adjacent_box_iou"] is not None
        ]
        confidences = [
            float(row["confidence"])
            for row in rows
            if row["confidence"] is not None
        ]
        return (
            0 if invalid else 1,
            -max(shifts, default=-1.0),
            min(ious, default=2.0),
            min(confidences, default=2.0),
            key[1],
            key[2],
        )

    representatives = {}
    by_video = defaultdict(list)
    for key, rows in grouped.items():
        by_video[(key[0], key[1])].append((key, rows))
    for mode_video, groups in by_video.items():
        representatives[mode_video] = min(groups, key=risk_key)

    by_mode = defaultdict(list)
    for (mode, _video_id), representative in representatives.items():
        by_mode[mode].append(representative)

    selected = []
    for mode in sorted(by_mode):
        for _key, rows in sorted(by_mode[mode], key=risk_key)[: max(0, int(limit_per_mode))]:
            reason = "valid_control" if all(row["valid"] for row in rows) else "invalid_candidate"
            selected.append((rows, reason))
    return selected


def _write_overlays(output_dir, candidate_rows, limit_per_mode):
    colors = {
        "eye_brow": (0, 180, 0),
        "nose_cheek": (0, 100, 255),
        "mouth_lower_face": (220, 40, 40),
    }
    rows = []
    for group, reason in _select_overlay_groups(candidate_rows, limit_per_mode):
        first = group[0]
        try:
            image = Image.open(first["image_path"]).convert("RGB")
        except (OSError, ValueError):
            continue
        draw = ImageDraw.Draw(image)
        for candidate in group:
            if candidate["x0"] is None:
                continue
            box = (candidate["x0"], candidate["y0"], candidate["x1"], candidate["y1"])
            color = colors[candidate["region"]]
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 1, max(0, box[1] + 1)), candidate["region"], fill=color)
        overlay_path = (
            Path(output_dir)
            / "overlays"
            / first["candidate_mode"]
            / f"{first['video_id']}_f{int(first['frame_id']):06d}.jpg"
        )
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(overlay_path, quality=95)
        rows.append(
            {
                "split": first["split"],
                "video_id": first["video_id"],
                "frame_id": first["frame_id"],
                "candidate_mode": first["candidate_mode"],
                "image_path": first["image_path"],
                "overlay_path": str(overlay_path),
                "selection_reason": reason,
                "review_status": "PENDING_HUMAN_REVIEW",
            }
        )
    return rows


def _write_report(path, source_rows, candidate_rows, summaries, issue_rows):
    status_counts = Counter(row["status"] for row in source_rows)
    valid_counts = Counter()
    total_counts = Counter()
    for row in candidate_rows:
        key = (row["candidate_mode"], row["region"])
        total_counts[key] += 1
        valid_counts[key] += int(bool(row["valid"]))
    lines = [
        "# Landmark RGB Region Contract Audit",
        "",
        f"- Policy status: `{POLICY_STATUS}`",
        f"- Videos: {len(source_rows)}",
        f"- Source PASS/FAIL/BLOCKED: {status_counts['PASS']}/{status_counts['FAIL']}/{status_counts['BLOCKED']}",
        f"- Candidate rows: {len(candidate_rows)}",
        f"- Issue rows: {len(issue_rows)}",
        "",
        "## Candidate Validity",
        "",
        "| Mode | Region | Valid | Total | Ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for key in sorted(total_counts):
        valid = valid_counts[key]
        total = total_counts[key]
        lines.append(f"| {key[0]} | {key[1]} | {valid} | {total} | {valid / total:.6f} |")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- This output is candidate geometry only; it does not approve a region policy or training input.",
            "- AU values, BDI labels, predictions, and checkpoints are not read.",
            "- Static canonical geometry is fit only from physical-train landmarks.",
            "- Validation/test rows are report-only and cannot tune geometry or thresholds.",
            "- Nose-bridge landmarks 27-29 are diagnostic, not strict containment anchors.",
            "- Any overlap or required-anchor containment failure is invalid rather than clipped or relaxed.",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _inventory_sources(split_map, image_root, landmark_root, video_ids, max_videos):
    image_map, image_duplicates = _unique_source_map(_find_image_dirs(image_root), lambda p: p.name)
    landmark_map, landmark_duplicates = _unique_source_map(
        _find_landmark_csvs(landmark_root), lambda p: p.stem
    )
    selected = sorted(split_map)
    if video_ids:
        requested = {normalize_video_id(value) for value in video_ids}
        unknown = sorted(requested - set(split_map))
        if unknown:
            raise ValueError(f"requested video ids are absent from split: {unknown}")
        selected = [video_id for video_id in selected if video_id in requested]
    if max_videos is not None:
        selected = selected[: max(0, int(max_videos))]

    records = []
    for video_id in selected:
        image_dir = image_map.get(video_id)
        landmark_csv = landmark_map.get(video_id)
        frame_paths = sorted(image_dir.glob("*.jpg")) if image_dir else []
        image_ids = [extract_frame_id(path) for path in frame_paths]
        landmark_header, raw_landmark_frame_ids = (
            _landmark_csv_inventory(landmark_csv) if landmark_csv else (set(), [])
        )
        landmark_rows = read_landmark_rows(landmark_csv) if landmark_csv else {}
        offset, joined = choose_frame_offset(image_ids, landmark_rows.keys())
        width = height = None
        image_decode_failure_count = 0
        image_size_mismatch_count = 0
        if frame_paths:
            for frame_path in frame_paths:
                try:
                    frame_width, frame_height = _image_size(frame_path)
                except (OSError, ValueError):
                    image_decode_failure_count += 1
                    continue
                if width is None:
                    width, height = frame_width, frame_height
                if (frame_width, frame_height) != EXPECTED_IMAGE_SIZE:
                    image_size_mismatch_count += 1
        issues = []
        if image_dir is None:
            issues.append("missing_image_directory")
        if landmark_csv is None:
            issues.append("missing_landmark_csv")
        if any(value is None for value in image_ids):
            issues.append("unparseable_image_frame_id")
        if image_ids and len(set(image_ids)) != len(image_ids):
            issues.append("duplicate_image_frame_id")
        parsed_image_ids = [value for value in image_ids if value is not None]
        if any(right <= left for left, right in zip(parsed_image_ids, parsed_image_ids[1:])):
            issues.append("non_monotonic_image_frame_id")
        if landmark_csv and not _required_landmark_schema().issubset(landmark_header):
            issues.append("landmark_schema_mismatch")
        if any(value is None for value in raw_landmark_frame_ids):
            issues.append("unparseable_landmark_frame_id")
        parsed_landmark_ids = [
            value for value in raw_landmark_frame_ids if value is not None
        ]
        if len(set(parsed_landmark_ids)) != len(parsed_landmark_ids):
            issues.append("duplicate_landmark_frame_id")
        if any(
            right <= left
            for left, right in zip(parsed_landmark_ids, parsed_landmark_ids[1:])
        ):
            issues.append("non_monotonic_landmark_frame_id")
        if frame_paths and len(frame_paths) != len(raw_landmark_frame_ids):
            issues.append("image_landmark_row_count_mismatch")
        if video_id in image_duplicates:
            issues.append("duplicate_image_directory")
        if video_id in landmark_duplicates:
            issues.append("duplicate_landmark_csv")
        if image_decode_failure_count:
            issues.append("image_decode_failure")
        if image_size_mismatch_count:
            issues.append("unexpected_image_size")
        join_rate = joined / len(frame_paths) if frame_paths else 0.0
        if (
            frame_paths
            and landmark_rows
            and (
                join_rate < 1.0
                or joined != len(frame_paths)
                or joined != len(raw_landmark_frame_ids)
            )
        ):
            issues.append("frame_join_not_exact")
        blocking = image_dir is None or landmark_csv is None or width is None or height is None
        status = "BLOCKED" if blocking else "FAIL" if issues else "PASS"
        subject_id, task_name = _subject_and_task(video_id)
        records.append(
            {
                "split": split_map[video_id],
                "subject_id": subject_id,
                "task_name": task_name,
                "video_id": video_id,
                "image_dir": image_dir,
                "landmark_csv": landmark_csv,
                "frame_paths": frame_paths,
                "image_ids": image_ids,
                "landmark_rows": landmark_rows,
                "landmark_row_count": len(raw_landmark_frame_ids),
                "best_frame_offset": offset,
                "joined_frame_count": joined,
                "frame_join_rate": join_rate,
                "width": width,
                "height": height,
                "status": status,
                "issues": issues,
            }
        )
    return records


def _canonical_from_records(records, confidence_threshold, sample_step):
    samples = []
    for record in records:
        if record["split"] != "train" or record["status"] != "PASS":
            continue
        width, height = record["width"], record["height"]
        for index, frame_id in enumerate(sorted(record["landmark_rows"])):
            if index % max(1, int(sample_step)):
                continue
            row = record["landmark_rows"][frame_id]
            valid, _reasons = source_validity(row, confidence_threshold)
            if not valid:
                continue
            points = _points_from_row(row)
            raw_geometry = build_raw_region_geometry(points, width, height)
            if not all(raw_geometry[region]["valid"] for region in REGION_ORDER):
                continue
            samples.append(points / np.asarray([width, height], dtype=np.float64))
    return fit_train_canonical(samples), len(samples)


def _process_record(
    record,
    canonical_points,
    confidence_threshold,
    stabilization_window,
    candidate_margin_ratio,
    modes,
):
    frames = []
    if record["status"] == "BLOCKED":
        return []
    offset = record["best_frame_offset"]
    for image_index, (image_path, frame_id) in enumerate(
        zip(record["frame_paths"], record["image_ids"])
    ):
        landmark_frame_id = frame_id + offset if frame_id is not None and offset is not None else None
        row = record["landmark_rows"].get(landmark_frame_id)
        valid, reasons = source_validity(row, confidence_threshold)
        if record["status"] != "PASS":
            valid = False
            reasons = [*reasons, "source_contract_not_pass"]
        points = _points_from_row(row)
        raw = (
            build_raw_region_geometry(
                points,
                record["width"],
                record["height"],
                margin_ratio=candidate_margin_ratio,
            )
            if valid
            else _blank_geometry(reasons)
        )
        static = (
            static_region_geometry(
                points,
                canonical_points,
                record["width"],
                record["height"],
                margin_ratio=candidate_margin_ratio,
            )
            if valid
            else _blank_geometry(reasons)
        )
        frames.append(
            {
                "split": record["split"],
                "subject_id": record["subject_id"],
                "task_name": record["task_name"],
                "video_id": record["video_id"],
                "frame_id": frame_id,
                "image_index": image_index,
                "image_path": image_path,
                "landmark_csv": record["landmark_csv"],
                "landmark_row_index": row.get("row_index") if row else None,
                "landmark_frame_id": landmark_frame_id,
                "success": row.get("success") if row else None,
                "confidence": row.get("confidence") if row else None,
                "landmark_count": len(row.get("landmarks", {})) if row else 0,
                "source_valid": valid,
                "width": record["width"],
                "height": record["height"],
                "points": points,
                "raw": raw,
                "static": static,
            }
        )
    stabilized = stabilize_region_sequence(frames, stabilization_window) if frames else []
    rows = []
    for index, frame in enumerate(frames):
        geometry_by_mode = {
            "raw_dynamic": frame["raw"],
            "static_canonical": frame["static"],
            "stabilized_dynamic": stabilized[index],
        }
        for mode in modes:
            rows.extend(_frame_candidate_rows(frame, mode, geometry_by_mode[mode]))
    return rows


def run_landmark_rgb_region_contract(
    dataset_split_file,
    image_root,
    aligned_landmark_root,
    output_dir,
    video_ids=None,
    max_videos=None,
    confidence_threshold=0.8,
    canonical_sample_step=30,
    stabilization_window=5,
    candidate_margin_ratio=0.0,
    candidate_modes=None,
    overlay_frames_per_mode=2,
    project_root=None,
    selection_manifest=None,
):
    """Generate candidate region-contract artifacts without approving policy."""

    confidence_threshold = float(confidence_threshold)
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be in [0,1]")
    candidate_margin_ratio = float(candidate_margin_ratio)
    if not 0.0 <= candidate_margin_ratio <= 0.25:
        raise ValueError("candidate margin ratio must be in [0,0.25]")
    modes = tuple(candidate_modes or CANDIDATE_MODES)
    if not modes or any(mode not in CANDIDATE_MODES for mode in modes):
        raise ValueError(f"candidate modes must be selected from {CANDIDATE_MODES}")
    if len(set(modes)) != len(modes):
        raise ValueError("candidate modes must not contain duplicates")

    dataset_split_file = Path(dataset_split_file).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    aligned_landmark_root = Path(aligned_landmark_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()

    split_map = load_split_map(dataset_split_file)
    selection = None
    if selection_manifest is not None:
        if video_ids:
            raise ValueError("selection manifest cannot be combined with video ids")
        if max_videos is not None:
            raise ValueError("selection manifest cannot be combined with max videos")
        selection = load_pilot_selection_manifest(
            selection_manifest=selection_manifest,
            dataset_split_file=dataset_split_file,
            split_map=split_map,
            project_root=project_root,
            confidence_threshold=confidence_threshold,
            canonical_sample_step=canonical_sample_step,
            stabilization_window=stabilization_window,
            candidate_margin_ratio=candidate_margin_ratio,
            candidate_modes=modes,
            overlay_frames_per_mode=overlay_frames_per_mode,
        )
        video_ids = selection["video_ids"]
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _inventory_sources(
        split_map, image_root, aligned_landmark_root, video_ids, max_videos
    )
    if selection is not None:
        expected = selection["expected_counts"]
        frame_count = sum(len(record["frame_paths"]) for record in records)
        if frame_count != expected["expected_frame_count"]:
            raise ValueError(
                f"REGION-P0B frame count mismatch: {frame_count} != "
                f"{expected['expected_frame_count']}"
            )
    canonical, canonical_count = _canonical_from_records(
        records, confidence_threshold, canonical_sample_step
    )
    candidate_rows = []
    for record in records:
        candidate_rows.extend(
            _process_record(
                record,
                canonical,
                confidence_threshold,
                stabilization_window,
                candidate_margin_ratio,
                modes,
            )
        )
    candidate_rows.sort(
        key=lambda row: (
            row["split"],
            row["video_id"],
            row["frame_id"] if row["frame_id"] is not None else -1,
            modes.index(row["candidate_mode"]),
            REGION_ORDER.index(row["region"]),
        )
    )
    _add_temporal_metrics(candidate_rows)
    if selection is not None:
        expected_rows = selection["expected_counts"]["expected_candidate_row_count"]
        if len(candidate_rows) != expected_rows:
            raise ValueError(
                f"REGION-P0B candidate row count mismatch: {len(candidate_rows)} != "
                f"{expected_rows}"
            )
    summaries = _summarize_candidates(candidate_rows)
    issues = _issue_rows(candidate_rows)

    source_rows = []
    for record in records:
        image_tree_sha = (
            _tree_sha256(record["frame_paths"], record["image_dir"])
            if record["frame_paths"] and record["image_dir"]
            else ""
        )
        landmark_sha = (
            _sha256_file(record["landmark_csv"]) if record["landmark_csv"] else ""
        )
        source_rows.append(
            {
                "split": record["split"],
                "subject_id": record["subject_id"],
                "task_name": record["task_name"],
                "video_id": record["video_id"],
                "image_dir": str(record["image_dir"] or ""),
                "landmark_csv": str(record["landmark_csv"] or ""),
                "image_frame_count": len(record["frame_paths"]),
                "landmark_row_count": record["landmark_row_count"],
                "best_frame_offset": record["best_frame_offset"],
                "joined_frame_count": record["joined_frame_count"],
                "frame_join_rate": record["frame_join_rate"],
                "image_width": record["width"],
                "image_height": record["height"],
                "image_tree_sha256": image_tree_sha,
                "landmark_csv_sha256": landmark_sha,
                "status": record["status"],
                "issues": ";".join(record["issues"]),
            }
        )

    tables_dir = output_dir / "tables"
    source_path = _write_csv(tables_dir / "region_source_manifest.csv", source_rows, SOURCE_FIELDS)
    frame_path = _write_csv(
        tables_dir / "region_frame_candidates.csv", candidate_rows, FRAME_FIELDS
    )
    video_path = _write_csv(
        tables_dir / "region_video_summary.csv", summaries, VIDEO_FIELDS
    )
    issue_path = _write_csv(tables_dir / "region_issues.csv", issues, ISSUE_FIELDS)
    overlay_rows = _write_overlays(output_dir, candidate_rows, overlay_frames_per_mode)
    if selection is not None:
        expected_overlays = selection["expected_counts"]["expected_overlay_count"]
        if len(overlay_rows) != expected_overlays:
            raise ValueError(
                f"REGION-P0B overlay count mismatch: {len(overlay_rows)} != "
                f"{expected_overlays}"
            )
    overlay_path = _write_csv(
        tables_dir / "region_overlay_review.csv", overlay_rows, OVERLAY_FIELDS
    )
    report_path = _write_report(
        output_dir / "reports" / "region_contract_report.md",
        source_rows,
        candidate_rows,
        summaries,
        issues,
    )

    implementation_files = {
        "core": Path(__file__).resolve(),
        "cli": project_root / "scripts" / "audit_landmark_rgb_region_contract.py",
    }
    if selection is not None:
        implementation_files["selection_manifest"] = selection["path"]
    output_paths = [source_path, frame_path, video_path, issue_path, overlay_path, report_path]
    payload = {
        "audit": (
            "REGION-P0B landmark-localized RGB pilot candidate contract"
            if selection is not None
            else "REGION-P0A landmark-localized RGB candidate contract"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(shlex.quote(value) for value in sys.argv),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "policy_status": POLICY_STATUS,
        "training_authorized": False,
        "data_materialization_performed": False,
        "labels_read": False,
        "au_values_read": False,
        "predictions_or_checkpoints_read": False,
        "candidate_modes": list(modes),
        "region_order": list(REGION_ORDER),
        "required_landmarks": {
            key: list(value) for key, value in REGION_REQUIRED_LANDMARKS.items()
        },
        "x_extent_landmarks": {
            key: list(value) for key, value in REGION_X_EXTENT_LANDMARKS.items()
        },
        "diagnostic_nose_bridge_landmarks": list(NOSE_BRIDGE_LANDMARKS),
        "confidence_threshold": confidence_threshold,
        "canonical_sample_step": int(canonical_sample_step),
        "canonical_train_sample_count": canonical_count,
        "stabilization_window": int(stabilization_window),
        "candidate_margin_ratio": candidate_margin_ratio,
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": {
            "dataset_split_file": {
                "path": str(dataset_split_file),
                "sha256": _sha256_file(dataset_split_file),
            },
            "image_root": str(image_root),
            "aligned_landmark_root": str(aligned_landmark_root),
            "selected_video_count": len(records),
        },
        "implementation_files": {
            key: {"path": str(path), "sha256": _sha256_file(path)}
            for key, path in implementation_files.items()
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256_file(path)}
            for path in output_paths
        },
    }
    if selection is not None:
        selection_payload = selection["payload"]
        payload["pilot_selection"] = {
            "path": str(selection["path"]),
            "sha256": selection["sha256"],
            "schema_version": selection_payload["schema_version"],
            "selected_subject_ids": selection_payload["selected_subject_ids"],
            "selected_video_ids": selection_payload["selected_video_ids"],
            "normalized_selected_video_ids": selection["video_ids"],
            "selection_algorithm": selection_payload.get("selection_algorithm"),
            "source_evidence": selection_payload["source_evidence"],
            "source_run_manifest": selection_payload["source_run_manifest"],
            **selection["expected_counts"],
        }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return [*output_paths, manifest_path]
