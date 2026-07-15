"""Read-only AU-T0b coordinate-contract audit.

The audit deliberately does not guess a detection-to-aligned coordinate
mapping.  A frame is mappable only when one of the following auditable sources
is available:

1. an explicit per-frame 2x3 affine transform;
2. landmarks re-detected directly on the aligned JPG sequence; or
3. a caller-supplied canonical aligned-space landmark template used to fit a
   per-frame similarity transform.

Independent x/y scaling from the OpenFace detection canvas to 112x112 is not
implemented because it cannot represent the rotation, translation and crop
used by face alignment.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.diagnostics.au_region_tracking import (
    ensure_dir,
    find_openface_csv_files,
    normalize_video_id,
    write_csv_rows,
)


MAPPING_METHODS = {
    "auto",
    "explicit_affine",
    "aligned_redetection",
    "canonical_similarity",
}

TRANSFORM_FIELDS = [
    "split",
    "video_id",
    "image_frame_id",
    "image_index",
    "mapping_method",
    "transform_source",
    "m00",
    "m01",
    "m02",
    "m10",
    "m11",
    "m12",
    "mapping_residual",
    "in_bounds_ratio",
    "mapping_valid",
]

FRAME_FIELDS = [
    "split",
    "video_id",
    "image_frame_id",
    "image_index",
    "image_path",
    "timestamp",
    "confidence",
    "success",
    "yaw",
    "pitch",
    "roll",
    "mapping_method",
    "source_coordinate_system",
    "transform_available",
    "landmark_count",
    "in_bounds_ratio",
    "mapping_residual",
    "mapping_valid",
    "status",
    "issues",
]

MANIFEST_FIELDS = [
    "split",
    "video_id",
    "image_dir",
    "openface_csv",
    "aligned_landmark_csv",
    "transform_csv",
    "mapping_method",
    "source_coordinate_system",
    "image_width",
    "image_height",
    "selected_frame_count",
    "mapped_frame_count",
    "mapping_valid_frame_count",
    "mapping_valid_frame_ratio",
    "transform_available_frame_count",
    "reference_frame_count",
    "median_in_bounds_ratio",
    "p10_in_bounds_ratio",
    "median_mapping_residual",
    "p95_mapping_residual",
    "status",
    "issues",
]

OVERLAY_FIELDS = [
    "split",
    "video_id",
    "image_frame_id",
    "image_path",
    "mapping_method",
    "selection_reason",
    "yaw",
    "pitch",
    "confidence",
    "mapping_residual",
    "in_bounds_ratio",
    "overlay_path",
    "review_status",
    "review_notes",
]

ISSUE_FIELDS = ["split", "video_id", "image_frame_id", "issue_type", "detail"]


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_int(value):
    value = _safe_float(value)
    if value is None or not float(value).is_integer():
        return None
    return int(value)


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.6f}"
    return str(value)


def _normalized_csv_row(row):
    return {str(key).strip(): value for key, value in row.items() if key is not None}


def _indexed_columns(fieldnames, prefix):
    result = {}
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for fieldname in fieldnames or []:
        normalized = str(fieldname).strip()
        match = pattern.match(normalized)
        if match:
            result[int(match.group(1))] = normalized
    return result


def read_landmark_rows(csv_path):
    """Return OpenFace rows keyed by frame id with landmark dictionaries."""
    csv_path = Path(csv_path)
    rows = {}
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(name).strip() for name in (reader.fieldnames or [])]
        x_columns = _indexed_columns(fieldnames, "x")
        y_columns = _indexed_columns(fieldnames, "y")
        landmark_ids = sorted(set(x_columns) & set(y_columns))
        for row_index, raw_row in enumerate(reader):
            row = _normalized_csv_row(raw_row)
            frame_id = _safe_int(row.get("frame"))
            if frame_id is None or frame_id in rows:
                continue
            landmarks = {}
            for landmark_id in landmark_ids:
                x_value = _safe_float(row.get(x_columns[landmark_id]))
                y_value = _safe_float(row.get(y_columns[landmark_id]))
                if x_value is not None and y_value is not None:
                    landmarks[landmark_id] = np.asarray([x_value, y_value], dtype=np.float64)
            rows[frame_id] = {
                "row_index": row_index,
                "frame": frame_id,
                "timestamp": _safe_float(row.get("timestamp")),
                "confidence": _safe_float(row.get("confidence")),
                "success": _safe_int(row.get("success")),
                "pitch": _safe_float(row.get("pose_Rx")),
                "yaw": _safe_float(row.get("pose_Ry")),
                "roll": _safe_float(row.get("pose_Rz")),
                "landmarks": landmarks,
            }
    return rows


def read_affine_rows(csv_path):
    """Read per-frame 2x3 affine transforms.

    Required columns are ``frame,m00,m01,m02,m10,m11,m12``.  The transform
    maps detection-space column vectors to aligned-image pixel coordinates.
    """
    required = ("m00", "m01", "m02", "m10", "m11", "m12")
    rows = {}
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = _normalized_csv_row(raw_row)
            frame_id = _safe_int(row.get("frame"))
            values = [_safe_float(row.get(field)) for field in required]
            if frame_id is None or frame_id in rows or any(value is None for value in values):
                continue
            rows[frame_id] = np.asarray(values, dtype=np.float64).reshape(2, 3)
    return rows


def read_canonical_template(csv_path):
    """Read ``landmark_id,x,y`` target landmarks in aligned-image pixels."""
    landmarks = {}
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = _normalized_csv_row(raw_row)
            landmark_id = _safe_int(row.get("landmark_id"))
            x_value = _safe_float(row.get("x"))
            y_value = _safe_float(row.get("y"))
            if landmark_id is None or x_value is None or y_value is None:
                continue
            landmarks[landmark_id] = np.asarray([x_value, y_value], dtype=np.float64)
    if len(landmarks) < 3:
        raise ValueError("Canonical template must contain at least 3 valid landmark_id,x,y rows.")
    return landmarks


def estimate_similarity_transform(source_points, target_points):
    """Estimate an isotropic scale + rotation + translation 2x3 transform."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source_points and target_points must both have shape [N, 2].")
    if source.shape[0] < 3:
        raise ValueError("At least 3 point pairs are required for a similarity transform.")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("Source landmarks are degenerate; similarity transform is undefined.")

    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular_values, right_t = np.linalg.svd(covariance)
    correction = np.eye(2, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    scale = float(np.sum(singular_values * np.diag(correction)) / source_variance)
    linear = scale * rotation
    translation = target_mean - linear @ source_mean
    return np.concatenate([linear, translation[:, None]], axis=1)


def apply_affine(landmarks, matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("Affine matrix must have shape [2, 3].")
    return {
        landmark_id: matrix[:, :2] @ point + matrix[:, 2]
        for landmark_id, point in landmarks.items()
    }


def landmark_rmse(left, right):
    landmark_ids = sorted(set(left) & set(right))
    if not landmark_ids:
        return None
    squared = [float(np.sum((left[idx] - right[idx]) ** 2)) for idx in landmark_ids]
    return math.sqrt(float(np.mean(squared)))


def in_bounds_ratio(landmarks, width, height):
    if not landmarks or width is None or height is None or width <= 0 or height <= 0:
        return 0.0
    valid = sum(
        1
        for point in landmarks.values()
        if 0.0 <= float(point[0]) < float(width) and 0.0 <= float(point[1]) < float(height)
    )
    return valid / len(landmarks)


def load_split_map(dataset_split_file):
    with Path(dataset_split_file).open("r", encoding="utf-8") as handle:
        split_data = json.load(handle)
    result = {}
    for split in ("train", "val", "test"):
        for item in split_data.get(split, []):
            video_id = normalize_video_id(Path(str(item)).name)
            previous = result.get(video_id)
            if previous is not None and previous != split:
                raise ValueError(f"Video {video_id!r} occurs in both {previous!r} and {split!r}.")
            result[video_id] = split
    return result


def _read_csv_rows(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        return [_normalized_csv_row(row) for row in csv.DictReader(handle)]


def _csv_map(root, require_landmarks=False, require_transform=False):
    if root is None:
        return {}
    root = Path(root)
    if not root.exists():
        return {}
    paths = sorted(root.rglob("*.csv")) if root.is_dir() else [root]
    result = {}
    for path in paths:
        try:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                fields = {str(item).strip() for item in (next(csv.reader(handle), []) or [])}
        except (OSError, UnicodeDecodeError):
            continue
        if require_landmarks and not ({"frame", "x_0", "y_0"} <= fields):
            continue
        if require_transform and not ({"frame", "m00", "m01", "m02", "m10", "m11", "m12"} <= fields):
            continue
        video_id = normalize_video_id(path.stem)
        if video_id in result:
            raise ValueError(f"Duplicate CSVs for video {video_id!r}: {result[video_id]} and {path}")
        result[video_id] = path
    return result


def _resolve_image_path(image_root, video_id, recorded_path):
    recorded = Path(str(recorded_path))
    if recorded.exists():
        return recorded
    image_root = Path(image_root)
    candidates = [
        image_root / f"{video_id}_aligned" / recorded.name,
        image_root / video_id / recorded.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _method_for_video(requested_method, transform_path, aligned_path, canonical_template):
    if requested_method not in MAPPING_METHODS:
        raise ValueError(f"Unsupported mapping method: {requested_method!r}")
    if requested_method != "auto":
        return requested_method
    if transform_path is not None:
        return "explicit_affine"
    if aligned_path is not None:
        return "aligned_redetection"
    if canonical_template is not None:
        return "canonical_similarity"
    return None


def _points_for_similarity(source, target):
    landmark_ids = sorted(set(source) & set(target))
    if len(landmark_ids) < 3:
        return None, None
    return (
        np.stack([source[idx] for idx in landmark_ids], axis=0),
        np.stack([target[idx] for idx in landmark_ids], axis=0),
    )


def _mapping_for_frame(method, raw_landmarks, aligned_landmarks, affine_matrix, template):
    if method == "explicit_affine":
        if not raw_landmarks or affine_matrix is None:
            return None, None, "detection_space", "explicit_affine_csv"
        return apply_affine(raw_landmarks, affine_matrix), affine_matrix, "detection_space", "explicit_affine_csv"
    if method == "aligned_redetection":
        if not aligned_landmarks:
            return None, None, "aligned_image_space", "aligned_openface_csv"
        identity = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        return aligned_landmarks, identity, "aligned_image_space", "aligned_openface_csv"
    if method == "canonical_similarity":
        if not raw_landmarks or template is None:
            return None, None, "detection_space", "canonical_template"
        source_points, target_points = _points_for_similarity(raw_landmarks, template)
        if source_points is None:
            return None, None, "detection_space", "canonical_template"
        try:
            matrix = estimate_similarity_transform(source_points, target_points)
        except ValueError:
            return None, None, "detection_space", "canonical_template"
        return apply_affine(raw_landmarks, matrix), matrix, "detection_space", "canonical_template"
    return None, None, "unknown", ""


def _candidate_update(candidates, category, score, context):
    previous = candidates.get(category)
    if previous is None or score > previous[0]:
        candidates[category] = (score, context)


def _update_overlay_candidates(candidates, context, previous_yaw):
    yaw = context.get("yaw")
    pitch = context.get("pitch")
    confidence = context.get("confidence")
    residual = context.get("mapping_residual")
    pose_magnitude = max(abs(yaw or 0.0), abs(pitch or 0.0))
    _candidate_update(candidates, "high_pose", pose_magnitude, context)
    _candidate_update(candidates, "low_confidence", -(confidence if confidence is not None else -1.0), context)
    frontal_score = -(pose_magnitude + (1.0 - (confidence if confidence is not None else 0.0)))
    _candidate_update(candidates, "frontal_control", frontal_score, context)
    if residual is not None:
        _candidate_update(candidates, "high_mapping_residual", residual, context)
    if yaw is not None and previous_yaw is not None:
        _candidate_update(candidates, "rapid_pose_change", abs(yaw - previous_yaw), context)


def _select_overlay_contexts(video_candidates, max_overlays):
    by_category = defaultdict(list)
    for candidates in video_candidates.values():
        for category, (score, context) in candidates.items():
            by_category[category].append((score, context))
    for category in by_category:
        by_category[category].sort(
            key=lambda item: (-item[0], item[1]["video_id"], item[1]["image_frame_id"])
        )

    categories = [
        "high_pose",
        "rapid_pose_change",
        "low_confidence",
        "high_mapping_residual",
        "frontal_control",
    ]
    selected = []
    seen = set()
    while len(selected) < max(0, int(max_overlays)):
        progressed = False
        for category in categories:
            queue = by_category.get(category, [])
            while queue:
                _score, context = queue.pop(0)
                key = (context["video_id"], context["image_frame_id"])
                if key in seen:
                    continue
                selected.append((category, context))
                seen.add(key)
                progressed = True
                break
            if len(selected) >= max_overlays:
                break
        if not progressed:
            break
    return selected


def _draw_overlay(context, output_path):
    image_path = Path(context["resolved_image_path"])
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    scale = 4
    image = image.resize((image.width * scale, image.height * scale), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)

    mapped = context["mapped_landmarks"]
    reference = context.get("reference_landmarks") or {}
    for point in mapped.values():
        x_value, y_value = float(point[0]) * scale, float(point[1]) * scale
        draw.ellipse((x_value - 3, y_value - 3, x_value + 3, y_value + 3), fill=(255, 70, 70))
    if context["mapping_method"] != "aligned_redetection":
        for point in reference.values():
            x_value, y_value = float(point[0]) * scale, float(point[1]) * scale
            draw.ellipse(
                (x_value - 3, y_value - 3, x_value + 3, y_value + 3),
                outline=(60, 255, 80),
                width=2,
            )

    label = (
        f"{context['video_id']} frame={context['image_frame_id']} "
        f"method={context['mapping_method']} red=mapped green=reference"
    )
    font = ImageFont.load_default()
    text_box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, min(image.width, text_box[2] + 8), text_box[3] + 8), fill=(0, 0, 0))
    draw.text((4, 4), label, fill=(255, 255, 255), font=font)
    ensure_dir(Path(output_path).parent)
    image.save(output_path)


def _percentile(values, percentile):
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.percentile(finite, percentile)) if finite else None


def _issue(split, video_id, frame_id, issue_type, detail=""):
    return {
        "split": split,
        "video_id": video_id,
        "image_frame_id": frame_id,
        "issue_type": issue_type,
        "detail": detail,
    }


def _write_coordinate_report(report_path, manifests, overlay_rows, thresholds):
    status_counts = Counter(row["status"] for row in manifests)
    method_counts = Counter(row["mapping_method"] or "unavailable" for row in manifests)
    valid_ratios = [row["mapping_valid_frame_ratio"] for row in manifests if row["selected_frame_count"]]
    lines = [
        "# AU-T0b Coordinate Contract Audit",
        "",
        f"- Videos inventoried: {len(manifests)}",
        f"- REVIEW_REQUIRED: {status_counts.get('REVIEW_REQUIRED', 0)}",
        f"- FAIL: {status_counts.get('FAIL', 0)}",
        f"- BLOCKED: {status_counts.get('BLOCKED', 0)}",
        f"- Minimum mapping-valid frame ratio: {min(valid_ratios) if valid_ratios else 0.0:.4f}",
        f"- Required mapping-valid frame ratio: {thresholds['min_mapping_valid_ratio']:.4f}",
        f"- Required per-frame landmark in-bounds ratio: {thresholds['min_in_bounds_ratio']:.4f}",
        f"- Train overlays generated: {len(overlay_rows)}",
        "",
        "## Mapping Method Distribution",
        "",
        "| Method | Videos |",
        "|---|---:|",
    ]
    lines.extend(f"| {method} | {count} |" for method, count in sorted(method_counts.items()))
    lines.extend(["", "## Non-reviewable Videos", ""])
    failed = [row for row in manifests if row["status"] != "REVIEW_REQUIRED"]
    if failed:
        lines.extend(["| Video | Split | Status | Method | Valid ratio | Issues |", "|---|---|---|---|---:|---|"])
        for row in failed[:100]:
            lines.append(
                f"| {row['video_id']} | {row['split']} | {row['status']} | "
                f"{row['mapping_method']} | {row['mapping_valid_frame_ratio']:.4f} | {row['issues']} |"
            )
    else:
        lines.append("All videos passed automated coordinate availability and in-bounds checks.")
    lines.extend(
        [
            "",
            "## Decision Boundary",
            "",
            "- `REVIEW_REQUIRED` is not a T0b PASS. Human overlay review must still "
            "confirm correct aligned-face placement.",
            "- Red points are mapped landmarks; green outlines are optional aligned-space re-detection references.",
            "- Validation/test frames are not selected for threshold or overlay tuning by this command.",
            "- Independent width/height scaling is intentionally unsupported.",
            "- Dynamic masks and training remain blocked until the overlay contract passes.",
        ]
    )
    ensure_dir(Path(report_path).parent)
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sha256(path):
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


def write_run_manifest(path, project_root, arguments, input_paths):
    payload = {
        "audit": "AU-T0b coordinate contract",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "command_line": " ".join(shlex.quote(item) for item in sys.argv),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in arguments.items()},
        "inputs": {
            name: {
                "path": str(value),
                "sha256": _sha256(value) if value is not None and Path(value).is_file() else "",
            }
            for name, value in input_paths.items()
        },
        "python": sys.version,
        "numpy": np.__version__,
        "pillow": Image.__version__,
    }
    ensure_dir(Path(path).parent)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return Path(path)


def run_coordinate_contract_audit(
    image_root,
    openface_root,
    frame_contract_summary,
    selected_frame_mapping,
    dataset_split_file,
    output_dir,
    mapping_method="auto",
    transform_root=None,
    aligned_openface_root=None,
    canonical_template_path=None,
    min_mapping_valid_ratio=0.995,
    min_in_bounds_ratio=0.80,
    max_overlays=120,
    max_videos=None,
):
    """Run the read-only T0b audit and return generated artifact paths."""
    image_root = Path(image_root)
    openface_root = Path(openface_root)
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    figures_dir = ensure_dir(output_dir / "figures" / "overlays")

    if mapping_method not in MAPPING_METHODS:
        raise ValueError(f"mapping_method must be one of {sorted(MAPPING_METHODS)}")

    summary_rows = _read_csv_rows(frame_contract_summary)
    selected_rows = _read_csv_rows(selected_frame_mapping)
    split_map = load_split_map(dataset_split_file)
    raw_csv_map = {normalize_video_id(path.stem): path for path in find_openface_csv_files(openface_root)}
    aligned_csv_map = _csv_map(aligned_openface_root, require_landmarks=True)
    transform_csv_map = _csv_map(transform_root, require_transform=True)
    canonical_template = (
        read_canonical_template(canonical_template_path) if canonical_template_path is not None else None
    )

    summary_by_video = {normalize_video_id(row.get("video_id")): row for row in summary_rows}
    selected_by_video = defaultdict(list)
    for row in selected_rows:
        selected_by_video[normalize_video_id(row.get("video_id"))].append(row)
    video_ids = sorted(summary_by_video)
    if max_videos is not None:
        video_ids = video_ids[: max(0, int(max_videos))]

    manifests = []
    frame_outputs = []
    transform_outputs = []
    issues = []
    video_candidates = {}

    for video_id in video_ids:
        summary = summary_by_video[video_id]
        split = split_map.get(video_id, "unknown")
        selected = sorted(
            selected_by_video.get(video_id, []),
            key=lambda row: _safe_int(row.get("selected_position")) or 0,
        )
        raw_csv_path = raw_csv_map.get(video_id)
        aligned_csv_path = aligned_csv_map.get(video_id)
        transform_csv_path = transform_csv_map.get(video_id)
        method = _method_for_video(
            mapping_method,
            transform_csv_path,
            aligned_csv_path,
            canonical_template,
        )
        width = _safe_int(summary.get("image_width"))
        height = _safe_int(summary.get("image_height"))
        video_issue_names = set()

        if summary.get("status") != "PASS":
            video_issue_names.add("t0a_not_passed")
        if split == "unknown":
            video_issue_names.add("split_unknown")
        if raw_csv_path is None:
            video_issue_names.add("missing_openface_csv")
        if method is None:
            video_issue_names.add("mapping_source_unavailable")
        if method == "explicit_affine" and transform_csv_path is None:
            video_issue_names.add("missing_transform_csv")
        if method == "aligned_redetection" and aligned_csv_path is None:
            video_issue_names.add("missing_aligned_landmark_csv")
        if method == "canonical_similarity" and canonical_template is None:
            video_issue_names.add("missing_canonical_template")

        raw_rows = read_landmark_rows(raw_csv_path) if raw_csv_path is not None else {}
        aligned_rows = read_landmark_rows(aligned_csv_path) if aligned_csv_path is not None else {}
        affine_rows = read_affine_rows(transform_csv_path) if transform_csv_path is not None else {}
        method_source_system = {
            "explicit_affine": "detection_space",
            "aligned_redetection": "aligned_image_space",
            "canonical_similarity": "detection_space",
        }.get(method, "unknown")

        mapped_count = 0
        valid_count = 0
        transform_count = 0
        reference_count = 0
        in_bounds_values = []
        residual_values = []
        candidates = {}
        previous_yaw = None

        for selected_row in selected:
            image_frame_id = _safe_int(selected_row.get("image_frame_id"))
            image_index = _safe_int(selected_row.get("image_index"))
            raw_frame_id = _safe_int(selected_row.get("expected_csv_frame"))
            aligned_frame_id = image_index + 1 if image_index is not None else None
            raw_row = raw_rows.get(raw_frame_id) if raw_frame_id is not None else None
            aligned_row = aligned_rows.get(aligned_frame_id) if aligned_frame_id is not None else None
            raw_landmarks = raw_row["landmarks"] if raw_row else {}
            aligned_landmarks = aligned_row["landmarks"] if aligned_row else {}
            affine_matrix = affine_rows.get(raw_frame_id) if raw_frame_id is not None else None
            mapped, matrix, source_system, transform_source = _mapping_for_frame(
                method,
                raw_landmarks,
                aligned_landmarks,
                affine_matrix,
                canonical_template,
            )
            mapped = mapped or {}
            residual = (
                landmark_rmse(mapped, aligned_landmarks)
                if mapped and aligned_landmarks and method != "aligned_redetection"
                else None
            )
            bounds_ratio = in_bounds_ratio(mapped, width, height)
            mapping_valid = bool(
                mapped
                and matrix is not None
                and len(mapped) >= 3
                and bounds_ratio >= float(min_in_bounds_ratio)
            )
            frame_issue_names = []
            if raw_row is None:
                frame_issue_names.append("missing_detection_landmarks")
            if method == "explicit_affine" and affine_matrix is None:
                frame_issue_names.append("missing_affine_transform")
            if method == "aligned_redetection" and aligned_row is None:
                frame_issue_names.append("missing_aligned_landmarks")
            if not mapped:
                frame_issue_names.append("mapping_unavailable")
            elif bounds_ratio < float(min_in_bounds_ratio):
                frame_issue_names.append("low_landmark_in_bounds_ratio")

            if mapped:
                mapped_count += 1
                in_bounds_values.append(bounds_ratio)
            if matrix is not None:
                transform_count += 1
            if aligned_landmarks:
                reference_count += 1
            if mapping_valid:
                valid_count += 1
            if residual is not None:
                residual_values.append(residual)

            image_path = _resolve_image_path(
                image_root,
                video_id,
                selected_row.get("image_path", ""),
            )
            metadata_row = raw_row or aligned_row or {}
            timestamp = metadata_row.get("timestamp")
            confidence = metadata_row.get("confidence")
            success = metadata_row.get("success")
            yaw = metadata_row.get("yaw")
            pitch = metadata_row.get("pitch")
            roll = metadata_row.get("roll")
            frame_status = "REVIEW_REQUIRED" if mapping_valid else "FAIL"
            frame_output = {
                "split": split,
                "video_id": video_id,
                "image_frame_id": image_frame_id,
                "image_index": image_index,
                "image_path": str(image_path),
                "timestamp": timestamp,
                "confidence": confidence,
                "success": success,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "mapping_method": method or "",
                "source_coordinate_system": source_system,
                "transform_available": matrix is not None,
                "landmark_count": len(mapped),
                "in_bounds_ratio": bounds_ratio if mapped else None,
                "mapping_residual": residual,
                "mapping_valid": mapping_valid,
                "status": frame_status,
                "issues": ";".join(sorted(set(frame_issue_names))),
            }
            frame_outputs.append(frame_output)
            matrix_values = matrix.reshape(-1).tolist() if matrix is not None else [None] * 6
            transform_outputs.append(
                {
                    "split": split,
                    "video_id": video_id,
                    "image_frame_id": image_frame_id,
                    "image_index": image_index,
                    "mapping_method": method or "",
                    "transform_source": transform_source,
                    "m00": matrix_values[0],
                    "m01": matrix_values[1],
                    "m02": matrix_values[2],
                    "m10": matrix_values[3],
                    "m11": matrix_values[4],
                    "m12": matrix_values[5],
                    "mapping_residual": residual,
                    "in_bounds_ratio": bounds_ratio if mapped else None,
                    "mapping_valid": mapping_valid,
                }
            )
            for issue_name in frame_issue_names:
                issues.append(_issue(split, video_id, image_frame_id, issue_name))

            if split == "train" and mapping_valid and image_path.exists():
                context = {
                    **frame_output,
                    "resolved_image_path": str(image_path),
                    "mapped_landmarks": mapped,
                    "reference_landmarks": aligned_landmarks,
                }
                _update_overlay_candidates(candidates, context, previous_yaw)
            if yaw is not None:
                previous_yaw = yaw

        selected_count = len(selected)
        valid_ratio = valid_count / selected_count if selected_count else 0.0
        if not selected:
            video_issue_names.add("missing_selected_frame_mapping")
        if valid_ratio < float(min_mapping_valid_ratio):
            video_issue_names.add("mapping_valid_ratio_below_threshold")
        blocking_names = {
            "t0a_not_passed",
            "split_unknown",
            "missing_openface_csv",
            "mapping_source_unavailable",
            "missing_transform_csv",
            "missing_aligned_landmark_csv",
            "missing_canonical_template",
            "missing_selected_frame_mapping",
        }
        if video_issue_names & blocking_names:
            status = "BLOCKED"
        elif valid_ratio < float(min_mapping_valid_ratio):
            status = "FAIL"
        else:
            status = "REVIEW_REQUIRED"

        for issue_name in sorted(video_issue_names):
            issues.append(_issue(split, video_id, "", issue_name))
        manifests.append(
            {
                "split": split,
                "video_id": video_id,
                "image_dir": str(image_root / f"{video_id}_aligned"),
                "openface_csv": str(raw_csv_path or ""),
                "aligned_landmark_csv": str(aligned_csv_path or ""),
                "transform_csv": str(transform_csv_path or ""),
                "mapping_method": method or "",
                "source_coordinate_system": method_source_system,
                "image_width": width,
                "image_height": height,
                "selected_frame_count": selected_count,
                "mapped_frame_count": mapped_count,
                "mapping_valid_frame_count": valid_count,
                "mapping_valid_frame_ratio": valid_ratio,
                "transform_available_frame_count": transform_count,
                "reference_frame_count": reference_count,
                "median_in_bounds_ratio": _percentile(in_bounds_values, 50),
                "p10_in_bounds_ratio": _percentile(in_bounds_values, 10),
                "median_mapping_residual": _percentile(residual_values, 50),
                "p95_mapping_residual": _percentile(residual_values, 95),
                "status": status,
                "issues": ";".join(sorted(video_issue_names)),
            }
        )
        video_candidates[video_id] = candidates

    overlay_rows = []
    for selection_reason, context in _select_overlay_contexts(video_candidates, max_overlays):
        overlay_path = figures_dir / context["video_id"] / f"frame_{int(context['image_frame_id']):06d}.png"
        _draw_overlay(context, overlay_path)
        overlay_rows.append(
            {
                "split": context["split"],
                "video_id": context["video_id"],
                "image_frame_id": context["image_frame_id"],
                "image_path": context["image_path"],
                "mapping_method": context["mapping_method"],
                "selection_reason": selection_reason,
                "yaw": context["yaw"],
                "pitch": context["pitch"],
                "confidence": context["confidence"],
                "mapping_residual": context["mapping_residual"],
                "in_bounds_ratio": context["in_bounds_ratio"],
                "overlay_path": str(overlay_path),
                "review_status": "",
                "review_notes": "",
            }
        )

    manifest_path = tables_dir / "coordinate_mapping_manifest.csv"
    frame_path = tables_dir / "coordinate_frame_summary.csv"
    transforms_path = tables_dir / "coordinate_transforms.csv"
    overlays_path = tables_dir / "overlay_manifest.csv"
    issues_path = tables_dir / "coordinate_contract_issues.csv"
    write_csv_rows(
        manifest_path,
        [{key: _format_value(row.get(key)) for key in MANIFEST_FIELDS} for row in manifests],
        MANIFEST_FIELDS,
    )
    write_csv_rows(
        frame_path,
        [{key: _format_value(row.get(key)) for key in FRAME_FIELDS} for row in frame_outputs],
        FRAME_FIELDS,
    )
    write_csv_rows(
        transforms_path,
        [{key: _format_value(row.get(key)) for key in TRANSFORM_FIELDS} for row in transform_outputs],
        TRANSFORM_FIELDS,
    )
    write_csv_rows(
        overlays_path,
        [{key: _format_value(row.get(key)) for key in OVERLAY_FIELDS} for row in overlay_rows],
        OVERLAY_FIELDS,
    )
    write_csv_rows(issues_path, issues, ISSUE_FIELDS)

    report_path = reports_dir / "coordinate_contract_report.md"
    _write_coordinate_report(
        report_path,
        manifests,
        overlay_rows,
        {
            "min_mapping_valid_ratio": float(min_mapping_valid_ratio),
            "min_in_bounds_ratio": float(min_in_bounds_ratio),
        },
    )
    return [manifest_path, frame_path, transforms_path, overlays_path, issues_path, report_path]
