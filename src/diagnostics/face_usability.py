"""Phase-1 read-only audit for frame-level face usability.

This module measures pixels and aligned-space 68-point landmark geometry for
every frame.  It deliberately does not freeze thresholds or approve frames for
training.  Train-only extreme examples are exported for manual review first.
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
from src.diagnostics.au_region_tracking import extract_frame_id
from src.diagnostics.frame_recovery import (
    FrameRecoveryPolicy,
    _metrics_from_rgb,
    validate_relocated_image_root,
)


FRAME_FIELDS = [
    "split",
    "subject_id",
    "video_id",
    "task_name",
    "frame_id",
    "frame_index",
    "timestamp",
    "image_path",
    "decode_status",
    "source_presence_status",
    "recovery_permission",
    "aligned_failure_status",
    "exposure_status",
    "nonblack_ratio",
    "visible_ratio",
    "global_mean_luma",
    "global_std_luma",
    "low_saturation_ratio",
    "high_saturation_ratio",
    "blur_score",
    "gradient_energy",
    "landmark_success",
    "confidence",
    "landmark_count",
    "landmark_in_frame_ratio",
    "face_hull_coverage",
    "face_hull_visible_ratio",
    "bbox_area_ratio",
    "center_offset_x",
    "center_offset_y",
    "eye_distance_ratio",
    "landmark_pose_yaw_deg",
    "landmark_pose_pitch_deg",
    "landmark_pose_roll_deg",
    "landmark_pose_solver",
    "landmark_pose_min_depth",
    "landmark_pose_reprojection_rmse",
    "transform_residual",
    "landmark_jump",
    "face_usability_status",
    "exclusion_reasons",
    "review_status",
]

VIDEO_FIELDS = [
    "split",
    "video_id",
    "subject_id",
    "task_name",
    "frame_count",
    "decode_ok_count",
    "landmark_success_count",
    "landmark_success_ratio",
    "person_absent_count",
    "aligned_failure_pending_raw_warp_count",
    "face_present_low_quality_count",
    "pending_threshold_review_count",
    "median_confidence",
    "median_landmark_in_frame_ratio",
    "median_face_hull_coverage",
    "median_face_hull_visible_ratio",
    "median_blur_score",
    "median_abs_yaw_deg",
    "median_abs_pitch_deg",
    "median_pose_reprojection_rmse",
    "median_transform_residual",
    "median_landmark_jump",
]

DISTRIBUTION_FIELDS = [
    "scope",
    "split",
    "task_name",
    "metric",
    "count",
    "mean",
    "std",
    "min",
    "q01",
    "q05",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "q95",
    "q99",
    "max",
]

STATUS_FIELDS = ["scope", "split", "task_name", "face_usability_status", "count"]

CONTACT_FIELDS = [
    "selection_reason",
    "rank",
    "split",
    "video_id",
    "frame_id",
    "image_path",
    "metric_name",
    "metric_value",
    "contact_sheet",
    "review_status",
    "review_label",
    "review_notes",
]

CANONICAL_FIELDS = ["landmark_id", "x_normalized", "y_normalized"]

DISTRIBUTION_METRICS = [
    "nonblack_ratio",
    "visible_ratio",
    "global_mean_luma",
    "global_std_luma",
    "low_saturation_ratio",
    "high_saturation_ratio",
    "blur_score",
    "gradient_energy",
    "confidence",
    "landmark_in_frame_ratio",
    "face_hull_coverage",
    "face_hull_visible_ratio",
    "bbox_area_ratio",
    "center_offset_x",
    "center_offset_y",
    "eye_distance_ratio",
    "landmark_pose_yaw_deg",
    "landmark_pose_pitch_deg",
    "landmark_pose_roll_deg",
    "landmark_pose_min_depth",
    "landmark_pose_reprojection_rmse",
    "transform_residual",
    "landmark_jump",
]

CONTACT_REASONS = {
    "abs_yaw_high": ("landmark_pose_yaw_deg", "abs_high"),
    "abs_pitch_high": ("landmark_pose_pitch_deg", "abs_high"),
    "hull_coverage_low": ("face_hull_coverage", "low"),
    "hull_visible_low": ("face_hull_visible_ratio", "low"),
    "in_frame_ratio_low": ("landmark_in_frame_ratio", "low"),
    "confidence_low": ("confidence", "low"),
    "blur_low": ("blur_score", "low"),
    "transform_residual_high": ("transform_residual", "high"),
    "landmark_jump_high": ("landmark_jump", "high"),
    "frontal_control": ("frontal_control_score", "low"),
    "visible_landmark_failure": ("confidence", "low"),
}

POSE_LANDMARK_IDS = [30, 8, 36, 45, 48, 54]
POSE_MODEL_POINTS = np.asarray(
    [
        (0.0, 0.0, 0.0),
        (0.0, 330.0, 65.0),
        (-225.0, -170.0, 135.0),
        (225.0, -170.0, 135.0),
        (-150.0, 150.0, 125.0),
        (150.0, 150.0, 125.0),
    ],
    dtype=np.float64,
)


def _format(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}" if math.isfinite(float(value)) else ""
    return str(value)


def _safe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_int(value):
    value = _safe_float(value)
    return int(value) if value is not None and value.is_integer() else None


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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
    digest = hashlib.sha256()
    for path in sorted(Path(value).resolve() for value in paths):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
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
    task_match = re.search(r"_(Freeform|Northwind)_", str(video_id))
    return (
        subject_match.group(1) if subject_match else "",
        task_match.group(1) if task_match else "",
    )


def normalize_landmarks(points):
    """Remove translation, inter-eye scale, and in-plane roll."""

    points = np.asarray(points, dtype=np.float64)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        return None
    left_eye = points[36:42].mean(axis=0)
    right_eye = points[42:48].mean(axis=0)
    eye_vector = right_eye - left_eye
    scale = float(np.linalg.norm(eye_vector))
    if scale <= 1e-6:
        return None
    center = 0.5 * (left_eye + right_eye)
    angle = math.atan2(float(eye_vector[1]), float(eye_vector[0]))
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return ((points - center) @ rotation) / scale


def _rotation_degrees(rotation_vector):
    import cv2

    rotation, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(float(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    if sy < 1e-8:
        pitch = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        yaw = math.atan2(float(-rotation[2, 0]), sy)
        roll = 0.0
    else:
        pitch = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(-rotation[2, 0]), sy)
        roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return tuple(float(math.degrees(value)) for value in (yaw, pitch, roll)), rotation


def estimate_pose_details(points, width, height):
    """Estimate positive-depth PnP pose and its fit diagnostics."""

    import cv2

    points = np.asarray(points, dtype=np.float64)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        return {}
    image_points = points[POSE_LANDMARK_IDS]
    focal = float(max(width, height))
    camera = np.asarray(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)
    solver_specs = [("ITERATIVE", cv2.SOLVEPNP_ITERATIVE)]
    fallback_specs = [
        ("SQPNP", cv2.SOLVEPNP_SQPNP),
        ("EPNP", cv2.SOLVEPNP_EPNP),
    ]
    candidates = []
    for solver, flag in solver_specs:
        try:
            ok, rotation_vector, translation = cv2.solvePnP(
                POSE_MODEL_POINTS,
                image_points,
                camera,
                distortion,
                flags=flag,
            )
        except cv2.error:
            continue
        if ok:
            angles, rotation = _rotation_degrees(rotation_vector)
            camera_points = POSE_MODEL_POINTS @ rotation.T + translation.reshape(1, 3)
            projected, _ = cv2.projectPoints(
                POSE_MODEL_POINTS, rotation_vector, translation, camera, distortion
            )
            rmse = float(
                np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_points) ** 2, axis=1)))
            )
            candidates.append(
                {
                    "angles": angles,
                    "solver": solver,
                    "min_depth": float(camera_points[:, 2].min()),
                    "reprojection_rmse": rmse,
                }
            )
    if not candidates or candidates[0]["min_depth"] <= 0.0:
        for solver, flag in fallback_specs:
            try:
                ok, rotation_vector, translation = cv2.solvePnP(
                    POSE_MODEL_POINTS,
                    image_points,
                    camera,
                    distortion,
                    flags=flag,
                )
            except cv2.error:
                continue
            if not ok:
                continue
            angles, rotation = _rotation_degrees(rotation_vector)
            camera_points = POSE_MODEL_POINTS @ rotation.T + translation.reshape(1, 3)
            min_depth = float(camera_points[:, 2].min())
            if min_depth <= 0.0:
                continue
            projected, _ = cv2.projectPoints(
                POSE_MODEL_POINTS, rotation_vector, translation, camera, distortion
            )
            candidates.append(
                {
                    "angles": angles,
                    "solver": solver,
                    "min_depth": min_depth,
                    "reprojection_rmse": float(
                        np.sqrt(
                            np.mean(
                                np.sum(
                                    (projected.reshape(-1, 2) - image_points) ** 2,
                                    axis=1,
                                )
                            )
                        )
                    ),
                }
            )
    positive = [candidate for candidate in candidates if candidate["min_depth"] > 0.0]
    if not positive:
        return {}
    return min(positive, key=lambda candidate: candidate["reprojection_rmse"])


def estimate_pose_degrees(points, width, height):
    """Estimate positive-depth PnP yaw/pitch/roll from aligned landmarks."""

    details = estimate_pose_details(points, width, height)
    return details.get("angles", (None, None, None))


def _landmark_geometry(points, width, height, canonical):
    import cv2

    points = np.asarray(points, dtype=np.float64)
    in_frame = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] < float(width))
        & (points[:, 1] >= 0.0)
        & (points[:, 1] < float(height))
    )
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    bbox_width = max(0.0, float(x1 - x0))
    bbox_height = max(0.0, float(y1 - y0))
    center_x = 0.5 * float(x0 + x1)
    center_y = 0.5 * float(y0 + y1)
    hull = cv2.convexHull(np.rint(points).astype(np.int32))
    normalized = normalize_landmarks(points)
    residual = None
    if normalized is not None and canonical is not None:
        residual = float(np.median(np.linalg.norm(normalized[17:68] - canonical[17:68], axis=1)))
    left_eye = points[36:42].mean(axis=0)
    right_eye = points[42:48].mean(axis=0)
    pose = estimate_pose_details(points, width, height)
    yaw, pitch, roll = pose.get("angles", (None, None, None))
    return {
        "landmark_in_frame_ratio": float(in_frame.mean()),
        "face_hull_coverage": float(cv2.contourArea(hull) / max(width * height, 1)),
        "bbox_area_ratio": float((bbox_width * bbox_height) / max(width * height, 1)),
        "center_offset_x": float((center_x - width / 2.0) / max(width, 1)),
        "center_offset_y": float((center_y - height / 2.0) / max(height, 1)),
        "eye_distance_ratio": float(np.linalg.norm(right_eye - left_eye) / max(width, 1)),
        "landmark_pose_yaw_deg": yaw,
        "landmark_pose_pitch_deg": pitch,
        "landmark_pose_roll_deg": roll,
        "landmark_pose_solver": pose.get("solver"),
        "landmark_pose_min_depth": pose.get("min_depth"),
        "landmark_pose_reprojection_rmse": pose.get("reprojection_rmse"),
        "transform_residual": residual,
        "normalized_landmarks": normalized,
        "hull": hull,
    }


def _pixel_metrics(rgb, hull=None):
    import cv2

    policy = FrameRecoveryPolicy()
    result = _metrics_from_rgb(rgb, policy)
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if hull is not None:
        mask = np.zeros(luma.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
    else:
        mask = (np.max(rgb, axis=2) > policy.black_threshold).astype(np.uint8)
    if int(mask.sum()) >= 64:
        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        if int(eroded.sum()) < 64:
            eroded = mask.astype(bool)
        laplacian = cv2.Laplacian(luma, cv2.CV_32F, ksize=3)
        gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        gradient = np.sqrt(gx * gx + gy * gy)
        result["blur_score"] = float(np.var(laplacian[eroded]))
        result["gradient_energy"] = float(np.mean(gradient[eroded]))
        result["face_hull_visible_ratio"] = (
            float(np.mean(np.max(rgb, axis=2)[mask.astype(bool)] > policy.black_threshold))
            if hull is not None
            else None
        )
    else:
        result["blur_score"] = None
        result["gradient_energy"] = None
        result["face_hull_visible_ratio"] = None
    return result


def build_train_canonical(split_by_video, landmark_root, video_ids, sample_step=30):
    samples = []
    for video_id in video_ids:
        if split_by_video.get(video_id) != "train":
            continue
        rows = read_landmark_rows(Path(landmark_root) / f"{video_id}.csv")
        for row_index, frame_id in enumerate(sorted(rows)):
            if row_index % max(1, int(sample_step)):
                continue
            row = rows[frame_id]
            if row.get("success") != 1 or len(row.get("landmarks", {})) != 68:
                continue
            points = np.stack([row["landmarks"][index] for index in range(68)])
            normalized = normalize_landmarks(points)
            if normalized is not None:
                samples.append(normalized.astype(np.float32))
    if len(samples) < 3:
        raise ValueError("at least three valid train landmark samples are required")
    return np.median(np.stack(samples), axis=0).astype(np.float64), len(samples)


def _exposure_segments(rows):
    grouped = defaultdict(list)
    for row in rows:
        start = _safe_int(row.get("segment_start_frame"))
        end = _safe_int(row.get("segment_end_frame"))
        if start is None or end is None:
            raise ValueError("exposure review contains invalid segment bounds")
        if row.get("review_status") != "REVIEWED":
            raise ValueError("exposure review contains non-REVIEWED segment")
        grouped[row["video_id"]].append((start, end, row.get("review_decision", "")))
    for video_id in grouped:
        grouped[video_id].sort()
    return grouped


def _exposure_status(segments, frame_id):
    for start, end, decision in segments:
        if start <= frame_id <= end:
            return decision
    return "normal_unmodified"


def _candidate_priority(reason, row):
    metric, mode = CONTACT_REASONS[reason]
    if reason == "frontal_control":
        yaw = _safe_float(row.get("landmark_pose_yaw_deg"))
        pitch = _safe_float(row.get("landmark_pose_pitch_deg"))
        residual = _safe_float(row.get("transform_residual"))
        if yaw is None or pitch is None or residual is None:
            return None
        value = abs(yaw) + abs(pitch) + 10.0 * residual
    else:
        value = _safe_float(row.get(metric))
    if value is None:
        return None
    if mode == "abs_high":
        return abs(value)
    if mode == "high":
        return value
    return -value


def _record_candidate(candidate_by_reason, reason, row, points=None):
    if row.get("split") != "train":
        return
    priority = _candidate_priority(reason, row)
    if priority is None:
        return
    video_id = row["video_id"]
    current = candidate_by_reason[reason].get(video_id)
    item = (priority, int(row["frame_id"]), dict(row), points)
    if current is None or item[:2] > current[:2]:
        candidate_by_reason[reason][video_id] = item


def _select_contacts(candidate_by_reason, count):
    selected = {}
    for reason in CONTACT_REASONS:
        values = sorted(
            candidate_by_reason.get(reason, {}).values(),
            key=lambda item: (item[0], item[2]["video_id"], item[1]),
            reverse=True,
        )
        selected[reason] = values[: max(1, int(count))]
    return selected


def _overlay_image(path, points, row, width=224):
    with Image.open(path) as image:
        image = image.convert("RGB")
    scale = width / image.width
    height = round(image.height * scale)
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    if points is not None:
        scaled = np.asarray(points, dtype=np.float64) * scale
        for x, y in scaled:
            draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=(0, 255, 0))
        x0, y0 = scaled.min(axis=0)
        x1, y1 = scaled.max(axis=0)
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=1)
    label_height = 42
    canvas = Image.new("RGB", (width, height + label_height), "white")
    canvas.paste(image, (0, 0))
    label = (
        f"{row['video_id']} f{row['frame_id']}\n"
        f"conf={_format(row.get('confidence'))} yaw={_format(row.get('landmark_pose_yaw_deg'))} "
        f"pitch={_format(row.get('landmark_pose_pitch_deg'))}"
    )
    ImageDraw.Draw(canvas).text((3, height + 3), label, fill="black")
    return canvas


def _write_contact_sheets(output_dir, selections, columns=4):
    output_dir = Path(output_dir)
    manifest_rows = []
    sheet_paths = []
    for reason, items in selections.items():
        if not items:
            continue
        panels = [_overlay_image(item[2]["image_path"], item[3], item[2]) for item in items]
        cell_width = panels[0].width
        cell_height = panels[0].height
        rows = math.ceil(len(panels) / max(1, int(columns)))
        canvas = Image.new(
            "RGB",
            (max(1, int(columns)) * cell_width, rows * cell_height + 24),
            "white",
        )
        ImageDraw.Draw(canvas).text((4, 5), f"FACE-S1 train-only: {reason}", fill="black")
        sheet_path = output_dir / "contact_sheets" / f"{reason}.jpg"
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        for index, panel in enumerate(panels):
            x = (index % int(columns)) * cell_width
            y = 24 + (index // int(columns)) * cell_height
            canvas.paste(panel, (x, y))
        canvas.save(sheet_path, format="JPEG", quality=92, subsampling=0, optimize=False)
        sheet_paths.append(sheet_path)
        metric_name = CONTACT_REASONS[reason][0]
        for rank, item in enumerate(items, start=1):
            row = item[2]
            metric_value = (
                abs(float(row.get(metric_name)))
                if CONTACT_REASONS[reason][1] == "abs_high" and row.get(metric_name) not in {None, ""}
                else row.get(metric_name, "")
            )
            manifest_rows.append(
                {
                    "selection_reason": reason,
                    "rank": rank,
                    "split": row["split"],
                    "video_id": row["video_id"],
                    "frame_id": row["frame_id"],
                    "image_path": row["image_path"],
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "contact_sheet": str(sheet_path),
                    "review_status": "PENDING",
                    "review_label": "",
                    "review_notes": "",
                }
            )
    return manifest_rows, sheet_paths


def _distribution_rows(metric_matrix, split_codes, task_codes):
    split_names = {0: "train", 1: "val", 2: "test"}
    task_names = {0: "Freeform", 1: "Northwind"}
    scopes = [("overall", "", "", np.ones(len(split_codes), dtype=bool))]
    for code, name in split_names.items():
        scopes.append(("split", name, "", split_codes == code))
    for code, name in task_names.items():
        scopes.append(("task", "", name, task_codes == code))
    for split_code, split_name in split_names.items():
        for task_code, task_name in task_names.items():
            scopes.append(
                (
                    "split_task",
                    split_name,
                    task_name,
                    (split_codes == split_code) & (task_codes == task_code),
                )
            )
    rows = []
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for scope, split, task, group_mask in scopes:
        for metric_index, metric in enumerate(DISTRIBUTION_METRICS):
            values = metric_matrix[group_mask, metric_index]
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            q = np.quantile(values.astype(np.float64), quantiles)
            rows.append(
                {
                    "scope": scope,
                    "split": split,
                    "task_name": task,
                    "metric": metric,
                    "count": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    **{f"q{int(value * 100):02d}": float(result) for value, result in zip(quantiles, q)},
                    "max": float(values.max()),
                }
            )
    return rows


def _status_rows(counter):
    rows = []
    for (split, task, status), count in sorted(counter.items()):
        rows.append(
            {
                "scope": "split_task",
                "split": split,
                "task_name": task,
                "face_usability_status": status,
                "count": count,
            }
        )
    return rows


def _median(rows, field, absolute=False):
    values = []
    for row in rows:
        value = _safe_float(row.get(field))
        if value is not None:
            values.append(abs(value) if absolute else value)
    return float(np.median(values)) if values else None


def _report(path, video_count, frame_count, canonical_count, status_counter, sheets):
    status_counts = Counter()
    for (_split, _task, status), count in status_counter.items():
        status_counts[status] += count
    lines = [
        "# FACE-S1 Phase-1 Face Usability Distribution Audit",
        "",
        "- This phase reads pixels and aligned-space landmarks only.",
        "- It does not read BDI labels, predictions, or AU values.",
        "- No threshold manifest was used and no frame was approved as face_usable.",
        f"- Videos: {video_count}",
        f"- Frames: {frame_count}",
        f"- Train canonical landmark samples: {canonical_count}",
        f"- Contact sheets: {len(sheets)}",
        "",
        "## Provisional Status Counts",
        "",
        "| Status | Frames |",
        "|---|---:|",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- pending_threshold_review is not face_usable approval.",
            "- PnP pose, blur, residual, jump, coverage, and confidence must be reviewed jointly.",
            "- Train-only contact sheets are used to freeze later thresholds; validation/test cannot tune them.",
            "- FACE-S2 clip generation remains blocked until a separate reviewed threshold manifest exists.",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_face_usability_distribution_audit(
    *,
    dataset_split_file,
    image_root,
    aligned_landmark_root,
    frame_audit_dir,
    source_presence_gate,
    exposure_review_manifest,
    output_dir,
    image_integrity_comparison_summary=None,
    video_ids=None,
    max_videos=None,
    canonical_sample_step=30,
    contact_frames_per_reason=12,
    contact_sheet_columns=4,
    project_root=None,
):
    """Run phase-1 FACE-S1 without freezing quality thresholds."""

    dataset_split_file = Path(dataset_split_file).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    aligned_landmark_root = Path(aligned_landmark_root).expanduser().resolve()
    frame_audit_dir = Path(frame_audit_dir).expanduser().resolve()
    source_presence_gate = Path(source_presence_gate).expanduser().resolve()
    exposure_review_manifest = Path(exposure_review_manifest).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output_dir must be empty or absent: {output_dir}")

    split_payload = json.loads(dataset_split_file.read_text(encoding="utf-8"))
    split_by_video = {}
    for split_name, values in split_payload.items():
        normalized = "val" if split_name == "validation" else split_name
        for video_id in values:
            if video_id in split_by_video:
                raise ValueError(f"video appears in multiple splits: {video_id}")
            split_by_video[video_id] = normalized
    selected = sorted(split_by_video)
    if video_ids:
        requested = {str(value) for value in video_ids}
        unknown = sorted(requested - set(selected))
        if unknown:
            raise ValueError(f"unknown requested videos: {unknown}")
        selected = [value for value in selected if value in requested]
    if max_videos is not None:
        selected = selected[: max(0, int(max_videos))]
    if not selected:
        raise ValueError("no videos selected")

    audit_manifest_path = frame_audit_dir / "run_manifest.json"
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    frame_failure_path = frame_audit_dir / "tables" / "frame_failure_manifest.csv"
    recorded_hash = audit_manifest["outputs"]["frame_failure_manifest"]["sha256"]
    if _sha256_file(frame_failure_path) != recorded_hash:
        raise ValueError("frame failure manifest differs from provenance-final hash")
    relocation = validate_relocated_image_root(
        image_root,
        Path(audit_manifest["image_root"]).expanduser().resolve(),
        image_integrity_comparison_summary,
        project_root,
    )

    failures = _read_csv(frame_failure_path)
    failure_by_key = {
        (row["video_id"], int(row["frame_id"])): row for row in failures
    }
    pure_black_keys = {
        key for key, row in failure_by_key.items() if row.get("failure_type") == "pure_black"
    }
    gates = _read_csv(source_presence_gate)
    gate_by_key = {(row["video_id"], int(row["frame_id"])): row for row in gates}
    if set(gate_by_key) != pure_black_keys:
        raise ValueError("source-presence gate does not exactly cover pure-black failures")
    if any(row.get("review_status") != "REVIEWED" for row in gates):
        raise ValueError("source-presence gate contains non-REVIEWED rows")
    exposure_by_video = _exposure_segments(_read_csv(exposure_review_manifest))

    for video_id in selected:
        if not (image_root / video_id).is_dir():
            raise FileNotFoundError(f"missing aligned image directory: {video_id}")
        if not (aligned_landmark_root / f"{video_id}.csv").is_file():
            raise FileNotFoundError(f"missing aligned landmark CSV: {video_id}")

    print(
        f"[FACE_USABILITY_PHASE1] building train-only canonical from {len(selected)} selected videos",
        flush=True,
    )
    canonical, canonical_count = build_train_canonical(
        split_by_video,
        aligned_landmark_root,
        selected,
        sample_step=canonical_sample_step,
    )
    print(
        f"[FACE_USABILITY_PHASE1] canonical ready from {canonical_count} sampled train frames",
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = _write_csv(
        tables_dir / "train_canonical_landmarks.csv",
        [
            {"landmark_id": index, "x_normalized": point[0], "y_normalized": point[1]}
            for index, point in enumerate(canonical)
        ],
        CANONICAL_FIELDS,
    )

    total_frames = sum(len(list((image_root / video_id).glob("*.jpg"))) for video_id in selected)
    print(
        f"[FACE_USABILITY_PHASE1] frame scan starting: {len(selected)} videos, {total_frames} JPG frames",
        flush=True,
    )
    metric_matrix = np.full((total_frames, len(DISTRIBUTION_METRICS)), np.nan, dtype=np.float32)
    split_codes = np.full(total_frames, -1, dtype=np.int8)
    task_codes = np.full(total_frames, -1, dtype=np.int8)
    split_code_map = {"train": 0, "val": 1, "test": 2}
    task_code_map = {"Freeform": 0, "Northwind": 1}
    status_counter = Counter()
    candidate_by_reason = defaultdict(dict)
    video_summaries = []
    frame_path = tables_dir / "face_usability_phase1_frames.csv"
    row_cursor = 0

    with frame_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        for video_number, video_id in enumerate(selected, start=1):
            split = split_by_video[video_id]
            subject_id, task_name = _subject_and_task(video_id)
            landmark_rows = read_landmark_rows(aligned_landmark_root / f"{video_id}.csv")
            image_paths = sorted((image_root / video_id).glob("*.jpg"))
            image_by_frame = {extract_frame_id(path): path for path in image_paths}
            if None in image_by_frame or len(image_by_frame) != len(image_paths):
                raise ValueError(f"invalid or duplicate image frame ids: {video_id}")
            if set(image_by_frame) != set(landmark_rows):
                raise ValueError(f"image/landmark frame set mismatch: {video_id}")
            previous_normalized = None
            previous_frame = None
            video_rows = []
            for frame_index, frame_id in enumerate(sorted(landmark_rows)):
                landmark_row = landmark_rows[frame_id]
                image_path = image_by_frame[frame_id]
                failure = failure_by_key.get((video_id, frame_id))
                gate = gate_by_key.get((video_id, frame_id), {})
                points = None
                geometry = {}
                if landmark_row.get("success") == 1 and len(landmark_row.get("landmarks", {})) == 68:
                    points = np.stack([landmark_row["landmarks"][index] for index in range(68)])
                    geometry = _landmark_geometry(points, 112, 112, canonical)
                try:
                    with Image.open(image_path) as image:
                        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    pixel = _pixel_metrics(rgb, geometry.get("hull"))
                    decode_status = "OK"
                except Exception as exc:
                    pixel = {}
                    decode_status = f"ERROR:{type(exc).__name__}"

                normalized = geometry.get("normalized_landmarks")
                jump = None
                if (
                    normalized is not None
                    and previous_normalized is not None
                    and previous_frame is not None
                    and frame_id == previous_frame + 1
                ):
                    jump = float(
                        np.median(np.linalg.norm(normalized[17:68] - previous_normalized[17:68], axis=1))
                    )
                previous_normalized = normalized
                previous_frame = frame_id if normalized is not None else None

                source_presence = gate.get("source_presence_status", "not_reviewed_nonblack")
                recovery_permission = gate.get("recovery_permission", "")
                aligned_failure = "none"
                status = "pending_threshold_review"
                reasons = []
                if decode_status != "OK":
                    status = "unreadable_or_missing"
                    reasons.append("image_unreadable")
                elif failure and failure.get("failure_type") == "pure_black":
                    if source_presence == "person_absent":
                        status = "person_absent"
                        aligned_failure = "pure_black_person_absent"
                        reasons.append("person_absent")
                    else:
                        status = "aligned_failure_pending_raw_warp"
                        aligned_failure = "pure_black_person_present_detection_failure"
                        reasons.append("aligned_failure_pending_raw_warp")
                elif landmark_row.get("success") != 1 or points is None:
                    status = "face_present_low_quality"
                    aligned_failure = failure.get("failure_type", "landmark_failure") if failure else "landmark_failure"
                    reasons.append("landmark_unreliable")

                row = {
                    "split": split,
                    "subject_id": subject_id,
                    "video_id": video_id,
                    "task_name": task_name,
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "timestamp": landmark_row.get("timestamp"),
                    "image_path": str(image_path),
                    "decode_status": decode_status,
                    "source_presence_status": source_presence,
                    "recovery_permission": recovery_permission,
                    "aligned_failure_status": aligned_failure,
                    "exposure_status": _exposure_status(exposure_by_video.get(video_id, []), frame_id),
                    "nonblack_ratio": pixel.get("nonblack_ratio"),
                    "visible_ratio": pixel.get("visible_ratio"),
                    "global_mean_luma": pixel.get("global_mean_luma"),
                    "global_std_luma": pixel.get("global_std_luma"),
                    "low_saturation_ratio": pixel.get("low_saturation_ratio"),
                    "high_saturation_ratio": pixel.get("high_saturation_ratio"),
                    "blur_score": pixel.get("blur_score"),
                    "gradient_energy": pixel.get("gradient_energy"),
                    "landmark_success": landmark_row.get("success"),
                    "confidence": landmark_row.get("confidence"),
                    "landmark_count": len(landmark_row.get("landmarks", {})),
                    "landmark_in_frame_ratio": geometry.get("landmark_in_frame_ratio"),
                    "face_hull_coverage": geometry.get("face_hull_coverage"),
                    "face_hull_visible_ratio": pixel.get("face_hull_visible_ratio"),
                    "bbox_area_ratio": geometry.get("bbox_area_ratio"),
                    "center_offset_x": geometry.get("center_offset_x"),
                    "center_offset_y": geometry.get("center_offset_y"),
                    "eye_distance_ratio": geometry.get("eye_distance_ratio"),
                    "landmark_pose_yaw_deg": geometry.get("landmark_pose_yaw_deg"),
                    "landmark_pose_pitch_deg": geometry.get("landmark_pose_pitch_deg"),
                    "landmark_pose_roll_deg": geometry.get("landmark_pose_roll_deg"),
                    "landmark_pose_solver": geometry.get("landmark_pose_solver"),
                    "landmark_pose_min_depth": geometry.get("landmark_pose_min_depth"),
                    "landmark_pose_reprojection_rmse": geometry.get("landmark_pose_reprojection_rmse"),
                    "transform_residual": geometry.get("transform_residual"),
                    "landmark_jump": jump,
                    "face_usability_status": status,
                    "exclusion_reasons": ";".join(reasons),
                    "review_status": "PENDING" if status == "pending_threshold_review" else "HARD_STATUS",
                }
                writer.writerow({field: _format(row.get(field)) for field in FRAME_FIELDS})
                video_rows.append(row)
                status_counter[(split, task_name, status)] += 1
                split_codes[row_cursor] = split_code_map[split]
                task_codes[row_cursor] = task_code_map[task_name]
                for metric_index, metric in enumerate(DISTRIBUTION_METRICS):
                    value = _safe_float(row.get(metric))
                    if value is not None:
                        metric_matrix[row_cursor, metric_index] = value
                row_cursor += 1

                if status == "pending_threshold_review":
                    for reason in CONTACT_REASONS:
                        if reason != "visible_landmark_failure":
                            _record_candidate(candidate_by_reason, reason, row, points)
                elif status == "face_present_low_quality" and pixel.get("nonblack_ratio", 0.0) > 0.1:
                    _record_candidate(candidate_by_reason, "visible_landmark_failure", row, None)

            counts = Counter(row["face_usability_status"] for row in video_rows)
            video_summaries.append(
                {
                    "split": split,
                    "video_id": video_id,
                    "subject_id": subject_id,
                    "task_name": task_name,
                    "frame_count": len(video_rows),
                    "decode_ok_count": sum(row["decode_status"] == "OK" for row in video_rows),
                    "landmark_success_count": sum(row["landmark_success"] == 1 for row in video_rows),
                    "landmark_success_ratio": sum(row["landmark_success"] == 1 for row in video_rows) / max(len(video_rows), 1),
                    "person_absent_count": counts["person_absent"],
                    "aligned_failure_pending_raw_warp_count": counts["aligned_failure_pending_raw_warp"],
                    "face_present_low_quality_count": counts["face_present_low_quality"],
                    "pending_threshold_review_count": counts["pending_threshold_review"],
                    "median_confidence": _median(video_rows, "confidence"),
                    "median_landmark_in_frame_ratio": _median(video_rows, "landmark_in_frame_ratio"),
                    "median_face_hull_coverage": _median(video_rows, "face_hull_coverage"),
                    "median_face_hull_visible_ratio": _median(video_rows, "face_hull_visible_ratio"),
                    "median_blur_score": _median(video_rows, "blur_score"),
                    "median_abs_yaw_deg": _median(video_rows, "landmark_pose_yaw_deg", absolute=True),
                    "median_abs_pitch_deg": _median(video_rows, "landmark_pose_pitch_deg", absolute=True),
                    "median_pose_reprojection_rmse": _median(
                        video_rows, "landmark_pose_reprojection_rmse"
                    ),
                    "median_transform_residual": _median(video_rows, "transform_residual"),
                    "median_landmark_jump": _median(video_rows, "landmark_jump"),
                }
            )
            print(f"[FACE_USABILITY_PHASE1] processed {video_number}/{len(selected)} videos", flush=True)

    if row_cursor != total_frames:
        raise RuntimeError(f"frame accounting mismatch: wrote {row_cursor}, expected {total_frames}")
    distributions = _distribution_rows(metric_matrix, split_codes, task_codes)
    distribution_path = _write_csv(
        tables_dir / "face_quality_distributions.csv",
        distributions,
        DISTRIBUTION_FIELDS,
    )
    video_path = _write_csv(
        tables_dir / "face_usability_video_summary.csv", video_summaries, VIDEO_FIELDS
    )
    status_path = _write_csv(
        tables_dir / "face_usability_status_summary.csv",
        _status_rows(status_counter),
        STATUS_FIELDS,
    )
    contact_rows, contact_sheets = _write_contact_sheets(
        output_dir,
        _select_contacts(candidate_by_reason, contact_frames_per_reason),
        columns=contact_sheet_columns,
    )
    contact_path = _write_csv(
        tables_dir / "face_usability_contact_review.csv", contact_rows, CONTACT_FIELDS
    )
    report_path = _report(
        output_dir / "reports" / "face_usability_phase1_report.md",
        len(selected),
        total_frames,
        canonical_count,
        status_counter,
        contact_sheets,
    )

    landmark_paths = [aligned_landmark_root / f"{video_id}.csv" for video_id in selected]
    landmark_tree_hash = _tree_sha256(landmark_paths, aligned_landmark_root)
    payload = {
        "audit": "FACE-S1 phase-1 face usability distributions and train-only review sheets",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(shlex.quote(item) for item in sys.argv),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "threshold_manifest_used": False,
        "final_face_usable_approval_generated": False,
        "labels_or_predictions_read": False,
        "au_values_read": False,
        "face_landmark_rerun_performed": False,
        "output_status": "DISTRIBUTION_REVIEW_REQUIRED",
        "video_count": len(selected),
        "frame_count": total_frames,
        "canonical_sample_step": int(canonical_sample_step),
        "canonical_sample_count": canonical_count,
        "contact_frames_per_reason": int(contact_frames_per_reason),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "relocated_image_root": relocation,
        "inputs": {
            "dataset_split_file": {"path": str(dataset_split_file), "sha256": _sha256_file(dataset_split_file)},
            "frame_audit_manifest": {"path": str(audit_manifest_path), "sha256": _sha256_file(audit_manifest_path)},
            "frame_failure_manifest": {"path": str(frame_failure_path), "sha256": _sha256_file(frame_failure_path)},
            "source_presence_gate": {"path": str(source_presence_gate), "sha256": _sha256_file(source_presence_gate)},
            "exposure_review_manifest": {"path": str(exposure_review_manifest), "sha256": _sha256_file(exposure_review_manifest)},
            "aligned_landmark_root": str(aligned_landmark_root),
            "aligned_landmark_csv_count": len(landmark_paths),
            "aligned_landmark_tree_sha256": landmark_tree_hash,
        },
        "implementation_files": {
            "core": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())},
            "cli": {
                "path": str(project_root / "scripts" / "audit_face_usability.py"),
                "sha256": _sha256_file(project_root / "scripts" / "audit_face_usability.py"),
            },
        },
        "outputs": {
            "frame_manifest": {"path": str(frame_path), "sha256": _sha256_file(frame_path)},
            "video_summary": {"path": str(video_path), "sha256": _sha256_file(video_path)},
            "distributions": {"path": str(distribution_path), "sha256": _sha256_file(distribution_path)},
            "status_summary": {"path": str(status_path), "sha256": _sha256_file(status_path)},
            "canonical_landmarks": {"path": str(canonical_path), "sha256": _sha256_file(canonical_path)},
            "contact_review": {"path": str(contact_path), "sha256": _sha256_file(contact_path)},
            "contact_sheets": [
                {"path": str(path), "sha256": _sha256_file(path)} for path in contact_sheets
            ],
            "report": {"path": str(report_path), "sha256": _sha256_file(report_path)},
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [frame_path, video_path, distribution_path, status_path, canonical_path, contact_path, report_path, manifest_path]
