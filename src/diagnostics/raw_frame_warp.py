"""Fail-closed smoke audit for warping real raw-video frames into aligned space.

The target aligned JPG is a pure-black placeholder.  Geometry is recovered
only from valid raw/aligned landmark anchors around that target, while raw
landmarks are propagated to the real target frame with adjacent-frame optical
flow.  Outputs remain review-required and are never installed into a dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.diagnostics.frame_recovery import validate_relocated_image_root
from src.diagnostics.raw_detail_recoverability import (
    ALIGNMENT_INDICES,
    _landmark_map,
    _raw_video_id,
    _raw_video_path,
)


SMOKE_FIELDS = [
    "split",
    "video_id",
    "pure_black_run_id",
    "target_frame",
    "previous_anchor_frame",
    "next_anchor_frame",
    "previous_anchor_gap",
    "next_anchor_gap",
    "raw_target_openface_success",
    "previous_anchor_inlier_ratio",
    "previous_anchor_rmse_px",
    "next_anchor_inlier_ratio",
    "next_anchor_rmse_px",
    "previous_tracking_valid_ratio",
    "previous_tracking_median_fb_error_px",
    "next_tracking_valid_ratio",
    "next_tracking_median_fb_error_px",
    "combined_tracking_valid_ratio",
    "target_transform_inlier_ratio",
    "target_transform_rmse_px",
    "transform_disagreement_median_px",
    "warped_landmark_in_bounds_ratio",
    "derived_nonblack_ratio",
    "raw_video_path",
    "raw_openface_csv",
    "aligned_openface_csv",
    "previous_aligned_path",
    "target_placeholder_path",
    "next_aligned_path",
    "derived_path",
    "contact_sheet",
    "raw_target_rgb_sha256",
    "derived_sha256",
    "automatic_status",
    "review_status",
    "issues",
]


def _format(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6f}" if math.isfinite(value) else ""
    return str(value)


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


def _sha256_array(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
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


def _safe_int(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def _estimate_similarity(source_points, target_points, valid, ransac_threshold_px):
    import cv2

    indices = np.asarray(ALIGNMENT_INDICES, dtype=np.int64)
    selected = np.asarray(valid, dtype=bool)[indices]
    source = np.asarray(source_points, dtype=np.float32)[indices][selected]
    target = np.asarray(target_points, dtype=np.float32)[indices][selected]
    if len(source) < 12:
        return None, None, None
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold_px),
        maxIters=5000,
        confidence=0.999,
        refineIters=20,
    )
    if matrix is None or inliers is None:
        return None, None, None
    inliers = inliers.reshape(-1).astype(bool)
    if not inliers.any():
        return matrix, 0.0, None
    predicted = cv2.transform(source[None, :, :], matrix)[0]
    rmse = float(
        np.sqrt(np.mean(np.sum((predicted[inliers] - target[inliers]) ** 2, axis=1)))
    )
    return matrix, float(inliers.mean()), rmse


def interpolate_similarity_matrix(previous, following, alpha):
    """Linearly interpolate two 2D similarity transforms without adding shear."""

    previous = np.asarray(previous, dtype=np.float64)
    following = np.asarray(following, dtype=np.float64)

    def components(matrix):
        a = 0.5 * (matrix[0, 0] + matrix[1, 1])
        b = 0.5 * (matrix[0, 1] - matrix[1, 0])
        return np.asarray([a, b, matrix[0, 2], matrix[1, 2]], dtype=np.float64)

    a, b, tx, ty = (1.0 - float(alpha)) * components(previous) + float(alpha) * components(
        following
    )
    return np.asarray([[a, b, tx], [-b, a, ty]], dtype=np.float64)


def _track_adjacent_sequence(frames, points, max_fb_error_px):
    """Track landmarks over adjacent raw frames and reject forward/backward drift."""

    import cv2

    points = np.asarray(points, dtype=np.float32).copy()
    valid = np.ones(len(points), dtype=bool)
    max_errors = np.zeros(len(points), dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    for source_bgr, target_bgr in zip(frames[:-1], frames[1:]):
        source_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
        forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            source_gray,
            target_gray,
            points.reshape(-1, 1, 2),
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=criteria,
        )
        if forward is None or forward_status is None:
            return points, np.zeros(len(points), dtype=bool), np.full(len(points), np.inf)
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            target_gray,
            source_gray,
            forward,
            None,
            winSize=(31, 31),
            maxLevel=3,
            criteria=criteria,
        )
        if backward is None or backward_status is None:
            return points, np.zeros(len(points), dtype=bool), np.full(len(points), np.inf)
        forward = forward.reshape(-1, 2)
        backward = backward.reshape(-1, 2)
        fb_error = np.linalg.norm(backward - points, axis=1)
        height, width = target_gray.shape
        in_bounds = (
            (forward[:, 0] >= 0.0)
            & (forward[:, 0] < width)
            & (forward[:, 1] >= 0.0)
            & (forward[:, 1] < height)
        )
        valid &= (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.isfinite(fb_error)
            & (fb_error <= float(max_fb_error_px))
            & in_bounds
        )
        max_errors = np.maximum(max_errors, np.nan_to_num(fb_error, nan=np.inf, posinf=np.inf))
        points = forward
    return points, valid, max_errors


def _combine_tracked_points(previous_points, previous_valid, next_points, next_valid):
    previous_points = np.asarray(previous_points, dtype=np.float32)
    next_points = np.asarray(next_points, dtype=np.float32)
    previous_valid = np.asarray(previous_valid, dtype=bool)
    next_valid = np.asarray(next_valid, dtype=bool)
    combined = np.zeros_like(previous_points)
    both = previous_valid & next_valid
    combined[both] = 0.5 * (previous_points[both] + next_points[both])
    combined[previous_valid & ~next_valid] = previous_points[previous_valid & ~next_valid]
    combined[next_valid & ~previous_valid] = next_points[next_valid & ~previous_valid]
    return combined, previous_valid | next_valid


def select_smoke_candidates(
    failure_rows,
    source_gate_rows,
    *,
    split="train",
    max_anchor_gap=8,
    requested_targets=None,
):
    """Validate the human gate and deterministically rank one target per run/video."""

    pure_black = {}
    for row in failure_rows:
        if row.get("failure_type") != "pure_black":
            continue
        frame = _safe_int(row.get("frame_id"))
        if frame is None:
            raise ValueError("pure-black failure row lacks an integer frame_id")
        key = (row["video_id"], frame)
        if key in pure_black:
            raise ValueError(f"duplicate pure-black failure key: {key}")
        pure_black[key] = row

    gate = {}
    for row in source_gate_rows:
        frame = _safe_int(row.get("frame_id"))
        if frame is None:
            raise ValueError("source-presence gate row lacks an integer frame_id")
        key = (row["video_id"], frame)
        if key in gate:
            raise ValueError(f"duplicate source-presence gate key: {key}")
        gate[key] = row
    if set(gate) != set(pure_black):
        raise ValueError("source-presence gate does not exactly cover the safe-audit pure-black frames")
    for key, row in gate.items():
        if row.get("review_status") != "REVIEWED":
            raise ValueError(f"source-presence gate is not REVIEWED: {key}")

    requested = None
    if requested_targets:
        requested = set()
        for value in requested_targets:
            video_id, separator, frame = str(value).rpartition(":")
            frame_id = _safe_int(frame)
            if not separator or not video_id or frame_id is None:
                raise ValueError(f"invalid target selector, expected VIDEO_ID:FRAME_ID: {value}")
            requested.add((video_id, frame_id))

    candidates = []
    for key, failure in pure_black.items():
        review = gate[key]
        if split and failure.get("split") != split:
            continue
        if requested is not None and key not in requested:
            continue
        if review.get("source_presence_status") != "person_present_detection_failure":
            continue
        if review.get("recovery_permission") != "raw_frame_warp_only":
            continue
        target = key[1]
        previous = _safe_int(failure.get("previous_valid_frame"))
        following = _safe_int(failure.get("next_valid_frame"))
        block_length = _safe_int(failure.get("failure_block_length"))
        if previous is None or following is None or not previous < target < following:
            continue
        previous_gap = target - previous
        next_gap = following - target
        if max(previous_gap, next_gap) > int(max_anchor_gap):
            continue
        candidates.append(
            {
                "failure": failure,
                "review": review,
                "target_frame": target,
                "previous_anchor_frame": previous,
                "next_anchor_frame": following,
                "previous_anchor_gap": previous_gap,
                "next_anchor_gap": next_gap,
                "block_length": block_length if block_length is not None else following - previous - 1,
                "rank": (
                    block_length if block_length is not None else following - previous - 1,
                    max(previous_gap, next_gap),
                    previous_gap + next_gap,
                    failure["video_id"],
                    target,
                ),
            }
        )
    if requested is not None:
        selected_keys = {(item["failure"]["video_id"], item["target_frame"]) for item in candidates}
        missing = sorted(requested - selected_keys)
        if missing:
            raise ValueError(f"requested targets are not eligible under the frozen gates: {missing}")

    candidates.sort(key=lambda item: item["rank"])
    one_per_run = []
    seen_runs = set()
    for item in candidates:
        run_id = item["review"].get("pure_black_run_id") or item["failure"].get("failure_block_id")
        if run_id in seen_runs:
            continue
        seen_runs.add(run_id)
        one_per_run.append(item)
    return one_per_run


def _decode_raw_range(video_path, start_frame, end_frame):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open raw video: {video_path}")
    frames = {}
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame) - 1)
        for frame_id in range(int(start_frame), int(end_frame) + 1):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode raw frame {video_path}:{frame_id}")
            frames[frame_id] = bgr
    finally:
        capture.release()
    return frames


def _load_aligned_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _draw_raw_landmarks(raw_bgr, points, valid):
    import cv2

    rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for point, is_valid in zip(points, valid):
        if not is_valid:
            continue
        x, y = float(point[0]), float(point[1])
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0))
    return np.asarray(image, dtype=np.uint8)


def _panel(rgb, width, height, label):
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    contained = ImageOps.contain(image, (width, height), method=Image.Resampling.BILINEAR)
    panel = Image.new("RGB", (width, height + 24), "white")
    panel.paste(contained, ((width - contained.width) // 2, (height - contained.height) // 2))
    ImageDraw.Draw(panel).text((4, height + 5), label, fill="black")
    return panel


def _write_contact_sheet(path, previous_aligned, raw_target, derived, next_aligned, row):
    width, height = 240, 180
    panels = [
        _panel(previous_aligned, width, height, f"prev aligned f{row['previous_anchor_frame']}"),
        _panel(raw_target, width, height, f"real raw target f{row['target_frame']}"),
        _panel(derived, width, height, "raw target warped to 112x112"),
        _panel(next_aligned, width, height, f"next aligned f{row['next_anchor_frame']}"),
    ]
    canvas = Image.new("RGB", (2 * width, 2 * (height + 24) + 30), "white")
    ImageDraw.Draw(canvas).text(
        (5, 7),
        "Smoke only: real raw pixels; green points are propagated landmarks",
        fill="black",
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 2) * width, 30 + (index // 2) * (height + 24)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=92, subsampling=0, optimize=False)
    return path


def _process_candidate(
    candidate,
    *,
    dataset_root,
    image_root,
    raw_openface_root,
    aligned_openface_root,
    output_dir,
    max_fb_error_px,
    min_tracking_valid_ratio,
    min_transform_inlier_ratio,
    max_transform_rmse_px,
    ransac_threshold_px,
    max_transform_disagreement_px,
    min_warped_landmark_in_bounds_ratio,
):
    import cv2

    failure = candidate["failure"]
    review = candidate["review"]
    video_id = failure["video_id"]
    target = candidate["target_frame"]
    previous = candidate["previous_anchor_frame"]
    following = candidate["next_anchor_frame"]
    raw_video = _raw_video_path(dataset_root, failure["split"], video_id)
    raw_csv = Path(raw_openface_root) / f"{_raw_video_id(video_id)}.csv"
    aligned_csv = Path(aligned_openface_root) / f"{video_id}.csv"
    missing = [str(path) for path in (raw_video, raw_csv, aligned_csv) if not path.exists()]
    base = {
        "split": failure["split"],
        "video_id": video_id,
        "pure_black_run_id": review.get("pure_black_run_id", ""),
        "target_frame": target,
        "previous_anchor_frame": previous,
        "next_anchor_frame": following,
        "previous_anchor_gap": candidate["previous_anchor_gap"],
        "next_anchor_gap": candidate["next_anchor_gap"],
        "raw_video_path": str(raw_video),
        "raw_openface_csv": str(raw_csv),
        "aligned_openface_csv": str(aligned_csv),
        "automatic_status": "AUTO_FAIL",
        "review_status": "NOT_APPLICABLE",
        "issues": "",
    }
    if missing:
        return {**base, "issues": f"missing_inputs:{'|'.join(missing)}"}

    frame_ids = [previous, target, following]
    raw_landmarks = _landmark_map(raw_csv, frame_ids)
    aligned_landmarks = _landmark_map(aligned_csv, frame_ids)
    base["raw_target_openface_success"] = (raw_landmarks.get(target) or {}).get("success")
    issues = []
    for name, frame_id in (("previous", previous), ("next", following)):
        raw_anchor = raw_landmarks.get(frame_id)
        aligned_anchor = aligned_landmarks.get(frame_id)
        if raw_anchor is None or raw_anchor.get("success") != 1 or raw_anchor.get("points") is None:
            issues.append(f"{name}_raw_anchor_invalid")
        if (
            aligned_anchor is None
            or aligned_anchor.get("success") != 1
            or aligned_anchor.get("points") is None
        ):
            issues.append(f"{name}_aligned_anchor_invalid")
    if issues:
        return {**base, "issues": ";".join(issues)}

    video_dir = Path(image_root) / video_id
    previous_path = video_dir / Path(failure["previous_valid_path"]).name
    target_path = video_dir / Path(failure["image_path"]).name
    next_path = video_dir / Path(failure["next_valid_path"]).name
    base.update(
        {
            "previous_aligned_path": str(previous_path),
            "target_placeholder_path": str(target_path),
            "next_aligned_path": str(next_path),
        }
    )
    if not all(path.exists() for path in (previous_path, target_path, next_path)):
        return {**base, "issues": "relocated_aligned_anchor_or_target_missing"}

    frames = _decode_raw_range(raw_video, previous, following)
    previous_raw = raw_landmarks[previous]["points"]
    next_raw = raw_landmarks[following]["points"]
    previous_aligned_points = aligned_landmarks[previous]["points"]
    next_aligned_points = aligned_landmarks[following]["points"]
    all_valid = np.ones(68, dtype=bool)
    previous_matrix, previous_inlier, previous_rmse = _estimate_similarity(
        previous_raw,
        previous_aligned_points,
        all_valid,
        ransac_threshold_px,
    )
    next_matrix, next_inlier, next_rmse = _estimate_similarity(
        next_raw,
        next_aligned_points,
        all_valid,
        ransac_threshold_px,
    )
    base.update(
        {
            "previous_anchor_inlier_ratio": previous_inlier,
            "previous_anchor_rmse_px": previous_rmse,
            "next_anchor_inlier_ratio": next_inlier,
            "next_anchor_rmse_px": next_rmse,
        }
    )
    for name, matrix, inlier, rmse in (
        ("previous", previous_matrix, previous_inlier, previous_rmse),
        ("next", next_matrix, next_inlier, next_rmse),
    ):
        if matrix is None or inlier is None or rmse is None:
            issues.append(f"{name}_anchor_transform_failed")
        elif inlier < float(min_transform_inlier_ratio):
            issues.append(f"{name}_anchor_inlier_below_gate")
        elif rmse > float(max_transform_rmse_px):
            issues.append(f"{name}_anchor_rmse_above_gate")
    if issues:
        return {**base, "issues": ";".join(issues)}

    previous_points, previous_valid, previous_errors = _track_adjacent_sequence(
        [frames[index] for index in range(previous, target + 1)],
        previous_raw,
        max_fb_error_px,
    )
    next_points, next_valid, next_errors = _track_adjacent_sequence(
        [frames[index] for index in range(following, target - 1, -1)],
        next_raw,
        max_fb_error_px,
    )
    tracked_points, tracked_valid = _combine_tracked_points(
        previous_points,
        previous_valid,
        next_points,
        next_valid,
    )
    inner = np.asarray(ALIGNMENT_INDICES, dtype=np.int64)

    def valid_ratio(mask):
        return float(np.mean(np.asarray(mask, dtype=bool)[inner]))

    def valid_median(values, mask):
        selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
        return float(np.median(selected)) if selected.size else None

    base.update(
        {
            "previous_tracking_valid_ratio": valid_ratio(previous_valid),
            "previous_tracking_median_fb_error_px": valid_median(previous_errors, previous_valid),
            "next_tracking_valid_ratio": valid_ratio(next_valid),
            "next_tracking_median_fb_error_px": valid_median(next_errors, next_valid),
            "combined_tracking_valid_ratio": valid_ratio(tracked_valid),
        }
    )
    if valid_ratio(tracked_valid) < float(min_tracking_valid_ratio):
        return {**base, "issues": "combined_tracking_valid_ratio_below_gate"}

    alpha = float(target - previous) / float(following - previous)
    aligned_target_points = (
        (1.0 - alpha) * previous_aligned_points + alpha * next_aligned_points
    ).astype(np.float32)
    target_matrix, target_inlier, target_rmse = _estimate_similarity(
        tracked_points,
        aligned_target_points,
        tracked_valid,
        ransac_threshold_px,
    )
    base.update(
        {
            "target_transform_inlier_ratio": target_inlier,
            "target_transform_rmse_px": target_rmse,
        }
    )
    if target_matrix is None or target_inlier is None or target_rmse is None:
        return {**base, "issues": "target_transform_failed"}
    if target_inlier < float(min_transform_inlier_ratio):
        issues.append("target_transform_inlier_below_gate")
    if target_rmse > float(max_transform_rmse_px):
        issues.append("target_transform_rmse_above_gate")

    interpolated_matrix = interpolate_similarity_matrix(previous_matrix, next_matrix, alpha)
    selected_points = tracked_points[tracked_valid]
    target_projection = cv2.transform(selected_points[None, :, :], target_matrix)[0]
    interpolated_projection = cv2.transform(
        selected_points[None, :, :], interpolated_matrix
    )[0]
    disagreement = float(np.median(np.linalg.norm(target_projection - interpolated_projection, axis=1)))
    base["transform_disagreement_median_px"] = disagreement
    if disagreement > float(max_transform_disagreement_px):
        issues.append("tracked_vs_interpolated_transform_disagreement_above_gate")

    transformed_landmarks = cv2.transform(
        tracked_points[tracked_valid][None, :, :], target_matrix
    )[0]
    in_bounds = (
        (transformed_landmarks[:, 0] >= 0.0)
        & (transformed_landmarks[:, 0] < 112.0)
        & (transformed_landmarks[:, 1] >= 0.0)
        & (transformed_landmarks[:, 1] < 112.0)
    )
    in_bounds_ratio = float(np.mean(in_bounds)) if len(in_bounds) else 0.0
    base["warped_landmark_in_bounds_ratio"] = in_bounds_ratio
    if in_bounds_ratio < float(min_warped_landmark_in_bounds_ratio):
        issues.append("warped_landmark_in_bounds_ratio_below_gate")
    if issues:
        return {**base, "issues": ";".join(issues)}

    target_bgr = frames[target]
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    derived = cv2.warpAffine(
        target_rgb,
        target_matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    nonblack_ratio = float(np.mean(np.max(derived, axis=2) > 8))
    base["derived_nonblack_ratio"] = nonblack_ratio
    if nonblack_ratio <= 0.10:
        return {**base, "issues": "derived_frame_is_effectively_black"}

    derived_path = Path(output_dir) / "derived" / video_id / target_path.name
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(derived, mode="RGB").save(
        derived_path,
        format="JPEG",
        quality=95,
        subsampling=0,
        optimize=False,
    )
    previous_aligned_rgb = _load_aligned_rgb(previous_path)
    next_aligned_rgb = _load_aligned_rgb(next_path)
    raw_overlay = _draw_raw_landmarks(target_bgr, tracked_points, tracked_valid)
    contact_path = Path(output_dir) / "contact_sheets" / f"{video_id}_f{target:06d}.jpg"
    row = {
        **base,
        "derived_path": str(derived_path),
        "contact_sheet": str(contact_path),
        "raw_target_rgb_sha256": _sha256_array(target_rgb),
        "derived_sha256": _sha256_file(derived_path),
        "automatic_status": "AUTO_PASS_REVIEW_REQUIRED",
        "review_status": "PENDING",
        "issues": "",
    }
    _write_contact_sheet(
        contact_path,
        previous_aligned_rgb,
        raw_overlay,
        derived,
        next_aligned_rgb,
        row,
    )
    return row


def _write_report(path, rows, requested_count):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    automatic = [row for row in rows if row["automatic_status"] == "AUTO_PASS_REVIEW_REQUIRED"]
    failed = [row for row in rows if row["automatic_status"] == "AUTO_FAIL"]
    lines = [
        "# Raw-frame Warp Smoke Report",
        "",
        "- This is a 1-3 frame smoke audit, not a repaired dataset.",
        "- Every derived JPG uses pixels from the real target raw-video frame.",
        "- The pure-black aligned target is never synthesized from aligned neighbors.",
        "- Automatic passage remains REVIEW_REQUIRED until the contact sheet is reviewed.",
        f"- Requested smoke frames: {requested_count}",
        f"- Automatic pass / review required: {len(automatic)}",
        f"- Automatic failures attempted: {len(failed)}",
        "",
        "## Automatic Results",
        "",
        "| Video | Target | Anchors | Track ratio | Transform RMSE | Disagreement | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {video_id} | {target_frame} | {previous_anchor_frame}/{next_anchor_frame} | "
            "{combined_tracking_valid_ratio} | {target_transform_rmse_px} | "
            "{transform_disagreement_median_px} | {automatic_status} |".format(
                **{key: _format(row.get(key)) for key in SMOKE_FIELDS}
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- AUTO_PASS_REVIEW_REQUIRED only proves that the audited geometry and optical-flow gates were finite and internally consistent.",
            "- Manual review must confirm that identity, pose, occlusion, and framing in the real target frame were preserved.",
            "- No OpenFace/FaceLandmark rerun was performed; derived landmark validity is unknown.",
            "- These files must not be used for training or merged into face_images.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_raw_frame_warp_smoke(
    *,
    audit_dir,
    dataset_root,
    image_root,
    source_presence_gate,
    raw_openface_root,
    aligned_openface_root,
    output_dir,
    image_integrity_comparison_summary=None,
    max_frames=3,
    split="train",
    target_frames=None,
    max_anchor_gap=8,
    max_fb_error_px=3.0,
    min_tracking_valid_ratio=0.60,
    min_transform_inlier_ratio=0.50,
    max_transform_rmse_px=2.50,
    ransac_threshold_px=2.0,
    max_transform_disagreement_px=5.0,
    min_warped_landmark_in_bounds_ratio=0.80,
    project_root=None,
):
    """Generate up to three review-required raw-frame warp smoke outputs."""

    audit_dir = Path(audit_dir).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    source_presence_gate = Path(source_presence_gate).expanduser().resolve()
    raw_openface_root = Path(raw_openface_root).expanduser().resolve()
    aligned_openface_root = Path(aligned_openface_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output_dir must be empty or absent: {output_dir}")
    if not 1 <= int(max_frames) <= 3:
        raise ValueError("raw-frame warp smoke is restricted to 1-3 frames")
    if split != "train":
        raise ValueError("raw-frame warp smoke selection is frozen to train split")
    if int(max_anchor_gap) < 1:
        raise ValueError("max_anchor_gap must be positive")
    if float(max_fb_error_px) <= 0.0:
        raise ValueError("max_fb_error_px must be positive")
    for name, value in (
        ("min_tracking_valid_ratio", min_tracking_valid_ratio),
        ("min_transform_inlier_ratio", min_transform_inlier_ratio),
        ("min_warped_landmark_in_bounds_ratio", min_warped_landmark_in_bounds_ratio),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    for name, value in (
        ("max_transform_rmse_px", max_transform_rmse_px),
        ("ransac_threshold_px", ransac_threshold_px),
        ("max_transform_disagreement_px", max_transform_disagreement_px),
    ):
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be positive")

    audit_manifest_path = audit_dir / "run_manifest.json"
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    if int(audit_manifest.get("video_count", -1)) != 300:
        raise ValueError("raw-frame warp smoke requires the full 300-video audit")
    if int(audit_manifest.get("failed_frame_count", -1)) != 6501:
        raise ValueError("raw-frame warp smoke requires the frozen 6501-frame failure inventory")
    if int(audit_manifest.get("failure_block_count", -1)) != 221:
        raise ValueError("raw-frame warp smoke requires the frozen 221 failure blocks")
    frame_manifest_path = audit_dir / "tables" / "frame_failure_manifest.csv"
    recorded_frame_hash = audit_manifest["outputs"]["frame_failure_manifest"]["sha256"]
    if _sha256_file(frame_manifest_path) != recorded_frame_hash:
        raise ValueError("frame failure manifest differs from the provenance-final hash")

    relocation = validate_relocated_image_root(
        image_root,
        Path(audit_manifest["image_root"]).expanduser().resolve(),
        image_integrity_comparison_summary,
        project_root,
    )
    failures = _read_csv(frame_manifest_path)
    gates = _read_csv(source_presence_gate)
    candidates = select_smoke_candidates(
        failures,
        gates,
        split=split,
        max_anchor_gap=max_anchor_gap,
        requested_targets=target_frames,
    )
    if not candidates:
        raise ValueError("no eligible train raw-frame warp smoke candidates")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    passed = 0
    seen_videos = set()
    for candidate in candidates:
        video_id = candidate["failure"]["video_id"]
        if not target_frames and video_id in seen_videos:
            continue
        row = _process_candidate(
            candidate,
            dataset_root=dataset_root,
            image_root=image_root,
            raw_openface_root=raw_openface_root,
            aligned_openface_root=aligned_openface_root,
            output_dir=output_dir,
            max_fb_error_px=max_fb_error_px,
            min_tracking_valid_ratio=min_tracking_valid_ratio,
            min_transform_inlier_ratio=min_transform_inlier_ratio,
            max_transform_rmse_px=max_transform_rmse_px,
            ransac_threshold_px=ransac_threshold_px,
            max_transform_disagreement_px=max_transform_disagreement_px,
            min_warped_landmark_in_bounds_ratio=min_warped_landmark_in_bounds_ratio,
        )
        rows.append(row)
        seen_videos.add(video_id)
        if row["automatic_status"] == "AUTO_PASS_REVIEW_REQUIRED":
            passed += 1
        if passed >= int(max_frames):
            break
        if target_frames and len(rows) >= len(target_frames):
            break

    table_path = _write_csv(output_dir / "tables" / "raw_frame_warp_smoke.csv", rows, SMOKE_FIELDS)
    report_path = _write_report(
        output_dir / "reports" / "raw_frame_warp_smoke_report.md",
        rows,
        int(max_frames),
    )
    raw_video_paths = sorted(
        {row["raw_video_path"] for row in rows if Path(row["raw_video_path"]).exists()}
    )
    raw_csv_paths = sorted(
        {row["raw_openface_csv"] for row in rows if Path(row["raw_openface_csv"]).exists()}
    )
    aligned_csv_paths = sorted(
        {
            row["aligned_openface_csv"]
            for row in rows
            if Path(row["aligned_openface_csv"]).exists()
        }
    )
    aligned_jpg_paths = sorted(
        {
            row[field]
            for row in rows
            for field in (
                "previous_aligned_path",
                "target_placeholder_path",
                "next_aligned_path",
            )
            if row.get(field) and Path(row[field]).exists()
        }
    )
    input_files = {
        "audit_manifest": {"path": str(audit_manifest_path), "sha256": _sha256_file(audit_manifest_path)},
        "frame_failure_manifest": {
            "path": str(frame_manifest_path),
            "sha256": _sha256_file(frame_manifest_path),
        },
        "source_presence_gate": {
            "path": str(source_presence_gate),
            "sha256": _sha256_file(source_presence_gate),
        },
        "raw_videos": [
            {"path": value, "sha256": _sha256_file(value)} for value in raw_video_paths
        ],
        "raw_openface_csvs": [
            {"path": value, "sha256": _sha256_file(value)} for value in raw_csv_paths
        ],
        "aligned_openface_csvs": [
            {"path": value, "sha256": _sha256_file(value)} for value in aligned_csv_paths
        ],
        "aligned_jpgs": [
            {"path": value, "sha256": _sha256_file(value)} for value in aligned_jpg_paths
        ],
    }
    payload = {
        "audit": "real raw-frame to aligned-space warp smoke",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(shlex.quote(item) for item in sys.argv),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "split": split,
        "requested_max_frames": int(max_frames),
        "attempted_frames": len(rows),
        "automatic_pass_review_required_frames": passed,
        "overall_status": (
            "AUTO_PASS_REVIEW_REQUIRED" if passed == int(max_frames) else "BLOCKED_INSUFFICIENT_AUTO_PASS"
        ),
        "training_use_authorized": False,
        "face_landmark_rerun_performed": False,
        "aligned_neighbor_synthesis_used": False,
        "target_raw_pixels_used": True,
        "relocated_image_root": relocation,
        "thresholds": {
            "max_anchor_gap": int(max_anchor_gap),
            "max_fb_error_px": float(max_fb_error_px),
            "min_tracking_valid_ratio": float(min_tracking_valid_ratio),
            "min_transform_inlier_ratio": float(min_transform_inlier_ratio),
            "max_transform_rmse_px": float(max_transform_rmse_px),
            "ransac_threshold_px": float(ransac_threshold_px),
            "max_transform_disagreement_px": float(max_transform_disagreement_px),
            "min_warped_landmark_in_bounds_ratio": float(
                min_warped_landmark_in_bounds_ratio
            ),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
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
        "inputs": input_files,
        "outputs": {
            "table": {"path": str(table_path), "sha256": _sha256_file(table_path)},
            "report": {"path": str(report_path), "sha256": _sha256_file(report_path)},
            "derived": [
                {"path": row["derived_path"], "sha256": row["derived_sha256"]}
                for row in rows
                if row.get("derived_path")
            ],
            "contact_sheets": [
                {"path": row["contact_sheet"], "sha256": _sha256_file(row["contact_sheet"])}
                for row in rows
                if row.get("contact_sheet")
            ],
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [table_path, report_path, manifest_path]
