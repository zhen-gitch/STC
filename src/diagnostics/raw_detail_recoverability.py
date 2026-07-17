"""Read-only raw-video versus aligned-JPG detail recoverability audit.

The audit uses existing raw-video and aligned-image OpenFace CSV files. It
never launches OpenFace, modifies source data, or claims that clipped detail
has been reconstructed. Raw frames are robustly warped into the current
aligned geometry so clipping and texture are compared inside one face mask.
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
from PIL import Image, ImageDraw, ImageOps

from src.diagnostics.frame_recovery import validate_relocated_image_root


FRAME_FIELDS = [
    "cohort",
    "split",
    "video_id",
    "frame_id",
    "exposure_status",
    "raw_video_path",
    "aligned_image_path",
    "raw_openface_csv",
    "aligned_openface_csv",
    "raw_openface_success",
    "aligned_openface_success",
    "raw_openface_confidence",
    "aligned_openface_confidence",
    "transform_inlier_ratio",
    "transform_rmse_px",
    "face_mask_pixels",
    "raw_low_clip_ratio",
    "aligned_low_clip_ratio",
    "aligned_minus_raw_low_clip",
    "raw_high_clip_ratio",
    "aligned_high_clip_ratio",
    "aligned_minus_raw_high_clip",
    "raw_luma_span_q90_q10",
    "aligned_luma_span_q90_q10",
    "raw_gradient_energy",
    "aligned_gradient_energy",
    "raw_to_aligned_gradient_ratio",
    "raw_laplacian_variance",
    "aligned_laplacian_variance",
    "raw_to_aligned_laplacian_ratio",
    "raw_entropy_bits",
    "aligned_entropy_bits",
    "raw_minus_aligned_entropy_bits",
    "relevant_clip_tail",
    "relevant_raw_clip_ratio",
    "relevant_aligned_clip_ratio",
    "relevant_clip_advantage",
    "frame_recoverability",
    "status",
    "issues",
]

VIDEO_FIELDS = [
    "split",
    "video_id",
    "exposure_status",
    "sampled_frame_count",
    "valid_comparison_count",
    "invalid_comparison_count",
    "raw_detail_recoverable_count",
    "raw_detail_recoverable_ratio",
    "raw_also_clipped_count",
    "raw_also_clipped_ratio",
    "raw_less_clipped_texture_inconclusive_count",
    "no_raw_recoverability_evidence_count",
    "median_relevant_raw_clip_ratio",
    "median_relevant_aligned_clip_ratio",
    "median_relevant_clip_advantage",
    "median_raw_to_aligned_gradient_ratio",
    "median_raw_minus_aligned_entropy_bits",
    "recommended_action",
    "review_status",
    "contact_sheet",
]

CALIBRATION_FIELDS = [
    "reference_video_count",
    "reference_valid_frame_count",
    "quantile",
    "raw_low_clip_upper",
    "raw_high_clip_upper",
    "low_clip_advantage_upper",
    "high_clip_advantage_upper",
    "gradient_ratio_upper",
    "laplacian_ratio_upper",
    "entropy_delta_upper",
    "min_practical_clip_advantage",
    "min_transform_inlier_ratio",
    "max_transform_rmse_px",
]

ALIGNMENT_INDICES = list(range(17, 68))


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
    value = _safe_float(value)
    return int(value) if value is not None and value.is_integer() else None


def _frame_id(path):
    groups = re.findall(r"\d+", Path(path).stem)
    return int(groups[-1]) if groups else None


def _source_split(split):
    return {"val": "dev", "validation": "dev"}.get(str(split).lower(), str(split).lower())


def _task_name(video_id):
    match = re.search(r"_(Freeform|Northwind)_", str(video_id))
    return match.group(1) if match else ""


def _raw_video_id(video_id):
    value = str(video_id)
    return value[: -len("_aligned")] if value.endswith("_aligned") else value


def _raw_video_path(dataset_root, split, video_id):
    return (
        Path(dataset_root)
        / _source_split(split)
        / _task_name(video_id)
        / f"{_raw_video_id(video_id)}.mp4"
    )


def _even_rows(rows, count):
    rows = sorted(rows, key=lambda row: int(row["frame_id"]))
    if not rows or count <= 0:
        return []
    indices = sorted(
        set(np.linspace(0, len(rows) - 1, min(len(rows), int(count))).round().astype(int))
    )
    return [rows[index] for index in indices]


def _landmark_map(path, selected_frames):
    selected_frames = set(int(value) for value in selected_frames)
    result = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for raw_row in csv.DictReader(handle):
            row = {str(key).strip(): value for key, value in raw_row.items()}
            frame = _safe_int(row.get("frame"))
            if frame not in selected_frames:
                continue
            points = []
            valid = True
            for index in range(68):
                x = _safe_float(row.get(f"x_{index}"))
                y = _safe_float(row.get(f"y_{index}"))
                if x is None or y is None:
                    valid = False
                    break
                points.append((x, y))
            result[frame] = {
                "success": _safe_int(row.get("success")),
                "confidence": _safe_float(row.get("confidence")),
                "points": np.asarray(points, dtype=np.float32) if valid else None,
            }
    return result


def _estimate_raw_to_aligned(raw_points, aligned_points, ransac_threshold):
    import cv2

    source = raw_points[ALIGNMENT_INDICES]
    target = aligned_points[ALIGNMENT_INDICES]
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_threshold),
        maxIters=5000,
        confidence=0.999,
        refineIters=20,
    )
    if matrix is None or inliers is None:
        return None, None, None
    inliers = inliers.reshape(-1).astype(bool)
    predicted = cv2.transform(source[None, :, :], matrix)[0]
    if not inliers.any():
        return matrix, 0.0, None
    rmse = float(np.sqrt(np.mean(np.sum((predicted[inliers] - target[inliers]) ** 2, axis=1))))
    return matrix, float(inliers.mean()), rmse


def _face_mask(shape, points, erosion_pixels=2):
    import cv2

    mask = np.zeros(shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(np.rint(points).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    if erosion_pixels > 0:
        size = 2 * int(erosion_pixels) + 1
        kernel = np.ones((size, size), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask.astype(bool)


def _luma(rgb):
    values = np.asarray(rgb, dtype=np.float32)
    return 0.2989 * values[:, :, 0] + 0.5870 * values[:, :, 1] + 0.1140 * values[:, :, 2]


def _normalized_texture_luma(luma, mask):
    values = luma[mask]
    if values.size < 64:
        return None
    low, high = np.quantile(values, [0.05, 0.95])
    if high - low < 1.0:
        return np.zeros_like(luma, dtype=np.float32)
    return np.clip((luma - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _face_metrics(rgb, mask):
    import cv2

    luma = _luma(rgb)
    values = luma[mask]
    if values.size < 64:
        return None
    q10, q50, q90 = np.quantile(values, [0.10, 0.50, 0.90])
    normalized = _normalized_texture_luma(luma, mask)
    if normalized is None:
        return None
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    laplacian = cv2.Laplacian(normalized, cv2.CV_32F, ksize=3)
    histogram, _ = np.histogram(normalized[mask], bins=64, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64)
    probabilities /= max(probabilities.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    return {
        "low_clip_ratio": float(np.mean(values <= 16.0)),
        "high_clip_ratio": float(np.mean(values >= 250.0)),
        "luma_median": float(q50),
        "luma_span_q90_q10": float(q90 - q10),
        "gradient_energy": float(np.mean(gradient[eroded])),
        "laplacian_variance": float(np.var(laplacian[eroded])),
        "entropy_bits": entropy,
    }


def _crop_raw_roi(raw_rgb, points, margin=0.20):
    height, width = raw_rgb.shape[:2]
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    dx = (x1 - x0) * float(margin)
    dy = (y1 - y0) * float(margin)
    left = max(0, int(math.floor(x0 - dx)))
    top = max(0, int(math.floor(y0 - dy)))
    right = min(width, int(math.ceil(x1 + dx)))
    bottom = min(height, int(math.ceil(y1 + dy)))
    if right <= left or bottom <= top:
        return None
    return raw_rgb[top:bottom, left:right]


def _process_frame(
    *,
    cohort,
    summary,
    frame_id,
    raw_bgr,
    aligned_rgb,
    raw_landmark,
    aligned_landmark,
    raw_video_path,
    aligned_path,
    raw_csv,
    aligned_csv,
    min_inlier_ratio,
    max_rmse,
    ransac_threshold,
):
    import cv2

    base = {
        "cohort": cohort,
        "split": summary["split"],
        "video_id": summary["video_id"],
        "frame_id": frame_id,
        "exposure_status": summary["video_exposure_status"],
        "raw_video_path": str(raw_video_path),
        "aligned_image_path": str(aligned_path),
        "raw_openface_csv": str(raw_csv),
        "aligned_openface_csv": str(aligned_csv),
        "raw_openface_success": raw_landmark.get("success") if raw_landmark else None,
        "aligned_openface_success": aligned_landmark.get("success") if aligned_landmark else None,
        "raw_openface_confidence": raw_landmark.get("confidence") if raw_landmark else None,
        "aligned_openface_confidence": aligned_landmark.get("confidence") if aligned_landmark else None,
        "status": "INVALID",
        "issues": "",
    }
    issues = []
    if raw_landmark is None or raw_landmark.get("points") is None:
        issues.append("raw_landmarks_missing")
    elif raw_landmark.get("success") != 1:
        issues.append("raw_openface_failed")
    if aligned_landmark is None or aligned_landmark.get("points") is None:
        issues.append("aligned_landmarks_missing")
    elif aligned_landmark.get("success") != 1:
        issues.append("aligned_openface_failed")
    if raw_bgr is None:
        issues.append("raw_frame_decode_failed")
    if issues:
        return {**base, "issues": ";".join(issues)}, None

    matrix, inlier_ratio, rmse = _estimate_raw_to_aligned(
        raw_landmark["points"],
        aligned_landmark["points"],
        ransac_threshold,
    )
    base["transform_inlier_ratio"] = inlier_ratio
    base["transform_rmse_px"] = rmse
    if matrix is None or inlier_ratio is None or rmse is None:
        return {**base, "issues": "transform_estimation_failed"}, None
    if inlier_ratio < min_inlier_ratio:
        issues.append("transform_inlier_ratio_below_gate")
    if rmse > max_rmse:
        issues.append("transform_rmse_above_gate")
    if issues:
        return {**base, "issues": ";".join(issues)}, None

    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    height, width = aligned_rgb.shape[:2]
    raw_warp = cv2.warpAffine(
        raw_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = _face_mask(aligned_rgb.shape, aligned_landmark["points"], erosion_pixels=2)
    raw_metrics = _face_metrics(raw_warp, mask)
    aligned_metrics = _face_metrics(aligned_rgb, mask)
    if raw_metrics is None or aligned_metrics is None:
        return {**base, "issues": "insufficient_common_face_mask"}, None

    gradient_ratio = raw_metrics["gradient_energy"] / max(aligned_metrics["gradient_energy"], 1e-8)
    laplacian_ratio = raw_metrics["laplacian_variance"] / max(
        aligned_metrics["laplacian_variance"], 1e-8
    )
    row = {
        **base,
        "face_mask_pixels": int(mask.sum()),
        "raw_low_clip_ratio": raw_metrics["low_clip_ratio"],
        "aligned_low_clip_ratio": aligned_metrics["low_clip_ratio"],
        "aligned_minus_raw_low_clip": (
            aligned_metrics["low_clip_ratio"] - raw_metrics["low_clip_ratio"]
        ),
        "raw_high_clip_ratio": raw_metrics["high_clip_ratio"],
        "aligned_high_clip_ratio": aligned_metrics["high_clip_ratio"],
        "aligned_minus_raw_high_clip": (
            aligned_metrics["high_clip_ratio"] - raw_metrics["high_clip_ratio"]
        ),
        "raw_luma_span_q90_q10": raw_metrics["luma_span_q90_q10"],
        "aligned_luma_span_q90_q10": aligned_metrics["luma_span_q90_q10"],
        "raw_gradient_energy": raw_metrics["gradient_energy"],
        "aligned_gradient_energy": aligned_metrics["gradient_energy"],
        "raw_to_aligned_gradient_ratio": gradient_ratio,
        "raw_laplacian_variance": raw_metrics["laplacian_variance"],
        "aligned_laplacian_variance": aligned_metrics["laplacian_variance"],
        "raw_to_aligned_laplacian_ratio": laplacian_ratio,
        "raw_entropy_bits": raw_metrics["entropy_bits"],
        "aligned_entropy_bits": aligned_metrics["entropy_bits"],
        "raw_minus_aligned_entropy_bits": (
            raw_metrics["entropy_bits"] - aligned_metrics["entropy_bits"]
        ),
        "status": "VALID",
        "issues": "",
    }
    if summary["video_exposure_status"] == "underexposed":
        row.update(
            {
                "relevant_clip_tail": "low",
                "relevant_raw_clip_ratio": raw_metrics["low_clip_ratio"],
                "relevant_aligned_clip_ratio": aligned_metrics["low_clip_ratio"],
                "relevant_clip_advantage": (
                    aligned_metrics["low_clip_ratio"] - raw_metrics["low_clip_ratio"]
                ),
            }
        )
    elif summary["video_exposure_status"] == "overexposed":
        row.update(
            {
                "relevant_clip_tail": "high",
                "relevant_raw_clip_ratio": raw_metrics["high_clip_ratio"],
                "relevant_aligned_clip_ratio": aligned_metrics["high_clip_ratio"],
                "relevant_clip_advantage": (
                    aligned_metrics["high_clip_ratio"] - raw_metrics["high_clip_ratio"]
                ),
            }
        )
    preview = {
        "frame_id": frame_id,
        "raw_roi": _crop_raw_roi(raw_rgb, raw_landmark["points"]),
        "raw_warp": raw_warp,
        "aligned": aligned_rgb,
        "row": row,
    }
    return row, preview


def _calibrate(
    reference_rows,
    quantile,
    min_practical_clip_advantage,
    min_inlier_ratio,
    max_rmse,
    reference_video_count,
):
    valid = [row for row in reference_rows if row["status"] == "VALID"]
    if len(valid) < 20:
        raise ValueError("at least 20 valid train-normal reference frames are required")

    def q(field):
        return float(np.quantile([float(row[field]) for row in valid], quantile))

    return {
        "reference_video_count": reference_video_count,
        "reference_valid_frame_count": len(valid),
        "quantile": quantile,
        "raw_low_clip_upper": q("raw_low_clip_ratio"),
        "raw_high_clip_upper": q("raw_high_clip_ratio"),
        "low_clip_advantage_upper": q("aligned_minus_raw_low_clip"),
        "high_clip_advantage_upper": q("aligned_minus_raw_high_clip"),
        "gradient_ratio_upper": q("raw_to_aligned_gradient_ratio"),
        "laplacian_ratio_upper": q("raw_to_aligned_laplacian_ratio"),
        "entropy_delta_upper": q("raw_minus_aligned_entropy_bits"),
        "min_practical_clip_advantage": min_practical_clip_advantage,
        "min_transform_inlier_ratio": min_inlier_ratio,
        "max_transform_rmse_px": max_rmse,
    }


def _classify_candidate_frame(row, calibration):
    if row["status"] != "VALID":
        return "invalid_comparison"
    if row["relevant_clip_tail"] == "low":
        raw_clip_upper = calibration["raw_low_clip_upper"]
        advantage_upper = calibration["low_clip_advantage_upper"]
    else:
        raw_clip_upper = calibration["raw_high_clip_upper"]
        advantage_upper = calibration["high_clip_advantage_upper"]
    raw_also_clipped = row["relevant_raw_clip_ratio"] > raw_clip_upper
    clipping_advantage = row["relevant_clip_advantage"] > max(
        advantage_upper,
        calibration["min_practical_clip_advantage"],
    )
    texture_advantage = (
        row["raw_to_aligned_gradient_ratio"] > calibration["gradient_ratio_upper"]
        or row["raw_to_aligned_laplacian_ratio"] > calibration["laplacian_ratio_upper"]
        or row["raw_minus_aligned_entropy_bits"] > calibration["entropy_delta_upper"]
    )
    if raw_also_clipped:
        return "raw_also_clipped"
    if clipping_advantage and texture_advantage:
        return "raw_detail_recoverable"
    if clipping_advantage:
        return "raw_less_clipped_texture_inconclusive"
    return "no_raw_recoverability_evidence"


def _thumbnail(rgb, width, label):
    if rgb is None:
        canvas = Image.new("RGB", (width, width + 22), (40, 40, 40))
        ImageDraw.Draw(canvas).text((4, width + 4), f"{label} unavailable", fill="white")
        return canvas
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    canvas = Image.new("RGB", (width, width + 22), "white")
    resampling = getattr(Image, "Resampling", Image)
    contained = ImageOps.contain(image, (width, width), method=resampling.BILINEAR)
    canvas.paste(contained, ((width - contained.width) // 2, (width - contained.height) // 2))
    ImageDraw.Draw(canvas).text((4, width + 4), label, fill="black")
    return canvas


def _write_contact_sheet(path, previews, columns=4, thumb_width=224):
    columns = max(1, int(columns))
    thumb_width = max(64, int(thumb_width))
    label_width = 92
    header_height = 28
    cell_height = thumb_width + 22
    pages = math.ceil(len(previews) / columns)
    canvas = Image.new(
        "RGB",
        (label_width + columns * thumb_width, header_height + pages * 3 * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 7), "Raw-vs-aligned detail audit; no detail is synthesized", fill="black")
    labels = ("RAW ROI", "RAW WARP", "ALIGNED")
    keys = ("raw_roi", "raw_warp", "aligned")
    for index, preview in enumerate(previews):
        page = index // columns
        column = index % columns
        x = label_width + column * thumb_width
        for row_index, (label, key) in enumerate(zip(labels, keys)):
            y = header_height + (page * 3 + row_index) * cell_height
            if column == 0:
                draw.text((5, y + 7), label, fill="black")
            metrics = preview["row"]
            suffix = f"f{preview['frame_id']}"
            if row_index == 1:
                suffix += f" Gx={metrics['raw_to_aligned_gradient_ratio']:.2f}"
            elif row_index == 2:
                suffix += f" {metrics['frame_recoverability']}"
            canvas.paste(_thumbnail(preview[key], thumb_width, suffix), (x, y))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=92, subsampling=0, optimize=False)
    return path


def _process_video_samples(
    *,
    cohort,
    summary,
    sample_rows,
    dataset_root,
    image_root,
    raw_openface_root,
    aligned_openface_root,
    min_inlier_ratio,
    max_rmse,
    ransac_threshold,
):
    import cv2

    video_id = summary["video_id"]
    frame_ids = [int(row["frame_id"]) for row in sample_rows]
    raw_video_path = _raw_video_path(dataset_root, summary["split"], video_id)
    raw_csv = Path(raw_openface_root) / f"{_raw_video_id(video_id)}.csv"
    aligned_csv = Path(aligned_openface_root) / f"{video_id}.csv"
    missing = [str(path) for path in (raw_video_path, raw_csv, aligned_csv) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing raw/aligned audit inputs for {video_id}: {missing}")
    raw_landmarks = _landmark_map(raw_csv, frame_ids)
    aligned_landmarks = _landmark_map(aligned_csv, frame_ids)
    image_by_frame = {
        _frame_id(path): path
        for path in (Path(image_root) / video_id).iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    }
    capture = cv2.VideoCapture(str(raw_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open raw video: {raw_video_path}")
    rows, previews = [], []
    try:
        for frame_id in frame_ids:
            aligned_path = image_by_frame.get(frame_id)
            if aligned_path is None:
                raise FileNotFoundError(f"missing aligned image {video_id}:{frame_id}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
            ok, raw_bgr = capture.read()
            aligned_rgb = np.asarray(Image.open(aligned_path).convert("RGB"), dtype=np.uint8)
            row, preview = _process_frame(
                cohort=cohort,
                summary=summary,
                frame_id=frame_id,
                raw_bgr=raw_bgr if ok else None,
                aligned_rgb=aligned_rgb,
                raw_landmark=raw_landmarks.get(frame_id),
                aligned_landmark=aligned_landmarks.get(frame_id),
                raw_video_path=raw_video_path,
                aligned_path=aligned_path,
                raw_csv=raw_csv,
                aligned_csv=aligned_csv,
                min_inlier_ratio=min_inlier_ratio,
                max_rmse=max_rmse,
                ransac_threshold=ransac_threshold,
            )
            rows.append(row)
            if preview is not None:
                previews.append(preview)
    finally:
        capture.release()
    return rows, previews


def run_raw_detail_recoverability_audit(
    *,
    frame_audit_dir,
    dataset_root,
    image_root,
    raw_openface_root,
    aligned_openface_root,
    output_dir,
    image_integrity_comparison_summary=None,
    video_ids=None,
    max_videos=None,
    candidate_frames_per_video=32,
    reference_frames_per_video=4,
    contact_frames_per_video=8,
    contact_sheet_columns=4,
    calibration_quantile=0.95,
    min_practical_clip_advantage=0.01,
    min_transform_inlier_ratio=0.50,
    max_transform_rmse_px=2.50,
    ransac_threshold_px=2.0,
    project_root=None,
):
    """Run the train-calibrated raw-vs-aligned detail recoverability audit."""

    frame_audit_dir = Path(frame_audit_dir).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    raw_openface_root = Path(raw_openface_root).expanduser().resolve()
    aligned_openface_root = Path(aligned_openface_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output_dir must be empty or absent: {output_dir}")
    if not 0.50 <= calibration_quantile < 1.0:
        raise ValueError("calibration_quantile must lie in [0.50, 1.0)")
    if not 0.0 <= min_practical_clip_advantage <= 1.0:
        raise ValueError("min_practical_clip_advantage must lie in [0, 1]")

    frame_manifest_path = frame_audit_dir / "run_manifest.json"
    frame_manifest = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
    audited_root = Path(frame_manifest["image_root"]).expanduser().resolve()
    relocation_evidence = validate_relocated_image_root(
        image_root,
        audited_root,
        image_integrity_comparison_summary,
        project_root,
    )
    summaries = _read_csv(frame_audit_dir / "tables" / "video_failure_summary.csv")
    samples = _read_csv(frame_audit_dir / "tables" / "exposure_sample_manifest.csv")
    summary_by_video = {row["video_id"]: row for row in summaries}
    samples_by_video = defaultdict(list)
    for row in samples:
        if (
            row.get("decode_status") == "OK"
            and row.get("pure_black") not in {"1", "true", "True"}
            and _safe_int(row.get("frame_id")) is not None
        ):
            samples_by_video[row["video_id"]].append(row)

    candidates = [
        row
        for row in summaries
        if row["video_exposure_status"] in {"underexposed", "overexposed"}
    ]
    if video_ids:
        requested = {str(value) for value in video_ids}
        known = {row["video_id"] for row in candidates}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"requested videos are not exposure candidates: {unknown}")
        candidates = [row for row in candidates if row["video_id"] in requested]
    candidates.sort(key=lambda row: row["video_id"])
    if max_videos is not None:
        candidates = candidates[: max(0, int(max_videos))]
    if not candidates:
        raise ValueError("no exposure candidates selected")

    references = [
        row
        for row in summaries
        if row["split"] == "train" and row["video_exposure_status"] == "normal"
    ]
    references.sort(key=lambda row: row["video_id"])
    reference_rows = []
    for index, summary in enumerate(references, start=1):
        selected = _even_rows(samples_by_video.get(summary["video_id"], []), reference_frames_per_video)
        rows, _ = _process_video_samples(
            cohort="train_normal_reference",
            summary=summary,
            sample_rows=selected,
            dataset_root=dataset_root,
            image_root=image_root,
            raw_openface_root=raw_openface_root,
            aligned_openface_root=aligned_openface_root,
            min_inlier_ratio=min_transform_inlier_ratio,
            max_rmse=max_transform_rmse_px,
            ransac_threshold=ransac_threshold_px,
        )
        reference_rows.extend(rows)
        if index % 20 == 0 or index == len(references):
            print(f"[RAW_DETAIL_AUDIT] reference {index}/{len(references)} videos", flush=True)

    calibration = _calibrate(
        reference_rows,
        calibration_quantile,
        min_practical_clip_advantage,
        min_transform_inlier_ratio,
        max_transform_rmse_px,
        len(references),
    )
    candidate_rows = []
    preview_by_video = {}
    for index, summary in enumerate(candidates, start=1):
        selected = _even_rows(
            samples_by_video.get(summary["video_id"], []),
            candidate_frames_per_video,
        )
        rows, previews = _process_video_samples(
            cohort="exposure_candidate",
            summary=summary,
            sample_rows=selected,
            dataset_root=dataset_root,
            image_root=image_root,
            raw_openface_root=raw_openface_root,
            aligned_openface_root=aligned_openface_root,
            min_inlier_ratio=min_transform_inlier_ratio,
            max_rmse=max_transform_rmse_px,
            ransac_threshold=ransac_threshold_px,
        )
        for row in rows:
            row["frame_recoverability"] = _classify_candidate_frame(row, calibration)
        by_frame = {row["frame_id"]: row for row in rows}
        for preview in previews:
            preview["row"] = by_frame[preview["frame_id"]]
        candidate_rows.extend(rows)
        preview_by_video[summary["video_id"]] = previews
        print(f"[RAW_DETAIL_AUDIT] candidate {index}/{len(candidates)} videos", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = output_dir / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    video_rows = []
    for summary in candidates:
        video_id = summary["video_id"]
        rows = [row for row in candidate_rows if row["video_id"] == video_id]
        valid = [row for row in rows if row["status"] == "VALID"]
        counts = Counter(row["frame_recoverability"] for row in rows)
        valid_count = len(valid)
        recoverable_ratio = counts["raw_detail_recoverable"] / valid_count if valid_count else 0.0
        raw_clipped_ratio = counts["raw_also_clipped"] / valid_count if valid_count else 0.0
        if recoverable_ratio >= 0.20 and raw_clipped_ratio >= 0.20:
            action = "segment_review_required"
        elif recoverable_ratio >= 0.25:
            action = "raw_space_tone_then_warp_candidate"
        elif raw_clipped_ratio >= 0.50:
            action = "tone_only_or_keep_raw"
        else:
            action = "inconclusive_keep_raw_until_review"
        previews = preview_by_video[video_id]
        previews.sort(
            key=lambda item: (
                item["row"]["frame_recoverability"] != "raw_detail_recoverable",
                -float(item["row"].get("relevant_clip_advantage") or 0.0),
                item["frame_id"],
            )
        )
        previews = previews[: max(1, int(contact_frames_per_video))]
        contact_path = _write_contact_sheet(
            contact_dir / f"{video_id}.jpg",
            previews,
            columns=contact_sheet_columns,
        )

        def median(field):
            values = [float(row[field]) for row in valid if row.get(field) is not None]
            return float(np.median(values)) if values else None

        video_rows.append(
            {
                "split": summary["split"],
                "video_id": video_id,
                "exposure_status": summary["video_exposure_status"],
                "sampled_frame_count": len(rows),
                "valid_comparison_count": valid_count,
                "invalid_comparison_count": len(rows) - valid_count,
                "raw_detail_recoverable_count": counts["raw_detail_recoverable"],
                "raw_detail_recoverable_ratio": recoverable_ratio,
                "raw_also_clipped_count": counts["raw_also_clipped"],
                "raw_also_clipped_ratio": raw_clipped_ratio,
                "raw_less_clipped_texture_inconclusive_count": counts[
                    "raw_less_clipped_texture_inconclusive"
                ],
                "no_raw_recoverability_evidence_count": counts[
                    "no_raw_recoverability_evidence"
                ],
                "median_relevant_raw_clip_ratio": median("relevant_raw_clip_ratio"),
                "median_relevant_aligned_clip_ratio": median("relevant_aligned_clip_ratio"),
                "median_relevant_clip_advantage": median("relevant_clip_advantage"),
                "median_raw_to_aligned_gradient_ratio": median(
                    "raw_to_aligned_gradient_ratio"
                ),
                "median_raw_minus_aligned_entropy_bits": median(
                    "raw_minus_aligned_entropy_bits"
                ),
                "recommended_action": action,
                "review_status": "REVIEW_REQUIRED",
                "contact_sheet": str(contact_path),
            }
        )

    all_rows = sorted(
        [*reference_rows, *candidate_rows],
        key=lambda row: (row["cohort"], row["video_id"], int(row["frame_id"])),
    )
    video_rows.sort(key=lambda row: row["video_id"])
    frame_path = _write_csv(output_dir / "tables" / "raw_aligned_detail_frames.csv", all_rows, FRAME_FIELDS)
    video_path = _write_csv(
        output_dir / "tables" / "raw_aligned_detail_video_summary.csv",
        video_rows,
        VIDEO_FIELDS,
    )
    calibration_path = _write_csv(
        output_dir / "tables" / "train_normal_detail_calibration.csv",
        [calibration],
        CALIBRATION_FIELDS,
    )
    action_counts = Counter(row["recommended_action"] for row in video_rows)
    lines = [
        "# Raw-video vs Aligned-JPG Detail Recoverability Audit",
        "",
        "- This audit used existing raw/aligned OpenFace landmarks and did not run OpenFace.",
        "- Raw frames were robustly warped into the current aligned geometry before comparison.",
        "- Texture metrics use robustly normalized luma inside one eroded face hull.",
        "- The train-normal reference cohort calibrates all evidence thresholds without BDI labels.",
        f"- Raw recoverability also requires at least {100.0 * min_practical_clip_advantage:.1f} percentage point absolute clipping advantage over aligned JPG.",
        "- No output reconstructs or hallucinates clipped texture; every action remains REVIEW_REQUIRED.",
        f"- Train-normal reference videos/valid frames: {len(references)}/{calibration['reference_valid_frame_count']}.",
        f"- Exposure candidate videos: {len(video_rows)}.",
        "",
        "## Recommended action counts",
        "",
        "| Action | Videos |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in sorted(action_counts.items()))
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `raw_space_tone_then_warp_candidate` means raw evidence survives unusually better than aligned evidence; it is not approval to materialize.",
            "- `raw_also_clipped` means the original video already lost the relevant tail; only tone normalization or keep_raw is defensible.",
            "- Mixed temporal evidence requires segment review and fixed per-segment decisions.",
            "- Validation/test outputs cannot change calibration quantiles or routing thresholds.",
        ]
    )
    report_path = output_dir / "reports" / "raw_aligned_detail_recoverability_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    outputs = [frame_path, video_path, calibration_path, report_path]
    payload = {
        "audit": "raw-video versus aligned-JPG detail recoverability",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(shlex.quote(item) for item in sys.argv),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "frame_audit_dir": str(frame_audit_dir),
        "frame_audit_manifest_sha256": _sha256_file(frame_manifest_path),
        "dataset_root": str(dataset_root),
        "image_root": str(image_root),
        "audited_image_root": str(audited_root),
        "image_root_relocation_evidence": relocation_evidence,
        "raw_openface_root": str(raw_openface_root),
        "aligned_openface_root": str(aligned_openface_root),
        "source_tree_modified": False,
        "face_landmark_rerun_performed": False,
        "detail_reconstruction_performed": False,
        "calibration": calibration,
        "candidate_video_ids": [row["video_id"] for row in candidates],
        "candidate_frames_per_video": candidate_frames_per_video,
        "reference_frames_per_video": reference_frames_per_video,
        "contact_frames_per_video": contact_frames_per_video,
        "min_practical_clip_advantage": min_practical_clip_advantage,
        "ransac_threshold_px": ransac_threshold_px,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256_file(path)} for path in outputs
        },
        "contact_sheets": [
            {"path": row["contact_sheet"], "sha256": _sha256_file(row["contact_sheet"])}
            for row in video_rows
        ],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [*outputs, manifest_path]
