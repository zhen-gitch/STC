"""Train-only temporal exposure stability audit.

This module scans aligned JPG frames read-only, calibrates one full-video
luma-IQR threshold from train-split normal-exposure videos, and applies that
threshold only as a review triage for audited exposure candidates.  It never
reads BDI labels or prediction metrics, never chooses segment boundaries, and
never materializes adjusted images.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shlex
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from src.diagnostics.frame_recovery import (
    IMAGE_SUFFIXES,
    FrameRecoveryPolicy,
    _frame_id,
    _git_value,
    _read_csv,
    _sha256_file,
    _write_csv,
    classify_failed_frame,
    ensure_dir,
    inspect_frame,
    validate_relocated_image_root,
)


FRAME_LUMA_FIELDS = [
    "split",
    "cohort",
    "video_id",
    "frame_id",
    "image_path",
    "decode_status",
    "pure_black",
    "visible_luma_median",
]

VIDEO_STABILITY_FIELDS = [
    "split",
    "cohort",
    "video_id",
    "exposure_status",
    "expected_frame_count",
    "image_count",
    "visible_frame_count",
    "pure_black_frame_count",
    "unreadable_frame_count",
    "visible_coverage_ratio",
    "frame_luma_q05",
    "frame_luma_q10",
    "frame_luma_q25",
    "frame_luma_median",
    "frame_luma_q75",
    "frame_luma_q90",
    "frame_luma_q95",
    "frame_luma_iqr",
    "frame_luma_q90_q10_span",
    "adjacent_abs_delta_median",
    "adjacent_abs_delta_q95",
    "adjacent_abs_delta_max",
    "train_normal_iqr_q90_threshold",
    "train_normal_iqr_q95_reference",
    "iqr_to_q90_ratio",
    "temporal_stability_class",
    "review_route",
    "issues",
]

THRESHOLD_FIELDS = [
    "source_cohort",
    "reference_video_count",
    "metric",
    "quantile",
    "value",
    "numpy_quantile_method",
    "role",
]


def _safe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def summarize_frame_luma(frame_rows, summary):
    """Summarize visible frame-level luma without filling missing frames."""

    visible = [
        row
        for row in frame_rows
        if row["decode_status"] == "OK"
        and not row["pure_black"]
        and _safe_float(row.get("visible_luma_median")) is not None
    ]
    values = np.asarray(
        [float(row["visible_luma_median"]) for row in visible],
        dtype=np.float64,
    )
    quantiles = [None] * 7
    if values.size:
        quantiles = np.quantile(
            values,
            [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
            method="linear",
        ).tolist()

    visible_by_frame = {
        int(row["frame_id"]): float(row["visible_luma_median"])
        for row in visible
        if row.get("frame_id") is not None
    }
    adjacent_deltas = np.asarray(
        [
            abs(visible_by_frame[frame + 1] - value)
            for frame, value in visible_by_frame.items()
            if frame + 1 in visible_by_frame
        ],
        dtype=np.float64,
    )
    adjacent_median = float(np.median(adjacent_deltas)) if adjacent_deltas.size else None
    adjacent_q95 = (
        float(np.quantile(adjacent_deltas, 0.95, method="linear"))
        if adjacent_deltas.size
        else None
    )
    adjacent_max = float(adjacent_deltas.max()) if adjacent_deltas.size else None

    expected = int(summary["frame_count"])
    image_count = len(frame_rows)
    issues = []
    if image_count != expected:
        issues.append(f"image_count_mismatch:{image_count}!={expected}")
    unreadable_count = sum(row["decode_status"] != "OK" for row in frame_rows)
    pure_black_count = sum(row["decode_status"] == "OK" and row["pure_black"] for row in frame_rows)
    if unreadable_count:
        issues.append(f"unreadable_frames:{unreadable_count}")
    if pure_black_count:
        issues.append(f"pure_black_frames:{pure_black_count}")
    if not values.size:
        issues.append("no_visible_luma")

    q05, q10, q25, median, q75, q90, q95 = quantiles
    return {
        "split": summary["split"],
        "cohort": summary["cohort"],
        "video_id": summary["video_id"],
        "exposure_status": summary["video_exposure_status"],
        "expected_frame_count": expected,
        "image_count": image_count,
        "visible_frame_count": int(values.size),
        "pure_black_frame_count": pure_black_count,
        "unreadable_frame_count": unreadable_count,
        "visible_coverage_ratio": float(values.size / expected) if expected else 0.0,
        "frame_luma_q05": q05,
        "frame_luma_q10": q10,
        "frame_luma_q25": q25,
        "frame_luma_median": median,
        "frame_luma_q75": q75,
        "frame_luma_q90": q90,
        "frame_luma_q95": q95,
        "frame_luma_iqr": (float(q75 - q25) if q25 is not None and q75 is not None else None),
        "frame_luma_q90_q10_span": (
            float(q90 - q10) if q10 is not None and q90 is not None else None
        ),
        "adjacent_abs_delta_median": adjacent_median,
        "adjacent_abs_delta_q95": adjacent_q95,
        "adjacent_abs_delta_max": adjacent_max,
        "train_normal_iqr_q90_threshold": None,
        "train_normal_iqr_q95_reference": None,
        "iqr_to_q90_ratio": None,
        "temporal_stability_class": "",
        "review_route": "",
        "issues": ";".join(issues),
    }


def calibrate_temporal_iqr_thresholds(
    video_rows,
    stability_quantile=0.90,
    diagnostic_quantiles=(0.50, 0.75, 0.80, 0.90, 0.95),
):
    """Calibrate q90 from train-normal videos and annotate all rows.

    The default q90 rule is pre-registered: candidate IQR above train-normal
    q90 requires segment review.  q95 is descriptive priority stratification
    only and cannot relax the q90 decision.
    """

    stability_quantile = float(stability_quantile)
    diagnostic_quantiles = tuple(float(value) for value in diagnostic_quantiles)
    if not 0.0 < stability_quantile < 1.0:
        raise ValueError("stability_quantile must lie strictly inside (0, 1)")
    if stability_quantile != 0.90:
        raise ValueError("the preregistered temporal stability quantile is fixed at 0.90")
    if 0.90 not in diagnostic_quantiles or 0.95 not in diagnostic_quantiles:
        raise ValueError("diagnostic_quantiles must include 0.90 and 0.95")
    if any(not 0.0 < value < 1.0 for value in diagnostic_quantiles):
        raise ValueError("diagnostic quantiles must lie strictly inside (0, 1)")

    reference = [
        float(row["frame_luma_iqr"])
        for row in video_rows
        if row["cohort"] == "train_normal_reference"
        and _safe_float(row.get("frame_luma_iqr")) is not None
    ]
    if not reference:
        raise ValueError("no valid train-normal full-frame luma IQR values")
    values = np.asarray(reference, dtype=np.float64)
    threshold_values = {
        quantile: float(np.quantile(values, quantile, method="linear"))
        for quantile in diagnostic_quantiles
    }
    q90 = threshold_values[0.90]
    q95 = threshold_values[0.95]
    if q90 <= 0.0:
        raise ValueError("train-normal q90 IQR threshold must be positive")

    threshold_rows = []
    for quantile in diagnostic_quantiles:
        if quantile == 0.90:
            role = "preregistered_stable_whole_video_upper_bound"
        elif quantile == 0.95:
            role = "diagnostic_extreme_instability_reference_only"
        else:
            role = "descriptive_reference_quantile"
        threshold_rows.append(
            {
                "source_cohort": "train_split_normal_exposure_full_frame_luma_iqr",
                "reference_video_count": len(reference),
                "metric": "frame_luma_q75_minus_q25",
                "quantile": quantile,
                "value": threshold_values[quantile],
                "numpy_quantile_method": "linear",
                "role": role,
            }
        )

    annotated = []
    for source in video_rows:
        row = dict(source)
        iqr = _safe_float(row.get("frame_luma_iqr"))
        row["train_normal_iqr_q90_threshold"] = q90
        row["train_normal_iqr_q95_reference"] = q95
        row["iqr_to_q90_ratio"] = (iqr / q90) if iqr is not None else None
        if iqr is None:
            row["temporal_stability_class"] = "unavailable"
            row["review_route"] = "keep_raw_pending_data_repair"
        elif iqr <= q90:
            row["temporal_stability_class"] = "within_train_normal_q90"
            row["review_route"] = "whole_video_curve_eligible_pending_visual_review"
        elif iqr <= q95:
            row["temporal_stability_class"] = "above_q90_at_or_below_q95"
            row["review_route"] = "segment_review_required"
        else:
            row["temporal_stability_class"] = "above_train_normal_q95"
            row["review_route"] = "segment_review_required_priority"
        annotated.append(row)
    return annotated, threshold_rows


def _scan_video(task):
    image_root, summary, policy = task
    video_dir = Path(image_root) / summary["video_id"]
    paths = []
    if video_dir.is_dir():
        paths = sorted(
            (
                path
                for path in video_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda path: (_frame_id(path) is None, _frame_id(path) or 0, path.name),
        )
    frame_rows = []
    for path in paths:
        metrics = inspect_frame(path, policy)
        failure_type = classify_failed_frame(metrics, policy)
        pure_black = failure_type == "pure_black"
        frame_rows.append(
            {
                "split": summary["split"],
                "cohort": summary["cohort"],
                "video_id": summary["video_id"],
                "frame_id": _frame_id(path),
                "image_path": str(path),
                "decode_status": metrics.get("decode_status", "ERROR"),
                "pure_black": pure_black,
                "visible_luma_median": (
                    metrics.get("luma_median")
                    if metrics.get("decode_status") == "OK" and not pure_black
                    else None
                ),
            }
        )
    return summarize_frame_luma(frame_rows, summary), frame_rows


def _write_report(path, video_rows, threshold_rows, relocation_evidence):
    q90 = next(row["value"] for row in threshold_rows if row["quantile"] == 0.90)
    q95 = next(row["value"] for row in threshold_rows if row["quantile"] == 0.95)
    references = [row for row in video_rows if row["cohort"] == "train_normal_reference"]
    candidates = [row for row in video_rows if row["cohort"] == "exposure_candidate"]
    route_counts = Counter(row["review_route"] for row in candidates)
    ranked = sorted(
        candidates,
        key=lambda row: _safe_float(row.get("frame_luma_iqr")) or -1.0,
        reverse=True,
    )
    lines = [
        "# Exposure temporal stability audit",
        "",
        "- This audit is read-only and did not run OpenFace or materialize adjusted images.",
        "- Calibration used only train-split videos previously classified as normal exposure.",
        "- BDI labels, predictions, validation metrics, and test metrics were not read.",
        "- The preregistered stable-whole-video upper bound is train-normal full-frame luma-IQR q90.",
        "- q95 is reported only to prioritize extreme manual review; it cannot relax the q90 rule.",
        f"- Train-normal reference videos: {len(references)}.",
        f"- Exposure candidate videos: {len(candidates)}.",
        f"- Frozen q90 IQR threshold: {q90:.6f}.",
        f"- Diagnostic q95 IQR reference: {q95:.6f}.",
        f"- Relocated image-root authorization: {relocation_evidence['mode']}.",
        "",
        "## Candidate triage counts",
        "",
        "| Review route | Videos |",
        "|---|---:|",
    ]
    for route, count in sorted(route_counts.items()):
        lines.append(f"| {route} | {count} |")
    lines.extend(
        [
            "",
            "## Candidates ranked by full-frame luma IQR",
            "",
            "| Video | Split | Exposure | IQR | IQR/q90 | Stability class | Review route |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in ranked:
        ratio = _safe_float(row.get("iqr_to_q90_ratio"))
        lines.append(
            f"| {row['video_id']} | {row['split']} | {row['exposure_status']} | "
            f"{float(row['frame_luma_iqr']):.6f} | {ratio:.3f} | "
            f"{row['temporal_stability_class']} | {row['review_route']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- IQR at or below q90 makes a whole-video fixed curve eligible for visual review; it does not approve the curve.",
            "- IQR above q90 requires manual segment review or keep_raw; the audit does not infer segment boundaries.",
            "- Occlusion-, pose-, or expression-driven luma changes must not be converted into content-conditioned exposure segments.",
            "- Safety-band placement remains separate from clipped-detail recoverability.",
            "",
        ]
    )
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_exposure_temporal_stability_audit(
    audit_dir,
    image_root,
    output_dir,
    image_integrity_comparison_summary=None,
    workers=None,
    stability_quantile=0.90,
    max_reference_videos=None,
    max_candidate_videos=None,
    project_root=None,
):
    """Scan train-normal references and exposure candidates at full frame rate."""

    audit_dir = Path(audit_dir).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output_dir must be empty or absent: {output_dir}")

    audit_manifest_path = audit_dir / "run_manifest.json"
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    policy = FrameRecoveryPolicy(**audit_manifest["policy"]).validate()
    audited_root = Path(audit_manifest["image_root"]).expanduser().resolve()
    relocation_evidence = validate_relocated_image_root(
        image_root,
        audited_root,
        image_integrity_comparison_summary,
        project_root,
    )
    summaries = _read_csv(audit_dir / "tables" / "video_failure_summary.csv")
    references = [
        {**row, "cohort": "train_normal_reference"}
        for row in summaries
        if row["split"] == "train" and row["video_exposure_status"] == "normal"
    ]
    candidates = [
        {**row, "cohort": "exposure_candidate"}
        for row in summaries
        if row["video_exposure_status"] in {"underexposed", "overexposed"}
    ]
    references.sort(key=lambda row: row["video_id"])
    candidates.sort(key=lambda row: row["video_id"])
    if max_reference_videos is not None:
        references = references[: max(0, int(max_reference_videos))]
    if max_candidate_videos is not None:
        candidates = candidates[: max(0, int(max_candidate_videos))]
    if not references:
        raise ValueError("no train-normal reference videos selected")
    if not candidates:
        raise ValueError("no exposure candidate videos selected")

    tasks = [(image_root, row, policy) for row in references + candidates]
    worker_count = max(1, int(workers or min(8, os.cpu_count() or 1)))
    executor = None
    results = map(_scan_video, tasks)
    if worker_count != 1:
        executor = ThreadPoolExecutor(max_workers=worker_count)
        results = executor.map(_scan_video, tasks)
    video_rows, frame_rows = [], []
    try:
        for index, (video_row, video_frames) in enumerate(results, 1):
            video_rows.append(video_row)
            frame_rows.extend(video_frames)
            if index % 10 == 0 or index == len(tasks):
                print(
                    f"[EXPOSURE_TEMPORAL_STABILITY] processed {index}/{len(tasks)} videos",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown()

    video_rows, threshold_rows = calibrate_temporal_iqr_thresholds(
        video_rows,
        stability_quantile=stability_quantile,
    )
    video_rows.sort(key=lambda row: (row["cohort"], row["video_id"]))
    frame_rows.sort(key=lambda row: (row["cohort"], row["video_id"], row["frame_id"] or -1))

    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    frame_path = _write_csv(tables_dir / "frame_luma.csv", frame_rows, FRAME_LUMA_FIELDS)
    video_path = _write_csv(
        tables_dir / "video_temporal_stability.csv",
        video_rows,
        VIDEO_STABILITY_FIELDS,
    )
    threshold_path = _write_csv(
        tables_dir / "temporal_iqr_thresholds.csv",
        threshold_rows,
        THRESHOLD_FIELDS,
    )
    report_path = _write_report(
        reports_dir / "exposure_temporal_stability_report.md",
        video_rows,
        threshold_rows,
        relocation_evidence,
    )

    manifest = {
        "audit": "train-only full-frame exposure temporal stability",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(shlex.quote(value) for value in sys.argv),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "audit_dir": str(audit_dir),
        "audit_manifest_sha256": _sha256_file(audit_manifest_path),
        "source_image_root": str(image_root),
        "audited_source_image_root": str(audited_root),
        "image_root_relocation_evidence": relocation_evidence,
        "source_tree_modified": False,
        "formal_materialization_performed": False,
        "face_landmark_rerun_performed": False,
        "bdi_label_or_prediction_metric_access": False,
        "policy": asdict(policy),
        "temporal_stability_policy": {
            "metric": "full-frame visible luma q75 minus q25",
            "reference_cohort": "train split normal-exposure videos only",
            "stability_quantile": float(stability_quantile),
            "decision_rule": "candidate IQR > train-normal q90 requires segment review",
            "q95_role": "priority stratification only; cannot relax q90",
            "segment_boundary_inference_performed": False,
            "numpy_quantile_method": "linear",
        },
        "workers": worker_count,
        "max_reference_videos": max_reference_videos,
        "max_candidate_videos": max_candidate_videos,
        "reference_video_count": len(references),
        "candidate_video_count": len(candidates),
        "frame_row_count": len(frame_rows),
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
                "path": str(project_root / "scripts" / "audit_exposure_temporal_stability.py"),
                "sha256": _sha256_file(
                    project_root / "scripts" / "audit_exposure_temporal_stability.py"
                ),
            },
        },
        "outputs": {
            "frame_luma": {"path": str(frame_path), "sha256": _sha256_file(frame_path)},
            "video_temporal_stability": {
                "path": str(video_path),
                "sha256": _sha256_file(video_path),
            },
            "temporal_iqr_thresholds": {
                "path": str(threshold_path),
                "sha256": _sha256_file(threshold_path),
            },
            "report": {"path": str(report_path), "sha256": _sha256_file(report_path)},
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [frame_path, video_path, threshold_path, report_path, manifest_path]
