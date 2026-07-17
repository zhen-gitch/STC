import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.diagnostics.exposure_temporal_stability import (
    calibrate_temporal_iqr_thresholds,
    run_exposure_temporal_stability_audit,
    summarize_frame_luma,
)
from src.diagnostics.frame_recovery import FrameRecoveryPolicy, VIDEO_FIELDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "audit_exposure_temporal_stability",
    PROJECT_ROOT / "scripts" / "audit_exposure_temporal_stability.py",
)
audit_exposure_temporal_stability = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(audit_exposure_temporal_stability)


def _write_video(root, video_id, values):
    video_dir = root / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    for frame_id, value in enumerate(values, start=1):
        rgb = np.full((16, 16, 3), int(value), dtype=np.uint8)
        Image.fromarray(rgb).save(
            video_dir / f"frame_det_00_{frame_id:06d}.jpg",
            format="JPEG",
            quality=95,
            subsampling=0,
        )


def _summary(video_id, split, status, frame_count):
    row = {field: "" for field in VIDEO_FIELDS}
    row.update(
        {
            "split": split,
            "video_id": video_id,
            "frame_count": frame_count,
            "video_exposure_status": status,
        }
    )
    return row


def test_cli_freezes_q90_and_exposes_no_facelandmark_action():
    args = audit_exposure_temporal_stability.build_parser().parse_args(
        [
            "--audit-dir",
            "/audit",
            "--image-root",
            "/images",
            "--output-dir",
            "/output",
        ]
    )
    assert args.stability_quantile == 0.90
    assert all("facelandmark" not in key.lower() for key in vars(args))


def test_frame_summary_excludes_black_and_uses_consecutive_visible_deltas():
    rows = [
        {"frame_id": 1, "decode_status": "OK", "pure_black": False, "visible_luma_median": 10},
        {"frame_id": 2, "decode_status": "OK", "pure_black": True, "visible_luma_median": None},
        {"frame_id": 3, "decode_status": "OK", "pure_black": False, "visible_luma_median": 30},
        {"frame_id": 4, "decode_status": "OK", "pure_black": False, "visible_luma_median": 34},
    ]
    summary = summarize_frame_luma(
        rows,
        {
            "split": "train",
            "cohort": "train_normal_reference",
            "video_id": "video",
            "video_exposure_status": "normal",
            "frame_count": "4",
        },
    )
    assert summary["visible_frame_count"] == 3
    assert summary["pure_black_frame_count"] == 1
    assert summary["frame_luma_iqr"] == pytest.approx(12.0)
    assert summary["adjacent_abs_delta_median"] == pytest.approx(4.0)


def test_threshold_uses_only_train_normal_reference_and_q95_cannot_relax_q90():
    rows = []
    for index, iqr in enumerate([1.0, 2.0, 3.0], start=1):
        rows.append(
            {
                "cohort": "train_normal_reference",
                "video_id": f"ref_{index}",
                "frame_luma_iqr": iqr,
            }
        )
    rows.extend(
        [
            {"cohort": "exposure_candidate", "video_id": "candidate_a", "frame_luma_iqr": 2.7},
            {"cohort": "exposure_candidate", "video_id": "candidate_b", "frame_luma_iqr": 100.0},
        ]
    )
    annotated, thresholds = calibrate_temporal_iqr_thresholds(rows)
    q90 = next(row["value"] for row in thresholds if row["quantile"] == 0.90)
    assert q90 == pytest.approx(2.8)
    by_id = {row["video_id"]: row for row in annotated}
    assert by_id["candidate_a"]["review_route"] == "whole_video_curve_eligible_pending_visual_review"
    assert by_id["candidate_b"]["review_route"] == "segment_review_required_priority"
    with pytest.raises(ValueError, match="fixed at 0.90"):
        calibrate_temporal_iqr_thresholds(rows, stability_quantile=0.95)


def test_realistic_synthetic_run_writes_separate_frame_video_and_threshold_tables(tmp_path):
    image_root = tmp_path / "images"
    ref_a = "201_1_Freeform_video_aligned"
    ref_b = "202_1_Freeform_video_aligned"
    candidate = "203_1_Freeform_video_aligned"
    _write_video(image_root, ref_a, [80, 81, 80, 81])
    _write_video(image_root, ref_b, [90, 92, 94, 96])
    _write_video(image_root, candidate, [20, 20, 80, 80])

    audit_dir = tmp_path / "audit"
    (audit_dir / "tables").mkdir(parents=True)
    summaries = [
        _summary(ref_a, "train", "normal", 4),
        _summary(ref_b, "train", "normal", 4),
        _summary(candidate, "val", "underexposed", 4),
    ]
    with (audit_dir / "tables" / "video_failure_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=VIDEO_FIELDS)
        writer.writeheader()
        writer.writerows(summaries)
    (audit_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "image_root": str(image_root.resolve()),
                "policy": FrameRecoveryPolicy().__dict__,
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    generated = run_exposure_temporal_stability_audit(
        audit_dir=audit_dir,
        image_root=image_root,
        output_dir=output_dir,
        workers=1,
        project_root=PROJECT_ROOT,
    )
    assert len(generated) == 5
    with (output_dir / "tables" / "frame_luma.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        frame_rows = list(csv.DictReader(handle))
    with (output_dir / "tables" / "video_temporal_stability.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        video_rows = list(csv.DictReader(handle))
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(frame_rows) == 12
    assert len(video_rows) == 3
    assert manifest["reference_video_count"] == 2
    assert manifest["candidate_video_count"] == 1
    candidate_row = next(row for row in video_rows if row["video_id"] == candidate)
    assert candidate_row["review_route"] == "segment_review_required_priority"
    assert manifest["bdi_label_or_prediction_metric_access"] is False
    assert manifest["formal_materialization_performed"] is False
