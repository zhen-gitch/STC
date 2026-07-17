import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.diagnostics.face_usability import (
    POSE_LANDMARK_IDS,
    POSE_MODEL_POINTS,
    _candidate_priority,
    estimate_pose_degrees,
    estimate_pose_details,
    normalize_landmarks,
    run_face_usability_distribution_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "audit_face_usability",
    PROJECT_ROOT / "scripts" / "audit_face_usability.py",
)
audit_face_usability = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(audit_face_usability)


def _base_points():
    angles = np.linspace(0.0, 2.0 * np.pi, 68, endpoint=False)
    points = np.stack([56.0 + 30.0 * np.cos(angles), 56.0 + 38.0 * np.sin(angles)], axis=1)
    points[36:42] = np.array([[38, 48], [40, 46], [44, 46], [46, 48], [44, 50], [40, 50]])
    points[42:48] = np.array([[66, 48], [68, 46], [72, 46], [74, 48], [72, 50], [68, 50]])
    return points.astype(np.float64)


def _write_landmark_csv(path, frames, points, success=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["frame", "timestamp", "confidence", "success"] + [
        f"{axis}_{index}" for axis in ("x", "y") for index in range(68)
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for offset, frame in enumerate(frames):
            frame_points = points + np.array([offset * 0.2, 0.0])
            row = {
                "frame": frame,
                "timestamp": offset / 30.0,
                "confidence": 0.95,
                "success": 1 if success is None else success[offset],
            }
            for index in range(68):
                row[f"x_{index}"] = frame_points[index, 0] if row["success"] else 0.0
                row[f"y_{index}"] = frame_points[index, 1] if row["success"] else 0.0
            writer.writerow(row)


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_normalize_landmarks_is_translation_scale_roll_invariant():
    points = _base_points()
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = 1.7 * (points @ rotation.T) + np.array([18.0, -11.0])

    assert np.allclose(normalize_landmarks(points), normalize_landmarks(transformed), atol=1e-6)


def test_pose_estimation_recovers_frontal_projection():
    camera = np.array([[112.0, 0.0, 56.0], [0.0, 112.0, 56.0], [0.0, 0.0, 1.0]])
    image_points, _ = cv2.projectPoints(
        POSE_MODEL_POINTS,
        np.zeros((3, 1)),
        np.array([[0.0], [0.0], [1000.0]]),
        camera,
        np.zeros((4, 1)),
    )
    points = _base_points()
    points[POSE_LANDMARK_IDS] = image_points.reshape(-1, 2)

    yaw, pitch, roll = estimate_pose_degrees(points, 112, 112)

    assert abs(yaw) < 1e-3
    assert abs(pitch) < 1e-3
    assert abs(roll) < 1e-3
    details = estimate_pose_details(points, 112, 112)
    assert details["solver"] == "ITERATIVE"
    assert details["min_depth"] > 0.0
    assert details["reprojection_rmse"] < 1e-5


def test_frontal_control_rejects_missing_pose_or_residual():
    complete = {
        "landmark_pose_yaw_deg": 1.0,
        "landmark_pose_pitch_deg": -2.0,
        "transform_residual": 0.03,
    }

    assert _candidate_priority("frontal_control", complete) == -3.3
    for missing_field in complete:
        incomplete = dict(complete)
        incomplete[missing_field] = None
        assert _candidate_priority("frontal_control", incomplete) is None


def test_cli_defaults_do_not_accept_threshold_manifest():
    args = audit_face_usability.build_parser().parse_args(
        [
            "--dataset-split-file",
            "/split.json",
            "--image-root",
            "/images",
            "--aligned-landmark-root",
            "/landmarks",
            "--frame-audit-dir",
            "/audit",
            "--source-presence-gate",
            "/gate.csv",
            "--exposure-review-manifest",
            "/exposure.csv",
            "--output-dir",
            "/output",
        ]
    )

    assert args.canonical_sample_step == 30
    assert args.contact_frames_per_reason == 12
    assert "threshold" not in vars(args)


def test_small_phase1_audit_writes_pending_distribution_outputs(tmp_path):
    video_id = "203_1_Freeform_video_aligned"
    image_root = tmp_path / "images"
    video_dir = image_root / video_id
    video_dir.mkdir(parents=True)
    points = _base_points()
    for frame in range(1, 7):
        image = np.full((112, 112, 3), 120 + frame, dtype=np.uint8)
        Image.fromarray(image).save(video_dir / f"frame_{frame:06d}.jpg")
    landmark_root = tmp_path / "landmarks"
    _write_landmark_csv(
        landmark_root / f"{video_id}.csv",
        range(1, 7),
        points,
        success=[1, 1, 0, 1, 1, 1],
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"train": [video_id], "val": [], "test": []}), encoding="utf-8")

    frame_audit = tmp_path / "frame_audit"
    failure_path = _write_csv(
        frame_audit / "tables" / "frame_failure_manifest.csv",
        ["split", "video_id", "frame_id", "failure_type"],
        [{"split": "train", "video_id": video_id, "frame_id": 3, "failure_type": "visible_detection_failed"}],
    )
    (frame_audit / "run_manifest.json").write_text(
        json.dumps(
            {
                "image_root": str(image_root.resolve()),
                "outputs": {"frame_failure_manifest": {"sha256": _sha256(failure_path)}},
            }
        ),
        encoding="utf-8",
    )
    gate_path = _write_csv(
        tmp_path / "source_gate.csv",
        [
            "split",
            "video_id",
            "frame_id",
            "source_presence_status",
            "recovery_permission",
            "review_status",
        ],
        [],
    )
    exposure_path = _write_csv(
        tmp_path / "exposure.csv",
        [
            "video_id",
            "segment_start_frame",
            "segment_end_frame",
            "review_status",
            "review_decision",
        ],
        [],
    )

    generated = run_face_usability_distribution_audit(
        dataset_split_file=split_path,
        image_root=image_root,
        aligned_landmark_root=landmark_root,
        frame_audit_dir=frame_audit,
        source_presence_gate=gate_path,
        exposure_review_manifest=exposure_path,
        output_dir=tmp_path / "output",
        canonical_sample_step=1,
        contact_frames_per_reason=1,
        project_root=PROJECT_ROOT,
    )

    assert len(generated) == 8
    with (tmp_path / "output/tables/face_usability_phase1_frames.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert rows[2]["face_usability_status"] == "face_present_low_quality"
    assert rows[2]["face_hull_visible_ratio"] == ""
    assert rows[0]["face_usability_status"] == "pending_threshold_review"
    manifest = json.loads((tmp_path / "output/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["threshold_manifest_used"] is False
    assert manifest["final_face_usable_approval_generated"] is False
