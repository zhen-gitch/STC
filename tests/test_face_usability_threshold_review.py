import csv
import hashlib
import json
from pathlib import Path

from src.diagnostics.face_usability_threshold_review import run_threshold_review_template


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_threshold_review_deduplicates_and_keeps_dual_lane_pending(tmp_path):
    source = tmp_path / "source"
    contact = _write_csv(
        source / "tables/contact.csv",
        [
            "selection_reason",
            "rank",
            "split",
            "video_id",
            "frame_id",
            "image_path",
            "contact_sheet",
            "review_status",
        ],
        [
            {
                "selection_reason": "landmark_jump_high",
                "rank": 1,
                "split": "train",
                "video_id": "203_1_Freeform_video_aligned",
                "frame_id": 2,
                "image_path": "/images/frame_2.jpg",
                "contact_sheet": "/sheets/jump.jpg",
                "review_status": "PENDING",
            },
            {
                "selection_reason": "transform_residual_high",
                "rank": 2,
                "split": "train",
                "video_id": "203_1_Freeform_video_aligned",
                "frame_id": 2,
                "image_path": "/images/frame_2.jpg",
                "contact_sheet": "/sheets/residual.jpg",
                "review_status": "PENDING",
            },
        ],
    )
    frame_fields = [
        "split",
        "video_id",
        "frame_id",
        "image_path",
        "face_usability_status",
        "exclusion_reasons",
        *[
            "source_presence_status",
            "aligned_failure_status",
            "exposure_status",
            "landmark_success",
            "confidence",
            "landmark_in_frame_ratio",
            "face_hull_coverage",
            "face_hull_visible_ratio",
            "blur_score",
            "gradient_energy",
            "global_mean_luma",
            "low_saturation_ratio",
            "high_saturation_ratio",
            "landmark_pose_yaw_deg",
            "landmark_pose_pitch_deg",
            "landmark_pose_roll_deg",
            "landmark_pose_reprojection_rmse",
            "transform_residual",
            "landmark_jump",
        ],
    ]
    frame = _write_csv(
        source / "tables/frames.csv",
        frame_fields,
        [
            {
                **{field: "" for field in frame_fields},
                "split": "train",
                "video_id": "203_1_Freeform_video_aligned",
                "frame_id": 2,
                "image_path": "/images/frame_2.jpg",
                "face_usability_status": "pending_threshold_review",
                "landmark_success": 1,
                "landmark_jump": 0.1,
            }
        ],
    )
    (source / "run_manifest.json").write_text(
        json.dumps(
            {
                "output_status": "DISTRIBUTION_REVIEW_REQUIRED",
                "threshold_manifest_used": False,
                "final_face_usable_approval_generated": False,
                "outputs": {
                    "contact_review": {"path": str(contact.resolve()), "sha256": _sha(contact)},
                    "frame_manifest": {"path": str(frame.resolve()), "sha256": _sha(frame)},
                },
            }
        ),
        encoding="utf-8",
    )

    temporal = tmp_path / "temporal"
    pair = _write_csv(
        temporal / "tables/pairs.csv",
        ["split", "video_id", "previous_frame_id", "current_frame_id", "review_status"],
        [
            {
                "split": "train",
                "video_id": "203_1_Freeform_video_aligned",
                "previous_frame_id": 1,
                "current_frame_id": 2,
                "review_status": "PENDING",
            }
        ],
    )
    (temporal / "run_manifest.json").write_text(
        json.dumps(
            {
                "review_status": "PENDING",
                "threshold_manifest_generated": False,
                "outputs": {"pair_review": {"path": str(pair.resolve()), "sha256": _sha(pair)}},
            }
        ),
        encoding="utf-8",
    )

    generated = run_threshold_review_template(
        source_run_dir=source,
        temporal_review_dir=temporal,
        output_dir=tmp_path / "output",
        project_root=PROJECT_ROOT,
    )

    assert len(generated) == 4
    rows = list(
        csv.DictReader(
            (tmp_path / "output/tables/face_usability_threshold_review_template.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert len(rows) == 1
    assert rows[0]["selection_reasons"] == "landmark_jump_high;transform_residual_high"
    assert rows[0]["previous_frame_id_for_jump"] == "1"
    assert rows[0]["global_face_review_status"] == "PENDING"
    assert rows[0]["local_geometry_review_status"] == "PENDING"
    assert rows[0]["temporal_boundary_review_status"] == "PENDING"
    manifest = json.loads((tmp_path / "output/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dual_lane_contract"] is True
    assert manifest["local_geometry_only"] is True
    assert manifest["local_crop_approval_generated"] is False
    assert manifest["threshold_manifest_generated"] is False
