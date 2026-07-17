import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.diagnostics.face_usability import normalize_landmarks
from src.diagnostics.face_usability_temporal_review import run_landmark_jump_pair_review


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _points():
    angles = np.linspace(0.0, 2.0 * np.pi, 68, endpoint=False)
    points = np.stack([56.0 + 30.0 * np.cos(angles), 56.0 + 38.0 * np.sin(angles)], axis=1)
    points[36:42] = np.array([[38, 48], [40, 46], [44, 46], [46, 48], [44, 50], [40, 50]])
    points[42:48] = np.array([[66, 48], [68, 46], [72, 46], [74, 48], [72, 50], [68, 50]])
    return points


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_pair_review_recomputes_jump_and_writes_pending_outputs(tmp_path):
    video_id = "203_1_Freeform_video_aligned"
    image_dir = tmp_path / "images" / video_id
    image_dir.mkdir(parents=True)
    base = _points()
    moved = base.copy()
    moved[17:68, 0] += np.linspace(0.0, 2.0, 51)
    for frame, value in [(1, 100), (2, 120)]:
        Image.fromarray(np.full((112, 112, 3), value, dtype=np.uint8)).save(
            image_dir / f"frame_{frame:06d}.jpg"
        )

    landmark_root = tmp_path / "landmarks"
    landmark_root.mkdir()
    fields = ["frame", "timestamp", "confidence", "success"] + [
        f"{axis}_{index}" for axis in ("x", "y") for index in range(68)
    ]
    with (landmark_root / f"{video_id}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame, points in [(1, base), (2, moved)]:
            row = {"frame": frame, "timestamp": frame / 30.0, "confidence": 0.98, "success": 1}
            for index in range(68):
                row[f"x_{index}"] = points[index, 0]
                row[f"y_{index}"] = points[index, 1]
            writer.writerow(row)

    jump = float(
        np.median(
            np.linalg.norm(
                normalize_landmarks(moved)[17:68] - normalize_landmarks(base)[17:68], axis=1
            )
        )
    )
    source = tmp_path / "source"
    contact_path = source / "tables" / "face_usability_contact_review.csv"
    contact_path.parent.mkdir(parents=True)
    with contact_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "selection_reason",
                "rank",
                "split",
                "video_id",
                "frame_id",
                "image_path",
                "metric_name",
                "metric_value",
                "review_status",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "selection_reason": "landmark_jump_high",
                "rank": 1,
                "split": "train",
                "video_id": video_id,
                "frame_id": 2,
                "image_path": str(image_dir / "frame_000002.jpg"),
                "metric_name": "landmark_jump",
                "metric_value": f"{jump:.6f}",
                "review_status": "PENDING",
            }
        )
    (source / "run_manifest.json").write_text(
        json.dumps(
            {
                "output_status": "DISTRIBUTION_REVIEW_REQUIRED",
                "threshold_manifest_used": False,
                "final_face_usable_approval_generated": False,
                "outputs": {
                    "contact_review": {
                        "path": str(contact_path.resolve()),
                        "sha256": _sha256(contact_path),
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    generated = run_landmark_jump_pair_review(
        source_run_dir=source,
        aligned_landmark_root=landmark_root,
        output_dir=tmp_path / "output",
        project_root=PROJECT_ROOT,
    )

    assert len(generated) == 4
    rows = list(
        csv.DictReader(
            (tmp_path / "output/tables/landmark_jump_pair_review.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    assert rows[0]["previous_frame_id"] == "1"
    assert rows[0]["current_frame_id"] == "2"
    assert rows[0]["review_status"] == "PENDING"
    assert float(rows[0]["absolute_difference"]) <= 1e-5
    manifest = json.loads((tmp_path / "output/run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["threshold_manifest_generated"] is False
    assert manifest["face_usable_approval_generated"] is False
