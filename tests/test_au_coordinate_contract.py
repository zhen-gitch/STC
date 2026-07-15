import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.diagnostics.au_coordinate_contract import (
    apply_affine,
    estimate_similarity_transform,
    run_coordinate_contract_audit,
    write_run_manifest,
)


VIDEO_ID = "203_1_Freeform_video"


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_landmark_csv(path, point_rows, success_values=None):
    fieldnames = [
        "frame",
        "timestamp",
        "confidence",
        "success",
        "pose_Rx",
        "pose_Ry",
        "pose_Rz",
        "x_0",
        "x_1",
        "x_2",
        "x_3",
        "y_0",
        "y_1",
        "y_2",
        "y_3",
    ]
    rows = []
    success_values = success_values or [1] * len(point_rows)
    for frame_id, (points, success) in enumerate(zip(point_rows, success_values), start=1):
        row = {
            "frame": frame_id,
            "timestamp": (frame_id - 1) / 30.0,
            "confidence": 0.95 - 0.05 * (frame_id - 1),
            "success": success,
            "pose_Rx": 0.01 * frame_id,
            "pose_Ry": 0.02 * frame_id,
            "pose_Rz": 0.0,
        }
        for landmark_id, (x_value, y_value) in enumerate(points):
            row[f"x_{landmark_id}"] = x_value
            row[f"y_{landmark_id}"] = y_value
        rows.append(row)
    return _write_csv(path, fieldnames, rows)


def _write_contract_inputs(root, image_root):
    video_dir = image_root / f"{VIDEO_ID}_aligned"
    video_dir.mkdir(parents=True)
    image_paths = []
    for frame_id in (1, 2):
        image_path = video_dir / f"frame_det_00_{frame_id:06d}.jpg"
        Image.new("RGB", (112, 112), color=(80 + frame_id, 90, 100)).save(image_path)
        image_paths.append(image_path)

    summary_path = _write_csv(
        root / "frame_contract_summary.csv",
        [
            "video_id",
            "image_width",
            "image_height",
            "selected_frame_count",
            "status",
        ],
        [
            {
                "video_id": VIDEO_ID,
                "image_width": 112,
                "image_height": 112,
                "selected_frame_count": 2,
                "status": "PASS",
            }
        ],
    )
    mapping_path = _write_csv(
        root / "selected_frame_mapping.csv",
        [
            "video_id",
            "selected_position",
            "image_index",
            "image_path",
            "image_frame_id",
            "expected_csv_frame",
        ],
        [
            {
                "video_id": VIDEO_ID,
                "selected_position": position,
                "image_index": position,
                "image_path": f"/server/path/{image_path.name}",
                "image_frame_id": position + 1,
                "expected_csv_frame": position + 1,
            }
            for position, image_path in enumerate(image_paths)
        ],
    )
    split_path = root / "dataset_split.json"
    split_path.write_text(
        json.dumps({"train": [f"{VIDEO_ID}_aligned"], "val": [], "test": []}),
        encoding="utf-8",
    )
    return summary_path, mapping_path, split_path


def _read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_similarity_transform_recovers_scale_rotation_and_translation():
    source = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [2.0, 3.0]])
    expected = np.asarray([[0.0, -1.5, 20.0], [1.5, 0.0, 30.0]])
    target = np.stack(
        [expected[:, :2] @ point + expected[:, 2] for point in source],
        axis=0,
    )

    actual = estimate_similarity_transform(source, target)

    assert np.allclose(actual, expected)
    mapped = apply_affine({idx: point for idx, point in enumerate(source)}, actual)
    assert np.allclose(np.stack([mapped[idx] for idx in range(4)]), target)


def test_coordinate_audit_prefers_explicit_affine_and_matches_reference(tmp_path):
    image_root = tmp_path / "images"
    openface_root = tmp_path / "openface"
    aligned_root = tmp_path / "aligned_openface"
    transform_root = tmp_path / "transforms"
    summary_path, mapping_path, split_path = _write_contract_inputs(tmp_path, image_root)
    raw_points = [
        [(20, 30), (60, 30), (20, 70), (60, 70)],
        [(22, 30), (62, 30), (22, 70), (62, 70)],
    ]
    matrix = np.asarray([[0.8, 0.0, 10.0], [0.0, 0.8, 5.0]])
    aligned_points = [
        [(matrix[:, :2] @ np.asarray(point) + matrix[:, 2]).tolist() for point in frame]
        for frame in raw_points
    ]
    _write_landmark_csv(openface_root / f"{VIDEO_ID}.csv", raw_points)
    _write_landmark_csv(aligned_root / f"{VIDEO_ID}_aligned.csv", aligned_points)
    _write_csv(
        transform_root / f"{VIDEO_ID}.csv",
        ["frame", "m00", "m01", "m02", "m10", "m11", "m12"],
        [
            {
                "frame": frame_id,
                "m00": 0.8,
                "m01": 0.0,
                "m02": 10.0,
                "m10": 0.0,
                "m11": 0.8,
                "m12": 5.0,
            }
            for frame_id in (1, 2)
        ],
    )

    generated = run_coordinate_contract_audit(
        image_root=image_root,
        openface_root=openface_root,
        frame_contract_summary=summary_path,
        selected_frame_mapping=mapping_path,
        dataset_split_file=split_path,
        output_dir=tmp_path / "audit",
        mapping_method="auto",
        transform_root=transform_root,
        aligned_openface_root=aligned_root,
        max_overlays=2,
    )

    assert len(generated) == 6
    manifest = _read_rows(tmp_path / "audit/tables/coordinate_mapping_manifest.csv")[0]
    assert manifest["mapping_method"] == "explicit_affine"
    assert manifest["status"] == "REVIEW_REQUIRED"
    assert float(manifest["mapping_valid_frame_ratio"]) == 1.0
    assert float(manifest["median_mapping_residual"]) < 1e-8
    transforms = _read_rows(tmp_path / "audit/tables/coordinate_transforms.csv")
    assert len(transforms) == 2
    assert all(float(row["m00"]) == 0.8 for row in transforms)
    overlays = _read_rows(tmp_path / "audit/tables/overlay_manifest.csv")
    assert overlays
    assert all(row["split"] == "train" for row in overlays)
    assert all(Path(row["overlay_path"]).exists() for row in overlays)


def test_coordinate_audit_uses_aligned_redetection_without_raw_scaling(tmp_path):
    image_root = tmp_path / "images"
    openface_root = tmp_path / "openface"
    aligned_root = tmp_path / "aligned_openface"
    summary_path, mapping_path, split_path = _write_contract_inputs(tmp_path, image_root)
    detection_points = [
        [(300, 200), (340, 200), (300, 240), (340, 240)],
        [(302, 200), (342, 200), (302, 240), (342, 240)],
    ]
    aligned_points = [
        [(30, 35), (70, 35), (30, 75), (70, 75)],
        [(31, 35), (71, 35), (31, 75), (71, 75)],
    ]
    _write_landmark_csv(openface_root / f"{VIDEO_ID}.csv", detection_points)
    _write_landmark_csv(aligned_root / f"{VIDEO_ID}_aligned.csv", aligned_points)

    run_coordinate_contract_audit(
        image_root=image_root,
        openface_root=openface_root,
        frame_contract_summary=summary_path,
        selected_frame_mapping=mapping_path,
        dataset_split_file=split_path,
        output_dir=tmp_path / "audit",
        mapping_method="auto",
        aligned_openface_root=aligned_root,
        max_overlays=1,
    )

    manifest = _read_rows(tmp_path / "audit/tables/coordinate_mapping_manifest.csv")[0]
    assert manifest["mapping_method"] == "aligned_redetection"
    assert manifest["source_coordinate_system"] == "aligned_image_space"
    assert manifest["status"] == "REVIEW_REQUIRED"
    transforms = _read_rows(tmp_path / "audit/tables/coordinate_transforms.csv")
    assert all(float(row["m00"]) == 1.0 for row in transforms)
    assert all(float(row["m02"]) == 0.0 for row in transforms)


def test_coordinate_audit_rejects_failed_aligned_redetection_rows(tmp_path):
    image_root = tmp_path / "images"
    openface_root = tmp_path / "openface"
    aligned_root = tmp_path / "aligned_openface"
    summary_path, mapping_path, split_path = _write_contract_inputs(tmp_path, image_root)
    detection_points = [
        [(300, 200), (340, 200), (300, 240), (340, 240)],
        [(302, 200), (342, 200), (302, 240), (342, 240)],
    ]
    aligned_points = [
        [(30, 35), (70, 35), (30, 75), (70, 75)],
        [(31, 35), (71, 35), (31, 75), (71, 75)],
    ]
    _write_landmark_csv(openface_root / f"{VIDEO_ID}.csv", detection_points)
    _write_landmark_csv(
        aligned_root / f"{VIDEO_ID}_aligned.csv",
        aligned_points,
        success_values=[1, 0],
    )

    run_coordinate_contract_audit(
        image_root=image_root,
        openface_root=openface_root,
        frame_contract_summary=summary_path,
        selected_frame_mapping=mapping_path,
        dataset_split_file=split_path,
        output_dir=tmp_path / "audit",
        mapping_method="aligned_redetection",
        aligned_openface_root=aligned_root,
        max_overlays=2,
    )

    manifest = _read_rows(tmp_path / "audit/tables/coordinate_mapping_manifest.csv")[0]
    assert manifest["status"] == "FAIL"
    assert float(manifest["mapping_valid_frame_ratio"]) == 0.5
    frame_rows = _read_rows(tmp_path / "audit/tables/coordinate_frame_summary.csv")
    assert frame_rows[1]["mapping_source_success"] == "0"
    assert frame_rows[1]["mapping_valid"] == "0"
    assert "mapping_source_detection_failed" in frame_rows[1]["issues"]


def test_coordinate_audit_fits_caller_supplied_canonical_template(tmp_path):
    image_root = tmp_path / "images"
    openface_root = tmp_path / "openface"
    summary_path, mapping_path, split_path = _write_contract_inputs(tmp_path, image_root)
    detection_points = [
        [(200, 100), (300, 100), (200, 200), (300, 200)],
        [(202, 100), (302, 100), (202, 200), (302, 200)],
    ]
    _write_landmark_csv(openface_root / f"{VIDEO_ID}.csv", detection_points)
    template_path = _write_csv(
        tmp_path / "canonical_template.csv",
        ["landmark_id", "x", "y"],
        [
            {"landmark_id": 0, "x": 30, "y": 30},
            {"landmark_id": 1, "x": 80, "y": 30},
            {"landmark_id": 2, "x": 30, "y": 80},
            {"landmark_id": 3, "x": 80, "y": 80},
        ],
    )

    run_coordinate_contract_audit(
        image_root=image_root,
        openface_root=openface_root,
        frame_contract_summary=summary_path,
        selected_frame_mapping=mapping_path,
        dataset_split_file=split_path,
        output_dir=tmp_path / "audit",
        mapping_method="canonical_similarity",
        canonical_template_path=template_path,
        max_overlays=1,
    )

    manifest = _read_rows(tmp_path / "audit/tables/coordinate_mapping_manifest.csv")[0]
    assert manifest["mapping_method"] == "canonical_similarity"
    assert manifest["source_coordinate_system"] == "detection_space"
    assert manifest["status"] == "REVIEW_REQUIRED"
    assert float(manifest["mapping_valid_frame_ratio"]) == 1.0


def test_coordinate_audit_blocks_when_no_legal_mapping_source_exists(tmp_path):
    image_root = tmp_path / "images"
    openface_root = tmp_path / "openface"
    summary_path, mapping_path, split_path = _write_contract_inputs(tmp_path, image_root)
    _write_landmark_csv(
        openface_root / f"{VIDEO_ID}.csv",
        [
            [(300, 200), (340, 200), (300, 240), (340, 240)],
            [(302, 200), (342, 200), (302, 240), (342, 240)],
        ],
    )

    run_coordinate_contract_audit(
        image_root=image_root,
        openface_root=openface_root,
        frame_contract_summary=summary_path,
        selected_frame_mapping=mapping_path,
        dataset_split_file=split_path,
        output_dir=tmp_path / "audit",
        mapping_method="auto",
        max_overlays=1,
    )

    manifest = _read_rows(tmp_path / "audit/tables/coordinate_mapping_manifest.csv")[0]
    assert manifest["mapping_method"] == ""
    assert manifest["status"] == "BLOCKED"
    assert "mapping_source_unavailable" in manifest["issues"]
    assert _read_rows(tmp_path / "audit/tables/overlay_manifest.csv") == []


def test_run_manifest_freezes_git_command_and_input_hashes(tmp_path):
    input_path = tmp_path / "input.csv"
    input_path.write_text("frame\n1\n", encoding="utf-8")

    manifest_path = write_run_manifest(
        tmp_path / "run_manifest.json",
        Path(__file__).resolve().parents[1],
        {"mapping_method": "aligned_redetection", "max_videos": 1},
        {"input": input_path, "optional": None},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["audit"] == "AU-T0b coordinate contract"
    assert payload["git_commit"]
    assert payload["git_branch"]
    assert payload["inputs"]["input"]["sha256"]
    assert payload["inputs"]["optional"]["sha256"] == ""


def test_coordinate_contract_cli_smoke_writes_provenance(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    image_root = tmp_path / "images"
    openface_root = tmp_path / "openface"
    aligned_root = tmp_path / "aligned_openface"
    summary_path, mapping_path, split_path = _write_contract_inputs(tmp_path, image_root)
    detection_points = [
        [(300, 200), (340, 200), (300, 240), (340, 240)],
        [(302, 200), (342, 200), (302, 240), (342, 240)],
    ]
    aligned_points = [
        [(30, 35), (70, 35), (30, 75), (70, 75)],
        [(31, 35), (71, 35), (31, 75), (71, 75)],
    ]
    _write_landmark_csv(openface_root / f"{VIDEO_ID}.csv", detection_points)
    _write_landmark_csv(aligned_root / f"{VIDEO_ID}_aligned.csv", aligned_points)
    output_dir = tmp_path / "cli_audit"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_au_coordinate_contract.py",
            "--image-root",
            str(image_root),
            "--openface-root",
            str(openface_root),
            "--frame-contract-summary",
            str(summary_path),
            "--selected-frame-mapping",
            str(mapping_path),
            "--dataset-split-file",
            str(split_path),
            "--aligned-openface-root",
            str(aligned_root),
            "--mapping-method",
            "aligned_redetection",
            "--output-dir",
            str(output_dir),
            "--max-overlays",
            "1",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "[AU-T0b] generated files:" in result.stdout
    payload = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert payload["arguments"]["mapping_method"] == "aligned_redetection"
    assert payload["inputs"]["frame_contract_summary"]["sha256"]
