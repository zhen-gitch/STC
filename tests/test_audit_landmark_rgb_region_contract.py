import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.diagnostics.landmark_rgb_region_contract import (
    CANDIDATE_MODES,
    POLICY_STATUS,
    run_landmark_rgb_region_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_landmark_rgb_region_contract",
    PROJECT_ROOT / "scripts" / "audit_landmark_rgb_region_contract.py",
)
audit_landmark_rgb_region_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_landmark_rgb_region_contract)


def _semantic_points():
    points = np.full((68, 2), [56.0, 56.0], dtype=np.float64)
    points[0:17, 0] = np.linspace(10.0, 102.0, 17)
    points[0:17, 1] = np.asarray(
        [56, 60, 64, 68, 78, 88, 96, 101, 104, 101, 96, 88, 78, 68, 64, 60, 56]
    )
    points[17:27, 0] = np.linspace(25.0, 87.0, 10)
    points[17:27, 1] = np.asarray([30, 28, 27, 28, 30, 30, 28, 27, 28, 30])
    points[36:42] = np.asarray([[27, 42], [32, 39], [38, 39], [42, 42], [37, 45], [31, 45]])
    points[42:48] = np.asarray([[70, 42], [75, 39], [81, 39], [85, 42], [80, 45], [74, 45]])
    points[27:30] = np.asarray([[56, 43], [56, 49], [56, 55]])
    points[30:36] = np.asarray([[56, 61], [45, 64], [50, 67], [56, 68], [62, 67], [67, 64]])
    points[48:60] = np.asarray(
        [[39, 78], [44, 74], [50, 72], [56, 71], [62, 72], [68, 74],
         [73, 78], [68, 82], [62, 84], [56, 85], [50, 84], [44, 82]]
    )
    points[60:68] = np.asarray(
        [[46, 78], [51, 75], [56, 75], [61, 76], [66, 78], [61, 81], [56, 82], [51, 81]]
    )
    return points


def _write_landmark_csv(path, points, frame_count=3):
    fields = ["frame", "timestamp", "confidence", "success"]
    fields += [f"x_{index}" for index in range(68)]
    fields += [f"y_{index}" for index in range(68)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame_id in range(1, frame_count + 1):
            row = {
                "frame": frame_id,
                "timestamp": (frame_id - 1) / 30,
                "confidence": 0.95,
                "success": 1,
            }
            for index in range(68):
                row[f"x_{index}"] = points[index, 0]
                row[f"y_{index}"] = points[index, 1]
            writer.writerow(row)


def _read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_parser_defaults_keep_policy_unfrozen():
    args = audit_landmark_rgb_region_contract.build_parser().parse_args(
        [
            "--dataset-split-file",
            "/split.json",
            "--image-root",
            "/images",
            "--aligned-landmark-root",
            "/landmarks",
            "--output-dir",
            "/output",
        ]
    )

    assert args.confidence_threshold == 0.8
    assert args.canonical_sample_step == 30
    assert args.stabilization_window == 5
    assert args.candidate_margin_ratio == 0.0
    assert args.candidate_mode is None
    assert args.overlay_frames_per_mode == 2


def test_region_contract_run_writes_hashed_candidate_artifacts(tmp_path):
    video_id = "203_1_Freeform_video_aligned"
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"train": [video_id], "val": [], "test": []}),
        encoding="utf-8",
    )
    image_root = tmp_path / "images"
    video_dir = image_root / video_id
    video_dir.mkdir(parents=True)
    for frame_id in range(3):
        Image.new("RGB", (112, 112), color=(120, 110, 100)).save(
            video_dir / f"frame_det_00_{frame_id:06d}.jpg"
        )
    landmark_root = tmp_path / "landmarks"
    landmark_root.mkdir()
    _write_landmark_csv(landmark_root / f"{video_id}.csv", _semantic_points())

    generated = run_landmark_rgb_region_contract(
        dataset_split_file=split_path,
        image_root=image_root,
        aligned_landmark_root=landmark_root,
        output_dir=tmp_path / "output",
        confidence_threshold=0.8,
        canonical_sample_step=1,
        stabilization_window=3,
        overlay_frames_per_mode=1,
        project_root=PROJECT_ROOT,
    )

    assert len(generated) == 7
    assert all(path.exists() for path in generated)
    source_rows = _read_csv(tmp_path / "output" / "tables" / "region_source_manifest.csv")
    assert source_rows[0]["status"] == "PASS"
    assert source_rows[0]["frame_join_rate"] == "1.000000"
    assert len(source_rows[0]["image_tree_sha256"]) == 64
    assert len(source_rows[0]["landmark_csv_sha256"]) == 64

    candidate_rows = _read_csv(tmp_path / "output" / "tables" / "region_frame_candidates.csv")
    assert len(candidate_rows) == 3 * len(CANDIDATE_MODES) * 3
    assert {row["candidate_mode"] for row in candidate_rows} == set(CANDIDATE_MODES)
    assert {row["policy_status"] for row in candidate_rows} == {POLICY_STATUS}
    assert all(row["valid"] == "1" for row in candidate_rows)
    assert not any("AU" in field for field in candidate_rows[0])

    manifest = json.loads((tmp_path / "output" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy_status"] == POLICY_STATUS
    assert manifest["labels_read"] is False
    assert manifest["au_values_read"] is False
    assert manifest["training_authorized"] is False
    assert manifest["canonical_train_sample_count"] == 3
    assert set(manifest["outputs"]) == {
        "region_source_manifest.csv",
        "region_frame_candidates.csv",
        "region_video_summary.csv",
        "region_issues.csv",
        "region_overlay_review.csv",
        "region_contract_report.md",
    }


def test_static_canonical_is_not_fit_from_validation_only(tmp_path):
    video_id = "203_1_Freeform_video_aligned"
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"train": [], "val": [video_id], "test": []}),
        encoding="utf-8",
    )
    image_root = tmp_path / "images"
    video_dir = image_root / video_id
    video_dir.mkdir(parents=True)
    Image.new("RGB", (112, 112)).save(video_dir / "frame_000000.jpg")
    landmark_root = tmp_path / "landmarks"
    landmark_root.mkdir()
    _write_landmark_csv(
        landmark_root / f"{video_id}.csv", _semantic_points(), frame_count=1
    )

    run_landmark_rgb_region_contract(
        dataset_split_file=split_path,
        image_root=image_root,
        aligned_landmark_root=landmark_root,
        output_dir=tmp_path / "output",
        candidate_modes=["static_canonical"],
        overlay_frames_per_mode=0,
        project_root=PROJECT_ROOT,
    )

    candidate_rows = _read_csv(tmp_path / "output" / "tables" / "region_frame_candidates.csv")
    assert all(row["valid"] == "0" for row in candidate_rows)
    assert all("train_canonical_unavailable" in row["failure_reasons"] for row in candidate_rows)
    manifest = json.loads((tmp_path / "output" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["canonical_train_sample_count"] == 0


def test_non_exact_source_contract_invalidates_all_candidate_rows(tmp_path):
    video_id = "203_1_Freeform_video_aligned"
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps({"train": [video_id], "val": [], "test": []}),
        encoding="utf-8",
    )
    image_root = tmp_path / "images"
    video_dir = image_root / video_id
    video_dir.mkdir(parents=True)
    Image.new("RGB", (112, 112)).save(video_dir / "frame_000000.jpg")
    landmark_root = tmp_path / "landmarks"
    landmark_root.mkdir()
    _write_landmark_csv(
        landmark_root / f"{video_id}.csv", _semantic_points(), frame_count=2
    )

    run_landmark_rgb_region_contract(
        dataset_split_file=split_path,
        image_root=image_root,
        aligned_landmark_root=landmark_root,
        output_dir=tmp_path / "output",
        candidate_modes=["raw_dynamic"],
        overlay_frames_per_mode=0,
        project_root=PROJECT_ROOT,
    )

    source_rows = _read_csv(tmp_path / "output" / "tables" / "region_source_manifest.csv")
    assert source_rows[0]["status"] == "FAIL"
    assert "image_landmark_row_count_mismatch" in source_rows[0]["issues"]
    candidate_rows = _read_csv(tmp_path / "output" / "tables" / "region_frame_candidates.csv")
    assert all(row["valid"] == "0" for row in candidate_rows)
    assert all("source_contract_not_pass" in row["failure_reasons"] for row in candidate_rows)
