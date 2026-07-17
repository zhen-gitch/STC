import csv
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from src.diagnostics.source_presence import (
    SOURCE_REVIEW_FIELDS,
    build_pure_black_runs,
    run_source_presence_audit,
    run_source_presence_review_report,
    validate_source_review_segments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "audit_source_presence",
    PROJECT_ROOT / "scripts" / "audit_source_presence.py",
)
audit_source_presence = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(audit_source_presence)


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_video(path, frame_count=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48)
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((48, 64, 3), 30 + index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_cli_is_read_only_and_has_no_facelandmark_action():
    parser = audit_source_presence.build_parser()
    args = parser.parse_args(
        [
            "--dataset-root",
            "/dataset",
            "--frame-failure-manifest",
            "/failure.csv",
            "--video-failure-summary",
            "/summary.csv",
            "--output-dir",
            "/audit",
        ]
    )
    assert args.max_run_samples == 7
    assert all("facelandmark" not in key.lower() for key in vars(args))


def test_build_runs_and_validate_review_coverage(tmp_path):
    failures = [
        {"split": "val", "video_id": "205_1_Freeform_video_aligned", "frame_id": str(frame), "failure_type": "pure_black"}
        for frame in (2, 3, 5)
    ]
    runs = build_pure_black_runs(failures)
    assert [(row["start_frame"], row["end_frame"]) for row in runs] == [(2, 3), (5, 5)]

    review_path = tmp_path / "review.csv"
    rows = [
        {
            "split": "val",
            "video_id": "205_1_Freeform_video_aligned",
            "pure_black_run_id": "205_1_Freeform_video_aligned:PB0001",
            "segment_start_frame": 2,
            "segment_end_frame": 3,
            "review_status": "REVIEWED",
            "source_presence_status": "person_present_detection_failure",
            "recovery_permission": "raw_frame_warp_only",
            "reviewer": "tester",
            "review_notes": "synthetic",
        },
        {
            "split": "val",
            "video_id": "205_1_Freeform_video_aligned",
            "pure_black_run_id": "205_1_Freeform_video_aligned:PB0002",
            "segment_start_frame": 5,
            "segment_end_frame": 5,
            "review_status": "REVIEWED",
            "source_presence_status": "person_absent",
            "recovery_permission": "keep_invalid",
            "reviewer": "tester",
            "review_notes": "synthetic",
        },
    ]
    _write_csv(review_path, rows, SOURCE_REVIEW_FIELDS)
    resolved = validate_source_review_segments(review_path, failures)
    assert resolved[("205_1_Freeform_video_aligned", 2)]["recovery_permission"] == "raw_frame_warp_only"
    assert resolved[("205_1_Freeform_video_aligned", 5)]["source_presence_status"] == "person_absent"


def test_source_presence_audit_maps_val_to_dev_and_writes_contract(tmp_path):
    video_id = "205_1_Freeform_video_aligned"
    raw_video_id = "205_1_Freeform_video"
    dataset_root = tmp_path / "AVEC2014"
    _write_video(dataset_root / "dev" / "Freeform" / f"{raw_video_id}.mp4")
    raw_openface_root = dataset_root / "openface_features"
    _write_csv(
        raw_openface_root / f"{raw_video_id}.csv",
        [
            {"frame": frame, "confidence": 0.98 if frame != 2 else 0.1, "success": int(frame != 2)}
            for frame in range(1, 6)
        ],
        ["frame", "confidence", "success"],
    )
    failure_path = tmp_path / "frame_failure_manifest.csv"
    _write_csv(
        failure_path,
        [
            {"split": "val", "video_id": video_id, "frame_id": 2, "failure_type": "pure_black"},
            {"split": "val", "video_id": video_id, "frame_id": 3, "failure_type": "visible_detection_failed"},
        ],
        ["split", "video_id", "frame_id", "failure_type"],
    )
    summary_path = tmp_path / "video_failure_summary.csv"
    _write_csv(
        summary_path,
        [{"split": "val", "video_id": video_id, "frame_count": 5}],
        ["split", "video_id", "frame_count"],
    )
    output_dir = tmp_path / "audit"

    generated = run_source_presence_audit(
        dataset_root=dataset_root,
        frame_failure_manifest=failure_path,
        video_failure_summary=summary_path,
        output_dir=output_dir,
        raw_openface_root=raw_openface_root,
        project_root=PROJECT_ROOT,
    )

    assert len(generated) == 5
    with (output_dir / "tables" / "source_video_contract.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        contract = next(csv.DictReader(handle))
    assert contract["source_split"] == "dev"
    assert contract["frame_count_match"] == "1"
    assert contract["contract_status"] == "PASS"
    with (output_dir / "tables" / "pure_black_source_runs.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        run = next(csv.DictReader(handle))
    assert run["start_frame"] == "2"
    assert Path(run["contact_sheet"]).exists()
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["face_landmark_rerun_performed"] is False
    assert manifest["source_tree_modified"] is False


def test_completed_review_report_expands_frame_gate(tmp_path):
    video_id = "205_1_Freeform_video_aligned"
    failure_path = tmp_path / "failures.csv"
    failures = [
        {"split": "val", "video_id": video_id, "frame_id": frame, "failure_type": "pure_black"}
        for frame in (2, 3, 5)
    ]
    _write_csv(failure_path, failures, ["split", "video_id", "frame_id", "failure_type"])
    source_run_path = tmp_path / "runs.csv"
    _write_csv(
        source_run_path,
        [
            {"video_id": video_id, "pure_black_run_id": f"{video_id}:PB0001", "start_frame": 2, "end_frame": 3},
            {"video_id": video_id, "pure_black_run_id": f"{video_id}:PB0002", "start_frame": 5, "end_frame": 5},
        ],
        ["video_id", "pure_black_run_id", "start_frame", "end_frame"],
    )
    review_path = tmp_path / "review.csv"
    _write_csv(
        review_path,
        [
            {
                "split": "val",
                "video_id": video_id,
                "pure_black_run_id": f"{video_id}:PB0001",
                "segment_start_frame": 2,
                "segment_end_frame": 3,
                "review_status": "REVIEWED",
                "source_presence_status": "person_present_detection_failure",
                "recovery_permission": "raw_frame_warp_only",
                "reviewer": "tester",
                "review_notes": "present",
            },
            {
                "split": "val",
                "video_id": video_id,
                "pure_black_run_id": f"{video_id}:PB0002",
                "segment_start_frame": 5,
                "segment_end_frame": 5,
                "review_status": "REVIEWED",
                "source_presence_status": "person_absent",
                "recovery_permission": "keep_invalid",
                "reviewer": "tester",
                "review_notes": "absent",
            },
        ],
        SOURCE_REVIEW_FIELDS,
    )

    generated = run_source_presence_review_report(
        frame_failure_manifest=failure_path,
        source_run_manifest=source_run_path,
        review_manifest=review_path,
        output_dir=tmp_path / "report",
        project_root=PROJECT_ROOT,
    )

    assert len(generated) == 4
    with (tmp_path / "report" / "tables" / "source_presence_frame_gate.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        gate = list(csv.DictReader(handle))
    assert len(gate) == 3
    assert sum(row["source_presence_status"] == "person_absent" for row in gate) == 1
    manifest = json.loads((tmp_path / "report" / "review_manifest.json").read_text())
    assert manifest["source_presence_status_counts"] == {
        "person_absent": 1,
        "person_present_detection_failure": 2,
    }
