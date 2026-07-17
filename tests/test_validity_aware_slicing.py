import csv
import importlib.util
import json
from pathlib import Path

import pytest

from src.diagnostics.validity_aware_slicing import (
    POLICY_AU_READY,
    POLICY_GLOBAL_POST_WARP,
    POLICY_GLOBAL_READY,
    POLICY_LANDMARK_READY,
    balanced_partition_windows,
    contiguous_valid_runs,
    run_validity_aware_slicing_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "audit_validity_aware_slicing",
    PROJECT_ROOT / "scripts" / "audit_validity_aware_slicing.py",
)
audit_validity_aware_slicing = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(audit_validity_aware_slicing)


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_contiguous_runs_never_bridge_invalid_frames():
    assert contiguous_valid_runs(10, {3, 4, 8}) == [(1, 2), (5, 7), (9, 10)]
    assert contiguous_valid_runs(3, {1, 2, 3}) == []
    with pytest.raises(ValueError, match="outside"):
        contiguous_valid_runs(3, {4})


def test_balanced_partition_is_exact_non_overlapping_and_avoids_duplicate_tail():
    windows = balanced_partition_windows(1, 2001, 2000)
    assert windows == [(1, 1001), (1002, 2001)]
    assert max(end - start + 1 for start, end in windows) <= 2000
    covered = [frame for start, end in windows for frame in range(start, end + 1)]
    assert covered == list(range(1, 2002))


def test_full_audit_separates_global_and_landmark_validity_contracts(tmp_path):
    video_id = "203_1_Freeform_video_aligned"
    split_path = tmp_path / "dataset_split.json"
    split_path.write_text(
        json.dumps({"train": [video_id], "val": [], "test": []}),
        encoding="utf-8",
    )
    summary_path = tmp_path / "video_summary.csv"
    _write_csv(
        summary_path,
        [{"split": "train", "video_id": video_id, "frame_count": 10}],
        ["split", "video_id", "frame_count"],
    )
    failure_path = tmp_path / "failures.csv"
    _write_csv(
        failure_path,
        [
            {
                "split": "train",
                "video_id": video_id,
                "frame_id": 3,
                "failure_type": "pure_black",
                "decode_status": "OK",
            },
            {
                "split": "train",
                "video_id": video_id,
                "frame_id": 7,
                "failure_type": "visible_detection_failed",
                "decode_status": "OK",
            },
        ],
        ["split", "video_id", "frame_id", "failure_type", "decode_status"],
    )
    presence_path = tmp_path / "presence.csv"
    _write_csv(
        presence_path,
        [
            {
                "split": "train",
                "video_id": video_id,
                "frame_id": 3,
                "source_presence_status": "person_absent",
                "recovery_permission": "keep_invalid",
                "review_status": "REVIEWED",
            }
        ],
        [
            "split",
            "video_id",
            "frame_id",
            "source_presence_status",
            "recovery_permission",
            "review_status",
        ],
    )
    output_dir = tmp_path / "audit"
    generated = run_validity_aware_slicing_audit(
        dataset_split_file=split_path,
        video_failure_summary=summary_path,
        frame_failure_manifest=failure_path,
        source_presence_gate=presence_path,
        output_dir=output_dir,
        sample_step=1,
        max_seq_len=5,
        window_frames_values=(4,),
        min_clip_frames=2,
        project_root=PROJECT_ROOT,
    )

    assert len(generated) == 6
    with (output_dir / "tables" / "slicing_video_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    by_policy = {row["policy"]: row for row in rows}
    assert by_policy[POLICY_GLOBAL_READY]["invalid_frame_count"] == "1"
    assert by_policy[POLICY_GLOBAL_POST_WARP]["invalid_frame_count"] == "1"
    assert POLICY_AU_READY == POLICY_LANDMARK_READY
    assert by_policy[POLICY_LANDMARK_READY]["invalid_frame_count"] == "2"
    assert by_policy[POLICY_GLOBAL_READY]["legacy_head_invalid_count"] == "1"
    assert by_policy[POLICY_LANDMARK_READY]["valid_run_count"] == "3"

    with (output_dir / "tables" / "candidate_clip_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        clips = [row for row in csv.DictReader(handle) if row["policy"] == POLICY_LANDMARK_READY]
    assert all(not (int(row["start_frame"]) <= 3 <= int(row["end_frame"])) for row in clips)
    assert all(not (int(row["start_frame"]) <= 7 <= int(row["end_frame"])) for row in clips)

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["labels_read"] is False
    assert manifest["source_tree_modified"] is False


def test_cli_has_no_training_or_materialization_action():
    parser = audit_validity_aware_slicing.build_parser()
    args = parser.parse_args(
        [
            "--dataset-split-file",
            "/split.json",
            "--video-failure-summary",
            "/summary.csv",
            "--frame-failure-manifest",
            "/failures.csv",
            "--source-presence-gate",
            "/presence.csv",
            "--output-dir",
            "/audit",
        ]
    )
    assert args.window_frames == [600, 1200, 2000]
    assert all("train" not in key and "material" not in key for key in vars(args))
