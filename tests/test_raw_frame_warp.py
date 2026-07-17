import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.diagnostics.raw_frame_warp import (
    _combine_tracked_points,
    _track_adjacent_sequence,
    interpolate_similarity_matrix,
    select_smoke_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "audit_frame_recovery",
    PROJECT_ROOT / "scripts" / "audit_frame_recovery.py",
)
audit_frame_recovery = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(audit_frame_recovery)


def _failure(video_id, frame, previous, following, block_id="B1", split="train"):
    return {
        "split": split,
        "video_id": video_id,
        "frame_id": str(frame),
        "failure_type": "pure_black",
        "failure_block_id": block_id,
        "failure_block_length": str(following - previous - 1),
        "previous_valid_frame": str(previous),
        "next_valid_frame": str(following),
    }


def _gate(video_id, frame, run_id="PB1", presence="person_present_detection_failure"):
    return {
        "video_id": video_id,
        "frame_id": str(frame),
        "pure_black_run_id": run_id,
        "source_presence_status": presence,
        "recovery_permission": "raw_frame_warp_only" if presence.endswith("failure") else "keep_invalid",
        "review_status": "REVIEWED",
    }


def test_similarity_interpolation_preserves_similarity_structure():
    previous = np.array([[0.4, 0.1, -20.0], [-0.1, 0.4, 5.0]])
    following = np.array([[0.6, -0.2, -10.0], [0.2, 0.6, 9.0]])

    result = interpolate_similarity_matrix(previous, following, 0.25)

    assert np.isclose(result[0, 0], result[1, 1])
    assert np.isclose(result[0, 1], -result[1, 0])
    assert np.allclose(result[:, 2], [-17.5, 6.0])


def test_candidate_selection_requires_exact_reviewed_gate_and_prefers_short_runs():
    failures = [
        _failure("video_b", 10, 9, 11, block_id="B1"),
        _failure("video_a", 20, 18, 22, block_id="B2"),
        {**_failure("video_c", 30, 29, 31, block_id="B3"), "split": "val"},
    ]
    gates = [
        _gate("video_b", 10, "PB1"),
        _gate("video_a", 20, "PB2"),
        _gate("video_c", 30, "PB3"),
    ]

    selected = select_smoke_candidates(failures, gates, split="train", max_anchor_gap=8)

    assert [item["failure"]["video_id"] for item in selected] == ["video_b", "video_a"]


def test_candidate_selection_rejects_incomplete_presence_gate():
    failures = [_failure("video", 10, 9, 11), _failure("video", 20, 19, 21, block_id="B2")]
    gates = [_gate("video", 10)]

    try:
        select_smoke_candidates(failures, gates)
    except ValueError as exc:
        assert "exactly cover" in str(exc)
    else:
        raise AssertionError("incomplete source-presence gate must fail closed")


def test_adjacent_optical_flow_tracks_translation_and_combines_directions():
    import cv2

    frames = []
    for offset in (0, 2, 4):
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        cv2.circle(image, (40 + offset, 44), 8, (255, 255, 255), -1)
        cv2.line(image, (30 + offset, 44), (50 + offset, 44), (120, 120, 120), 2)
        cv2.line(image, (40 + offset, 34), (40 + offset, 54), (120, 120, 120), 2)
        frames.append(image)
    points = np.array([[40.0, 44.0], [36.0, 44.0], [44.0, 44.0]], dtype=np.float32)

    forward, forward_valid, _ = _track_adjacent_sequence(frames[:2], points, 1.0)
    backward, backward_valid, _ = _track_adjacent_sequence(frames[:0:-1], points + [4.0, 0.0], 1.0)
    combined, valid = _combine_tracked_points(forward, forward_valid, backward, backward_valid)

    assert valid.all()
    assert np.allclose(combined, points + [2.0, 0.0], atol=0.5)


def test_cli_raw_warp_smoke_defaults_to_train_and_three_frames():
    args = audit_frame_recovery.build_parser().parse_args(
        [
            "raw-warp-smoke",
            "--audit-dir",
            "/audit",
            "--dataset-root",
            "/dataset",
            "--image-root",
            "/images",
            "--source-presence-gate",
            "/gate.csv",
            "--raw-openface-root",
            "/raw-csv",
            "--aligned-openface-root",
            "/aligned-csv",
            "--output-dir",
            "/output",
        ]
    )

    assert args.split == "train"
    assert args.max_frames == 3
    assert args.max_anchor_gap == 8
    assert args.target_frame is None


def test_candidate_selection_rejects_non_train_explicit_target():
    failures = [{**_failure("video", 10, 9, 11), "split": "val"}]
    gates = [_gate("video", 10)]

    with pytest.raises(ValueError, match="not eligible"):
        select_smoke_candidates(
            failures,
            gates,
            split="train",
            requested_targets=["video:10"],
        )
