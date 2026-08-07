import numpy as np
import pytest

from src.diagnostics.landmark_rgb_region_contract import (
    REGION_ORDER,
    _intersection_area,
    build_raw_region_geometry,
    fit_train_canonical,
    source_validity,
    stabilize_region_sequence,
    static_region_geometry,
)


def _semantic_points():
    points = np.full((68, 2), [56.0, 56.0], dtype=np.float64)

    # Jaw contour. Cheek proxies are 2-4 and 12-14; lower-face anchors are 5-11.
    jaw_x = np.linspace(10.0, 102.0, 17)
    jaw_y = np.asarray(
        [56, 60, 64, 68, 78, 88, 96, 101, 104, 101, 96, 88, 78, 68, 64, 60, 56],
        dtype=np.float64,
    )
    points[0:17, 0] = jaw_x
    points[0:17, 1] = jaw_y

    # Eyebrows and eyes.
    points[17:27, 0] = np.linspace(25.0, 87.0, 10)
    points[17:27, 1] = np.asarray([30, 28, 27, 28, 30, 30, 28, 27, 28, 30])
    points[36:42, 0] = np.asarray([27, 32, 38, 42, 37, 31])
    points[36:42, 1] = np.asarray([42, 39, 39, 42, 45, 45])
    points[42:48, 0] = np.asarray([70, 75, 81, 85, 80, 74])
    points[42:48, 1] = np.asarray([42, 39, 39, 42, 45, 45])

    # Nose bridge is diagnostic; lower nose is the strict nose anchor.
    points[27:30, 0] = 56.0
    points[27:30, 1] = np.asarray([43, 49, 55])
    points[30:36, 0] = np.asarray([56, 45, 50, 56, 62, 67])
    points[30:36, 1] = np.asarray([61, 64, 67, 68, 67, 64])

    # Outer and inner lips.
    points[48:60, 0] = np.asarray([39, 44, 50, 56, 62, 68, 73, 68, 62, 56, 50, 44])
    points[48:60, 1] = np.asarray([78, 74, 72, 71, 72, 74, 78, 82, 84, 85, 84, 82])
    points[60:68, 0] = np.asarray([46, 51, 56, 61, 66, 61, 56, 51])
    points[60:68, 1] = np.asarray([78, 75, 75, 76, 78, 81, 82, 81])
    return points


def _row(points, success=1, confidence=0.95):
    return {
        "success": success,
        "confidence": confidence,
        "landmarks": {index: value for index, value in enumerate(points)},
    }


def test_raw_regions_are_non_overlapping_and_contain_required_landmarks():
    geometry = build_raw_region_geometry(_semantic_points(), 112, 112)

    assert all(geometry[region]["valid"] for region in REGION_ORDER)
    boxes = [geometry[region]["box"] for region in REGION_ORDER]
    assert _intersection_area(boxes[0], boxes[1]) == 0
    assert _intersection_area(boxes[0], boxes[2]) == 0
    assert _intersection_area(boxes[1], boxes[2]) == 0
    assert all(
        geometry[region]["required_containment_ratio"] == pytest.approx(1.0)
        for region in REGION_ORDER
    )
    assert geometry["nose_cheek"]["nose_bridge_containment_ratio"] < 1.0


def test_semantic_order_conflict_fails_closed_without_clipping():
    points = _semantic_points()
    points[30:36, 1] = 43.0

    geometry = build_raw_region_geometry(points, 112, 112)

    assert not any(geometry[region]["valid"] for region in REGION_ORDER)
    assert all(
        "eye_nose_semantic_order_conflict" in geometry[region]["failure_reasons"]
        for region in REGION_ORDER
    )


def test_static_geometry_uses_fixed_train_template_and_checks_current_frame():
    train_points = _semantic_points()
    canonical = fit_train_canonical([train_points / 112.0, train_points / 112.0])
    accepted = static_region_geometry(train_points, canonical, 112, 112)
    assert all(accepted[region]["valid"] for region in REGION_ORDER)

    shifted = train_points.copy()
    shifted[:, 0] += 2.0
    accepted_with_margin = static_region_geometry(
        shifted, canonical, 112, 112, margin_ratio=0.05
    )
    assert all(accepted_with_margin[region]["valid"] for region in REGION_ORDER)

    shifted = train_points.copy()
    shifted[:, 0] += 50.0
    rejected = static_region_geometry(shifted, canonical, 112, 112)
    assert not any(rejected[region]["valid"] for region in REGION_ORDER)
    assert any(
        "semantic_containment_conflict" in rejected[region]["failure_reasons"]
        for region in REGION_ORDER
    )


def test_stabilized_geometry_reduces_one_frame_box_jump():
    frames = []
    for frame_id, shift in enumerate((0.0, 0.0, 6.0, 0.0, 0.0), start=1):
        points = _semantic_points()
        points[:, 0] += shift
        frames.append(
            {
                "frame_id": frame_id,
                "points": points,
                "width": 112,
                "height": 112,
                "raw": build_raw_region_geometry(points, 112, 112),
            }
        )

    stabilized = stabilize_region_sequence(frames, window_size=5)

    raw_x0 = frames[2]["raw"]["eye_brow"]["box"][0]
    stabilized_x0 = stabilized[2]["eye_brow"]["box"][0]
    baseline_x0 = frames[0]["raw"]["eye_brow"]["box"][0]
    assert raw_x0 != baseline_x0
    assert stabilized_x0 == baseline_x0
    assert not stabilized[2]["eye_brow"]["valid"]
    assert "semantic_containment_conflict" in stabilized[2]["eye_brow"]["failure_reasons"]


def test_source_validity_requires_success_confidence_and_all_landmarks():
    points = _semantic_points()
    assert source_validity(_row(points), 0.8) == (True, [])

    valid, reasons = source_validity(_row(points, confidence=0.2), 0.8)
    assert not valid
    assert reasons == ["confidence_below_threshold"]

    incomplete = _row(points)
    incomplete["landmarks"].pop(67)
    valid, reasons = source_validity(incomplete, 0.8)
    assert not valid
    assert "incomplete_or_non_finite_68_landmarks" in reasons
