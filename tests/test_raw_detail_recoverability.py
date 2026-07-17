import importlib.util
from pathlib import Path

import numpy as np

from src.diagnostics.raw_detail_recoverability import (
    _classify_candidate_frame,
    _estimate_raw_to_aligned,
    _face_mask,
    _face_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "audit_raw_detail_recoverability",
    PROJECT_ROOT / "scripts" / "audit_raw_detail_recoverability.py",
)
audit_raw_detail_recoverability = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(audit_raw_detail_recoverability)


def _ellipse_points(center, radii):
    angles = np.linspace(0.0, 2.0 * np.pi, 68, endpoint=False)
    return np.stack(
        [
            center[0] + radii[0] * np.cos(angles),
            center[1] + radii[1] * np.sin(angles),
        ],
        axis=1,
    ).astype(np.float32)


def test_raw_to_aligned_similarity_transform_is_recovered():
    raw_points = _ellipse_points((320.0, 220.0), (90.0, 120.0))
    expected = np.array([[0.42, -0.03, -72.0], [0.03, 0.42, -48.0]], dtype=np.float64)
    homogeneous = np.concatenate([raw_points, np.ones((68, 1), dtype=np.float32)], axis=1)
    aligned_points = homogeneous @ expected.T

    matrix, inlier_ratio, rmse = _estimate_raw_to_aligned(
        raw_points,
        aligned_points.astype(np.float32),
        ransac_threshold=0.5,
    )

    assert matrix is not None
    assert inlier_ratio > 0.99
    assert rmse < 1e-3
    assert np.allclose(matrix, expected, atol=1e-3)


def test_common_mask_texture_metrics_detect_raw_detail_advantage():
    size = 112
    points = _ellipse_points((56.0, 58.0), (34.0, 43.0))
    mask = _face_mask((size, size, 3), points, erosion_pixels=2)
    yy, xx = np.mgrid[:size, :size]
    checker = ((xx // 4 + yy // 4) % 2) * 70 + 80
    raw = np.stack([checker, checker, checker], axis=2).astype(np.uint8)
    aligned = np.full((size, size, 3), 90, dtype=np.uint8)

    raw_metrics = _face_metrics(raw, mask)
    aligned_metrics = _face_metrics(aligned, mask)

    assert raw_metrics["gradient_energy"] > aligned_metrics["gradient_energy"]
    assert raw_metrics["entropy_bits"] > aligned_metrics["entropy_bits"]
    assert raw_metrics["luma_span_q90_q10"] > aligned_metrics["luma_span_q90_q10"]


def test_candidate_frame_routing_separates_recoverable_and_raw_clipped():
    calibration = {
        "raw_low_clip_upper": 0.05,
        "raw_high_clip_upper": 0.05,
        "low_clip_advantage_upper": 0.03,
        "high_clip_advantage_upper": 0.03,
        "gradient_ratio_upper": 1.20,
        "laplacian_ratio_upper": 1.30,
        "entropy_delta_upper": 0.10,
        "min_practical_clip_advantage": 0.01,
    }
    base = {
        "status": "VALID",
        "relevant_clip_tail": "low",
        "relevant_raw_clip_ratio": 0.01,
        "relevant_clip_advantage": 0.15,
        "raw_to_aligned_gradient_ratio": 1.50,
        "raw_to_aligned_laplacian_ratio": 1.10,
        "raw_minus_aligned_entropy_bits": 0.05,
    }

    assert _classify_candidate_frame(base, calibration) == "raw_detail_recoverable"
    assert (
        _classify_candidate_frame(
            {**base, "relevant_raw_clip_ratio": 0.20},
            calibration,
        )
        == "raw_also_clipped"
    )


def test_cli_defaults_are_read_only_and_train_calibrated():
    args = audit_raw_detail_recoverability.build_parser().parse_args(
        [
            "--frame-audit-dir",
            "/audit",
            "--dataset-root",
            "/dataset",
            "--image-root",
            "/images",
            "--raw-openface-root",
            "/raw_csv",
            "--aligned-openface-root",
            "/aligned_csv",
            "--output-dir",
            "/output",
        ]
    )

    assert args.calibration_quantile == 0.95
    assert args.min_practical_clip_advantage == 0.01
    assert args.reference_frames_per_video == 4
    assert args.candidate_frames_per_video == 32
    assert all("run_openface" not in key for key in vars(args))
