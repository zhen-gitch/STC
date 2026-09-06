"""contracts: 42-column CSV lock, behavior axis lock, strict bool parse."""

from src.eva_di.contracts import (
    BEHAVIOR_DIM,
    FEATURES_CSV_COLUMNS,
    SCHEMA_FRAME_CACHE,
    EvaDiSchemaError,
    expected_behavior_axis_names,
    parse_bool,
)
import pytest


def test_features_csv_columns_frozen_lock():
    # Second literal list: a silent edit to contracts.py must break here too.
    expected = [
        "sample_id", "split", "task_id", "frame_index_zero", "frame_index_one",
        "timestamp", "source_video", "source_video_sha256", "frame_quality_sha256",
        "decoded", "detection_count", "confidence", "quality_status",
        "quality_geometry_status", "bbox_xyxy", "retinaface_5pt_xy_original",
        "technical_valid", "alignment_matrix_2x3", "landmarks_98_xy_original",
        "landmarks_98_xy_aligned", "aligned_image_path", "aligned_image_sha256",
        "aligned_mask_schema", "gaze_angle_x", "gaze_angle_y",
        "au_score_0", "au_score_1", "au_score_2", "au_score_3",
        "au_score_4", "au_score_5", "au_score_6", "au_score_7",
        "emotion_index", "alignment_status", "landmark_status", "behavior_status",
        "image_valid", "behavior_valid", "pair_valid", "inference_seconds", "issues",
    ]
    assert list(FEATURES_CSV_COLUMNS) == expected
    assert len(FEATURES_CSV_COLUMNS) == 42
    assert len(set(FEATURES_CSV_COLUMNS)) == 42


def test_behavior_axis_lock():
    names = expected_behavior_axis_names()
    assert len(names) == BEHAVIOR_DIM == 206
    assert names[0] == "landmark_00_x_norm"
    assert names[194:198] == ("landmark_97_x_norm", "landmark_97_y_norm",
                              "gaze_angle_x", "gaze_angle_y")
    assert names[-1] == "au_score_7"


def test_parse_bool_strict():
    assert parse_bool("True", field="f", row=1) is True
    assert parse_bool("False", field="f", row=1) is False
    for bad in ("true", "1", "", "TRUE ", "yes"):
        with pytest.raises(EvaDiSchemaError):
            parse_bool(bad, field="f", row=2)
    # trailing whitespace around the canonical spellings is accepted:
    assert parse_bool(" True ") is True


def test_cache_schema_constant():
    assert SCHEMA_FRAME_CACHE == "eva_di_frame_feature_cache_v1"
