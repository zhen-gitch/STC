import math

import pytest

from src.diagnostics.privileged_behavior_contract import (
    AU_SOURCE_COLUMNS,
    POSE_ROTATION_SOURCE_COLUMNS,
    FrameReference,
    PrivilegedBehaviorContractError,
    audit_privileged_behavior_contract,
    fit_train_only_statistics,
    validate_behavior_schema,
)


VIDEO_ID = "203_1_Freeform_video_aligned"
TASK_NAME = "Freeform"
BASE_COLUMNS = [
    "frame",
    "timestamp",
    "confidence",
    "success",
    *AU_SOURCE_COLUMNS,
    *POSE_ROTATION_SOURCE_COLUMNS,
]
FULL_PROFILE_COLUMNS = [
    *BASE_COLUMNS,
    "gaze_angle_x",
    "gaze_0_x",
    "pose_Tx",
    "pose_Ty",
    "AU01_r",
    "AU12_c",
    "AU45_c",
    "x_0",
]


class GuardedRow(dict):
    """Raise if the audit attempts to read any forbidden feature value."""

    forbidden = {
        "gaze_angle_x",
        "gaze_0_x",
        "pose_Tx",
        "pose_Ty",
        "AU01_r",
        "AU12_c",
        "AU45_c",
    }

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"forbidden feature value was accessed: {key}")
        return super().get(key, default)


def _row(
    frame,
    timestamp,
    *,
    success=1,
    confidence=0.95,
    au=(1.0, 2.0, 3.0),
    pose=(0.0, 0.0, 0.0),
    row_identity=False,
):
    values = {
        "frame": frame,
        "timestamp": timestamp,
        "success": success,
        "confidence": confidence,
        **dict(zip(AU_SOURCE_COLUMNS, au)),
        **dict(zip(POSE_ROTATION_SOURCE_COLUMNS, pose)),
        "gaze_angle_x": "must-not-read",
        "gaze_0_x": "must-not-read",
        "pose_Tx": "must-not-read",
        "pose_Ty": "must-not-read",
        "AU01_r": "must-not-read",
        "AU12_c": "must-not-read",
        "AU45_c": "must-not-read",
        "x_0": "ignored",
    }
    if row_identity:
        values.update(video_id=VIDEO_ID, task_name=TASK_NAME)
    return GuardedRow(values)


def _expected(frames_and_timestamps):
    return [
        FrameReference(VIDEO_ID, TASK_NAME, frame, timestamp)
        for frame, timestamp in frames_and_timestamps
    ]


def _audit(rows, *, fieldnames=FULL_PROFILE_COLUMNS, expected=None, **kwargs):
    if expected is None:
        expected = _expected(
            sorted((int(row["frame"]), float(row["timestamp"])) for row in rows)
        )
    return audit_privileged_behavior_contract(
        rows,
        fieldnames=fieldnames,
        video_id=VIDEO_ID,
        task_name=TASK_NAME,
        physical_split=kwargs.pop("physical_split", "train"),
        expected_frames=expected,
        **kwargs,
    )


def test_schema_projects_only_whitelist_and_never_reads_forbidden_values():
    schema = validate_behavior_schema(FULL_PROFILE_COLUMNS)

    assert schema.selected_behavior_columns == (
        "AU12_r",
        "AU14_r",
        "AU15_r",
        "pose_Rx",
        "pose_Ry",
        "pose_Rz",
    )
    assert set(schema.forbidden_columns_present) == GuardedRow.forbidden
    assert not set(schema.forbidden_columns_present) & set(schema.accessed_value_columns)

    audit = _audit(
        [
            _row(2, 0.04, pose=(0.2, 0.4, 0.6)),
            _row(1, 0.00, pose=(0.1, 0.2, 0.3)),
        ]
    )

    assert [frame.reference.frame for frame in audit.frames] == [1, 2]
    assert audit.frames[1].pose_velocity_valid
    assert audit.frames[1].pose_velocity == pytest.approx((2.5, 5.0, 7.5))
    assert audit.coverage.au_valid_ratio == pytest.approx(1.0)
    assert audit.coverage.pose_velocity_valid_ratio == pytest.approx(1.0)


def test_schema_fails_closed_for_duplicate_missing_and_partial_identity_columns():
    with pytest.raises(PrivilegedBehaviorContractError, match="duplicate OpenFace columns"):
        validate_behavior_schema([*BASE_COLUMNS, " AU12_r "])

    with pytest.raises(PrivilegedBehaviorContractError, match="missing required"):
        validate_behavior_schema([column for column in BASE_COLUMNS if column != "pose_Rz"])

    with pytest.raises(PrivilegedBehaviorContractError, match="both video_id and task_name"):
        validate_behavior_schema([*BASE_COLUMNS, "video_id"])


@pytest.mark.parametrize("identity_key,bad_value", [("video_id", "203"), ("task_name", "Northwind")])
def test_exact_row_video_and_task_identity_are_required(identity_key, bad_value):
    fieldnames = [*BASE_COLUMNS, "video_id", "task_name"]
    row = _row(1, 0.0, row_identity=True)
    row[identity_key] = bad_value

    with pytest.raises(PrivilegedBehaviorContractError, match="row identity differs"):
        _audit([row], fieldnames=fieldnames)


def test_exact_frame_join_fails_missing_extra_and_identity_mismatch():
    rows = [_row(1, 0.0), _row(2, 0.0)]

    with pytest.raises(PrivilegedBehaviorContractError, match="frame sets differ"):
        _audit(rows, expected=_expected([(1, 0.0)]))

    with pytest.raises(PrivilegedBehaviorContractError, match="expected frame identity differs"):
        _audit(
            rows,
            expected=[
                FrameReference(VIDEO_ID, TASK_NAME, 1, 0.0),
                FrameReference(VIDEO_ID, "Northwind", 2, 0.04),
            ],
        )


def test_duplicate_frames_are_ambiguous_but_constant_openface_timestamps_are_allowed():
    with pytest.raises(PrivilegedBehaviorContractError, match="duplicate OpenFace frames"):
        _audit(
            [_row(1, 0.0), _row(1, 0.0)],
            expected=_expected([(1, 0.0)]),
        )

    audit = _audit(
        [_row(1, 0.0, pose=(0.0, 0.0, 0.0)), _row(2, 0.0, pose=(0.4, 0.8, 1.2))],
        expected=_expected([(1, 0.00), (2, 0.04)]),
    )

    assert audit.frames[1].pose_velocity_valid is True
    assert audit.frames[1].pose_velocity == pytest.approx((10.0, 20.0, 30.0))


def test_quality_and_nonfinite_values_only_change_group_masks_without_pseudo_zero():
    rows = [
        _row(1, 0.00, pose=(0.0, 0.0, 0.0)),
        _row(2, 0.04, success=0, pose=(0.1, 0.1, 0.1)),
        _row(3, 0.08, pose=(0.2, 0.2, 0.2)),
        _row(4, 0.12, confidence=0.5, pose=(0.3, 0.3, 0.3)),
        _row(5, 0.16, au=(math.nan, 2.0, 3.0), pose=(0.4, 0.4, 0.4)),
        _row(6, 0.20, pose=(math.inf, 0.5, 0.5)),
        _row(7, 0.24, pose=(0.6, 0.6, 0.6)),
        _row(8, 0.28, pose=(0.7, 0.7, 0.7)),
    ]

    audit = _audit(rows)
    frames = audit.frames

    assert frames[1].quality_valid is False
    assert frames[2].pose_valid is True
    assert frames[2].pose_velocity_valid is False
    assert "pose_velocity_invalid_endpoint" in frames[2].invalid_reasons
    assert frames[4].au_valid is False
    assert frames[4].au_values[0] is None
    assert frames[4].pose_valid is True
    assert frames[5].pose_valid is False
    assert frames[5].pose_rotation[0] is None
    assert frames[6].pose_velocity_valid is False
    assert frames[7].pose_velocity_valid is True

    coverage = audit.coverage
    assert coverage.total_frame_count == 8
    assert coverage.quality_valid_count == 6
    assert coverage.au_valid_count == 5
    assert coverage.pose_valid_count == 5
    assert coverage.pose_velocity_candidate_count == 7
    assert coverage.pose_velocity_valid_count == 1
    assert coverage.au_nonfinite_frame_count == 1
    assert coverage.pose_nonfinite_frame_count == 1


def test_pose_velocity_uses_full_rate_legal_dt_and_never_bridges_invalid_gap():
    rows = [
        _row(1, 0.0, pose=(0.0, 0.0, 0.0)),
        _row(2, 0.0, pose=(0.4, 0.8, 1.2)),
        _row(3, 0.0, pose=(0.8, 1.6, 2.4)),
        _row(4, 0.0, success=0, pose=(1.2, 2.4, 3.6)),
        _row(5, 0.0, pose=(1.6, 3.2, 4.8)),
        _row(6, 0.0, pose=(2.0, 4.0, 6.0)),
    ]

    audit = _audit(
        rows,
        expected=_expected(
            [(1, 0.00), (2, 0.04), (3, 0.20), (4, 0.24), (5, 0.28), (6, 0.32)]
        ),
        max_pose_velocity_dt_seconds=0.05,
    )
    frames = audit.frames

    assert frames[1].pose_velocity == pytest.approx((10.0, 20.0, 30.0))
    assert frames[2].pose_velocity_valid is False
    assert "pose_velocity_illegal_dt" in frames[2].invalid_reasons
    assert frames[3].pose_velocity_valid is False
    assert frames[4].pose_velocity_valid is False
    assert frames[5].pose_velocity == pytest.approx((10.0, 20.0, 30.0))


def test_pose_velocity_rejects_nonconsecutive_frame_ids_even_with_legal_dt():
    audit = _audit(
        [
            _row(1, 0.0, pose=(0.0, 0.0, 0.0)),
            _row(3, 0.0, pose=(0.4, 0.8, 1.2)),
        ],
        expected=_expected([(1, 0.00), (3, 0.04)]),
    )

    assert audit.frames[1].pose_velocity_candidate is False
    assert audit.frames[1].pose_velocity_valid is False
    assert "pose_velocity_nonconsecutive_frame" in audit.frames[1].invalid_reasons
    assert audit.coverage.pose_velocity_nonconsecutive_pair_count == 1


def test_pose_velocity_wraps_radian_delta_across_pi_boundary():
    rows = [
        _row(1, 0.0, pose=(math.pi - 0.01, -math.pi + 0.02, 0.0)),
        _row(2, 0.0, pose=(-math.pi + 0.01, math.pi - 0.02, 0.04)),
    ]

    audit = _audit(
        rows,
        expected=_expected([(1, 0.00), (2, 0.04)]),
    )

    assert audit.frames[1].pose_velocity_valid is True
    assert audit.frames[1].pose_velocity == pytest.approx((0.5, -1.0, 1.0))


def test_missing_external_timestamps_block_head_but_preserve_au_audit():
    rows = [
        _row(1, 0.0, au=(1.0, 2.0, 3.0), pose=(0.0, 0.0, 0.0)),
        _row(2, 0.0, au=(3.0, 4.0, 5.0), pose=(0.4, 0.8, 1.2)),
    ]
    expected = [
        FrameReference(VIDEO_ID, TASK_NAME, 1, None),
        FrameReference(VIDEO_ID, TASK_NAME, 2, None),
    ]

    audit = _audit(rows, expected=expected)

    assert audit.coverage.au_valid_count == 2
    assert audit.coverage.pose_velocity_candidate_count == 0
    assert audit.coverage.pose_velocity_valid_count == 0
    assert audit.coverage.pose_velocity_missing_timestamp_pair_count == 1
    assert "pose_velocity_missing_reference_timestamp" in audit.frames[1].invalid_reasons

    au_stats = fit_train_only_statistics(
        [audit],
        physical_split="train",
        groups=("au",),
    )
    assert au_stats.groups == ("au",)
    assert set(au_stats.features) == set(AU_SOURCE_COLUMNS)
    with pytest.raises(PrivilegedBehaviorContractError, match="no valid physical-train values"):
        fit_train_only_statistics(
            [audit],
            physical_split="train",
            groups=("head",),
        )


@pytest.mark.parametrize(
    "column,value,error",
    [
        ("frame", math.nan, "non-finite required value"),
        ("success", math.nan, "non-finite required value"),
        ("confidence", math.inf, "non-finite required value"),
        ("success", 0.5, "must be 0 or 1"),
        ("confidence", 1.5, "within \\[0, 1\\]"),
    ],
)
def test_nonfinite_or_ambiguous_key_and_quality_values_fail_closed(column, value, error):
    row = _row(1, 0.0)
    row[column] = value

    with pytest.raises(PrivilegedBehaviorContractError, match=error):
        _audit([row], expected=_expected([(1, 0.0)]))


def test_train_only_statistics_use_only_valid_targets_and_refuse_nontrain_split():
    first = _audit(
        [
            _row(1, 0.00, au=(1.0, 2.0, 3.0), pose=(0.0, 0.0, 0.0)),
            _row(2, 0.04, au=(3.0, 4.0, 5.0), pose=(0.4, 0.8, 1.2)),
            _row(3, 0.08, success=0, au=(99.0, 99.0, 99.0), pose=(9.0, 9.0, 9.0)),
        ]
    )
    second_rows = [
        _row(1, 0.00, au=(5.0, 6.0, 7.0), pose=(0.0, 0.0, 0.0)),
        _row(2, 0.04, au=(7.0, 8.0, 9.0), pose=(0.8, 1.2, 1.6)),
    ]
    second = audit_privileged_behavior_contract(
        second_rows,
        fieldnames=FULL_PROFILE_COLUMNS,
        video_id="204_1_Freeform_video_aligned",
        task_name=TASK_NAME,
        physical_split="train",
        expected_frames=[
            FrameReference("204_1_Freeform_video_aligned", TASK_NAME, 1, 0.0),
            FrameReference("204_1_Freeform_video_aligned", TASK_NAME, 2, 0.04),
        ],
    )

    stats = fit_train_only_statistics([first, second], physical_split="train")

    assert stats.physical_split == "train"
    assert stats.features["AU12_r"].count == 4
    assert stats.features["AU12_r"].mean == pytest.approx(4.0)
    assert stats.features["d_pose_Rx_dt"].count == 2
    assert stats.features["d_pose_Rx_dt"].mean == pytest.approx(15.0)
    assert stats.features["d_pose_Rx_dt"].std == pytest.approx(5.0)

    with pytest.raises(PrivilegedBehaviorContractError, match="physical_split='train'"):
        fit_train_only_statistics([first], physical_split="val")

    val_audit = _audit(
        [_row(1, 0.0), _row(2, 0.0, pose=(0.4, 0.8, 1.2))],
        expected=_expected([(1, 0.0), (2, 0.04)]),
        physical_split="val",
    )
    with pytest.raises(PrivilegedBehaviorContractError, match="non-train audits"):
        fit_train_only_statistics([val_audit], physical_split="train")


def test_train_statistics_fail_for_duplicate_audits_and_constant_targets():
    audit = _audit(
        [
            _row(1, 0.00, au=(1.0, 2.0, 3.0), pose=(0.0, 0.0, 0.0)),
            _row(2, 0.04, au=(1.0, 2.0, 3.0), pose=(0.4, 0.8, 1.2)),
        ]
    )

    with pytest.raises(PrivilegedBehaviorContractError, match="duplicate video/task audits"):
        fit_train_only_statistics([audit, audit], physical_split="train")

    with pytest.raises(PrivilegedBehaviorContractError, match="constant or near-constant"):
        fit_train_only_statistics([audit], physical_split="train")
