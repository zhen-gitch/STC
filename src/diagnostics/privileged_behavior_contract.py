"""Fail-closed contracts for privileged AU/head training targets.

This module deliberately contains no dataset, model, or torch dependencies. It
audits a single exact ``video_id + task_name`` OpenFace sequence against the
complete RGB frame manifest that the model will use. Only the depression-
relevant source columns below are ever read as behavior values:

``AU12_r, AU14_r, AU15_r, pose_Rx, pose_Ry, pose_Rz``.

Gaze, pose translation, other AUs, landmarks, and quality fields are never
promoted to targets. ``success`` and ``confidence`` are used only to construct
validity masks and coverage summaries.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


AU_SOURCE_COLUMNS = ("AU12_r", "AU14_r", "AU15_r")
POSE_ROTATION_SOURCE_COLUMNS = ("pose_Rx", "pose_Ry", "pose_Rz")
SELECTED_BEHAVIOR_SOURCE_COLUMNS = AU_SOURCE_COLUMNS + POSE_ROTATION_SOURCE_COLUMNS
POSE_VELOCITY_TARGET_COLUMNS = (
    "d_pose_Rx_dt",
    "d_pose_Ry_dt",
    "d_pose_Rz_dt",
)
TRAIN_STATISTIC_COLUMNS = AU_SOURCE_COLUMNS + POSE_VELOCITY_TARGET_COLUMNS
PHYSICAL_SPLITS = ("train", "val", "test")

FRAME_COLUMNS = ("frame", "timestamp")
QUALITY_COLUMNS = ("success", "confidence")
OPTIONAL_ROW_IDENTITY_COLUMNS = ("video_id", "task_name")
REQUIRED_OPENFACE_COLUMNS = FRAME_COLUMNS + QUALITY_COLUMNS + SELECTED_BEHAVIOR_SOURCE_COLUMNS


class PrivilegedBehaviorContractError(ValueError):
    """Raised when privileged behavior data cannot be used unambiguously."""


@dataclass(frozen=True)
class FrameReference:
    """Exact RGB frame identity and optional external source-video time."""

    video_id: str
    task_name: str
    frame: int
    timestamp: float | None

    def __post_init__(self) -> None:
        video_id = _validate_identity(self.video_id, "video_id")
        task_name = _validate_identity(self.task_name, "task_name")
        frame = _parse_frame_number(self.frame, "frame reference")
        timestamp = _parse_optional_finite_float(self.timestamp)
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "task_name", task_name)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "timestamp", timestamp)


@dataclass(frozen=True)
class BehaviorSchema:
    """Frozen schema projection proving which CSV values may be accessed."""

    fieldnames: tuple[str, ...]
    accessed_value_columns: tuple[str, ...]
    selected_behavior_columns: tuple[str, ...]
    forbidden_columns_present: tuple[str, ...]
    ignored_columns: tuple[str, ...]


@dataclass(frozen=True)
class BehaviorFrame:
    """One audited frame with separate masks for each behavior group."""

    reference: FrameReference
    au_values: tuple[float | None, float | None, float | None]
    pose_rotation: tuple[float | None, float | None, float | None]
    pose_velocity: tuple[float | None, float | None, float | None]
    pose_velocity_dt: float | None
    quality_valid: bool
    au_valid: bool
    pose_valid: bool
    pose_velocity_candidate: bool
    pose_velocity_valid: bool
    invalid_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CoverageSummary:
    """Group-specific valid counts and ratios for one exact video/task."""

    total_frame_count: int
    quality_valid_count: int
    au_valid_count: int
    pose_valid_count: int
    pose_velocity_candidate_count: int
    pose_velocity_valid_count: int
    pose_velocity_missing_timestamp_pair_count: int
    pose_velocity_illegal_dt_pair_count: int
    pose_velocity_nonconsecutive_pair_count: int
    au_nonfinite_frame_count: int
    pose_nonfinite_frame_count: int

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return float(numerator) / float(denominator) if denominator else 0.0

    @property
    def quality_valid_ratio(self) -> float:
        return self._ratio(self.quality_valid_count, self.total_frame_count)

    @property
    def au_valid_ratio(self) -> float:
        return self._ratio(self.au_valid_count, self.total_frame_count)

    @property
    def pose_valid_ratio(self) -> float:
        return self._ratio(self.pose_valid_count, self.total_frame_count)

    @property
    def pose_velocity_valid_ratio(self) -> float:
        return self._ratio(
            self.pose_velocity_valid_count,
            self.pose_velocity_candidate_count,
        )


@dataclass(frozen=True)
class PrivilegedBehaviorAudit:
    """Complete P0 result for one exact video/task sequence."""

    video_id: str
    task_name: str
    physical_split: str
    schema: BehaviorSchema
    frames: tuple[BehaviorFrame, ...]
    coverage: CoverageSummary


@dataclass(frozen=True)
class FeatureStatistics:
    """Population statistics computed exclusively from physical train data."""

    count: int
    mean: float
    std: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class TrainOnlyStatistics:
    """Frozen train-only normalization statistics keyed by target name."""

    physical_split: str
    groups: tuple[str, ...]
    features: Mapping[str, FeatureStatistics]


@dataclass(frozen=True)
class _ParsedSourceFrame:
    reference: FrameReference
    au_values: tuple[float | None, float | None, float | None]
    pose_rotation: tuple[float | None, float | None, float | None]
    quality_valid: bool
    au_valid: bool
    pose_valid: bool
    invalid_reasons: tuple[str, ...]


_MISSING = object()


def _validate_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PrivilegedBehaviorContractError(f"{label} must be a non-empty exact string")
    if value != value.strip():
        raise PrivilegedBehaviorContractError(
            f"{label} contains leading or trailing whitespace: {value!r}"
        )
    return value


def _validate_physical_split(value: Any) -> str:
    if value not in PHYSICAL_SPLITS:
        raise PrivilegedBehaviorContractError(
            f"physical_split must be one of {PHYSICAL_SPLITS}, got {value!r}"
        )
    return value


def _parse_required_finite_float(value: Any, label: str) -> float:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PrivilegedBehaviorContractError(f"missing required value: {label}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PrivilegedBehaviorContractError(f"invalid numeric value for {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise PrivilegedBehaviorContractError(f"non-finite required value for {label}: {value!r}")
    return result


def _parse_optional_finite_float(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_frame_number(value: Any, label: str) -> int:
    number = _parse_required_finite_float(value, label)
    if not number.is_integer() or number < 0:
        raise PrivilegedBehaviorContractError(
            f"{label} must be a non-negative integer, got {value!r}"
        )
    return int(number)


def _parse_success(value: Any, frame: int) -> bool:
    number = _parse_required_finite_float(value, f"success at frame {frame}")
    if number not in (0.0, 1.0):
        raise PrivilegedBehaviorContractError(
            f"success at frame {frame} must be 0 or 1, got {value!r}"
        )
    return bool(int(number))


def _parse_confidence(value: Any, frame: int) -> float:
    confidence = _parse_required_finite_float(value, f"confidence at frame {frame}")
    if not 0.0 <= confidence <= 1.0:
        raise PrivilegedBehaviorContractError(
            f"confidence at frame {frame} must be within [0, 1], got {value!r}"
        )
    return confidence


def _is_forbidden_behavior_column(column: str) -> bool:
    if column.startswith("gaze_"):
        return True
    if column.startswith("pose_T"):
        return True
    if column.startswith("AU") and column.endswith(("_r", "_c")):
        return column not in AU_SOURCE_COLUMNS
    return False


def _wrapped_angle_delta(current: float, previous: float) -> float:
    """Return the shortest signed angular difference for radian rotations."""

    raw_delta = current - previous
    return math.atan2(math.sin(raw_delta), math.cos(raw_delta))


def validate_behavior_schema(fieldnames: Sequence[Any]) -> BehaviorSchema:
    """Validate and freeze the only permitted value projection.

    Forbidden OpenFace columns may physically exist in a full-profile CSV, but
    they are recorded only by name and are never included in
    ``accessed_value_columns``.
    """

    normalized = tuple(str(field).strip() for field in fieldnames)
    if not normalized:
        raise PrivilegedBehaviorContractError("OpenFace schema is empty")
    if any(not field for field in normalized):
        raise PrivilegedBehaviorContractError("OpenFace schema contains an empty column name")

    duplicates = sorted(field for field, count in Counter(normalized).items() if count > 1)
    if duplicates:
        raise PrivilegedBehaviorContractError(
            f"duplicate OpenFace columns after whitespace normalization: {duplicates}"
        )

    missing = [column for column in REQUIRED_OPENFACE_COLUMNS if column not in normalized]
    if missing:
        raise PrivilegedBehaviorContractError(
            f"missing required privileged behavior columns: {missing}"
        )

    identity_present = tuple(
        column for column in OPTIONAL_ROW_IDENTITY_COLUMNS if column in normalized
    )
    if len(identity_present) == 1:
        raise PrivilegedBehaviorContractError(
            "row identity columns must contain both video_id and task_name, or neither"
        )

    accessed = FRAME_COLUMNS + QUALITY_COLUMNS + identity_present + SELECTED_BEHAVIOR_SOURCE_COLUMNS
    forbidden = tuple(column for column in normalized if _is_forbidden_behavior_column(column))
    ignored = tuple(
        column for column in normalized if column not in accessed and column not in forbidden
    )
    return BehaviorSchema(
        fieldnames=normalized,
        accessed_value_columns=accessed,
        selected_behavior_columns=SELECTED_BEHAVIOR_SOURCE_COLUMNS,
        forbidden_columns_present=forbidden,
        ignored_columns=ignored,
    )


def _row_value(row: Mapping[str, Any], column: str, row_index: int) -> Any:
    value = row.get(column, _MISSING)
    if value is _MISSING:
        raise PrivilegedBehaviorContractError(
            f"row {row_index} is missing schema column {column!r}"
        )
    return value


def _coerce_reference(value: FrameReference | Mapping[str, Any], index: int) -> FrameReference:
    if isinstance(value, FrameReference):
        return value
    if not isinstance(value, Mapping):
        raise PrivilegedBehaviorContractError(
            f"expected frame {index} must be FrameReference or a mapping"
        )
    missing = [
        column
        for column in (*OPTIONAL_ROW_IDENTITY_COLUMNS, "frame")
        if column not in value
    ]
    if missing:
        raise PrivilegedBehaviorContractError(
            f"expected frame {index} is missing columns: {missing}"
        )
    return FrameReference(
        video_id=value["video_id"],
        task_name=value["task_name"],
        frame=value["frame"],
        timestamp=value.get("timestamp"),
    )


def _validate_expected_frames(
    expected_frames: Sequence[FrameReference | Mapping[str, Any]],
    video_id: str,
    task_name: str,
) -> tuple[FrameReference, ...]:
    if not expected_frames:
        raise PrivilegedBehaviorContractError("expected RGB frame manifest is empty")
    references = tuple(
        _coerce_reference(reference, index)
        for index, reference in enumerate(expected_frames)
    )

    for reference in references:
        if reference.video_id != video_id or reference.task_name != task_name:
            raise PrivilegedBehaviorContractError(
                "expected frame identity differs from exact source identity: "
                f"expected {video_id!r}/{task_name!r}, got "
                f"{reference.video_id!r}/{reference.task_name!r} at frame {reference.frame}"
            )

    frame_counts = Counter(reference.frame for reference in references)
    duplicate_frames = sorted(frame for frame, count in frame_counts.items() if count > 1)
    if duplicate_frames:
        raise PrivilegedBehaviorContractError(
            f"duplicate frames in expected RGB manifest: {duplicate_frames}"
        )
    ordered = tuple(sorted(references, key=lambda reference: reference.frame))
    return ordered


def _parse_source_rows(
    rows: Sequence[Mapping[str, Any]],
    schema: BehaviorSchema,
    video_id: str,
    task_name: str,
    confidence_threshold: float,
) -> tuple[_ParsedSourceFrame, ...]:
    if not rows:
        raise PrivilegedBehaviorContractError("OpenFace behavior sequence is empty")

    parsed = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PrivilegedBehaviorContractError(f"row {row_index} is not a mapping")

        frame = _parse_frame_number(_row_value(row, "frame", row_index), f"frame at row {row_index}")
        # Aligned-image ``-fdir`` OpenFace runs can serialize timestamp=0 for
        # every row. Preserve a finite value only as source provenance; exact
        # joining and pose velocity use the external RGB/source-video manifest.
        timestamp = _parse_optional_finite_float(
            _row_value(row, "timestamp", row_index)
        )

        if "video_id" in schema.accessed_value_columns:
            row_video_id = _validate_identity(
                _row_value(row, "video_id", row_index),
                f"video_id at frame {frame}",
            )
            row_task_name = _validate_identity(
                _row_value(row, "task_name", row_index),
                f"task_name at frame {frame}",
            )
            if row_video_id != video_id or row_task_name != task_name:
                raise PrivilegedBehaviorContractError(
                    "row identity differs from exact source identity at frame "
                    f"{frame}: {row_video_id!r}/{row_task_name!r}"
                )

        success = _parse_success(_row_value(row, "success", row_index), frame)
        confidence = _parse_confidence(_row_value(row, "confidence", row_index), frame)
        quality_valid = success and confidence >= confidence_threshold

        au_values = tuple(
            _parse_optional_finite_float(_row_value(row, column, row_index))
            for column in AU_SOURCE_COLUMNS
        )
        pose_rotation = tuple(
            _parse_optional_finite_float(_row_value(row, column, row_index))
            for column in POSE_ROTATION_SOURCE_COLUMNS
        )
        au_finite = all(value is not None for value in au_values)
        pose_finite = all(value is not None for value in pose_rotation)
        au_valid = quality_valid and au_finite
        pose_valid = quality_valid and pose_finite

        reasons = []
        if not success:
            reasons.append("openface_unsuccessful")
        if confidence < confidence_threshold:
            reasons.append("low_confidence")
        if not au_finite:
            reasons.append("nonfinite_au")
        if not pose_finite:
            reasons.append("nonfinite_pose")

        parsed.append(
            _ParsedSourceFrame(
                reference=FrameReference(video_id, task_name, frame, timestamp),
                au_values=au_values,
                pose_rotation=pose_rotation,
                quality_valid=quality_valid,
                au_valid=au_valid,
                pose_valid=pose_valid,
                invalid_reasons=tuple(reasons),
            )
        )

    frame_counts = Counter(item.reference.frame for item in parsed)
    duplicate_frames = sorted(frame for frame, count in frame_counts.items() if count > 1)
    if duplicate_frames:
        raise PrivilegedBehaviorContractError(
            f"ambiguous duplicate OpenFace frames: {duplicate_frames}"
        )
    ordered = tuple(sorted(parsed, key=lambda item: item.reference.frame))
    return ordered


def _join_exact_frame_contract(
    parsed: Sequence[_ParsedSourceFrame],
    expected: Sequence[FrameReference],
) -> tuple[_ParsedSourceFrame, ...]:
    parsed_by_frame = {item.reference.frame: item for item in parsed}
    expected_by_frame = {item.frame: item for item in expected}
    parsed_frames = set(parsed_by_frame)
    expected_frames = set(expected_by_frame)
    if parsed_frames != expected_frames:
        missing = sorted(expected_frames - parsed_frames)
        extra = sorted(parsed_frames - expected_frames)
        raise PrivilegedBehaviorContractError(
            "OpenFace/RGB frame sets differ; "
            f"missing OpenFace frames={missing}, extra OpenFace frames={extra}"
        )

    joined = []
    for frame in sorted(expected_frames):
        source = parsed_by_frame[frame]
        reference = expected_by_frame[frame]
        joined.append(
            _ParsedSourceFrame(
                reference=reference,
                au_values=source.au_values,
                pose_rotation=source.pose_rotation,
                quality_valid=source.quality_valid,
                au_valid=source.au_valid,
                pose_valid=source.pose_valid,
                invalid_reasons=source.invalid_reasons,
            )
        )
    return tuple(joined)


def _build_behavior_frames(
    joined: Sequence[_ParsedSourceFrame],
    max_pose_velocity_dt_seconds: float,
) -> tuple[BehaviorFrame, ...]:
    frames = []
    for index, source in enumerate(joined):
        velocity = (None, None, None)
        velocity_dt = None
        velocity_candidate = False
        velocity_valid = False
        reasons = list(source.invalid_reasons)

        if index > 0:
            previous = joined[index - 1]
            contiguous = source.reference.frame == previous.reference.frame + 1
            current_timestamp = source.reference.timestamp
            previous_timestamp = previous.reference.timestamp
            timestamps_available = (
                current_timestamp is not None and previous_timestamp is not None
            )
            dt = (
                current_timestamp - previous_timestamp
                if timestamps_available
                else None
            )
            legal_dt = (
                contiguous
                and dt is not None
                and 0.0 < dt <= max_pose_velocity_dt_seconds
            )
            velocity_candidate = legal_dt
            if not contiguous:
                reasons.append("pose_velocity_nonconsecutive_frame")
            elif not timestamps_available:
                reasons.append("pose_velocity_missing_reference_timestamp")
            elif dt is None or not 0.0 < dt <= max_pose_velocity_dt_seconds:
                reasons.append("pose_velocity_illegal_dt")
            elif source.pose_valid and previous.pose_valid:
                velocity = tuple(
                    _wrapped_angle_delta(current_value, previous_value) / dt
                    for current_value, previous_value in zip(
                        source.pose_rotation,
                        previous.pose_rotation,
                    )
                )
                velocity_dt = dt
                velocity_valid = True
            else:
                reasons.append("pose_velocity_invalid_endpoint")

        frames.append(
            BehaviorFrame(
                reference=source.reference,
                au_values=source.au_values,
                pose_rotation=source.pose_rotation,
                pose_velocity=velocity,
                pose_velocity_dt=velocity_dt,
                quality_valid=source.quality_valid,
                au_valid=source.au_valid,
                pose_valid=source.pose_valid,
                pose_velocity_candidate=velocity_candidate,
                pose_velocity_valid=velocity_valid,
                invalid_reasons=tuple(dict.fromkeys(reasons)),
            )
        )
    return tuple(frames)


def _summarize_coverage(frames: Sequence[BehaviorFrame]) -> CoverageSummary:
    return CoverageSummary(
        total_frame_count=len(frames),
        quality_valid_count=sum(frame.quality_valid for frame in frames),
        au_valid_count=sum(frame.au_valid for frame in frames),
        pose_valid_count=sum(frame.pose_valid for frame in frames),
        pose_velocity_candidate_count=sum(
            frame.pose_velocity_candidate for frame in frames
        ),
        pose_velocity_valid_count=sum(frame.pose_velocity_valid for frame in frames),
        pose_velocity_missing_timestamp_pair_count=sum(
            "pose_velocity_missing_reference_timestamp" in frame.invalid_reasons
            for frame in frames
        ),
        pose_velocity_illegal_dt_pair_count=sum(
            "pose_velocity_illegal_dt" in frame.invalid_reasons for frame in frames
        ),
        pose_velocity_nonconsecutive_pair_count=sum(
            "pose_velocity_nonconsecutive_frame" in frame.invalid_reasons
            for frame in frames
        ),
        au_nonfinite_frame_count=sum(
            "nonfinite_au" in frame.invalid_reasons for frame in frames
        ),
        pose_nonfinite_frame_count=sum(
            "nonfinite_pose" in frame.invalid_reasons for frame in frames
        ),
    )


def audit_privileged_behavior_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[Any],
    video_id: str,
    task_name: str,
    physical_split: str,
    expected_frames: Sequence[FrameReference | Mapping[str, Any]],
    confidence_threshold: float = 0.8,
    max_pose_velocity_dt_seconds: float = 0.1,
) -> PrivilegedBehaviorAudit:
    """Audit one exact full-frame OpenFace sequence against its RGB manifest.

    The function sorts by explicit frame id and verifies an exact frame-set
    join. Aligned-image OpenFace timestamps are not trusted; pose velocity uses
    the optional external timestamps in ``expected_frames``. Velocity is valid
    only for consecutive frame ids with legal ``dt`` and two valid pose
    endpoints, so missing time and detection failures are never bridged. AU
    masks remain auditable when the external time axis is unavailable.
    """

    video_id = _validate_identity(video_id, "video_id")
    task_name = _validate_identity(task_name, "task_name")
    physical_split = _validate_physical_split(physical_split)
    confidence_threshold = _parse_required_finite_float(
        confidence_threshold,
        "confidence_threshold",
    )
    if not 0.0 <= confidence_threshold <= 1.0:
        raise PrivilegedBehaviorContractError("confidence_threshold must be within [0, 1]")
    max_pose_velocity_dt_seconds = _parse_required_finite_float(
        max_pose_velocity_dt_seconds,
        "max_pose_velocity_dt_seconds",
    )
    if max_pose_velocity_dt_seconds <= 0.0:
        raise PrivilegedBehaviorContractError(
            "max_pose_velocity_dt_seconds must be positive"
        )

    schema = validate_behavior_schema(fieldnames)
    expected = _validate_expected_frames(expected_frames, video_id, task_name)
    parsed = _parse_source_rows(
        rows,
        schema,
        video_id,
        task_name,
        confidence_threshold,
    )
    joined = _join_exact_frame_contract(parsed, expected)
    frames = _build_behavior_frames(joined, max_pose_velocity_dt_seconds)
    return PrivilegedBehaviorAudit(
        video_id=video_id,
        task_name=task_name,
        physical_split=physical_split,
        schema=schema,
        frames=frames,
        coverage=_summarize_coverage(frames),
    )


def _feature_statistics(values: Sequence[float], feature_name: str) -> FeatureStatistics:
    if not values:
        raise PrivilegedBehaviorContractError(
            f"no valid physical-train values for feature {feature_name}"
        )
    if any(not math.isfinite(value) for value in values):
        raise PrivilegedBehaviorContractError(
            f"non-finite physical-train value reached statistics for {feature_name}"
        )
    count = len(values)
    mean = math.fsum(values) / count
    variance = math.fsum((value - mean) ** 2 for value in values) / count
    std = math.sqrt(max(0.0, variance))
    return FeatureStatistics(
        count=count,
        mean=mean,
        std=std,
        minimum=min(values),
        maximum=max(values),
    )


def fit_train_only_statistics(
    audits: Iterable[PrivilegedBehaviorAudit],
    *,
    physical_split: str,
    groups: Sequence[str] = ("au", "head"),
    minimum_std: float = 0.0,
) -> TrainOnlyStatistics:
    """Fit target statistics while refusing validation/test or mixed inputs."""

    if physical_split != "train":
        raise PrivilegedBehaviorContractError(
            "privileged behavior statistics may only be fit on physical_split='train'"
        )
    normalized_groups = tuple(str(group).strip().lower() for group in groups)
    if not normalized_groups:
        raise PrivilegedBehaviorContractError("at least one statistics group is required")
    duplicate_groups = sorted(
        group for group, count in Counter(normalized_groups).items() if count > 1
    )
    if duplicate_groups:
        raise PrivilegedBehaviorContractError(
            f"duplicate statistics groups: {duplicate_groups}"
        )
    unsupported_groups = sorted(set(normalized_groups) - {"au", "head"})
    if unsupported_groups:
        raise PrivilegedBehaviorContractError(
            f"unsupported statistics groups: {unsupported_groups}"
        )

    minimum_std = _parse_required_finite_float(minimum_std, "minimum_std")
    if minimum_std < 0.0:
        raise PrivilegedBehaviorContractError("minimum_std must be non-negative")

    audit_list = tuple(audits)
    if not audit_list:
        raise PrivilegedBehaviorContractError("cannot fit train-only statistics from no audits")
    nontrain_audits = sorted(
        (audit.video_id, audit.task_name, audit.physical_split)
        for audit in audit_list
        if audit.physical_split != "train"
    )
    if nontrain_audits:
        raise PrivilegedBehaviorContractError(
            "train-only statistics received non-train audits: "
            f"{nontrain_audits}"
        )
    identities = [(audit.video_id, audit.task_name) for audit in audit_list]
    duplicate_identities = sorted(
        identity for identity, count in Counter(identities).items() if count > 1
    )
    if duplicate_identities:
        raise PrivilegedBehaviorContractError(
            f"duplicate video/task audits in train statistics: {duplicate_identities}"
        )

    selected_statistic_columns = tuple(
        feature
        for group in normalized_groups
        for feature in (
            AU_SOURCE_COLUMNS if group == "au" else POSE_VELOCITY_TARGET_COLUMNS
        )
    )
    values_by_feature = {feature: [] for feature in selected_statistic_columns}
    for audit in audit_list:
        for frame in audit.frames:
            if "au" in normalized_groups and frame.au_valid:
                for feature, value in zip(AU_SOURCE_COLUMNS, frame.au_values):
                    if value is None:
                        raise PrivilegedBehaviorContractError(
                            f"AU mask/value inconsistency for {feature} at frame {frame.reference.frame}"
                        )
                    values_by_feature[feature].append(value)
            if "head" in normalized_groups and frame.pose_velocity_valid:
                for feature, value in zip(
                    POSE_VELOCITY_TARGET_COLUMNS,
                    frame.pose_velocity,
                ):
                    if value is None:
                        raise PrivilegedBehaviorContractError(
                            "pose velocity mask/value inconsistency at frame "
                            f"{frame.reference.frame}"
                        )
                    values_by_feature[feature].append(value)

    statistics = {
        feature: _feature_statistics(values, feature)
        for feature, values in values_by_feature.items()
    }
    near_constant = sorted(
        feature
        for feature, stats in statistics.items()
        if stats.std <= minimum_std
    )
    if near_constant:
        raise PrivilegedBehaviorContractError(
            "constant or near-constant physical-train targets are not authorized: "
            f"{near_constant}"
        )
    return TrainOnlyStatistics(
        physical_split="train",
        groups=normalized_groups,
        features=MappingProxyType(statistics),
    )
