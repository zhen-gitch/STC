"""Fail-closed PB-P0C physical-fidelity audit for the frozen core AU group.

The audit compares OpenFace AU intensities extracted from the original video
with the matching aligned-JPG extraction.  It is deliberately independent of
datasets, models, labels, predictions, checkpoints, and torch.  Only
``frame``, ``success``, ``confidence``, and the policy-selected AU columns are
projected from either OpenFace CSV.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CORE_AU_COLUMNS = ("AU12_r", "AU14_r", "AU15_r")
QUALITY_COLUMNS = ("frame", "success", "confidence")
PHYSICAL_SPLITS = ("train", "val", "test")
TASK_NAMES = ("Freeform", "Northwind")
FORBIDDEN_AU_COLUMNS = ("AU04_r", "AU06_r", "AU07_r", "AU10_r", "AU17_r")
FORBIDDEN_POSE_COLUMNS = (
    "pose_Rx",
    "pose_Ry",
    "pose_Rz",
    "pose_Tx",
    "pose_Ty",
    "pose_Tz",
)
FORBIDDEN_GAZE_COLUMNS = (
    "gaze_0_x",
    "gaze_0_y",
    "gaze_0_z",
    "gaze_1_x",
    "gaze_1_y",
    "gaze_1_z",
    "gaze_angle_x",
    "gaze_angle_y",
)


class AuFidelityError(ValueError):
    """Raised when P0C input evidence is ambiguous or violates its policy."""


@dataclass(frozen=True)
class OpenFaceSequence:
    """Projected values for one exact OpenFace frame sequence."""

    frames: np.ndarray
    success: np.ndarray
    confidence: np.ndarray
    au_values: Mapping[str, np.ndarray]
    schema_sha256: str
    schema_column_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AuFidelityError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuFidelityError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuFidelityError(f"{label} must contain one JSON object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AuFidelityError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AuFidelityError(f"{label} must be an array")
    return value


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AuFidelityError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise AuFidelityError(f"{label} is not finite: {value!r}")
    return result


def _exact_int(value: Any, label: str) -> int:
    result = _finite_float(value, label)
    if not result.is_integer():
        raise AuFidelityError(f"{label} is not an integer: {value!r}")
    return int(result)


def _require_hash(value: Any, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise AuFidelityError(f"{label} must be a lowercase SHA-256")
    return normalized


def _require_equal(observed: Any, expected: Any, label: str) -> None:
    if observed != expected:
        raise AuFidelityError(f"{label} mismatch: observed={observed!r} expected={expected!r}")


def load_au_fidelity_policy(path: Path) -> dict[str, Any]:
    """Load and validate the frozen P0C policy without reading experiment data."""

    policy = _read_json(Path(path), "AU fidelity policy")
    _require_equal(policy.get("policy_id"), "pb_p0c_core_au_fidelity_v1", "policy_id")
    _require_equal(policy.get("policy_version"), 1, "policy_version")
    _require_equal(policy.get("status"), "FROZEN", "policy status")
    _require_equal(policy.get("label_blind"), True, "label_blind")

    scope = _mapping(policy.get("scope"), "scope")
    _require_equal(tuple(scope.get("au_columns", ())), CORE_AU_COLUMNS, "scope.au_columns")
    _require_equal(tuple(scope.get("quality_columns", ())), QUALITY_COLUMNS, "scope.quality_columns")
    _require_equal(scope.get("head_motion_in_eligibility"), False, "head eligibility")
    _require_equal(scope.get("gaze_in_eligibility"), False, "gaze eligibility")
    _require_equal(scope.get("extension_au_in_eligibility"), False, "extension AU eligibility")
    forbidden = set(_sequence(scope.get("forbidden_value_columns"), "forbidden value columns"))
    required_forbidden = set((*FORBIDDEN_AU_COLUMNS, *FORBIDDEN_POSE_COLUMNS, *FORBIDDEN_GAZE_COLUMNS))
    if not required_forbidden <= forbidden:
        raise AuFidelityError(
            f"forbidden value columns omit frozen exclusions: {sorted(required_forbidden - forbidden)}"
        )

    source = _mapping(policy.get("source_contract"), "source_contract")
    _require_equal(
        source.get("feature_profile"),
        "quality_2d_pose_au_no_gaze_no_hog_v1",
        "feature profile",
    )
    _require_equal(tuple(source.get("feature_arguments", ())), ("-2Dfp", "-pose", "-aus"), "feature arguments")
    confidence_threshold = _finite_float(source.get("confidence_threshold"), "confidence threshold")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise AuFidelityError("confidence threshold must be within [0, 1]")
    minimum = _finite_float(source.get("au_value_minimum"), "AU minimum")
    maximum = _finite_float(source.get("au_value_maximum"), "AU maximum")
    if minimum >= maximum:
        raise AuFidelityError("AU value range must be increasing")
    for field, expected in (
        ("require_exact_video_set", True),
        ("require_exact_frame_join", True),
        ("frame_offset_allowed", False),
        ("interpolation_allowed", False),
        ("imputation_allowed", False),
        ("smoothing_allowed", False),
    ):
        _require_equal(source.get(field), expected, f"source_contract.{field}")
    for field in (
        "required_dataset_split_sha256",
        "required_source_video_contract_sha256",
        "required_aligned_run_manifest_sha256",
        "required_p0b_run_manifest_sha256",
        "required_p0b_selected_target_manifest_sha256",
        "required_feature_extraction_sha256",
        "required_model_sha256",
        "required_openface_readme_sha256",
        "required_au_predictor_manifest_sha256",
    ):
        _require_hash(source.get(field), f"source_contract.{field}")

    statistics = _mapping(policy.get("statistics"), "statistics")
    _require_equal(
        tuple(statistics.get("aggregation_order", ())),
        ("video", "subject", "physical_split"),
        "statistics.aggregation_order",
    )
    _require_equal(
        statistics.get("task_sequence_concatenation_allowed"),
        False,
        "task sequence concatenation",
    )
    if _exact_int(statistics.get("minimum_matched_frames_per_video"), "minimum matched frames") <= 1:
        raise AuFidelityError("minimum matched frames must exceed one")
    if _exact_int(statistics.get("minimum_unique_values_per_side"), "minimum unique values") < 2:
        raise AuFidelityError("minimum unique values must be at least two")
    spearman = _mapping(statistics.get("spearman"), "statistics.spearman")
    if _exact_int(spearman.get("bootstrap_samples"), "bootstrap samples") <= 0:
        raise AuFidelityError("bootstrap samples must be positive")
    lag = _mapping(statistics.get("lag"), "statistics.lag")
    minimum_lag = _exact_int(lag.get("minimum_lag_frames"), "minimum lag")
    maximum_lag = _exact_int(lag.get("maximum_lag_frames"), "maximum lag")
    if not minimum_lag < 0 < maximum_lag or abs(minimum_lag) != abs(maximum_lag):
        raise AuFidelityError("lag range must be symmetric around zero")
    _require_equal(
        _mapping(statistics.get("cross_talk"), "statistics.cross_talk").get("decision_role"),
        "REPORT_ONLY",
        "cross-talk decision role",
    )
    split_policy = _mapping(policy.get("split_policy"), "split_policy")
    _require_equal(
        split_policy.get("eligibility_split"),
        "train",
        "eligibility split",
    )
    _require_equal(split_policy.get("labels_accessed"), False, "label access policy")
    decision = _mapping(policy.get("decision_contract"), "decision_contract")
    for field in ("training_authorized", "p0d_authorized", "p0e_authorized"):
        _require_equal(decision.get(field), False, f"decision_contract.{field}")
    return policy


def _assert_file_hash(path: Path, expected: Any, label: str) -> str:
    expected_hash = _require_hash(expected, f"expected {label} hash")
    if not path.is_file():
        raise AuFidelityError(f"{label} does not exist: {path}")
    observed = _sha256_file(path)
    if observed != expected_hash:
        raise AuFidelityError(f"{label} SHA-256 mismatch: {observed} != {expected_hash}")
    return observed


def _read_exact_split(path: Path, expected_count: int) -> tuple[dict[str, str], dict[str, list[str]]]:
    payload = _read_json(path, "dataset split")
    if tuple(payload) != PHYSICAL_SPLITS:
        raise AuFidelityError(
            f"dataset split keys/order must be exactly {PHYSICAL_SPLITS}, got {tuple(payload)}"
        )
    split_by_video: dict[str, str] = {}
    videos_by_split: dict[str, list[str]] = {}
    for split in PHYSICAL_SPLITS:
        values = list(_sequence(payload.get(split), f"dataset split {split}"))
        if any(not isinstance(value, str) or not value for value in values):
            raise AuFidelityError(f"dataset split {split} contains an invalid video id")
        if len(values) != len(set(values)):
            raise AuFidelityError(f"dataset split {split} contains duplicate video ids")
        videos_by_split[split] = values
        for video_id in values:
            if video_id in split_by_video:
                raise AuFidelityError(f"video occurs in multiple physical splits: {video_id}")
            split_by_video[video_id] = split
    if len(split_by_video) != expected_count:
        raise AuFidelityError(
            f"dataset split video count mismatch: {len(split_by_video)} != {expected_count}"
        )
    return split_by_video, videos_by_split


def _read_projected_csv_rows(
    path: Path,
    required_columns: Sequence[str],
    label: str,
) -> tuple[list[str], list[list[str]]]:
    if not path.is_file():
        raise AuFidelityError(f"{label} does not exist: {path}")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            raw_header = next(reader, None)
            if raw_header is None:
                raise AuFidelityError(f"{label} has no header: {path}")
            header = [value.strip() for value in raw_header]
            if len(header) != len(set(header)):
                duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
                raise AuFidelityError(f"{label} has duplicate columns: {duplicates}")
            missing = [column for column in required_columns if column not in header]
            if missing:
                raise AuFidelityError(f"{label} is missing required columns: {missing}")
            indexes = [header.index(column) for column in required_columns]
            rows: list[list[str]] = []
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise AuFidelityError(
                        f"{label} row {row_number} has {len(row)} columns, expected {len(header)}"
                    )
                rows.append([row[index].strip() for index in indexes])
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AuFidelityError(f"cannot read {label}: {path}: {exc}") from exc
    return header, rows


def _read_source_contract(
    path: Path,
    split_by_video: Mapping[str, str],
    expected_frame_count: int,
    access_counts: Counter[str],
) -> dict[str, dict[str, Any]]:
    selected_columns = (
        "split",
        "video_id",
        "raw_video_id",
        "task_name",
        "aligned_frame_count",
        "raw_frame_count",
        "frame_count_match",
        "raw_fps",
        "contract_status",
        "issues",
    )
    _, values = _read_projected_csv_rows(path, selected_columns, "source-video contract")
    records: dict[str, dict[str, Any]] = {}
    total_frames = 0
    for row_number, row in enumerate(values, start=2):
        record = dict(zip(selected_columns, row))
        access_counts.update(selected_columns)
        video_id = record["video_id"]
        if not video_id or video_id in records:
            raise AuFidelityError(f"source-video contract has empty/duplicate video id at row {row_number}")
        if video_id not in split_by_video:
            raise AuFidelityError(f"source-video contract contains unexpected video: {video_id}")
        expected_split = split_by_video[video_id]
        _require_equal(record["split"], expected_split, f"source split for {video_id}")
        task_name = _task_from_video_id(video_id)
        _require_equal(record["task_name"], task_name, f"source task for {video_id}")
        expected_raw_id = video_id.removesuffix("_aligned")
        _require_equal(record["raw_video_id"], expected_raw_id, f"raw video id for {video_id}")
        aligned_count = _exact_int(record["aligned_frame_count"], f"aligned frames for {video_id}")
        raw_count = _exact_int(record["raw_frame_count"], f"raw frames for {video_id}")
        _require_equal(aligned_count, raw_count, f"raw/aligned frame count for {video_id}")
        _require_equal(_exact_int(record["frame_count_match"], f"frame match for {video_id}"), 1, f"frame match for {video_id}")
        if abs(_finite_float(record["raw_fps"], f"raw fps for {video_id}") - 30.0) > 1e-6:
            raise AuFidelityError(f"source-video FPS is not 30 for {video_id}")
        _require_equal(record["contract_status"], "PASS", f"source contract status for {video_id}")
        if record["issues"]:
            raise AuFidelityError(f"source-video contract has issues for {video_id}: {record['issues']}")
        record["frame_count"] = aligned_count
        records[video_id] = record
        total_frames += aligned_count
    if set(records) != set(split_by_video):
        missing = sorted(set(split_by_video) - set(records))
        raise AuFidelityError(f"source-video contract video set mismatch; missing={missing[:20]}")
    if total_frames != expected_frame_count:
        raise AuFidelityError(
            f"source-video contract frame count mismatch: {total_frames} != {expected_frame_count}"
        )
    return records


def _task_from_video_id(video_id: str) -> str:
    matches = [task for task in TASK_NAMES if f"_{task}_" in video_id]
    if len(matches) != 1:
        raise AuFidelityError(f"video id has no unambiguous task token: {video_id}")
    return matches[0]


def _subject_from_video_id(video_id: str) -> str:
    parts = video_id.split("_")
    if len(parts) < 4 or not parts[0].isdigit() or not parts[1].isdigit():
        raise AuFidelityError(f"cannot derive NNN_M subject id from video: {video_id}")
    return f"{parts[0]}_{parts[1]}"


def _read_content_manifest(path: Path, expected_videos: Iterable[str], label: str) -> dict[str, dict[str, Any]]:
    required = (
        "video_id",
        "csv_rows",
        "schema_sha256",
        "csv_size_bytes",
        "csv_sha256",
        "status",
    )
    _, rows = _read_projected_csv_rows(path, required, label)
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        record = dict(zip(required, row))
        video_id = record["video_id"]
        if not video_id or video_id in records:
            raise AuFidelityError(f"{label} has an empty or duplicate video id: {video_id!r}")
        record["csv_rows"] = _exact_int(record["csv_rows"], f"{label} csv_rows for {video_id}")
        record["csv_size_bytes"] = _exact_int(record["csv_size_bytes"], f"{label} size for {video_id}")
        record["schema_sha256"] = _require_hash(record["schema_sha256"], f"{label} schema hash for {video_id}")
        record["csv_sha256"] = _require_hash(record["csv_sha256"], f"{label} CSV hash for {video_id}")
        records[video_id] = record
    expected_set = set(expected_videos)
    if set(records) != expected_set:
        missing = sorted(expected_set - set(records))
        extra = sorted(set(records) - expected_set)
        raise AuFidelityError(f"{label} video set mismatch: missing={missing[:20]} extra={extra[:20]}")
    return records


def _validate_common_openface_manifest(
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    label: str,
) -> None:
    source = _mapping(policy.get("source_contract"), "source_contract")
    _require_equal(manifest.get("feature_profile"), source.get("feature_profile"), f"{label} feature profile")
    _require_equal(tuple(manifest.get("feature_arguments", ())), tuple(source.get("feature_arguments", ())), f"{label} feature arguments")
    bindings = (
        ("feature_extraction_sha256", "required_feature_extraction_sha256"),
        ("model_sha256", "required_model_sha256"),
        ("openface_readme_sha256", "required_openface_readme_sha256"),
        ("au_predictor_manifest_sha256", "required_au_predictor_manifest_sha256"),
        ("source_video_contract_sha256", "required_source_video_contract_sha256"),
    )
    for manifest_field, policy_field in bindings:
        observed = _require_hash(manifest.get(manifest_field), f"{label}.{manifest_field}")
        expected = _require_hash(source.get(policy_field), f"policy.{policy_field}")
        _require_equal(observed, expected, f"{label} {manifest_field}")
    for field in ("gaze_requested", "hog_enabled", "tracked_video_enabled", "aligned_image_generation_enabled"):
        _require_equal(manifest.get(field), False, f"{label} {field}")


def _validate_p0b_binding(
    p0b_run: Mapping[str, Any],
    selected: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    source = _mapping(policy.get("source_contract"), "source_contract")
    _require_equal(p0b_run.get("status"), "PASS", "P0B run status")
    _require_equal(p0b_run.get("training_authorized"), False, "P0B training authorization")
    counts = _mapping(p0b_run.get("counts"), "P0B counts")
    _require_equal(counts.get("video_count"), source.get("expected_video_count"), "P0B video count")
    _require_equal(counts.get("frame_count"), source.get("expected_frame_count"), "P0B frame count")
    _require_equal(counts.get("core_audited_video_count"), source.get("expected_video_count"), "P0B core video count")
    _require_equal(counts.get("exact_join_video_count"), source.get("expected_video_count"), "P0B exact join count")
    _require_equal(counts.get("blocking_issue_count"), 0, "P0B blocking issues")

    _require_equal(selected.get("status"), "PASS", "P0B selected-target status")
    _require_equal(selected.get("training_authorized"), False, "selected-target training authorization")
    groups = _mapping(selected.get("groups"), "P0B selected groups")
    _require_equal(tuple(groups.get("au", ())), CORE_AU_COLUMNS, "P0B core AU group")
    provenance = _mapping(selected.get("provenance"), "P0B provenance")
    hash_bindings = (
        ("dataset_split_sha256", "required_dataset_split_sha256"),
        ("source_run_manifest_sha256", "required_aligned_run_manifest_sha256"),
        ("source_video_contract_sha256", "required_source_video_contract_sha256"),
    )
    for selected_field, policy_field in hash_bindings:
        _require_equal(
            _require_hash(provenance.get(selected_field), f"P0B {selected_field}"),
            _require_hash(source.get(policy_field), f"policy {policy_field}"),
            f"P0B {selected_field}",
        )
    raw_openface = _mapping(selected.get("raw_openface"), "P0B raw_openface access")
    _require_equal(raw_openface.get("feature_file_open_count"), 0, "P0B raw OpenFace file access")
    _require_equal(raw_openface.get("feature_value_access_count"), 0, "P0B raw OpenFace value access")


def _validate_mask_aware_aligned_compatibility(
    p0b_run: Mapping[str, Any],
    selected: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen evidence that makes legacy aligned FAIL non-structural."""

    source = _mapping(policy.get("source_contract"), "source_contract")
    expected_videos = _exact_int(source.get("expected_video_count"), "expected video count")
    expected_frames = _exact_int(source.get("expected_frame_count"), "expected frame count")
    p0b_counts = _mapping(p0b_run.get("counts"), "P0B counts")
    _require_equal(
        p0b_counts.get("rich_schema_video_count"),
        expected_videos,
        "P0B rich schema video count",
    )

    rich = _mapping(selected.get("rich_provenance"), "P0B rich provenance")
    _require_equal(rich.get("mode"), "strict_rich", "P0B rich provenance mode")
    _require_equal(rich.get("status"), "PASS", "P0B rich provenance status")
    _require_equal(
        rich.get("legacy_strict_status_observed"),
        "FAIL",
        "P0B legacy strict status",
    )
    _require_equal(
        rich.get("legacy_strict_status_used_as_gate"),
        False,
        "P0B legacy strict gate",
    )
    _require_equal(
        rich.get("mask_aware_decision_required"),
        True,
        "P0B mask-aware decision requirement",
    )
    legacy_pass_count = _exact_int(
        rich.get("legacy_strict_pass_count"),
        "P0B legacy strict pass count",
    )
    legacy_fail_count = _exact_int(
        rich.get("legacy_strict_fail_count"),
        "P0B legacy strict fail count",
    )
    _require_equal(
        legacy_pass_count + legacy_fail_count,
        expected_videos,
        "P0B legacy strict video count",
    )
    if legacy_fail_count <= 0:
        raise AuFidelityError("P0B mask-aware compatibility requires legacy strict failures")

    binding = _mapping(
        rich.get("coverage_policy_binding"),
        "P0B coverage policy binding",
    )
    _require_equal(binding.get("required"), True, "P0B coverage policy requirement")
    _require_equal(binding.get("status"), "PASS", "P0B coverage policy binding status")
    provided = _mapping(binding.get("provided"), "P0B coverage policy provided inputs")
    for field in ("decision", "decision_sha256", "policy"):
        _require_equal(provided.get(field), True, f"P0B coverage policy provided {field}")

    decision = _mapping(binding.get("decision"), "P0B coverage policy decision")
    _require_equal(
        decision.get("status"),
        "PASS_FULL_SOURCE_COVERAGE",
        "P0B coverage policy decision status",
    )
    _require_equal(decision.get("scope"), "full", "P0B coverage policy decision scope")
    expected_decision_sha256 = _require_hash(
        decision.get("expected_sha256"),
        "P0B expected coverage decision hash",
    )
    observed_decision_sha256 = _require_hash(
        decision.get("observed_sha256"),
        "P0B observed coverage decision hash",
    )
    _require_equal(
        observed_decision_sha256,
        expected_decision_sha256,
        "P0B coverage decision hash",
    )

    policy_binding = _mapping(binding.get("policy"), "P0B coverage policy")
    coverage_policy_sha256 = _require_hash(
        policy_binding.get("sha256"),
        "P0B coverage policy hash",
    )
    _require_equal(
        _require_hash(decision.get("policy_sha256"), "P0B decision policy hash"),
        coverage_policy_sha256,
        "P0B coverage decision policy hash",
    )

    observed_counts = _mapping(
        binding.get("observed_counts"),
        "P0B coverage policy observed counts",
    )
    _require_equal(
        observed_counts.get("video_count"),
        expected_videos,
        "P0B mask-aware video count",
    )
    _require_equal(
        observed_counts.get("frame_count"),
        expected_frames,
        "P0B mask-aware frame count",
    )

    p0b_inputs = _mapping(p0b_run.get("inputs"), "P0B inputs")
    run_decision = _mapping(
        p0b_inputs.get("coverage_policy_decision"),
        "P0B run coverage policy decision",
    )
    _require_equal(run_decision.get("provided"), True, "P0B run coverage decision provided")
    _require_equal(
        _require_hash(run_decision.get("expected_sha256"), "P0B run expected decision hash"),
        observed_decision_sha256,
        "P0B run expected coverage decision hash",
    )
    _require_equal(
        _require_hash(run_decision.get("sha256"), "P0B run coverage decision hash"),
        observed_decision_sha256,
        "P0B run coverage decision hash",
    )
    run_policy = _mapping(
        p0b_inputs.get("coverage_policy"),
        "P0B run coverage policy",
    )
    _require_equal(run_policy.get("provided"), True, "P0B run coverage policy provided")
    _require_equal(
        _require_hash(run_policy.get("sha256"), "P0B run coverage policy hash"),
        coverage_policy_sha256,
        "P0B run coverage policy hash",
    )

    return {
        "enabled": True,
        "evidence_source": "frozen_p0b_selected_target_manifest",
        "accepted_aligned_status": "FAIL",
        "legacy_strict_status_observed": "FAIL",
        "legacy_strict_status_used_as_gate": False,
        "legacy_strict_pass_count": legacy_pass_count,
        "legacy_strict_fail_count": legacy_fail_count,
        "mask_aware_decision_status": "PASS_FULL_SOURCE_COVERAGE",
        "mask_aware_decision_sha256": observed_decision_sha256,
        "mask_aware_policy_sha256": coverage_policy_sha256,
    }


def _resolve_aligned_status_compatibility(
    aligned_content: Mapping[str, Mapping[str, Any]],
    p0b_run: Mapping[str, Any],
    selected: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(str(record.get("status")) for record in aligned_content.values())
    unexpected = sorted(set(status_counts) - {"PASS", "FAIL"})
    if unexpected:
        raise AuFidelityError(f"aligned content manifest has unsupported statuses: {unexpected}")
    if status_counts.get("FAIL", 0) == 0:
        return {
            "enabled": False,
            "reason": "all_aligned_content_status_pass",
            "aligned_content_status_counts": dict(sorted(status_counts.items())),
            "accepted_legacy_fail_count": 0,
        }

    compatibility = _validate_mask_aware_aligned_compatibility(
        p0b_run,
        selected,
        policy,
    )
    _require_equal(
        status_counts.get("PASS", 0),
        compatibility["legacy_strict_pass_count"],
        "aligned/P0B legacy strict PASS count",
    )
    _require_equal(
        status_counts.get("FAIL", 0),
        compatibility["legacy_strict_fail_count"],
        "aligned/P0B legacy strict FAIL count",
    )
    return {
        **compatibility,
        "aligned_content_status_counts": dict(sorted(status_counts.items())),
        "accepted_legacy_fail_count": status_counts["FAIL"],
    }


def _validate_input_manifests(
    *,
    policy: Mapping[str, Any],
    aligned_manifest: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    raw_openface_root: Path,
    aligned_openface_root: Path,
    p0b_run: Mapping[str, Any],
    p0b_selected: Mapping[str, Any],
    expected_video_ids: Iterable[str],
    raw_extractor_script: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    source = _mapping(policy.get("source_contract"), "source_contract")
    expected_videos = _exact_int(source.get("expected_video_count"), "expected video count")
    expected_frames = _exact_int(source.get("expected_frame_count"), "expected frame count")
    _validate_common_openface_manifest(aligned_manifest, policy, "aligned manifest")
    _validate_common_openface_manifest(raw_manifest, policy, "raw manifest")
    _validate_p0b_binding(p0b_run, p0b_selected, policy)

    _require_equal(aligned_manifest.get("run_scope"), "full_dataset", "aligned run scope")
    _require_equal(aligned_manifest.get("selected_video_count"), expected_videos, "aligned selected videos")
    _require_equal(aligned_manifest.get("input_frame_count"), expected_frames, "aligned input frames")

    _require_equal(raw_manifest.get("run_scope"), "full_dataset", "raw run scope")
    _require_equal(raw_manifest.get("status"), "PASS", "raw run status")
    _require_equal(raw_manifest.get("input_mode"), "raw_video", "raw input mode")
    _require_equal(raw_manifest.get("selected_video_count"), expected_videos, "raw selected videos")
    _require_equal(raw_manifest.get("input_frame_count"), expected_frames, "raw input frames")
    _require_equal(raw_manifest.get("selected_frame_count"), expected_frames, "raw selected frames")
    _require_equal(raw_manifest.get("output_root_must_be_new"), True, "raw output immutability")
    _require_equal(raw_manifest.get("raw_openface_csv_path_field_access_count"), 0, "historical raw OpenFace path access")
    _require_equal(raw_manifest.get("historical_raw_openface_feature_file_open_count"), 0, "historical raw OpenFace file access")
    _require_equal(tuple(raw_manifest.get("selected_value_columns", ())), CORE_AU_COLUMNS, "raw selected AU values")
    _require_equal(raw_manifest.get("pose_value_access_count"), 0, "raw pose value access")
    _require_equal(raw_manifest.get("gaze_value_access_count"), 0, "raw gaze value access")
    _require_equal(raw_manifest.get("extension_au_value_access_count"), 0, "raw extension AU value access")
    _require_equal(raw_manifest.get("git_status_available"), True, "raw git-status availability")
    _require_equal(raw_manifest.get("provenance_stable"), True, "raw extraction provenance stability")
    for field in (
        "script_unchanged",
        "source_video_contract_unchanged",
        "binary_files_unchanged",
        "model_files_unchanged",
        "git_provenance_unchanged",
    ):
        _require_equal(raw_manifest.get(field), True, f"raw extraction {field}")
    _require_equal(
        _require_hash(raw_manifest.get("script_sha256"), "raw extractor script hash"),
        _sha256_file(raw_extractor_script),
        "raw extractor implementation hash",
    )
    if str(raw_manifest.get("git_status_short", "")).strip():
        raise AuFidelityError("full raw extraction manifest records a dirty git checkout")

    for manifest_name in ("binary_manifest", "model_manifest"):
        hash_field = f"{manifest_name}_sha256"
        aligned_hash = _require_hash(
            aligned_manifest.get(hash_field),
            f"aligned {manifest_name} hash",
        )
        raw_hash = _require_hash(raw_manifest.get(hash_field), f"raw {manifest_name} hash")
        _require_equal(raw_hash, aligned_hash, f"raw/aligned {manifest_name} hash")
        _assert_file_hash(
            aligned_openface_root / "_audit" / f"{manifest_name}.csv",
            aligned_hash,
            f"aligned {manifest_name}",
        )
        _assert_file_hash(
            raw_openface_root / "_audit" / f"{manifest_name}.csv",
            raw_hash,
            f"raw {manifest_name}",
        )

    aligned_content_path = aligned_openface_root / "_audit" / "csv_content_manifest.csv"
    raw_content_path = raw_openface_root / "_audit" / "csv_content_manifest.csv"
    aligned_content_hash = _require_hash(
        aligned_manifest.get("csv_content_manifest_sha256"),
        "aligned content manifest hash",
    )
    raw_content_hash = _require_hash(raw_manifest.get("csv_content_manifest_sha256"), "raw content manifest hash")
    _assert_file_hash(aligned_content_path, aligned_content_hash, "aligned CSV content manifest")
    _assert_file_hash(raw_content_path, raw_content_hash, "raw CSV content manifest")

    raw_summary_path = raw_openface_root / "_audit" / "extraction_summary.json"
    raw_summary_hash = _require_hash(raw_manifest.get("extraction_summary_sha256"), "raw extraction summary hash")
    _assert_file_hash(raw_summary_path, raw_summary_hash, "raw extraction summary")
    raw_summary = _read_json(raw_summary_path, "raw extraction summary")
    _require_equal(raw_summary.get("status"), "PASS", "raw extraction summary status")
    _require_equal(raw_summary.get("run_scope"), "full_dataset", "raw extraction summary scope")
    _require_equal(raw_summary.get("video_count"), expected_videos, "raw summary video count")
    _require_equal(raw_summary.get("fail_count"), 0, "raw summary failure count")
    _require_equal(raw_summary.get("total_source_frames"), expected_frames, "raw summary source frames")
    _require_equal(raw_summary.get("total_csv_rows"), expected_frames, "raw summary CSV rows")
    _require_equal(raw_summary.get("schema_count"), 1, "raw summary schema count")
    _require_equal(raw_summary.get("schema_consistent"), True, "raw summary schema consistency")
    _require_equal(raw_summary.get("hog_file_count"), 0, "raw summary HOG files")
    _require_equal(raw_summary.get("tracked_video_file_count"), 0, "raw summary tracked videos")
    _require_equal(raw_summary.get("generated_image_count"), 0, "raw summary generated images")
    _require_equal(raw_summary.get("provenance_stable"), True, "raw summary provenance stability")

    aligned_content = _read_content_manifest(
        aligned_content_path,
        expected_video_ids,
        "aligned CSV content manifest",
    )
    raw_content = _read_content_manifest(
        raw_content_path,
        expected_video_ids,
        "raw CSV content manifest",
    )
    aligned_status_compatibility = _resolve_aligned_status_compatibility(
        aligned_content,
        p0b_run,
        p0b_selected,
        policy,
    )
    return aligned_content, raw_content, aligned_status_compatibility


def _read_openface_sequence(
    *,
    path: Path,
    expected_frames: int,
    content_record: Mapping[str, Any],
    au_columns: Sequence[str],
    value_minimum: float,
    value_maximum: float,
    source_name: str,
    access_counts: Counter[str],
) -> OpenFaceSequence:
    if not path.is_file():
        raise AuFidelityError(f"{source_name} CSV does not exist: {path}")
    observed_size = path.stat().st_size
    _require_equal(observed_size, content_record.get("csv_size_bytes"), f"{source_name} CSV size")
    _assert_file_hash(path, content_record.get("csv_sha256"), f"{source_name} CSV")
    _require_equal(content_record.get("csv_rows"), expected_frames, f"{source_name} manifest row count")

    selected_columns = (*QUALITY_COLUMNS, *au_columns)
    if not path.is_file():
        raise AuFidelityError(f"{source_name} CSV does not exist: {path}")
    frames: list[int] = []
    successes: list[bool] = []
    confidences: list[float] = []
    values: dict[str, list[float]] = {column: [] for column in au_columns}
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            raw_header = next(reader, None)
            if raw_header is None:
                raise AuFidelityError(f"{source_name} CSV has no header: {path}")
            header = [name.strip() for name in raw_header]
            if len(header) != len(set(header)):
                raise AuFidelityError(f"{source_name} CSV has duplicate columns: {path}")
            missing = [column for column in selected_columns if column not in header]
            if missing:
                raise AuFidelityError(f"{source_name} CSV is missing selected columns: {missing}")
            indexes = {column: header.index(column) for column in selected_columns}
            schema_sha256 = _sha256_text(",".join(header))
            _require_equal(schema_sha256, content_record.get("schema_sha256"), f"{source_name} schema hash")

            for ordinal, row in enumerate(reader, start=1):
                if len(row) != len(header):
                    raise AuFidelityError(
                        f"{source_name} CSV row {ordinal} has {len(row)} columns, expected {len(header)}"
                    )
                frame = _exact_int(row[indexes["frame"]].strip(), f"{source_name} frame {ordinal}")
                access_counts[f"{source_name}:frame"] += 1
                if frame != ordinal:
                    raise AuFidelityError(
                        f"{source_name} frame sequence is not exact at row {ordinal}: frame={frame}"
                    )
                success_value = _finite_float(
                    row[indexes["success"]].strip(), f"{source_name} success frame {frame}"
                )
                access_counts[f"{source_name}:success"] += 1
                if success_value not in (0.0, 1.0):
                    raise AuFidelityError(
                        f"{source_name} success is outside {{0,1}} at frame {frame}: {success_value}"
                    )
                confidence = _finite_float(
                    row[indexes["confidence"]].strip(), f"{source_name} confidence frame {frame}"
                )
                access_counts[f"{source_name}:confidence"] += 1
                if not 0.0 <= confidence <= 1.0:
                    raise AuFidelityError(
                        f"{source_name} confidence is outside [0,1] at frame {frame}: {confidence}"
                    )
                frames.append(frame)
                successes.append(success_value == 1.0)
                confidences.append(confidence)
                for au in au_columns:
                    value = _finite_float(row[indexes[au]].strip(), f"{source_name} {au} frame {frame}")
                    access_counts[f"{source_name}:{au}"] += 1
                    if not value_minimum <= value <= value_maximum:
                        raise AuFidelityError(
                            f"{source_name} {au} is outside [{value_minimum},{value_maximum}] "
                            f"at frame {frame}: {value}"
                        )
                    values[au].append(value)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AuFidelityError(f"cannot read {source_name} CSV {path}: {exc}") from exc

    if len(frames) != expected_frames:
        raise AuFidelityError(
            f"{source_name} CSV row count mismatch: {len(frames)} != {expected_frames}: {path}"
        )
    return OpenFaceSequence(
        frames=np.asarray(frames, dtype=np.int64),
        success=np.asarray(successes, dtype=np.bool_),
        confidence=np.asarray(confidences, dtype=np.float64),
        au_values={column: np.asarray(column_values, dtype=np.float64) for column, column_values in values.items()},
        schema_sha256=schema_sha256,
        schema_column_count=len(header),
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    index = 0
    while index < values.size:
        end = index
        while end + 1 < values.size and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        ranks[order[index : end + 1]] = (index + end) / 2.0 + 1.0
        index = end + 1
    return ranks


def _pearson(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size < 2 or b.size != a.size:
        return None
    centered_a = a - float(np.mean(a))
    centered_b = b - float(np.mean(b))
    denominator = math.sqrt(float(np.sum(centered_a**2) * np.sum(centered_b**2)))
    if denominator <= 1e-15:
        return None
    return float(np.sum(centered_a * centered_b) / denominator)


def _spearman(values_a: np.ndarray, values_b: np.ndarray) -> float | None:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return None
    return _pearson(_average_ranks(values_a), _average_ranks(values_b))


def _estimable_spearman(
    values_a: np.ndarray,
    values_b: np.ndarray,
    valid_mask: np.ndarray,
    minimum_count: int,
    minimum_unique: int,
) -> tuple[float | None, int]:
    selected_a = np.asarray(values_a[valid_mask], dtype=np.float64)
    selected_b = np.asarray(values_b[valid_mask], dtype=np.float64)
    count = int(selected_a.size)
    if count < minimum_count:
        return None, count
    if np.unique(selected_a).size < minimum_unique or np.unique(selected_b).size < minimum_unique:
        return None, count
    return _spearman(selected_a, selected_b), count


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(np.asarray(finite, dtype=np.float64))) if finite else None


def _lag_slices(length: int, lag: int) -> tuple[slice, slice]:
    if lag > 0:
        return slice(0, length - lag), slice(lag, length)
    if lag < 0:
        return slice(-lag, length), slice(0, length + lag)
    return slice(0, length), slice(0, length)


def compute_video_au_fidelity(
    *,
    raw: OpenFaceSequence,
    aligned: OpenFaceSequence,
    policy: Mapping[str, Any],
    video_id: str,
    physical_split: str,
    task_name: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Compute all label-blind video-level P0C metrics for one exact pair."""

    if physical_split not in PHYSICAL_SPLITS:
        raise AuFidelityError(f"invalid physical split: {physical_split}")
    if task_name not in TASK_NAMES:
        raise AuFidelityError(f"invalid task name: {task_name}")
    if not np.array_equal(raw.frames, aligned.frames):
        raise AuFidelityError(f"raw/aligned frame join differs for {video_id}; offsets are forbidden")
    if tuple(raw.au_values) != CORE_AU_COLUMNS or tuple(aligned.au_values) != CORE_AU_COLUMNS:
        raise AuFidelityError(f"projected AU order differs from frozen core group for {video_id}")

    source = _mapping(policy.get("source_contract"), "source_contract")
    statistics = _mapping(policy.get("statistics"), "statistics")
    direction_policy = _mapping(statistics.get("direction"), "statistics.direction")
    amplitude_policy = _mapping(statistics.get("amplitude"), "statistics.amplitude")
    lag_policy = _mapping(statistics.get("lag"), "statistics.lag")
    confidence_threshold = _finite_float(source.get("confidence_threshold"), "confidence threshold")
    minimum_count = _exact_int(statistics.get("minimum_matched_frames_per_video"), "minimum matched frames")
    minimum_unique = _exact_int(statistics.get("minimum_unique_values_per_side"), "minimum unique values")
    raw_quality = raw.success & (raw.confidence >= confidence_threshold)
    aligned_quality = aligned.success & (aligned.confidence >= confidence_threshold)
    joint_quality = raw_quality & aligned_quality
    subject_id = _subject_from_video_id(video_id)

    coverage_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    lags = range(
        _exact_int(lag_policy.get("minimum_lag_frames"), "minimum lag"),
        _exact_int(lag_policy.get("maximum_lag_frames"), "maximum lag") + 1,
    )

    for au in CORE_AU_COLUMNS:
        raw_values = raw.au_values[au]
        aligned_values = aligned.au_values[au]
        matched_count = int(np.sum(joint_quality))
        consecutive_mask = joint_quality[:-1] & joint_quality[1:]
        consecutive_count = int(np.sum(consecutive_mask))
        coverage_rows.append(
            {
                "split": physical_split,
                "subject_id": subject_id,
                "video_id": video_id,
                "task_name": task_name,
                "au": au,
                "total_frame_count": int(raw.frames.size),
                "raw_quality_valid_count": int(np.sum(raw_quality)),
                "aligned_quality_valid_count": int(np.sum(aligned_quality)),
                "matched_valid_count": matched_count,
                "consecutive_matched_pair_count": consecutive_count,
                "raw_only_quality_valid_count": int(np.sum(raw_quality & ~aligned_quality)),
                "aligned_only_quality_valid_count": int(np.sum(aligned_quality & ~raw_quality)),
                "status": "PASS" if matched_count else "NO_MATCHED_VALID_FRAMES",
                "issues": "" if matched_count else "no_joint_quality_valid_frames",
            }
        )

        rho0, rho_count = _estimable_spearman(
            raw_values,
            aligned_values,
            joint_quality,
            minimum_count,
            minimum_unique,
        )
        raw_amplitude = None
        aligned_amplitude = None
        amplitude_ratio = None
        if matched_count >= minimum_count:
            selected_raw = raw_values[joint_quality]
            selected_aligned = aligned_values[joint_quality]
            raw_amplitude = float(np.quantile(selected_raw, 0.95) - np.quantile(selected_raw, 0.05))
            aligned_amplitude = float(
                np.quantile(selected_aligned, 0.95) - np.quantile(selected_aligned, 0.05)
            )
            minimum_raw_amplitude = _finite_float(
                amplitude_policy.get("minimum_raw_amplitude"),
                "minimum raw amplitude",
            )
            if raw_amplitude >= minimum_raw_amplitude:
                amplitude_ratio = aligned_amplitude / raw_amplitude

        delta_raw = raw_values[1:] - raw_values[:-1]
        delta_aligned = aligned_values[1:] - aligned_values[:-1]
        reference_events = consecutive_mask & (
            np.abs(delta_raw)
            >= _finite_float(
                direction_policy.get("reference_raw_absolute_delta_minimum"),
                "raw direction delta threshold",
            )
        )
        event_count = int(np.sum(reference_events))
        direction_matches = reference_events & (
            np.abs(delta_aligned)
            >= _finite_float(
                direction_policy.get("aligned_absolute_delta_minimum"),
                "aligned direction delta threshold",
            )
        ) & (np.sign(delta_raw) == np.sign(delta_aligned))
        match_count = int(np.sum(direction_matches))
        direction_rate = None
        if event_count >= _exact_int(direction_policy.get("minimum_events_per_video"), "minimum direction events"):
            direction_rate = match_count / event_count
        aligned_only_events = consecutive_mask & ~reference_events & (
            np.abs(delta_aligned)
            >= _finite_float(
                direction_policy.get("aligned_absolute_delta_minimum"),
                "aligned direction delta threshold",
            )
        )
        aligned_only_event_rate = (
            float(np.sum(aligned_only_events)) / consecutive_count if consecutive_count else None
        )

        lag_values: dict[int, float | None] = {}
        lag_counts: dict[int, int] = {}
        for lag in lags:
            raw_slice, aligned_slice = _lag_slices(len(raw.frames), lag)
            lag_valid = raw_quality[raw_slice] & aligned_quality[aligned_slice]
            lag_rho, lag_count = _estimable_spearman(
                raw_values[raw_slice],
                aligned_values[aligned_slice],
                lag_valid,
                minimum_count,
                minimum_unique,
            )
            lag_values[lag] = lag_rho
            lag_counts[lag] = lag_count
            lag_rows.append(
                {
                    "split": physical_split,
                    "aggregation": "video",
                    "subject_id": subject_id,
                    "video_id": video_id,
                    "task_name": task_name,
                    "au": au,
                    "lag_frames": lag,
                    "rho": lag_rho,
                    "pair_count": lag_count,
                    "video_count": 1,
                    "subject_count": 1,
                }
            )
        estimable_lag_values = {lag: value for lag, value in lag_values.items() if value is not None}
        strict_peak_lags: list[int] = []
        strict_peak_lag = None
        if len(estimable_lag_values) == len(tuple(lags)):
            maximum_rho = max(estimable_lag_values.values())
            strict_peak_lags = sorted(
                lag for lag, value in estimable_lag_values.items() if abs(value - maximum_rho) <= 1e-12
            )
            if len(strict_peak_lags) == 1:
                strict_peak_lag = strict_peak_lags[0]

        issues = []
        if rho0 is None:
            issues.append("rho_not_estimable")
        if amplitude_ratio is None:
            issues.append("amplitude_not_estimable")
        if direction_rate is None:
            issues.append("direction_not_estimable")
        if len(estimable_lag_values) != len(tuple(lags)):
            issues.append("lag_curve_not_estimable")
        metric_rows.append(
            {
                "split": physical_split,
                "subject_id": subject_id,
                "video_id": video_id,
                "task_name": task_name,
                "au": au,
                "matched_valid_count": matched_count,
                "rho_pair_count": rho_count,
                "rho0": rho0,
                "rho_estimable": rho0 is not None,
                "direction_event_count": event_count,
                "direction_match_count": match_count,
                "direction_rate": direction_rate,
                "direction_estimable": direction_rate is not None,
                "aligned_only_event_rate": aligned_only_event_rate,
                "raw_amplitude_q95_q05": raw_amplitude,
                "aligned_amplitude_q95_q05": aligned_amplitude,
                "amplitude_ratio": amplitude_ratio,
                "amplitude_estimable": amplitude_ratio is not None,
                "strict_peak_lag": strict_peak_lag,
                "strict_peak_lags": ";".join(str(value) for value in strict_peak_lags),
                "lag_curve_estimable": len(estimable_lag_values) == len(tuple(lags)),
                "status": "ESTIMABLE" if not issues else "PARTIAL_OR_NOT_ESTIMABLE",
                "issues": ";".join(issues),
            }
        )

    matrix_sources = {
        "raw_to_aligned": (raw.au_values, aligned.au_values, raw_quality, aligned_quality),
        "raw_to_raw": (raw.au_values, raw.au_values, raw_quality, raw_quality),
        "aligned_to_aligned": (
            aligned.au_values,
            aligned.au_values,
            aligned_quality,
            aligned_quality,
        ),
    }
    for matrix_name, (source_values, target_values, source_valid, target_valid) in matrix_sources.items():
        for source_au in CORE_AU_COLUMNS:
            for target_au in CORE_AU_COLUMNS:
                valid = source_valid & target_valid
                rho, pair_count = _estimable_spearman(
                    source_values[source_au],
                    target_values[target_au],
                    valid,
                    minimum_count,
                    minimum_unique,
                )
                cross_rows.append(
                    {
                        "split": physical_split,
                        "aggregation": "video",
                        "subject_id": subject_id,
                        "video_id": video_id,
                        "task_name": task_name,
                        "matrix": matrix_name,
                        "source_au": source_au,
                        "target_au": target_au,
                        "rho": rho,
                        "pair_count": pair_count,
                        "video_count": 1,
                        "subject_count": 1,
                        "decision_role": "REPORT_ONLY",
                    }
                )
    return coverage_rows, metric_rows, lag_rows, cross_rows


def _aggregate_group_median(
    rows: Sequence[Mapping[str, Any]],
    value_field: str,
) -> float | None:
    return _median(row.get(value_field) for row in rows)


def aggregate_au_fidelity_metrics(
    video_metric_rows: Sequence[Mapping[str, Any]],
    video_lag_rows: Sequence[Mapping[str, Any]],
    video_cross_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate video metrics to subject and split without frame weighting."""

    by_subject: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in video_metric_rows:
        by_subject[(str(row["split"]), str(row["subject_id"]), str(row["au"]))].append(row)

    subject_rows: list[dict[str, Any]] = []
    for (physical_split, subject_id, au), rows in sorted(by_subject.items()):
        tasks = sorted({str(row["task_name"]) for row in rows})
        rho_task_count = sum(row.get("rho0") is not None for row in rows)
        direction_task_count = sum(row.get("direction_rate") is not None for row in rows)
        amplitude_task_count = sum(row.get("amplitude_ratio") is not None for row in rows)
        lag_task_count = sum(bool(row.get("lag_curve_estimable")) for row in rows)
        metric_task_counts = {
            "rho": rho_task_count,
            "direction": direction_task_count,
            "amplitude": amplitude_task_count,
            "lag": lag_task_count,
        }
        available_counts = [count for count in metric_task_counts.values() if count > 0]
        single_task_metrics = [
            metric for metric, count in metric_task_counts.items() if count == 1
        ]
        if all(count == len(tasks) for count in metric_task_counts.values()):
            status = "ESTIMABLE"
        elif any(count > 0 for count in metric_task_counts.values()):
            status = "PARTIAL_ESTIMABLE"
        else:
            status = "NOT_ESTIMABLE"
        subject_rows.append(
            {
                "split": physical_split,
                "subject_id": subject_id,
                "au": au,
                "task_count": len(tasks),
                "tasks": ";".join(tasks),
                "rho_task_count": rho_task_count,
                "rho0": _aggregate_group_median(rows, "rho0"),
                "direction_task_count": direction_task_count,
                "direction_rate": _aggregate_group_median(rows, "direction_rate"),
                "amplitude_task_count": amplitude_task_count,
                "amplitude_ratio": _aggregate_group_median(rows, "amplitude_ratio"),
                "lag_task_count": lag_task_count,
                "single_task_only": bool(available_counts)
                and all(count <= 1 for count in available_counts),
                "single_task_metrics": ";".join(single_task_metrics),
                "status": status,
            }
        )

    subject_lag_rows: list[dict[str, Any]] = []
    lag_by_subject: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in video_lag_rows:
        if row.get("aggregation") == "video":
            lag_by_subject[
                (
                    str(row["split"]),
                    str(row["subject_id"]),
                    str(row["au"]),
                    int(row["lag_frames"]),
                )
            ].append(row)
    for (physical_split, subject_id, au, lag), rows in sorted(lag_by_subject.items()):
        subject_lag_rows.append(
            {
                "split": physical_split,
                "aggregation": "subject",
                "subject_id": subject_id,
                "video_id": "",
                "task_name": "",
                "au": au,
                "lag_frames": lag,
                "rho": _aggregate_group_median(rows, "rho"),
                "pair_count": sum(int(row.get("pair_count") or 0) for row in rows),
                "video_count": sum(row.get("rho") is not None for row in rows),
                "subject_count": 1,
            }
        )

    split_lag_rows: list[dict[str, Any]] = []
    lag_by_split: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in subject_lag_rows:
        lag_by_split[(str(row["split"]), str(row["au"]), int(row["lag_frames"]))].append(row)
    for (physical_split, au, lag), rows in sorted(lag_by_split.items()):
        split_lag_rows.append(
            {
                "split": physical_split,
                "aggregation": "split_subject_median",
                "subject_id": "",
                "video_id": "",
                "task_name": "",
                "au": au,
                "lag_frames": lag,
                "rho": _aggregate_group_median(rows, "rho"),
                "pair_count": sum(int(row.get("pair_count") or 0) for row in rows),
                "video_count": sum(int(row.get("video_count") or 0) for row in rows),
                "subject_count": sum(row.get("rho") is not None for row in rows),
            }
        )

    subject_cross_rows: list[dict[str, Any]] = []
    cross_by_subject: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in video_cross_rows:
        if row.get("aggregation") == "video":
            cross_by_subject[
                (
                    str(row["split"]),
                    str(row["subject_id"]),
                    str(row["matrix"]),
                    str(row["source_au"]),
                    str(row["target_au"]),
                )
            ].append(row)
    for (physical_split, subject_id, matrix_name, source_au, target_au), rows in sorted(
        cross_by_subject.items()
    ):
        subject_cross_rows.append(
            {
                "split": physical_split,
                "aggregation": "subject",
                "subject_id": subject_id,
                "video_id": "",
                "task_name": "",
                "matrix": matrix_name,
                "source_au": source_au,
                "target_au": target_au,
                "rho": _aggregate_group_median(rows, "rho"),
                "pair_count": sum(int(row.get("pair_count") or 0) for row in rows),
                "video_count": sum(row.get("rho") is not None for row in rows),
                "subject_count": 1,
                "decision_role": "REPORT_ONLY",
            }
        )

    split_cross_rows: list[dict[str, Any]] = []
    cross_by_split: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in subject_cross_rows:
        cross_by_split[
            (
                str(row["split"]),
                str(row["matrix"]),
                str(row["source_au"]),
                str(row["target_au"]),
            )
        ].append(row)
    for (physical_split, matrix_name, source_au, target_au), rows in sorted(cross_by_split.items()):
        split_cross_rows.append(
            {
                "split": physical_split,
                "aggregation": "split_subject_median",
                "subject_id": "",
                "video_id": "",
                "task_name": "",
                "matrix": matrix_name,
                "source_au": source_au,
                "target_au": target_au,
                "rho": _aggregate_group_median(rows, "rho"),
                "pair_count": sum(int(row.get("pair_count") or 0) for row in rows),
                "video_count": sum(int(row.get("video_count") or 0) for row in rows),
                "subject_count": sum(row.get("rho") is not None for row in rows),
                "decision_role": "REPORT_ONLY",
            }
        )
    for row in split_cross_rows:
        peers = [
            candidate
            for candidate in split_cross_rows
            if candidate["split"] == row["split"]
            and candidate["matrix"] == row["matrix"]
            and candidate["source_au"] == row["source_au"]
        ]
        diagonal = next(
            (
                candidate.get("rho")
                for candidate in peers
                if candidate["target_au"] == row["source_au"]
            ),
            None,
        )
        off_diagonal = [
            abs(float(candidate["rho"]))
            for candidate in peers
            if candidate["target_au"] != row["source_au"]
            and candidate.get("rho") is not None
        ]
        maximum_off_diagonal = max(off_diagonal) if off_diagonal else None
        row["diagonal_rho"] = diagonal
        row["maximum_off_diagonal_absolute_rho"] = maximum_off_diagonal
        row["diagonal_margin"] = (
            float(diagonal) - maximum_off_diagonal
            if diagonal is not None and maximum_off_diagonal is not None
            else None
        )
    return (
        subject_rows,
        [*video_lag_rows, *subject_lag_rows, *split_lag_rows],
        [*video_cross_rows, *subject_cross_rows, *split_cross_rows],
    )


def _subject_bootstrap_median_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> tuple[float | None, float | None]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size < 2 or samples <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.choice(finite, size=finite.size, replace=True)
        estimates[index] = np.median(selected)
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def _task_estimable_counts(
    video_rows: Sequence[Mapping[str, Any]],
    *,
    physical_split: str,
    au: str,
    field: str,
    require_truthy: bool = False,
) -> dict[str, int]:
    return {
        task: sum(
            (bool(row.get(field)) if require_truthy else row.get(field) is not None)
            for row in video_rows
            if row.get("split") == physical_split
            and row.get("au") == au
            and row.get("task_name") == task
        )
        for task in TASK_NAMES
    }


def _split_lag_curve(
    lag_rows: Sequence[Mapping[str, Any]],
    physical_split: str,
    au: str,
) -> dict[int, tuple[float | None, int]]:
    return {
        int(row["lag_frames"]): (
            float(row["rho"]) if row.get("rho") is not None else None,
            int(row.get("subject_count") or 0),
        )
        for row in lag_rows
        if row.get("aggregation") == "split_subject_median"
        and row.get("split") == physical_split
        and row.get("au") == au
    }


def build_au_fidelity_decision(
    *,
    policy: Mapping[str, Any],
    video_rows: Sequence[Mapping[str, Any]],
    subject_rows: Sequence[Mapping[str, Any]],
    lag_rows: Sequence[Mapping[str, Any]],
    technical_issues: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the machine decision; only physical train can affect eligibility."""

    statistics = _mapping(policy.get("statistics"), "statistics")
    spearman = _mapping(statistics.get("spearman"), "statistics.spearman")
    direction = _mapping(statistics.get("direction"), "statistics.direction")
    amplitude = _mapping(statistics.get("amplitude"), "statistics.amplitude")
    lag_policy = _mapping(statistics.get("lag"), "statistics.lag")
    contract = _mapping(policy.get("decision_contract"), "decision_contract")
    minimum_subjects = _exact_int(statistics.get("minimum_train_subjects"), "minimum train subjects")
    minimum_task_videos = _exact_int(
        statistics.get("minimum_train_videos_per_task"),
        "minimum train videos per task",
    )
    minimum_direction_subjects = _exact_int(
        direction.get("minimum_train_subjects"),
        "minimum direction subjects",
    )
    decisions: dict[str, Any] = {}

    if technical_issues:
        for au in CORE_AU_COLUMNS:
            decisions[au] = {
                "status": contract["au_not_evaluated_status"],
                "reasons": ["technical_audit_blocked"],
            }
        group_status = contract["group_not_evaluable_status"]
        audit_status = contract["audit_blocked_status"]
    else:
        for au in CORE_AU_COLUMNS:
            train_subject_rows = [
                row
                for row in subject_rows
                if row.get("split") == "train" and row.get("au") == au
            ]
            rho_values = [float(row["rho0"]) for row in train_subject_rows if row.get("rho0") is not None]
            direction_values = [
                float(row["direction_rate"])
                for row in train_subject_rows
                if row.get("direction_rate") is not None
            ]
            amplitude_values = [
                float(row["amplitude_ratio"])
                for row in train_subject_rows
                if row.get("amplitude_ratio") is not None
            ]
            rho_median = _median(rho_values)
            direction_median = _median(direction_values)
            amplitude_median = _median(amplitude_values)
            bootstrap_low, bootstrap_high = _subject_bootstrap_median_ci(
                rho_values,
                samples=_exact_int(spearman.get("bootstrap_samples"), "bootstrap samples"),
                seed=_exact_int(spearman.get("bootstrap_seed"), "bootstrap seed"),
                confidence_level=_finite_float(
                    spearman.get("bootstrap_confidence_level"),
                    "bootstrap confidence level",
                ),
            )
            rho_task_counts = _task_estimable_counts(
                video_rows, physical_split="train", au=au, field="rho0"
            )
            amplitude_task_counts = _task_estimable_counts(
                video_rows,
                physical_split="train",
                au=au,
                field="amplitude_ratio",
            )
            lag_task_counts = _task_estimable_counts(
                video_rows,
                physical_split="train",
                au=au,
                field="lag_curve_estimable",
                require_truthy=True,
            )
            curve = _split_lag_curve(lag_rows, "train", au)
            expected_lags = list(
                range(
                    _exact_int(lag_policy.get("minimum_lag_frames"), "minimum lag"),
                    _exact_int(lag_policy.get("maximum_lag_frames"), "maximum lag") + 1,
                )
            )
            curve_complete = set(curve) == set(expected_lags) and all(
                curve[lag][0] is not None and curve[lag][1] >= minimum_subjects
                for lag in expected_lags
            )
            strict_peak_lags: list[int] = []
            strict_peak_lag = None
            canonical_peak_lag = None
            if curve_complete:
                curve_values = {lag: float(curve[lag][0]) for lag in expected_lags}
                maximum_rho = max(curve_values.values())
                strict_peak_lags = sorted(
                    lag for lag, value in curve_values.items() if abs(value - maximum_rho) <= 1e-12
                )
                if len(strict_peak_lags) == 1:
                    strict_peak_lag = strict_peak_lags[0]
                tolerance = _finite_float(
                    lag_policy.get("canonical_zero_tolerance"),
                    "canonical zero tolerance",
                )
                if curve_values[0] >= maximum_rho - tolerance:
                    canonical_peak_lag = 0
                else:
                    canonical_peak_lag = min(strict_peak_lags, key=lambda value: (abs(value), value))

            evidence_checks = {
                "rho_subject_count": len(rho_values) >= minimum_subjects,
                "rho_task_video_counts": all(
                    rho_task_counts[task] >= minimum_task_videos for task in TASK_NAMES
                ),
                "direction_subject_count": len(direction_values) >= minimum_direction_subjects,
                "amplitude_subject_count": len(amplitude_values) >= minimum_subjects,
                "amplitude_task_video_counts": all(
                    amplitude_task_counts[task] >= minimum_task_videos for task in TASK_NAMES
                ),
                "lag_curve_subject_count": curve_complete,
                "lag_task_video_counts": all(
                    lag_task_counts[task] >= minimum_task_videos for task in TASK_NAMES
                ),
            }
            evidence_pass = all(evidence_checks.values())
            metric_checks = {
                "rho_subject_median": rho_median is not None
                and rho_median >= _finite_float(
                    spearman.get("minimum_subject_median"),
                    "minimum subject rho",
                ),
                "rho_bootstrap_lower": bootstrap_low is not None
                and bootstrap_low
                > _finite_float(
                    spearman.get("minimum_bootstrap_lower_bound"),
                    "minimum bootstrap lower bound",
                ),
                "direction_subject_median": direction_median is not None
                and direction_median
                >= _finite_float(
                    direction.get("minimum_subject_median_rate"),
                    "minimum direction median",
                ),
                "amplitude_subject_median": amplitude_median is not None
                and _finite_float(amplitude.get("minimum_median_ratio"), "minimum amplitude ratio")
                <= amplitude_median
                <= _finite_float(amplitude.get("maximum_median_ratio"), "maximum amplitude ratio"),
                "canonical_peak_lag": canonical_peak_lag
                == _exact_int(lag_policy.get("required_canonical_peak_lag"), "required peak lag"),
            }
            if not evidence_pass:
                status = contract["au_evidence_blocked_status"]
                reasons = [name for name, passed in evidence_checks.items() if not passed]
            elif not all(metric_checks.values()):
                status = contract["au_metric_fail_status"]
                reasons = [name for name, passed in metric_checks.items() if not passed]
            else:
                status = contract["au_pass_status"]
                reasons = []
            decisions[au] = {
                "status": status,
                "reasons": reasons,
                "evidence_checks": evidence_checks,
                "metric_checks": metric_checks,
                "observed": {
                    "rho_subject_count": len(rho_values),
                    "rho_subject_median": rho_median,
                    "rho_bootstrap_ci_lower": bootstrap_low,
                    "rho_bootstrap_ci_upper": bootstrap_high,
                    "rho_task_video_counts": rho_task_counts,
                    "direction_subject_count": len(direction_values),
                    "direction_subject_median": direction_median,
                    "amplitude_subject_count": len(amplitude_values),
                    "amplitude_subject_median": amplitude_median,
                    "amplitude_task_video_counts": amplitude_task_counts,
                    "lag_task_video_counts": lag_task_counts,
                    "strict_peak_lag": strict_peak_lag,
                    "strict_peak_lags": strict_peak_lags,
                    "canonical_peak_lag": canonical_peak_lag,
                    "lag_curve": {
                        str(lag): {
                            "rho": curve.get(lag, (None, 0))[0],
                            "subject_count": curve.get(lag, (None, 0))[1],
                        }
                        for lag in expected_lags
                    },
                },
            }

        statuses = [decisions[au]["status"] for au in CORE_AU_COLUMNS]
        if all(status == contract["au_pass_status"] for status in statuses):
            group_status = contract["group_eligible_status"]
        elif any(status == contract["au_metric_fail_status"] for status in statuses):
            group_status = contract["group_metric_fail_status"]
        else:
            group_status = contract["group_evidence_blocked_status"]
        audit_status = contract["audit_pass_status"]

    group_eligible = group_status == contract["group_eligible_status"]
    split_reports: dict[str, dict[str, Any]] = {}
    for physical_split in PHYSICAL_SPLITS:
        split_reports[physical_split] = {}
        for au in CORE_AU_COLUMNS:
            rows = [
                row
                for row in subject_rows
                if row.get("split") == physical_split and row.get("au") == au
            ]
            split_reports[physical_split][au] = {
                "decision_role": "ELIGIBILITY" if physical_split == "train" else "REPORT_ONLY",
                "subject_count": len(rows),
                "rho_subject_count": sum(row.get("rho0") is not None for row in rows),
                "rho_subject_median": _median(row.get("rho0") for row in rows),
                "direction_subject_count": sum(
                    row.get("direction_rate") is not None for row in rows
                ),
                "direction_subject_median": _median(
                    row.get("direction_rate") for row in rows
                ),
                "amplitude_subject_count": sum(
                    row.get("amplitude_ratio") is not None for row in rows
                ),
                "amplitude_subject_median": _median(
                    row.get("amplitude_ratio") for row in rows
                ),
            }
    return {
        "audit": "PB-P0C core AU raw/aligned physical fidelity",
        "implementation_package_id": "CODE-20260731-PB-P0C-CORE-AU-FIDELITY-v1",
        "audit_status": audit_status,
        "eligibility_split": "train",
        "validation_test_policy": "REPORT_ONLY_NO_THRESHOLD_CALLBACK",
        "core_au_group": list(CORE_AU_COLUMNS),
        "au_decisions": decisions,
        "group_decision": {
            "group": "core",
            "au_members": list(CORE_AU_COLUMNS),
            "au_statuses": {au: decisions[au]["status"] for au in CORE_AU_COLUMNS},
            "status": group_status,
            "eligible": group_eligible,
        },
        "eligibility_status": group_status,
        "cross_talk_decision_role": "REPORT_ONLY",
        "split_reports": split_reports,
        "technical_issue_count": len(technical_issues),
        "training_authorized": False,
        "p0d_authorized": False,
        "p0e_authorized": False,
        "next_action": (
            contract["pass_next_action"] if group_eligible else contract["stop_next_action"]
        ),
    }


COVERAGE_FIELDS = (
    "split",
    "subject_id",
    "video_id",
    "task_name",
    "au",
    "total_frame_count",
    "raw_quality_valid_count",
    "aligned_quality_valid_count",
    "matched_valid_count",
    "consecutive_matched_pair_count",
    "raw_only_quality_valid_count",
    "aligned_only_quality_valid_count",
    "status",
    "issues",
)

VIDEO_METRIC_FIELDS = (
    "split",
    "subject_id",
    "video_id",
    "task_name",
    "au",
    "matched_valid_count",
    "rho_pair_count",
    "rho0",
    "rho_estimable",
    "direction_event_count",
    "direction_match_count",
    "direction_rate",
    "direction_estimable",
    "aligned_only_event_rate",
    "raw_amplitude_q95_q05",
    "aligned_amplitude_q95_q05",
    "amplitude_ratio",
    "amplitude_estimable",
    "strict_peak_lag",
    "strict_peak_lags",
    "lag_curve_estimable",
    "status",
    "issues",
)

SUBJECT_METRIC_FIELDS = (
    "split",
    "subject_id",
    "au",
    "task_count",
    "tasks",
    "rho_task_count",
    "rho0",
    "direction_task_count",
    "direction_rate",
    "amplitude_task_count",
    "amplitude_ratio",
    "lag_task_count",
    "single_task_only",
    "single_task_metrics",
    "status",
)

LAG_FIELDS = (
    "split",
    "aggregation",
    "subject_id",
    "video_id",
    "task_name",
    "au",
    "lag_frames",
    "rho",
    "pair_count",
    "video_count",
    "subject_count",
)

CROSS_TALK_FIELDS = (
    "split",
    "aggregation",
    "subject_id",
    "video_id",
    "task_name",
    "matrix",
    "source_au",
    "target_au",
    "rho",
    "pair_count",
    "video_count",
    "subject_count",
    "diagonal_rho",
    "maximum_off_diagonal_absolute_rho",
    "diagonal_margin",
    "decision_role",
)

GROUP_DECISION_FIELDS = (
    "group",
    "au_members",
    "au_statuses",
    "eligibility_status",
    "eligible",
    "reasons",
)

ISSUE_FIELDS = (
    "severity",
    "code",
    "split",
    "video_id",
    "task_name",
    "details",
)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _new_output_dir(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"output directory already exists; refusing overwrite/resume: {resolved}")
    resolved.mkdir(parents=True)
    (resolved / "tables").mkdir()
    (resolved / "reports").mkdir()
    return resolved


def _path_is_within(candidate: Path, root: Path) -> bool:
    return candidate == root or root in candidate.parents


def _paths_overlap(path_a: Path, path_b: Path) -> bool:
    return _path_is_within(path_a, path_b) or _path_is_within(path_b, path_a)


def _validate_audit_path_boundaries(
    *,
    aligned_openface_root: Path,
    aligned_run_manifest: Path,
    raw_openface_root: Path,
    raw_run_manifest: Path,
    output_dir: Path,
) -> None:
    if _paths_overlap(raw_openface_root, aligned_openface_root):
        raise AuFidelityError("raw and aligned OpenFace roots must be distinct and non-overlapping")
    _require_equal(
        aligned_run_manifest,
        aligned_openface_root / "_audit" / "run_manifest.json",
        "aligned run-manifest location",
    )
    _require_equal(
        raw_run_manifest,
        raw_openface_root / "_audit" / "run_manifest.json",
        "raw run-manifest location",
    )
    if _paths_overlap(output_dir, raw_openface_root) or _paths_overlap(
        output_dir, aligned_openface_root
    ):
        raise AuFidelityError("audit output must not overlap either OpenFace input root")


def _git_value(project_root: Path, arguments: Sequence[str]) -> tuple[str, bool]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "", False
    return completed.stdout.rstrip(), completed.returncode == 0


def _issue(
    code: str,
    details: str,
    *,
    physical_split: str = "",
    video_id: str = "",
    task_name: str = "",
) -> dict[str, Any]:
    return {
        "severity": "BLOCKER",
        "code": code,
        "split": physical_split,
        "video_id": video_id,
        "task_name": task_name,
        "details": details,
    }


def _build_markdown_report(
    decision: Mapping[str, Any],
    coverage_rows: Sequence[Mapping[str, Any]],
    video_rows: Sequence[Mapping[str, Any]],
    issues: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# PB-P0C Core AU Physical Fidelity",
        "",
        f"- Audit status: `{decision['audit_status']}`",
        f"- Core eligibility: `{decision['eligibility_status']}`",
        f"- Technical blockers: {len(issues)}",
        f"- Audited coverage rows: {len(coverage_rows)}",
        f"- Video metric rows: {len(video_rows)}",
        "- Eligibility split: physical `train` only; `val/test` are report-only.",
        "- Cross-talk: report-only with no threshold in v1.",
        "- Training/P0D/P0E authorization: `false/false/false`.",
        "",
        "## AU decisions",
        "",
        "| AU | Status | Rho median | Bootstrap lower | Direction | Amplitude | Canonical lag |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for au in CORE_AU_COLUMNS:
        au_decision = decision["au_decisions"][au]
        observed = au_decision.get("observed", {})

        def render(value: Any) -> str:
            return "" if value is None else f"{float(value):.6f}"

        lines.append(
            f"| {au} | {au_decision['status']} | "
            f"{render(observed.get('rho_subject_median'))} | "
            f"{render(observed.get('rho_bootstrap_ci_lower'))} | "
            f"{render(observed.get('direction_subject_median'))} | "
            f"{render(observed.get('amplitude_subject_median'))} | "
            f"{'' if observed.get('canonical_peak_lag') is None else observed.get('canonical_peak_lag')} |"
        )
    lines.extend(["", "## Stop conditions", ""])
    if issues:
        for row in issues[:20]:
            lines.append(f"- `{row['code']}`: {row['details']}")
        if len(issues) > 20:
            lines.append(f"- ... {len(issues) - 20} additional blockers are in the issue table.")
    else:
        lines.append("- No technical source, provenance, schema, hash, or exact-join blocker was observed.")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This audit measures raw/aligned OpenFace representation stability. It does not prove AU ground-truth accuracy, depression relevance, model utility, or training authorization.",
            "",
        ]
    )
    return "\n".join(lines)


def run_privileged_behavior_au_fidelity(
    *,
    policy_path: Path,
    policy_sha256: str,
    dataset_split_file: Path,
    source_video_contract: Path,
    aligned_openface_root: Path,
    aligned_run_manifest: Path,
    raw_openface_root: Path,
    raw_run_manifest: Path,
    raw_run_manifest_sha256: str,
    p0b_run_manifest: Path,
    p0b_selected_target_manifest: Path,
    output_dir: Path,
    project_root: Path | None = None,
    command_line: str | None = None,
) -> list[Path]:
    """Run one immutable P0C audit and return all generated artifact paths."""

    policy_path = Path(policy_path).expanduser().resolve()
    dataset_split_file = Path(dataset_split_file).expanduser().resolve()
    source_video_contract = Path(source_video_contract).expanduser().resolve()
    aligned_openface_root = Path(aligned_openface_root).expanduser().resolve()
    aligned_run_manifest = Path(aligned_run_manifest).expanduser().resolve()
    raw_openface_root = Path(raw_openface_root).expanduser().resolve()
    raw_run_manifest = Path(raw_run_manifest).expanduser().resolve()
    p0b_run_manifest = Path(p0b_run_manifest).expanduser().resolve()
    p0b_selected_target_manifest = Path(p0b_selected_target_manifest).expanduser().resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    policy_sha256 = _require_hash(policy_sha256, "expected policy hash")
    raw_run_manifest_sha256 = _require_hash(
        raw_run_manifest_sha256,
        "expected raw run manifest hash",
    )
    output_dir = Path(output_dir).expanduser().resolve()
    _validate_audit_path_boundaries(
        aligned_openface_root=aligned_openface_root,
        aligned_run_manifest=aligned_run_manifest,
        raw_openface_root=raw_openface_root,
        raw_run_manifest=raw_run_manifest,
        output_dir=output_dir,
    )
    policy = load_au_fidelity_policy(policy_path)
    output_dir = _new_output_dir(output_dir)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    run_manifest_path = output_dir / "run_manifest.json"
    _write_json(
        run_manifest_path,
        {
            "audit": "PB-P0C core AU raw/aligned physical fidelity run",
            "implementation_package_id": "CODE-20260731-PB-P0C-CORE-AU-FIDELITY-v1",
            "compatibility_package_id": "CODE-20260805-PB-P0C-MASKAWARE-COMPAT-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "RUNNING",
            "command_line": command_line or " ".join(sys.argv),
            "policy": {
                "path": str(policy_path),
                "expected_sha256": policy_sha256,
            },
            "raw_run_manifest": {
                "path": str(raw_run_manifest),
                "expected_sha256": raw_run_manifest_sha256,
            },
            "output_root_must_be_new": True,
            "training_authorized": False,
            "p0d_authorized": False,
            "p0e_authorized": False,
        },
    )
    source = _mapping(policy.get("source_contract"), "source_contract")

    coverage_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    video_lag_rows: list[dict[str, Any]] = []
    video_cross_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    lag_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    openface_access_counts: Counter[str] = Counter()
    source_access_counts: Counter[str] = Counter()
    aligned_schema_counts: Counter[str] = Counter()
    raw_schema_counts: Counter[str] = Counter()
    processed_video_count = 0
    processed_frame_count = 0
    aligned_status_compatibility: dict[str, Any] = {
        "enabled": False,
        "reason": "input_contract_not_evaluated",
        "aligned_content_status_counts": {},
        "accepted_legacy_fail_count": 0,
    }

    input_hashes: dict[str, str] = {}
    try:
        input_hashes = {
            "policy": _assert_file_hash(
                policy_path,
                policy_sha256,
                "AU fidelity policy",
            ),
            "dataset_split": _assert_file_hash(
                dataset_split_file,
                source.get("required_dataset_split_sha256"),
                "dataset split",
            ),
            "source_video_contract": _assert_file_hash(
                source_video_contract,
                source.get("required_source_video_contract_sha256"),
                "source-video contract",
            ),
            "aligned_run_manifest": _assert_file_hash(
                aligned_run_manifest,
                source.get("required_aligned_run_manifest_sha256"),
                "aligned run manifest",
            ),
            "raw_run_manifest": _assert_file_hash(
                raw_run_manifest,
                raw_run_manifest_sha256,
                "raw run manifest",
            ),
            "p0b_run_manifest": _assert_file_hash(
                p0b_run_manifest,
                source.get("required_p0b_run_manifest_sha256"),
                "P0B run manifest",
            ),
            "p0b_selected_target_manifest": _assert_file_hash(
                p0b_selected_target_manifest,
                source.get("required_p0b_selected_target_manifest_sha256"),
                "P0B selected-target manifest",
            ),
        }
        split_by_video, videos_by_split = _read_exact_split(
            dataset_split_file,
            _exact_int(source.get("expected_video_count"), "expected video count"),
        )
        source_records = _read_source_contract(
            source_video_contract,
            split_by_video,
            _exact_int(source.get("expected_frame_count"), "expected frame count"),
            source_access_counts,
        )
        aligned_manifest = _read_json(aligned_run_manifest, "aligned run manifest")
        raw_manifest = _read_json(raw_run_manifest, "raw run manifest")
        p0b_run = _read_json(p0b_run_manifest, "P0B run manifest")
        p0b_selected = _read_json(
            p0b_selected_target_manifest,
            "P0B selected-target manifest",
        )
        aligned_content, raw_content, aligned_status_compatibility = _validate_input_manifests(
            policy=policy,
            aligned_manifest=aligned_manifest,
            raw_manifest=raw_manifest,
            raw_openface_root=raw_openface_root,
            aligned_openface_root=aligned_openface_root,
            p0b_run=p0b_run,
            p0b_selected=p0b_selected,
            expected_video_ids=split_by_video,
            raw_extractor_script=(
                project_root
                / "scripts"
                / "run_openface_raw_au_fidelity_reference.ps1"
            ),
        )

        value_minimum = _finite_float(source.get("au_value_minimum"), "AU minimum")
        value_maximum = _finite_float(source.get("au_value_maximum"), "AU maximum")
        for physical_split in PHYSICAL_SPLITS:
            for video_id in videos_by_split[physical_split]:
                task_name = _task_from_video_id(video_id)
                source_record = source_records[video_id]
                expected_frames = int(source_record["frame_count"])
                raw_video_id = str(source_record["raw_video_id"])
                aligned_csv = aligned_openface_root / f"{video_id}.csv"
                raw_csv = raw_openface_root / f"{raw_video_id}.csv"
                try:
                    aligned_status = str(aligned_content[video_id].get("status"))
                    aligned_status_accepted = aligned_status == "PASS" or (
                        aligned_status == "FAIL"
                        and aligned_status_compatibility.get("enabled") is True
                    )
                    if not aligned_status_accepted:
                        raise AuFidelityError(
                            f"aligned content manifest status is not accepted for {video_id}: "
                            f"{aligned_status}"
                        )
                    if str(raw_content[video_id].get("status")) != "PASS":
                        raise AuFidelityError(
                            f"raw content manifest status is not structural PASS for {video_id}: "
                            f"{raw_content[video_id].get('status')}"
                        )
                    raw_sequence = _read_openface_sequence(
                        path=raw_csv,
                        expected_frames=expected_frames,
                        content_record=raw_content[video_id],
                        au_columns=CORE_AU_COLUMNS,
                        value_minimum=value_minimum,
                        value_maximum=value_maximum,
                        source_name="raw",
                        access_counts=openface_access_counts,
                    )
                    aligned_sequence = _read_openface_sequence(
                        path=aligned_csv,
                        expected_frames=expected_frames,
                        content_record=aligned_content[video_id],
                        au_columns=CORE_AU_COLUMNS,
                        value_minimum=value_minimum,
                        value_maximum=value_maximum,
                        source_name="aligned",
                        access_counts=openface_access_counts,
                    )
                    current_coverage, current_metrics, current_lags, current_cross = (
                        compute_video_au_fidelity(
                            raw=raw_sequence,
                            aligned=aligned_sequence,
                            policy=policy,
                            video_id=video_id,
                            physical_split=physical_split,
                            task_name=task_name,
                        )
                    )
                    coverage_rows.extend(current_coverage)
                    video_rows.extend(current_metrics)
                    video_lag_rows.extend(current_lags)
                    video_cross_rows.extend(current_cross)
                    raw_schema_counts[raw_sequence.schema_sha256] += 1
                    aligned_schema_counts[aligned_sequence.schema_sha256] += 1
                    processed_video_count += 1
                    processed_frame_count += expected_frames
                except AuFidelityError as exc:
                    issues.append(
                        _issue(
                            "video_pair_audit_failed",
                            str(exc),
                            physical_split=physical_split,
                            video_id=video_id,
                            task_name=task_name,
                        )
                    )
        expected_video_count = _exact_int(source.get("expected_video_count"), "expected video count")
        expected_frame_count = _exact_int(source.get("expected_frame_count"), "expected frame count")
        if processed_video_count != expected_video_count or processed_frame_count != expected_frame_count:
            issues.append(
                _issue(
                    "incomplete_pair_audit",
                    f"processed videos/frames={processed_video_count}/{processed_frame_count}; "
                    f"expected={expected_video_count}/{expected_frame_count}",
                )
            )
        if len(raw_schema_counts) != 1 or len(aligned_schema_counts) != 1:
            issues.append(
                _issue(
                    "schema_inconsistency",
                    f"raw schemas={dict(raw_schema_counts)} aligned schemas={dict(aligned_schema_counts)}",
                )
            )
        subject_rows, lag_rows, cross_rows = aggregate_au_fidelity_metrics(
            video_rows,
            video_lag_rows,
            video_cross_rows,
        )
    except AuFidelityError as exc:
        issues.append(_issue("input_contract_failed", str(exc)))

    decision = build_au_fidelity_decision(
        policy=policy,
        video_rows=video_rows,
        subject_rows=subject_rows,
        lag_rows=lag_rows,
        technical_issues=issues,
    )
    decision.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "policy": {
                "path": str(policy_path),
                "sha256": input_hashes.get("policy", _sha256_file(policy_path)),
                "expected_sha256": policy_sha256,
                "policy_id": policy["policy_id"],
                "policy_version": policy["policy_version"],
            },
            "counts": {
                "processed_video_count": processed_video_count,
                "processed_frame_count": processed_frame_count,
                "coverage_row_count": len(coverage_rows),
                "video_metric_row_count": len(video_rows),
                "subject_metric_row_count": len(subject_rows),
            },
        }
    )

    coverage_path = tables_dir / "au_fidelity_coverage.csv"
    video_path = tables_dir / "au_fidelity_video_metrics.csv"
    subject_path = tables_dir / "au_fidelity_subject_metrics.csv"
    lag_path = tables_dir / "au_fidelity_lag_curve.csv"
    cross_path = tables_dir / "au_fidelity_cross_talk.csv"
    group_path = tables_dir / "au_fidelity_group_decision.csv"
    issues_path = tables_dir / "au_fidelity_issues.csv"
    decision_path = output_dir / "au_fidelity_decision.json"
    report_path = reports_dir / "au_fidelity_report.md"

    group = decision["group_decision"]
    group_reasons = sorted(
        {
            reason
            for au_decision in decision["au_decisions"].values()
            for reason in au_decision.get("reasons", [])
        }
    )
    group_rows = [
        {
            "group": group["group"],
            "au_members": ";".join(group["au_members"]),
            "au_statuses": ";".join(
                f"{au}={status}" for au, status in group["au_statuses"].items()
            ),
            "eligibility_status": group["status"],
            "eligible": group["eligible"],
            "reasons": ";".join(group_reasons),
        }
    ]
    _write_csv(coverage_path, COVERAGE_FIELDS, coverage_rows)
    _write_csv(video_path, VIDEO_METRIC_FIELDS, video_rows)
    _write_csv(subject_path, SUBJECT_METRIC_FIELDS, subject_rows)
    _write_csv(lag_path, LAG_FIELDS, lag_rows)
    _write_csv(cross_path, CROSS_TALK_FIELDS, cross_rows)
    _write_csv(group_path, GROUP_DECISION_FIELDS, group_rows)
    _write_csv(issues_path, ISSUE_FIELDS, issues)
    _write_json(decision_path, decision)
    report_path.write_text(
        _build_markdown_report(decision, coverage_rows, video_rows, issues),
        encoding="utf-8",
    )

    commit, commit_available = _git_value(project_root, ("rev-parse", "HEAD"))
    branch, branch_available = _git_value(project_root, ("branch", "--show-current"))
    status_short, status_available = _git_value(project_root, ("status", "--short"))
    output_paths = (
        coverage_path,
        video_path,
        subject_path,
        lag_path,
        cross_path,
        group_path,
        issues_path,
        decision_path,
        report_path,
    )
    implementation_files = {
        "audit_core": Path(__file__).resolve(),
        "cli": project_root / "scripts" / "audit_privileged_behavior_au_fidelity.py",
        "raw_reference_extractor": project_root
        / "scripts"
        / "run_openface_raw_au_fidelity_reference.ps1",
        "policy": policy_path,
    }
    run_manifest = {
        "audit": "PB-P0C core AU raw/aligned physical fidelity run",
        "implementation_package_id": "CODE-20260731-PB-P0C-CORE-AU-FIDELITY-v1",
        "compatibility_package_id": "CODE-20260805-PB-P0C-MASKAWARE-COMPAT-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": decision["audit_status"],
        "eligibility_status": decision["eligibility_status"],
        "command_line": command_line or " ".join(sys.argv),
        "git_commit": commit,
        "git_commit_available": commit_available,
        "git_branch": branch,
        "git_branch_available": branch_available,
        "git_status_short": status_short,
        "git_status_available": status_available,
        "inputs": {
            "policy": {
                "path": str(policy_path),
                "sha256": input_hashes.get("policy"),
                "expected_sha256": policy_sha256,
            },
            "dataset_split": {
                "path": str(dataset_split_file),
                "sha256": input_hashes.get("dataset_split"),
            },
            "source_video_contract": {
                "path": str(source_video_contract),
                "sha256": input_hashes.get("source_video_contract"),
                "accessed_columns": sorted(source_access_counts),
                "raw_video_path_access_count": 0,
                "raw_openface_csv_path_access_count": 0,
            },
            "aligned_openface_root": str(aligned_openface_root),
            "aligned_run_manifest": {
                "path": str(aligned_run_manifest),
                "sha256": input_hashes.get("aligned_run_manifest"),
            },
            "raw_openface_root": str(raw_openface_root),
            "raw_run_manifest": {
                "path": str(raw_run_manifest),
                "sha256": input_hashes.get("raw_run_manifest"),
                "expected_sha256": raw_run_manifest_sha256,
            },
            "p0b_run_manifest": {
                "path": str(p0b_run_manifest),
                "sha256": input_hashes.get("p0b_run_manifest"),
            },
            "p0b_selected_target_manifest": {
                "path": str(p0b_selected_target_manifest),
                "sha256": input_hashes.get("p0b_selected_target_manifest"),
            },
        },
        "access_contract": {
            "selected_value_columns": list((*QUALITY_COLUMNS, *CORE_AU_COLUMNS)),
            "openface_value_access_counts": dict(sorted(openface_access_counts.items())),
            "pose_value_access_count": 0,
            "gaze_value_access_count": 0,
            "extension_au_value_access_count": 0,
            "label_access_count": 0,
            "prediction_access_count": 0,
            "checkpoint_access_count": 0,
            "validation_test_eligibility_callback_count": 0,
            "count_semantics": "selected value projection/use counts; CSV tokenization is not value access",
        },
        "aligned_mask_aware_compatibility": aligned_status_compatibility,
        "counts": decision["counts"],
        "schema_sha256_video_counts": {
            "raw": dict(raw_schema_counts),
            "aligned": dict(aligned_schema_counts),
        },
        "implementation_files": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path) if path.is_file() else None,
            }
            for name, path in implementation_files.items()
        },
        "outputs": {
            path.relative_to(output_dir).as_posix(): {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for path in output_paths
        },
        "read_only_inputs": True,
        "source_data_modified": False,
        "model_modified": False,
        "training_started": False,
        "device": "cpu",
        "precision": "float64_metrics_no_amp",
        "python": sys.version,
        "platform": platform.platform(),
        "training_authorized": False,
        "p0d_authorized": False,
        "p0e_authorized": False,
    }
    _write_json(run_manifest_path, run_manifest)
    return [*output_paths, run_manifest_path]
