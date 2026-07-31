"""Label-blind PB-P0A2 mask-aware source/coverage policy validation.

The extractor's historical per-video ``success_ratio >= 0.995`` result is
retained as evidence, but it is not used as the mask-aware decision.  Source
completeness (video/row/schema/hash/provenance) remains hard-fail.  Coverage is
recomputed from the immutable CSV rows using ``success`` and ``confidence``;
validation/test coverage is report-only and never feeds a model or threshold.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VIDEO_FIELDS = (
    "physical_split",
    "video_id",
    "task_name",
    "frame_count",
    "quality_valid_count",
    "quality_valid_ratio",
    "au_valid_count",
    "au_valid_ratio",
    "pose_valid_count",
    "pose_valid_ratio",
    "head_candidate_pair_count",
    "head_valid_pair_count",
    "head_valid_pair_ratio",
    "legacy_success_ratio",
    "legacy_status",
    "policy_enforcement",
    "status",
    "issues",
)

AGGREGATE_FIELDS = (
    "scope",
    "physical_split",
    "task_name",
    "video_count",
    "frame_count",
    "quality_valid_count",
    "quality_valid_ratio",
    "au_valid_count",
    "au_valid_ratio",
    "head_candidate_pair_count",
    "head_valid_pair_count",
    "head_valid_pair_ratio",
    "policy_enforcement",
    "status",
    "issues",
)

ISSUE_FIELDS = ("severity", "scope", "video_id", "issue_code", "detail")

FULL_PASS_NEXT_ACTION = "REQUEST_SEPARATE_P0B_AUTHORIZATION"
BLOCKED_NEXT_ACTION = "STOP_AND_REVIEW_POLICY_OR_SOURCE_ISSUES"


class CoveragePolicyError(ValueError):
    """Raised when a policy or immutable source contract is unreadable."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoveragePolicyError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CoveragePolicyError(f"{label} must be a JSON object: {path}")
    return value


def _read_csv(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise CoveragePolicyError(f"{label} has no header: {path}")
            normalized = [str(field).strip() for field in reader.fieldnames]
            if len(set(normalized)) != len(normalized):
                raise CoveragePolicyError(f"{label} has duplicate normalized columns: {path}")
            rows = []
            for raw in reader:
                row = {
                    str(key).strip(): "" if value is None else str(value).strip()
                    for key, value in raw.items()
                }
                rows.append(row)
            return rows
    except OSError as exc:
        raise CoveragePolicyError(f"cannot read {label}: {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.12g}" if math.isfinite(value) else ""
    return str(value)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field)) for field in fieldnames})
    return path


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise CoveragePolicyError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise CoveragePolicyError(f"{label} must be finite")
    return parsed


def _value_is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise CoveragePolicyError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CoveragePolicyError(f"{label} must be an integer") from exc
    if parsed < 0:
        raise CoveragePolicyError(f"{label} must be non-negative")
    return parsed


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _issue(scope: str, code: str, detail: str, video_id: str = "") -> dict[str, str]:
    return {
        "severity": "BLOCKING",
        "scope": scope,
        "video_id": video_id,
        "issue_code": code,
        "detail": detail,
    }


def _warning(scope: str, code: str, detail: str) -> dict[str, str]:
    return {
        "severity": "WARNING",
        "scope": scope,
        "video_id": "",
        "issue_code": code,
        "detail": detail,
    }


def _require_keys(value: dict[str, Any], keys: Iterable[str], label: str) -> None:
    missing = sorted(set(keys) - set(value))
    if missing:
        raise CoveragePolicyError(f"{label} is missing keys: {missing}")


def _decision_next_action(policy: dict[str, Any], status: str) -> str:
    contract = policy["decision_contract"]
    if status == contract["pilot_pass_status"]:
        return contract["pilot_pass_next_action"]
    if status == contract["full_pass_status"]:
        return contract.get("full_pass_next_action", FULL_PASS_NEXT_ACTION)
    return BLOCKED_NEXT_ACTION


def load_coverage_policy(path: Path) -> dict[str, Any]:
    policy = _read_json_object(path, "coverage policy")
    _require_keys(
        policy,
        (
            "policy_id",
            "policy_version",
            "status",
            "label_blind",
            "source_contract",
            "pilot_evidence",
            "full_source_expectations",
            "pilot_feasibility_thresholds",
            "physical_train_full_thresholds",
            "validation_test_policy",
            "decision_contract",
        ),
        "coverage policy",
    )
    if policy["status"] != "FROZEN" or policy["label_blind"] is not True:
        raise CoveragePolicyError("coverage policy must be FROZEN and label_blind=true")
    if not isinstance(policy["source_contract"], dict):
        raise CoveragePolicyError("source_contract must be an object")
    source = policy["source_contract"]
    _require_keys(
        source,
        (
            "feature_profile",
            "confidence_threshold",
            "legacy_strict_success_ratio",
            "legacy_strict_status_is_diagnostic_only",
            "expected_csv_column_count",
            "required_au_columns",
            "required_pose_columns",
            "required_source_video_contract_sha256",
        ),
        "source_contract",
    )
    confidence = _finite_float(source["confidence_threshold"], "confidence_threshold")
    if not 0.0 <= confidence <= 1.0:
        raise CoveragePolicyError("confidence_threshold must be within [0, 1]")
    if source["legacy_strict_status_is_diagnostic_only"] is not True:
        raise CoveragePolicyError("v1 mask-aware policy must retain legacy strict status as diagnostic only")
    for group in ("required_au_columns", "required_pose_columns"):
        columns = source[group]
        if not isinstance(columns, list) or not columns or any(not str(item) for item in columns):
            raise CoveragePolicyError(f"{group} must be a non-empty list")
        if len(set(columns)) != len(columns):
            raise CoveragePolicyError(f"{group} must not contain duplicates")
    for section_name in ("pilot_feasibility_thresholds", "physical_train_full_thresholds"):
        section = policy[section_name]
        if not isinstance(section, dict):
            raise CoveragePolicyError(f"{section_name} must be an object")
        for key, value in section.items():
            if key.startswith("minimum_") or key.startswith("maximum_"):
                number = _finite_float(value, f"{section_name}.{key}")
                if number < 0.0:
                    raise CoveragePolicyError(f"{section_name}.{key} must be non-negative")
    decision = policy["decision_contract"]
    if not isinstance(decision, dict):
        raise CoveragePolicyError("decision_contract must be an object")
    _require_keys(
        decision,
        (
            "pilot_pass_status",
            "full_pass_status",
            "blocked_status",
            "full_rich_authorized",
            "p0b_authorized",
            "training_authorized",
            "pilot_pass_next_action",
        ),
        "decision_contract",
    )
    if any(
        decision[field] is not False
        for field in ("full_rich_authorized", "p0b_authorized", "training_authorized")
    ):
        raise CoveragePolicyError("decision_contract authorization flags must remain false")
    for field in (
        "pilot_pass_status",
        "full_pass_status",
        "blocked_status",
        "pilot_pass_next_action",
    ):
        if not isinstance(decision[field], str) or not decision[field].strip():
            raise CoveragePolicyError(f"decision_contract.{field} must be a non-empty string")
    if "full_pass_next_action" in decision and (
        not isinstance(decision["full_pass_next_action"], str)
        or not decision["full_pass_next_action"].strip()
    ):
        raise CoveragePolicyError(
            "decision_contract.full_pass_next_action must be a non-empty string when present"
        )
    return policy


def _read_split_map(path: Path) -> dict[str, str]:
    payload = _read_json_object(path, "dataset split")
    mapping: dict[str, str] = {}
    for split in ("train", "val", "test"):
        videos = payload.get(split)
        if not isinstance(videos, list):
            raise CoveragePolicyError(f"dataset split {split} must be a list")
        for raw in videos:
            video_id = str(raw).strip()
            if not video_id or video_id in mapping:
                raise CoveragePolicyError(f"duplicate or empty video id in dataset split: {video_id!r}")
            mapping[video_id] = split
    return mapping


def _task_name(video_id: str) -> str:
    for task in ("Freeform", "Northwind"):
        if f"_{task}_" in video_id:
            return task
    raise CoveragePolicyError(f"cannot infer task from video id: {video_id}")


def _safe_output_dir(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory must be empty or absent: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_source_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true"}


def _validate_pinned_hash(
    issues: list[dict[str, str]], path: Path, expected: str, code: str
) -> None:
    observed = _sha256_file(path) if path.is_file() else "MISSING"
    if observed != expected:
        issues.append(_issue("pilot_evidence", code, f"expected={expected} observed={observed}"))


def _scan_behavior_csv(
    path: Path,
    *,
    confidence_threshold: float,
    required_au_columns: list[str],
    required_pose_columns: list[str],
    expected_column_count: int,
) -> dict[str, Any]:
    rows = _read_csv(path, "behavior CSV")
    if not rows:
        raise CoveragePolicyError(f"behavior CSV is empty: {path}")
    columns = list(rows[0])
    required = {"frame", "confidence", "success", *required_au_columns, *required_pose_columns}
    missing = sorted(required - set(columns))
    if missing:
        raise CoveragePolicyError(f"behavior CSV missing required columns {missing}: {path}")
    if len(columns) != expected_column_count:
        raise CoveragePolicyError(
            f"behavior CSV column count mismatch: expected={expected_column_count} observed={len(columns)}: {path}"
        )

    quality_valid = []
    au_valid = []
    pose_valid = []
    success_count = 0
    frame_ids = []
    for index, row in enumerate(rows, start=1):
        frame = _nonnegative_int(row["frame"], f"{path.name}.frame")
        if frame != index:
            raise CoveragePolicyError(
                f"behavior CSV frame sequence must be 1..N: {path}: row={index} frame={frame}"
            )
        confidence = _finite_float(row["confidence"], f"{path.name}.confidence")
        success_value = _finite_float(row["success"], f"{path.name}.success")
        if success_value not in (0.0, 1.0):
            raise CoveragePolicyError(f"success must be 0 or 1: {path}: frame={frame}")
        quality = success_value == 1.0 and confidence >= confidence_threshold
        success_count += int(success_value == 1.0)
        au_finite = all(_value_is_finite(row[column]) for column in required_au_columns)
        pose_finite = all(_value_is_finite(row[column]) for column in required_pose_columns)
        frame_ids.append(frame)
        quality_valid.append(quality)
        au_valid.append(quality and au_finite)
        pose_valid.append(quality and pose_finite)

    head_candidate = max(len(rows) - 1, 0)
    head_valid = sum(
        frame_ids[index] == frame_ids[index - 1] + 1
        and pose_valid[index - 1]
        and pose_valid[index]
        for index in range(1, len(rows))
    )
    return {
        "frame_count": len(rows),
        "success_count": success_count,
        "quality_valid_count": sum(quality_valid),
        "au_valid_count": sum(au_valid),
        "pose_valid_count": sum(pose_valid),
        "head_candidate_pair_count": head_candidate,
        "head_valid_pair_count": head_valid,
    }


def _aggregate(rows: list[dict[str, Any]], scope: str, split: str, task: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if (split == "ALL" or row["physical_split"] == split)
        and (task == "ALL" or row["task_name"] == task)
    ]
    totals = {
        key: sum(int(row[key]) for row in selected)
        for key in (
            "frame_count",
            "quality_valid_count",
            "au_valid_count",
            "head_candidate_pair_count",
            "head_valid_pair_count",
        )
    }
    return {
        "scope": scope,
        "physical_split": split,
        "task_name": task,
        "video_count": len(selected),
        **totals,
        "quality_valid_ratio": _ratio(totals["quality_valid_count"], totals["frame_count"]),
        "au_valid_ratio": _ratio(totals["au_valid_count"], totals["frame_count"]),
        "head_valid_pair_ratio": _ratio(
            totals["head_valid_pair_count"], totals["head_candidate_pair_count"]
        ),
        "policy_enforcement": "PENDING",
        "status": "REPORT_ONLY",
        "issues": "",
    }


def validate_behavior_source_coverage(
    *,
    policy_path: Path,
    dataset_split_file: Path,
    extraction_root: Path,
    output_dir: Path,
    project_root: Path | None = None,
) -> list[Path]:
    """Validate one immutable rich extraction against the frozen A2 policy."""

    policy_path = Path(policy_path).expanduser().resolve()
    dataset_split_file = Path(dataset_split_file).expanduser().resolve()
    extraction_root = Path(extraction_root).expanduser().resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    policy = load_coverage_policy(policy_path)
    split_map = _read_split_map(dataset_split_file)
    output_dir = _safe_output_dir(output_dir)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"

    audit_dir = extraction_root / "_audit"
    paths = {
        "run_manifest": audit_dir / "run_manifest.json",
        "extraction_summary": audit_dir / "extraction_summary.json",
        "video_run_summary": audit_dir / "video_run_summary.csv",
        "input_frame_contract": audit_dir / "input_frame_contract.csv",
        "csv_content_manifest": audit_dir / "csv_content_manifest.csv",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise CoveragePolicyError(f"missing immutable source artifact {label}: {path}")

    source_manifest = _read_json_object(paths["run_manifest"], "source run manifest")
    summary = _read_json_object(paths["extraction_summary"], "extraction summary")
    video_summary = _read_csv(paths["video_run_summary"], "video run summary")
    input_contract = _read_csv(paths["input_frame_contract"], "input frame contract")
    content_manifest = _read_csv(paths["csv_content_manifest"], "CSV content manifest")

    selected_contract = {
        row["video_id"]: row for row in input_contract if _parse_source_bool(row["selected_for_run"])
    }
    video_rows_by_id = {row["video_id"]: row for row in video_summary}
    content_by_id = {row["video_id"]: row for row in content_manifest}
    issues: list[dict[str, str]] = []

    if len(video_rows_by_id) != len(video_summary):
        issues.append(_issue("source", "duplicate_video_run_summary_id", "video ids must be unique"))
    if len(content_by_id) != len(content_manifest):
        issues.append(_issue("source", "duplicate_csv_content_manifest_id", "video ids must be unique"))
    selected_ids = set(selected_contract)
    if set(video_rows_by_id) != selected_ids or set(content_by_id) != selected_ids:
        issues.append(
            _issue(
                "source",
                "selected_video_set_mismatch",
                f"contract={len(selected_ids)} summary={len(video_rows_by_id)} content={len(content_by_id)}",
            )
        )

    source = policy["source_contract"]
    scope = "pilot" if source_manifest.get("run_scope") == "debug_subset" else "full"
    if source_manifest.get("feature_profile") != source["feature_profile"]:
        issues.append(
            _issue(
                "source",
                "feature_profile_mismatch",
                f"expected={source['feature_profile']} observed={source_manifest.get('feature_profile')}",
            )
        )
    if source.get("require_clean_source_git") and (
        source_manifest.get("git_status_available") is not True
        or str(source_manifest.get("git_status_short", "")) != ""
    ):
        issues.append(_issue("source", "source_git_not_clean", "source git status must be available and empty"))
    if source.get("require_no_gaze_request") and source_manifest.get("gaze_requested") is not False:
        issues.append(_issue("source", "gaze_was_requested", "gaze_requested must be false"))
    for field in ("hog_enabled", "tracked_video_enabled", "aligned_image_generation_enabled"):
        if source_manifest.get(field) is not False:
            issues.append(
                _issue("source", f"forbidden_{field}", f"source manifest {field} must be false")
            )
    observed_source_contract_sha = str(source_manifest.get("source_video_contract_sha256", ""))
    if observed_source_contract_sha != source["required_source_video_contract_sha256"]:
        issues.append(
            _issue(
                "source",
                "source_video_contract_sha256_mismatch",
                f"expected={source['required_source_video_contract_sha256']} observed={observed_source_contract_sha}",
            )
        )

    actual_input_contract_sha = _sha256_file(paths["input_frame_contract"])
    actual_content_manifest_sha = _sha256_file(paths["csv_content_manifest"])
    actual_summary_sha = _sha256_file(paths["extraction_summary"])
    hash_bindings = (
        (
            "source_input_frame_contract_sha256_mismatch",
            source_manifest.get("input_frame_contract_sha256"),
            actual_input_contract_sha,
        ),
        (
            "source_csv_content_manifest_sha256_mismatch",
            source_manifest.get("csv_content_manifest_sha256"),
            actual_content_manifest_sha,
        ),
        (
            "source_extraction_summary_sha256_mismatch",
            source_manifest.get("extraction_summary_sha256"),
            actual_summary_sha,
        ),
        (
            "summary_csv_content_manifest_sha256_mismatch",
            summary.get("csv_content_manifest_sha256"),
            actual_content_manifest_sha,
        ),
    )
    for code, observed, expected in hash_bindings:
        if observed != expected:
            issues.append(
                _issue("source", code, f"expected={expected} observed={observed}")
            )

    summary_contract = {
        "schema_count": summary.get("schema_count") == 1,
        "schema_consistent": summary.get("schema_consistent") is True,
        "hog_file_count": summary.get("hog_file_count") == 0,
        "tracked_video_file_count": summary.get("tracked_video_file_count") == 0,
        "regenerated_aligned_image_count": summary.get("regenerated_aligned_image_count") == 0,
    }
    for field, passed in summary_contract.items():
        if not passed:
            issues.append(
                _issue(
                    "source",
                    f"invalid_summary_{field}",
                    f"observed={summary.get(field)!r}",
                )
            )

    if scope == "pilot":
        evidence = policy["pilot_evidence"]
        for key in ("expected_video_count", "expected_frame_count"):
            _nonnegative_int(evidence[key], f"pilot_evidence.{key}")
        _validate_pinned_hash(
            issues, dataset_split_file, evidence["dataset_split_sha256"], "pilot_split_sha256_mismatch"
        )
        for label in (
            "run_manifest",
            "extraction_summary",
            "video_run_summary",
            "input_frame_contract",
            "csv_content_manifest",
        ):
            _validate_pinned_hash(
                issues,
                paths[label],
                evidence[f"{label}_sha256"],
                f"pilot_{label}_sha256_mismatch",
            )
        if source_manifest.get("git_commit") != evidence["source_git_commit"]:
            issues.append(
                _issue(
                    "pilot_evidence",
                    "pilot_source_git_commit_mismatch",
                    f"expected={evidence['source_git_commit']} observed={source_manifest.get('git_commit')}",
                )
            )
        expected_video_count = int(evidence["expected_video_count"])
        expected_frame_count = int(evidence["expected_frame_count"])
    else:
        expectations = policy["full_source_expectations"]
        expected_video_count = int(expectations["expected_video_count"])
        expected_frame_count = int(expectations["expected_frame_count"])
        if expectations.get("require_exact_dataset_split_video_set") and selected_ids != set(split_map):
            issues.append(
                _issue(
                    "source",
                    "full_source_video_set_mismatch",
                    f"expected={len(split_map)} observed={len(selected_ids)}",
                )
            )

    if len(selected_ids) != expected_video_count:
        issues.append(
            _issue(
                "source",
                "selected_video_count_mismatch",
                f"expected={expected_video_count} observed={len(selected_ids)}",
            )
        )
    selected_frames = sum(_nonnegative_int(row["image_count"], "image_count") for row in selected_contract.values())
    if selected_frames != expected_frame_count:
        issues.append(
            _issue(
                "source",
                "selected_frame_count_mismatch",
                f"expected={expected_frame_count} observed={selected_frames}",
            )
        )
    if int(summary.get("video_count", -1)) != len(selected_ids):
        issues.append(_issue("source", "summary_video_count_mismatch", "summary video count disagrees"))
    if int(summary.get("total_images", -1)) != selected_frames or int(
        summary.get("total_csv_rows", -1)
    ) != selected_frames:
        issues.append(_issue("source", "summary_frame_or_row_count_mismatch", "summary counts disagree"))

    for video_id, row in selected_contract.items():
        if row.get("status") != "PASS" or row.get("issues", ""):
            issues.append(
                _issue(
                    "source",
                    "invalid_selected_input_frame_contract_row",
                    f"status={row.get('status')} issues={row.get('issues')}",
                    video_id,
                )
            )

    video_policy_rows: list[dict[str, Any]] = []
    expected_column_count = int(source["expected_csv_column_count"])
    total_success_count = 0
    observed_schema_hashes: set[str] = set()
    for video_id in sorted(selected_ids):
        if video_id not in split_map:
            issues.append(_issue("source", "video_missing_from_dataset_split", video_id, video_id))
            continue
        summary_row = video_rows_by_id.get(video_id)
        content_row = content_by_id.get(video_id)
        contract_row = selected_contract[video_id]
        if summary_row is None or content_row is None:
            continue
        content_status = content_row.get("status", "")
        summary_status = summary_row.get("status", "")
        if content_status not in {"PASS", "FAIL"} or summary_status not in {"PASS", "FAIL"}:
            issues.append(
                _issue(
                    "source",
                    "invalid_legacy_extractor_status",
                    f"content_status={content_status} summary_status={summary_status}",
                    video_id,
                )
            )
        elif content_status != summary_status:
            issues.append(
                _issue(
                    "source",
                    "csv_content_manifest_status_mismatch",
                    f"content_status={content_status} summary_status={summary_status}",
                    video_id,
                )
            )
        observed_schema_hashes.add(content_row.get("schema_sha256", ""))
        csv_path = extraction_root / f"{video_id}.csv"
        if not csv_path.is_file():
            issues.append(_issue("source", "missing_behavior_csv", str(csv_path), video_id))
            continue
        observed_sha = _sha256_file(csv_path)
        observed_size = csv_path.stat().st_size
        expected_sha = content_row["csv_sha256"]
        expected_size = _nonnegative_int(content_row["csv_size_bytes"], "csv_size_bytes")
        if observed_sha != expected_sha or observed_size != expected_size:
            issues.append(
                _issue(
                    "source",
                    "behavior_csv_content_mismatch",
                    f"expected_sha={expected_sha} observed_sha={observed_sha} expected_size={expected_size} observed_size={observed_size}",
                    video_id,
                )
            )
            continue
        try:
            counts = _scan_behavior_csv(
                csv_path,
                confidence_threshold=float(source["confidence_threshold"]),
                required_au_columns=list(source["required_au_columns"]),
                required_pose_columns=list(source["required_pose_columns"]),
                expected_column_count=expected_column_count,
            )
        except CoveragePolicyError as exc:
            issues.append(_issue("source", "invalid_behavior_csv", str(exc), video_id))
            continue
        expected_rows = _nonnegative_int(content_row["csv_rows"], "csv_rows")
        contract_frames = _nonnegative_int(contract_row["image_count"], "image_count")
        if counts["frame_count"] != expected_rows or expected_rows != contract_frames:
            issues.append(
                _issue(
                    "source",
                    "behavior_csv_row_count_mismatch",
                    f"actual={counts['frame_count']} content={expected_rows} contract={contract_frames}",
                    video_id,
                )
            )
        expected_success_count = _nonnegative_int(
            summary_row.get("success_count"), "success_count"
        )
        observed_success_ratio = _ratio(counts["success_count"], counts["frame_count"])
        expected_success_ratio = _finite_float(
            summary_row["success_ratio"], f"{video_id}.legacy_success_ratio"
        )
        if counts["success_count"] != expected_success_count or not math.isclose(
            observed_success_ratio, expected_success_ratio, rel_tol=0.0, abs_tol=1e-6
        ):
            issues.append(
                _issue(
                    "source",
                    "video_success_summary_mismatch",
                    f"actual_count={counts['success_count']} summary_count={expected_success_count} actual_ratio={observed_success_ratio:.12g} summary_ratio={expected_success_ratio:.12g}",
                    video_id,
                )
            )
        total_success_count += counts["success_count"]
        if _nonnegative_int(summary_row.get("schema_column_count"), "schema_column_count") != expected_column_count:
            issues.append(
                _issue(
                    "source",
                    "video_schema_column_count_mismatch",
                    f"expected={expected_column_count} observed={summary_row.get('schema_column_count')}",
                    video_id,
                )
            )
        legacy_issues = [part for part in summary_row.get("issues", "").split(";") if part]
        unexpected_legacy = [part for part in legacy_issues if not part.startswith("success_ratio:")]
        if unexpected_legacy:
            issues.append(
                _issue(
                    "source",
                    "noncoverage_extractor_issue",
                    ";".join(unexpected_legacy),
                    video_id,
                )
            )
        row = {
            "physical_split": split_map[video_id],
            "video_id": video_id,
            "task_name": _task_name(video_id),
            **counts,
            "quality_valid_ratio": _ratio(counts["quality_valid_count"], counts["frame_count"]),
            "au_valid_ratio": _ratio(counts["au_valid_count"], counts["frame_count"]),
            "pose_valid_ratio": _ratio(counts["pose_valid_count"], counts["frame_count"]),
            "head_valid_pair_ratio": _ratio(
                counts["head_valid_pair_count"], counts["head_candidate_pair_count"]
            ),
            "legacy_success_ratio": expected_success_ratio,
            "legacy_status": summary_row["status"],
            "policy_enforcement": (
                "PILOT_FEASIBILITY"
                if scope == "pilot" and split_map[video_id] == "train"
                else (
                    "ENFORCED"
                    if scope == "full" and split_map[video_id] == "train"
                    else "REPORT_ONLY"
                )
            ),
            "status": "PENDING",
            "issues": "",
        }
        video_policy_rows.append(row)

    if len(observed_schema_hashes) != 1 or "" in observed_schema_hashes:
        issues.append(
            _issue(
                "source",
                "csv_schema_hash_set_invalid",
                f"observed={sorted(observed_schema_hashes)}",
            )
        )
    if total_success_count != int(summary.get("total_successes", -1)):
        issues.append(
            _issue(
                "source",
                "summary_total_successes_mismatch",
                f"actual={total_success_count} summary={summary.get('total_successes')}",
            )
        )

    aggregate_rows = [
        _aggregate(video_policy_rows, scope, split, task)
        for split in ("ALL", "train", "val", "test")
        for task in ("ALL", "Freeform", "Northwind")
        if any(
            (split == "ALL" or row["physical_split"] == split)
            and (task == "ALL" or row["task_name"] == task)
            for row in video_policy_rows
        )
    ]

    threshold_issues: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    if scope == "pilot":
        thresholds = policy["pilot_feasibility_thresholds"]
        if thresholds.get("aggregate_ratio_policy") != "REPORT_ONLY_INCOMPLETE_PHYSICAL_TRAIN":
            raise CoveragePolicyError(
                "pilot aggregate ratios must be report-only because the pilot is not the complete physical train split"
            )
        enforcement_split = str(thresholds.get("enforcement_split", ""))
        if enforcement_split != "train":
            raise CoveragePolicyError("pilot feasibility may enforce per-video minima only on physical train")
        enforced_videos = [
            row for row in video_policy_rows if row["physical_split"] == enforcement_split
        ]
        full_thresholds = policy["physical_train_full_thresholds"]
        train_overall = next(
            (
                row
                for row in aggregate_rows
                if row["physical_split"] == "train" and row["task_name"] == "ALL"
            ),
            None,
        )
        if train_overall and train_overall["quality_valid_ratio"] < full_thresholds[
            "minimum_overall_quality_valid_ratio"
        ]:
            warnings.append(
                _warning(
                    "pilot_future_full_diagnostic",
                    "pilot_train_subset_overall_below_future_full_minimum",
                    f"observed={train_overall['quality_valid_ratio']:.12g} future_full_minimum={full_thresholds['minimum_overall_quality_valid_ratio']}",
                )
            )
        train_tasks = [
            row
            for row in aggregate_rows
            if row["physical_split"] == "train" and row["task_name"] != "ALL"
        ]
        for row in train_tasks:
            if row["quality_valid_ratio"] < full_thresholds["minimum_task_quality_valid_ratio"]:
                warnings.append(
                    _warning(
                        "pilot_future_full_diagnostic",
                        "pilot_train_subset_task_below_future_full_minimum",
                        f"task={row['task_name']} observed={row['quality_valid_ratio']:.12g} future_full_minimum={full_thresholds['minimum_task_quality_valid_ratio']}",
                    )
                )
        task_ratios = [row["quality_valid_ratio"] for row in train_tasks]
        if task_ratios and max(task_ratios) - min(task_ratios) > full_thresholds[
            "maximum_task_quality_valid_ratio_gap"
        ]:
            warnings.append(
                _warning(
                    "pilot_future_full_diagnostic",
                    "pilot_train_subset_task_gap_above_future_full_maximum",
                    f"observed={max(task_ratios) - min(task_ratios):.12g} future_full_maximum={full_thresholds['maximum_task_quality_valid_ratio_gap']}",
                )
            )
    else:
        thresholds = policy["physical_train_full_thresholds"]
        enforced_videos = [row for row in video_policy_rows if row["physical_split"] == "train"]
        train_overall = next(
            row
            for row in aggregate_rows
            if row["physical_split"] == "train" and row["task_name"] == "ALL"
        )
        if train_overall["quality_valid_ratio"] < thresholds["minimum_overall_quality_valid_ratio"]:
            threshold_issues.append(
                _issue(
                    "train_coverage",
                    "train_quality_valid_ratio_below_minimum",
                    f"observed={train_overall['quality_valid_ratio']:.12g} minimum={thresholds['minimum_overall_quality_valid_ratio']}",
                )
            )
        train_tasks = [
            row
            for row in aggregate_rows
            if row["physical_split"] == "train" and row["task_name"] != "ALL"
        ]
        for row in train_tasks:
            if row["quality_valid_ratio"] < thresholds["minimum_task_quality_valid_ratio"]:
                threshold_issues.append(
                    _issue(
                        "train_coverage",
                        "train_task_quality_valid_ratio_below_minimum",
                        f"task={row['task_name']} observed={row['quality_valid_ratio']:.12g} minimum={thresholds['minimum_task_quality_valid_ratio']}",
                    )
                )
        task_ratios = [row["quality_valid_ratio"] for row in train_tasks]
        if task_ratios and max(task_ratios) - min(task_ratios) > thresholds[
            "maximum_task_quality_valid_ratio_gap"
        ]:
            threshold_issues.append(
                _issue(
                    "train_coverage",
                    "train_task_quality_valid_ratio_gap_above_maximum",
                    f"observed={max(task_ratios) - min(task_ratios):.12g} maximum={thresholds['maximum_task_quality_valid_ratio_gap']}",
                )
            )

    for row in enforced_videos:
        row_issues = []
        if row["au_valid_count"] < thresholds["minimum_au_valid_frames_per_video"]:
            row_issues.append("au_valid_frames_below_minimum")
        if row["head_valid_pair_count"] < thresholds["minimum_head_valid_pairs_per_video"]:
            row_issues.append("head_valid_pairs_below_minimum")
        if thresholds.get("require_each_group_nonzero") and (
            row["quality_valid_count"] == 0
            or row["au_valid_count"] == 0
            or row["head_valid_pair_count"] == 0
        ):
            row_issues.append("required_group_has_zero_coverage")
        row["issues"] = ";".join(row_issues)
        row["status"] = "PASS" if not row_issues else "BLOCKED"
        for code in row_issues:
            threshold_issues.append(
                _issue("video_coverage", code, "mask-aware minimum not met", row["video_id"])
            )

    enforced_ids = {row["video_id"] for row in enforced_videos}
    for row in video_policy_rows:
        if row["video_id"] not in enforced_ids:
            row["policy_enforcement"] = "REPORT_ONLY"
            row["status"] = "REPORT_ONLY"
            row["issues"] = ""

    issues.extend(threshold_issues)
    status = (
        policy["decision_contract"]["blocked_status"]
        if issues
        else policy["decision_contract"][
            "pilot_pass_status" if scope == "pilot" else "full_pass_status"
        ]
    )
    for row in aggregate_rows:
        if scope == "pilot":
            row["policy_enforcement"] = "REPORT_ONLY_INCOMPLETE_SPLIT"
        elif row["physical_split"] == "train":
            row["policy_enforcement"] = "ENFORCED"
        else:
            row["policy_enforcement"] = "REPORT_ONLY"
        report_only = str(row["policy_enforcement"]).startswith("REPORT_ONLY")
        row["status"] = "BLOCKED" if issues and not report_only else (
            "PASS" if not report_only else "REPORT_ONLY"
        )
        row["issues"] = ""

    video_path = _write_csv(tables_dir / "video_coverage_policy.csv", video_policy_rows, VIDEO_FIELDS)
    aggregate_path = _write_csv(
        tables_dir / "aggregate_coverage_policy.csv", aggregate_rows, AGGREGATE_FIELDS
    )
    issues_path = _write_csv(tables_dir / "coverage_policy_issues.csv", issues, ISSUE_FIELDS)
    warnings_path = _write_csv(
        tables_dir / "coverage_policy_warnings.csv", warnings, ISSUE_FIELDS
    )

    decision = {
        "policy_id": policy["policy_id"],
        "policy_version": policy["policy_version"],
        "policy_sha256": _sha256_file(policy_path),
        "scope": scope,
        "status": status,
        "source_structural_complete": not any(row["scope"] in {"source", "pilot_evidence"} for row in issues),
        "mask_aware_coverage_pass": not threshold_issues,
        "full_train_aggregate_thresholds_evaluated": scope == "full",
        "legacy_strict_status_observed": summary.get("status"),
        "legacy_strict_status_used_as_gate": False,
        "full_rich_authorized": False,
        "p0b_authorized": False,
        "training_authorized": False,
        "next_action": _decision_next_action(policy, status),
        "nonblocking_future_full_warnings": warnings,
        "counts": {
            "video_count": len(video_policy_rows),
            "frame_count": sum(row["frame_count"] for row in video_policy_rows),
            "quality_valid_count": sum(row["quality_valid_count"] for row in video_policy_rows),
            "au_valid_count": sum(row["au_valid_count"] for row in video_policy_rows),
            "head_valid_pair_count": sum(row["head_valid_pair_count"] for row in video_policy_rows),
            "blocking_issue_count": len(issues),
            "warning_count": len(warnings),
        },
        "label_access": {
            "bdi_label_access_count": 0,
            "prediction_access_count": 0,
            "checkpoint_access_count": 0,
            "validation_test_threshold_callback_count": 0,
        },
    }
    decision_path = _write_json(output_dir / "coverage_policy_decision.json", decision)

    report_lines = [
        "# PB-P0A2 mask-aware source/coverage policy",
        "",
        f"- Policy: `{policy['policy_id']}` v{policy['policy_version']}.",
        f"- Scope: `{scope}`.",
        f"- Status: `{status}`.",
        f"- Videos/frames: {decision['counts']['video_count']}/{decision['counts']['frame_count']}.",
        f"- Quality-valid frames: {decision['counts']['quality_valid_count']}.",
        f"- AU-valid frames: {decision['counts']['au_valid_count']}.",
        f"- Head-valid adjacent pairs: {decision['counts']['head_valid_pair_count']}.",
        f"- Blocking issues: {len(issues)}.",
        f"- Non-blocking future-full warnings: {len(warnings)}.",
        "- Legacy strict 0.995 status is retained as diagnostic evidence, not used as the mask-aware gate.",
        "- Pilot aggregate ratios are report-only because the 20-video subset is not the complete physical train split.",
        "- Pilot per-video minimums are enforced only for selected physical-train videos; val/test are report-only.",
        "- No BDI label, prediction or checkpoint was read; validation/test thresholds were not tuned.",
        "- This decision does not authorize full-rich extraction, P0B, model changes or training.",
        f"- Next action: `{decision['next_action']}`.",
    ]
    if warnings:
        report_lines.extend(
            ["", "## Future-full diagnostics", ""]
            + [f"- `{row['issue_code']}`: {row['detail']}" for row in warnings]
        )
    report_path = reports_dir / "coverage_policy_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    output_files = [
        video_path,
        aggregate_path,
        issues_path,
        warnings_path,
        decision_path,
        report_path,
    ]
    manifest = {
        "audit": "PB-P0A2 mask-aware source/coverage policy validation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "command_line": " ".join(sys.argv),
        "inputs": {
            "policy": {"path": str(policy_path), "sha256": _sha256_file(policy_path)},
            "dataset_split": {
                "path": str(dataset_split_file),
                "sha256": _sha256_file(dataset_split_file),
            },
            "extraction_root": str(extraction_root),
            **{
                label: {"path": str(path), "sha256": _sha256_file(path)}
                for label, path in paths.items()
            },
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256_file(path)}
            for path in output_files
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "read_only_inputs": True,
        "source_data_modified": False,
        "model_modified": False,
        "training_started": False,
        "full_rich_authorized": False,
        "p0b_authorized": False,
        "training_authorized": False,
        "python": sys.version,
        "platform": platform.platform(),
    }
    manifest_path = _write_json(output_dir / "run_manifest.json", manifest)
    return [*output_files, manifest_path]
