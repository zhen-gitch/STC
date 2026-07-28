"""PB-P0 orchestration for aligned-JPG privileged behavior targets.

This module is deliberately an audit/materialization boundary, not a dataset
or training implementation.  It joins the exact physical dataset split to the
exact aligned JPG frame ids, projects only the approved OpenFace AU/head
columns, and derives the external frame clock from ``source_video_contract``.

The historical raw-video OpenFace CSV paths recorded by that source contract
are never opened or copied.  Gaze values are never projected, normalized, or
written.  A landmark-only OpenFace run therefore produces a complete,
reproducible ``BLOCKED`` audit rather than an exception or a partial target
cache.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.diagnostics.privileged_behavior_contract import (
    AU_SOURCE_COLUMNS,
    OPTIONAL_ROW_IDENTITY_COLUMNS,
    POSE_ROTATION_SOURCE_COLUMNS,
    POSE_VELOCITY_TARGET_COLUMNS,
    REQUIRED_OPENFACE_COLUMNS,
    SELECTED_BEHAVIOR_SOURCE_COLUMNS,
    FrameReference,
    PrivilegedBehaviorContractError,
    audit_privileged_behavior_contract,
    fit_train_only_statistics,
)


PHYSICAL_SPLITS = ("train", "val", "test")
SELECTED_TARGET_COLUMNS = AU_SOURCE_COLUMNS + POSE_VELOCITY_TARGET_COLUMNS
FROZEN_OPENFACE_PACKAGE = "OpenFace_2.2.0_win_x64"
FROZEN_FEATURE_EXTRACTION_SHA256 = (
    "a29ba49cfc59039bfe5e2f141898b2a110da420f6f520d6a923a86ac78cd96ae"
)
FROZEN_MODEL_SHA256 = (
    "7efbef33dbc3e54197960300827657f9fe7a42c0953ef52c2af054a6fdbc3598"
)
FROZEN_OPENFACE_README_SHA256 = (
    "4ccdd65f992124db8127688a545a9344b537bdeb2d97371cfddfd65fc68a1d93"
)
FROZEN_BEHAVIOR_PROFILE = "quality_2d_pose_au_no_gaze_no_hog_v1"
FROZEN_MIN_SUCCESS_RATIO = 0.995
REQUIRED_BEHAVIOR_ARGUMENTS = ("-fdir", "-2Dfp", "-pose", "-aus")
CSV_CONTENT_MANIFEST_FIELDS = (
    "video_id",
    "csv_rows",
    "schema_sha256",
    "csv_size_bytes",
    "csv_sha256",
    "status",
)
SOURCE_CLOCK_COLUMNS = (
    "split",
    "video_id",
    "task_name",
    "aligned_frame_count",
    "raw_frame_count",
    "frame_count_match",
    "raw_fps",
    "contract_status",
    "issues",
)

FRAME_CONTRACT_FIELDS = (
    "physical_split",
    "video_id",
    "task_name",
    "frame_id",
    "image_filename",
    "image_path",
    "source_fps",
    "source_timestamp_seconds",
    "openface_csv",
    "openface_frame_present",
    "exact_frame_join",
    "quality_valid",
    "au_valid",
    "pose_valid",
    "pose_velocity_candidate",
    "pose_velocity_valid",
    "pose_velocity_dt_seconds",
    *AU_SOURCE_COLUMNS,
    *POSE_VELOCITY_TARGET_COLUMNS,
    "invalid_reasons",
    "status",
)

GROUP_COVERAGE_FIELDS = (
    "physical_split",
    "video_id",
    "task_name",
    "image_frame_count",
    "openface_row_count",
    "joined_frame_count",
    "frame_join_ratio",
    "quality_valid_count",
    "quality_valid_ratio",
    "au_valid_count",
    "au_valid_ratio",
    "pose_valid_count",
    "pose_valid_ratio",
    "head_candidate_count",
    "head_valid_count",
    "head_valid_ratio",
    "head_missing_timestamp_pair_count",
    "head_illegal_dt_pair_count",
    "head_nonconsecutive_pair_count",
    "schema_status",
    "missing_behavior_columns",
    "status",
    "issues",
)

TARGET_STATS_FIELDS = (
    "physical_split",
    "target_group",
    "target_name",
    "count",
    "mean",
    "std",
    "minimum",
    "maximum",
    "status",
    "reason",
)

ISSUE_FIELDS = (
    "severity",
    "scope",
    "physical_split",
    "video_id",
    "task_name",
    "frame_id",
    "issue_code",
    "detail",
)

SOURCE_FIDELITY_FIELDS = (
    "scope",
    "physical_split",
    "video_id",
    "task_name",
    "check_name",
    "expected",
    "observed",
    "status",
    "detail",
)

IDENTITY_TASK_RISK_FIELDS = (
    "risk_type",
    "target_group",
    "target_names",
    "physical_split_policy",
    "status",
    "probe_value_access_count",
    "reason",
)


class PrivilegedBehaviorP0InputError(ValueError):
    """Raised before output creation when the frozen split is unreadable."""


def _format(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def _write_csv(path: Path, rows, fieldnames) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field)) for field in fieldnames})
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_capture(project_root: Path, arguments: list[str]) -> tuple[str, bool]:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "", False
    return result.stdout.strip(), True


def _canonical_cross_platform_path(value: Any) -> str:
    """Normalize Windows/WSL spellings without erasing POSIX case semantics."""

    text = str(value or "").strip().replace("\\", "/")
    wsl_unc = re.match(
        r"^//(?:wsl\.localhost|wsl\$)/ubuntu(?P<path>/.*)?$",
        text,
        flags=re.IGNORECASE,
    )
    if wsl_unc:
        text = wsl_unc.group("path") or "/"
    else:
        drive = re.match(r"^([A-Za-z]):(?:/(.*))?$", text)
        if drive:
            suffix = drive.group(2) or ""
            text = f"/mnt/{drive.group(1).lower()}"
            if suffix:
                text += f"/{suffix}"
        else:
            mounted_drive = re.match(r"^/mnt/([A-Za-z])(?:/(.*))?$", text)
            if mounted_drive:
                suffix = mounted_drive.group(2) or ""
                text = f"/mnt/{mounted_drive.group(1).lower()}"
                if suffix:
                    text += f"/{suffix}"
    return text if text == "/" else text.rstrip("/")


def _paths_cross_platform_equivalent(left: Any, right: Any) -> bool:
    left_value = _canonical_cross_platform_path(left)
    right_value = _canonical_cross_platform_path(right)
    if not left_value or not right_value:
        return False
    drive_pattern = re.compile(r"^/mnt/[a-z](?:/|$)")
    if drive_pattern.match(left_value) and drive_pattern.match(right_value):
        return left_value.casefold() == right_value.casefold()
    return left_value == right_value


def _resolve_cross_platform_path(value: Any, project_root: Path) -> Path | None:
    canonical = _canonical_cross_platform_path(value)
    if not canonical:
        return None
    candidate = Path(canonical).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    project_root = Path(project_root).resolve()
    resolved = (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        return None
    return resolved


def _iter_integrity_image_files(image_root: Path):
    for video_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
        for image_path in sorted(
            path
            for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
        ):
            yield image_path.relative_to(image_root).as_posix(), image_path


def _safe_output_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output_dir must be empty or absent: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer() or number <= 0:
        return None
    return int(number)


def _parse_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        return None
    return int(number)


def _parse_positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def _parse_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy_contract_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def _valid_git_branch_name(value: Any) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if value == "@" or value.startswith(('.', '/')) or value.endswith(('/', '.', '.lock')):
        return False
    if ".." in value or "//" in value or "@{" in value:
        return False
    return not any(
        ord(character) <= 0x20
        or ord(character) == 0x7F
        or character in "~^:?*[\\"
        for character in value
    )


def _task_from_video_id(video_id: str) -> str:
    match = re.search(r"_(Freeform|Northwind)_", video_id)
    return match.group(1) if match else ""


def _read_exact_split(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrivilegedBehaviorP0InputError(
            f"cannot read exact dataset split {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PrivilegedBehaviorP0InputError("dataset split must be a JSON object")
    missing = [split for split in PHYSICAL_SPLITS if split not in payload]
    extra = sorted(set(payload) - set(PHYSICAL_SPLITS))
    if missing or extra:
        raise PrivilegedBehaviorP0InputError(
            f"dataset split keys must be exactly {PHYSICAL_SPLITS}; missing={missing} extra={extra}"
        )

    split_by_video: dict[str, str] = {}
    videos_by_split: dict[str, list[str]] = {split: [] for split in PHYSICAL_SPLITS}
    for split in PHYSICAL_SPLITS:
        values = payload[split]
        if not isinstance(values, list):
            raise PrivilegedBehaviorP0InputError(f"dataset split {split!r} must be a list")
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value or value != value.strip():
                raise PrivilegedBehaviorP0InputError(
                    f"dataset split {split}[{index}] must be a non-empty exact string"
                )
            if value in split_by_video:
                raise PrivilegedBehaviorP0InputError(
                    f"video appears more than once in exact dataset split: {value}"
                )
            split_by_video[value] = split
            videos_by_split[split].append(value)
    if not split_by_video:
        raise PrivilegedBehaviorP0InputError("dataset split contains no videos")
    if not videos_by_split["train"]:
        raise PrivilegedBehaviorP0InputError("physical train split contains no videos")
    return split_by_video, videos_by_split


def _read_source_clock_contract(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Read only identity/frame-clock/status fields from the raw source contract."""

    problems: list[str] = []
    rows_by_video: dict[str, dict[str, str]] = {}
    try:
        handle = path.open("r", newline="", encoding="utf-8-sig")
    except OSError as exc:
        return {}, [f"cannot_open:{exc}"]
    with handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        fieldnames = tuple(str(value).strip() for value in (reader.fieldnames or ()))
        missing = [column for column in SOURCE_CLOCK_COLUMNS if column not in fieldnames]
        if missing:
            return {}, [f"missing_columns:{','.join(missing)}"]
        for row_index, raw_row in enumerate(reader, start=2):
            # Intentionally project an allowlist.  In particular, raw_video_path
            # and raw_openface_csv are never looked up and can never become an
            # input path to this audit.
            row = {
                column: (
                    raw_row.get(column, "").strip()
                    if isinstance(raw_row.get(column, ""), str)
                    else raw_row.get(column, "")
                )
                for column in SOURCE_CLOCK_COLUMNS
            }
            video_id = str(row["video_id"])
            if not video_id:
                problems.append(f"empty_video_id:row={row_index}")
                continue
            if video_id in rows_by_video:
                problems.append(f"duplicate_video_id:{video_id}")
                continue
            rows_by_video[video_id] = row
    return rows_by_video, problems


def _read_source_run_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {}, f"cannot_read_source_run_manifest:{exc}"
    if not isinstance(payload, dict):
        return {}, "source_run_manifest_is_not_an_object"
    return payload, ""


def _frame_id_from_jpg(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    return _parse_positive_int(match.group(1)) if match else None


def _read_image_frames(video_dir: Path) -> tuple[list[tuple[int, Path]], list[str]]:
    if not video_dir.is_dir():
        return [], ["missing_aligned_image_dir"]
    jpgs = sorted(
        path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"
    )
    issues: list[str] = []
    parsed: list[tuple[int, Path]] = []
    for path in jpgs:
        frame_id = _frame_id_from_jpg(path)
        if frame_id is None:
            issues.append(f"unparseable_jpg_frame_id:{path.name}")
        else:
            parsed.append((frame_id, path))
    counts = Counter(frame_id for frame_id, _path in parsed)
    duplicates = sorted(frame_id for frame_id, count in counts.items() if count > 1)
    if duplicates:
        issues.append(f"duplicate_jpg_frame_ids:{duplicates[:20]}")
    parsed.sort(key=lambda item: item[0])
    if not jpgs:
        issues.append("no_aligned_jpgs")
    return parsed, issues


def _read_openface_projection(path: Path) -> dict[str, Any]:
    """Read frame ids and, only for a complete rich schema, approved values."""

    result = {
        "fieldnames": (),
        "rows": [],
        "frame_ids": [],
        "row_count": 0,
        "missing_behavior_columns": list(SELECTED_BEHAVIOR_SOURCE_COLUMNS),
        "schema_complete": False,
        "issues": [],
        "gaze_header_columns": [],
        "access_counts": Counter(),
    }
    if not path.is_file():
        result["issues"].append("missing_openface_csv")
        return result

    try:
        handle = path.open("r", newline="", encoding="utf-8-sig")
    except OSError as exc:
        result["issues"].append(f"cannot_open_openface_csv:{exc}")
        return result
    with handle:
        reader = csv.reader(handle, skipinitialspace=True)
        try:
            raw_header = next(reader)
        except StopIteration:
            result["issues"].append("empty_openface_csv")
            return result
        header = tuple(str(value).strip() for value in raw_header)
        result["fieldnames"] = header
        duplicate_columns = sorted(
            column for column, count in Counter(header).items() if count > 1
        )
        if duplicate_columns:
            result["issues"].append(f"duplicate_openface_columns:{duplicate_columns}")
        missing_behavior = [
            column for column in SELECTED_BEHAVIOR_SOURCE_COLUMNS if column not in header
        ]
        missing_required = [column for column in REQUIRED_OPENFACE_COLUMNS if column not in header]
        result["missing_behavior_columns"] = missing_behavior
        result["gaze_header_columns"] = [
            column for column in header if column.startswith("gaze_")
        ]
        schema_complete = not duplicate_columns and not missing_required
        result["schema_complete"] = schema_complete
        missing_nonbehavior = [
            column for column in missing_required if column not in SELECTED_BEHAVIOR_SOURCE_COLUMNS
        ]
        if missing_nonbehavior:
            result["issues"].append(
                f"missing_required_openface_columns:{missing_nonbehavior}"
            )
        if "frame" not in header:
            result["issues"].append("missing_openface_frame_column")
            return result

        indices = {column: header.index(column) for column in set(header)}
        projected_columns = ["frame"]
        if schema_complete:
            projected_columns = list(REQUIRED_OPENFACE_COLUMNS)
            identity_columns = [
                column for column in OPTIONAL_ROW_IDENTITY_COLUMNS if column in header
            ]
            if identity_columns:
                projected_columns.extend(identity_columns)
        max_index = max(indices[column] for column in projected_columns)
        for row_number, raw_values in enumerate(reader, start=2):
            if not raw_values or all(not str(value).strip() for value in raw_values):
                result["issues"].append(f"blank_openface_row:{row_number}")
                continue
            result["row_count"] += 1
            if len(raw_values) <= max_index:
                result["issues"].append(f"short_openface_row:{row_number}")
                continue
            frame_value = raw_values[indices["frame"]].strip()
            frame_id = _parse_positive_int(frame_value)
            result["access_counts"]["frame"] += 1
            if frame_id is None:
                result["issues"].append(f"invalid_openface_frame:{row_number}:{frame_value!r}")
                continue
            result["frame_ids"].append(frame_id)
            if schema_complete:
                row = {
                    column: raw_values[indices[column]].strip()
                    for column in projected_columns
                }
                result["rows"].append(row)
                for column in projected_columns:
                    if column != "frame":
                        result["access_counts"][column] += 1

    duplicates = sorted(
        frame_id
        for frame_id, count in Counter(result["frame_ids"]).items()
        if count > 1
    )
    if duplicates:
        result["issues"].append(f"duplicate_openface_frame_ids:{duplicates[:20]}")
    return result


def _issue_row(
    code: str,
    detail: str,
    *,
    scope: str = "global",
    physical_split: str = "",
    video_id: str = "",
    task_name: str = "",
    frame_id: Any = "",
    severity: str = "BLOCKER",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "scope": scope,
        "physical_split": physical_split,
        "video_id": video_id,
        "task_name": task_name,
        "frame_id": frame_id,
        "issue_code": code,
        "detail": detail,
    }


def _fidelity_row(
    check_name: str,
    expected: Any,
    observed: Any,
    status: str,
    detail: str,
    *,
    scope: str = "global",
    physical_split: str = "",
    video_id: str = "",
    task_name: str = "",
) -> dict[str, Any]:
    return {
        "scope": scope,
        "physical_split": physical_split,
        "video_id": video_id,
        "task_name": task_name,
        "check_name": check_name,
        "expected": expected,
        "observed": observed,
        "status": status,
        "detail": detail,
    }


def _validate_current_image_integrity(
    *,
    manifest: dict[str, Any],
    image_root: Path,
    project_root: Path,
    total_image_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Bind the current aligned-JPG bytes to the source candidate manifest."""

    issues: list[dict[str, Any]] = []
    fidelity: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "status": "BLOCKED",
        "comparison_summary": "",
        "comparison_summary_sha256": "",
        "candidate_manifest": "",
        "candidate_manifest_sha256": "",
        "candidate_row_count": 0,
        "current_image_count": 0,
        "exact_match_count": 0,
        "missing_current_file_count": 0,
        "extra_current_file_count": 0,
        "size_mismatch_count": 0,
        "sha256_mismatch_count": 0,
        "current_file_read_error_count": 0,
        "samples": [],
    }

    def check(code: str, expected: Any, observed: Any, passed: bool, detail: str) -> None:
        fidelity.append(
            _fidelity_row(
                code,
                expected,
                observed,
                "PASS" if passed else "FAIL",
                detail,
            )
        )
        if not passed:
            issues.append(
                _issue_row(
                    f"rich_{code}_mismatch",
                    f"expected={expected!r} observed={observed!r}",
                )
            )

    recorded_summary_path = manifest.get("integrity_comparison_summary", "")
    comparison_summary_path = _resolve_cross_platform_path(
        recorded_summary_path, project_root
    )
    evidence["comparison_summary"] = (
        str(comparison_summary_path) if comparison_summary_path is not None else ""
    )
    summary_path_valid = (
        comparison_summary_path is not None and comparison_summary_path.is_file()
    )
    check(
        "source_integrity_comparison_summary_path",
        "existing absolute/cross-platform or project-root-relative file",
        recorded_summary_path,
        summary_path_valid,
        "The source run must identify the exact image-integrity comparison summary.",
    )
    if not summary_path_valid:
        return issues, fidelity, evidence

    try:
        summary_bytes = comparison_summary_path.read_bytes()
        summary = json.loads(summary_bytes.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append(
            _issue_row("rich_integrity_comparison_summary_invalid", str(exc))
        )
        return issues, fidelity, evidence
    if not isinstance(summary, dict):
        issues.append(
            _issue_row(
                "rich_integrity_comparison_summary_invalid",
                "comparison summary is not a JSON object",
            )
        )
        return issues, fidelity, evidence

    actual_summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    evidence["comparison_summary_sha256"] = actual_summary_sha256
    recorded_summary_sha256 = str(
        manifest.get("integrity_comparison_summary_sha256", "")
    ).lower()
    check(
        "source_integrity_comparison_summary_sha256",
        actual_summary_sha256,
        recorded_summary_sha256,
        recorded_summary_sha256 == actual_summary_sha256,
        "The source run manifest must bind the parsed image-integrity summary bytes.",
    )

    summary_reference_count = _parse_positive_int(summary.get("reference_count"))
    summary_candidate_count = _parse_positive_int(summary.get("candidate_count"))
    status_counts = summary.get("status_counts")
    summary_exact_count = (
        _parse_positive_int(status_counts.get("EXACT_MATCH"))
        if isinstance(status_counts, dict)
        else None
    )
    summary_reference_sha256 = str(
        summary.get("reference_manifest_sha256", "")
    ).lower()
    summary_candidate_sha256 = str(
        summary.get("candidate_manifest_sha256", "")
    ).lower()
    summary_contract = {
        "audit": summary.get("audit"),
        "status": summary.get("status"),
        "reference_count": summary_reference_count,
        "candidate_count": summary_candidate_count,
        "exact_match_count": summary_exact_count,
        "reference_manifest_sha256": summary_reference_sha256,
        "candidate_manifest_sha256": summary_candidate_sha256,
    }
    summary_contract_valid = (
        summary.get("audit") == "aligned image cross-machine comparison"
        and summary.get("status") == "EXACT_PASS"
        and summary_reference_count == total_image_frames
        and summary_candidate_count == total_image_frames
        and summary_exact_count == total_image_frames
        and bool(re.fullmatch(r"[0-9a-f]{64}", summary_reference_sha256))
        and summary_reference_sha256 == summary_candidate_sha256
    )
    check(
        "integrity_comparison_summary_contract",
        {
            "audit": "aligned image cross-machine comparison",
            "status": "EXACT_PASS",
            "counts": total_image_frames,
            "equal_manifest_sha256": True,
        },
        summary_contract,
        summary_contract_valid,
        "The comparison summary must prove exact equality for every current split frame.",
    )

    recorded_candidate_path = summary.get("candidate_manifest", "")
    candidate_manifest_path = _resolve_cross_platform_path(
        recorded_candidate_path, project_root
    )
    evidence["candidate_manifest"] = (
        str(candidate_manifest_path) if candidate_manifest_path is not None else ""
    )
    candidate_path_valid = (
        candidate_manifest_path is not None and candidate_manifest_path.is_file()
    )
    check(
        "integrity_candidate_manifest_path",
        "existing absolute/cross-platform or project-root-relative file",
        recorded_candidate_path,
        candidate_path_valid,
        "The comparison summary must identify the candidate manifest used for extraction.",
    )
    if not candidate_path_valid:
        return issues, fidelity, evidence

    try:
        actual_candidate_sha256 = _sha256_file(candidate_manifest_path)
    except OSError as exc:
        issues.append(_issue_row("rich_integrity_candidate_manifest_invalid", str(exc)))
        return issues, fidelity, evidence
    evidence["candidate_manifest_sha256"] = actual_candidate_sha256
    manifest_candidate_sha256 = str(
        manifest.get("integrity_candidate_manifest_sha256", "")
    ).lower()
    candidate_hash_valid = (
        actual_candidate_sha256 == summary_candidate_sha256
        and actual_candidate_sha256 == manifest_candidate_sha256
    )
    check(
        "integrity_candidate_manifest_sha256",
        {
            "summary": actual_candidate_sha256,
            "source_manifest": actual_candidate_sha256,
        },
        {
            "summary": summary_candidate_sha256,
            "source_manifest": manifest_candidate_sha256,
        },
        candidate_hash_valid,
        "Both provenance documents must bind the exact candidate image manifest.",
    )
    if not summary_contract_valid or not candidate_hash_valid:
        return issues, fidelity, evidence
    if not image_root.is_dir():
        check(
            "current_image_content_binding",
            "existing current image root",
            str(image_root),
            False,
            "The current aligned-image root must remain readable for byte binding.",
        )
        return issues, fidelity, evidence

    samples: list[str] = []

    def sample(detail: str) -> None:
        if len(samples) < 20:
            samples.append(detail)

    candidate_row_count = 0
    current_image_count = 0
    exact_match_count = 0
    missing_count = 0
    extra_count = 0
    size_mismatch_count = 0
    sha_mismatch_count = 0
    read_error_count = 0
    manifest_structure_valid = True
    previous_relative_path = ""
    actual_iterator = iter(_iter_integrity_image_files(image_root))
    actual_entry = next(actual_iterator, None)
    try:
        handle = candidate_manifest_path.open(
            "r", newline="", encoding="utf-8-sig"
        )
    except OSError as exc:
        issues.append(_issue_row("rich_integrity_candidate_manifest_invalid", str(exc)))
        return issues, fidelity, evidence

    with handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        fields = tuple(str(value).strip() for value in (reader.fieldnames or ()))
        required_fields = ("relative_path", "file_size", "file_sha256")
        duplicate_fields = sorted(
            field for field, count in Counter(fields).items() if count > 1
        )
        missing_fields = [field for field in required_fields if field not in fields]
        if missing_fields or duplicate_fields:
            manifest_structure_valid = False
            sample(
                f"manifest_header:missing={missing_fields}:duplicates={duplicate_fields}"
            )
        else:
            for row_number, row in enumerate(reader, start=2):
                candidate_row_count += 1
                relative_path = str(row.get("relative_path", ""))
                path_parts = relative_path.split("/")
                relative_valid = (
                    bool(relative_path)
                    and relative_path == relative_path.strip()
                    and "\\" not in relative_path
                    and not relative_path.startswith("/")
                    and all(part not in {"", ".", ".."} for part in path_parts)
                )
                file_size = _parse_positive_int(row.get("file_size"))
                file_sha256 = str(row.get("file_sha256", "")).strip().lower()
                row_valid = (
                    relative_valid
                    and file_size is not None
                    and bool(re.fullmatch(r"[0-9a-f]{64}", file_sha256))
                    and (not previous_relative_path or relative_path > previous_relative_path)
                )
                if not row_valid:
                    manifest_structure_valid = False
                    sample(
                        f"candidate_row:{row_number}:relative={relative_path!r}:size={row.get('file_size')!r}:sha256={row.get('file_sha256')!r}"
                    )
                    break
                previous_relative_path = relative_path

                while actual_entry is not None and actual_entry[0] < relative_path:
                    extra_count += 1
                    current_image_count += 1
                    sample(f"extra_current:{actual_entry[0]}")
                    actual_entry = next(actual_iterator, None)
                if actual_entry is None or relative_path < actual_entry[0]:
                    missing_count += 1
                    sample(f"missing_current:{relative_path}")
                    continue

                actual_relative_path, actual_path = actual_entry
                current_image_count += 1
                try:
                    actual_size = actual_path.stat().st_size
                    actual_sha256 = _sha256_file(actual_path)
                except OSError as exc:
                    read_error_count += 1
                    sample(f"current_read_error:{actual_relative_path}:{exc}")
                else:
                    if actual_size != file_size:
                        size_mismatch_count += 1
                        sample(
                            f"size_mismatch:{actual_relative_path}:{file_size}!={actual_size}"
                        )
                    if actual_sha256 != file_sha256:
                        sha_mismatch_count += 1
                        sample(
                            f"sha256_mismatch:{actual_relative_path}:{file_sha256}!={actual_sha256}"
                        )
                    if actual_size == file_size and actual_sha256 == file_sha256:
                        exact_match_count += 1
                actual_entry = next(actual_iterator, None)

    while actual_entry is not None:
        extra_count += 1
        current_image_count += 1
        sample(f"extra_current:{actual_entry[0]}")
        actual_entry = next(actual_iterator, None)

    evidence.update(
        {
            "candidate_row_count": candidate_row_count,
            "current_image_count": current_image_count,
            "exact_match_count": exact_match_count,
            "missing_current_file_count": missing_count,
            "extra_current_file_count": extra_count,
            "size_mismatch_count": size_mismatch_count,
            "sha256_mismatch_count": sha_mismatch_count,
            "current_file_read_error_count": read_error_count,
            "samples": samples,
        }
    )
    binding_valid = (
        manifest_structure_valid
        and candidate_row_count == total_image_frames
        and current_image_count == total_image_frames
        and exact_match_count == total_image_frames
        and missing_count == 0
        and extra_count == 0
        and size_mismatch_count == 0
        and sha_mismatch_count == 0
        and read_error_count == 0
    )
    check(
        "current_image_content_binding",
        {
            "candidate_rows": total_image_frames,
            "current_images": total_image_frames,
            "exact_matches": total_image_frames,
            "all_issue_counts": 0,
        },
        {
            "candidate_rows": candidate_row_count,
            "current_images": current_image_count,
            "exact_matches": exact_match_count,
            "missing": missing_count,
            "extra": extra_count,
            "size_mismatch": size_mismatch_count,
            "sha256_mismatch": sha_mismatch_count,
            "read_errors": read_error_count,
            "manifest_structure_valid": manifest_structure_valid,
            "samples": samples,
        },
        binding_valid,
        "Every current aligned image must match the source candidate manifest by exact relative path, byte size and SHA-256.",
    )
    evidence["status"] = "PASS" if binding_valid and not issues else "BLOCKED"
    return issues, fidelity, evidence


def _validate_manifest_provenance(
    manifest: dict[str, Any],
    source_video_contract: Path,
    image_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    fidelity: list[dict[str, Any]] = []

    checks = {
        "feature_extraction_sha256": manifest.get("feature_extraction_sha256", ""),
        "model_sha256": manifest.get("model_sha256", ""),
        "image_root": manifest.get("image_root", ""),
    }
    for key, value in checks.items():
        passed = bool(value)
        fidelity.append(
            _fidelity_row(
                f"source_manifest_{key}",
                "non-empty",
                value,
                "PASS" if passed else "FAIL",
                "OpenFace source provenance must be frozen and attributable.",
            )
        )
        if not passed:
            issues.append(_issue_row(f"missing_source_manifest_{key}", "value is empty"))

    recorded_image_root = manifest.get("image_root", "")
    image_root_equivalent = _paths_cross_platform_equivalent(recorded_image_root, image_root)
    fidelity.append(
        _fidelity_row(
            "source_manifest_image_root",
            str(image_root),
            recorded_image_root,
            "PASS" if image_root_equivalent else "RELOCATION_REQUIRES_INTEGRITY_PROOF",
            "Windows drive paths and their /mnt/<drive> WSL spelling are treated as equivalent; other relocation is accepted only by the rich exact-integrity gate.",
        )
    )

    arguments = manifest.get("arguments", [])
    uses_fdir = isinstance(arguments, list) and "-fdir" in [str(value) for value in arguments]
    fidelity.append(
        _fidelity_row(
            "aligned_image_directory_invocation",
            "arguments contain -fdir",
            arguments if isinstance(arguments, list) else type(arguments).__name__,
            "PASS" if uses_fdir else "FAIL",
            "Behavior targets must be extracted from the aligned JPG sequence.",
        )
    )
    if not uses_fdir:
        issues.append(
            _issue_row(
                "source_manifest_not_fdir",
                "source run manifest does not prove aligned-image -fdir extraction",
            )
        )

    integrity_status = str(manifest.get("integrity_status", ""))
    integrity_pass = integrity_status == "EXACT_PASS"
    fidelity.append(
        _fidelity_row(
            "aligned_image_integrity",
            "EXACT_PASS",
            integrity_status,
            "PASS" if integrity_pass else "FAIL",
            "The aligned-JPG input tree must have frozen exact integrity provenance.",
        )
    )
    if not integrity_pass:
        issues.append(
            _issue_row(
                "source_manifest_integrity_not_exact",
                f"integrity_status={integrity_status!r}",
            )
        )

    recorded_contract_hash = str(manifest.get("source_video_contract_sha256", "")).lower()
    actual_contract_hash = (
        _sha256_file(source_video_contract) if source_video_contract.is_file() else ""
    )
    if recorded_contract_hash:
        matches = bool(actual_contract_hash) and recorded_contract_hash == actual_contract_hash
        fidelity.append(
            _fidelity_row(
                "source_video_contract_hash",
                actual_contract_hash,
                recorded_contract_hash,
                "PASS" if matches else "FAIL",
                "Present hashes must match the exact frame-clock contract used by PB-P0.",
            )
        )
        if not matches:
            issues.append(
                _issue_row(
                    "source_video_contract_hash_mismatch",
                    f"recorded={recorded_contract_hash} actual={actual_contract_hash}",
                )
            )
    else:
        fidelity.append(
            _fidelity_row(
                "source_video_contract_hash",
                actual_contract_hash,
                "not recorded",
                "NOT_RECORDED",
                "Legacy landmark-only manifests may omit this field; rich extraction must record it.",
            )
        )

    timestamp_authorized = manifest.get("csv_timestamp_authorized_for_head_velocity")
    if timestamp_authorized is True:
        issues.append(
            _issue_row(
                "openface_timestamp_incorrectly_authorized",
                "aligned -fdir CSV timestamps must never drive head velocity",
            )
        )
    fidelity.append(
        _fidelity_row(
            "head_velocity_timebase",
            "external source FPS/frame clock",
            "OpenFace timestamp forbidden; t=(frame-1)/fps",
            "PASS" if timestamp_authorized is not True else "FAIL",
            "PB-P0 ignores the aligned-image CSV timestamp when deriving d_pose/dt.",
        )
    )

    gaze_counts = {
        key: manifest.get(key, 0)
        for key in (
            "gaze_training_access_count",
            "gaze_normalizer_access_count",
            "gaze_loss_access_count",
        )
    }
    bad_gaze = {key: value for key, value in gaze_counts.items() if value not in (0, "0", None)}
    if manifest.get("gaze_requested") is True:
        bad_gaze["gaze_requested"] = True
    fidelity.append(
        _fidelity_row(
            "gaze_exclusion",
            "requested=false and all access counts=0",
            {"gaze_requested": manifest.get("gaze_requested", False), **gaze_counts},
            "PASS" if not bad_gaze else "FAIL",
            "Header names may be inspected, but gaze values are never projected.",
        )
    )
    if bad_gaze:
        issues.append(_issue_row("source_manifest_gaze_not_excluded", str(bad_gaze)))
    return issues, fidelity


def _validate_rich_source_provenance(
    *,
    manifest: dict[str, Any],
    source_run_manifest: Path,
    source_video_contract: Path,
    image_root: Path,
    project_root: Path,
    expected_video_ids: set[str],
    total_image_frames: int,
    total_openface_rows: int,
    csv_evidence: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Apply strict provenance gates only when every video has a rich schema."""

    issues: list[dict[str, Any]] = []
    fidelity: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {
        "mode": "strict_rich",
        "status": "BLOCKED",
        "extraction_summary": "",
        "csv_content_manifest": "",
    }

    def check(code: str, expected: Any, observed: Any, passed: bool, detail: str) -> None:
        fidelity.append(
            _fidelity_row(
                code,
                expected,
                observed,
                "PASS" if passed else "FAIL",
                detail,
            )
        )
        if not passed:
            issues.append(_issue_row(f"rich_{code}_mismatch", f"expected={expected!r} observed={observed!r}"))

    check(
        "source_feature_extraction_sha256",
        FROZEN_FEATURE_EXTRACTION_SHA256,
        str(manifest.get("feature_extraction_sha256", "")).lower(),
        str(manifest.get("feature_extraction_sha256", "")).lower()
        == FROZEN_FEATURE_EXTRACTION_SHA256,
        "Rich behavior targets require the frozen OpenFace 2.2.0 FeatureExtraction binary.",
    )
    check(
        "source_model_sha256",
        FROZEN_MODEL_SHA256,
        str(manifest.get("model_sha256", "")).lower(),
        str(manifest.get("model_sha256", "")).lower() == FROZEN_MODEL_SHA256,
        "Rich behavior targets require the frozen CE-CLM model.",
    )
    check(
        "source_openface_readme_sha256",
        FROZEN_OPENFACE_README_SHA256,
        str(manifest.get("openface_readme_sha256", "")).lower(),
        str(manifest.get("openface_readme_sha256", "")).lower()
        == FROZEN_OPENFACE_README_SHA256,
        "The release readme hash identifies the frozen OpenFace package.",
    )
    package = str(manifest.get("openface_package_directory", ""))
    check(
        "source_openface_package",
        FROZEN_OPENFACE_PACKAGE,
        package,
        package == FROZEN_OPENFACE_PACKAGE,
        "The rich source must identify the expected OpenFace release directory.",
    )
    profile = str(manifest.get("feature_profile", ""))
    check(
        "source_feature_profile",
        FROZEN_BEHAVIOR_PROFILE,
        profile,
        profile == FROZEN_BEHAVIOR_PROFILE,
        "Only the frozen no-gaze AU/pose profile may authorize PB-P0 rich targets.",
    )
    extractor_script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_openface_aligned_behavior_features.ps1"
    )
    extractor_script_sha256 = (
        _sha256_file(extractor_script_path) if extractor_script_path.is_file() else ""
    )
    check(
        "source_extractor_script_sha256",
        extractor_script_sha256,
        str(manifest.get("script_sha256", "")).lower(),
        bool(extractor_script_sha256)
        and str(manifest.get("script_sha256", "")).lower()
        == extractor_script_sha256,
        "The full rich run must come from the exact currently reviewed extractor script.",
    )
    raw_source_git_commit = manifest.get("git_commit", "")
    raw_source_git_branch = manifest.get("git_branch", "")
    source_git_commit = (
        raw_source_git_commit if isinstance(raw_source_git_commit, str) else ""
    )
    source_git_branch = (
        raw_source_git_branch if isinstance(raw_source_git_branch, str) else ""
    )
    source_git_commit_valid = bool(
        re.fullmatch(r"[0-9A-Fa-f]{7,40}", source_git_commit)
    )
    source_git_branch_valid = _valid_git_branch_name(source_git_branch)
    source_git_status_available = manifest.get("git_status_available") is True
    source_git_status_short = str(manifest.get("git_status_short", ""))
    check(
        "source_git_provenance",
        {
            "commit": "7-40 hexadecimal characters",
            "branch": "valid non-empty branch name",
            "status_available": True,
            "status_short": "",
        },
        {
            "commit": source_git_commit,
            "branch": source_git_branch,
            "status_available": manifest.get("git_status_available"),
            "status_short": source_git_status_short,
        },
        source_git_commit_valid
        and source_git_branch_valid
        and source_git_status_available
        and source_git_status_short == "",
        "A full rich extraction must be attributable to a clean Windows checkout; unavailable status cannot authorize PASS.",
    )

    arguments = manifest.get("arguments", [])
    argument_values = [str(value) for value in arguments] if isinstance(arguments, list) else []
    missing_arguments = [flag for flag in REQUIRED_BEHAVIOR_ARGUMENTS if flag not in argument_values]
    forbidden_arguments = [flag for flag in ("-gaze", "-hogalign", "-tracked") if flag in argument_values]
    check(
        "source_behavior_arguments",
        {"required": list(REQUIRED_BEHAVIOR_ARGUMENTS), "forbidden": []},
        {"missing": missing_arguments, "forbidden_present": forbidden_arguments},
        not missing_arguments and not forbidden_arguments,
        "The source command must be aligned-image -fdir with 2D landmarks, pose and AUs only.",
    )

    declared_columns = manifest.get("behavior_source_columns")
    if declared_columns is None:
        declared_columns = manifest.get("downstream_behavior_columns")
    declared_tuple = tuple(str(value) for value in declared_columns) if isinstance(declared_columns, list) else ()
    check(
        "source_behavior_columns",
        list(SELECTED_BEHAVIOR_SOURCE_COLUMNS),
        list(declared_tuple),
        declared_tuple == SELECTED_BEHAVIOR_SOURCE_COLUMNS,
        "The source manifest must freeze the exact AU12/AU14/AU15 and pose rotation projection.",
    )

    expected_video_count = len(expected_video_ids)
    source_scope = str(manifest.get("run_scope", ""))
    check(
        "source_run_scope",
        "full_dataset",
        source_scope,
        source_scope == "full_dataset",
        "Debug extraction roots cannot authorize a full PB-P0 contract.",
    )
    for field, expected in (
        ("input_video_count", expected_video_count),
        ("selected_video_count", expected_video_count),
        ("input_frame_count", total_image_frames),
        ("max_videos", 0),
    ):
        observed = (
            _parse_nonnegative_int(manifest.get(field))
            if field == "max_videos"
            else _parse_positive_int(manifest.get(field))
        )
        passed = observed == expected
        check(
            f"source_{field}",
            expected,
            observed,
            passed,
            "Rich source scope/count fields must describe the exact frozen full dataset.",
        )

    source_min_success_ratio = _parse_finite_float(
        manifest.get("min_success_ratio")
    )
    check(
        "source_min_success_ratio",
        FROZEN_MIN_SUCCESS_RATIO,
        source_min_success_ratio,
        source_min_success_ratio == FROZEN_MIN_SUCCESS_RATIO,
        "The full rich extraction must use the frozen per-video success threshold.",
    )

    actual_source_contract_sha256 = (
        _sha256_file(source_video_contract) if source_video_contract.is_file() else ""
    )
    recorded_source_contract_sha256 = str(
        manifest.get("source_video_contract_sha256", "")
    ).lower()
    check(
        "source_video_contract_sha256",
        actual_source_contract_sha256,
        recorded_source_contract_sha256,
        bool(actual_source_contract_sha256)
        and recorded_source_contract_sha256 == actual_source_contract_sha256,
        "Rich head velocity must bind to the exact external FPS/frame-count contract.",
    )

    reference_integrity_sha = str(
        manifest.get("integrity_reference_manifest_sha256", "")
    ).lower()
    candidate_integrity_sha = str(
        manifest.get("integrity_candidate_manifest_sha256", "")
    ).lower()
    integrity_pair_valid = (
        bool(reference_integrity_sha)
        and reference_integrity_sha == candidate_integrity_sha
    )
    check(
        "source_image_integrity_manifest_pair",
        "non-empty equal reference/candidate SHA-256",
        {
            "reference": reference_integrity_sha,
            "candidate": candidate_integrity_sha,
        },
        integrity_pair_valid,
        "The extraction input must have an exact per-image integrity comparison.",
    )
    image_root_equivalent = _paths_cross_platform_equivalent(
        manifest.get("image_root", ""), image_root
    )
    image_binding_valid = image_root_equivalent and integrity_pair_valid
    check(
        "source_image_root_binding",
        str(image_root),
        {
            "manifest_image_root": manifest.get("image_root", ""),
            "cross_platform_equivalent": image_root_equivalent,
            "exact_integrity_pair": integrity_pair_valid,
        },
        image_binding_valid,
        "The current image root must be the manifest root (allowing D:\\ and /mnt/d spelling) and the reference/candidate integrity hashes must also be non-empty and equal.",
    )
    check(
        "source_integrity_status",
        "EXACT_PASS",
        manifest.get("integrity_status", ""),
        manifest.get("integrity_status") == "EXACT_PASS",
        "The aligned JPG integrity audit must pass exactly.",
    )
    for field in ("integrity_reference_count", "integrity_candidate_count"):
        observed = _parse_positive_int(manifest.get(field))
        check(
            f"source_{field}",
            total_image_frames,
            observed,
            observed == total_image_frames,
            "Image integrity counts must equal the exact P0 aligned-frame count.",
        )

    image_integrity_issues, image_integrity_fidelity, image_integrity_evidence = (
        _validate_current_image_integrity(
            manifest=manifest,
            image_root=image_root,
            project_root=project_root,
            total_image_frames=total_image_frames,
        )
    )
    issues.extend(image_integrity_issues)
    fidelity.extend(image_integrity_fidelity)
    evidence["current_image_integrity"] = image_integrity_evidence

    check(
        "source_gaze_requested",
        False,
        manifest.get("gaze_requested"),
        manifest.get("gaze_requested") is False,
        "The frozen rich extraction must not request gaze.",
    )
    check(
        "source_csv_timestamp_authorized",
        False,
        manifest.get("csv_timestamp_authorized_for_head_velocity"),
        manifest.get("csv_timestamp_authorized_for_head_velocity") is False,
        "Aligned image-directory timestamps may not drive head velocity.",
    )

    summary_path = source_run_manifest.parent / "extraction_summary.json"
    evidence["extraction_summary"] = str(summary_path)
    check(
        "source_extraction_summary_path",
        str(summary_path),
        manifest.get("extraction_summary", ""),
        _paths_cross_platform_equivalent(
            manifest.get("extraction_summary", ""), summary_path
        ),
        "The source run manifest must identify the adjacent final extraction summary.",
    )
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            loaded = json.loads(summary_path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            issues.append(_issue_row("rich_extraction_summary_invalid", str(exc)))
        else:
            if isinstance(loaded, dict):
                summary = loaded
            else:
                issues.append(
                    _issue_row("rich_extraction_summary_invalid", "summary is not a JSON object")
                )
    else:
        issues.append(
            _issue_row("rich_extraction_summary_missing", str(summary_path))
        )

    if summary:
        check(
            "extraction_summary_run_manifest_path",
            str(source_run_manifest),
            summary.get("run_manifest", ""),
            _paths_cross_platform_equivalent(
                summary.get("run_manifest", ""), source_run_manifest
            ),
            "The final extraction summary must point back to the exact source run manifest.",
        )
        summary_sha256 = _sha256_file(summary_path)
        recorded_summary_sha256 = str(
            manifest.get("extraction_summary_sha256", "")
        ).lower()
        check(
            "extraction_summary_sha256",
            summary_sha256,
            recorded_summary_sha256,
            bool(recorded_summary_sha256) and recorded_summary_sha256 == summary_sha256,
            "The final rewritten run manifest must bind the adjacent extraction summary.",
        )
        summary_checks = (
            ("status", "PASS"),
            ("run_scope", "full_dataset"),
            ("feature_profile", FROZEN_BEHAVIOR_PROFILE),
            ("video_count", expected_video_count),
            ("pass_count", expected_video_count),
            ("fail_count", 0),
            ("total_images", total_image_frames),
            ("total_csv_rows", total_openface_rows),
            ("schema_count", 1),
            ("schema_consistent", True),
            ("hog_file_count", 0),
            ("tracked_video_file_count", 0),
            ("regenerated_aligned_image_count", 0),
        )
        for field, expected in summary_checks:
            observed = summary.get(field)
            if isinstance(expected, int) and not isinstance(expected, bool):
                observed = _parse_nonnegative_int(observed)
            check(
                f"extraction_summary_{field}",
                expected,
                observed,
                observed == expected,
                "The adjacent final summary must describe a successful full-dataset rich extraction.",
            )
        summary_successes = _parse_nonnegative_int(summary.get("total_successes"))
        summary_success_ratio = _parse_finite_float(summary.get("success_ratio"))
        summary_min_success_ratio = _parse_finite_float(
            summary.get("min_required_success_ratio")
        )
        check(
            "extraction_summary_min_required_success_ratio",
            FROZEN_MIN_SUCCESS_RATIO,
            summary_min_success_ratio,
            summary_min_success_ratio == FROZEN_MIN_SUCCESS_RATIO,
            "The final extraction summary must preserve the frozen success threshold.",
        )
        success_counts_valid = (
            summary_successes is not None
            and 0 <= summary_successes <= total_openface_rows
            and summary_success_ratio is not None
            and summary_success_ratio >= FROZEN_MIN_SUCCESS_RATIO
            and abs(
                summary_success_ratio
                - (summary_successes / total_openface_rows if total_openface_rows else 0.0)
            )
            <= 1e-6
        )
        check(
            "extraction_summary_success_counts",
            f"0<=successes<=rows; ratio=successes/rows; ratio>={FROZEN_MIN_SUCCESS_RATIO}",
            {
                "total_successes": summary_successes,
                "success_ratio": summary_success_ratio,
                "min_required_success_ratio": summary_min_success_ratio,
            },
            success_counts_valid,
            "Final rich success counts and ratios must be internally consistent and meet the frozen extraction threshold.",
        )

    content_manifest_path = source_run_manifest.parent / "csv_content_manifest.csv"
    evidence["csv_content_manifest"] = str(content_manifest_path)
    for owner, recorded in (
        ("source", manifest.get("csv_content_manifest", "")),
        ("extraction_summary", summary.get("csv_content_manifest", "")),
    ):
        check(
            f"{owner}_csv_content_manifest_path",
            str(content_manifest_path),
            recorded,
            _paths_cross_platform_equivalent(recorded, content_manifest_path),
            "Final provenance paths must identify the adjacent CSV content manifest.",
        )
    if not content_manifest_path.is_file():
        issues.append(
            _issue_row("rich_csv_content_manifest_missing", str(content_manifest_path))
        )
    else:
        content_manifest_sha256 = _sha256_file(content_manifest_path)
        for owner, recorded in (
            ("summary", summary.get("csv_content_manifest_sha256", "")),
            ("source_manifest", manifest.get("csv_content_manifest_sha256", "")),
        ):
            recorded_value = str(recorded).lower()
            check(
                f"csv_content_manifest_sha256_{owner}",
                content_manifest_sha256,
                recorded_value,
                bool(recorded_value) and recorded_value == content_manifest_sha256,
                "Both final provenance documents must bind the CSV content manifest.",
            )

        with content_manifest_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as handle:
            reader = csv.DictReader(handle, skipinitialspace=True)
            fields = tuple(str(value).strip() for value in (reader.fieldnames or ()))
            missing_fields = [
                field for field in CSV_CONTENT_MANIFEST_FIELDS if field not in fields
            ]
            content_rows = list(reader) if not missing_fields else []
        if missing_fields:
            issues.append(
                _issue_row(
                    "rich_csv_content_manifest_invalid",
                    f"missing fields={missing_fields}",
                )
            )
        else:
            rows_by_video: dict[str, dict[str, Any]] = {}
            duplicate_ids: list[str] = []
            for row in content_rows:
                video_id = str(row.get("video_id", "")).strip()
                if video_id in rows_by_video:
                    duplicate_ids.append(video_id)
                else:
                    rows_by_video[video_id] = row
            if duplicate_ids or set(rows_by_video) != expected_video_ids:
                issues.append(
                    _issue_row(
                        "rich_csv_content_manifest_video_set_mismatch",
                        f"duplicates={sorted(set(duplicate_ids))} missing={sorted(expected_video_ids-set(rows_by_video))[:20]} extra={sorted(set(rows_by_video)-expected_video_ids)[:20]}",
                    )
                )
            for video_id in sorted(expected_video_ids & set(rows_by_video)):
                row = rows_by_video[video_id]
                actual = csv_evidence[video_id]
                comparisons = {
                    "status": ("PASS", str(row.get("status", "")).strip()),
                    "csv_rows": (actual["csv_rows"], _parse_positive_int(row.get("csv_rows"))),
                    "schema_sha256": (
                        actual["schema_sha256"],
                        str(row.get("schema_sha256", "")).strip().lower(),
                    ),
                    "csv_size_bytes": (
                        actual["csv_size_bytes"],
                        _parse_positive_int(row.get("csv_size_bytes")),
                    ),
                    "csv_sha256": (
                        actual["csv_sha256"],
                        str(row.get("csv_sha256", "")).strip().lower(),
                    ),
                }
                mismatches = {
                    key: {"expected": expected, "observed": observed}
                    for key, (expected, observed) in comparisons.items()
                    if expected != observed
                }
                if mismatches:
                    issues.append(
                        _issue_row(
                            "rich_csv_content_mismatch",
                            f"video_id={video_id} mismatches={mismatches}",
                            scope="video",
                            video_id=video_id,
                        )
                    )
    evidence["status"] = "PASS" if not issues else "BLOCKED"
    return issues, fidelity, evidence


def _target_stats_rows(statistics, reason: str = "") -> list[dict[str, Any]]:
    rows = []
    for target in SELECTED_TARGET_COLUMNS:
        group = "au" if target in AU_SOURCE_COLUMNS else "head"
        if statistics is None:
            rows.append(
                {
                    "physical_split": "train",
                    "target_group": group,
                    "target_name": target,
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "minimum": None,
                    "maximum": None,
                    "status": "BLOCKED",
                    "reason": reason,
                }
            )
            continue
        feature = statistics.features[target]
        rows.append(
            {
                "physical_split": statistics.physical_split,
                "target_group": group,
                "target_name": target,
                "count": feature.count,
                "mean": feature.mean,
                "std": feature.std,
                "minimum": feature.minimum,
                "maximum": feature.maximum,
                "status": "PASS",
                "reason": "physical-train valid-mask values only",
            }
        )
    return rows


def _write_report(
    path: Path,
    *,
    status: str,
    video_count: int,
    frame_count: int,
    audited_video_count: int,
    exact_join_video_count: int,
    issue_rows: list[dict[str, Any]],
    missing_columns: list[str],
    statistics_available: bool,
) -> Path:
    lines = [
        "# Privileged behavior PB-P0 contract audit",
        "",
        f"- Status: `{status}`",
        f"- Exact physical-split videos: {video_count}",
        f"- Aligned JPG frames: {frame_count}",
        f"- Rich-schema core audits completed: {audited_video_count}/{video_count}",
        f"- 100% structural frame joins: {exact_join_video_count}/{video_count}",
        f"- Physical-train normalization statistics available: {statistics_available}",
        f"- Blocking issues: {sum(row['severity'] == 'BLOCKER' for row in issue_rows)}",
        "",
        "## Selected target contract",
        "",
        "- AU: `AU12_r`, `AU14_r`, `AU15_r`.",
        "- Head dynamics: `d_pose_Rx_dt`, `d_pose_Ry_dt`, `d_pose_Rz_dt`.",
        "- Head velocity uses `t=(frame_id-1)/fps` from `source_video_contract.csv`; aligned OpenFace timestamps are ignored.",
        "- Gaze target projection/training/normalizer/loss use counts are all zero; this does not claim a generic CSV parser never tokenized an unselected physical column.",
        "- Historical raw-video OpenFace paths were never selected or dereferenced, and their feature files were not opened.",
    ]
    if missing_columns:
        lines.extend(
            [
                "",
                "## Blocking schema gap",
                "",
                "Missing selected source columns: "
                + ", ".join(f"`{column}`" for column in missing_columns)
                + ".",
                "Run the frozen aligned-JPG behavior extraction profile, then rerun this audit into a new output directory.",
            ]
        )
    lines.extend(
        [
            "",
            "## Authorization boundary",
            "",
            "This audit does not modify a model, create an auxiliary head, start training, or authorize PB-P1. Identity/task risk rows are explicit placeholders for the next offline gate.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_privileged_behavior_p0(
    *,
    dataset_split_file,
    image_root,
    openface_root,
    source_run_manifest,
    source_video_contract,
    output_dir,
    confidence_threshold: float = 0.8,
    max_pose_velocity_dt_seconds: float = 0.1,
    project_root=None,
) -> list[Path]:
    """Run the complete PB-P0 audit and emit ``BLOCKED`` artifacts on schema gaps."""

    dataset_split_file = Path(dataset_split_file).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    openface_root = Path(openface_root).expanduser().resolve()
    source_run_manifest = Path(source_run_manifest).expanduser().resolve()
    source_video_contract = Path(source_video_contract).expanduser().resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()

    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence_threshold must be within [0, 1]")
    if not math.isfinite(float(max_pose_velocity_dt_seconds)) or float(
        max_pose_velocity_dt_seconds
    ) <= 0.0:
        raise ValueError("max_pose_velocity_dt_seconds must be positive and finite")
    split_by_video, videos_by_split = _read_exact_split(dataset_split_file)
    output_dir = _safe_output_dir(Path(output_dir))
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    issue_rows: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    train_audits = []
    all_audits = []
    all_schema_hashes: Counter[str] = Counter()
    all_access_counts: Counter[str] = Counter()
    gaze_header_columns: set[str] = set()
    csv_evidence: dict[str, dict[str, Any]] = {}
    rich_schema_video_count = 0
    total_openface_rows = 0
    rich_provenance: dict[str, Any] = {
        "mode": "not_applicable_landmark_or_partial_schema",
        "status": "NOT_APPLICABLE",
    }
    total_image_frames = 0
    exact_join_video_count = 0
    missing_behavior_columns: set[str] = set()

    source_rows, source_contract_problems = _read_source_clock_contract(source_video_contract)
    for problem in source_contract_problems:
        issue_rows.append(_issue_row("invalid_source_video_contract", problem))

    expected_video_ids = set(split_by_video)
    source_video_ids = set(source_rows)
    if source_video_ids != expected_video_ids:
        missing = sorted(expected_video_ids - source_video_ids)
        extra = sorted(source_video_ids - expected_video_ids)
        issue_rows.append(
            _issue_row(
                "source_contract_video_set_mismatch",
                f"missing={missing[:20]} extra={extra[:20]}",
            )
        )
    fidelity_rows.append(
        _fidelity_row(
            "source_contract_exact_video_set",
            len(expected_video_ids),
            len(source_video_ids),
            "PASS" if source_video_ids == expected_video_ids else "FAIL",
            "The source frame clock must cover the exact frozen dataset split once.",
        )
    )

    manifest, manifest_problem = _read_source_run_manifest(source_run_manifest)
    if manifest_problem:
        issue_rows.append(_issue_row("invalid_source_run_manifest", manifest_problem))
    else:
        manifest_issues, manifest_fidelity = _validate_manifest_provenance(
            manifest, source_video_contract, image_root
        )
        issue_rows.extend(manifest_issues)
        fidelity_rows.extend(manifest_fidelity)

    if not image_root.is_dir():
        issue_rows.append(_issue_row("missing_image_root", str(image_root)))
        image_dir_ids: set[str] = set()
    else:
        image_dir_ids = {path.name for path in image_root.iterdir() if path.is_dir()}
    if image_dir_ids != expected_video_ids:
        issue_rows.append(
            _issue_row(
                "aligned_image_video_set_mismatch",
                f"missing={sorted(expected_video_ids-image_dir_ids)[:20]} "
                f"extra={sorted(image_dir_ids-expected_video_ids)[:20]}",
            )
        )
    fidelity_rows.append(
        _fidelity_row(
            "aligned_image_exact_video_set",
            len(expected_video_ids),
            len(image_dir_ids),
            "PASS" if image_dir_ids == expected_video_ids else "FAIL",
            "Only exact split identities are accepted; names are not normalized or remapped.",
        )
    )

    if not openface_root.is_dir():
        issue_rows.append(_issue_row("missing_openface_root", str(openface_root)))
        openface_csv_ids: set[str] = set()
    else:
        openface_csv_ids = {path.stem for path in openface_root.glob("*.csv") if path.is_file()}
    if openface_csv_ids != expected_video_ids:
        issue_rows.append(
            _issue_row(
                "openface_csv_video_set_mismatch",
                f"missing={sorted(expected_video_ids-openface_csv_ids)[:20]} "
                f"extra={sorted(openface_csv_ids-expected_video_ids)[:20]}",
            )
        )
    fidelity_rows.append(
        _fidelity_row(
            "openface_exact_video_set",
            len(expected_video_ids),
            len(openface_csv_ids),
            "PASS" if openface_csv_ids == expected_video_ids else "FAIL",
            "One exact per-video aligned-JPG OpenFace CSV is required.",
        )
    )

    frame_path = tables_dir / "behavior_frame_contract.csv"
    with frame_path.open("w", newline="", encoding="utf-8") as frame_handle:
        frame_writer = csv.DictWriter(frame_handle, fieldnames=FRAME_CONTRACT_FIELDS)
        frame_writer.writeheader()

        for physical_split in PHYSICAL_SPLITS:
            for video_id in videos_by_split[physical_split]:
                task_name = _task_from_video_id(video_id)
                local_issue_codes: list[str] = []
                if not task_name:
                    issue_rows.append(
                        _issue_row(
                            "unparseable_task_name",
                            "expected exact _Freeform_ or _Northwind_ token",
                            scope="video",
                            physical_split=physical_split,
                            video_id=video_id,
                        )
                    )
                    local_issue_codes.append("unparseable_task_name")

                source_row = source_rows.get(video_id)
                fps = None
                source_aligned_count = None
                source_raw_count = None
                if source_row is None:
                    local_issue_codes.append("missing_source_clock_row")
                    issue_rows.append(
                        _issue_row(
                            "missing_source_clock_row",
                            "video absent from source_video_contract.csv",
                            scope="video",
                            physical_split=physical_split,
                            video_id=video_id,
                            task_name=task_name,
                        )
                    )
                else:
                    fps = _parse_positive_float(source_row["raw_fps"])
                    source_aligned_count = _parse_positive_int(source_row["aligned_frame_count"])
                    source_raw_count = _parse_positive_int(source_row["raw_frame_count"])
                    source_checks = {
                        "split": source_row["split"] == physical_split,
                        "task_name": source_row["task_name"] == task_name,
                        "fps": fps is not None,
                        "aligned_frame_count": source_aligned_count is not None,
                        "raw_frame_count": source_raw_count is not None,
                        "frame_count_match": _truthy_contract_value(source_row["frame_count_match"]),
                        "contract_status": source_row["contract_status"].upper() == "PASS",
                    }
                    failed_source_checks = sorted(
                        name for name, passed in source_checks.items() if not passed
                    )
                    if failed_source_checks:
                        local_issue_codes.append("invalid_source_clock_row")
                        issue_rows.append(
                            _issue_row(
                                "invalid_source_clock_row",
                                f"failed checks={failed_source_checks}; recorded issues={source_row['issues']!r}",
                                scope="video",
                                physical_split=physical_split,
                                video_id=video_id,
                                task_name=task_name,
                            )
                        )

                image_frames, image_problems = _read_image_frames(image_root / video_id)
                for problem in image_problems:
                    issue_rows.append(
                        _issue_row(
                            "aligned_image_contract_error",
                            problem,
                            scope="video",
                            physical_split=physical_split,
                            video_id=video_id,
                            task_name=task_name,
                        )
                    )
                    local_issue_codes.append("aligned_image_contract_error")
                image_frame_ids = [frame_id for frame_id, _path in image_frames]
                unique_image_ids = set(image_frame_ids)
                total_image_frames += len(image_frames)
                consecutive_images = image_frame_ids == list(range(1, len(image_frame_ids) + 1))
                if image_frames and not consecutive_images:
                    issue_rows.append(
                        _issue_row(
                            "nonconsecutive_aligned_jpg_frames",
                            f"first={image_frame_ids[0]} last={image_frame_ids[-1]} count={len(image_frame_ids)}",
                            scope="video",
                            physical_split=physical_split,
                            video_id=video_id,
                            task_name=task_name,
                        )
                    )
                    local_issue_codes.append("nonconsecutive_aligned_jpg_frames")

                if source_aligned_count is not None and source_aligned_count != len(image_frames):
                    issue_rows.append(
                        _issue_row(
                            "source_aligned_frame_count_mismatch",
                            f"contract={source_aligned_count} jpg={len(image_frames)}",
                            scope="video",
                            physical_split=physical_split,
                            video_id=video_id,
                            task_name=task_name,
                        )
                    )
                    local_issue_codes.append("source_aligned_frame_count_mismatch")
                if source_raw_count is not None and source_raw_count != len(image_frames):
                    issue_rows.append(
                        _issue_row(
                            "source_raw_frame_count_mismatch",
                            f"contract={source_raw_count} jpg={len(image_frames)}",
                            scope="video",
                            physical_split=physical_split,
                            video_id=video_id,
                            task_name=task_name,
                        )
                    )
                    local_issue_codes.append("source_raw_frame_count_mismatch")

                openface_csv = openface_root / f"{video_id}.csv"
                projection = _read_openface_projection(openface_csv)
                total_openface_rows += int(projection["row_count"])
                if projection["schema_complete"]:
                    rich_schema_video_count += 1
                all_access_counts.update(projection["access_counts"])
                gaze_header_columns.update(projection["gaze_header_columns"])
                if projection["fieldnames"]:
                    schema_text = ",".join(projection["fieldnames"])
                    schema_sha256 = _sha256_text(schema_text)
                    all_schema_hashes[schema_sha256] += 1
                else:
                    schema_sha256 = ""
                csv_evidence[video_id] = {
                    "csv_rows": int(projection["row_count"]),
                    "schema_sha256": schema_sha256,
                    "csv_size_bytes": (
                        openface_csv.stat().st_size
                        if projection["schema_complete"] and openface_csv.is_file()
                        else 0
                    ),
                    "csv_sha256": (
                        _sha256_file(openface_csv)
                        if projection["schema_complete"] and openface_csv.is_file()
                        else ""
                    ),
                }
                missing_behavior_columns.update(projection["missing_behavior_columns"])
                for problem in projection["issues"]:
                    issue_rows.append(
                        _issue_row(
                            "openface_contract_error",
                            problem,
                            scope="video",
                            physical_split=physical_split,
                            video_id=video_id,
                            task_name=task_name,
                        )
                    )
                    local_issue_codes.append("openface_contract_error")
                if projection["missing_behavior_columns"]:
                    detail = ",".join(projection["missing_behavior_columns"])
                    issue_rows.append(
                        _issue_row(
                            "missing_behavior_columns",
                            detail,
                            scope="video",
                            physical_split=physical_split,
                            video_id=video_id,
                            task_name=task_name,
                        )
                    )
                    local_issue_codes.append("missing_behavior_columns")

                openface_frame_ids = projection["frame_ids"]
                unique_openface_ids = set(openface_frame_ids)
                joined_ids = unique_image_ids & unique_openface_ids
                exact_join = (
                    bool(image_frames)
                    and len(unique_image_ids) == len(image_frames)
                    and len(unique_openface_ids) == len(openface_frame_ids)
                    and unique_image_ids == unique_openface_ids
                )
                if exact_join:
                    exact_join_video_count += 1
                else:
                    issue_rows.append(
                        _issue_row(
                            "exact_frame_join_failed",
                            f"missing_openface={sorted(unique_image_ids-unique_openface_ids)[:20]} "
                            f"extra_openface={sorted(unique_openface_ids-unique_image_ids)[:20]}",
                            scope="video",
                            physical_split=physical_split,
                            video_id=video_id,
                            task_name=task_name,
                        )
                    )
                    local_issue_codes.append("exact_frame_join_failed")

                expected_frames = [
                    FrameReference(
                        video_id=video_id,
                        task_name=task_name,
                        frame=frame_id,
                        timestamp=((frame_id - 1) / fps if fps is not None else None),
                    )
                    for frame_id, _path in image_frames
                ] if task_name else []
                audit = None
                if projection["schema_complete"] and expected_frames:
                    try:
                        audit = audit_privileged_behavior_contract(
                            projection["rows"],
                            fieldnames=projection["fieldnames"],
                            video_id=video_id,
                            task_name=task_name,
                            physical_split=physical_split,
                            expected_frames=expected_frames,
                            confidence_threshold=confidence_threshold,
                            max_pose_velocity_dt_seconds=max_pose_velocity_dt_seconds,
                        )
                    except PrivilegedBehaviorContractError as exc:
                        issue_rows.append(
                            _issue_row(
                                "behavior_core_contract_failed",
                                str(exc),
                                scope="video",
                                physical_split=physical_split,
                                video_id=video_id,
                                task_name=task_name,
                            )
                        )
                        local_issue_codes.append("behavior_core_contract_failed")
                    else:
                        all_audits.append(audit)
                        if physical_split == "train":
                            train_audits.append(audit)

                local_status = "PASS" if not local_issue_codes and audit is not None else "BLOCKED"
                audit_by_frame = (
                    {frame.reference.frame: frame for frame in audit.frames} if audit else {}
                )
                for frame_id, image_path in image_frames:
                    behavior_frame = audit_by_frame.get(frame_id)
                    frame_row = {
                        "physical_split": physical_split,
                        "video_id": video_id,
                        "task_name": task_name,
                        "frame_id": frame_id,
                        "image_filename": image_path.name,
                        "image_path": str(image_path),
                        "source_fps": fps,
                        "source_timestamp_seconds": (
                            (frame_id - 1) / fps if fps is not None else None
                        ),
                        "openface_csv": str(openface_csv),
                        "openface_frame_present": frame_id in unique_openface_ids,
                        "exact_frame_join": exact_join and frame_id in unique_openface_ids,
                        "quality_valid": behavior_frame.quality_valid if behavior_frame else None,
                        "au_valid": behavior_frame.au_valid if behavior_frame else None,
                        "pose_valid": behavior_frame.pose_valid if behavior_frame else None,
                        "pose_velocity_candidate": (
                            behavior_frame.pose_velocity_candidate if behavior_frame else None
                        ),
                        "pose_velocity_valid": (
                            behavior_frame.pose_velocity_valid if behavior_frame else None
                        ),
                        "pose_velocity_dt_seconds": (
                            behavior_frame.pose_velocity_dt if behavior_frame else None
                        ),
                        "invalid_reasons": (
                            ";".join(behavior_frame.invalid_reasons) if behavior_frame else ""
                        ),
                        "status": local_status,
                    }
                    if behavior_frame:
                        frame_row.update(dict(zip(AU_SOURCE_COLUMNS, behavior_frame.au_values)))
                        frame_row.update(
                            dict(zip(POSE_VELOCITY_TARGET_COLUMNS, behavior_frame.pose_velocity))
                        )
                    frame_writer.writerow(
                        {field: _format(frame_row.get(field)) for field in FRAME_CONTRACT_FIELDS}
                    )

                coverage = audit.coverage if audit else None
                coverage_rows.append(
                    {
                        "physical_split": physical_split,
                        "video_id": video_id,
                        "task_name": task_name,
                        "image_frame_count": len(image_frames),
                        "openface_row_count": projection["row_count"],
                        "joined_frame_count": len(joined_ids),
                        "frame_join_ratio": (
                            len(joined_ids) / len(unique_image_ids) if unique_image_ids else 0.0
                        ),
                        "quality_valid_count": coverage.quality_valid_count if coverage else None,
                        "quality_valid_ratio": coverage.quality_valid_ratio if coverage else None,
                        "au_valid_count": coverage.au_valid_count if coverage else None,
                        "au_valid_ratio": coverage.au_valid_ratio if coverage else None,
                        "pose_valid_count": coverage.pose_valid_count if coverage else None,
                        "pose_valid_ratio": coverage.pose_valid_ratio if coverage else None,
                        "head_candidate_count": (
                            coverage.pose_velocity_candidate_count if coverage else None
                        ),
                        "head_valid_count": coverage.pose_velocity_valid_count if coverage else None,
                        "head_valid_ratio": (
                            coverage.pose_velocity_valid_ratio if coverage else None
                        ),
                        "head_missing_timestamp_pair_count": (
                            coverage.pose_velocity_missing_timestamp_pair_count
                            if coverage
                            else None
                        ),
                        "head_illegal_dt_pair_count": (
                            coverage.pose_velocity_illegal_dt_pair_count if coverage else None
                        ),
                        "head_nonconsecutive_pair_count": (
                            coverage.pose_velocity_nonconsecutive_pair_count if coverage else None
                        ),
                        "schema_status": "PASS" if projection["schema_complete"] else "BLOCKED",
                        "missing_behavior_columns": ";".join(
                            projection["missing_behavior_columns"]
                        ),
                        "status": local_status,
                        "issues": ";".join(dict.fromkeys(local_issue_codes)),
                    }
                )

                fidelity_status = (
                    "PASS"
                    if fps is not None
                    and consecutive_images
                    and exact_join
                    and source_aligned_count == len(image_frames)
                    and source_raw_count == len(image_frames)
                    else "FAIL"
                )
                fidelity_rows.append(
                    _fidelity_row(
                        "per_video_full_rate_source_clock",
                        "JPG ids 1..N; raw=aligned=N; exact OpenFace join; t=(frame-1)/fps",
                        {
                            "fps": fps,
                            "jpg_frames": len(image_frames),
                            "source_aligned_frames": source_aligned_count,
                            "source_raw_frames": source_raw_count,
                            "joined_frames": len(joined_ids),
                        },
                        fidelity_status,
                        "Raw OpenFace feature values are not consulted for this fidelity check.",
                        scope="video",
                        physical_split=physical_split,
                        video_id=video_id,
                        task_name=task_name,
                    )
                )

    if rich_schema_video_count == len(expected_video_ids):
        strict_issues, strict_fidelity, rich_provenance = _validate_rich_source_provenance(
            manifest=manifest,
            source_run_manifest=source_run_manifest,
            source_video_contract=source_video_contract,
            image_root=image_root,
            project_root=project_root,
            expected_video_ids=expected_video_ids,
            total_image_frames=total_image_frames,
            total_openface_rows=total_openface_rows,
            csv_evidence=csv_evidence,
        )
        issue_rows.extend(strict_issues)
        fidelity_rows.extend(strict_fidelity)
    else:
        fidelity_rows.append(
            _fidelity_row(
                "strict_rich_provenance_gate",
                len(expected_video_ids),
                rich_schema_video_count,
                "NOT_APPLICABLE",
                "Legacy landmark-only or partial-rich inputs remain schema-BLOCKED; strict rich extractor provenance is enforced only when every exact split video has the complete selected schema.",
            )
        )

    statistics = None
    statistics_reason = "rich-schema exact contract is incomplete"
    expected_train_audit_count = len(videos_by_split["train"])
    if len(train_audits) == expected_train_audit_count:
        try:
            statistics = fit_train_only_statistics(
                train_audits,
                physical_split="train",
                groups=("au", "head"),
            )
        except PrivilegedBehaviorContractError as exc:
            statistics_reason = str(exc)
            issue_rows.append(_issue_row("train_only_statistics_failed", str(exc)))
        else:
            statistics_reason = ""
    else:
        statistics_reason = (
            f"physical train audits available={len(train_audits)} "
            f"expected={expected_train_audit_count}"
        )
        issue_rows.append(
            _issue_row("train_only_statistics_unavailable", statistics_reason)
        )

    target_stats_rows = _target_stats_rows(statistics, statistics_reason)
    identity_task_rows = [
        {
            "risk_type": risk_type,
            "target_group": group,
            "target_names": ";".join(
                AU_SOURCE_COLUMNS if group == "au" else POSE_VELOCITY_TARGET_COLUMNS
            ),
            "physical_split_policy": "future probe calibration uses physical train only",
            "status": "NOT_RUN_P0_PLACEHOLDER",
            "probe_value_access_count": 0,
            "reason": "PB-P0 freezes data/provenance only; no identity/task probe is fitted here.",
        }
        for risk_type in ("subject_identity", "task")
        for group in ("au", "head")
    ]

    blocking_issue_count = sum(row["severity"] == "BLOCKER" for row in issue_rows)
    status = "PASS" if blocking_issue_count == 0 and statistics is not None else "BLOCKED"
    selected_manifest = {
        "audit": "PB-P0 selected privileged behavior target contract",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "training_authorized": False,
        "provenance": {
            "dataset_split_file": str(dataset_split_file),
            "dataset_split_sha256": _sha256_file(dataset_split_file),
            "source_run_manifest": str(source_run_manifest),
            "source_run_manifest_sha256": (
                _sha256_file(source_run_manifest) if source_run_manifest.is_file() else ""
            ),
            "source_video_contract": str(source_video_contract),
            "source_video_contract_sha256": (
                _sha256_file(source_video_contract) if source_video_contract.is_file() else ""
            ),
        },
        "physical_split_contract": list(PHYSICAL_SPLITS),
        "source_columns": list(SELECTED_BEHAVIOR_SOURCE_COLUMNS),
        "target_columns": list(SELECTED_TARGET_COLUMNS),
        "groups": {
            "au": list(AU_SOURCE_COLUMNS),
            "head": list(POSE_VELOCITY_TARGET_COLUMNS),
        },
        "head_source_rotation_columns": list(POSE_ROTATION_SOURCE_COLUMNS),
        "frame_join": "exact video_id + task_name + JPG frame_id; offset forbidden",
        "source_clock": {
            "formula": "t=(frame_id-1)/fps",
            "fps_source": "source_video_contract.csv raw_fps",
            "openface_csv_timestamp_authorized": False,
            "full_rate_difference_before_sampling": True,
            "max_pose_velocity_dt_seconds": float(max_pose_velocity_dt_seconds),
        },
        "quality_mask": {
            "success_required": True,
            "confidence_threshold": float(confidence_threshold),
        },
        "normalization": {
            "physical_train_only": True,
            "available": statistics is not None,
            "validation_or_test_access_count": 0,
        },
        "rich_provenance": rich_provenance,
        "gaze": {
            "target_count": 0,
            "value_access_count": 0,
            "value_projection_count": 0,
            "normalizer_access_count": 0,
            "loss_access_count": 0,
            "access_semantics": "projection/use counts only; no claim that a generic CSV parser did not tokenize physically present unselected columns",
            "header_columns_seen_only": sorted(gaze_header_columns),
            "output_directories": [],
        },
        "raw_openface": {
            "feature_file_open_count": 0,
            "feature_value_access_count": 0,
            "source_contract_path_field_access_count": 0,
            "access_semantics": "no raw OpenFace path is selected, dereferenced, opened or used; source-contract CSV tokenization is not counted as feature access",
        },
        "openface_value_access_counts": dict(sorted(all_access_counts.items())),
        "schema_sha256_video_counts": dict(sorted(all_schema_hashes.items())),
        "missing_behavior_columns": sorted(missing_behavior_columns),
        "identity_task_risk_status": "NOT_RUN_P0_PLACEHOLDER",
        "next_gate": (
            "run aligned-JPG rich AU/pose extraction and rerun PB-P0"
            if status == "BLOCKED"
            else "complete offline identity/task risk audit before PB-P1"
        ),
    }

    coverage_path = _write_csv(
        tables_dir / "behavior_group_coverage.csv",
        coverage_rows,
        GROUP_COVERAGE_FIELDS,
    )
    target_stats_path = _write_csv(
        tables_dir / "behavior_target_stats.csv",
        target_stats_rows,
        TARGET_STATS_FIELDS,
    )
    issues_path = _write_csv(
        tables_dir / "behavior_contract_issues.csv",
        issue_rows,
        ISSUE_FIELDS,
    )
    fidelity_path = _write_csv(
        tables_dir / "behavior_source_fidelity.csv",
        fidelity_rows,
        SOURCE_FIDELITY_FIELDS,
    )
    identity_task_path = _write_csv(
        tables_dir / "behavior_identity_task_risk.csv",
        identity_task_rows,
        IDENTITY_TASK_RISK_FIELDS,
    )
    selected_manifest_path = _write_json(
        output_dir / "selected_target_manifest.json", selected_manifest
    )
    report_path = _write_report(
        reports_dir / "behavior_contract_report.md",
        status=status,
        video_count=len(split_by_video),
        frame_count=total_image_frames,
        audited_video_count=len(all_audits),
        exact_join_video_count=exact_join_video_count,
        issue_rows=issue_rows,
        missing_columns=sorted(missing_behavior_columns),
        statistics_available=statistics is not None,
    )

    # These names are intentionally never created.  Keep the assertion close
    # to manifest finalization so a future refactor cannot silently materialize
    # a gaze cache under this audit root.
    forbidden_gaze_dirs = [
        output_dir / name for name in ("gaze", "gaze_features", "gaze_targets")
    ]
    if any(path.exists() and (path.is_file() or any(path.iterdir())) for path in forbidden_gaze_dirs):
        raise RuntimeError("PB-P0 gaze output directories must be absent or empty")

    output_files = [
        frame_path,
        coverage_path,
        target_stats_path,
        issues_path,
        fidelity_path,
        identity_task_path,
        selected_manifest_path,
        report_path,
    ]
    git_commit, git_commit_available = _git_capture(project_root, ["rev-parse", "HEAD"])
    git_branch, git_branch_available = _git_capture(
        project_root, ["branch", "--show-current"]
    )
    git_status_short, git_status_available = _git_capture(
        project_root, ["status", "--short"]
    )
    git_status_interpretation = (
        "captured; empty means clean"
        if git_status_available
        else "unavailable; do not interpret empty as clean"
    )
    contract_core_path = Path(__file__).with_name("privileged_behavior_contract.py").resolve()
    code_root = Path(__file__).resolve().parents[2]
    cli_path = (code_root / "scripts" / "audit_privileged_behavior_contract.py").resolve()
    run_manifest = {
        "audit": "PB-P0 aligned-JPG privileged behavior data contract",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "training_authorized": False,
        "command_line": " ".join(shlex.quote(value) for value in sys.argv),
        "git_commit": git_commit,
        "git_commit_available": git_commit_available,
        "git_branch": git_branch,
        "git_branch_available": git_branch_available,
        "git_status_short": git_status_short,
        "git_status_available": git_status_available,
        "git_status_interpretation": git_status_interpretation,
        "inputs": {
            "dataset_split_file": {
                "path": str(dataset_split_file),
                "sha256": _sha256_file(dataset_split_file),
            },
            "image_root": str(image_root),
            "openface_root": str(openface_root),
            "source_run_manifest": {
                "path": str(source_run_manifest),
                "sha256": _sha256_file(source_run_manifest)
                if source_run_manifest.is_file()
                else "",
            },
            "source_video_contract": {
                "path": str(source_video_contract),
                "sha256": _sha256_file(source_video_contract)
                if source_video_contract.is_file()
                else "",
                "accessed_columns": list(SOURCE_CLOCK_COLUMNS),
                "raw_video_path_accessed": False,
                "raw_openface_csv_path_accessed": False,
            },
        },
        "parameters": {
            "confidence_threshold": float(confidence_threshold),
            "max_pose_velocity_dt_seconds": float(max_pose_velocity_dt_seconds),
        },
        "counts": {
            "video_count": len(split_by_video),
            "frame_count": total_image_frames,
            "core_audited_video_count": len(all_audits),
            "rich_schema_video_count": rich_schema_video_count,
            "exact_join_video_count": exact_join_video_count,
            "blocking_issue_count": blocking_issue_count,
        },
        "access_contract": {
            "selected_source_columns": list(SELECTED_BEHAVIOR_SOURCE_COLUMNS),
            "selected_target_columns": list(SELECTED_TARGET_COLUMNS),
            "gaze_target_count": 0,
            "gaze_value_access_count": 0,
            "gaze_value_projection_count": 0,
            "gaze_normalizer_access_count": 0,
            "gaze_loss_access_count": 0,
            "raw_openface_feature_file_open_count": 0,
            "raw_openface_feature_value_access_count": 0,
            "validation_test_normalization_access_count": 0,
            "count_semantics": "projection/use/open counts; generic CSV tokenization of unselected columns is not claimed to be zero",
        },
        "read_only_inputs": True,
        "source_data_modified": False,
        "model_modified": False,
        "training_started": False,
        "python": sys.version,
        "platform": platform.platform(),
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "implementation_files": {
            "p0_orchestration": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "behavior_contract_core": {
                "path": str(contract_core_path),
                "sha256": _sha256_file(contract_core_path),
            },
            "cli": {
                "path": str(cli_path),
                "sha256": _sha256_file(cli_path) if cli_path.is_file() else "",
            },
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256_file(path)}
            for path in output_files
        },
    }
    run_manifest_path = _write_json(output_dir / "run_manifest.json", run_manifest)
    return [*output_files, run_manifest_path]
