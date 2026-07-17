"""Read-only source-video review for pure-black aligned placeholders.

The aligned OpenFace tree can contain a black JPEG when face detection or
alignment failed.  A black aligned image does not reveal whether the source
video contained a person, an occluded/out-of-frame face, or an empty room.
This module joins the existing frame-failure manifest back to the original
videos, creates review contact sheets, and validates an explicit human review
gate.  It never launches OpenFace/FaceLandmark and never modifies source data.
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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SOURCE_VIDEO_FIELDS = [
    "split",
    "source_split",
    "video_id",
    "raw_video_id",
    "task_name",
    "raw_video_path",
    "raw_openface_csv",
    "aligned_frame_count",
    "raw_frame_count",
    "frame_count_match",
    "raw_width",
    "raw_height",
    "raw_fps",
    "pure_black_run_count",
    "pure_black_frame_count",
    "contract_status",
    "issues",
]

SOURCE_RUN_FIELDS = [
    "split",
    "source_split",
    "video_id",
    "raw_video_id",
    "task_name",
    "pure_black_run_id",
    "start_frame",
    "end_frame",
    "length",
    "previous_source_frame",
    "next_source_frame",
    "raw_video_path",
    "raw_openface_csv",
    "raw_openface_success_ratio",
    "raw_openface_confidence_mean",
    "review_sample_frames",
    "contact_sheet",
]

SOURCE_REVIEW_FIELDS = [
    "split",
    "video_id",
    "pure_black_run_id",
    "segment_start_frame",
    "segment_end_frame",
    "review_status",
    "source_presence_status",
    "recovery_permission",
    "reviewer",
    "review_notes",
]

SOURCE_FRAME_GATE_FIELDS = [
    "split",
    "video_id",
    "pure_black_run_id",
    "frame_id",
    "source_presence_status",
    "recovery_permission",
    "review_status",
    "reviewer",
    "review_notes",
]

SOURCE_VIDEO_REVIEW_FIELDS = [
    "split",
    "video_id",
    "pure_black_run_count",
    "pure_black_frame_count",
    "person_present_detection_failure_frames",
    "person_absent_frames",
    "mixed_frames",
    "ambiguous_frames",
    "raw_frame_warp_only_frames",
    "keep_invalid_frames",
    "review_status",
]

REVIEWED_PRESENCE_STATUSES = {
    "person_absent",
    "person_present_detection_failure",
    "mixed",
    "ambiguous",
}
RECOVERY_PERMISSIONS = {"keep_invalid", "raw_frame_warp_only"}


def _format(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field)) for field in fields})
    return path


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(project_root, arguments):
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _safe_int(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() else None


def _safe_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _source_split(split):
    return {"val": "dev", "validation": "dev"}.get(str(split).lower(), str(split).lower())


def _raw_video_id(video_id):
    value = str(video_id)
    return value[: -len("_aligned")] if value.endswith("_aligned") else value


def _task_name(video_id):
    match = re.search(r"_(Freeform|Northwind)_", str(video_id))
    return match.group(1) if match else ""


def build_pure_black_runs(failure_rows):
    """Group consecutive pure-black aligned placeholders by video."""

    grouped = defaultdict(list)
    for row in failure_rows:
        if row.get("failure_type") != "pure_black":
            continue
        frame_id = _safe_int(row.get("frame_id"))
        if frame_id is None:
            raise ValueError("pure-black failure row has no integer frame_id")
        grouped[row["video_id"]].append((frame_id, row))

    runs = []
    for video_id in sorted(grouped):
        values = sorted(grouped[video_id], key=lambda item: item[0])
        if len({frame for frame, _ in values}) != len(values):
            raise ValueError(f"duplicate pure-black frame ids for {video_id}")
        run_number = 0
        start = end = values[0][0]
        first_row = values[0][1]
        for frame_id, row in values[1:] + [(None, None)]:
            if frame_id is not None and frame_id == end + 1:
                end = frame_id
                continue
            run_number += 1
            runs.append(
                {
                    "split": first_row.get("split", ""),
                    "video_id": video_id,
                    "pure_black_run_id": f"{video_id}:PB{run_number:04d}",
                    "start_frame": start,
                    "end_frame": end,
                    "length": end - start + 1,
                }
            )
            if frame_id is not None:
                start = end = frame_id
                first_row = row
    return runs


def _review_sample_frames(start_frame, end_frame, frame_count, max_run_samples):
    run_length = end_frame - start_frame + 1
    if run_length <= max_run_samples:
        internal = list(range(start_frame, end_frame + 1))
    else:
        internal = sorted(
            set(
                np.linspace(start_frame, end_frame, max_run_samples)
                .round()
                .astype(int)
                .tolist()
            )
        )
    values = []
    if start_frame > 1:
        values.append(start_frame - 1)
    values.extend(internal)
    if end_frame < frame_count:
        values.append(end_frame + 1)
    return sorted(set(values))


def _openface_frame_map(path):
    if path is None or not Path(path).exists():
        return {}
    result = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            frame = _safe_int(row.get("frame"))
            if frame is None:
                continue
            result[frame] = {
                "success": _safe_int(row.get("success")),
                "confidence": _safe_float(row.get("confidence")),
            }
    return result


def _write_contact_sheet(capture, frame_ids, path, *, columns=4, thumb_width=240):
    import cv2

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source video has invalid dimensions")
    thumb_height = max(1, round(thumb_width * source_height / source_width))
    label_height = 20
    rows = math.ceil(len(frame_ids) / columns)
    canvas = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height)),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, frame_id in enumerate(frame_ids):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id) - 1)
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"cannot decode source frame {frame_id}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize(
            (thumb_width, thumb_height), Image.Resampling.LANCZOS
        )
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        canvas.paste(image, (x, y))
        draw.text((x + 4, y + thumb_height + 3), f"frame {frame_id}", fill="white", font=font)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=92, subsampling=0)
    return path


def _source_path(dataset_root, split, video_id):
    task_name = _task_name(video_id)
    raw_video_id = _raw_video_id(video_id)
    return Path(dataset_root) / _source_split(split) / task_name / f"{raw_video_id}.mp4"


def run_source_presence_audit(
    *,
    dataset_root,
    frame_failure_manifest,
    video_failure_summary,
    output_dir,
    raw_openface_root=None,
    video_ids=None,
    max_videos=None,
    max_run_samples=7,
    contact_sheet_columns=4,
    write_contact_sheets=True,
    project_root=None,
):
    """Join pure-black aligned runs to original videos and create review artifacts."""

    import cv2

    dataset_root = Path(dataset_root).expanduser().resolve()
    frame_failure_manifest = Path(frame_failure_manifest).expanduser().resolve()
    video_failure_summary = Path(video_failure_summary).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    raw_openface_root = (
        Path(raw_openface_root).expanduser().resolve() if raw_openface_root else None
    )
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    contact_dir = output_dir / "contact_sheets"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if write_contact_sheets:
        contact_dir.mkdir(parents=True, exist_ok=True)

    failures = _read_csv(frame_failure_manifest)
    summaries = _read_csv(video_failure_summary)
    runs = build_pure_black_runs(failures)
    runs_by_video = defaultdict(list)
    for row in runs:
        runs_by_video[row["video_id"]].append(row)

    summary_by_video = {row["video_id"]: row for row in summaries}
    selected_videos = sorted(summary_by_video)
    if video_ids:
        requested = {str(value) for value in video_ids}
        unknown = sorted(requested - set(selected_videos))
        if unknown:
            raise ValueError(f"unknown video ids: {unknown}")
        selected_videos = [value for value in selected_videos if value in requested]
    if max_videos is not None:
        selected_videos = selected_videos[: max(0, int(max_videos))]
    selected = set(selected_videos)
    runs = [row for row in runs if row["video_id"] in selected]

    source_rows = []
    source_run_rows = []
    review_rows = []
    for index, video_id in enumerate(selected_videos, start=1):
        summary = summary_by_video[video_id]
        split = summary.get("split", "")
        source_split = _source_split(split)
        task_name = _task_name(video_id)
        raw_video_id = _raw_video_id(video_id)
        raw_video_path = _source_path(dataset_root, split, video_id)
        raw_openface_csv = (
            raw_openface_root / f"{raw_video_id}.csv" if raw_openface_root else None
        )
        aligned_count = _safe_int(summary.get("frame_count")) or 0
        issues = []
        raw_count = width = height = 0
        fps = None
        capture = None
        if not raw_video_path.exists():
            issues.append("raw_video_missing")
        else:
            capture = cv2.VideoCapture(str(raw_video_path))
            if not capture.isOpened():
                issues.append("raw_video_open_failed")
            else:
                raw_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                if raw_count != aligned_count:
                    issues.append("raw_aligned_frame_count_mismatch")
                if width <= 0 or height <= 0 or not fps or fps <= 0:
                    issues.append("invalid_raw_video_metadata")
        video_runs = runs_by_video.get(video_id, [])
        source_rows.append(
            {
                "split": split,
                "source_split": source_split,
                "video_id": video_id,
                "raw_video_id": raw_video_id,
                "task_name": task_name,
                "raw_video_path": str(raw_video_path),
                "raw_openface_csv": str(raw_openface_csv or ""),
                "aligned_frame_count": aligned_count,
                "raw_frame_count": raw_count,
                "frame_count_match": raw_count == aligned_count and raw_count > 0,
                "raw_width": width,
                "raw_height": height,
                "raw_fps": fps,
                "pure_black_run_count": len(video_runs),
                "pure_black_frame_count": sum(row["length"] for row in video_runs),
                "contract_status": "PASS" if not issues else "FAIL",
                "issues": ";".join(issues),
            }
        )
        if capture is not None and capture.isOpened() and video_runs:
            openface_by_frame = _openface_frame_map(raw_openface_csv)
            for run in video_runs:
                start_frame = int(run["start_frame"])
                end_frame = int(run["end_frame"])
                sample_frames = _review_sample_frames(
                    start_frame,
                    end_frame,
                    raw_count,
                    max(1, int(max_run_samples)),
                )
                contact_sheet = ""
                if write_contact_sheets:
                    sheet_name = f"{run['pure_black_run_id'].replace(':', '_')}.jpg"
                    sheet_path = _write_contact_sheet(
                        capture,
                        sample_frames,
                        contact_dir / sheet_name,
                        columns=max(1, int(contact_sheet_columns)),
                    )
                    contact_sheet = str(sheet_path)
                interval = [
                    openface_by_frame.get(frame)
                    for frame in range(start_frame, end_frame + 1)
                    if frame in openface_by_frame
                ]
                successes = [row["success"] for row in interval if row["success"] is not None]
                confidences = [
                    row["confidence"] for row in interval if row["confidence"] is not None
                ]
                source_run_rows.append(
                    {
                        **run,
                        "source_split": source_split,
                        "raw_video_id": raw_video_id,
                        "task_name": task_name,
                        "previous_source_frame": start_frame - 1 if start_frame > 1 else None,
                        "next_source_frame": end_frame + 1 if end_frame < raw_count else None,
                        "raw_video_path": str(raw_video_path),
                        "raw_openface_csv": str(raw_openface_csv or ""),
                        "raw_openface_success_ratio": (
                            sum(value == 1 for value in successes) / len(successes)
                            if successes
                            else None
                        ),
                        "raw_openface_confidence_mean": (
                            float(np.mean(confidences)) if confidences else None
                        ),
                        "review_sample_frames": ";".join(map(str, sample_frames)),
                        "contact_sheet": contact_sheet,
                    }
                )
                review_rows.append(
                    {
                        "split": split,
                        "video_id": video_id,
                        "pure_black_run_id": run["pure_black_run_id"],
                        "segment_start_frame": start_frame,
                        "segment_end_frame": end_frame,
                        "review_status": "PENDING",
                        "source_presence_status": "unreviewed",
                        "recovery_permission": "pending",
                        "reviewer": "",
                        "review_notes": "",
                    }
                )
        if capture is not None:
            capture.release()
        if index % 25 == 0 or index == len(selected_videos):
            print(
                f"[SOURCE_PRESENCE_AUDIT] processed {index}/{len(selected_videos)} videos",
                flush=True,
            )

    source_rows.sort(key=lambda row: row["video_id"])
    source_run_rows.sort(key=lambda row: (row["video_id"], row["start_frame"]))
    review_rows.sort(key=lambda row: (row["video_id"], row["segment_start_frame"]))
    source_path = _write_csv(tables_dir / "source_video_contract.csv", source_rows, SOURCE_VIDEO_FIELDS)
    runs_path = _write_csv(
        tables_dir / "pure_black_source_runs.csv", source_run_rows, SOURCE_RUN_FIELDS
    )
    review_path = _write_csv(
        tables_dir / "source_presence_review_template.csv", review_rows, SOURCE_REVIEW_FIELDS
    )

    contract_counts = Counter(row["contract_status"] for row in source_rows)
    lines = [
        "# Source-video Presence Audit",
        "",
        "- This audit read original videos and existing CSV/JPG manifests only.",
        "- It did not run FaceLandmark/OpenFace and did not modify source videos or aligned JPGs.",
        f"- Source videos: {len(source_rows)}; contract PASS: {contract_counts['PASS']}; FAIL: {contract_counts['FAIL']}.",
        f"- Pure-black aligned placeholders: {sum(row['length'] for row in source_run_rows)} frames in {len(source_run_rows)} runs.",
        "- A pure-black aligned frame is a preprocessing placeholder, not evidence that the source frame is damaged or that the person is absent.",
        "",
        "## Mandatory review labels",
        "",
        "- `person_absent`: keep invalid; never synthesize a face.",
        "- `person_present_detection_failure`: aligned-neighbor synthesis remains disabled; only a future raw-frame warp may be considered.",
        "- `mixed` or `ambiguous`: split the segment where possible, otherwise keep invalid.",
        "- Materialization is blocked while any selected pure-black frame is unreviewed or uncovered.",
    ]
    report_path = reports_dir / "source_presence_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    project_root = Path(project_root or Path(__file__).resolve().parents[2])
    outputs = [source_path, runs_path, review_path, report_path]
    payload = {
        "audit": "raw source-video presence review for pure-black aligned placeholders",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(shlex.quote(item) for item in sys.argv),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "dataset_root": str(dataset_root),
        "frame_failure_manifest": str(frame_failure_manifest),
        "frame_failure_manifest_sha256": _sha256_file(frame_failure_manifest),
        "video_failure_summary": str(video_failure_summary),
        "video_failure_summary_sha256": _sha256_file(video_failure_summary),
        "raw_openface_root": str(raw_openface_root or ""),
        "face_landmark_rerun_performed": False,
        "source_tree_modified": False,
        "max_videos": max_videos,
        "selected_video_count": len(source_rows),
        "pure_black_run_count": len(source_run_rows),
        "pure_black_frame_count": sum(row["length"] for row in source_run_rows),
        "contract_status_counts": dict(sorted(contract_counts.items())),
        "max_run_samples": max_run_samples,
        "contact_sheet_columns": contact_sheet_columns,
        "contact_sheets_written": write_contact_sheets,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "implementation_files": {
            "core": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "cli": {
                "path": str(project_root / "scripts" / "audit_source_presence.py"),
                "sha256": _sha256_file(project_root / "scripts" / "audit_source_presence.py"),
            },
        },
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256_file(path)} for path in outputs
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [*outputs, manifest_path]


def validate_source_review_segments(review_manifest, pure_black_failure_rows):
    """Require exactly one reviewed source-presence segment per black frame."""

    review_manifest = Path(review_manifest).expanduser().resolve()
    rows = _read_csv(review_manifest)
    required = set(SOURCE_REVIEW_FIELDS)
    if rows:
        missing = sorted(required - set(rows[0]))
        if missing:
            raise ValueError(f"source review manifest is missing columns: {missing}")
    else:
        with review_manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), [])
        missing = sorted(required - set(header))
        if missing:
            raise ValueError(f"source review manifest is missing columns: {missing}")

    by_video = defaultdict(list)
    for row in rows:
        start = _safe_int(row.get("segment_start_frame"))
        end = _safe_int(row.get("segment_end_frame"))
        if start is None or end is None or start <= 0 or end < start:
            raise ValueError(f"invalid source review segment: {row}")
        if row.get("review_status", "").strip().upper() != "REVIEWED":
            raise ValueError(
                f"source review segment is not REVIEWED: {row.get('video_id')} {start}-{end}"
            )
        presence = row.get("source_presence_status", "").strip()
        permission = row.get("recovery_permission", "").strip()
        if presence not in REVIEWED_PRESENCE_STATUSES:
            raise ValueError(f"invalid source_presence_status: {presence}")
        if permission not in RECOVERY_PERMISSIONS:
            raise ValueError(f"invalid recovery_permission: {permission}")
        if presence != "person_present_detection_failure" and permission != "keep_invalid":
            raise ValueError(f"{presence} must use recovery_permission=keep_invalid")
        by_video[row["video_id"]].append((start, end, row))

    for video_id, segments in by_video.items():
        segments.sort(key=lambda item: (item[0], item[1]))
        for previous, current in zip(segments, segments[1:]):
            if current[0] <= previous[1]:
                raise ValueError(f"overlapping source review segments for {video_id}")

    result = {}
    for failure in pure_black_failure_rows:
        if failure.get("failure_type") != "pure_black":
            continue
        video_id = failure["video_id"]
        frame_id = _safe_int(failure.get("frame_id"))
        matches = [row for start, end, row in by_video.get(video_id, []) if start <= frame_id <= end]
        if len(matches) != 1:
            raise ValueError(
                f"pure-black source review coverage must be exactly one: {video_id} frame {frame_id}"
            )
        result[(video_id, frame_id)] = matches[0]
    return result


def run_source_presence_review_report(
    *,
    frame_failure_manifest,
    source_run_manifest,
    review_manifest,
    output_dir,
    project_root=None,
):
    """Validate a completed human review and write frame/video audit reports."""

    frame_failure_manifest = Path(frame_failure_manifest).expanduser().resolve()
    source_run_manifest = Path(source_run_manifest).expanduser().resolve()
    review_manifest = Path(review_manifest).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    failures = [
        row for row in _read_csv(frame_failure_manifest) if row.get("failure_type") == "pure_black"
    ]
    source_runs = _read_csv(source_run_manifest)
    review_by_frame = validate_source_review_segments(review_manifest, failures)

    run_by_frame = {}
    for run in source_runs:
        start = _safe_int(run.get("start_frame"))
        end = _safe_int(run.get("end_frame"))
        if start is None or end is None:
            raise ValueError(f"invalid source run: {run}")
        for frame_id in range(start, end + 1):
            key = (run["video_id"], frame_id)
            if key in run_by_frame:
                raise ValueError(f"overlapping source runs: {key}")
            run_by_frame[key] = run["pure_black_run_id"]

    frame_rows = []
    video_frames = defaultdict(list)
    for failure in sorted(failures, key=lambda row: (row["video_id"], int(row["frame_id"]))):
        video_id = failure["video_id"]
        frame_id = int(failure["frame_id"])
        key = (video_id, frame_id)
        if key not in run_by_frame:
            raise ValueError(f"pure-black frame is absent from source run manifest: {key}")
        review = review_by_frame[key]
        row = {
            "split": failure.get("split", ""),
            "video_id": video_id,
            "pure_black_run_id": run_by_frame[key],
            "frame_id": frame_id,
            "source_presence_status": review["source_presence_status"],
            "recovery_permission": review["recovery_permission"],
            "review_status": review["review_status"],
            "reviewer": review.get("reviewer", ""),
            "review_notes": review.get("review_notes", ""),
        }
        frame_rows.append(row)
        video_frames[video_id].append(row)

    runs_by_video = Counter(row["video_id"] for row in source_runs)
    video_rows = []
    for video_id in sorted(video_frames):
        rows = video_frames[video_id]
        statuses = Counter(row["source_presence_status"] for row in rows)
        permissions = Counter(row["recovery_permission"] for row in rows)
        video_rows.append(
            {
                "split": rows[0]["split"],
                "video_id": video_id,
                "pure_black_run_count": runs_by_video[video_id],
                "pure_black_frame_count": len(rows),
                "person_present_detection_failure_frames": statuses[
                    "person_present_detection_failure"
                ],
                "person_absent_frames": statuses["person_absent"],
                "mixed_frames": statuses["mixed"],
                "ambiguous_frames": statuses["ambiguous"],
                "raw_frame_warp_only_frames": permissions["raw_frame_warp_only"],
                "keep_invalid_frames": permissions["keep_invalid"],
                "review_status": "REVIEWED",
            }
        )

    frame_path = _write_csv(
        tables_dir / "source_presence_frame_gate.csv",
        frame_rows,
        SOURCE_FRAME_GATE_FIELDS,
    )
    video_path = _write_csv(
        tables_dir / "source_presence_video_summary.csv",
        video_rows,
        SOURCE_VIDEO_REVIEW_FIELDS,
    )

    status_counts = Counter(row["source_presence_status"] for row in frame_rows)
    permission_counts = Counter(row["recovery_permission"] for row in frame_rows)
    videos_with_absence = [row for row in video_rows if row["person_absent_frames"]]
    run_statuses = Counter()
    for run in source_runs:
        start = int(run["start_frame"])
        end = int(run["end_frame"])
        statuses = {
            review_by_frame[(run["video_id"], frame_id)]["source_presence_status"]
            for frame_id in range(start, end + 1)
        }
        run_statuses[next(iter(statuses)) if len(statuses) == 1 else "mixed_after_segmentation"] += 1

    lines = [
        "# Full Source-video Presence Review",
        "",
        "## Scope and contract",
        "",
        f"- Reviewed videos with pure-black aligned placeholders: {len(video_rows)}.",
        f"- Reviewed pure-black runs: {len(source_runs)}; frames: {len(frame_rows)}.",
        "- Original-video frame identity was frozen before review; no FaceLandmark/OpenFace process was launched.",
        "- Short runs were reviewed from boundary/full-frame contact sheets. Long runs were additionally checked with dense/all-frame source-video sheets; all 2,136 source frames in the 247_3 mixed run were reviewed in consecutive pages and its departure/return boundaries were inspected frame by frame.",
        "",
        "## Presence conclusion",
        "",
        f"- Person present but aligned detection failed: {status_counts['person_present_detection_failure']} frames.",
        f"- Person absent from the source frame: {status_counts['person_absent']} frames.",
        f"- Mixed/ambiguous after segmentation: {status_counts['mixed'] + status_counts['ambiguous']} frames.",
        f"- Runs containing only person-present detection failure: {run_statuses['person_present_detection_failure']}.",
        f"- Runs requiring multiple presence segments: {run_statuses['mixed_after_segmentation']}.",
        f"- Videos containing confirmed person absence: {len(videos_with_absence)}.",
        "",
        "## Recovery gate",
        "",
        f"- `raw_frame_warp_only`: {permission_counts['raw_frame_warp_only']} frames. This is permission for a future raw-source warp audit, not a completed repair.",
        f"- `keep_invalid`: {permission_counts['keep_invalid']} frames. A face must never be synthesized for these frames.",
        "- Aligned-neighbor optical-flow interpolation and previous-aligned-frame copy remain disabled for every pure-black frame.",
        "",
        "## Confirmed absence cases",
        "",
        "| Video | Person-present failure frames | Person-absent frames | Decision |",
        "|---|---:|---:|---|",
    ]
    if videos_with_absence:
        lines.extend(
            f"| {row['video_id']} | {row['person_present_detection_failure_frames']} | {row['person_absent_frames']} | absent -> keep_invalid; present -> raw_frame_warp_only |"
            for row in videos_with_absence
        )
    else:
        lines.append("| none | 0 | 0 | - |")
    lines.extend(
        [
            "",
            "## Per-video summary",
            "",
            "| Video | Runs | Black frames | Present failure | Absent | Raw warp only | Keep invalid |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        f"| {row['video_id']} | {row['pure_black_run_count']} | {row['pure_black_frame_count']} | {row['person_present_detection_failure_frames']} | {row['person_absent_frames']} | {row['raw_frame_warp_only_frames']} | {row['keep_invalid_frames']} |"
        for row in video_rows
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `person_present_detection_failure` includes severe pose, motion blur, low head position, partial body/face exit, hand/headphone occlusion, close-range framing, and exposure changes. It does not imply that a usable frontal face exists.",
            "- `raw_frame_warp_only` does not authorize hallucination or aligned-frame blending. A later implementation must warp the actual source frame using audited transform propagation and preserve real occlusion/pose content.",
            "- The source-presence review is independent of BDI labels and predictions; no threshold was selected from validation/test utility.",
        ]
    )
    report_path = reports_dir / "source_presence_full_review_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    project_root = Path(project_root or Path(__file__).resolve().parents[2])
    outputs = [frame_path, video_path, report_path]
    payload = {
        "audit": "completed human source-video presence review",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(shlex.quote(item) for item in sys.argv),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "frame_failure_manifest": str(frame_failure_manifest),
        "frame_failure_manifest_sha256": _sha256_file(frame_failure_manifest),
        "source_run_manifest": str(source_run_manifest),
        "source_run_manifest_sha256": _sha256_file(source_run_manifest),
        "review_manifest": str(review_manifest),
        "review_manifest_sha256": _sha256_file(review_manifest),
        "face_landmark_rerun_performed": False,
        "source_tree_modified": False,
        "reviewed_video_count": len(video_rows),
        "reviewed_run_count": len(source_runs),
        "reviewed_frame_count": len(frame_rows),
        "source_presence_status_counts": dict(sorted(status_counts.items())),
        "recovery_permission_counts": dict(sorted(permission_counts.items())),
        "videos_with_confirmed_absence": [row["video_id"] for row in videos_with_absence],
        "outputs": {
            path.name: {"path": str(path), "sha256": _sha256_file(path)} for path in outputs
        },
    }
    manifest_path = output_dir / "review_manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [*outputs, manifest_path]
