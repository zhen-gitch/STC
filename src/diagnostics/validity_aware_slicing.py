"""Read-only feasibility audit for validity-aware temporal slicing.

The current AVEC dataset loader samples directly from the full aligned JPG
sequence.  Padding is masked, but source-person absence, pure-black aligned
placeholders, and landmark failures are not represented in that mask.  This
module joins the already-audited failure/presence manifests and measures how
different input contracts would affect continuous usable runs and long-video
windowing.  It never reads BDI labels, changes split membership, or modifies
source images.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.datasets.temporal_sampling import select_temporal_indices


POLICY_LEGACY_ALL = "legacy_all_frames"
POLICY_GLOBAL_READY = "global_rgb_ready_now"
POLICY_GLOBAL_POST_WARP = "global_rgb_after_approved_raw_warp"
POLICY_LANDMARK_READY = "landmark_local_ready_now"
# Backward-compatible import alias. AU values are not read by this audit.
POLICY_AU_READY = POLICY_LANDMARK_READY

POLICY_ORDER = (
    POLICY_LEGACY_ALL,
    POLICY_GLOBAL_READY,
    POLICY_GLOBAL_POST_WARP,
    POLICY_LANDMARK_READY,
)

POLICY_STATUS = {
    POLICY_LEGACY_ALL: "historical_loader_contract",
    POLICY_GLOBAL_READY: "current_data_contract",
    POLICY_GLOBAL_POST_WARP: "hypothetical_until_raw_warp_and_validation",
    POLICY_LANDMARK_READY: "current_landmark_contract",
}

RUN_FIELDS = [
    "policy",
    "policy_status",
    "split",
    "video_id",
    "run_id",
    "start_frame",
    "end_frame",
    "raw_frame_count",
    "sampled_frame_count",
    "below_min_clip_frames",
]

CLIP_FIELDS = [
    "policy",
    "policy_status",
    "split",
    "video_id",
    "run_id",
    "clip_id",
    "window_frames",
    "start_frame",
    "end_frame",
    "raw_frame_count",
    "sampled_frame_count",
    "below_min_clip_frames",
    "aggregation_weight_frames",
]

VIDEO_FIELDS = [
    "policy",
    "policy_status",
    "split",
    "video_id",
    "frame_count",
    "invalid_frame_count",
    "valid_frame_count",
    "valid_ratio",
    "valid_run_count",
    "longest_valid_run",
    "runs_below_min_clip_frames",
    "legacy_head_selected_count",
    "legacy_head_invalid_count",
    "legacy_head_invalid_ratio",
    "legacy_head_span_end_frame",
    "legacy_head_temporal_span_ratio",
    "legacy_head_unseen_tail_frames",
    "legacy_head_unseen_valid_frames",
    "legacy_head_unseen_valid_ratio",
    "window_frames",
    "deterministic_clip_count",
    "clips_below_min_clip_frames",
    "multi_clip_required",
]

SUMMARY_FIELDS = [
    "policy",
    "policy_status",
    "split",
    "window_frames",
    "video_count",
    "total_frame_count",
    "valid_frame_count",
    "invalid_frame_count",
    "valid_ratio",
    "valid_run_count",
    "videos_with_multiple_runs",
    "runs_below_min_clip_frames",
    "videos_longer_than_window",
    "deterministic_clip_count",
    "videos_requiring_multiple_clips",
    "clips_below_min_clip_frames",
    "clip_augmentation_factor",
    "legacy_head_invalid_count",
    "legacy_head_selected_count",
    "legacy_head_invalid_ratio",
    "legacy_head_temporal_span_ratio_mean",
    "legacy_head_temporal_span_ratio_weighted",
    "legacy_head_unseen_tail_frames",
    "legacy_head_unseen_valid_frames",
    "legacy_head_unseen_valid_ratio",
]


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def _write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format(row.get(field)) for field in fields})
    return path


def _safe_int(value, field):
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc
    return number


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


def sampled_frame_count(raw_frame_count, sample_step):
    raw_frame_count = max(0, int(raw_frame_count))
    sample_step = max(1, int(sample_step))
    if raw_frame_count == 0:
        return 0
    return ((raw_frame_count - 1) // sample_step) + 1


def contiguous_valid_runs(frame_count, invalid_frames):
    """Return inclusive 1-based valid runs without deleting timeline gaps."""

    frame_count = int(frame_count)
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    invalid = sorted({int(frame) for frame in invalid_frames})
    if invalid and (invalid[0] < 1 or invalid[-1] > frame_count):
        raise ValueError("invalid frame id lies outside the video frame range")

    runs = []
    start = 1
    for frame_id in invalid:
        if start <= frame_id - 1:
            runs.append((start, frame_id - 1))
        start = frame_id + 1
    if start <= frame_count:
        runs.append((start, frame_count))
    return runs


def balanced_partition_windows(start_frame, end_frame, window_frames):
    """Partition one valid run into the fewest near-equal non-overlap clips.

    Using equal partitions avoids a nearly duplicated tail window when a run is
    only slightly longer than ``window_frames``.  Every valid frame is covered
    exactly once, every clip is contiguous, and no clip crosses an invalid gap.
    """

    start_frame = int(start_frame)
    end_frame = int(end_frame)
    window_frames = int(window_frames)
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if end_frame < start_frame:
        return []

    length = end_frame - start_frame + 1
    clip_count = int(math.ceil(length / window_frames))
    base, remainder = divmod(length, clip_count)
    windows = []
    cursor = start_frame
    for index in range(clip_count):
        clip_length = base + (1 if index < remainder else 0)
        clip_end = cursor + clip_length - 1
        windows.append((cursor, clip_end))
        cursor = clip_end + 1
    if cursor != end_frame + 1:
        raise AssertionError("balanced partition did not fully cover the run")
    return windows


def _load_split_map(dataset_split_file):
    payload = json.loads(Path(dataset_split_file).read_text(encoding="utf-8"))
    split_map = {}
    for split in ("train", "val", "test"):
        values = payload.get(split)
        if not isinstance(values, list):
            raise ValueError(f"dataset split JSON has no list for {split!r}")
        for video_id in values:
            video_id = str(video_id)
            if video_id in split_map:
                raise ValueError(f"video appears in multiple splits: {video_id}")
            split_map[video_id] = split
    return split_map


def _load_video_inventory(video_summary_path, split_map):
    inventory = {}
    for row in _read_csv(video_summary_path):
        video_id = row.get("video_id", "")
        if video_id not in split_map:
            raise ValueError(f"video summary contains video outside split file: {video_id}")
        frame_count = _safe_int(row.get("frame_count"), "frame_count")
        if frame_count <= 0:
            raise ValueError(f"non-positive frame count for {video_id}")
        if video_id in inventory:
            raise ValueError(f"duplicate video summary row: {video_id}")
        row_split = row.get("split", "")
        if row_split and row_split != split_map[video_id]:
            raise ValueError(
                f"split mismatch for {video_id}: summary={row_split}, json={split_map[video_id]}"
            )
        inventory[video_id] = {
            "video_id": video_id,
            "split": split_map[video_id],
            "frame_count": frame_count,
        }
    missing = sorted(set(split_map) - set(inventory))
    if missing:
        raise ValueError(f"video summary is missing {len(missing)} split videos; first={missing[0]}")
    return inventory


def _build_policy_invalid_frames(failure_rows, presence_rows, inventory):
    failures = defaultdict(set)
    pure_black = defaultdict(set)
    unreadable = defaultdict(set)
    failure_keys = set()
    for row in failure_rows:
        video_id = row.get("video_id", "")
        if video_id not in inventory:
            raise ValueError(f"failure row contains unknown video: {video_id}")
        frame_id = _safe_int(row.get("frame_id"), "frame_id")
        frame_count = inventory[video_id]["frame_count"]
        if not 1 <= frame_id <= frame_count:
            raise ValueError(f"failure frame outside range: {video_id} frame {frame_id}")
        key = (video_id, frame_id)
        if key in failure_keys:
            raise ValueError(f"duplicate failure frame row: {video_id} frame {frame_id}")
        failure_keys.add(key)
        failures[video_id].add(frame_id)
        failure_type = row.get("failure_type", "")
        if failure_type == "pure_black":
            pure_black[video_id].add(frame_id)
        if failure_type == "unreadable" or row.get("decode_status", "OK") != "OK":
            unreadable[video_id].add(frame_id)

    presence_by_frame = {}
    for row in presence_rows:
        video_id = row.get("video_id", "")
        if video_id not in inventory:
            raise ValueError(f"presence gate contains unknown video: {video_id}")
        frame_id = _safe_int(row.get("frame_id"), "frame_id")
        key = (video_id, frame_id)
        if key in presence_by_frame:
            raise ValueError(f"duplicate presence gate frame: {video_id} frame {frame_id}")
        if frame_id not in pure_black[video_id]:
            raise ValueError(f"presence gate frame is not pure black: {video_id} frame {frame_id}")
        if row.get("review_status") != "REVIEWED":
            raise ValueError(f"presence gate is not REVIEWED: {video_id} frame {frame_id}")
        presence_by_frame[key] = row

    expected_presence = {
        (video_id, frame_id)
        for video_id, frame_ids in pure_black.items()
        for frame_id in frame_ids
    }
    if set(presence_by_frame) != expected_presence:
        missing = sorted(expected_presence - set(presence_by_frame))
        extra = sorted(set(presence_by_frame) - expected_presence)
        raise ValueError(
            "presence gate must cover every pure-black frame exactly once; "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    person_absent = defaultdict(set)
    for (video_id, frame_id), row in presence_by_frame.items():
        status = row.get("source_presence_status", "")
        permission = row.get("recovery_permission", "")
        if status == "person_present_detection_failure":
            if permission != "raw_frame_warp_only":
                raise ValueError(
                    f"present failure lacks raw_frame_warp_only: {video_id} frame {frame_id}"
                )
        elif status in {"person_absent", "mixed", "ambiguous"}:
            if permission != "keep_invalid":
                raise ValueError(
                    f"non-present frame lacks keep_invalid: {video_id} frame {frame_id}"
                )
            person_absent[video_id].add(frame_id)
        else:
            raise ValueError(
                f"unsupported source presence status: {video_id} frame {frame_id}: {status!r}"
            )

    result = {policy: defaultdict(set) for policy in POLICY_ORDER}
    for video_id in inventory:
        result[POLICY_LEGACY_ALL][video_id] = set()
        result[POLICY_GLOBAL_READY][video_id] = set(pure_black[video_id]) | set(unreadable[video_id])
        result[POLICY_GLOBAL_POST_WARP][video_id] = set(person_absent[video_id]) | set(unreadable[video_id])
        result[POLICY_LANDMARK_READY][video_id] = set(failures[video_id]) | set(unreadable[video_id])
    return result


def build_slicing_audit_rows(
    inventory,
    invalid_by_policy,
    *,
    sample_step,
    max_seq_len,
    window_frames_values,
    min_clip_frames,
):
    run_rows = []
    clip_rows = []
    video_rows = []

    for policy in POLICY_ORDER:
        for video_id in sorted(inventory):
            item = inventory[video_id]
            frame_count = item["frame_count"]
            invalid_frames = invalid_by_policy[policy][video_id]
            runs = contiguous_valid_runs(frame_count, invalid_frames)
            head_indices = select_temporal_indices(
                frame_count,
                sample_step=sample_step,
                max_seq_len=max_seq_len,
                strategy="stride_head",
            )
            head_frame_ids = [index + 1 for index in head_indices]
            head_invalid = sum(frame_id in invalid_frames for frame_id in head_frame_ids)
            head_span_end = head_frame_ids[-1] if head_frame_ids else 0
            unseen_tail_frames = max(0, frame_count - head_span_end)
            unseen_invalid_frames = sum(frame_id > head_span_end for frame_id in invalid_frames)
            unseen_valid_frames = max(0, unseen_tail_frames - unseen_invalid_frames)

            for run_index, (start, end) in enumerate(runs, start=1):
                length = end - start + 1
                run_rows.append(
                    {
                        "policy": policy,
                        "policy_status": POLICY_STATUS[policy],
                        "split": item["split"],
                        "video_id": video_id,
                        "run_id": f"{video_id}:{policy}:R{run_index:04d}",
                        "start_frame": start,
                        "end_frame": end,
                        "raw_frame_count": length,
                        "sampled_frame_count": sampled_frame_count(length, sample_step),
                        "below_min_clip_frames": length < min_clip_frames,
                    }
                )

            for window_frames in window_frames_values:
                video_clip_rows = []
                for run_index, (start, end) in enumerate(runs, start=1):
                    run_id = f"{video_id}:{policy}:R{run_index:04d}"
                    partitions = balanced_partition_windows(start, end, window_frames)
                    for partition_index, (clip_start, clip_end) in enumerate(partitions, start=1):
                        length = clip_end - clip_start + 1
                        row = {
                            "policy": policy,
                            "policy_status": POLICY_STATUS[policy],
                            "split": item["split"],
                            "video_id": video_id,
                            "run_id": run_id,
                            "clip_id": f"{run_id}:W{window_frames}:C{partition_index:04d}",
                            "window_frames": window_frames,
                            "start_frame": clip_start,
                            "end_frame": clip_end,
                            "raw_frame_count": length,
                            "sampled_frame_count": sampled_frame_count(length, sample_step),
                            "below_min_clip_frames": length < min_clip_frames,
                            "aggregation_weight_frames": length,
                        }
                        video_clip_rows.append(row)
                        clip_rows.append(row)

                valid_count = frame_count - len(invalid_frames)
                video_rows.append(
                    {
                        "policy": policy,
                        "policy_status": POLICY_STATUS[policy],
                        "split": item["split"],
                        "video_id": video_id,
                        "frame_count": frame_count,
                        "invalid_frame_count": len(invalid_frames),
                        "valid_frame_count": valid_count,
                        "valid_ratio": valid_count / frame_count,
                        "valid_run_count": len(runs),
                        "longest_valid_run": max((end - start + 1 for start, end in runs), default=0),
                        "runs_below_min_clip_frames": sum(
                            (end - start + 1) < min_clip_frames for start, end in runs
                        ),
                        "legacy_head_selected_count": len(head_frame_ids),
                        "legacy_head_invalid_count": head_invalid,
                        "legacy_head_invalid_ratio": head_invalid / len(head_frame_ids) if head_frame_ids else 0.0,
                        "legacy_head_span_end_frame": head_span_end,
                        "legacy_head_temporal_span_ratio": head_span_end / frame_count if frame_count else 0.0,
                        "legacy_head_unseen_tail_frames": unseen_tail_frames,
                        "legacy_head_unseen_valid_frames": unseen_valid_frames,
                        "legacy_head_unseen_valid_ratio": unseen_valid_frames / valid_count if valid_count else 0.0,
                        "window_frames": window_frames,
                        "deterministic_clip_count": len(video_clip_rows),
                        "clips_below_min_clip_frames": sum(
                            row["below_min_clip_frames"] for row in video_clip_rows
                        ),
                        "multi_clip_required": len(video_clip_rows) > 1,
                    }
                )
    return run_rows, clip_rows, video_rows


def aggregate_slicing_rows(video_rows):
    grouped = defaultdict(list)
    for row in video_rows:
        grouped[(row["policy"], row["split"], row["window_frames"])].append(row)
        grouped[(row["policy"], "all", row["window_frames"])].append(row)

    summaries = []
    for (policy, split, window_frames), rows in sorted(grouped.items()):
        total_frames = sum(row["frame_count"] for row in rows)
        valid_frames = sum(row["valid_frame_count"] for row in rows)
        invalid_frames = sum(row["invalid_frame_count"] for row in rows)
        head_invalid = sum(row["legacy_head_invalid_count"] for row in rows)
        head_selected = sum(row["legacy_head_selected_count"] for row in rows)
        video_count = len(rows)
        summaries.append(
            {
                "policy": policy,
                "policy_status": POLICY_STATUS[policy],
                "split": split,
                "window_frames": window_frames,
                "video_count": video_count,
                "total_frame_count": total_frames,
                "valid_frame_count": valid_frames,
                "invalid_frame_count": invalid_frames,
                "valid_ratio": valid_frames / total_frames if total_frames else 0.0,
                "valid_run_count": sum(row["valid_run_count"] for row in rows),
                "videos_with_multiple_runs": sum(row["valid_run_count"] > 1 for row in rows),
                "runs_below_min_clip_frames": sum(row["runs_below_min_clip_frames"] for row in rows),
                "videos_longer_than_window": sum(row["frame_count"] > window_frames for row in rows),
                "deterministic_clip_count": sum(row["deterministic_clip_count"] for row in rows),
                "videos_requiring_multiple_clips": sum(row["multi_clip_required"] for row in rows),
                "clips_below_min_clip_frames": sum(row["clips_below_min_clip_frames"] for row in rows),
                "clip_augmentation_factor": (
                    sum(row["deterministic_clip_count"] for row in rows) / video_count
                    if video_count
                    else 0.0
                ),
                "legacy_head_invalid_count": head_invalid,
                "legacy_head_selected_count": head_selected,
                "legacy_head_invalid_ratio": head_invalid / head_selected if head_selected else 0.0,
                "legacy_head_temporal_span_ratio_mean": (
                    sum(row["legacy_head_temporal_span_ratio"] for row in rows) / video_count
                    if video_count
                    else 0.0
                ),
                "legacy_head_temporal_span_ratio_weighted": (
                    sum(row["legacy_head_span_end_frame"] for row in rows) / total_frames
                    if total_frames
                    else 0.0
                ),
                "legacy_head_unseen_tail_frames": sum(
                    row["legacy_head_unseen_tail_frames"] for row in rows
                ),
                "legacy_head_unseen_valid_frames": sum(
                    row["legacy_head_unseen_valid_frames"] for row in rows
                ),
                "legacy_head_unseen_valid_ratio": (
                    sum(row["legacy_head_unseen_valid_frames"] for row in rows) / valid_frames
                    if valid_frames
                    else 0.0
                ),
            }
        )
    return summaries


def _report_lines(summaries, sample_step, max_seq_len, min_clip_frames):
    lines = [
        "# Validity-aware Temporal Slicing Feasibility Audit",
        "",
        "## Contract",
        "",
        "- Read-only: no image, split, label, OpenFace CSV, or checkpoint was modified.",
        "- No BDI label or prediction was used to define a boundary or window.",
        "- Invalid frames are treated as timeline barriers; frames on both sides are never concatenated as if continuous.",
        f"- Historical sampling reference: `SAMPLE_STEP={sample_step}`, `MAX_SEQ_LEN={max_seq_len}`.",
        f"- Candidate clips shorter than {min_clip_frames} raw frames are reported, not silently discarded.",
        "- Deterministic clips use balanced non-overlapping partitions, covering each valid run exactly once.",
        "",
        "## Policy meanings",
        "",
        "- `legacy_all_frames`: reproduces the historical assumption that every JPG is valid.",
        "- `global_rgb_ready_now`: excludes unreadable and pure-black aligned placeholders; visible OpenFace failures remain usable RGB.",
        "- `global_rgb_after_approved_raw_warp`: hypothetical future contract; only source-confirmed absence/unreadable frames remain invalid.",
        "- `landmark_local_ready_now`: excludes every current OpenFace failure because landmark-guided local crops need a valid coordinate contract; AU values are not read.",
        "",
        "## Aggregate results",
        "",
        "| Policy | Split | Window | Videos | Valid ratio | Runs | Multi-run videos | Clips | Multi-clip videos | Short clips | Head invalid ratio | Unseen valid ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {policy} | {split} | {window} | {videos} | {valid:.4f} | {runs} | {multi_runs} | {clips} | {multi_clips} | {short} | {invalid:.4f} | {unseen:.4f} |".format(
                policy=row["policy"],
                split=row["split"],
                window=row["window_frames"],
                videos=row["video_count"],
                valid=row["valid_ratio"],
                runs=row["valid_run_count"],
                multi_runs=row["videos_with_multiple_runs"],
                clips=row["deterministic_clip_count"],
                multi_clips=row["videos_requiring_multiple_clips"],
                short=row["clips_below_min_clip_frames"],
                invalid=row["legacy_head_invalid_ratio"],
                unseen=row["legacy_head_unseen_valid_ratio"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `global_rgb_after_approved_raw_warp` is a capacity estimate, not permission to train on repaired data before the warp provenance and rerun gate pass.",
            "- A visible OpenFace failure is not automatically a bad global RGB frame; it is invalid only for AU/landmark-dependent local views.",
            "- Treating each clip as an independent subject would duplicate labels and overweight long videos. Training should sample video first, then a valid clip; validation/test must aggregate clips back to one video prediction before metrics.",
            "- Current GRU masking only removes padded outputs after recurrent encoding. Arbitrary internal invalid slots must not be introduced without a mask-aware recurrent contract or real-frame repair.",
        ]
    )
    return lines


def run_validity_aware_slicing_audit(
    *,
    dataset_split_file,
    video_failure_summary,
    frame_failure_manifest,
    source_presence_gate,
    output_dir,
    sample_step=10,
    max_seq_len=2000,
    window_frames_values=(600, 1200, 2000),
    min_clip_frames=600,
    project_root=None,
):
    dataset_split_file = Path(dataset_split_file)
    video_failure_summary = Path(video_failure_summary)
    frame_failure_manifest = Path(frame_failure_manifest)
    source_presence_gate = Path(source_presence_gate)
    output_dir = Path(output_dir)
    project_root = Path(project_root or Path(__file__).resolve().parents[2])

    sample_step = max(1, int(sample_step))
    max_seq_len = max(1, int(max_seq_len))
    min_clip_frames = max(1, int(min_clip_frames))
    window_frames_values = tuple(sorted({int(value) for value in window_frames_values}))
    if not window_frames_values or window_frames_values[0] <= 0:
        raise ValueError("window_frames_values must contain positive integers")

    split_map = _load_split_map(dataset_split_file)
    inventory = _load_video_inventory(video_failure_summary, split_map)
    failure_rows = _read_csv(frame_failure_manifest)
    presence_rows = _read_csv(source_presence_gate)
    invalid_by_policy = _build_policy_invalid_frames(failure_rows, presence_rows, inventory)
    run_rows, clip_rows, video_rows = build_slicing_audit_rows(
        inventory,
        invalid_by_policy,
        sample_step=sample_step,
        max_seq_len=max_seq_len,
        window_frames_values=window_frames_values,
        min_clip_frames=min_clip_frames,
    )
    summary_rows = aggregate_slicing_rows(video_rows)

    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    generated = [
        _write_csv(tables_dir / "validity_run_manifest.csv", run_rows, RUN_FIELDS),
        _write_csv(tables_dir / "candidate_clip_manifest.csv", clip_rows, CLIP_FIELDS),
        _write_csv(tables_dir / "slicing_video_summary.csv", video_rows, VIDEO_FIELDS),
        _write_csv(tables_dir / "slicing_aggregate_summary.csv", summary_rows, SUMMARY_FIELDS),
    ]

    report_path = reports_dir / "validity_aware_slicing_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(_report_lines(summary_rows, sample_step, max_seq_len, min_clip_frames)) + "\n",
        encoding="utf-8",
    )
    generated.append(report_path)

    inputs = [
        dataset_split_file,
        video_failure_summary,
        frame_failure_manifest,
        source_presence_gate,
    ]
    manifest = {
        "audit": "validity_aware_temporal_slicing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "command": shlex.join(sys.argv),
        "python_version": sys.version,
        "platform": platform.platform(),
        "sample_step": sample_step,
        "max_seq_len": max_seq_len,
        "window_frames_values": list(window_frames_values),
        "min_clip_frames": min_clip_frames,
        "random_seed": None,
        "deterministic": True,
        "gpu_device": "not_applicable_read_only_audit",
        "precision": "not_applicable_read_only_audit",
        "video_count": len(inventory),
        "failure_frame_count": len(failure_rows),
        "source_presence_frame_count": len(presence_rows),
        "policies": list(POLICY_ORDER),
        "input_files": [
            {"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in inputs
        ],
        "implementation_files": [
            {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            {
                "path": str((project_root / "scripts" / "audit_validity_aware_slicing.py").resolve()),
                "sha256": _sha256_file(project_root / "scripts" / "audit_validity_aware_slicing.py"),
            },
        ],
        "output_files": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": _sha256_file(path),
            }
            for path in generated
        ],
        "source_tree_modified": False,
        "split_modified": False,
        "labels_read": False,
        "predictions_read": False,
        "openface_rerun_performed": False,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    generated.append(manifest_path)
    return generated
