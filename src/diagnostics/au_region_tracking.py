import csv
import math
import re
from collections import Counter
from pathlib import Path

from src.datasets.temporal_sampling import (
    normalize_temporal_sampling_strategy,
    select_temporal_indices,
)


FRAME_CONTRACT_FIELDS = [
    "video_id",
    "image_dir",
    "openface_csv",
    "image_frame_count",
    "csv_row_count",
    "image_frame_id_parse_count",
    "csv_frame_id_parse_count",
    "image_duplicate_id_count",
    "csv_duplicate_id_count",
    "image_ids_monotonic",
    "csv_frames_monotonic",
    "timestamps_monotonic",
    "image_width",
    "image_height",
    "best_frame_offset",
    "joined_frame_count",
    "frame_join_rate",
    "selected_frame_count",
    "selected_joined_frame_count",
    "selected_join_rate",
    "sample_step",
    "max_seq_len",
    "sampling_strategy",
    "status",
    "issues",
]

SELECTED_FRAME_FIELDS = [
    "video_id",
    "selected_position",
    "image_index",
    "image_path",
    "image_frame_id",
    "best_frame_offset",
    "expected_csv_frame",
    "csv_row_index",
    "csv_timestamp",
    "match_status",
]

ISSUE_FIELDS = ["video_id", "issue_type", "item", "detail"]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv_rows(csv_path, rows, fieldnames):
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_video_id(video_id):
    text = str(video_id or "")
    if text.endswith(".csv"):
        text = Path(text).stem
    if text.endswith("_aligned"):
        text = text[: -len("_aligned")]
    return text


def looks_like_openface_csv(csv_path):
    try:
        with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), [])
    except (OSError, UnicodeDecodeError):
        return False
    return "frame" in header


def find_openface_csv_files(openface_root):
    openface_root = Path(openface_root).expanduser()
    if not openface_root.exists():
        return []
    return sorted(path for path in openface_root.rglob("*.csv") if looks_like_openface_csv(path))


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_int(value):
    number = _safe_float(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def extract_frame_id(frame_path, frame_id_regex=None):
    """Extract a numeric source-frame id from an aligned image filename.

    With no explicit regex, the last digit group in the stem is used. This
    supports common names such as ``frame_000001.jpg`` and
    ``frame_det_00_000001.jpg`` while avoiding assumptions about prefixes.
    """
    stem = Path(frame_path).stem
    if frame_id_regex:
        match = re.search(frame_id_regex, stem)
        if match is None:
            return None
        value = match.group(1) if match.lastindex else match.group(0)
        return _safe_int(value)
    matches = re.findall(r"\d+", stem)
    return int(matches[-1]) if matches else None


def _is_strictly_increasing(values):
    values = [value for value in values if value is not None]
    return all(right > left for left, right in zip(values, values[1:]))


def _duplicate_extra_count(values):
    counts = Counter(value for value in values if value is not None)
    return sum(max(0, count - 1) for count in counts.values())


def _read_image_size(frame_path):
    if frame_path is None:
        return None, None
    try:
        from torchvision.io import ImageReadMode, read_image

        image = read_image(str(frame_path), mode=ImageReadMode.RGB)
    except Exception:
        return None, None
    return int(image.shape[-1]), int(image.shape[-2])


def read_openface_frame_rows(csv_path):
    rows = []
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            rows.append(
                {
                    "row_index": row_index,
                    "frame": _safe_int(row.get("frame")),
                    "timestamp": _safe_float(row.get("timestamp")),
                }
            )
    return rows


def choose_frame_offset(image_frame_ids, csv_frame_ids, candidate_offsets=(-2, -1, 0, 1, 2)):
    image_ids = {value for value in image_frame_ids if value is not None}
    csv_ids = {value for value in csv_frame_ids if value is not None}
    if not image_ids or not csv_ids:
        return None, 0

    scores = []
    for offset in candidate_offsets:
        joined = sum(1 for frame_id in image_ids if frame_id + int(offset) in csv_ids)
        scores.append((joined, -abs(int(offset)), 1 if int(offset) == 0 else 0, int(offset)))
    joined, _distance, _zero_preference, offset = max(scores)
    return offset, joined


def _issue_row(video_id, issue_type, item="", detail=""):
    return {
        "video_id": str(video_id),
        "issue_type": str(issue_type),
        "item": str(item),
        "detail": str(detail),
    }


def summarize_frame_contract(
    video_dir,
    openface_csv,
    video_id=None,
    sample_step=10,
    max_seq_len=2000,
    sampling_strategy="stride_head",
    frame_id_regex=None,
    join_threshold=0.995,
):
    video_dir = Path(video_dir) if video_dir is not None else None
    openface_csv = Path(openface_csv) if openface_csv is not None else None
    inferred_video_id = video_id
    if not inferred_video_id and video_dir is not None:
        inferred_video_id = video_dir.name
    if not inferred_video_id and openface_csv is not None:
        inferred_video_id = openface_csv.stem
    video_id = normalize_video_id(inferred_video_id)
    sampling_strategy = normalize_temporal_sampling_strategy(sampling_strategy)
    sample_step = max(1, int(sample_step))
    max_seq_len = max(1, int(max_seq_len))

    frame_paths = sorted(video_dir.glob("*.jpg")) if video_dir is not None and video_dir.exists() else []
    image_frame_ids = [extract_frame_id(path, frame_id_regex=frame_id_regex) for path in frame_paths]
    csv_rows = read_openface_frame_rows(openface_csv) if openface_csv is not None and openface_csv.exists() else []
    csv_frame_ids = [row["frame"] for row in csv_rows]
    csv_timestamps = [row["timestamp"] for row in csv_rows]

    best_offset, joined_count = choose_frame_offset(image_frame_ids, csv_frame_ids)
    image_count = len(frame_paths)
    frame_join_rate = joined_count / image_count if image_count else 0.0
    csv_by_frame = {}
    for row in csv_rows:
        frame_id = row["frame"]
        if frame_id is not None and frame_id not in csv_by_frame:
            csv_by_frame[frame_id] = row

    selected_indices = select_temporal_indices(
        image_count,
        sample_step=sample_step,
        max_seq_len=max_seq_len,
        strategy=sampling_strategy,
        seed=str(video_dir) if video_dir is not None else str(video_id),
    )
    selected_rows = []
    selected_joined = 0
    for selected_position, image_index in enumerate(selected_indices):
        frame_id = image_frame_ids[image_index]
        expected_csv_frame = (
            frame_id + best_offset
            if frame_id is not None and best_offset is not None
            else None
        )
        csv_row = csv_by_frame.get(expected_csv_frame)
        if csv_row is not None:
            selected_joined += 1
        selected_rows.append(
            {
                "video_id": video_id,
                "selected_position": selected_position,
                "image_index": image_index,
                "image_path": str(frame_paths[image_index]),
                "image_frame_id": frame_id,
                "best_frame_offset": best_offset,
                "expected_csv_frame": expected_csv_frame,
                "csv_row_index": csv_row["row_index"] if csv_row else None,
                "csv_timestamp": csv_row["timestamp"] if csv_row else None,
                "match_status": "matched" if csv_row else "unmatched",
            }
        )

    selected_count = len(selected_indices)
    selected_join_rate = selected_joined / selected_count if selected_count else 0.0
    image_duplicate_count = _duplicate_extra_count(image_frame_ids)
    csv_duplicate_count = _duplicate_extra_count(csv_frame_ids)
    image_parse_count = sum(value is not None for value in image_frame_ids)
    csv_parse_count = sum(value is not None for value in csv_frame_ids)
    image_ids_monotonic = _is_strictly_increasing(image_frame_ids)
    csv_frames_monotonic = _is_strictly_increasing(csv_frame_ids)
    timestamps_monotonic = _is_strictly_increasing(csv_timestamps)
    image_width, image_height = _read_image_size(frame_paths[0] if frame_paths else None)

    issues = []
    if not frame_paths:
        issues.append(_issue_row(video_id, "missing_image_frames", video_dir or ""))
    if not csv_rows:
        issues.append(_issue_row(video_id, "missing_openface_rows", openface_csv or ""))
    if image_parse_count != image_count:
        issues.append(
            _issue_row(
                video_id,
                "unparsed_image_frame_ids",
                image_count - image_parse_count,
                "Use --frame-id-regex when the filename frame id is not the last digit group.",
            )
        )
    if csv_parse_count != len(csv_rows):
        issues.append(_issue_row(video_id, "unparsed_csv_frame_ids", len(csv_rows) - csv_parse_count))
    if image_duplicate_count:
        issues.append(_issue_row(video_id, "duplicate_image_frame_ids", image_duplicate_count))
    if csv_duplicate_count:
        issues.append(_issue_row(video_id, "duplicate_csv_frame_ids", csv_duplicate_count))
    if image_frame_ids and not image_ids_monotonic:
        issues.append(_issue_row(video_id, "non_monotonic_image_frame_ids"))
    if csv_frame_ids and not csv_frames_monotonic:
        issues.append(_issue_row(video_id, "non_monotonic_csv_frames"))
    if csv_timestamps and not timestamps_monotonic:
        issues.append(_issue_row(video_id, "non_monotonic_timestamps"))

    image_id_set = {value for value in image_frame_ids if value is not None}
    csv_id_set = {value for value in csv_frame_ids if value is not None}
    if best_offset is not None:
        for frame_id in sorted(image_id_set):
            if frame_id + best_offset not in csv_id_set:
                issues.append(_issue_row(video_id, "unmatched_image_frame", frame_id))
        expected_csv_ids = {frame_id + best_offset for frame_id in image_id_set}
        for frame_id in sorted(csv_id_set - expected_csv_ids):
            issues.append(_issue_row(video_id, "unused_csv_frame", frame_id))

    blocking = (
        not frame_paths
        or not csv_rows
        or image_parse_count != image_count
        or csv_parse_count != len(csv_rows)
        or best_offset is None
    )
    failed = (
        frame_join_rate < float(join_threshold)
        or selected_join_rate < 1.0
        or image_duplicate_count > 0
        or csv_duplicate_count > 0
        or not image_ids_monotonic
        or not csv_frames_monotonic
        or not timestamps_monotonic
    )
    status = "BLOCKED" if blocking else "FAIL" if failed else "PASS"
    issue_names = sorted({row["issue_type"] for row in issues})
    summary = {
        "video_id": video_id,
        "image_dir": str(video_dir) if video_dir is not None else "",
        "openface_csv": str(openface_csv) if openface_csv is not None else "",
        "image_frame_count": image_count,
        "csv_row_count": len(csv_rows),
        "image_frame_id_parse_count": image_parse_count,
        "csv_frame_id_parse_count": csv_parse_count,
        "image_duplicate_id_count": image_duplicate_count,
        "csv_duplicate_id_count": csv_duplicate_count,
        "image_ids_monotonic": image_ids_monotonic,
        "csv_frames_monotonic": csv_frames_monotonic,
        "timestamps_monotonic": timestamps_monotonic,
        "image_width": image_width,
        "image_height": image_height,
        "best_frame_offset": best_offset,
        "joined_frame_count": joined_count,
        "frame_join_rate": frame_join_rate,
        "selected_frame_count": selected_count,
        "selected_joined_frame_count": selected_joined,
        "selected_join_rate": selected_join_rate,
        "sample_step": sample_step,
        "max_seq_len": max_seq_len,
        "sampling_strategy": sampling_strategy,
        "status": status,
        "issues": ";".join(issue_names),
    }
    return summary, selected_rows, issues


def find_aligned_video_dirs(image_root):
    image_root = Path(image_root).expanduser()
    if not image_root.exists():
        return []
    return sorted({path.parent for path in image_root.rglob("*.jpg")})


def _unique_path_map(paths, id_getter):
    result = {}
    duplicates = {}
    for path in paths:
        video_id = normalize_video_id(id_getter(path))
        if video_id in result:
            duplicates.setdefault(video_id, [result[video_id]]).append(path)
            continue
        result[video_id] = path
    return result, duplicates


def write_frame_contract_report(report_path, summaries, generated_files, join_threshold):
    report_path = Path(report_path)
    ensure_dir(report_path.parent)
    status_counts = Counter(row["status"] for row in summaries)
    rates = [row["frame_join_rate"] for row in summaries if row["image_frame_count"]]
    offsets = Counter(row["best_frame_offset"] for row in summaries if row["best_frame_offset"] is not None)
    lines = [
        "# AU Region Frame Contract Audit",
        "",
        f"- Videos inventoried: {len(summaries)}",
        f"- PASS: {status_counts.get('PASS', 0)}",
        f"- FAIL: {status_counts.get('FAIL', 0)}",
        f"- BLOCKED: {status_counts.get('BLOCKED', 0)}",
        f"- Minimum frame join rate: {min(rates) if rates else 0.0:.4f}",
        f"- Required frame join rate: {float(join_threshold):.4f}",
        "",
        "## Frame Offset Distribution",
        "",
    ]
    if offsets:
        lines.append("| Offset | Videos |")
        lines.append("|---:|---:|")
        for offset, count in sorted(offsets.items()):
            lines.append(f"| {offset} | {count} |")
    else:
        lines.append("No valid frame offset could be inferred.")

    lines.extend(["", "## Non-passing Videos", ""])
    failed = [row for row in summaries if row["status"] != "PASS"]
    if failed:
        lines.append("| Video | Status | Join rate | Selected join | Issues |")
        lines.append("|---|---|---:|---:|---|")
        for row in failed[:50]:
            lines.append(
                f"| {row['video_id']} | {row['status']} | {row['frame_join_rate']:.4f} | "
                f"{row['selected_join_rate']:.4f} | {row['issues']} |"
            )
    else:
        lines.append("All inventoried videos passed the frozen frame contract.")

    lines.extend(["", "## Generated Files", ""])
    lines.extend(f"- `{path}`" for path in generated_files)
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- This audit proves only frame/file/CSV alignment and reproduces the configured temporal sampling indices.",
            "- It does not prove that OpenFace detection-space landmarks map to aligned-image coordinates.",
            "- Coordinate recovery and mask overlays remain the separate AU-T0b gate.",
            "- Validation/test data must not be used to tune filename regexes, offsets, or sampling parameters.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_frame_contract_audit(
    image_root,
    openface_root,
    output_dir,
    sample_step=10,
    max_seq_len=2000,
    sampling_strategy="stride_head",
    frame_id_regex=None,
    join_threshold=0.995,
    max_videos=None,
):
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")

    image_map, duplicate_image_dirs = _unique_path_map(
        find_aligned_video_dirs(image_root), lambda path: path.name
    )
    csv_map, duplicate_csvs = _unique_path_map(
        find_openface_csv_files(openface_root), lambda path: path.stem
    )
    video_ids = sorted(set(image_map) | set(csv_map))
    if max_videos is not None:
        video_ids = video_ids[: max(0, int(max_videos))]

    summaries = []
    selected_rows = []
    issues = []
    for video_id in video_ids:
        summary, video_selected, video_issues = summarize_frame_contract(
            video_dir=image_map.get(video_id),
            openface_csv=csv_map.get(video_id),
            video_id=video_id,
            sample_step=sample_step,
            max_seq_len=max_seq_len,
            sampling_strategy=sampling_strategy,
            frame_id_regex=frame_id_regex,
            join_threshold=join_threshold,
        )
        duplicate_types = []
        if video_id in duplicate_image_dirs:
            duplicate_types.append("duplicate_video_directories")
        if video_id in duplicate_csvs:
            duplicate_types.append("duplicate_openface_csvs")
        if duplicate_types:
            summary["status"] = "FAIL"
            summary_issue_names = {item for item in summary["issues"].split(";") if item}
            summary_issue_names.update(duplicate_types)
            summary["issues"] = ";".join(sorted(summary_issue_names))
        summaries.append(summary)
        selected_rows.extend(video_selected)
        issues.extend(video_issues)

    for video_id, paths in duplicate_image_dirs.items():
        issues.append(_issue_row(video_id, "duplicate_video_directories", "", ";".join(map(str, paths))))
    for video_id, paths in duplicate_csvs.items():
        issues.append(_issue_row(video_id, "duplicate_openface_csvs", "", ";".join(map(str, paths))))

    summary_path = tables_dir / "frame_contract_summary.csv"
    write_csv_rows(
        summary_path,
        [{field: _format_value(row.get(field)) for field in FRAME_CONTRACT_FIELDS} for row in summaries],
        FRAME_CONTRACT_FIELDS,
    )
    selected_path = tables_dir / "selected_frame_mapping.csv"
    write_csv_rows(
        selected_path,
        [{field: _format_value(row.get(field)) for field in SELECTED_FRAME_FIELDS} for row in selected_rows],
        SELECTED_FRAME_FIELDS,
    )
    issues_path = tables_dir / "frame_contract_issues.csv"
    write_csv_rows(issues_path, issues, ISSUE_FIELDS)
    generated = [summary_path, selected_path, issues_path]
    report_path = write_frame_contract_report(
        reports_dir / "frame_contract_report.md",
        summaries=summaries,
        generated_files=generated,
        join_threshold=join_threshold,
    )
    generated.append(report_path)
    return generated
