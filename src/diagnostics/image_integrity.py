"""Read-only integrity and cross-machine identity audit for aligned frames."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg"}

MANIFEST_FIELDS = [
    "relative_path",
    "video_id",
    "file_name",
    "file_size",
    "file_sha256",
    "pixel_sha256",
    "decode_status",
    "image_format",
    "image_mode",
    "width",
    "height",
    "issue",
]

ISSUE_FIELDS = [
    "relative_path",
    "video_id",
    "file_name",
    "issue_type",
    "detail",
]

COMPARISON_FIELDS = [
    "relative_path",
    "comparison_status",
    "reference_file_size",
    "candidate_file_size",
    "reference_file_sha256",
    "candidate_file_sha256",
    "reference_pixel_sha256",
    "candidate_pixel_sha256",
    "reference_decode_status",
    "candidate_decode_status",
    "detail",
]


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(image):
    rgb = image.convert("RGB")
    rgb.load()
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB\0".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _inspect_image_task(task):
    (
        absolute_path,
        relative_path,
        compute_pixel_hash,
        decode_check,
        expected_width,
        expected_height,
    ) = task
    path = Path(absolute_path)
    relative = Path(relative_path)
    row = {
        "relative_path": relative.as_posix(),
        "video_id": relative.parts[0] if len(relative.parts) > 1 else "",
        "file_name": relative.name,
        "file_size": "",
        "file_sha256": "",
        "pixel_sha256": "",
        "decode_status": "NOT_CHECKED" if not decode_check else "ERROR",
        "image_format": "",
        "image_mode": "",
        "width": "",
        "height": "",
        "issue": "",
    }
    try:
        encoded_bytes = path.read_bytes()
        row["file_size"] = len(encoded_bytes)
        row["file_sha256"] = hashlib.sha256(encoded_bytes).hexdigest()
    except OSError as exc:
        row["issue"] = f"file_read_error:{type(exc).__name__}:{exc}"
        return row

    if not decode_check:
        return row

    try:
        with Image.open(BytesIO(encoded_bytes)) as image:
            row["image_format"] = str(image.format or "")
            row["image_mode"] = str(image.mode or "")
            row["width"], row["height"] = image.size
            image.verify()
        with Image.open(BytesIO(encoded_bytes)) as image:
            image.load()
            if compute_pixel_hash:
                row["pixel_sha256"] = _pixel_sha256(image)
        row["decode_status"] = "OK"
        if (
            expected_width is not None
            and expected_height is not None
            and (row["width"] != expected_width or row["height"] != expected_height)
        ):
            row["issue"] = (
                f"unexpected_dimensions:expected={expected_width}x{expected_height},"
                f"actual={row['width']}x{row['height']}"
            )
    except Exception as exc:  # Pillow exposes multiple decoder-specific exceptions.
        row["decode_status"] = "ERROR"
        row["issue"] = f"image_decode_error:{type(exc).__name__}:{exc}"
    return row


def find_training_image_paths(image_root, max_videos=None):
    """Mirror the dataset contract: direct JPG children of direct video dirs."""
    image_root = Path(image_root).expanduser().resolve()
    if not image_root.exists() or not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found or not a directory: {image_root}")
    paths = []
    video_dirs = sorted(path for path in image_root.iterdir() if path.is_dir())
    if max_videos is not None:
        video_dirs = video_dirs[: max(0, int(max_videos))]
    for video_dir in video_dirs:
        for image_path in sorted(
            path
            for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ):
            paths.append(image_path)
    return image_root, paths


def _update_root_digest(digest, row, include_pixel_hash):
    fields = [
        row["relative_path"],
        str(row["file_size"]),
        row["file_sha256"],
        row["decode_status"],
    ]
    if include_pixel_hash:
        fields.append(row["pixel_sha256"])
    digest.update("\0".join(fields).encode("utf-8", errors="replace"))
    digest.update(b"\n")


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


def _write_inventory_report(path, summary):
    dimension_rows = sorted(
        summary["dimension_distribution"].items(),
        key=lambda item: (-item[1], item[0]),
    )
    lines = [
        "# Aligned Image Integrity Inventory",
        "",
        f"- Label: {summary['label']}",
        f"- Image root: `{summary['image_root']}`",
        f"- Video directories: {summary['video_dir_count']}",
        f"- Images: {summary['image_count']}",
        f"- Total bytes: {summary['total_bytes']}",
        f"- Decode OK: {summary['decode_ok_count']}",
        f"- Decode errors: {summary['decode_error_count']}",
        f"- Integrity issues: {summary['integrity_issue_count']}",
        f"- Expected dimensions: {summary['expected_width']}x{summary['expected_height']}",
        f"- Pixel hash enabled: {summary['pixel_hash_enabled']}",
        f"- Root digest: `{summary['root_digest']}`",
        f"- Status: {summary['status']}",
        "",
        "## Dimension Distribution",
        "",
        "| Width x Height | Images |",
        "|---|---:|",
    ]
    if dimension_rows:
        lines.extend(f"| {name} | {count} |" for name, count in dimension_rows)
    else:
        lines.append("| unavailable | 0 |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `PASS` means the inventory is non-empty and every image is readable, fully decodable, "
            "and matches the expected dimensions.",
            "- Equality with another machine is established only by the separate compare command.",
            "- Exact file SHA-256 equality proves byte-for-byte identity.",
            "- Pixel SHA-256 can distinguish harmless JPEG metadata/re-encoding differences from pixel changes.",
        ]
    )
    ensure_dir(Path(path).parent)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_image_inventory(
    image_root,
    output_dir,
    label,
    workers=None,
    pixel_hash=False,
    decode_check=True,
    expected_width=112,
    expected_height=112,
    max_videos=None,
    progress_every=10000,
    project_root=None,
):
    image_root, image_paths = find_training_image_paths(image_root, max_videos=max_videos)
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    manifest_path = tables_dir / "image_manifest.csv"
    issues_path = tables_dir / "image_issues.csv"

    worker_count = max(1, int(workers or min(16, os.cpu_count() or 1)))
    tasks = (
        (
            str(path),
            path.relative_to(image_root).as_posix(),
            bool(pixel_hash),
            bool(decode_check),
            int(expected_width) if expected_width is not None else None,
            int(expected_height) if expected_height is not None else None,
        )
        for path in image_paths
    )
    counts = Counter()
    dimensions = Counter()
    formats = Counter()
    modes = Counter()
    videos = set()
    total_bytes = 0
    integrity_issue_count = 0
    root_digest = hashlib.sha256()

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_handle, issues_path.open(
        "w", newline="", encoding="utf-8"
    ) as issue_handle:
        manifest_writer = csv.DictWriter(manifest_handle, fieldnames=MANIFEST_FIELDS)
        issue_writer = csv.DictWriter(issue_handle, fieldnames=ISSUE_FIELDS)
        manifest_writer.writeheader()
        issue_writer.writeheader()

        if worker_count == 1:
            results = map(_inspect_image_task, tasks)
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=worker_count)
            results = executor.map(_inspect_image_task, tasks, chunksize=64)
        try:
            for processed_count, row in enumerate(results, start=1):
                manifest_writer.writerow(row)
                videos.add(row["video_id"])
                counts[row["decode_status"]] += 1
                if row["file_size"] != "":
                    total_bytes += int(row["file_size"])
                if row["width"] != "" and row["height"] != "":
                    dimensions[f"{row['width']}x{row['height']}"] += 1
                if row["image_format"]:
                    formats[row["image_format"]] += 1
                if row["image_mode"]:
                    modes[row["image_mode"]] += 1
                _update_root_digest(root_digest, row, include_pixel_hash=bool(pixel_hash))
                if row["issue"]:
                    integrity_issue_count += 1
                    issue_type, _, detail = row["issue"].partition(":")
                    issue_writer.writerow(
                        {
                            "relative_path": row["relative_path"],
                            "video_id": row["video_id"],
                            "file_name": row["file_name"],
                            "issue_type": issue_type,
                            "detail": detail,
                        }
                    )
                if progress_every and processed_count % int(progress_every) == 0:
                    print(
                        f"[IMAGE_INTEGRITY] processed {processed_count}/{len(image_paths)} images",
                        flush=True,
                    )
        finally:
            if executor is not None:
                executor.shutdown()

    project_root = Path(project_root or Path(__file__).resolve().parents[2])
    decode_error_count = counts.get("ERROR", 0)
    if not image_paths:
        inventory_status = "FAIL"
    elif not decode_check:
        inventory_status = "NOT_CHECKED"
    elif integrity_issue_count:
        inventory_status = "FAIL"
    else:
        inventory_status = "PASS"
    summary = {
        "audit": "aligned image integrity inventory",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label": str(label),
        "image_root": str(image_root),
        "video_dir_count": len(videos),
        "image_count": len(image_paths),
        "total_bytes": total_bytes,
        "decode_check_enabled": bool(decode_check),
        "pixel_hash_enabled": bool(pixel_hash),
        "decode_ok_count": counts.get("OK", 0),
        "decode_not_checked_count": counts.get("NOT_CHECKED", 0),
        "decode_error_count": decode_error_count,
        "integrity_issue_count": integrity_issue_count,
        "expected_width": expected_width,
        "expected_height": expected_height,
        "dimension_distribution": dict(sorted(dimensions.items())),
        "format_distribution": dict(sorted(formats.items())),
        "mode_distribution": dict(sorted(modes.items())),
        "root_digest": root_digest.hexdigest(),
        "status": inventory_status,
        "workers": worker_count,
        "max_videos": max_videos,
        "hash_algorithm": "sha256",
        "python": sys.version,
        "platform": platform.platform(),
        "pillow": Image.__version__,
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "manifest_path": str(manifest_path),
        "issues_path": str(issues_path),
    }
    summary_path = output_dir / "inventory_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = reports_dir / "image_inventory_report.md"
    _write_inventory_report(report_path, summary)
    return [manifest_path, issues_path, summary_path, report_path]


def _next_row(reader):
    try:
        return next(reader)
    except StopIteration:
        return None


def _comparison_for_pair(reference, candidate):
    if reference is None:
        return "EXTRA_CANDIDATE", "File exists only in candidate manifest."
    if candidate is None:
        return "MISSING_CANDIDATE", "File exists only in reference manifest."
    if reference.get("decode_status") == "ERROR" or candidate.get("decode_status") == "ERROR":
        return "DECODE_ERROR", "At least one side failed image decoding."
    if reference.get("decode_status") != "OK" or candidate.get("decode_status") != "OK":
        return "DECODE_NOT_CHECKED", "At least one manifest did not perform image decoding."
    if reference.get("issue") or candidate.get("issue"):
        return "INTEGRITY_ISSUE", "At least one inventory row contains a dimension or file-integrity issue."
    if (
        reference.get("file_sha256")
        and reference.get("file_sha256") == candidate.get("file_sha256")
        and reference.get("file_size") == candidate.get("file_size")
    ):
        return "EXACT_MATCH", "Byte-for-byte SHA-256 match."
    reference_pixel = reference.get("pixel_sha256", "")
    candidate_pixel = candidate.get("pixel_sha256", "")
    if reference_pixel and candidate_pixel and reference_pixel == candidate_pixel:
        return "PIXEL_MATCH_BINARY_DIFFERENT", "Decoded RGB pixels match but file bytes differ."
    if not reference_pixel or not candidate_pixel:
        return "BINARY_MISMATCH_PIXEL_UNKNOWN", "File bytes differ and pixel hashes are unavailable."
    return "PIXEL_MISMATCH", "Decoded RGB pixels differ."


def _comparison_row(relative_path, status, reference, candidate, detail):
    reference = reference or {}
    candidate = candidate or {}
    return {
        "relative_path": relative_path,
        "comparison_status": status,
        "reference_file_size": reference.get("file_size", ""),
        "candidate_file_size": candidate.get("file_size", ""),
        "reference_file_sha256": reference.get("file_sha256", ""),
        "candidate_file_sha256": candidate.get("file_sha256", ""),
        "reference_pixel_sha256": reference.get("pixel_sha256", ""),
        "candidate_pixel_sha256": candidate.get("pixel_sha256", ""),
        "reference_decode_status": reference.get("decode_status", ""),
        "candidate_decode_status": candidate.get("decode_status", ""),
        "detail": detail,
    }


def _write_comparison_report(path, summary):
    lines = [
        "# Aligned Image Cross-Machine Comparison",
        "",
        f"- Reference manifest: `{summary['reference_manifest']}`",
        f"- Candidate manifest: `{summary['candidate_manifest']}`",
        f"- Reference rows: {summary['reference_count']}",
        f"- Candidate rows: {summary['candidate_count']}",
        f"- Exact matches: {summary['status_counts'].get('EXACT_MATCH', 0)}",
        f"- Pixel-equivalent binary differences: {summary['status_counts'].get('PIXEL_MATCH_BINARY_DIFFERENT', 0)}",
        f"- Missing candidate files: {summary['status_counts'].get('MISSING_CANDIDATE', 0)}",
        f"- Extra candidate files: {summary['status_counts'].get('EXTRA_CANDIDATE', 0)}",
        f"- Pixel mismatches: {summary['status_counts'].get('PIXEL_MISMATCH', 0)}",
        f"- Decode errors: {summary['status_counts'].get('DECODE_ERROR', 0)}",
        f"- Decode not checked: {summary['status_counts'].get('DECODE_NOT_CHECKED', 0)}",
        f"- Inventory integrity issues: {summary['status_counts'].get('INTEGRITY_ISSUE', 0)}",
        f"- Status: {summary['status']}",
        "",
        "## Decision Rule",
        "",
        "- `EXACT_PASS`: every training-visible JPG is byte-for-byte identical and decodes successfully.",
        "- `PIXEL_EQUIVALENT`: paths and decoded pixels match, but at least one JPEG differs at the byte level.",
        "- `FAIL`: missing/extra files, unchecked/failed decoding, dimension issues, or pixel mismatches are present.",
    ]
    ensure_dir(Path(path).parent)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_image_manifests(reference_manifest, candidate_manifest, output_dir):
    reference_manifest = Path(reference_manifest)
    candidate_manifest = Path(candidate_manifest)
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")
    issues_path = tables_dir / "image_comparison_issues.csv"
    counts = Counter()
    reference_count = 0
    candidate_count = 0

    with reference_manifest.open("r", newline="", encoding="utf-8") as reference_handle, candidate_manifest.open(
        "r", newline="", encoding="utf-8"
    ) as candidate_handle, issues_path.open("w", newline="", encoding="utf-8") as issue_handle:
        reference_reader = csv.DictReader(reference_handle)
        candidate_reader = csv.DictReader(candidate_handle)
        issue_writer = csv.DictWriter(issue_handle, fieldnames=COMPARISON_FIELDS)
        issue_writer.writeheader()
        reference = _next_row(reference_reader)
        candidate = _next_row(candidate_reader)

        while reference is not None or candidate is not None:
            reference_path = reference.get("relative_path") if reference is not None else None
            candidate_path = candidate.get("relative_path") if candidate is not None else None
            if reference is not None and (candidate is None or reference_path < candidate_path):
                status, detail = _comparison_for_pair(reference, None)
                relative_path = reference_path
                reference_count += 1
                issue_writer.writerow(_comparison_row(relative_path, status, reference, None, detail))
                reference = _next_row(reference_reader)
            elif candidate is not None and (reference is None or candidate_path < reference_path):
                status, detail = _comparison_for_pair(None, candidate)
                relative_path = candidate_path
                candidate_count += 1
                issue_writer.writerow(_comparison_row(relative_path, status, None, candidate, detail))
                candidate = _next_row(candidate_reader)
            else:
                status, detail = _comparison_for_pair(reference, candidate)
                relative_path = reference_path
                reference_count += 1
                candidate_count += 1
                if status != "EXACT_MATCH":
                    issue_writer.writerow(
                        _comparison_row(relative_path, status, reference, candidate, detail)
                    )
                reference = _next_row(reference_reader)
                candidate = _next_row(candidate_reader)
            counts[status] += 1

    failing_statuses = {
        "MISSING_CANDIDATE",
        "EXTRA_CANDIDATE",
        "DECODE_ERROR",
        "DECODE_NOT_CHECKED",
        "INTEGRITY_ISSUE",
        "BINARY_MISMATCH_PIXEL_UNKNOWN",
        "PIXEL_MISMATCH",
    }
    if any(counts.get(status, 0) for status in failing_statuses):
        status = "FAIL"
    elif counts.get("PIXEL_MATCH_BINARY_DIFFERENT", 0):
        status = "PIXEL_EQUIVALENT"
    else:
        status = "EXACT_PASS"
    summary = {
        "audit": "aligned image cross-machine comparison",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_manifest": str(reference_manifest),
        "candidate_manifest": str(candidate_manifest),
        "reference_manifest_sha256": _sha256_file(reference_manifest),
        "candidate_manifest_sha256": _sha256_file(candidate_manifest),
        "reference_count": reference_count,
        "candidate_count": candidate_count,
        "status_counts": dict(sorted(counts.items())),
        "status": status,
        "issues_path": str(issues_path),
    }
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = reports_dir / "image_comparison_report.md"
    _write_comparison_report(report_path, summary)
    return [issues_path, summary_path, report_path]
