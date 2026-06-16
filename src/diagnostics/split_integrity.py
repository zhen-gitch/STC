import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, read_prediction_table, severity_group, write_csv_rows


SUBJECT_RE = re.compile(r"(?P<subject>\d{3}_\d)")
TASK_RE = re.compile(r"^\d{3}_\d_(?P<task>.+?)(?:_video(?:_aligned)?|_aligned)?$")

MANIFEST_FIELDS = [
    "split",
    "video_id",
    "subject_id",
    "task_name",
    "label_path",
    "true_bdi",
    "severity_group",
    "image_dir",
    "image_exists",
    "duplicate_video_id",
    "subject_split_count",
    "issues",
]

OVERLAP_FIELDS = ["subject_id", "splits", "video_count", "video_ids"]
DISTRIBUTION_FIELDS = [
    "split",
    "severity_group",
    "video_count",
    "subject_count",
    "mean_bdi",
    "min_bdi",
    "max_bdi",
]
PREDICTION_ALIGNMENT_FIELDS = [
    "prediction_video_id",
    "prediction_subject_id",
    "matched_split",
    "matched_video_id",
    "status",
]


def infer_subject_id(video_id):
    match = SUBJECT_RE.search(str(video_id))
    if match:
        return match.group("subject")
    return str(video_id)[:5]


def infer_task_name(video_id):
    match = TASK_RE.match(str(video_id))
    if match:
        return match.group("task")
    return ""


def normalize_video_id(video_id):
    value = Path(str(video_id)).name
    for suffix in ("_aligned", "_video_aligned"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            if suffix == "_video_aligned":
                value = f"{value}_video"
            break
    return value


def compatible_image_dirs(image_root, video_id):
    image_root = Path(image_root)
    normalized = normalize_video_id(video_id)
    candidates = [
        str(video_id),
        normalized,
        f"{normalized}_aligned",
    ]
    if normalized.endswith("_video"):
        candidates.append(f"{normalized}_aligned")
    else:
        candidates.extend([f"{normalized}_video", f"{normalized}_video_aligned"])

    seen = set()
    paths = []
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        paths.append(image_root / name)
    return paths


def load_split_entries(split_file):
    split_file = Path(split_file)
    with split_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = []
    for split, entries in payload.items():
        if entries is None:
            continue
        for entry in entries:
            video_id = Path(str(entry)).name
            rows.append(
                {
                    "split": str(split),
                    "video_id": video_id,
                    "subject_id": infer_subject_id(video_id),
                    "task_name": infer_task_name(video_id),
                }
            )
    return rows


def read_label(label_dir, subject_id):
    label_path = Path(label_dir) / f"{subject_id}_Depression.csv"
    if not label_path.exists():
        return label_path, None, "missing_label"
    try:
        return label_path, float(label_path.read_text(encoding="utf-8").strip()), ""
    except ValueError:
        return label_path, None, "invalid_label"


def _format_number(value):
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"


def build_split_manifest(split_file, label_dir, image_root=None):
    entries = load_split_entries(split_file)
    video_counts = Counter(row["video_id"] for row in entries)
    subject_splits = defaultdict(set)
    for row in entries:
        subject_splits[row["subject_id"]].add(row["split"])

    manifest = []
    for row in entries:
        label_path, label, label_issue = read_label(label_dir, row["subject_id"])
        image_dir = ""
        image_exists = ""
        image_issue = ""
        if image_root is not None:
            matches = [path for path in compatible_image_dirs(image_root, row["video_id"]) if path.exists()]
            image_dir = str(matches[0]) if matches else str(Path(image_root) / row["video_id"])
            image_exists = bool(matches)
            if not matches:
                image_issue = "missing_image_dir"

        issues = []
        if video_counts[row["video_id"]] > 1:
            issues.append("duplicate_video_id")
        if len(subject_splits[row["subject_id"]]) > 1:
            issues.append("subject_split_overlap")
        if label_issue:
            issues.append(label_issue)
        if image_issue:
            issues.append(image_issue)

        manifest.append(
            {
                "split": row["split"],
                "video_id": row["video_id"],
                "subject_id": row["subject_id"],
                "task_name": row["task_name"],
                "label_path": str(label_path),
                "true_bdi": _format_number(label),
                "severity_group": severity_group(label) if label is not None else "",
                "image_dir": image_dir,
                "image_exists": str(image_exists) if image_root is not None else "",
                "duplicate_video_id": str(video_counts[row["video_id"]] > 1),
                "subject_split_count": str(len(subject_splits[row["subject_id"]])),
                "issues": ";".join(issues),
            }
        )
    return manifest


def summarize_subject_overlap(manifest):
    grouped = defaultdict(list)
    for row in manifest:
        grouped[row["subject_id"]].append(row)

    rows = []
    for subject_id, subject_rows in sorted(grouped.items()):
        splits = sorted({row["split"] for row in subject_rows})
        if len(splits) <= 1:
            continue
        rows.append(
            {
                "subject_id": subject_id,
                "splits": ";".join(splits),
                "video_count": str(len(subject_rows)),
                "video_ids": ";".join(sorted(row["video_id"] for row in subject_rows)),
            }
        )
    return rows


def summarize_label_distribution(manifest):
    grouped = defaultdict(list)
    for row in manifest:
        if not row["true_bdi"]:
            continue
        grouped[(row["split"], row["severity_group"])].append(row)

    rows = []
    for (split, group_name), group_rows in sorted(grouped.items()):
        values = np.asarray([float(row["true_bdi"]) for row in group_rows], dtype=float)
        rows.append(
            {
                "split": split,
                "severity_group": group_name,
                "video_count": str(len(group_rows)),
                "subject_count": str(len({row["subject_id"] for row in group_rows})),
                "mean_bdi": f"{float(np.mean(values)):.6f}",
                "min_bdi": f"{float(np.min(values)):.6f}",
                "max_bdi": f"{float(np.max(values)):.6f}",
            }
        )
    return rows


def align_predictions_to_split(predictions_csv, manifest):
    predictions = read_prediction_table(predictions_csv)
    by_video = defaultdict(list)
    for row in manifest:
        by_video[normalize_video_id(row["video_id"])].append(row)

    rows = []
    for pred in predictions:
        key = normalize_video_id(pred["video_id"])
        matches = by_video.get(key, [])
        if len(matches) == 1:
            status = "matched"
            matched_split = matches[0]["split"]
            matched_video_id = matches[0]["video_id"]
        elif len(matches) > 1:
            status = "ambiguous"
            matched_split = ";".join(sorted({row["split"] for row in matches}))
            matched_video_id = ";".join(sorted(row["video_id"] for row in matches))
        else:
            status = "missing_in_split"
            matched_split = ""
            matched_video_id = ""
        rows.append(
            {
                "prediction_video_id": pred["video_id"],
                "prediction_subject_id": pred["subject_id"],
                "matched_split": matched_split,
                "matched_video_id": matched_video_id,
                "status": status,
            }
        )
    return rows


def summarize_counts(manifest, overlap_rows, prediction_alignment=None, image_root_provided=False):
    issues = Counter()
    for row in manifest:
        for issue in filter(None, row["issues"].split(";")):
            issues[issue] += 1

    prediction_mismatches = 0
    if prediction_alignment is not None:
        prediction_mismatches = sum(1 for row in prediction_alignment if row["status"] != "matched")

    return {
        "split_count": len({row["split"] for row in manifest}),
        "video_count": len(manifest),
        "subject_count": len({row["subject_id"] for row in manifest}),
        "overlapping_subject_count": len(overlap_rows),
        "duplicate_video_count": issues["duplicate_video_id"],
        "missing_label_count": issues["missing_label"],
        "invalid_label_count": issues["invalid_label"],
        "missing_image_dir_count": issues["missing_image_dir"] if image_root_provided else 0,
        "prediction_mismatch_count": prediction_mismatches,
    }


def write_split_integrity_report(report_path, counts, generated_tables):
    critical = [
        counts["overlapping_subject_count"],
        counts["duplicate_video_count"],
        counts["missing_label_count"],
        counts["invalid_label_count"],
        counts["missing_image_dir_count"],
        counts["prediction_mismatch_count"],
    ]
    status = "PASS" if sum(critical) == 0 else "REVIEW_REQUIRED"
    lines = [
        "# Split Integrity Audit Report",
        "",
        f"- status: {status}",
        f"- splits: {counts['split_count']}",
        f"- videos: {counts['video_count']}",
        f"- subjects: {counts['subject_count']}",
        f"- overlapping_subjects: {counts['overlapping_subject_count']}",
        f"- duplicate_video_rows: {counts['duplicate_video_count']}",
        f"- missing_labels: {counts['missing_label_count']}",
        f"- invalid_labels: {counts['invalid_label_count']}",
        f"- missing_image_dirs: {counts['missing_image_dir_count']}",
        f"- prediction_alignment_mismatches: {counts['prediction_mismatch_count']}",
        "",
        "## Generated Tables",
        "",
    ]
    lines.extend(f"- {path}" for path in generated_tables)
    ensure_dir(Path(report_path).parent)
    Path(report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_split_integrity_audit(split_file, label_dir, output_dir, image_root=None, predictions_csv=None):
    output_dir = ensure_dir(output_dir)
    tables_dir = ensure_dir(output_dir / "tables")
    reports_dir = ensure_dir(output_dir / "reports")

    manifest = build_split_manifest(split_file, label_dir, image_root=image_root)
    overlap_rows = summarize_subject_overlap(manifest)
    distribution_rows = summarize_label_distribution(manifest)

    manifest_path = tables_dir / "split_video_manifest.csv"
    overlap_path = tables_dir / "split_subject_overlap.csv"
    distribution_path = tables_dir / "split_label_distribution.csv"

    write_csv_rows(manifest_path, manifest, MANIFEST_FIELDS)
    write_csv_rows(overlap_path, overlap_rows, OVERLAP_FIELDS)
    write_csv_rows(distribution_path, distribution_rows, DISTRIBUTION_FIELDS)

    generated = [manifest_path, overlap_path, distribution_path]
    prediction_alignment = None
    if predictions_csv is not None:
        prediction_alignment = align_predictions_to_split(predictions_csv, manifest)
        prediction_path = tables_dir / "split_prediction_alignment.csv"
        write_csv_rows(prediction_path, prediction_alignment, PREDICTION_ALIGNMENT_FIELDS)
        generated.append(prediction_path)

    counts = summarize_counts(
        manifest,
        overlap_rows,
        prediction_alignment=prediction_alignment,
        image_root_provided=image_root is not None,
    )
    report_path = reports_dir / "split_integrity_report.md"
    write_split_integrity_report(report_path, counts, generated)
    generated.append(report_path)
    return generated
