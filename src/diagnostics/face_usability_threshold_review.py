"""Prepare a dual-lane train-only review template for FACE-S1 thresholds."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


METRIC_FIELDS = [
    "source_presence_status",
    "aligned_failure_status",
    "exposure_status",
    "landmark_success",
    "confidence",
    "landmark_in_frame_ratio",
    "face_hull_coverage",
    "face_hull_visible_ratio",
    "blur_score",
    "gradient_energy",
    "global_mean_luma",
    "low_saturation_ratio",
    "high_saturation_ratio",
    "landmark_pose_yaw_deg",
    "landmark_pose_pitch_deg",
    "landmark_pose_roll_deg",
    "landmark_pose_reprojection_rmse",
    "transform_residual",
    "landmark_jump",
]

REVIEW_FIELDS = [
    "review_id",
    "split",
    "video_id",
    "frame_id",
    "previous_frame_id_for_jump",
    "image_path",
    "selection_reasons",
    "selection_ranks",
    "contact_sheets",
    "source_face_usability_status",
    "source_exclusion_reasons",
    *METRIC_FIELDS,
    "global_face_review_status",
    "global_face_label",
    "local_geometry_review_status",
    "local_geometry_label",
    "temporal_boundary_review_status",
    "temporal_boundary_label",
    "reviewer",
    "review_notes",
]

SUMMARY_FIELDS = ["selection_reason", "contact_rows", "unique_frames"]


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


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _validated_output(manifest, name):
    info = manifest["outputs"][name]
    path = Path(info["path"]).expanduser().resolve()
    if _sha256_file(path) != info["sha256"]:
        raise ValueError(f"{name} differs from recorded manifest hash")
    return path


def _load_selected_frames(frame_path, keys):
    selected = {}
    with Path(frame_path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (row["video_id"], int(row["frame_id"]))
            if key in keys:
                if key in selected:
                    raise ValueError(f"duplicate frame row: {key}")
                selected[key] = row
    missing = sorted(keys - set(selected))
    if missing:
        raise ValueError(f"selected review frames are missing from phase-1 table: {missing[:5]}")
    return selected


def run_threshold_review_template(
    *,
    source_run_dir,
    temporal_review_dir,
    output_dir,
    project_root=None,
):
    """Create a PENDING dual-lane review table without freezing thresholds."""

    source_run_dir = Path(source_run_dir).expanduser().resolve()
    temporal_review_dir = Path(temporal_review_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output_dir must be empty or absent: {output_dir}")

    source_manifest_path = source_run_dir / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("output_status") != "DISTRIBUTION_REVIEW_REQUIRED":
        raise ValueError("source FACE-S1 run is not a completed phase-1 distribution audit")
    if source_manifest.get("threshold_manifest_used") is not False:
        raise ValueError("source FACE-S1 run unexpectedly used thresholds")
    if source_manifest.get("final_face_usable_approval_generated") is not False:
        raise ValueError("source FACE-S1 run unexpectedly approved face_usable")
    contact_path = _validated_output(source_manifest, "contact_review")
    frame_path = _validated_output(source_manifest, "frame_manifest")

    temporal_manifest_path = temporal_review_dir / "run_manifest.json"
    temporal_manifest = json.loads(temporal_manifest_path.read_text(encoding="utf-8"))
    if temporal_manifest.get("review_status") != "PENDING":
        raise ValueError("temporal review must remain PENDING")
    if temporal_manifest.get("threshold_manifest_generated") is not False:
        raise ValueError("temporal review unexpectedly generated thresholds")
    pair_path = _validated_output(temporal_manifest, "pair_review")

    contact_rows = _read_csv(contact_path)
    if not contact_rows:
        raise ValueError("source contact review is empty")
    if any(row.get("split") != "train" or row.get("review_status") != "PENDING" for row in contact_rows):
        raise ValueError("threshold review inputs must be train-only and PENDING")

    grouped = defaultdict(lambda: {"reasons": [], "ranks": [], "sheets": []})
    reason_contacts = Counter()
    reason_keys = defaultdict(set)
    for row in contact_rows:
        key = (row["video_id"], int(row["frame_id"]))
        reason = row["selection_reason"]
        grouped[key]["reasons"].append(reason)
        grouped[key]["ranks"].append(f"{reason}:{row['rank']}")
        grouped[key]["sheets"].append(row.get("contact_sheet", ""))
        reason_contacts[reason] += 1
        reason_keys[reason].add(key)

    pair_by_key = {}
    for row in _read_csv(pair_path):
        if row.get("split") != "train" or row.get("review_status") != "PENDING":
            raise ValueError("paired jump review must be train-only and PENDING")
        key = (row["video_id"], int(row["current_frame_id"]))
        pair_by_key[key] = int(row["previous_frame_id"])
    jump_keys = {key for key, value in grouped.items() if "landmark_jump_high" in value["reasons"]}
    if set(pair_by_key) != jump_keys:
        raise ValueError("paired jump review does not exactly cover jump candidates")

    source_rows = _load_selected_frames(frame_path, set(grouped))
    review_rows = []
    for index, key in enumerate(sorted(grouped), start=1):
        source = source_rows[key]
        selection = grouped[key]
        row = {
            "review_id": f"FACE-S1-R{index:04d}",
            "split": source["split"],
            "video_id": source["video_id"],
            "frame_id": source["frame_id"],
            "previous_frame_id_for_jump": pair_by_key.get(key, ""),
            "image_path": source["image_path"],
            "selection_reasons": ";".join(sorted(set(selection["reasons"]))),
            "selection_ranks": ";".join(sorted(selection["ranks"])),
            "contact_sheets": ";".join(sorted(set(selection["sheets"]))),
            "source_face_usability_status": source["face_usability_status"],
            "source_exclusion_reasons": source["exclusion_reasons"],
            **{field: source.get(field, "") for field in METRIC_FIELDS},
            "global_face_review_status": "PENDING",
            "global_face_label": "",
            "local_geometry_review_status": "PENDING",
            "local_geometry_label": "",
            "temporal_boundary_review_status": "PENDING",
            "temporal_boundary_label": "",
            "reviewer": "",
            "review_notes": "",
        }
        review_rows.append(row)

    summary_rows = [
        {
            "selection_reason": reason,
            "contact_rows": reason_contacts[reason],
            "unique_frames": len(reason_keys[reason]),
        }
        for reason in sorted(reason_contacts)
    ]
    summary_rows.append(
        {
            "selection_reason": "ALL_DEDUPLICATED",
            "contact_rows": len(contact_rows),
            "unique_frames": len(review_rows),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    review_path = _write_csv(
        output_dir / "tables" / "face_usability_threshold_review_template.csv",
        review_rows,
        REVIEW_FIELDS,
    )
    summary_path = _write_csv(
        output_dir / "tables" / "face_usability_threshold_review_selection_summary.csv",
        summary_rows,
        SUMMARY_FIELDS,
    )
    report_path = output_dir / "reports" / "face_usability_threshold_review_instructions.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# FACE-S1 Dual-Lane Threshold Review Instructions\n\n"
        f"- Source contact rows: {len(contact_rows)}\n"
        f"- Deduplicated train frames: {len(review_rows)}\n"
        f"- Paired jump frames: {len(pair_by_key)}\n"
        "- This table is PENDING and does not freeze thresholds.\n\n"
        "## Required labels\n\n"
        "`global_face_label`: `global_usable`, `global_unusable_no_face`, "
        "`global_unusable_major_occlusion`, `global_unusable_extreme_pose`, "
        "`global_unusable_out_of_frame`, `global_unusable_blur`, "
        "`global_unusable_other`, or `global_uncertain`.\n\n"
        "`local_geometry_label`: `local_geometry_candidate`, "
        "`local_geometry_ineligible_landmark_missing`, "
        "`local_geometry_ineligible_extreme_pose`, "
        "`local_geometry_ineligible_out_of_frame`, "
        "`local_geometry_ineligible_unstable`, "
        "`local_geometry_ineligible_other`, or `local_geometry_uncertain`.\n\n"
        "`temporal_boundary_label`: `boundary`, `no_boundary`, or `boundary_uncertain`.\n\n"
        "Set each corresponding review-status field to `REVIEWED` only after visual "
        "inspection. A frame may be global_usable but local-geometry-ineligible. "
        "This review does not approve any brow/eye/nose/mouth crop; per-region "
        "eligibility requires the later frozen polygon/margin overlay gate. Landmark "
        "jump and blur cannot independently force a boundary.\n",
        encoding="utf-8",
    )

    manifest = {
        "audit": "FACE-S1 train-only dual-lane threshold review template",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(shlex.quote(item) for item in sys.argv),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "python": sys.version,
        "platform": platform.platform(),
        "review_status": "PENDING",
        "threshold_manifest_generated": False,
        "face_usable_approval_generated": False,
        "dual_lane_contract": True,
        "local_geometry_only": True,
        "local_crop_approval_generated": False,
        "source_contact_row_count": len(contact_rows),
        "deduplicated_review_frame_count": len(review_rows),
        "paired_jump_frame_count": len(pair_by_key),
        "inputs": {
            "source_run_manifest": {
                "path": str(source_manifest_path),
                "sha256": _sha256_file(source_manifest_path),
            },
            "source_contact_review": {"path": str(contact_path), "sha256": _sha256_file(contact_path)},
            "source_frame_manifest": {"path": str(frame_path), "sha256": _sha256_file(frame_path)},
            "temporal_review_manifest": {
                "path": str(temporal_manifest_path),
                "sha256": _sha256_file(temporal_manifest_path),
            },
            "temporal_pair_review": {"path": str(pair_path), "sha256": _sha256_file(pair_path)},
        },
        "implementation_files": {
            "core": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "cli": {
                "path": str(project_root / "scripts" / "prepare_face_usability_threshold_review.py"),
                "sha256": _sha256_file(
                    project_root / "scripts" / "prepare_face_usability_threshold_review.py"
                ),
            },
        },
        "outputs": {
            "review_template": {"path": str(review_path), "sha256": _sha256_file(review_path)},
            "selection_summary": {"path": str(summary_path), "sha256": _sha256_file(summary_path)},
            "instructions": {"path": str(report_path), "sha256": _sha256_file(report_path)},
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [review_path, summary_path, report_path, manifest_path]
