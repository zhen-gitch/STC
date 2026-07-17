"""Read-only paired-frame review for FACE-S1 landmark jump candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.diagnostics.au_coordinate_contract import read_landmark_rows
from src.diagnostics.au_region_tracking import extract_frame_id
from src.diagnostics.face_usability import normalize_landmarks


PAIR_FIELDS = [
    "rank",
    "split",
    "video_id",
    "previous_frame_id",
    "current_frame_id",
    "previous_image_path",
    "current_image_path",
    "source_landmark_jump",
    "recomputed_landmark_jump",
    "absolute_difference",
    "review_status",
    "review_label",
    "review_notes",
]


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


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _points(landmark_row):
    if landmark_row.get("success") != 1 or len(landmark_row.get("landmarks", {})) != 68:
        return None
    return np.stack([landmark_row["landmarks"][index] for index in range(68)])


def _frame_panel(image_path, points, frame_id, role, width=160):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    scale = width / image.width
    height = round(image.height * scale)
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    scaled = np.asarray(points, dtype=np.float64) * scale
    for x, y in scaled:
        draw.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=(0, 255, 0))
    x0, y0 = scaled.min(axis=0)
    x1, y1 = scaled.max(axis=0)
    draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=1)
    canvas = Image.new("RGB", (width, height + 24), "white")
    canvas.paste(image, (0, 0))
    ImageDraw.Draw(canvas).text((3, height + 4), f"{role}: frame {frame_id}", fill="black")
    return canvas


def _pair_panel(previous_path, current_path, previous_points, current_points, row):
    previous = _frame_panel(previous_path, previous_points, row["previous_frame_id"], "t-1")
    current = _frame_panel(current_path, current_points, row["current_frame_id"], "t")
    label_height = 36
    canvas = Image.new(
        "RGB",
        (previous.width + current.width, previous.height + label_height),
        "white",
    )
    canvas.paste(previous, (0, 0))
    canvas.paste(current, (previous.width, 0))
    label = (
        f"{row['video_id']} rank={row['rank']}\n"
        f"jump={float(row['recomputed_landmark_jump']):.6f}"
    )
    ImageDraw.Draw(canvas).text((3, previous.height + 2), label, fill="black")
    return canvas


def _write_sheet(path, panels, columns=4):
    if not panels:
        raise ValueError("no landmark-jump panels were generated")
    columns = max(1, int(columns))
    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    row_count = math.ceil(len(panels) / columns)
    canvas = Image.new("RGB", (columns * cell_width, row_count * cell_height + 26), "white")
    ImageDraw.Draw(canvas).text(
        (4, 5),
        "FACE-S1 train-only landmark jump: paired t-1 / t review",
        fill="black",
    )
    for index, panel in enumerate(panels):
        x = (index % columns) * cell_width
        y = 26 + (index // columns) * cell_height
        canvas.paste(panel, (x, y))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=92, subsampling=0, optimize=False)
    return path


def run_landmark_jump_pair_review(
    *,
    source_run_dir,
    aligned_landmark_root,
    output_dir,
    columns=4,
    project_root=None,
):
    """Generate paired evidence for train-only landmark-jump candidates."""

    source_run_dir = Path(source_run_dir).expanduser().resolve()
    aligned_landmark_root = Path(aligned_landmark_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output_dir must be empty or absent: {output_dir}")

    source_manifest_path = source_run_dir / "run_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("output_status") != "DISTRIBUTION_REVIEW_REQUIRED":
        raise ValueError("source FACE-S1 run is not a completed distribution review")
    if source_manifest.get("threshold_manifest_used") is not False:
        raise ValueError("source FACE-S1 run unexpectedly used a threshold manifest")
    if source_manifest.get("final_face_usable_approval_generated") is not False:
        raise ValueError("source FACE-S1 run unexpectedly generated face_usable approval")

    contact_info = source_manifest["outputs"]["contact_review"]
    contact_path = Path(contact_info["path"]).expanduser().resolve()
    if _sha256_file(contact_path) != contact_info["sha256"]:
        raise ValueError("source contact review differs from FACE-S1 manifest hash")
    candidates = [
        row for row in _read_csv(contact_path) if row.get("selection_reason") == "landmark_jump_high"
    ]
    if not candidates:
        raise ValueError("source contact review has no landmark_jump_high candidates")
    if any(row.get("split") != "train" or row.get("review_status") != "PENDING" for row in candidates):
        raise ValueError("landmark jump candidates must be train-only and PENDING")

    landmark_cache = {}
    image_cache = {}
    rows = []
    panels = []
    for candidate in sorted(candidates, key=lambda row: int(row["rank"])):
        video_id = candidate["video_id"]
        current_frame_id = int(candidate["frame_id"])
        previous_frame_id = current_frame_id - 1
        if video_id not in landmark_cache:
            landmark_cache[video_id] = read_landmark_rows(
                aligned_landmark_root / f"{video_id}.csv"
            )
        landmark_rows = landmark_cache[video_id]
        previous_points = _points(landmark_rows.get(previous_frame_id, {}))
        current_points = _points(landmark_rows.get(current_frame_id, {}))
        if previous_points is None or current_points is None:
            raise ValueError(f"jump candidate lacks valid consecutive landmarks: {video_id}:{current_frame_id}")

        image_dir = Path(candidate["image_path"]).expanduser().resolve().parent
        if video_id not in image_cache:
            paths = sorted(image_dir.glob("*.jpg"))
            image_cache[video_id] = {extract_frame_id(path): path for path in paths}
        previous_path = image_cache[video_id].get(previous_frame_id)
        current_path = image_cache[video_id].get(current_frame_id)
        if previous_path is None or current_path is None:
            raise FileNotFoundError(f"jump candidate lacks consecutive JPGs: {video_id}:{current_frame_id}")

        previous_normalized = normalize_landmarks(previous_points)
        current_normalized = normalize_landmarks(current_points)
        if previous_normalized is None or current_normalized is None:
            raise ValueError(f"jump candidate cannot be normalized: {video_id}:{current_frame_id}")
        recomputed = float(
            np.median(
                np.linalg.norm(current_normalized[17:68] - previous_normalized[17:68], axis=1)
            )
        )
        source_value = float(candidate["metric_value"])
        difference = abs(recomputed - source_value)
        if difference > 1e-5:
            raise ValueError(f"jump metric differs from source frame table: {video_id}:{current_frame_id}")
        row = {
            "rank": int(candidate["rank"]),
            "split": candidate["split"],
            "video_id": video_id,
            "previous_frame_id": previous_frame_id,
            "current_frame_id": current_frame_id,
            "previous_image_path": str(previous_path),
            "current_image_path": str(current_path),
            "source_landmark_jump": f"{source_value:.6f}",
            "recomputed_landmark_jump": f"{recomputed:.6f}",
            "absolute_difference": f"{difference:.9f}",
            "review_status": "PENDING",
            "review_label": "",
            "review_notes": "",
        }
        rows.append(row)
        panels.append(
            _pair_panel(previous_path, current_path, previous_points, current_points, row)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = _write_csv(output_dir / "tables" / "landmark_jump_pair_review.csv", rows)
    sheet_path = _write_sheet(
        output_dir / "contact_sheets" / "landmark_jump_pairs.jpg",
        panels,
        columns=columns,
    )
    report_path = output_dir / "reports" / "landmark_jump_pair_review.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# FACE-S1 Landmark Jump Paired Review\n\n"
        f"- Train-only candidate pairs: {len(rows)}\n"
        "- Each panel shows the exact consecutive frames used by the jump metric.\n"
        "- Source and recomputed jump values agree within 1e-5.\n"
        "- Review remains PENDING; this output does not freeze a threshold or approve face_usable.\n",
        encoding="utf-8",
    )

    manifest = {
        "audit": "FACE-S1 train-only paired landmark-jump review",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(shlex.quote(item) for item in sys.argv),
        "git_commit": _git_value(project_root, ["rev-parse", "HEAD"]),
        "git_branch": _git_value(project_root, ["branch", "--show-current"]),
        "git_status_short": _git_value(project_root, ["status", "--short"]),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "review_status": "PENDING",
        "threshold_manifest_generated": False,
        "face_usable_approval_generated": False,
        "pair_count": len(rows),
        "inputs": {
            "source_run_manifest": {
                "path": str(source_manifest_path),
                "sha256": _sha256_file(source_manifest_path),
            },
            "source_contact_review": {
                "path": str(contact_path),
                "sha256": _sha256_file(contact_path),
            },
            "aligned_landmark_root": str(aligned_landmark_root),
        },
        "implementation_files": {
            "core": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "cli": {
                "path": str(project_root / "scripts" / "audit_face_usability_temporal_review.py"),
                "sha256": _sha256_file(
                    project_root / "scripts" / "audit_face_usability_temporal_review.py"
                ),
            },
        },
        "outputs": {
            "pair_review": {"path": str(table_path), "sha256": _sha256_file(table_path)},
            "contact_sheet": {"path": str(sheet_path), "sha256": _sha256_file(sheet_path)},
            "report": {"path": str(report_path), "sha256": _sha256_file(report_path)},
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [table_path, sheet_path, report_path, manifest_path]
