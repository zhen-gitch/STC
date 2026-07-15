import csv
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from src.diagnostics.image_integrity import (
    compare_image_manifests,
    find_training_image_paths,
    run_image_inventory,
)


def _write_jpeg(path, color=(20, 40, 60), size=(112, 112)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, format="JPEG", quality=95)
    return path


def _read_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_image_integrity_import_does_not_require_numpy():
    project_root = Path(__file__).resolve().parents[1]
    code = """
import builtins
original_import = builtins.__import__
def deny_numpy(name, *args, **kwargs):
    if name == 'numpy' or name.startswith('numpy.'):
        raise ModuleNotFoundError("numpy intentionally unavailable")
    return original_import(name, *args, **kwargs)
builtins.__import__ = deny_numpy
import src.diagnostics.image_integrity
print('ok')
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "ok"


def test_find_training_image_paths_ignores_nested_notebook_files(tmp_path):
    image_root = tmp_path / "images"
    visible = _write_jpeg(image_root / "video_aligned/frame_000001.jpg")
    _write_jpeg(image_root / "video_aligned/.ipynb_checkpoints/frame_000002.jpg")

    resolved_root, paths = find_training_image_paths(image_root)

    assert resolved_root == image_root.resolve()
    assert paths == [visible]


def test_find_training_image_paths_supports_deterministic_video_limit(tmp_path):
    image_root = tmp_path / "images"
    first = _write_jpeg(image_root / "a_video_aligned/frame_000001.jpg")
    _write_jpeg(image_root / "b_video_aligned/frame_000001.jpg")

    _resolved_root, paths = find_training_image_paths(image_root, max_videos=1)

    assert paths == [first]


def test_empty_inventory_fails_instead_of_vacuously_passing(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()

    run_image_inventory(
        image_root=image_root,
        output_dir=tmp_path / "audit",
        label="empty",
        workers=1,
    )

    summary = _read_json(tmp_path / "audit/inventory_summary.json")
    assert summary["image_count"] == 0
    assert summary["status"] == "FAIL"


def test_inventory_hashes_and_decodes_images_and_reports_corruption(tmp_path):
    image_root = tmp_path / "images"
    _write_jpeg(image_root / "video_a_aligned/frame_000001.jpg")
    corrupt = image_root / "video_a_aligned/frame_000002.jpg"
    corrupt.write_bytes(b"not-a-jpeg")

    generated = run_image_inventory(
        image_root=image_root,
        output_dir=tmp_path / "audit",
        label="local",
        workers=1,
        pixel_hash=True,
    )

    assert len(generated) == 4
    rows = _read_rows(tmp_path / "audit/tables/image_manifest.csv")
    assert len(rows) == 2
    valid = next(row for row in rows if row["file_name"] == "frame_000001.jpg")
    invalid = next(row for row in rows if row["file_name"] == "frame_000002.jpg")
    assert valid["decode_status"] == "OK"
    assert valid["width"] == "112"
    assert valid["height"] == "112"
    assert valid["file_sha256"]
    assert valid["pixel_sha256"]
    assert invalid["decode_status"] == "ERROR"
    assert "image_decode_error" in invalid["issue"]
    summary = _read_json(tmp_path / "audit/inventory_summary.json")
    assert summary["image_count"] == 2
    assert summary["decode_ok_count"] == 1
    assert summary["decode_error_count"] == 1
    assert summary["status"] == "FAIL"
    issues = _read_rows(tmp_path / "audit/tables/image_issues.csv")
    assert len(issues) == 1
    assert issues[0]["issue_type"] == "image_decode_error"


def test_inventory_fails_decodable_image_with_unexpected_dimensions(tmp_path):
    image_root = tmp_path / "images"
    _write_jpeg(image_root / "video_a_aligned/frame_000001.jpg", size=(100, 112))

    run_image_inventory(
        image_root=image_root,
        output_dir=tmp_path / "audit",
        label="local",
        workers=1,
        pixel_hash=True,
    )

    row = _read_rows(tmp_path / "audit/tables/image_manifest.csv")[0]
    assert row["decode_status"] == "OK"
    assert row["width"] == "100"
    assert row["height"] == "112"
    assert row["issue"].startswith("unexpected_dimensions:")
    summary = _read_json(tmp_path / "audit/inventory_summary.json")
    assert summary["integrity_issue_count"] == 1
    assert summary["status"] == "FAIL"


def test_compare_reports_exact_pass_for_identical_manifests(tmp_path):
    image_root = tmp_path / "images"
    _write_jpeg(image_root / "video_a_aligned/frame_000001.jpg")
    _write_jpeg(image_root / "video_b_aligned/frame_000001.jpg", color=(80, 20, 10))
    run_image_inventory(image_root, tmp_path / "reference", "server", workers=1, pixel_hash=True)
    run_image_inventory(image_root, tmp_path / "candidate", "local", workers=1, pixel_hash=True)

    compare_image_manifests(
        tmp_path / "reference/tables/image_manifest.csv",
        tmp_path / "candidate/tables/image_manifest.csv",
        tmp_path / "comparison",
    )

    summary = _read_json(tmp_path / "comparison/comparison_summary.json")
    assert summary["status"] == "EXACT_PASS"
    assert summary["status_counts"] == {"EXACT_MATCH": 2}
    assert _read_rows(tmp_path / "comparison/tables/image_comparison_issues.csv") == []


def test_compare_distinguishes_pixel_equivalent_jpeg_bytes(tmp_path):
    reference_root = tmp_path / "reference_images"
    candidate_root = tmp_path / "candidate_images"
    reference = _write_jpeg(reference_root / "video_a_aligned/frame_000001.jpg")
    candidate = candidate_root / "video_a_aligned/frame_000001.jpg"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(reference.read_bytes() + b"harmless-trailing-bytes")
    run_image_inventory(reference_root, tmp_path / "reference", "server", workers=1, pixel_hash=True)
    run_image_inventory(candidate_root, tmp_path / "candidate", "local", workers=1, pixel_hash=True)

    compare_image_manifests(
        tmp_path / "reference/tables/image_manifest.csv",
        tmp_path / "candidate/tables/image_manifest.csv",
        tmp_path / "comparison",
    )

    summary = _read_json(tmp_path / "comparison/comparison_summary.json")
    assert summary["status"] == "PIXEL_EQUIVALENT"
    assert summary["status_counts"] == {"PIXEL_MATCH_BINARY_DIFFERENT": 1}
    issue = _read_rows(tmp_path / "comparison/tables/image_comparison_issues.csv")[0]
    assert issue["comparison_status"] == "PIXEL_MATCH_BINARY_DIFFERENT"


def test_compare_fails_for_missing_extra_and_pixel_changed_images(tmp_path):
    reference_root = tmp_path / "reference_images"
    candidate_root = tmp_path / "candidate_images"
    _write_jpeg(reference_root / "video_a_aligned/frame_000001.jpg", color=(10, 10, 10))
    _write_jpeg(reference_root / "video_a_aligned/frame_000002.jpg", color=(20, 20, 20))
    _write_jpeg(candidate_root / "video_a_aligned/frame_000001.jpg", color=(200, 10, 10))
    _write_jpeg(candidate_root / "video_a_aligned/frame_000003.jpg", color=(30, 30, 30))
    run_image_inventory(reference_root, tmp_path / "reference", "server", workers=1, pixel_hash=True)
    run_image_inventory(candidate_root, tmp_path / "candidate", "local", workers=1, pixel_hash=True)

    compare_image_manifests(
        tmp_path / "reference/tables/image_manifest.csv",
        tmp_path / "candidate/tables/image_manifest.csv",
        tmp_path / "comparison",
    )

    summary = _read_json(tmp_path / "comparison/comparison_summary.json")
    assert summary["status"] == "FAIL"
    assert summary["status_counts"]["PIXEL_MISMATCH"] == 1
    assert summary["status_counts"]["MISSING_CANDIDATE"] == 1
    assert summary["status_counts"]["EXTRA_CANDIDATE"] == 1


def test_compare_rejects_exact_but_wrong_sized_images(tmp_path):
    image_root = tmp_path / "images"
    _write_jpeg(image_root / "video_a_aligned/frame_000001.jpg", size=(100, 112))
    run_image_inventory(image_root, tmp_path / "reference", "server", workers=1, pixel_hash=True)
    run_image_inventory(image_root, tmp_path / "candidate", "local", workers=1, pixel_hash=True)

    compare_image_manifests(
        tmp_path / "reference/tables/image_manifest.csv",
        tmp_path / "candidate/tables/image_manifest.csv",
        tmp_path / "comparison",
    )

    summary = _read_json(tmp_path / "comparison/comparison_summary.json")
    assert summary["status"] == "FAIL"
    assert summary["status_counts"] == {"INTEGRITY_ISSUE": 1}
