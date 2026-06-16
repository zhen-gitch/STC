import csv
import json

from src.diagnostics.io import write_prediction_table
from src.diagnostics.split_integrity import (
    align_predictions_to_split,
    build_split_manifest,
    run_split_integrity_audit,
    summarize_subject_overlap,
)


def _write_split(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_label(label_dir, subject_id, score):
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / f"{subject_id}_Depression.csv").write_text(str(score), encoding="utf-8")


def _make_image_dir(image_root, video_id):
    video_dir = image_root / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir


def test_run_split_integrity_audit_writes_expected_outputs(tmp_path):
    split_file = _write_split(
        tmp_path / "split.json",
        {
            "train": ["203_1_Freeform_video", "203_1_Northwind_video"],
            "val": ["204_1_Freeform_video"],
            "test": ["205_1_Freeform_video"],
        },
    )
    label_dir = tmp_path / "labels"
    _write_label(label_dir, "203_1", 8)
    _write_label(label_dir, "204_1", 18)
    _write_label(label_dir, "205_1", 30)

    image_root = tmp_path / "images"
    _make_image_dir(image_root, "203_1_Freeform_video_aligned")
    _make_image_dir(image_root, "203_1_Northwind_video_aligned")
    _make_image_dir(image_root, "204_1_Freeform_video_aligned")
    _make_image_dir(image_root, "205_1_Freeform_video_aligned")

    generated = run_split_integrity_audit(
        split_file=split_file,
        label_dir=label_dir,
        image_root=image_root,
        output_dir=tmp_path / "split_integrity",
    )

    assert generated
    tables = tmp_path / "split_integrity" / "tables"
    reports = tmp_path / "split_integrity" / "reports"
    assert (tables / "split_video_manifest.csv").exists()
    assert (tables / "split_subject_overlap.csv").exists()
    assert (tables / "split_label_distribution.csv").exists()
    assert (reports / "split_integrity_report.md").exists()

    with (tables / "split_subject_overlap.csv").open(newline="", encoding="utf-8") as f:
        overlap_rows = list(csv.DictReader(f))
    assert overlap_rows == []

    report = (reports / "split_integrity_report.md").read_text(encoding="utf-8")
    assert "status: PASS" in report


def test_split_integrity_detects_subject_overlap_duplicate_video_and_prediction_mismatch(tmp_path):
    split_file = _write_split(
        tmp_path / "split.json",
        {
            "train": ["203_1_Freeform_video", "203_1_Freeform_video"],
            "val": ["203_1_Northwind_video"],
            "test": ["205_1_Freeform_video"],
        },
    )
    label_dir = tmp_path / "labels"
    _write_label(label_dir, "203_1", 8)
    _write_label(label_dir, "205_1", 30)

    manifest = build_split_manifest(split_file, label_dir)
    overlaps = summarize_subject_overlap(manifest)

    assert len(overlaps) == 1
    assert overlaps[0]["subject_id"] == "203_1"
    assert any(row["duplicate_video_id"] == "True" for row in manifest)

    predictions_csv = tmp_path / "predictions.csv"
    write_prediction_table(
        predictions_csv,
        subject_ids=["999_1"],
        video_ids=["999_1_Freeform_video_aligned"],
        targets=[10],
        preds=[12],
    )

    alignment = align_predictions_to_split(predictions_csv, manifest)

    assert alignment[0]["status"] == "missing_in_split"

    generated = run_split_integrity_audit(
        split_file=split_file,
        label_dir=label_dir,
        predictions_csv=predictions_csv,
        output_dir=tmp_path / "split_integrity",
    )
    assert generated
    report = (tmp_path / "split_integrity" / "reports" / "split_integrity_report.md").read_text(
        encoding="utf-8"
    )
    assert "status: REVIEW_REQUIRED" in report
    assert "overlapping_subjects: 1" in report
    assert "prediction_alignment_mismatches: 1" in report
