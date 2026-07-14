import csv
import math

from src.diagnostics.au_region_tracking import (
    choose_frame_offset,
    extract_frame_id,
    run_frame_contract_audit,
    summarize_frame_contract,
)


def _make_frame_dir(root, name, frame_ids, prefix="frame"):
    video_dir = root / name
    video_dir.mkdir(parents=True)
    for frame_id in frame_ids:
        (video_dir / f"{prefix}_{frame_id:06d}.jpg").write_bytes(b"")
    return video_dir


def _write_openface_csv(path, frame_ids, timestamps=None):
    timestamps = timestamps or [index * 0.04 for index in range(len(frame_ids))]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["frame", "timestamp", "confidence", "success"],
        )
        writer.writeheader()
        for frame_id, timestamp in zip(frame_ids, timestamps):
            writer.writerow(
                {
                    "frame": frame_id,
                    "timestamp": timestamp,
                    "confidence": 0.95,
                    "success": 1,
                }
            )
    return path


def test_extract_frame_id_uses_last_digit_group_and_optional_regex():
    assert extract_frame_id("frame_det_00_000123.jpg") == 123
    assert extract_frame_id("subject203_frame0042.jpg", r"frame(\d+)") == 42
    assert extract_frame_id("no_frame_id.jpg") is None


def test_choose_frame_offset_detects_zero_to_one_based_mapping():
    offset, joined = choose_frame_offset([0, 1, 2, 3], [1, 2, 3, 4])

    assert offset == 1
    assert joined == 4


def test_summarize_frame_contract_reproduces_selected_indices(tmp_path):
    video_dir = _make_frame_dir(
        tmp_path,
        "203_1_Freeform_video_aligned",
        range(10),
    )
    csv_path = _write_openface_csv(
        tmp_path / "203_1_Freeform_video.csv",
        range(1, 11),
    )

    summary, selected, issues = summarize_frame_contract(
        video_dir,
        csv_path,
        sample_step=3,
        max_seq_len=9,
        sampling_strategy="stride_head",
    )

    assert summary["status"] == "PASS"
    assert summary["best_frame_offset"] == 1
    assert math.isclose(summary["frame_join_rate"], 1.0)
    assert summary["selected_frame_count"] == 3
    assert math.isclose(summary["selected_join_rate"], 1.0)
    assert [row["image_index"] for row in selected] == [0, 3, 6]
    assert [row["expected_csv_frame"] for row in selected] == [1, 4, 7]
    assert issues == []


def test_summarize_frame_contract_fails_duplicate_and_missing_ids(tmp_path):
    video_dir = tmp_path / "204_1_Freeform_video_aligned"
    video_dir.mkdir()
    (video_dir / "frame_000001.jpg").write_bytes(b"")
    (video_dir / "copy_000001.jpg").write_bytes(b"")
    (video_dir / "frame_000003.jpg").write_bytes(b"")
    csv_path = _write_openface_csv(
        tmp_path / "204_1_Freeform_video.csv",
        [1, 2, 3],
    )

    summary, _selected, issues = summarize_frame_contract(
        video_dir,
        csv_path,
        sample_step=1,
        max_seq_len=3,
    )

    issue_types = {row["issue_type"] for row in issues}
    assert summary["status"] == "FAIL"
    assert summary["image_duplicate_id_count"] == 1
    assert "duplicate_image_frame_ids" in issue_types
    assert "unused_csv_frame" in issue_types


def test_summarize_frame_contract_blocks_unparseable_filenames(tmp_path):
    video_dir = tmp_path / "205_1_Freeform_video_aligned"
    video_dir.mkdir()
    (video_dir / "first.jpg").write_bytes(b"")
    csv_path = _write_openface_csv(
        tmp_path / "205_1_Freeform_video.csv",
        [1],
    )

    summary, selected, issues = summarize_frame_contract(video_dir, csv_path)

    assert summary["status"] == "BLOCKED"
    assert selected[0]["match_status"] == "unmatched"
    assert "unparsed_image_frame_ids" in {row["issue_type"] for row in issues}


def test_run_frame_contract_audit_writes_tables_and_report(tmp_path):
    image_root = tmp_path / "images"
    openface_root = tmp_path / "openface"
    image_root.mkdir()
    openface_root.mkdir()
    _make_frame_dir(image_root, "203_1_Freeform_video_aligned", range(5))
    _write_openface_csv(openface_root / "203_1_Freeform_video.csv", range(1, 6))

    generated = run_frame_contract_audit(
        image_root=image_root,
        openface_root=openface_root,
        output_dir=tmp_path / "audit",
        sample_step=2,
        max_seq_len=6,
    )

    assert len(generated) == 4
    assert (tmp_path / "audit" / "tables" / "frame_contract_summary.csv").exists()
    assert (tmp_path / "audit" / "tables" / "selected_frame_mapping.csv").exists()
    assert (tmp_path / "audit" / "tables" / "frame_contract_issues.csv").exists()
    report = tmp_path / "audit" / "reports" / "frame_contract_report.md"
    assert report.exists()
    assert "PASS: 1" in report.read_text(encoding="utf-8")
