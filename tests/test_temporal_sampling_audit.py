import csv

import pytest

from src.diagnostics.io import write_prediction_table
from src.diagnostics.temporal_sampling import (
    merge_predictions_with_temporal_sampling,
    run_temporal_sampling_audit,
    summarize_video_temporal_sampling,
)


def _make_frame_dir(root, name, count):
    video_dir = root / name
    video_dir.mkdir(parents=True)
    for idx in range(count):
        (video_dir / f"frame_{idx:06d}.jpg").write_bytes(b"")
    return video_dir


def test_summarize_video_temporal_sampling_matches_dataset_selection(tmp_path):
    video_dir = _make_frame_dir(tmp_path, "203_1_Freeform_video_aligned", 25)

    summary = summarize_video_temporal_sampling(video_dir, sample_step=3, max_seq_len=12)

    assert summary["frame_count"] == 25
    assert summary["sampled_frame_count"] == 9
    assert summary["model_max_len"] == 4
    assert summary["selected_frame_count"] == 4
    assert summary["truncated_frame_count"] == 5
    assert summary["padding_frame_count"] == 0
    assert summary["valid_ratio"] == pytest.approx(1.0)
    assert summary["truncated_ratio"] == pytest.approx(5 / 9)


def test_summarize_video_temporal_sampling_reports_padding(tmp_path):
    video_dir = _make_frame_dir(tmp_path, "204_1_Freeform_video_aligned", 5)

    summary = summarize_video_temporal_sampling(video_dir, sample_step=2, max_seq_len=12)

    assert summary["sampled_frame_count"] == 3
    assert summary["model_max_len"] == 6
    assert summary["selected_frame_count"] == 3
    assert summary["padding_frame_count"] == 3
    assert summary["valid_ratio"] == pytest.approx(0.5)
    assert summary["padding_ratio"] == pytest.approx(0.5)


def test_summarize_video_temporal_sampling_supports_uniform_strategy(tmp_path):
    video_dir = _make_frame_dir(tmp_path, "205_1_Freeform_video_aligned", 25)

    summary = summarize_video_temporal_sampling(video_dir, sample_step=1, max_seq_len=8, sampling_strategy="uniform")

    assert summary["sampling_strategy"] == "uniform"
    assert summary["frame_count"] == 25
    assert summary["sampled_frame_count"] == 25
    assert summary["selected_frame_count"] == 8
    assert summary["truncated_frame_count"] == 17
    assert summary["padding_frame_count"] == 0


def test_merge_predictions_with_temporal_sampling_matches_aligned_ids():
    prediction_rows = [
        {
            "video_id": "203_1_Freeform_video_aligned",
            "subject_id": "203_1",
            "task_name": "",
            "true_bdi": 8.0,
            "pred_bdi": 10.0,
            "residual": 2.0,
            "abs_error": 2.0,
            "severity_group": "minimal",
        }
    ]
    sampling_rows = [
        {
            "video_id": "203_1_Freeform_video",
            "frame_count": 30,
            "sampled_frame_count": 3,
            "selected_frame_count": 3,
        }
    ]

    merged = merge_predictions_with_temporal_sampling(prediction_rows, sampling_rows)

    assert len(merged) == 1
    assert merged[0]["frame_count"] == 30
    assert merged[0]["task_name"] == "Freeform"


def test_run_temporal_sampling_audit_writes_expected_outputs(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    _make_frame_dir(image_root, "203_1_Freeform_video_aligned", 30)
    _make_frame_dir(image_root, "204_1_Freeform_video_aligned", 5)

    predictions_csv = tmp_path / "predictions.csv"
    write_prediction_table(
        predictions_csv,
        subject_ids=["203_1", "204_1"],
        video_ids=["203_1_Freeform_video_aligned", "204_1_Freeform_video_aligned"],
        targets=[8, 30],
        preds=[10, 20],
    )

    generated = run_temporal_sampling_audit(
        predictions_csv=predictions_csv,
        image_root=image_root,
        output_dir=tmp_path / "temporal_sampling",
        sample_step=3,
        max_seq_len=12,
    )

    assert generated
    tables = tmp_path / "temporal_sampling" / "tables"
    reports = tmp_path / "temporal_sampling" / "reports"
    assert (tables / "temporal_sampling_summary.csv").exists()
    assert (tables / "temporal_sampling_merged.csv").exists()
    assert (tables / "temporal_sampling_correlation.csv").exists()
    assert (tables / "temporal_sampling_group_summary.csv").exists()
    assert (reports / "temporal_sampling_audit_report.md").exists()

    with (tables / "temporal_sampling_merged.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
