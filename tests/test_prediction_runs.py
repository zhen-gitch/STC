import csv

from src.diagnostics.io import write_prediction_table
from src.diagnostics.prediction_runs import summarize_prediction_runs


def test_summarize_prediction_runs_outputs_core_tables(tmp_path):
    rgb_csv = tmp_path / "rgb.csv"
    improved_csv = tmp_path / "improved.csv"
    video_ids = [
        "203_1_Freeform_video_aligned",
        "203_1_Northwind_video_aligned",
        "204_1_Freeform_video_aligned",
        "204_1_Northwind_video_aligned",
    ]
    subject_ids = ["203_1", "203_1", "204_1", "204_1"]
    targets = [4, 4, 30, 30]

    write_prediction_table(rgb_csv, subject_ids=subject_ids, video_ids=video_ids, targets=targets, preds=[10, 12, 18, 20])
    write_prediction_table(improved_csv, subject_ids=subject_ids, video_ids=video_ids, targets=targets, preds=[6, 5, 24, 25])

    generated = summarize_prediction_runs(
        [("rgb", rgb_csv), ("improved", improved_csv)],
        output_dir=tmp_path / "summary",
        baseline_name="rgb",
    )

    assert len(generated) == 5
    for path in generated:
        assert path.exists()

    overall_path = tmp_path / "summary" / "tables" / "prediction_run_summary.csv"
    with overall_path.open(newline="", encoding="utf-8") as f:
        overall = list(csv.DictReader(f))
    assert overall[0]["run"] == "improved"
    assert float(overall[0]["mae"]) < float(overall[1]["mae"])
    assert "pred_std" in overall[0]

    severity_path = tmp_path / "summary" / "tables" / "severity_bias_summary.csv"
    with severity_path.open(newline="", encoding="utf-8") as f:
        severity_rows = list(csv.DictReader(f))
    assert {row["severity_group"] for row in severity_rows} == {"minimal", "severe"}

    task_path = tmp_path / "summary" / "tables" / "task_consistency_summary.csv"
    with task_path.open(newline="", encoding="utf-8") as f:
        task_rows = list(csv.DictReader(f))
    assert all(row["pair_count"] == "2" for row in task_rows)

    pairwise_path = tmp_path / "summary" / "tables" / "pairwise_baseline_improvement.csv"
    with pairwise_path.open(newline="", encoding="utf-8") as f:
        pairwise = list(csv.DictReader(f))
    assert pairwise[0]["run"] == "improved"
    assert float(pairwise[0]["mean_abs_error_improvement"]) > 0
