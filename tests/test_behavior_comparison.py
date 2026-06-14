import csv

from src.diagnostics.behavior_comparison import compare_behavior_predictions
from src.diagnostics.io import write_prediction_table


def test_compare_behavior_predictions_aligns_video_ids_and_writes_summary(tmp_path):
    rgb_csv = tmp_path / "rgb_predictions.csv"
    behavior_csv = tmp_path / "behavior_predictions.csv"
    output_dir = tmp_path / "comparison"

    write_prediction_table(
        rgb_csv,
        subject_ids=["001_1", "002_1"],
        video_ids=["001_1_Freeform_video_aligned", "002_1_Northwind_video_aligned"],
        task_names=["Freeform", "Northwind"],
        targets=[10, 30],
        preds=[12, 18],
    )
    write_prediction_table(
        behavior_csv,
        subject_ids=["001_1", "002_1"],
        video_ids=["001_1_Freeform_video", "002_1_Northwind_video"],
        task_names=["Freeform", "Northwind"],
        targets=[10, 30],
        preds=[11, 25],
    )

    comparison_path, summary_path = compare_behavior_predictions(rgb_csv, behavior_csv, output_dir)

    with comparison_path.open(newline="", encoding="utf-8") as f:
        comparison_rows = list(csv.DictReader(f))
    with summary_path.open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))

    assert len(comparison_rows) == 2
    assert comparison_rows[0]["video_id"] == "001_1_Freeform_video"
    assert comparison_rows[0]["better_model"] == "behavior"
    assert summary_rows[0]["group"] == "all"
    assert summary_rows[0]["count"] == "2"
