import csv

from src.diagnostics.case_studies import build_case_study_manifest, write_case_study_manifest


def test_build_case_study_manifest_selects_expected_case_types():
    records = [
        {
            "video_id": "246_1_Freeform_video_aligned",
            "subject_id": "246_1",
            "true_bdi": 43.0,
            "pred_bdi": 12.0,
            "residual": -31.0,
            "abs_error": 31.0,
            "severity_group": "severe",
        },
        {
            "video_id": "345_3_Northwind_video_aligned",
            "subject_id": "345_3",
            "true_bdi": 0.0,
            "pred_bdi": 19.5,
            "residual": 19.5,
            "abs_error": 19.5,
            "severity_group": "minimal",
        },
        {
            "video_id": "237_1_Freeform_video_aligned",
            "subject_id": "237_1",
            "true_bdi": 41.0,
            "pred_bdi": 16.0,
            "residual": -25.0,
            "abs_error": 25.0,
            "severity_group": "severe",
        },
        {
            "video_id": "237_1_Northwind_video_aligned",
            "subject_id": "237_1",
            "true_bdi": 41.0,
            "pred_bdi": 29.5,
            "residual": -11.5,
            "abs_error": 11.5,
            "severity_group": "severe",
        },
        {
            "video_id": "203_1_Freeform_video_aligned",
            "subject_id": "203_1",
            "true_bdi": 8.0,
            "pred_bdi": 8.2,
            "residual": 0.2,
            "abs_error": 0.2,
            "severity_group": "minimal",
        },
    ]

    manifest = build_case_study_manifest(records, top_k_per_type=4, low_error_k=2)
    case_types = {row["case_type"] for row in manifest}

    assert "severe_underestimate" in case_types
    assert "minimal_overestimate" in case_types
    assert "task_inconsistency" in case_types
    assert "low_error_reference" in case_types

    task_rows = [row for row in manifest if row["case_type"] == "task_inconsistency"]
    assert task_rows
    assert task_rows[0]["subject_id"] == "237_1"
    assert abs(task_rows[0]["task_pred_diff"]) == 13.5


def test_write_case_study_manifest_outputs_csv_and_markdown(tmp_path):
    records = [
        {
            "video_id": "001_Freeform_video",
            "subject_id": "001",
            "true_bdi": 30.0,
            "pred_bdi": 10.0,
            "residual": -20.0,
            "abs_error": 20.0,
            "severity_group": "severe",
        }
    ]

    table_path, report_path = write_case_study_manifest(
        records,
        table_path=tmp_path / "tables" / "case_study_manifest.csv",
        report_path=tmp_path / "reports" / "case_study_manifest.md",
    )

    assert table_path.exists()
    assert report_path.exists()
    with table_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["case_type"] == "severe_underestimate"
    assert "Case Study Manifest" in report_path.read_text(encoding="utf-8")
