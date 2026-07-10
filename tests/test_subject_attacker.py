"""Tests for the offline subject-attacker accuracy aggregation.

Exercises the pure-numpy/csv logic in ``src.diagnostics.subject_attacker``:
seen/unseen partitioning, top-1/top-3 accuracy + coverage, single-run summary
schema, and multi-run aggregation.  No torch is required -- the per-query rows
are constructed directly, mimicking what the CLI's forward pass would produce.
"""

from pathlib import Path

from src.diagnostics.subject_attacker import (
    PER_QUERY_COLUMNS,
    RUN_SUMMARY_COLUMNS,
    SUMMARY_COLUMNS,
    compute_attacker_metrics,
    load_subject_attacker_summary,
    skipped_summary,
    summarize_subject_attacker_runs,
    write_subject_attacker_per_query,
    write_subject_attacker_summary,
)


def _row(video_id, subject_id, seen, true_idx, pred_idx, top3, c1, c3):
    return {
        "video_id": video_id,
        "subject_id": subject_id,
        "seen": seen,
        "true_subject_index": true_idx,
        "pred_subject_index": pred_idx,
        "pred_top3_subject_indices": top3,
        "correct_top1": c1,
        "correct_top3": c3,
    }


def test_metrics_seen_subset_accuracy_and_coverage():
    # 5 videos: 3 seen (2 correct top1, all 3 correct top3), 2 unseen.
    rows = [
        _row("v1", "101", True, 0, 0, [0, 1, 2], True, True),
        _row("v2", "101", True, 0, 1, [0, 1, 2], False, True),
        _row("v3", "102", True, 1, 1, [1, 0, 2], True, True),
        _row("v4", "999", False, None, None, None, False, False),
        _row("v5", "999", False, None, None, None, False, False),
    ]
    m = compute_attacker_metrics(rows, num_subject_classes=3)
    assert m["num_evaluated"] == 5
    assert m["num_seen"] == 3
    assert m["num_unseen_excluded"] == 2
    assert m["coverage"] == 0.6
    assert m["top1_accuracy"] == 2 / 3
    assert m["top3_accuracy"] == 1.0
    assert m["num_subject_classes"] == 3
    assert m["chance_top1"] == 1 / 3
    assert m["skipped"] is False


def test_metrics_all_unseen_returns_none_accuracy():
    rows = [
        _row("v1", "999", False, None, None, None, False, False),
        _row("v2", "998", False, None, None, None, False, False),
    ]
    m = compute_attacker_metrics(rows, num_subject_classes=5)
    assert m["num_seen"] == 0
    assert m["coverage"] == 0.0
    assert m["top1_accuracy"] is None
    assert m["top3_accuracy"] is None
    assert m["chance_top1"] == 0.2


def test_metrics_empty():
    m = compute_attacker_metrics([], num_subject_classes=0)
    assert m["num_evaluated"] == 0
    assert m["coverage"] is None
    assert m["top1_accuracy"] is None
    assert m["chance_top1"] is None


def test_skipped_summary():
    s = skipped_summary()
    assert s["skipped"] is True
    assert s["num_evaluated"] == 0
    assert s["top1_accuracy"] is None


def test_summary_csv_single_row_schema(tmp_path):
    rows = [_row("v1", "101", True, 0, 0, [0, 1], True, True)]
    m = compute_attacker_metrics(rows, num_subject_classes=2)
    out = tmp_path / "tables" / "subject_attacker_summary.csv"
    write_subject_attacker_summary(m, out)
    assert out.exists()

    from src.diagnostics.io import read_csv_rows

    loaded = read_csv_rows(out)
    assert len(loaded) == 1
    assert set(SUMMARY_COLUMNS).issubset(set(loaded[0].keys()))
    # round-trips through the loader (numeric coercion + skipped bool).
    parsed = load_subject_attacker_summary(out)
    assert parsed["num_seen"] == 1
    assert parsed["top1_accuracy"] == 1.0
    assert parsed["skipped"] is False


def test_per_query_csv_schema(tmp_path):
    rows = [
        _row("v1", "101", True, 0, 0, [0, 1, 2], True, True),
        _row("v2", "999", False, None, None, None, False, False),
    ]
    out = tmp_path / "tables" / "subject_attacker_per_query.csv"
    write_subject_attacker_per_query(rows, out)

    import csv

    with open(out) as f:
        header = next(csv.reader(f))
    assert set(PER_QUERY_COLUMNS).issubset(set(header))


def test_multirun_aggregation_includes_skipped_runs(tmp_path):
    # Two runs: one evaluated, one skipped (E0/E2).  Each in its own dir.
    run_a = tmp_path / "e1"
    run_b = tmp_path / "e0"

    rows = [_row("v1", "101", True, 0, 0, [0, 1], True, True)]
    m = compute_attacker_metrics(rows, num_subject_classes=2)
    write_subject_attacker_summary(m, run_a / "tables" / "subject_attacker_summary.csv")
    write_subject_attacker_summary(skipped_summary(), run_b / "tables" / "subject_attacker_summary.csv")

    out = tmp_path / "aggregate"
    generated = summarize_subject_attacker_runs(
        [("e1", run_a), ("e0", run_b)], out
    )
    assert any("subject_attacker_run_summary.csv" in str(p) for p in generated)

    from src.diagnostics.io import read_csv_rows

    run_rows = read_csv_rows(out / "tables" / "subject_attacker_run_summary.csv")
    by_name = {r["run_name"]: r for r in run_rows}
    assert set(by_name) == {"e1", "e0"}
    assert by_name["e1"]["top1_accuracy"] != ""
    assert by_name["e0"]["skipped"] in ("1", "True", "true")
    assert by_name["e0"]["top1_accuracy"] == ""


def test_multirun_skips_missing_run(tmp_path):
    run_a = tmp_path / "e1"
    rows = [_row("v1", "101", True, 0, 0, [0, 1], True, True)]
    write_subject_attacker_summary(
        compute_attacker_metrics(rows, num_subject_classes=2),
        run_a / "tables" / "subject_attacker_summary.csv",
    )
    out = tmp_path / "aggregate"
    generated = summarize_subject_attacker_runs(
        [("e1", run_a), ("e2", tmp_path / "missing_e2")], out
    )
    from src.diagnostics.io import read_csv_rows

    run_rows = read_csv_rows(out / "tables" / "subject_attacker_run_summary.csv")
    assert [r["run_name"] for r in run_rows] == ["e1"]
