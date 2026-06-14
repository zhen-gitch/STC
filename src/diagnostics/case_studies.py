import math
from pathlib import Path

from src.diagnostics.io import ensure_dir, severity_group, write_csv_rows


CASE_STUDY_FIELDS = [
    "case_type",
    "rank",
    "video_id",
    "subject_id",
    "task_name",
    "true_bdi",
    "pred_bdi",
    "residual",
    "abs_error",
    "severity_group",
    "paired_video_id",
    "paired_task_name",
    "paired_task_pred_bdi",
    "task_pred_diff",
    "recommended_diagnostics",
]


def _safe_float(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _format_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _task_name(row):
    task = str(row.get("task_name") or "")
    if task:
        return task
    video_id = str(row.get("video_id") or "")
    for candidate in ("Freeform", "Northwind"):
        if candidate.lower() in video_id.lower():
            return candidate
    return ""


def _base_case_row(row, case_type, rank, recommended_diagnostics, paired=None):
    true_bdi = _safe_float(row.get("true_bdi"))
    pred_bdi = _safe_float(row.get("pred_bdi"))
    residual = _safe_float(row.get("residual"))
    abs_error = _safe_float(row.get("abs_error"))
    group = row.get("severity_group") or (severity_group(true_bdi) if true_bdi is not None else "")

    paired = paired or {}
    paired_pred = _safe_float(paired.get("pred_bdi")) if paired else None
    task_diff = None
    if pred_bdi is not None and paired_pred is not None:
        task_diff = pred_bdi - paired_pred

    return {
        "case_type": case_type,
        "rank": int(rank),
        "video_id": row.get("video_id", ""),
        "subject_id": row.get("subject_id", ""),
        "task_name": _task_name(row),
        "true_bdi": true_bdi,
        "pred_bdi": pred_bdi,
        "residual": residual,
        "abs_error": abs_error,
        "severity_group": group,
        "paired_video_id": paired.get("video_id", ""),
        "paired_task_name": _task_name(paired),
        "paired_task_pred_bdi": paired_pred,
        "task_pred_diff": task_diff,
        "recommended_diagnostics": recommended_diagnostics,
    }


def _valid_prediction_rows(records):
    valid = []
    for row in records:
        true_bdi = _safe_float(row.get("true_bdi"))
        pred_bdi = _safe_float(row.get("pred_bdi"))
        residual = _safe_float(row.get("residual"))
        abs_error = _safe_float(row.get("abs_error"))
        if true_bdi is None or pred_bdi is None:
            continue
        if residual is None:
            residual = pred_bdi - true_bdi
        if abs_error is None:
            abs_error = abs(residual)
        normalized = dict(row)
        normalized["true_bdi"] = true_bdi
        normalized["pred_bdi"] = pred_bdi
        normalized["residual"] = residual
        normalized["abs_error"] = abs_error
        normalized.setdefault("severity_group", severity_group(true_bdi))
        valid.append(normalized)
    return valid


def _task_inconsistency_rows(records, top_k):
    by_subject = {}
    for row in records:
        subject_id = str(row.get("subject_id") or "")
        if not subject_id:
            continue
        by_subject.setdefault(subject_id, []).append(row)

    pairs = []
    for subject_rows in by_subject.values():
        if len(subject_rows) < 2:
            continue
        rows = sorted(subject_rows, key=lambda item: float(item["pred_bdi"]))
        low = rows[0]
        high = rows[-1]
        diff = float(high["pred_bdi"]) - float(low["pred_bdi"])
        if diff <= 1e-8:
            continue
        pairs.append((diff, high, low))
        pairs.append((diff, low, high))

    pairs.sort(key=lambda item: item[0], reverse=True)
    output = []
    for rank, (_diff, row, paired) in enumerate(pairs[:top_k], start=1):
        output.append(
            _base_case_row(
                row,
                "task_inconsistency",
                rank,
                "aligned_face;model_attention;spatial_occlusion;keyframes;paired_task_comparison",
                paired=paired,
            )
        )
    return output


def build_case_study_manifest(records, top_k_per_type=10, low_error_k=8):
    records = _valid_prediction_rows(records)
    if not records:
        return []

    rows = []

    severe = [
        row
        for row in records
        if row.get("severity_group") == "severe" and float(row["residual"]) < 0.0
    ]
    severe.sort(key=lambda row: (float(row["residual"]), -float(row["abs_error"])))
    for rank, row in enumerate(severe[:top_k_per_type], start=1):
        rows.append(
            _base_case_row(
                row,
                "severe_underestimate",
                rank,
                "aligned_face;model_attention;spatial_occlusion;keyframes;shortcut_features",
            )
        )

    minimal = [
        row
        for row in records
        if row.get("severity_group") == "minimal" and float(row["residual"]) > 0.0
    ]
    minimal.sort(key=lambda row: (float(row["residual"]), float(row["abs_error"])), reverse=True)
    for rank, row in enumerate(minimal[:top_k_per_type], start=1):
        rows.append(
            _base_case_row(
                row,
                "minimal_overestimate",
                rank,
                "aligned_face;model_attention;spatial_occlusion;keyframes;shortcut_features",
            )
        )

    rows.extend(_task_inconsistency_rows(records, top_k=top_k_per_type))

    low_error = sorted(records, key=lambda row: float(row["abs_error"]))[:low_error_k]
    for rank, row in enumerate(low_error, start=1):
        rows.append(
            _base_case_row(
                row,
                "low_error_reference",
                rank,
                "aligned_face;model_attention;spatial_occlusion;keyframes",
            )
        )

    return rows


def write_case_study_manifest(records, table_path, report_path, top_k_per_type=10, low_error_k=8):
    manifest_rows = build_case_study_manifest(
        records,
        top_k_per_type=top_k_per_type,
        low_error_k=low_error_k,
    )

    table_path = Path(table_path)
    report_path = Path(report_path)
    ensure_dir(table_path.parent)
    ensure_dir(report_path.parent)

    formatted_rows = [
        {field: _format_value(row.get(field)) for field in CASE_STUDY_FIELDS}
        for row in manifest_rows
    ]
    write_csv_rows(table_path, formatted_rows, CASE_STUDY_FIELDS)
    _write_case_study_report(report_path, manifest_rows, table_path)
    return table_path, report_path


def _write_case_study_report(report_path, manifest_rows, table_path):
    counts = {}
    for row in manifest_rows:
        counts[row["case_type"]] = counts.get(row["case_type"], 0) + 1

    lines = [
        "# Case Study Manifest",
        "",
        "该清单用于固定后续逐案复查样本，重点覆盖 severe 低估、minimal 高估、任务间预测不一致和低误差对照。",
        "",
        f"- Manifest table: `{table_path}`",
        f"- Total cases: {len(manifest_rows)}",
        "",
        "## Case Counts",
        "",
    ]
    if counts:
        for case_type in sorted(counts):
            lines.append(f"- {case_type}: {counts[case_type]}")
    else:
        lines.append("- No valid cases were selected.")

    lines.extend(["", "## Top Cases", ""])
    if manifest_rows:
        lines.append("| Case Type | Rank | Video ID | True BDI | Pred BDI | Residual | Abs Error | Paired Task Diff |")
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
        for row in manifest_rows[:30]:
            task_diff = row.get("task_pred_diff")
            task_diff_text = f"{task_diff:.3f}" if isinstance(task_diff, float) else ""
            lines.append(
                f"| {row['case_type']} | {row['rank']} | {row['video_id']} | "
                f"{row['true_bdi']:.3f} | {row['pred_bdi']:.3f} | "
                f"{row['residual']:.3f} | {row['abs_error']:.3f} | {task_diff_text} |"
            )
    else:
        lines.append("No valid prediction rows were available.")

    lines.extend(
        [
            "",
            "## Usage Notes",
            "",
            "- `severe_underestimate` 和 `minimal_overestimate` 用于检查预测范围压缩。",
            "- `task_inconsistency` 用于检查同一 subject 在不同任务视频上的预测稳定性。",
            "- `low_error_reference` 用作归因图和遮挡图的对照组，避免只解释失败样本。",
            "- 推荐为每个 case 同步查看 aligned face、model attention、spatial occlusion 和 keyframe diagnostics。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
