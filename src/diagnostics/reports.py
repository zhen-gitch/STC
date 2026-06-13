from pathlib import Path

from src.diagnostics.io import ensure_dir
from src.diagnostics.regression import regression_summary


def write_diagnostic_report(report_path, run_dir, checkpoint_path, records, generated_files):
    report_path = Path(report_path)
    ensure_dir(report_path.parent)
    summary = regression_summary(records)

    high_error = sorted(records, key=lambda row: row["abs_error"], reverse=True)[:10]
    low_error = sorted(records, key=lambda row: row["abs_error"])[:10]

    lines = [
        "# MTL-Lite Diagnostic Report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Samples: {summary['count']}",
        f"- MAE: {summary['mae']:.4f}",
        f"- RMSE: {summary['rmse']:.4f}",
        f"- Pearson: {summary['pearson']:.4f}",
        "",
        "## Generated Files",
        "",
    ]
    for file_path in generated_files:
        lines.append(f"- `{file_path}`")

    lines.extend(["", "## High Error Subjects", ""])
    for row in high_error:
        lines.append(
            f"- {row['subject_id']}: true={row['true_bdi']:.2f}, "
            f"pred={row['pred_bdi']:.2f}, abs_error={row['abs_error']:.2f}"
        )

    lines.extend(["", "## Low Error Subjects", ""])
    for row in low_error:
        lines.append(
            f"- {row['subject_id']}: true={row['true_bdi']:.2f}, "
            f"pred={row['pred_bdi']:.2f}, abs_error={row['abs_error']:.2f}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
