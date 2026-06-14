from pathlib import Path

import numpy as np

from src.diagnostics.case_studies import write_case_study_manifest
from src.diagnostics.io import ensure_dir, read_prediction_table, write_csv_rows


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for regression diagnostics.") from exc
    return plt


def prediction_arrays(records):
    targets = np.asarray([row["true_bdi"] for row in records], dtype=float)
    preds = np.asarray([row["pred_bdi"] for row in records], dtype=float)
    residuals = np.asarray([row["residual"] for row in records], dtype=float)
    abs_errors = np.asarray([row["abs_error"] for row in records], dtype=float)
    return targets, preds, residuals, abs_errors


def regression_summary(records):
    if not records:
        return {"count": 0, "mae": np.nan, "rmse": np.nan, "pearson": np.nan}
    targets, preds, residuals, abs_errors = prediction_arrays(records)
    pearson = np.nan
    if len(records) >= 2 and np.std(targets) > 1e-8 and np.std(preds) > 1e-8:
        pearson = float(np.corrcoef(targets, preds)[0, 1])
    return {
        "count": int(len(records)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "pearson": pearson,
    }


def plot_prediction_target_scatter(records, save_dir, max_score=63):
    plt = _require_matplotlib()
    save_dir = ensure_dir(save_dir)
    targets, preds, _, _ = prediction_arrays(records)
    summary = regression_summary(records)

    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.scatter(targets, preds, s=42, alpha=0.75, edgecolors="none")
    ax.plot([0, max_score], [0, max_score], linestyle="--", color="black", linewidth=1.1)
    ax.set_xlim(0, max_score)
    ax.set_ylim(0, max_score)
    ax.set_xlabel("True BDI")
    ax.set_ylabel("Predicted BDI")
    ax.set_title("Prediction vs Target")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.text(
        0.04,
        0.96,
        f"MAE={summary['mae']:.2f}\nRMSE={summary['rmse']:.2f}\nPearson={summary['pearson']:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.84, edgecolor="none"),
    )
    save_path = save_dir / "prediction_target_scatter.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_residual_histogram(records, save_dir):
    plt = _require_matplotlib()
    save_dir = ensure_dir(save_dir)
    _, _, residuals, _ = prediction_arrays(records)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hist(residuals, bins=min(24, max(6, len(residuals) // 2)), alpha=0.82)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Residual (Pred - True)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    save_path = save_dir / "residual_histogram.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_residual_by_bdi_bin(records, save_dir, max_score=63, bin_size=5):
    plt = _require_matplotlib()
    save_dir = ensure_dir(save_dir)
    targets, _, _, abs_errors = prediction_arrays(records)

    edges = np.arange(0, max_score + bin_size, bin_size)
    if edges[-1] < max_score:
        edges = np.append(edges, max_score)

    centers = []
    maes = []
    counts = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == edges[-1]:
            mask = (targets >= left) & (targets <= right)
        else:
            mask = (targets >= left) & (targets < right)
        if np.any(mask):
            centers.append((left + right) / 2)
            maes.append(float(np.mean(abs_errors[mask])))
            counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(centers, maes, width=max(1.0, bin_size * 0.82), alpha=0.82)
    for x, y, count in zip(centers, maes, counts):
        ax.text(x, y, str(count), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("BDI Bin Center")
    ax.set_ylabel("MAE")
    ax.set_title("MAE by BDI Bin")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    save_path = save_dir / "residual_by_bdi_bin.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_severity_group_error(records, save_dir):
    plt = _require_matplotlib()
    save_dir = ensure_dir(save_dir)
    groups = ["minimal", "mild", "moderate", "severe"]
    maes = []
    counts = []
    for group in groups:
        values = [row["abs_error"] for row in records if row["severity_group"] == group]
        maes.append(float(np.mean(values)) if values else 0.0)
        counts.append(len(values))

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.bar(groups, maes, alpha=0.82)
    for idx, (mae, count) in enumerate(zip(maes, counts)):
        ax.text(idx, mae, str(count), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Severity Group")
    ax.set_ylabel("MAE")
    ax.set_title("MAE by Severity Group")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    save_path = save_dir / "severity_group_error.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def write_error_rankings(records, save_dir, top_k=20):
    save_dir = ensure_dir(save_dir)
    high_rows = sorted(records, key=lambda row: row["abs_error"], reverse=True)[:top_k]
    low_rows = sorted(records, key=lambda row: row["abs_error"])[:top_k]
    fields = ["video_id", "subject_id", "true_bdi", "pred_bdi", "residual", "abs_error", "severity_group"]
    high_path = save_dir / "high_error_subjects.csv"
    low_path = save_dir / "low_error_subjects.csv"
    write_csv_rows(high_path, high_rows, fields)
    write_csv_rows(low_path, low_rows, fields)
    return high_path, low_path


def plot_regression_diagnostics(prediction_csv, save_dir, max_score=63, bin_size=5):
    records = read_prediction_table(prediction_csv)
    if not records:
        return []
    paths = [
        plot_prediction_target_scatter(records, save_dir, max_score=max_score),
        plot_residual_histogram(records, save_dir),
        plot_residual_by_bdi_bin(records, save_dir, max_score=max_score, bin_size=bin_size),
        plot_severity_group_error(records, save_dir),
    ]
    paths.extend(write_error_rankings(records, save_dir))
    paths.extend(
        write_case_study_manifest(
            records,
            table_path=Path(save_dir) / "case_study_manifest.csv",
            report_path=Path(save_dir) / "case_study_manifest.md",
        )
    )
    return paths
