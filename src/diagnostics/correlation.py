from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, numeric_columns_from_rows, read_csv_rows


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for correlation heatmaps.") from exc
    return plt


def _safe_correlation_matrix(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return None
    std = values.std(axis=0)
    keep = std > 1e-8
    if keep.sum() < 2:
        return None
    return np.corrcoef(values[:, keep], rowvar=False), keep


def plot_correlation_heatmap_from_rows(rows, save_path, excluded=(), title="Correlation Heatmap"):
    plt = _require_matplotlib()
    columns, names = numeric_columns_from_rows(rows, excluded=excluded)
    if len(names) < 2:
        return None

    matrix = np.column_stack([columns[name] for name in names])
    corr_result = _safe_correlation_matrix(matrix)
    if corr_result is None:
        return None
    corr, keep = corr_result
    names = [name for name, should_keep in zip(names, keep) if should_keep]

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig_size = max(6.0, len(names) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.82)

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_metrics_correlation_heatmap(metrics_csv, save_dir):
    rows = read_csv_rows(metrics_csv)
    return plot_correlation_heatmap_from_rows(
        rows,
        Path(save_dir) / "correlation_heatmap_metrics.png",
        excluded=("step",),
        title="Metrics Correlation Heatmap",
    )


def plot_predictions_correlation_heatmap(prediction_csv, save_dir):
    rows = read_csv_rows(prediction_csv)
    return plot_correlation_heatmap_from_rows(
        rows,
        Path(save_dir) / "correlation_heatmap_predictions.png",
        excluded=("subject_id", "severity_group"),
        title="Prediction Correlation Heatmap",
    )
