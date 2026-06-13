from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir, read_csv_rows


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for diagnostic plots.") from exc
    return plt


def _epoch_mean_series(rows, column):
    values_by_epoch = {}
    for row in rows:
        raw_epoch = row.get("epoch", "")
        raw_value = row.get(column, "")
        if raw_epoch == "" or raw_value == "":
            continue
        try:
            epoch = int(float(raw_epoch))
            value = float(raw_value)
        except ValueError:
            continue
        values_by_epoch.setdefault(epoch, []).append(value)

    if not values_by_epoch:
        return [], []

    epochs = sorted(values_by_epoch)
    means = [float(np.mean(values_by_epoch[epoch])) for epoch in epochs]
    return epochs, means


def plot_training_curves(metrics_csv, save_dir):
    plt = _require_matplotlib()
    metrics_csv = Path(metrics_csv)
    rows = read_csv_rows(metrics_csv)
    save_dir = ensure_dir(save_dir)
    if not rows:
        return None

    available_columns = set(rows[0].keys())
    groups = {
        "losses": [
            "train_loss",
            "train_reg_loss",
            "train_ordinal_loss",
            "train_ccc_loss",
            "val_loss",
            "val_reg_loss",
            "val_ordinal_loss",
            "val_ccc_loss",
            "test_loss",
        ],
        "metrics": [
            "train_RMSE_epoch",
            "train_MAE_epoch",
            "train_CCC_epoch",
            "val_RMSE_epoch",
            "val_MAE_epoch",
            "val_CCC_epoch",
            "test_RMSE_epoch",
            "test_MAE_epoch",
            "test_CCC_epoch",
        ],
        "learning_rate": [col for col in available_columns if "lr" in col.lower()],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (title, columns) in zip(axes, groups.items()):
        plotted = False
        for column in columns:
            if column not in available_columns:
                continue
            epochs, values = _epoch_mean_series(rows, column)
            if not epochs:
                continue
            ax.plot(epochs, values, marker="o", linewidth=1.7, label=column)
            plotted = True
        ax.set_title(title.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle=":", alpha=0.4)
        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    save_path = save_dir / "training_curves.png"
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path
