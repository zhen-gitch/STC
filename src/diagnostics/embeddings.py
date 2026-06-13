from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for embedding diagnostics.") from exc
    return plt


def _load_features(features_npz):
    data = np.load(features_npz, allow_pickle=True)
    return (
        np.asarray(data["features"], dtype=float),
        np.asarray(data["subject_ids"]).astype(str),
        np.asarray(data["true_bdi"], dtype=float),
    )


def _encode_subjects(subject_ids):
    unique = {value: idx for idx, value in enumerate(sorted(set(subject_ids)))}
    return np.asarray([unique[value] for value in subject_ids], dtype=int)


def _pca_2d(features):
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def _plot_projection(coords, subject_ids, bdi_scores, title_prefix, save_path):
    plt = _require_matplotlib()
    subject_labels = _encode_subjects(subject_ids)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    scatter_subject = axes[0].scatter(coords[:, 0], coords[:, 1], c=subject_labels, cmap="tab20", s=36, alpha=0.78)
    axes[0].set_title(f"{title_prefix} by Subject")
    axes[0].grid(True, linestyle=":", alpha=0.35)
    if len(set(subject_ids)) <= 20:
        fig.colorbar(scatter_subject, ax=axes[0], label="Subject ID")

    scatter_bdi = axes[1].scatter(coords[:, 0], coords[:, 1], c=bdi_scores, cmap="viridis", s=36, alpha=0.78)
    axes[1].set_title(f"{title_prefix} by BDI")
    axes[1].grid(True, linestyle=":", alpha=0.35)
    fig.colorbar(scatter_bdi, ax=axes[1], label="True BDI")

    for ax in axes:
        ax.set_xlabel(f"{title_prefix} 1")
        ax.set_ylabel(f"{title_prefix} 2")

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_embedding_diagnostics(features_npz, save_dir):
    features, subject_ids, bdi_scores = _load_features(features_npz)
    save_dir = ensure_dir(save_dir)
    if features.ndim != 2 or features.shape[0] < 3:
        return []

    paths = []
    paths.append(_plot_projection(_pca_2d(features), subject_ids, bdi_scores, "PCA", save_dir / "embedding_pca.png"))

    try:
        from sklearn.manifold import TSNE

        if features.shape[0] >= 10:
            perplexity = min(30, max(5, features.shape[0] // 4), features.shape[0] - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
            coords = tsne.fit_transform(features)
            paths.append(_plot_projection(coords, subject_ids, bdi_scores, "t-SNE", save_dir / "embedding_tsne.png"))
    except Exception:
        pass

    try:
        import umap

        n_neighbors = min(15, max(2, features.shape[0] - 1))
        coords = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42).fit_transform(features)
        paths.append(_plot_projection(coords, subject_ids, bdi_scores, "UMAP", save_dir / "embedding_umap.png"))
    except Exception:
        pass

    return paths
