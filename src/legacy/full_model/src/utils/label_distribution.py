import numpy as np
import torch


def _gaussian_smooth_1d(values, sigma):
    """Smooth a 1D array with a normalized Gaussian kernel."""
    if sigma <= 0:
        return values.astype(float)

    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    padded = np.pad(values.astype(float), (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def compute_bdi_loss_weights(bdi_scores, max_score, sigma=2.0, severity_alpha=0.8):
    """Compute Label Distribution Smoothing weights for discrete BDI scores.

    Args:
        bdi_scores: Iterable of integer BDI labels. Callers should pass labels
            directly rather than Dataset samples so image/video loading is not
            triggered before training.
        max_score: Maximum valid BDI score.
        sigma: Gaussian smoothing width for label density.
        severity_alpha: Additional severity-aware multiplier.

    Returns:
        A float tensor of shape ``[max_score + 1]`` normalized to mean 1.0.
    """
    all_train_bdis = np.clip(np.array(list(bdi_scores), dtype=int), 0, max_score)

    counts, _ = np.histogram(all_train_bdis, bins=max_score + 1, range=(0, max_score + 1))
    smoothed_counts = _gaussian_smooth_1d(counts.astype(float), sigma=sigma)
    smoothed_counts[smoothed_counts == 0] = 1e-6

    base_weights = 1.0 / smoothed_counts
    severity_scores = np.arange(max_score + 1) / float(max_score)
    final_weights = base_weights * (1.0 + severity_alpha * severity_scores)
    final_weights = final_weights / np.mean(final_weights)

    return torch.from_numpy(final_weights).float()
