from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for occlusion diagnostics.") from exc
    return plt


def frame_tensor_to_image(frame_tensor):
    frame = frame_tensor.detach().cpu().float()
    image = frame.permute(1, 2, 0).numpy()
    image = image * 0.5 + 0.5
    return np.clip(image, 0.0, 1.0)


def _scalar_prediction(model, video_tensor, mask):
    outputs = model(video_tensor, mask)
    pred = model.prediction_for_metrics(outputs.bdi_pred)
    return float(pred.reshape(-1)[0].detach().cpu().item())


def spatial_occlusion_map(model, video_tensor, mask, frame_idx, patch_size=24, stride=8, fill_value=0.0):
    """Return a spatial occlusion sensitivity map for one video/frame."""
    if video_tensor.size(0) != 1:
        raise ValueError("spatial_occlusion_map expects batch size 1.")

    model.eval()
    device = next(model.parameters()).device
    video_tensor = video_tensor.to(device)
    mask = mask.to(device)

    frame_idx = min(int(frame_idx), video_tensor.size(1) - 1)
    _, _, _, height, width = video_tensor.shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    import torch

    with torch.no_grad():
        base_score = _scalar_prediction(model, video_tensor, mask)

    for y in range(0, max(1, height - patch_size + 1), stride):
        for x in range(0, max(1, width - patch_size + 1), stride):
            perturbed = video_tensor.clone()
            perturbed[:, frame_idx, :, y:y + patch_size, x:x + patch_size] = fill_value
            with torch.no_grad():
                score = _scalar_prediction(model, perturbed, mask)
            delta = abs(score - base_score)
            heatmap[y:y + patch_size, x:x + patch_size] += delta
            counts[y:y + patch_size, x:x + patch_size] += 1.0

    heatmap = heatmap / np.maximum(counts, 1e-8)
    max_value = float(np.max(heatmap))
    min_value = float(np.min(heatmap))
    if max_value - min_value > 1e-8:
        heatmap = (heatmap - min_value) / (max_value - min_value)
    return heatmap


def plot_heatmap_overlay(frame_tensor, heatmap, save_path, title):
    plt = _require_matplotlib()
    image = frame_tensor_to_image(frame_tensor)
    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    axes[0].imshow(image)
    axes[0].set_title("Frame")
    axes[0].axis("off")

    axes[1].imshow(image)
    im = axes[1].imshow(heatmap, cmap="magma", alpha=0.48, vmin=0.0, vmax=1.0)
    axes[1].set_title(title)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_spatial_occlusion_diagnostic(
    model,
    video_tensor,
    mask,
    frame_idx,
    save_path,
    patch_size=24,
    stride=8,
):
    heatmap = spatial_occlusion_map(
        model=model,
        video_tensor=video_tensor,
        mask=mask,
        frame_idx=frame_idx,
        patch_size=patch_size,
        stride=stride,
    )
    frame_idx = min(int(frame_idx), video_tensor.size(1) - 1)
    return plot_heatmap_overlay(
        video_tensor[0, frame_idx],
        heatmap,
        save_path,
        title="Occlusion Impact",
    )
