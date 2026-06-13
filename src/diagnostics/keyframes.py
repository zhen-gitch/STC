from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir
from src.diagnostics.occlusion import _scalar_prediction, frame_tensor_to_image


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for keyframe diagnostics.") from exc
    return plt


def temporal_occlusion_importance(model, video_tensor, mask, temporal_window=5, fill_value=0.0):
    """Estimate per-frame importance by masking temporal windows."""
    if video_tensor.size(0) != 1:
        raise ValueError("temporal_occlusion_importance expects batch size 1.")

    import torch

    model.eval()
    device = next(model.parameters()).device
    video_tensor = video_tensor.to(device)
    mask = mask.to(device)
    valid_count = int(mask[0].sum().detach().cpu().item())
    seq_len = video_tensor.size(1)
    importance = np.zeros(seq_len, dtype=np.float32)

    with torch.no_grad():
        base_score = _scalar_prediction(model, video_tensor, mask)

    half_window = max(0, int(temporal_window) // 2)
    for idx in range(valid_count):
        left = max(0, idx - half_window)
        right = min(valid_count, idx + half_window + 1)
        perturbed = video_tensor.clone()
        perturbed[:, left:right] = fill_value
        with torch.no_grad():
            score = _scalar_prediction(model, perturbed, mask)
        importance[idx] = abs(score - base_score)

    if np.max(importance) > 1e-8:
        importance = importance / np.max(importance)
    return importance, valid_count


def plot_keyframe_importance(importance, valid_count, save_path):
    plt = _require_matplotlib()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    x = np.arange(valid_count)

    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.plot(x, importance[:valid_count], linewidth=1.8)
    ax.fill_between(x, 0, importance[:valid_count], alpha=0.24)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Normalized Importance")
    ax.set_title("Temporal Occlusion Keyframe Importance")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_keyframe_strip(video_tensor, importance, valid_count, save_path, num_frames=8):
    plt = _require_matplotlib()
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    if valid_count <= 0:
        return None

    top_indices = np.argsort(importance[:valid_count])[::-1][:num_frames]
    top_indices = sorted(int(idx) for idx in top_indices)
    fig, axes = plt.subplots(1, len(top_indices), figsize=(2.0 * len(top_indices), 2.8))
    if len(top_indices) == 1:
        axes = [axes]

    for ax, frame_idx in zip(axes, top_indices):
        ax.imshow(frame_tensor_to_image(video_tensor[0, frame_idx]))
        ax.set_title(f"#{frame_idx}\n{importance[frame_idx]:.2f}", fontsize=9)
        ax.axis("off")

    fig.suptitle("Top Keyframes")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_keyframe_diagnostics(model, video_tensor, mask, save_dir, subject_id, temporal_window=5):
    save_dir = ensure_dir(save_dir)
    importance, valid_count = temporal_occlusion_importance(
        model=model,
        video_tensor=video_tensor,
        mask=mask,
        temporal_window=temporal_window,
    )
    timeline_path = plot_keyframe_importance(
        importance,
        valid_count,
        save_dir / f"keyframe_importance_subject_{subject_id}.png",
    )
    strip_path = plot_keyframe_strip(
        video_tensor,
        importance,
        valid_count,
        save_dir / f"keyframe_strip_subject_{subject_id}.png",
    )
    key_frame = int(np.argmax(importance[:valid_count])) if valid_count > 0 else 0
    return key_frame, [path for path in (timeline_path, strip_path) if path is not None]
