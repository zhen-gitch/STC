from pathlib import Path

import numpy as np

from src.diagnostics.io import ensure_dir
from src.diagnostics.occlusion import plot_heatmap_overlay


def find_last_conv2d(module):
    import torch.nn as nn

    last_conv = None
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            last_conv = child
    return last_conv


def _normalize_heatmap(heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.maximum(heatmap, 0.0)
    min_value = float(np.min(heatmap))
    max_value = float(np.max(heatmap))
    if max_value - min_value > 1e-8:
        heatmap = (heatmap - min_value) / (max_value - min_value)
    return heatmap


def input_gradient_attention_map(model, video_tensor, mask, frame_idx):
    """General model-dependent attention proxy based on input gradients."""
    import torch

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    video_tensor = video_tensor.to(device).detach().clone().requires_grad_(True)
    mask = mask.to(device)
    frame_idx = min(int(frame_idx), video_tensor.size(1) - 1)

    try:
        model.zero_grad(set_to_none=True)
        with torch.backends.cudnn.flags(enabled=False):
            outputs = model(video_tensor, mask)
            target = outputs.bdi_pred.reshape(-1)[0]
            target.backward()

        grad = video_tensor.grad[0, frame_idx].detach().abs().mean(dim=0).cpu().numpy()
        return _normalize_heatmap(grad)
    finally:
        model.train(was_training)


def gradcam_attention_map(model, video_tensor, mask, frame_idx, target_layer):
    """Grad-CAM over the last convolutional backbone layer when available."""
    import torch
    import torch.nn.functional as F

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    video_tensor = video_tensor.to(device)
    mask = mask.to(device)
    frame_idx = min(int(frame_idx), video_tensor.size(1) - 1)

    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.backends.cudnn.flags(enabled=False):
            outputs = model(video_tensor, mask)
            outputs.bdi_pred.reshape(-1)[0].backward()

        if "value" not in activations or "value" not in gradients:
            return None

        valid_before = int(mask[0, :frame_idx].sum().detach().cpu().item())
        if not bool(mask[0, frame_idx].detach().cpu().item()):
            return None
        if valid_before >= activations["value"].shape[0]:
            return None

        activation = activations["value"][valid_before]
        gradient = gradients["value"][valid_before]
        if activation.ndim != 3 or gradient.ndim != 3:
            return None

        weights = gradient.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * activation).sum(dim=0))
        cam = F.interpolate(
            cam[None, None],
            size=video_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return _normalize_heatmap(cam.detach().cpu().numpy())
    finally:
        handle_fwd.remove()
        handle_bwd.remove()
        model.train(was_training)


def model_attention_map(model, video_tensor, mask, frame_idx, method="auto"):
    target_layer = find_last_conv2d(model.backbone)
    if method in {"auto", "gradcam"} and target_layer is not None:
        cam = gradcam_attention_map(model, video_tensor, mask, frame_idx, target_layer)
        if cam is not None:
            return cam, "gradcam"

    cam = input_gradient_attention_map(model, video_tensor, mask, frame_idx)
    return cam, "input_gradient"


def save_model_attention_diagnostic(model, video_tensor, mask, frame_idx, save_path, method="auto"):
    heatmap, used_method = model_attention_map(model, video_tensor, mask, frame_idx, method=method)
    frame_idx = min(int(frame_idx), video_tensor.size(1) - 1)
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    return plot_heatmap_overlay(
        video_tensor[0, frame_idx],
        heatmap,
        save_path,
        title=f"Model Attention ({used_method})",
    )
