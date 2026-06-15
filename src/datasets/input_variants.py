import torch
import torch.nn.functional as F


SUPPORTED_INPUT_VARIANTS = {
    "rgb",
    "grayscale",
    "blur",
    "center_mask",
    "boundary_erased",
    "black_to_gray",
    "black_to_mean",
    "black_to_blur",
    "soft_center_mask",
    "inner_crop_resize",
}
RESERVED_INPUT_VARIANTS = {"landmark_heatmap"}


def normalize_input_variant(variant):
    variant = str(variant or "rgb").strip().lower()
    aliases = {
        "aligned_rgb": "rgb",
        "gray": "grayscale",
        "grey": "grayscale",
        "masked_face": "center_mask",
        "boundary_mask": "boundary_erased",
        "black_fill_gray": "black_to_gray",
        "black_fill_mean": "black_to_mean",
        "black_fill_blur": "black_to_blur",
        "soft_mask": "soft_center_mask",
        "inner_crop": "inner_crop_resize",
    }
    variant = aliases.get(variant, variant)
    if variant in RESERVED_INPUT_VARIANTS:
        raise ValueError(
            "DATASET.INPUT_VARIANT='landmark_heatmap' requires OpenFace landmark "
            "coordinates and is reserved for the behavior/landmark baseline path."
        )
    if variant not in SUPPORTED_INPUT_VARIANTS:
        supported = sorted(SUPPORTED_INPUT_VARIANTS | RESERVED_INPUT_VARIANTS)
        raise ValueError(f"Unsupported DATASET.INPUT_VARIANT='{variant}'. Supported values: {supported}")
    return variant


def apply_input_variant(video_tensor, variant):
    """Apply a deterministic ablation variant to raw RGB video frames.

    Args:
        video_tensor: Tensor shaped [T, 3, H, W], usually uint8 from torchvision.io.read_image.
        variant: One of the supported DATASET.INPUT_VARIANT values.
    """
    variant = normalize_input_variant(variant)
    if variant == "rgb":
        return video_tensor
    if variant == "grayscale":
        return _to_grayscale_rgb(video_tensor)
    if variant == "blur":
        return _blur_video(video_tensor)
    if variant == "center_mask":
        return _apply_ellipse_mask(video_tensor, radius_y=0.46, radius_x=0.38)
    if variant == "boundary_erased":
        return _apply_ellipse_mask(video_tensor, radius_y=0.72, radius_x=0.60)
    if variant == "black_to_gray":
        return _replace_black_pixels(video_tensor, mode="gray")
    if variant == "black_to_mean":
        return _replace_black_pixels(video_tensor, mode="mean")
    if variant == "black_to_blur":
        return _replace_black_pixels(video_tensor, mode="blur")
    if variant == "soft_center_mask":
        return _apply_soft_ellipse_mask(video_tensor, radius_y=0.52, radius_x=0.43)
    if variant == "inner_crop_resize":
        return _inner_crop_resize(video_tensor)
    raise AssertionError(f"Unhandled input variant: {variant}")


def _restore_dtype(values, dtype):
    if dtype.is_floating_point:
        return values.to(dtype=dtype)
    info = torch.iinfo(dtype)
    return values.round().clamp(info.min, info.max).to(dtype=dtype)


def _to_grayscale_rgb(video_tensor):
    dtype = video_tensor.dtype
    values = video_tensor.to(dtype=torch.float32)
    weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=values.dtype, device=values.device).view(1, 3, 1, 1)
    gray = (values * weights).sum(dim=1, keepdim=True)
    return _restore_dtype(gray.repeat(1, 3, 1, 1), dtype)


def _blur_video(video_tensor, kernel_size=7):
    dtype = video_tensor.dtype
    if kernel_size % 2 == 0:
        kernel_size += 1
    values = video_tensor.to(dtype=torch.float32)
    blurred = F.avg_pool2d(
        values,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    )
    return _restore_dtype(blurred, dtype)


def _ellipse_mask(height, width, radius_y, radius_x, device):
    y = torch.linspace(-1.0, 1.0, steps=height, device=device).view(height, 1)
    x = torch.linspace(-1.0, 1.0, steps=width, device=device).view(1, width)
    center_y = -0.08
    center_x = 0.0
    mask = (((y - center_y) / radius_y) ** 2 + ((x - center_x) / radius_x) ** 2) <= 1.0
    return mask.to(dtype=torch.float32).view(1, 1, height, width)


def _apply_ellipse_mask(video_tensor, radius_y, radius_x):
    dtype = video_tensor.dtype
    _, _, height, width = video_tensor.shape
    values = video_tensor.to(dtype=torch.float32)
    mask = _ellipse_mask(height, width, radius_y=radius_y, radius_x=radius_x, device=values.device)
    return _restore_dtype(values * mask, dtype)


def _black_pixel_mask(video_tensor, threshold=8):
    values = video_tensor.to(dtype=torch.float32)
    return (values <= float(threshold)).all(dim=1, keepdim=True)


def _replace_black_pixels(video_tensor, mode="gray", threshold=8):
    dtype = video_tensor.dtype
    values = video_tensor.to(dtype=torch.float32)
    black_mask = _black_pixel_mask(values, threshold=threshold)
    if not black_mask.any():
        return video_tensor

    if mode == "gray":
        fill = torch.full_like(values, 127.0)
    elif mode == "blur":
        neutral_base = torch.where(black_mask.expand_as(values), torch.full_like(values, 127.0), values)
        fill = _blur_video(neutral_base, kernel_size=15).to(dtype=torch.float32)
    elif mode == "mean":
        fill = torch.empty_like(values)
        valid = (~black_mask).expand_as(values)
        for frame_idx in range(values.size(0)):
            frame_valid = valid[frame_idx]
            if frame_valid.any():
                channel_means = []
                for channel_idx in range(values.size(1)):
                    channel_values = values[frame_idx, channel_idx][frame_valid[channel_idx]]
                    channel_means.append(channel_values.mean())
                frame_fill = torch.stack(channel_means).view(-1, 1, 1)
            else:
                frame_fill = torch.full((values.size(1), 1, 1), 127.0, device=values.device, dtype=values.dtype)
            fill[frame_idx] = frame_fill
    else:
        raise ValueError(f"Unknown black pixel replacement mode: {mode}")

    return _restore_dtype(torch.where(black_mask.expand_as(values), fill, values), dtype)


def _soft_ellipse_alpha(height, width, radius_y, radius_x, device, softness=0.10):
    y = torch.linspace(-1.0, 1.0, steps=height, device=device).view(height, 1)
    x = torch.linspace(-1.0, 1.0, steps=width, device=device).view(1, width)
    center_y = -0.08
    center_x = 0.0
    distance = torch.sqrt(((y - center_y) / radius_y) ** 2 + ((x - center_x) / radius_x) ** 2)
    alpha = torch.clamp((1.0 + softness - distance) / max(softness, 1e-6), 0.0, 1.0)
    return alpha.view(1, 1, height, width)


def _apply_soft_ellipse_mask(video_tensor, radius_y, radius_x, fill_value=127.0):
    dtype = video_tensor.dtype
    _, _, height, width = video_tensor.shape
    values = video_tensor.to(dtype=torch.float32)
    alpha = _soft_ellipse_alpha(height, width, radius_y=radius_y, radius_x=radius_x, device=values.device)
    fill = torch.full_like(values, float(fill_value))
    return _restore_dtype(values * alpha + fill * (1.0 - alpha), dtype)


def _inner_crop_resize(video_tensor, top=0.12, bottom=0.90, left=0.16, right=0.84):
    dtype = video_tensor.dtype
    _, _, height, width = video_tensor.shape
    y0 = max(0, min(height - 1, int(round(height * top))))
    y1 = max(y0 + 1, min(height, int(round(height * bottom))))
    x0 = max(0, min(width - 1, int(round(width * left))))
    x1 = max(x0 + 1, min(width, int(round(width * right))))
    cropped = video_tensor[:, :, y0:y1, x0:x1].to(dtype=torch.float32)
    resized = F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)
    return _restore_dtype(resized, dtype)
