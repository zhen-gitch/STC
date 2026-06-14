import torch
import torch.nn.functional as F


SUPPORTED_INPUT_VARIANTS = {
    "rgb",
    "grayscale",
    "blur",
    "center_mask",
    "boundary_erased",
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
        variant: One of rgb, grayscale, blur, center_mask, boundary_erased.
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
