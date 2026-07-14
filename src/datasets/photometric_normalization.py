"""Deterministic video-level photometric normalization for RGB inputs.

The transform estimates one robust luminance mapping per sampled video tensor
and applies that same mapping to every frame.  This avoids frame-to-frame
flicker while weakening subject/video-specific exposure differences.  Chroma
is left unchanged (before clipping), and axis-connected OpenFace black padding
is excluded from statistics and preserved in the output.
"""

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch


_LUMA_WEIGHTS = (0.2989, 0.5870, 0.1140)
_SUPPORTED_MODES = {"none", "center", "center_contrast"}
_APPLY_CHUNK_SIZE = 32


@dataclass(frozen=True)
class PhotometricNormalizationConfig:
    """Validated configuration for video-level luminance normalization."""

    mode: str = "none"
    target_median: float = 0.5
    target_span: float = 0.5
    low_quantile: float = 0.10
    high_quantile: float = 0.90
    max_luma_shift: float = 0.20
    min_contrast_scale: float = 0.75
    max_contrast_scale: float = 1.33
    black_threshold: int = 8
    stats_max_frames: int = 32
    stats_spatial_stride: int = 4
    min_valid_stat_pixels: int = 256
    min_valid_stat_fraction: float = 0.05

    @property
    def enabled(self):
        return self.mode != "none"


def _section_value(section, key, default):
    if section is None:
        return default
    return getattr(section, key, default)


def _finite_real(value, field_name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"DATASET.PHOTOMETRIC_NORMALIZATION.{field_name} must be a finite number, "
            f"got: {value!r}"
        )
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(
            f"DATASET.PHOTOMETRIC_NORMALIZATION.{field_name} must be a finite number, "
            f"got: {value!r}"
        )
    return value


def _positive_int(value, field_name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(
            f"DATASET.PHOTOMETRIC_NORMALIZATION.{field_name} must be a positive integer, "
            f"got: {value!r}"
        )
    return int(value)


def resolve_photometric_normalization_config(configs):
    """Resolve and validate the optional photometric normalization policy.

    A missing section resolves to ``mode=none`` so existing configs retain
    exactly the previous RGB input path.
    """

    dataset_section = getattr(configs, "DATASET", None)
    section = (
        getattr(dataset_section, "PHOTOMETRIC_NORMALIZATION", None)
        if dataset_section is not None
        else None
    )
    allowed_keys = {
        "MODE",
        "TARGET_MEDIAN",
        "TARGET_SPAN",
        "LOW_QUANTILE",
        "HIGH_QUANTILE",
        "MAX_LUMA_SHIFT",
        "MIN_CONTRAST_SCALE",
        "MAX_CONTRAST_SCALE",
        "BLACK_THRESHOLD",
        "STATS_MAX_FRAMES",
        "STATS_SPATIAL_STRIDE",
        "MIN_VALID_STAT_PIXELS",
        "MIN_VALID_STAT_FRACTION",
    }
    if section is not None:
        if not hasattr(section, "keys"):
            raise ValueError(
                "DATASET.PHOTOMETRIC_NORMALIZATION must be a mapping of policy fields"
            )
        unknown_keys = set(section.keys()) - allowed_keys
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(
                "Unknown DATASET.PHOTOMETRIC_NORMALIZATION config field(s): "
                f"{unknown}"
            )

    defaults = PhotometricNormalizationConfig()
    mode = str(_section_value(section, "MODE", defaults.mode)).strip().lower()
    mode_aliases = {
        "off": "none",
        "disabled": "none",
        "p0": "none",
        "p1": "center",
        "luma_center": "center",
        "p2": "center_contrast",
        "luma_center_contrast": "center_contrast",
    }
    mode = mode_aliases.get(mode, mode)
    if mode not in _SUPPORTED_MODES:
        raise ValueError(
            "DATASET.PHOTOMETRIC_NORMALIZATION.MODE must be one of "
            f"{sorted(_SUPPORTED_MODES)}, got: {mode!r}"
        )

    target_median = _finite_real(
        _section_value(section, "TARGET_MEDIAN", defaults.target_median),
        "TARGET_MEDIAN",
    )
    target_span = _finite_real(
        _section_value(section, "TARGET_SPAN", defaults.target_span),
        "TARGET_SPAN",
    )
    low_quantile = _finite_real(
        _section_value(section, "LOW_QUANTILE", defaults.low_quantile),
        "LOW_QUANTILE",
    )
    high_quantile = _finite_real(
        _section_value(section, "HIGH_QUANTILE", defaults.high_quantile),
        "HIGH_QUANTILE",
    )
    max_luma_shift = _finite_real(
        _section_value(section, "MAX_LUMA_SHIFT", defaults.max_luma_shift),
        "MAX_LUMA_SHIFT",
    )
    min_contrast_scale = _finite_real(
        _section_value(
            section, "MIN_CONTRAST_SCALE", defaults.min_contrast_scale
        ),
        "MIN_CONTRAST_SCALE",
    )
    max_contrast_scale = _finite_real(
        _section_value(
            section, "MAX_CONTRAST_SCALE", defaults.max_contrast_scale
        ),
        "MAX_CONTRAST_SCALE",
    )

    black_threshold = _section_value(
        section, "BLACK_THRESHOLD", defaults.black_threshold
    )
    if (
        isinstance(black_threshold, bool)
        or not isinstance(black_threshold, Integral)
        or not 0 <= black_threshold <= 255
    ):
        raise ValueError(
            "DATASET.PHOTOMETRIC_NORMALIZATION.BLACK_THRESHOLD must be an integer "
            f"in [0, 255], got: {black_threshold!r}"
        )
    black_threshold = int(black_threshold)

    stats_max_frames = _positive_int(
        _section_value(section, "STATS_MAX_FRAMES", defaults.stats_max_frames),
        "STATS_MAX_FRAMES",
    )
    stats_spatial_stride = _positive_int(
        _section_value(
            section, "STATS_SPATIAL_STRIDE", defaults.stats_spatial_stride
        ),
        "STATS_SPATIAL_STRIDE",
    )
    min_valid_stat_pixels = _positive_int(
        _section_value(
            section,
            "MIN_VALID_STAT_PIXELS",
            defaults.min_valid_stat_pixels,
        ),
        "MIN_VALID_STAT_PIXELS",
    )
    min_valid_stat_fraction = _finite_real(
        _section_value(
            section,
            "MIN_VALID_STAT_FRACTION",
            defaults.min_valid_stat_fraction,
        ),
        "MIN_VALID_STAT_FRACTION",
    )

    if not 0.0 <= target_median <= 1.0:
        raise ValueError(
            "DATASET.PHOTOMETRIC_NORMALIZATION.TARGET_MEDIAN must be in [0, 1]"
        )
    if not 0.0 < target_span <= 1.0:
        raise ValueError(
            "DATASET.PHOTOMETRIC_NORMALIZATION.TARGET_SPAN must be in (0, 1]"
        )
    if not 0.0 <= low_quantile < 0.5 < high_quantile <= 1.0:
        raise ValueError(
            "DATASET.PHOTOMETRIC_NORMALIZATION quantiles must satisfy "
            "0 <= LOW_QUANTILE < 0.5 < HIGH_QUANTILE <= 1"
        )
    if not 0.0 <= max_luma_shift <= 1.0:
        raise ValueError(
            "DATASET.PHOTOMETRIC_NORMALIZATION.MAX_LUMA_SHIFT must be in [0, 1]"
        )
    if not 0.0 < min_contrast_scale <= 1.0:
        raise ValueError(
            "DATASET.PHOTOMETRIC_NORMALIZATION.MIN_CONTRAST_SCALE must be in (0, 1]"
        )
    if not 1.0 <= max_contrast_scale:
        raise ValueError(
            "DATASET.PHOTOMETRIC_NORMALIZATION.MAX_CONTRAST_SCALE must be >= 1"
        )
    if min_contrast_scale > max_contrast_scale:
        raise ValueError(
            "DATASET.PHOTOMETRIC_NORMALIZATION.MIN_CONTRAST_SCALE must not exceed "
            "MAX_CONTRAST_SCALE"
        )
    if not 0.0 <= min_valid_stat_fraction <= 1.0:
        raise ValueError(
            "DATASET.PHOTOMETRIC_NORMALIZATION.MIN_VALID_STAT_FRACTION must be "
            "in [0, 1]"
        )

    return PhotometricNormalizationConfig(
        mode=mode,
        target_median=target_median,
        target_span=target_span,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        max_luma_shift=max_luma_shift,
        min_contrast_scale=min_contrast_scale,
        max_contrast_scale=max_contrast_scale,
        black_threshold=black_threshold,
        stats_max_frames=stats_max_frames,
        stats_spatial_stride=stats_spatial_stride,
        min_valid_stat_pixels=min_valid_stat_pixels,
        min_valid_stat_fraction=min_valid_stat_fraction,
    )


def _axis_connected_black_padding_mask(values, threshold):
    """Return near-black runs reaching a border along a row or column.

    This intentionally matches the project's established strict OpenFace
    padding mask.  It is not a general 4/8-connected component operation;
    bent or diagonal near-black paths can remain part of the valid content.
    """

    black_mask = (values <= float(threshold)).all(dim=1, keepdim=True)
    black_int = black_mask.to(dtype=torch.int8)
    top = black_int.cumprod(dim=2).bool()
    bottom = black_int.flip(dims=(2,)).cumprod(dim=2).flip(dims=(2,)).bool()
    left = black_int.cumprod(dim=3).bool()
    right = black_int.flip(dims=(3,)).cumprod(dim=3).flip(dims=(3,)).bool()
    return top | bottom | left | right


def _sample_frame_indices(num_frames, max_frames, device):
    sample_count = min(num_frames, max_frames)
    if sample_count <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    return (
        torch.linspace(0, num_frames - 1, steps=sample_count, device=device)
        .round()
        .long()
        .unique()
    )


def _sample_stat_values(video_tensor, scale, config, weights):
    frame_indices = _sample_frame_indices(
        video_tensor.size(0), config.stats_max_frames, video_tensor.device
    )
    sampled_values = video_tensor[frame_indices].to(dtype=torch.float32) / scale
    sampled_luma = (sampled_values * weights).sum(dim=1, keepdim=True)
    threshold = float(config.black_threshold) / 255.0
    sampled_valid = ~_axis_connected_black_padding_mask(
        sampled_values, threshold=threshold
    )
    sampled_luma = sampled_luma[
        :, :, :: config.stats_spatial_stride, :: config.stats_spatial_stride
    ]
    sampled_valid = sampled_valid[
        :, :, :: config.stats_spatial_stride, :: config.stats_spatial_stride
    ]
    return sampled_luma[sampled_valid], sampled_valid.numel()


def _apply_mapping_chunk(
    video_chunk,
    *,
    scale,
    weights,
    threshold,
    median,
    target_center,
    contrast_scale,
):
    values = video_chunk.to(dtype=torch.float32) / scale
    luma = (values * weights).sum(dim=1, keepdim=True)
    border_black = _axis_connected_black_padding_mask(values, threshold=threshold)
    adjusted_luma = target_center + contrast_scale * (luma - median)

    # Use one shared RGB delta per pixel and constrain it to the intersection
    # of all three channels' available gamut.  This keeps channel differences
    # (and therefore chroma) unchanged instead of clipping channels separately.
    desired_delta = adjusted_luma - luma
    min_delta = -values.amin(dim=1, keepdim=True)
    max_delta = 1.0 - values.amax(dim=1, keepdim=True)
    safe_delta = torch.maximum(torch.minimum(desired_delta, max_delta), min_delta)
    adjusted_values = values + safe_delta
    output = torch.where((~border_black).expand_as(values), adjusted_values, values)
    if video_chunk.dtype.is_floating_point:
        return output.to(dtype=video_chunk.dtype)

    # Quantize the shared delta once, then add that same integer offset to all
    # channels.  This preserves uint8 channel differences exactly as well.
    quantized_delta = (safe_delta * scale).round()
    output_counts = video_chunk.to(dtype=torch.float32) + quantized_delta
    output_counts = torch.where(
        (~border_black).expand_as(values),
        output_counts,
        video_chunk.to(dtype=torch.float32),
    )
    return output_counts.clamp(0.0, scale).to(dtype=video_chunk.dtype)


def apply_photometric_normalization(video_tensor, config):
    """Apply one robust luminance transform to an entire RGB video tensor.

    Args:
        video_tensor: Tensor shaped ``[T, 3, H, W]``. Inputs must be uint8 in
            ``[0, 255]`` or floating point in ``[0, 1]``.
        config: :class:`PhotometricNormalizationConfig`.

    Returns:
        A tensor with the original shape and dtype.  ``mode=none`` returns the
        input object unchanged to preserve the legacy P0 path exactly.
    """

    if not isinstance(config, PhotometricNormalizationConfig):
        raise TypeError(
            "config must be a PhotometricNormalizationConfig, "
            f"got: {type(config).__name__}"
        )
    if not config.enabled:
        return video_tensor
    if video_tensor.ndim != 4 or video_tensor.size(1) != 3:
        raise ValueError(
            "video_tensor must have shape [T, 3, H, W], "
            f"got: {tuple(video_tensor.shape)}"
        )
    if video_tensor.size(0) == 0:
        return video_tensor

    if video_tensor.dtype != torch.uint8 and not video_tensor.dtype.is_floating_point:
        raise TypeError(
            "video_tensor must be uint8 in [0, 255] or floating point in [0, 1], "
            f"got: {video_tensor.dtype}"
        )

    scale = 1.0 if video_tensor.dtype.is_floating_point else 255.0
    if video_tensor.dtype.is_floating_point:
        if not bool(torch.isfinite(video_tensor).all()):
            raise ValueError("floating video_tensor contains NaN or infinity")
        if float(video_tensor.min()) < 0.0 or float(video_tensor.max()) > 1.0:
            raise ValueError("floating video_tensor values must be in [0, 1]")

    weights = torch.tensor(
        _LUMA_WEIGHTS, dtype=torch.float32, device=video_tensor.device
    ).view(1, 3, 1, 1)
    stat_values, sampled_pixel_count = _sample_stat_values(
        video_tensor, scale=scale, config=config, weights=weights
    )
    if stat_values.numel() < config.min_valid_stat_pixels:
        return video_tensor
    valid_fraction = stat_values.numel() / max(sampled_pixel_count, 1)
    if valid_fraction < config.min_valid_stat_fraction:
        return video_tensor

    median = torch.quantile(stat_values, 0.5)
    shift = torch.clamp(
        stat_values.new_tensor(config.target_median) - median,
        min=-config.max_luma_shift,
        max=config.max_luma_shift,
    )
    target_center = median + shift

    if config.mode == "center":
        contrast_scale = stat_values.new_tensor(1.0)
    elif config.mode == "center_contrast":
        quantiles = torch.quantile(
            stat_values,
            stat_values.new_tensor([config.low_quantile, config.high_quantile]),
        )
        observed_span = quantiles[1] - quantiles[0]
        if float(observed_span) <= 1e-6:
            contrast_scale = stat_values.new_tensor(1.0)
        else:
            contrast_scale = torch.clamp(
                stat_values.new_tensor(config.target_span) / observed_span,
                min=config.min_contrast_scale,
                max=config.max_contrast_scale,
            )
    else:
        raise AssertionError(f"Unhandled photometric mode: {config.mode}")

    threshold = float(config.black_threshold) / 255.0
    output = torch.empty_like(video_tensor)
    for start in range(0, video_tensor.size(0), _APPLY_CHUNK_SIZE):
        end = min(start + _APPLY_CHUNK_SIZE, video_tensor.size(0))
        output[start:end] = _apply_mapping_chunk(
            video_tensor[start:end],
            scale=scale,
            weights=weights,
            threshold=threshold,
            median=median,
            target_center=target_center,
            contrast_scale=contrast_scale,
        )
    return output
