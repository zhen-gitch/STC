from dataclasses import replace

import pytest
import torch
from omegaconf import OmegaConf

import src.datasets.dataset as dataset_module
import src.datasets.photometric_normalization as photometric_module
from src.datasets.photometric_normalization import (
    PhotometricNormalizationConfig,
    apply_photometric_normalization,
    resolve_photometric_normalization_config,
)


def _config(mode, **overrides):
    base = PhotometricNormalizationConfig(
        mode=mode,
        stats_max_frames=32,
        stats_spatial_stride=1,
        min_valid_stat_pixels=16,
    )
    return replace(base, **overrides)


def _luma(video):
    values = video.float() / 255.0
    weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    return (values * weights).sum(dim=1)


def test_p0_none_is_an_exact_noop():
    video = torch.randint(0, 256, (3, 3, 12, 12), dtype=torch.uint8)

    output = apply_photometric_normalization(video, _config("none"))

    assert output is video
    assert torch.equal(output, video)


def test_p1_aligns_video_luma_center_and_preserves_frame_difference():
    video = torch.empty((2, 3, 16, 16), dtype=torch.uint8)
    video[0].fill_(80)
    video[1].fill_(100)

    output = apply_photometric_normalization(video, _config("center"))

    assert output.dtype == video.dtype
    assert output.shape == video.shape
    assert abs(float(torch.quantile(_luma(output).flatten(), 0.5)) - 0.5) <= 1 / 255
    original_delta = video[1].to(torch.int16) - video[0].to(torch.int16)
    output_delta = output[1].to(torch.int16) - output[0].to(torch.int16)
    assert torch.equal(output_delta, original_delta)


def test_p2_increases_low_contrast_with_bounded_shared_scale():
    video = torch.full((2, 3, 16, 16), 80, dtype=torch.uint8)
    video[:, :, :, 8:] = 160
    p1 = apply_photometric_normalization(video, _config("center"))
    p2 = apply_photometric_normalization(video, _config("center_contrast"))

    p1_values = _luma(p1).flatten()
    p2_values = _luma(p2).flatten()
    p1_span = torch.quantile(p1_values, 0.9) - torch.quantile(p1_values, 0.1)
    p2_span = torch.quantile(p2_values, 0.9) - torch.quantile(p2_values, 0.1)

    assert p2_span > p1_span
    observed_scale = float(p2_span / p1_span)
    assert observed_scale == pytest.approx(1.33, abs=2 / 255)
    assert abs(float(torch.quantile(p2_values, 0.5)) - 0.5) <= 2 / 255


def test_p2_reduces_high_contrast_at_configured_lower_bound():
    video = torch.full((2, 3, 16, 16), 20, dtype=torch.uint8)
    video[:, :, :, 8:] = 220
    output = apply_photometric_normalization(
        video,
        _config(
            "center_contrast",
            min_contrast_scale=0.75,
            max_luma_shift=1.0,
        ),
    )

    source_values = _luma(video).flatten()
    output_values = _luma(output).flatten()
    source_span = torch.quantile(source_values, 0.9) - torch.quantile(source_values, 0.1)
    output_span = torch.quantile(output_values, 0.9) - torch.quantile(output_values, 0.1)

    assert float(output_span / source_span) == pytest.approx(0.75, abs=2 / 255)


def test_luma_delta_preserves_channel_differences_without_clipping():
    video = torch.empty((2, 3, 16, 16), dtype=torch.uint8)
    video[:, 0].fill_(140)
    video[:, 1].fill_(110)
    video[:, 2].fill_(80)

    output = apply_photometric_normalization(video, _config("center"))

    assert torch.equal(
        output[:, 0].to(torch.int16) - output[:, 1].to(torch.int16),
        video[:, 0].to(torch.int16) - video[:, 1].to(torch.int16),
    )
    assert torch.equal(
        output[:, 1].to(torch.int16) - output[:, 2].to(torch.int16),
        video[:, 1].to(torch.int16) - video[:, 2].to(torch.int16),
    )


@pytest.mark.parametrize(
    "rgb",
    [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (10, 80, 250),
    ],
)
def test_saturated_colors_keep_exact_uint8_channel_differences(rgb):
    video = torch.empty((2, 3, 16, 16), dtype=torch.uint8)
    for channel, value in enumerate(rgb):
        video[:, channel].fill_(value)

    output = apply_photometric_normalization(video, _config("center"))

    for left, right in ((0, 1), (0, 2), (1, 2)):
        assert torch.equal(
            output[:, left].to(torch.int16) - output[:, right].to(torch.int16),
            video[:, left].to(torch.int16) - video[:, right].to(torch.int16),
        )


def test_border_black_padding_is_excluded_and_preserved():
    video = torch.full((2, 3, 16, 16), 80, dtype=torch.uint8)
    video[:, :, :3, :] = 0
    video[:, :, :, :2] = 0

    output = apply_photometric_normalization(video, _config("center"))

    assert torch.equal(output[:, :, :3, :], video[:, :, :3, :])
    assert torch.equal(output[:, :, :, :2], video[:, :, :, :2])
    assert output[:, :, 8, 8].float().mean() > video[:, :, 8, 8].float().mean()


def test_black_threshold_keeps_eight_but_treats_nine_as_content():
    video = torch.full((2, 3, 16, 16), 80, dtype=torch.uint8)
    video[:, :, 0, :8] = 8
    video[:, :, 0, 8:] = 9

    output = apply_photometric_normalization(video, _config("center"))

    assert torch.equal(output[:, :, 0, :8], video[:, :, 0, :8])
    assert torch.all(output[:, :, 0, 8:] > video[:, :, 0, 8:])


@pytest.mark.parametrize(
    "video",
    [
        torch.zeros((2, 3, 16, 16), dtype=torch.uint8),
        torch.empty((0, 3, 16, 16), dtype=torch.uint8),
    ],
)
def test_degenerate_video_returns_unchanged(video):
    output = apply_photometric_normalization(video, _config("center_contrast"))

    assert torch.equal(output, video)


def test_too_few_valid_stat_pixels_returns_unchanged():
    video = torch.zeros((1, 3, 16, 16), dtype=torch.uint8)
    video[:, :, 7:9, 7:9] = 100

    output = apply_photometric_normalization(
        video,
        _config("center", min_valid_stat_pixels=16),
    )

    assert torch.equal(output, video)


def test_frame_permutation_is_equivariant_when_all_frames_feed_statistics():
    video = torch.stack(
        [torch.full((3, 16, 16), value, dtype=torch.uint8) for value in (60, 90, 120, 150)]
    )
    permutation = torch.tensor([2, 0, 3, 1])
    config = _config("center_contrast")

    output = apply_photometric_normalization(video, config)
    permuted_output = apply_photometric_normalization(video[permutation], config)

    assert torch.equal(permuted_output, output[permutation])


def test_chunked_application_matches_single_chunk(monkeypatch):
    generator = torch.Generator().manual_seed(42)
    video = torch.randint(
        20,
        220,
        (65, 3, 16, 16),
        dtype=torch.uint8,
        generator=generator,
    )
    config = _config("center_contrast", stats_max_frames=32)

    chunked = apply_photometric_normalization(video, config)
    monkeypatch.setattr(photometric_module, "_APPLY_CHUNK_SIZE", 128)
    single_chunk = apply_photometric_normalization(video, config)

    assert torch.equal(chunked, single_chunk)


def test_float_and_uint8_paths_agree_after_quantization():
    generator = torch.Generator().manual_seed(7)
    video = torch.randint(
        20,
        220,
        (4, 3, 16, 16),
        dtype=torch.uint8,
        generator=generator,
    )
    config = _config("center_contrast")

    uint8_output = apply_photometric_normalization(video, config)
    float_output = apply_photometric_normalization(video.float() / 255.0, config)
    requantized_float = (float_output * 255.0).round().to(torch.uint8)

    assert torch.equal(uint8_output, requantized_float)


def test_constant_nonblack_p2_falls_back_to_p1_contrast():
    video = torch.full((2, 3, 16, 16), 80, dtype=torch.uint8)

    p1 = apply_photometric_normalization(video, _config("center"))
    p2 = apply_photometric_normalization(video, _config("center_contrast"))

    assert torch.equal(p2, p1)


def test_isolated_center_black_pixel_is_treated_as_content():
    video = torch.full((2, 3, 16, 16), 80, dtype=torch.uint8)
    video[:, :, 8, 8] = 0

    output = apply_photometric_normalization(video, _config("center"))

    assert torch.all(output[:, :, 8, 8] > 0)


def test_low_valid_fraction_returns_unchanged_even_above_absolute_minimum():
    video = torch.zeros((32, 3, 32, 32), dtype=torch.uint8)
    video[:, :, 14:18, 14:18] = 100
    config = _config(
        "center",
        min_valid_stat_pixels=16,
        min_valid_stat_fraction=0.05,
    )

    output = apply_photometric_normalization(video, config)

    assert torch.equal(output, video)


def test_padding_length_does_not_change_valid_frame_output():
    dataset = dataset_module.AVECDataset.__new__(dataset_module.AVECDataset)
    dataset.photometric_normalization = _config("center")
    dataset.input_variant = "rgb"
    dataset.dataset_name = "val"
    dataset.transform = lambda value: value.float() / 255.0
    raw = torch.full((1, 3, 16, 16), 80, dtype=torch.uint8)

    dataset.max_len = 1
    unpadded = dataset._build_video_output(raw.clone())
    dataset.max_len = 3
    padded = dataset._build_video_output(raw.clone())

    assert torch.equal(padded[:1], unpadded)
    assert torch.count_nonzero(padded[1:]) == 0


def test_dataset_applies_photometric_before_input_variant(monkeypatch):
    calls = []
    dataset = dataset_module.AVECDataset.__new__(dataset_module.AVECDataset)
    dataset.photometric_normalization = _config("center")
    dataset.input_variant = "rgb"
    dataset.dataset_name = "val"
    dataset.max_len = 1
    dataset.transform = lambda value: value
    dataset._pad_video = lambda value: value

    def photometric(value, config):
        calls.append("photometric")
        return value

    def input_variant(value, variant):
        calls.append("input_variant")
        return value

    monkeypatch.setattr(dataset_module, "apply_photometric_normalization", photometric)
    monkeypatch.setattr(dataset_module, "apply_input_variant", input_variant)

    dataset._build_video_output(torch.zeros((1, 3, 4, 4), dtype=torch.uint8))

    assert calls == ["photometric", "input_variant"]


def test_missing_photometric_section_defaults_to_disabled():
    config = resolve_photometric_normalization_config(OmegaConf.create({"DATASET": {}}))

    assert config == PhotometricNormalizationConfig()
    assert config.enabled is False


@pytest.mark.parametrize(
    ("mode", "resolved"),
    [
        ("p0", "none"),
        ("p1", "center"),
        ("luma_center", "center"),
        ("p2", "center_contrast"),
        ("luma_center_contrast", "center_contrast"),
    ],
)
def test_mode_aliases_are_explicit(mode, resolved):
    cfg = OmegaConf.create(
        {"DATASET": {"PHOTOMETRIC_NORMALIZATION": {"MODE": mode}}}
    )

    assert resolve_photometric_normalization_config(cfg).mode == resolved


@pytest.mark.parametrize(
    "section",
    [
        {"MODE": "unknown"},
        {"LOW_QUANTILE": 0.6},
        {"HIGH_QUANTILE": 0.4},
        {"TARGET_SPAN": 0.0},
        {"MAX_LUMA_SHIFT": 1.1},
        {"MIN_CONTRAST_SCALE": 0.0},
        {"MAX_CONTRAST_SCALE": 0.9},
        {"BLACK_THRESHOLD": 256},
        {"STATS_MAX_FRAMES": 0},
        {"MIN_VALID_STAT_PIXELS": False},
        {"MIN_VALID_STAT_FRACTION": 1.1},
        {"MONITER": "typo"},
    ],
)
def test_invalid_photometric_config_is_rejected(section):
    cfg = OmegaConf.create(
        {"DATASET": {"PHOTOMETRIC_NORMALIZATION": section}}
    )

    with pytest.raises(ValueError):
        resolve_photometric_normalization_config(cfg)


def test_wrong_video_shape_is_rejected_when_enabled():
    with pytest.raises(ValueError, match=r"\[T, 3, H, W\]"):
        apply_photometric_normalization(
            torch.zeros((3, 16, 16), dtype=torch.uint8),
            _config("center"),
        )


@pytest.mark.parametrize(
    "video",
    [
        torch.full((1, 3, 4, 4), 2.0),
        torch.full((1, 3, 4, 4), float("nan")),
    ],
)
def test_invalid_float_domain_is_rejected(video):
    with pytest.raises(ValueError):
        apply_photometric_normalization(video, _config("center"))
