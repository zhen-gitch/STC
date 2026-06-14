import pytest
import torch

from src.datasets.input_variants import apply_input_variant, normalize_input_variant


def _sample_video():
    return torch.arange(2 * 3 * 8 * 8, dtype=torch.uint8).reshape(2, 3, 8, 8)


def test_rgb_variant_preserves_input():
    video = _sample_video()
    output = apply_input_variant(video, "rgb")

    assert torch.equal(output, video)
    assert output.dtype == video.dtype
    assert output.shape == video.shape


def test_grayscale_variant_replicates_channels():
    video = _sample_video()
    output = apply_input_variant(video, "grayscale")

    assert output.dtype == video.dtype
    assert output.shape == video.shape
    assert torch.equal(output[:, 0], output[:, 1])
    assert torch.equal(output[:, 1], output[:, 2])


def test_blur_variant_preserves_shape_and_changes_values():
    video = _sample_video()
    output = apply_input_variant(video, "blur")

    assert output.dtype == video.dtype
    assert output.shape == video.shape
    assert not torch.equal(output, video)


def test_mask_variants_preserve_center_and_erase_boundaries():
    video = torch.full((1, 3, 16, 16), 255, dtype=torch.uint8)
    center_masked = apply_input_variant(video, "center_mask")
    boundary_erased = apply_input_variant(video, "boundary_erased")

    assert center_masked[:, :, 8, 8].sum() > 0
    assert boundary_erased[:, :, 8, 8].sum() > 0
    assert center_masked[:, :, 0, 0].sum() == 0
    assert boundary_erased[:, :, 0, 0].sum() == 0
    assert boundary_erased.sum() > center_masked.sum()


def test_input_variant_aliases_and_reserved_values():
    assert normalize_input_variant("gray") == "grayscale"
    assert normalize_input_variant("masked_face") == "center_mask"

    with pytest.raises(ValueError, match="landmark_heatmap"):
        normalize_input_variant("landmark_heatmap")

    with pytest.raises(ValueError, match="Unsupported"):
        normalize_input_variant("unknown_variant")
