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


def test_black_replacement_variants_remove_near_black_pixels():
    video = torch.full((1, 3, 8, 8), 200, dtype=torch.uint8)
    video[:, :, :2, :] = 0
    video[:, :, 4, 4] = 0

    gray = apply_input_variant(video, "black_to_gray")
    mean = apply_input_variant(video, "black_to_mean")
    blurred = apply_input_variant(video, "black_to_blur")

    assert gray.dtype == video.dtype
    assert gray.shape == video.shape
    assert torch.equal(gray[:, :, 0, 0], torch.full((1, 3), 127, dtype=torch.uint8))
    assert torch.equal(mean[:, :, 0, 0], torch.full((1, 3), 200, dtype=torch.uint8))
    assert blurred[:, :, 0, 0].sum() > 0
    assert torch.equal(gray[:, :, 7, 7], video[:, :, 7, 7])


def test_border_black_to_gray_preserves_center_black_regions():
    video = torch.full((1, 3, 10, 10), 200, dtype=torch.uint8)
    video[:, :, 0, :] = 0
    video[:, :, :, 0] = 0
    video[:, :, 5, 5] = 0

    output = apply_input_variant(video, "border_black_to_gray")

    assert output.dtype == video.dtype
    assert output.shape == video.shape
    assert torch.equal(output[:, :, 0, 5], torch.full((1, 3), 127, dtype=torch.uint8))
    assert torch.equal(output[:, :, 5, 0], torch.full((1, 3), 127, dtype=torch.uint8))
    assert torch.equal(output[:, :, 5, 5], torch.zeros((1, 3), dtype=torch.uint8))
    assert torch.equal(output[:, :, 8, 8], video[:, :, 8, 8])


def test_border_black_feather_softens_boundary_without_touching_center_black():
    video = torch.full((1, 3, 12, 12), 220, dtype=torch.uint8)
    video[:, :, :2, :] = 0
    video[:, :, 6, 6] = 0

    output = apply_input_variant(video, "border_black_feather")

    assert output.dtype == video.dtype
    assert output.shape == video.shape
    assert output[:, :, 0, 6].float().mean() > 100
    assert 127 < output[:, :, 2, 6].float().mean() < 220
    assert torch.equal(output[:, :, 6, 6], torch.zeros((1, 3), dtype=torch.uint8))


def test_center_mask_black_to_gray_uses_gray_background():
    video = torch.full((1, 3, 16, 16), 255, dtype=torch.uint8)
    output = apply_input_variant(video, "center_mask_black_to_gray")

    assert output.dtype == video.dtype
    assert output.shape == video.shape
    assert output[:, :, 8, 8].float().mean() > 240
    assert torch.allclose(output[:, :, 0, 0].float(), torch.full((1, 3), 127.0), atol=1.0)


def test_soft_center_mask_blends_boundaries_to_gray():
    video = torch.full((1, 3, 16, 16), 255, dtype=torch.uint8)
    output = apply_input_variant(video, "soft_center_mask")

    assert output.dtype == video.dtype
    assert output.shape == video.shape
    assert output[:, :, 8, 8].float().mean() > 240
    assert torch.allclose(output[:, :, 0, 0].float(), torch.full((1, 3), 127.0), atol=1.0)


def test_inner_crop_resize_preserves_shape_and_removes_outer_border():
    video = torch.zeros((1, 3, 16, 16), dtype=torch.uint8)
    video[:, :, 4:12, 4:12] = 255
    output = apply_input_variant(video, "inner_crop_resize")

    assert output.dtype == video.dtype
    assert output.shape == video.shape
    assert output.sum() > video.sum()


def test_input_variant_aliases_and_reserved_values():
    assert normalize_input_variant("gray") == "grayscale"
    assert normalize_input_variant("masked_face") == "center_mask"
    assert normalize_input_variant("black_fill_gray") == "black_to_gray"
    assert normalize_input_variant("soft_mask") == "soft_center_mask"
    assert normalize_input_variant("border_black_gray") == "border_black_to_gray"
    assert normalize_input_variant("center_mask_gray_border") == "center_mask_black_to_gray"

    with pytest.raises(ValueError, match="landmark_heatmap"):
        normalize_input_variant("landmark_heatmap")

    with pytest.raises(ValueError, match="Unsupported"):
        normalize_input_variant("unknown_variant")
