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
    central_face_masked = apply_input_variant(video, "central_face_mask")
    boundary_erased = apply_input_variant(video, "boundary_erased")

    assert center_masked[:, :, 8, 8].sum() > 0
    assert central_face_masked[:, :, 8, 8].sum() > 0
    assert boundary_erased[:, :, 8, 8].sum() > 0
    assert center_masked[:, :, 0, 0].sum() == 0
    assert central_face_masked[:, :, 0, 0].sum() == 0
    assert boundary_erased[:, :, 0, 0].sum() == 0
    assert central_face_masked.sum() > boundary_erased.sum()
    assert boundary_erased.sum() > center_masked.sum()


def test_central_face_mask_covers_face_landmark_regions_more_broadly():
    video = torch.full((1, 3, 112, 112), 255, dtype=torch.uint8)

    center_masked = apply_input_variant(video, "center_mask")
    central_face_masked = apply_input_variant(video, "central_face_mask")

    # Current center_mask is intentionally preserved as a tiny nose/mouth patch
    # for historical ablation reproducibility.
    assert center_masked[:, :, 31, 35].sum() == 0
    assert center_masked[:, :, 80, 56].sum() == 0

    # central_face_mask is the new control: it retains approximate eye, nose,
    # mouth, and cheek regions while still erasing image corners.
    assert central_face_masked[:, :, 31, 35].sum() > 0
    assert central_face_masked[:, :, 31, 76].sum() > 0
    assert central_face_masked[:, :, 80, 56].sum() > 0
    assert central_face_masked[:, :, 56, 24].sum() > 0
    assert central_face_masked[:, :, 0, 0].sum() == 0

    center_area = (center_masked[:, 0] > 0).float().mean()
    central_face_area = (central_face_masked[:, 0] > 0).float().mean()
    assert 0.12 < float(center_area) < 0.15
    assert 0.45 < float(central_face_area) < 0.50


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


def test_edge_soften_only_preserves_large_black_areas_and_softens_boundary():
    video = torch.full((1, 3, 24, 24), 220, dtype=torch.uint8)
    # Wide border-connected black strips; the interior of the strip should stay black.
    video[:, :, :6, :] = 0
    video[:, :, :, :6] = 0
    # Center black dot should remain untouched.
    video[:, :, 12, 12] = 0

    output = apply_input_variant(video, "edge_soften_only")

    assert output.dtype == video.dtype
    assert output.shape == video.shape
    # Center black pixel preserved.
    assert torch.equal(output[:, :, 12, 12], torch.zeros((1, 3), dtype=torch.uint8))
    # Far interior preserved.
    assert torch.equal(output[:, :, 15, 15], video[:, :, 15, 15])
    # Large black area interior remains black (well inside the 2-pixel edge band).
    assert torch.equal(output[:, :, 2, 2], torch.zeros((1, 3), dtype=torch.uint8))
    # Boundary edge pixels should differ from original (softened).
    assert not torch.equal(output[:, :, 6, 10], video[:, :, 6, 10])


def test_border_blur_fill_replaces_border_black_with_neighbor_blur():
    video = torch.full((1, 3, 16, 16), 200, dtype=torch.uint8)
    # Border-connected black strip on the left.
    video[:, :, :, :3] = 0
    # Center black dot should remain untouched.
    video[:, :, 8, 8] = 0

    output = apply_input_variant(video, "border_blur_fill")

    assert output.dtype == video.dtype
    assert output.shape == video.shape
    # Center black pixel preserved.
    assert torch.equal(output[:, :, 8, 8], torch.zeros((1, 3), dtype=torch.uint8))
    # Border black pixels are filled with neighbor-blur values (no longer black).
    assert output[:, :, 0, 1].float().mean() > 50
    assert output[:, :, 0, 1].float().mean() > 0
    # The fill should be close to the bright neighbor value (200).
    assert output[:, :, 0, 1].float().mean() >= 180
    # Interior bright region preserved.
    assert torch.equal(output[:, :, 5, 10], video[:, :, 5, 10])


def test_identity_texture_suppressed_preserves_shape_and_reduces_high_freq_variance():
    # Create a frame with strong high-frequency checkerboard noise on top of a
    # smooth foreground region.
    video = torch.full((1, 3, 32, 32), 150, dtype=torch.uint8)
    video[:, :, 8:24, 8:24] = 200
    # Add checkerboard noise inside the bright region.
    for i in range(8, 24):
        for j in range(8, 24):
            if (i + j) % 2 == 0:
                video[:, :, i, j] = 230
            else:
                video[:, :, i, j] = 170

    output = apply_input_variant(video, "identity_texture_suppressed")

    assert output.dtype == video.dtype
    assert output.shape == video.shape
    # The checkerboard high-frequency variance should be reduced inside the region.
    input_std = float(video[:, :, 8:24, 8:24].float().std())
    output_std = float(output[:, :, 8:24, 8:24].float().std())
    assert output_std < input_std
    # Coarse mean structure should remain similar (not collapsed to uniform).
    assert abs(float(output[:, :, 8:24, 8:24].float().mean()) - 200.0) < 20


def test_identity_texture_suppressed_edge_soften_combines_both_effects():
    video = torch.full((1, 3, 24, 24), 220, dtype=torch.uint8)
    # Wide border-connected black strip.
    video[:, :, :6, :] = 0
    # Add high-frequency noise in the interior.
    for i in range(10, 20):
        for j in range(10, 20):
            if (i + j) % 2 == 0:
                video[:, :, i, j] = 230
            else:
                video[:, :, i, j] = 210

    output = apply_input_variant(video, "identity_texture_suppressed_edge_soften")

    assert output.dtype == video.dtype
    assert output.shape == video.shape
    # High-frequency variance is reduced.
    input_std = float(video[:, :, 10:20, 10:20].float().std())
    output_std = float(output[:, :, 10:20, 10:20].float().std())
    assert output_std < input_std
    # Large black area interior remains black.
    assert torch.equal(output[:, :, 2, 2], torch.zeros((1, 3), dtype=torch.uint8))
    # Boundary edge pixels are softened.
    assert not torch.equal(output[:, :, 6, 10], video[:, :, 6, 10])


def test_input_variant_aliases_and_reserved_values():
    assert normalize_input_variant("gray") == "grayscale"
    assert normalize_input_variant("masked_face") == "center_mask"
    assert normalize_input_variant("face_oval_mask") == "central_face_mask"
    assert normalize_input_variant("central_face") == "central_face_mask"
    assert normalize_input_variant("black_fill_gray") == "black_to_gray"
    assert normalize_input_variant("soft_mask") == "soft_center_mask"
    assert normalize_input_variant("border_black_gray") == "border_black_to_gray"
    assert normalize_input_variant("center_mask_gray_border") == "center_mask_black_to_gray"
    assert normalize_input_variant("edge_soften") == "edge_soften_only"
    assert normalize_input_variant("border_blur") == "border_blur_fill"
    assert normalize_input_variant("identity_suppressed") == "identity_texture_suppressed"
    assert normalize_input_variant("id_suppressed") == "identity_texture_suppressed"
    assert normalize_input_variant("id_suppressed_edge") == "identity_texture_suppressed_edge_soften"

    with pytest.raises(ValueError, match="landmark_heatmap"):
        normalize_input_variant("landmark_heatmap")

    with pytest.raises(ValueError, match="Unsupported"):
        normalize_input_variant("unknown_variant")
