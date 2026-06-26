"""Tests for the RPDF Stage A1 layer-wise identity probe hooks.

These tests verify that ``MTLLiteDepressionModel`` can register forward hooks
on backbone sub-modules and collect per-layer video-level embeddings, while the
training path (no hooks registered, ``return_layer_features=False``) is
unchanged.

A small transformer-style dummy backbone (with ``patch_embed`` and ``blocks``)
is used so that hook registration, accumulation across chunks, and per-video
slicing are exercised end-to-end without requiring timm/torchvision.
"""

import math

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class _DummyPatchEmbed(nn.Module):
    """Project a frame to a sequence of tokens, mimicking ViT patch embedding."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=4, stride=4)

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, dim, h, w) -> (B, num_tokens, dim)
        out = self.proj(x)
        return out.flatten(2).transpose(1, 2)


class _DummyBlock(nn.Module):
    """A minimal transformer block returning (B, num_tokens, dim)."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x) + x


class DummyTransformerBackbone(nn.Module):
    """A ViT/DeiT-like backbone exposing ``patch_embed`` and ``blocks``.

    Its forward returns a pooled feature vector of size ``dim`` so it is
    compatible with ``MTLLiteDepressionModel.input_dim``.
    """

    def __init__(self, dim=8, num_blocks=12):
        super().__init__()
        self.dim = dim
        self.patch_embed = _DummyPatchEmbed(dim)
        self.blocks = nn.ModuleList([_DummyBlock(dim) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        # Pool to a single vector per frame (cls token), like num_classes=0.
        return tokens[:, 0]


def _minimal_config(dim=8):
    return OmegaConf.create(
        {
            "MODE": "mtl_lite",
            "ACCELERATOR": "cpu",
            "PRECISION": "32-true",
            "LOG_DIR": "logs/test",
            "BACKBONE_OUT_DIMS": {"dummy_transformer": dim},
            "EXTRACT_FEATURE": {
                "MAX_SCORE": 63,
                "BATCH_SIZE": 2,
                "MODEL_NAME": "dummy_transformer",
                "TIMM_PRETRAINED": False,
                "MODEL_WEIGHT_PATH": None,
                "CHUNK_SIZE": 2,
                "FREEZE_BACKBONE": False,
                "FINETUNE_LAST_N_BLOCKS": 0,
            },
            "PROCESS_TEMPORAL": {
                "HIDDEN_DIM": dim,
                "CLASS_STEP": 2,
                "DROPOUT": 0.0,
                "LEARNING_RATE": 1e-4,
                "WEIGHT_DECAY": 5e-4,
            },
            "MODEL": {
                "AUXILIARY_TASKS": {"ORDINAL_CLASSIFICATION": True},
            },
            "LOSSES": {
                "ORDINAL_WEIGHT": 1.0,
                "CCC_WEIGHT": 0.0,
            },
        }
    )


@pytest.fixture()
def probe_model(monkeypatch):
    import src.models.mtl_lite as mtl_lite

    cfg = _minimal_config(dim=8)

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        return DummyTransformerBackbone(dim=8, num_blocks=12)

    monkeypatch.setattr(mtl_lite, "build_feature_backbone", build_dummy_backbone)
    model = mtl_lite.MTLLiteDepressionModel(cfg)
    model.eval()
    return model, cfg


def _video_and_mask():
    # 2 videos, 4 frames each, last frame of video 2 is padding.
    video = torch.randn(2, 4, 3, 16, 16)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    return video, mask


def test_register_layer_hooks_resolves_transformer_blocks(probe_model):
    """All backbone layer names resolve on a transformer-style backbone."""
    import src.models.mtl_lite as mtl_lite

    model, _ = probe_model
    registered = model.register_layer_hooks(list(mtl_lite.LAYERWISE_PROBE_LAYERS))

    assert "layer_stem" in registered
    assert "layer_block_0" in registered
    assert "layer_block_11" in registered
    assert "layer_backbone_out" in registered
    assert "layer_temporal" in registered  # captured inline, not hooked
    assert "layer_shared" in registered
    assert model._layer_probe_active is True
    model.clear_layer_hooks()
    assert model._layer_probe_active is False


def test_forward_returns_layer_features_when_requested(probe_model):
    model, _ = probe_model
    video, mask = _video_and_mask()

    model.register_layer_hooks()
    try:
        with torch.no_grad():
            outputs = model(video, mask, return_layer_features=True)
    finally:
        model.clear_layer_hooks()

    assert outputs.layer_features is not None
    # Inline-captured layers are always present.
    assert "layer_temporal" in outputs.layer_features
    assert "layer_shared" in outputs.layer_features
    assert "layer_backbone_out" in outputs.layer_features
    # Hooked backbone layers should also be present.
    assert "layer_stem" in outputs.layer_features
    assert "layer_block_0" in outputs.layer_features
    assert "layer_block_11" in outputs.layer_features

    for name, vec in outputs.layer_features.items():
        assert vec.shape[0] == 2, f"{name} should have one vector per video"
        assert vec.dim() == 2, f"{name} should be (B, D)"
        assert torch.isfinite(vec).all(), f"{name} has non-finite values"


def test_training_path_is_unchanged_without_hooks(probe_model):
    """When no hooks are registered, forward behaves exactly as before."""
    model, _ = probe_model
    video, mask = _video_and_mask()

    assert model._layer_probe_active is False
    with torch.no_grad():
        outputs = model(video, mask, return_features=True)

    # No layer features collected when return_layer_features is False (default).
    assert outputs.layer_features is None
    assert outputs.shared_features is not None
    assert outputs.bdi_pred.shape == (2,)


def test_return_layer_features_no_hooks_only_inline_layers(probe_model):
    """return_layer_features without registered hooks yields inline layers only."""
    model, _ = probe_model
    video, mask = _video_and_mask()

    with torch.no_grad():
        outputs = model(video, mask, return_layer_features=True)

    assert outputs.layer_features is not None
    # Inline layers are always available even without hooks.
    assert set(outputs.layer_features.keys()) >= {
        "layer_temporal",
        "layer_shared",
        "layer_backbone_out",
    }
    # No backbone sub-module layers without hooks.
    assert "layer_stem" not in outputs.layer_features
    assert "layer_block_0" not in outputs.layer_features


def test_layer_features_match_valid_frame_counts(probe_model):
    """Hooked layers accumulate per valid-frame order across chunks."""
    model, _ = probe_model
    video, mask = _video_and_mask()

    model.register_layer_hooks(["layer_block_3"])
    try:
        with torch.no_grad():
            outputs = model(video, mask, return_layer_features=True)
    finally:
        model.clear_layer_hooks()

    assert "layer_block_3" in outputs.layer_features
    vec = outputs.layer_features["layer_block_3"]
    # 2 videos -> 2 pooled vectors.
    assert vec.shape[0] == 2
    assert torch.isfinite(vec).all()


def test_clear_layer_hooks_removes_handles(probe_model):
    model, _ = probe_model
    model.register_layer_hooks()
    assert model._layer_probe_handles, "handles should be registered"
    model.clear_layer_hooks()
    assert model._layer_probe_handles == []
    assert model._layer_probe_cache == {}
    assert model._layer_probe_active is False
