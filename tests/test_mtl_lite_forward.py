import math

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class DummyBackbone(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))
        repeat_count = math.ceil(self.output_dim / pooled.size(1))
        return pooled.repeat(1, repeat_count)[:, : self.output_dim]


def minimal_mtl_lite_config():
    return OmegaConf.create(
        {
            "MODE": "full",
            "ACCELERATOR": "cpu",
            "PRECISION": "32-true",
            "LOG_DIR": "logs/test",
            "BACKBONE_OUT_DIMS": {
                "dummy_backbone": 8,
            },
            "EXTRACT_FEATURE": {
                "MAX_SCORE": 63,
                "BATCH_SIZE": 2,
                "MODEL_NAME": "dummy_backbone",
                "TIMM_PRETRAINED": False,
                "MODEL_WEIGHT_PATH": None,
                "CHUNK_SIZE": 3,
            },
            "PROCESS_TEMPORAL": {
                "HIDDEN_DIM": 8,
                "CLASS_STEP": 10,
                "DROPOUT": 0.0,
                "CCC_LOSS_WEIGHT": 0.05,
                "LEARNING_RATE": 1e-4,
                "WEIGHT_DECAY": 5e-4,
            },
            "LOSSES": {
                "ORDINAL_WEIGHT": 1.0,
                "CCC_WEIGHT": 0.05,
            },
        }
    )


@pytest.fixture()
def mtl_lite_model(monkeypatch):
    import src.models.mtl_lite as mtl_lite

    cfg = minimal_mtl_lite_config()

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        return DummyBackbone(output_dim=cfg.BACKBONE_OUT_DIMS[model_name])

    monkeypatch.setattr(mtl_lite, "build_feature_backbone", build_dummy_backbone)
    model = mtl_lite.MTLLiteDepressionModel(cfg)
    model.eval()
    return model, cfg


def test_mtl_lite_forward_returns_expected_shapes(mtl_lite_model):
    model, cfg = mtl_lite_model
    video = torch.randn(2, 4, 3, 16, 16)
    mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )

    with torch.no_grad():
        outputs = model(video, mask, return_features=True)

    assert outputs.bdi_pred.shape == (2,)
    assert outputs.ordinal_logits.shape == (2, model.num_classes - 1)
    assert outputs.shared_features.shape == (2, cfg.PROCESS_TEMPORAL.HIDDEN_DIM)
    assert torch.isfinite(outputs.bdi_pred).all()
    assert torch.isfinite(outputs.ordinal_logits).all()
    assert torch.isfinite(outputs.shared_features).all()


def test_mtl_lite_forward_handles_empty_mask(mtl_lite_model):
    model, _ = mtl_lite_model
    video = torch.randn(2, 4, 3, 16, 16)
    mask = torch.zeros(2, 4, dtype=torch.bool)

    with torch.no_grad():
        outputs = model(video, mask)

    assert outputs.bdi_pred.shape == (2,)
    assert outputs.ordinal_logits.shape == (2, model.num_classes - 1)
    assert outputs.shared_features is None
    assert torch.isfinite(outputs.bdi_pred).all()
    assert torch.isfinite(outputs.ordinal_logits).all()
