import math

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class DummyBackbone(nn.Module):
    """Small feature extractor used to avoid real timm/backbone dependencies."""

    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))
        repeat_count = math.ceil(self.output_dim / pooled.size(1))
        return pooled.repeat(1, repeat_count)[:, : self.output_dim]


def _minimal_config():
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
                "NUM_WORKERS": 0,
                "BATCH_SIZE": 2,
                "MODEL_NAME": "dummy_backbone",
                "TIMM_PRETRAINED": False,
                "MODEL_WEIGHT_PATH": None,
                "CHUNK_SIZE": 3,
                "LEARNING_RATE": 1e-5,
                "WEIGHT_DECAY": 5e-4,
                "USE_GRAD_CHECKPOINTING": False,
                "FINETUNE_LAST_N_BLOCKS": 0,
                "CLEAR_CUDA_CACHE": False,
            },
            "PROCESS_TEMPORAL": {
                "HIDDEN_DIM": 8,
                "EXPERT_DIM": 8,
                "CONTRASTIVE_DIM": 4,
                "MAX_SEQ_LEN": 4,
                "CLASS_STEP": 10,
                "SAMPLE_STEP": 1,
                "NUM_SHARED_GEN": 1,
                "NUM_SHARED_CON": 1,
                "NUM_SPECIFIC": 1,
                "TEMPERATURE": 0.10,
                "BDI_SIGMA": 5.0,
                "CCC_LOSS_WEIGHT": 0.05,
                "LDS_SIGMA": 2.0,
                "LDS_SEVERITY_ALPHA": 0.2,
                "PRED_MEAN_LOSS_WEIGHT": 0.02,
                "PRED_STD_LOSS_WEIGHT": 0.05,
                "CLS_LOSS_WEIGHT": 1.0,
                "CON_LOSS_WEIGHT": 0.0,
                "DISABLE_MULTIVIEW_CONTRASTIVE": True,
                "USE_PCGRAD": False,
                "USE_UNCERTAINTY_WEIGHT": False,
                "DROPOUT": 0.0,
                "ADVERSARIAL_MASK": 999,
                "MASK_UPDATE_INTERVAL": 10,
                "BASE_MASK_PROB": 0.05,
                "MAX_MASK_PROB": 0.20,
                "KERNEL_SIZE": 1,
                "SIGMA": 1.0,
                "MOVING_AVG_KERNELS": [1],
                "GATE_TEMPERATURE": 4.0,
                "GATE_ENTROPY_WEIGHT": 0.02,
                "GATE_BALANCE_WEIGHT": 0.01,
                "MAX_EPOCHS": 1,
                "FREEZE_EPOCHS": 1,
                "WARMUP_EPOCHS": 1,
                "LEARNING_RATE": 1e-4,
                "WEIGHT_DECAY": 5e-4,
            },
            "VISUALIZATION": {
                "ENABLE": False,
                "REGRESSION_INTERVAL": 1,
                "EMBEDDING_INTERVAL": 1,
                "GATING_INTERVAL": 1,
            },
        }
    )


@pytest.fixture()
def model_with_dummy_backbone(monkeypatch):
    import src.models.end_to_end as end_to_end

    cfg = _minimal_config()

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        output_dim = cfg.BACKBONE_OUT_DIMS[model_name]
        return DummyBackbone(output_dim=output_dim)

    monkeypatch.setattr(end_to_end, "build_feature_backbone", build_dummy_backbone)
    model = end_to_end.EndToEndDepressionModel(cfg)
    model.eval()
    return model, cfg


def test_model_forward_returns_expected_head_shapes(model_with_dummy_backbone):
    model, cfg = model_with_dummy_backbone
    video = torch.randn(2, 4, 3, 16, 16)
    mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )

    with torch.no_grad():
        bdi_pred, cls_pred, con_embeds = model(video, mask, need_all_heads=True)

    assert bdi_pred.shape == (2,)
    assert cls_pred.shape == (2, model.num_classes - 1)
    assert con_embeds.shape == (2, cfg.PROCESS_TEMPORAL.CONTRASTIVE_DIM)
    assert torch.isfinite(bdi_pred).all()
    assert torch.isfinite(cls_pred).all()
    assert torch.isfinite(con_embeds).all()


def test_model_forward_regression_only_handles_empty_mask(model_with_dummy_backbone):
    model, _ = model_with_dummy_backbone
    video = torch.randn(2, 4, 3, 16, 16)
    mask = torch.zeros(2, 4, dtype=torch.bool)

    with torch.no_grad():
        bdi_pred, cls_pred, con_embeds = model(video, mask, need_all_heads=False)

    assert bdi_pred.shape == (2,)
    assert cls_pred is None
    assert con_embeds is None
    assert torch.isfinite(bdi_pred).all()
