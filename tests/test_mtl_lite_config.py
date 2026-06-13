import torch
import torch.nn as nn
from omegaconf import OmegaConf

from tests.test_mtl_lite_forward import DummyBackbone, minimal_mtl_lite_config


class BlockedDummyBackbone(nn.Module):
    def __init__(self, output_dim, num_blocks=3):
        super().__init__()
        self.output_dim = output_dim
        self.stem = nn.Linear(3, output_dim)
        self.blocks = nn.ModuleList(
            [nn.Linear(output_dim, output_dim) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))
        features = self.stem(pooled)
        for block in self.blocks:
            features = block(features)
        return self.norm(features)


def test_mtl_lite_uses_loss_defaults_when_losses_section_is_missing(monkeypatch):
    import src.models.mtl_lite as mtl_lite

    cfg = minimal_mtl_lite_config()
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    del cfg.LOSSES

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        return DummyBackbone(output_dim=cfg.BACKBONE_OUT_DIMS[model_name])

    monkeypatch.setattr(mtl_lite, "build_feature_backbone", build_dummy_backbone)
    model = mtl_lite.MTLLiteDepressionModel(cfg)

    assert model.ordinal_weight == 1.0
    assert model.ccc_loss_weight == cfg.PROCESS_TEMPORAL.CCC_LOSS_WEIGHT


def test_mtl_lite_prediction_for_metrics_restores_bdi_scale(monkeypatch):
    import src.models.mtl_lite as mtl_lite

    cfg = minimal_mtl_lite_config()

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        return DummyBackbone(output_dim=cfg.BACKBONE_OUT_DIMS[model_name])

    monkeypatch.setattr(mtl_lite, "build_feature_backbone", build_dummy_backbone)
    model = mtl_lite.MTLLiteDepressionModel(cfg)
    normalized_preds = torch.tensor([-0.1, 0.0, 0.5, 1.2])

    metric_preds = model.prediction_for_metrics(normalized_preds)

    assert torch.allclose(metric_preds, torch.tensor([0.0, 0.0, 31.5, 63.0]))


def test_mtl_lite_can_disable_ordinal_auxiliary_task(monkeypatch):
    import src.models.mtl_lite as mtl_lite

    cfg = minimal_mtl_lite_config()
    cfg.MODEL = {
        "AUXILIARY_TASKS": {
            "ORDINAL_CLASSIFICATION": False,
        }
    }
    cfg.LOSSES.ORDINAL_WEIGHT = 0.0

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        return DummyBackbone(output_dim=cfg.BACKBONE_OUT_DIMS[model_name])

    monkeypatch.setattr(mtl_lite, "build_feature_backbone", build_dummy_backbone)
    model = mtl_lite.MTLLiteDepressionModel(cfg)
    video = torch.randn(2, 4, 3, 16, 16)
    mask = torch.ones(2, 4, dtype=torch.bool)
    labels = {
        "bdi_score": torch.tensor([10.0, 20.0]),
        "class_label": torch.tensor([1, 2]),
    }

    outputs = model(video, mask)
    losses = model.compute_losses(outputs, labels)

    assert model.ordinal_task_head is None
    assert outputs.ordinal_logits is None
    assert losses.ordinal is None
    assert torch.allclose(losses.total, losses.regression + model.ccc_loss_weight * losses.ccc)


def test_mtl_lite_keeps_backbone_trainable_by_default(monkeypatch):
    import src.models.mtl_lite as mtl_lite

    cfg = minimal_mtl_lite_config()

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        return BlockedDummyBackbone(output_dim=cfg.BACKBONE_OUT_DIMS[model_name])

    monkeypatch.setattr(mtl_lite, "build_feature_backbone", build_dummy_backbone)
    model = mtl_lite.MTLLiteDepressionModel(cfg)

    assert all(param.requires_grad for param in model.backbone.parameters())


def test_mtl_lite_can_freeze_entire_backbone(monkeypatch):
    import src.models.mtl_lite as mtl_lite

    cfg = minimal_mtl_lite_config()
    cfg.EXTRACT_FEATURE.FREEZE_BACKBONE = True
    cfg.EXTRACT_FEATURE.FINETUNE_LAST_N_BLOCKS = 0

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        return BlockedDummyBackbone(output_dim=cfg.BACKBONE_OUT_DIMS[model_name])

    monkeypatch.setattr(mtl_lite, "build_feature_backbone", build_dummy_backbone)
    model = mtl_lite.MTLLiteDepressionModel(cfg)

    assert not any(param.requires_grad for param in model.backbone.parameters())
    assert any(param.requires_grad for param in model.temporal_encoder.parameters())
    assert any(param.requires_grad for param in model.reg_task_head.parameters())


def test_mtl_lite_can_finetune_last_backbone_blocks(monkeypatch):
    import src.models.mtl_lite as mtl_lite

    cfg = minimal_mtl_lite_config()
    cfg.EXTRACT_FEATURE.FREEZE_BACKBONE = True
    cfg.EXTRACT_FEATURE.FINETUNE_LAST_N_BLOCKS = 1

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        return BlockedDummyBackbone(output_dim=cfg.BACKBONE_OUT_DIMS[model_name], num_blocks=3)

    monkeypatch.setattr(mtl_lite, "build_feature_backbone", build_dummy_backbone)
    model = mtl_lite.MTLLiteDepressionModel(cfg)

    assert not any(param.requires_grad for param in model.backbone.stem.parameters())
    assert not any(param.requires_grad for param in model.backbone.blocks[0].parameters())
    assert not any(param.requires_grad for param in model.backbone.blocks[1].parameters())
    assert all(param.requires_grad for param in model.backbone.blocks[2].parameters())
    assert all(param.requires_grad for param in model.backbone.norm.parameters())
