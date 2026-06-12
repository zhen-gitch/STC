import torch
from omegaconf import OmegaConf

from tests.test_mtl_lite_forward import DummyBackbone, minimal_mtl_lite_config


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
