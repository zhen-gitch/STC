import torch
import torch.nn.functional as F

from tests.test_model_forward import DummyBackbone, _minimal_config


def _build_model_with_dummy_backbone(monkeypatch):
    import src.models.end_to_end as end_to_end

    cfg = _minimal_config()

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        output_dim = cfg.BACKBONE_OUT_DIMS[model_name]
        return DummyBackbone(output_dim=output_dim)

    monkeypatch.setattr(end_to_end, "build_feature_backbone", build_dummy_backbone)
    model = end_to_end.EndToEndDepressionModel(cfg)
    model.eval()
    return model, cfg


def test_ccc_loss_matches_one_minus_ccc_metric():
    from src.metrics.metrics import ConcordanceCorrCoefMetric, concordance_ccc_loss

    preds = torch.tensor([0.05, 0.25, 0.55, 0.80], dtype=torch.float32)
    targets = torch.tensor([0.00, 0.30, 0.50, 0.95], dtype=torch.float32)

    loss = concordance_ccc_loss(preds, targets)
    metric = ConcordanceCorrCoefMetric()
    metric.update(preds, targets)
    ccc = metric.compute()

    assert torch.allclose(loss, 1.0 - ccc, atol=1e-6)


def test_prediction_for_metrics_restores_real_bdi_scale(monkeypatch):
    model, _ = _build_model_with_dummy_backbone(monkeypatch)
    normalized_preds = torch.tensor([-0.5, 0.0, 0.5, 1.2], dtype=torch.float32)

    metric_preds = model._prediction_for_metrics(normalized_preds)

    expected = torch.tensor([0.0, 0.0, 31.5, 63.0], dtype=torch.float32)
    assert torch.allclose(metric_preds, expected)


def test_multitask_regression_loss_uses_normalized_bdi_scale(monkeypatch):
    from src.metrics.metrics import concordance_ccc_loss
    from src.models.task_heads import coral_loss, get_coral_levels

    model, cfg = _build_model_with_dummy_backbone(monkeypatch)
    true_bdi = torch.tensor([0.0, 31.5, 63.0], dtype=torch.float32)
    true_bdi_norm = true_bdi / float(cfg.EXTRACT_FEATURE.MAX_SCORE)
    bdi_preds = torch.tensor([0.05, 0.45, 0.90], dtype=torch.float32)
    true_cls_levels = get_coral_levels(torch.tensor([0, 3, 6]), model.num_classes)
    cls_preds = torch.randn(3, model.num_classes - 1)
    loss_con = torch.tensor(0.25)

    loss_reg, loss_cls, losses, loss_mse, loss_ccc, loss_dist = model._compute_multitask_losses(
        bdi_preds=bdi_preds,
        cls_preds=cls_preds,
        true_bdi=true_bdi,
        true_bdi_norm=true_bdi_norm,
        true_cls_levels=true_cls_levels,
        loss_con=loss_con,
    )

    expected_mse = F.mse_loss(bdi_preds, true_bdi_norm)
    expected_ccc = concordance_ccc_loss(bdi_preds, true_bdi_norm)
    expected_dist = (
        model.pred_mean_loss_weight * (bdi_preds.mean() - true_bdi_norm.mean()).pow(2)
        + model.pred_std_loss_weight
        * F.relu(true_bdi_norm.std(unbiased=False) - bdi_preds.std(unbiased=False)).pow(2)
    )
    expected_reg = expected_mse + model.ccc_loss_weight * expected_ccc + expected_dist

    assert torch.allclose(loss_mse, expected_mse)
    assert torch.allclose(loss_ccc, expected_ccc)
    assert torch.allclose(loss_dist, expected_dist)
    assert torch.allclose(loss_reg, expected_reg)
    assert torch.allclose(loss_cls, coral_loss(cls_preds, true_cls_levels))
    assert losses[0] is loss_reg
    assert losses[1] is loss_cls
    assert losses[2] is loss_con
