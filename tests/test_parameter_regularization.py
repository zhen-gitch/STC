import pytest
import torch

from tests.test_mtl_lite_forward import build_dummy_mtl_lite_model, minimal_mtl_lite_config


def _batch():
    video = torch.randn(2, 4, 3, 16, 16)
    mask = torch.ones(2, 4, dtype=torch.bool)
    labels = {
        "bdi_score": torch.tensor([10.0, 30.0]),
        "class_label": torch.tensor([1, 3]),
    }
    return video, mask, labels


def test_regularization_is_train_only_and_matches_weighted_formula(monkeypatch):
    cfg = minimal_mtl_lite_config()
    cfg.LOSSES.L1_WEIGHT = 0.01
    cfg.LOSSES.L2_WEIGHT = 0.1
    model = build_dummy_mtl_lite_model(monkeypatch, cfg)
    video, mask, labels = _batch()
    outputs = model(video, mask)

    train_losses = model.compute_losses(outputs, labels, stage="train")
    val_losses = model.compute_losses(outputs, labels, stage="val")

    expected = (
        train_losses.regression
        + model.ccc_loss_weight * train_losses.ccc
        + model.ordinal_weight * train_losses.ordinal
        + model.l1_weight * train_losses.l1
        + model.l2_weight * train_losses.l2
    )
    assert train_losses.l1 is not None and train_losses.l2 is not None
    assert torch.allclose(train_losses.total, expected)
    assert val_losses.l1 is None and val_losses.l2 is None
    val_expected = val_losses.regression + model.ccc_loss_weight * val_losses.ccc
    assert torch.allclose(val_losses.total, val_expected)
    train_losses.total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_regularization_excludes_bias_and_one_dimensional_parameters(monkeypatch):
    cfg = minimal_mtl_lite_config()
    cfg.LOSSES.L1_WEIGHT = 1.0
    cfg.LOSSES.L2_WEIGHT = 1.0
    model = build_dummy_mtl_lite_model(monkeypatch, cfg)
    for parameter in model.parameters():
        parameter.data.fill_(2.0)

    l1, l2 = model._parameter_regularization("train")
    eligible = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    assert eligible
    assert torch.allclose(l1, torch.tensor(2.0, device=l1.device, dtype=l1.dtype))
    assert torch.allclose(l2, torch.tensor(4.0, device=l2.device, dtype=l2.dtype))


def test_zero_regularization_preserves_baseline_total(monkeypatch):
    cfg = minimal_mtl_lite_config()
    model = build_dummy_mtl_lite_model(monkeypatch, cfg)
    video, mask, labels = _batch()
    outputs = model(video, mask)
    losses = model.compute_losses(outputs, labels, stage="train")
    baseline = losses.regression + model.ccc_loss_weight * losses.ccc + model.ordinal_weight * losses.ordinal
    assert torch.allclose(losses.total, baseline)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), "0.1", True])
def test_regularization_weights_reject_invalid_values(monkeypatch, value):
    cfg = minimal_mtl_lite_config()
    cfg.LOSSES.L1_WEIGHT = value
    with pytest.raises(ValueError, match="L1_WEIGHT"):
        build_dummy_mtl_lite_model(monkeypatch, cfg)
