import torch

from tests.test_mtl_lite_forward import (
    DummyBackbone,
    build_dummy_mtl_lite_model,
    minimal_mtl_lite_config,
    stage_c_mtl_lite_config,
)


def test_mtl_lite_regression_head_receives_nonzero_gradients(monkeypatch):
    import src.models.mtl_lite as mtl_lite

    cfg = minimal_mtl_lite_config()

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        return DummyBackbone(output_dim=cfg.BACKBONE_OUT_DIMS[model_name])

    monkeypatch.setattr(mtl_lite, "build_feature_backbone", build_dummy_backbone)
    model = mtl_lite.MTLLiteDepressionModel(cfg)
    model.train()
    model.zero_grad(set_to_none=True)

    video = torch.randn(2, 4, 3, 16, 16)
    mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    labels = {
        "bdi_score": torch.tensor([12.0, 36.0]),
        "class_label": torch.tensor([1, 4]),
    }

    outputs = model(video, mask)
    losses = model.compute_losses(outputs, labels)
    losses.total.backward()

    reg_grads = [
        param.grad
        for param in model.reg_task_head.parameters()
        if param.grad is not None
    ]

    assert torch.isfinite(losses.total)
    assert torch.isfinite(losses.regression)
    assert losses.ordinal is not None and torch.isfinite(losses.ordinal)
    assert losses.ccc is not None and torch.isfinite(losses.ccc)
    assert reg_grads
    assert all(torch.isfinite(grad).all() for grad in reg_grads)
    assert sum(grad.detach().abs().sum().item() for grad in reg_grads) > 0.0


def _stage_c_batch(batch_size=4):
    video = torch.randn(batch_size, 4, 3, 16, 16)
    mask = torch.ones(batch_size, 4, dtype=torch.bool)
    labels = {
        "bdi_score": torch.linspace(5.0, 45.0, batch_size),
        "class_label": torch.arange(batch_size) % 5,
    }
    return video, mask, labels


def test_stage_c_full_loss_matches_weighted_formula_and_backpropagates(monkeypatch):
    cfg = stage_c_mtl_lite_config(
        reconstruction=True,
        cross_correlation=True,
        reconstruction_weight=0.01,
        cross_correlation_weight=0.02,
    )
    model = build_dummy_mtl_lite_model(monkeypatch, cfg)
    model.train()
    video, mask, labels = _stage_c_batch()

    outputs = model(video, mask)
    losses = model.compute_losses(outputs, labels, stage="train")
    expected = (
        losses.regression
        + model.ccc_loss_weight * losses.ccc
        + model.ordinal_weight * losses.ordinal
        + 0.01 * losses.reconstruction
        + 0.02 * losses.cross_correlation
    )
    assert torch.allclose(losses.total, expected)
    assert losses.reconstruction.dtype == torch.float32
    assert losses.cross_correlation.dtype == torch.float32

    losses.total.backward()
    block = model.task_nuisance_block
    parameter_groups = (
        list(block.dep_encoder.parameters()),
        list(block.nuisance_encoder.parameters()),
        list(block.reconstructor.parameters()),
    )
    for parameters in parameter_groups:
        gradients = [param.grad for param in parameters if param.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum().item() for gradient in gradients) > 0.0


def test_stage_c_reconstruction_only_does_not_emit_cross_correlation(monkeypatch):
    cfg = stage_c_mtl_lite_config(
        reconstruction=True,
        reconstruction_weight=0.01,
    )
    model = build_dummy_mtl_lite_model(monkeypatch, cfg)
    video, mask, labels = _stage_c_batch()

    losses = model.compute_losses(model(video, mask), labels, stage="train")

    assert losses.reconstruction is not None
    assert losses.cross_correlation is None


def test_calibration_only_reports_raw_train_losses_without_optimizing_them(
    monkeypatch,
):
    cfg = stage_c_mtl_lite_config(
        reconstruction=True,
        cross_correlation=True,
        calibration_only=True,
    )
    model = build_dummy_mtl_lite_model(monkeypatch, cfg)
    video, mask, labels = _stage_c_batch()
    outputs = model(video, mask)

    train_losses = model.compute_losses(outputs, labels, stage="train")
    primary = (
        train_losses.regression
        + model.ccc_loss_weight * train_losses.ccc
        + model.ordinal_weight * train_losses.ordinal
    )
    val_losses = model.compute_losses(outputs, labels, stage="val")

    assert train_losses.reconstruction is not None
    assert train_losses.cross_correlation is not None
    assert torch.allclose(train_losses.total, primary)
    assert val_losses.reconstruction is None
    assert val_losses.cross_correlation is None

    train_losses.total.backward()
    for module in (
        model.task_nuisance_block.nuisance_encoder,
        model.task_nuisance_block.reconstructor,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients)
