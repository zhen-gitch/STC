import torch

from tests.test_mtl_lite_forward import minimal_mtl_lite_config, DummyBackbone


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
