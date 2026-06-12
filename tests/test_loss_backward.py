import torch
import torch.nn.functional as F

from tests.test_model_forward import DummyBackbone, _minimal_config


def test_regression_head_receives_nonzero_gradients(monkeypatch):
    import src.models.end_to_end as end_to_end

    cfg = _minimal_config()

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        output_dim = cfg.BACKBONE_OUT_DIMS[model_name]
        return DummyBackbone(output_dim=output_dim)

    monkeypatch.setattr(end_to_end, "build_feature_backbone", build_dummy_backbone)
    model = end_to_end.EndToEndDepressionModel(cfg)
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

    bdi_pred, cls_pred, con_embeds = model(video, mask, need_all_heads=False)
    target = bdi_pred.detach() + 0.5
    loss = F.mse_loss(bdi_pred, target)
    loss.backward()

    reg_grads = [
        param.grad
        for param in model.reg_task_head.parameters()
        if param.grad is not None
    ]

    assert cls_pred is None
    assert con_embeds is None
    assert torch.isfinite(loss)
    assert reg_grads
    assert all(torch.isfinite(grad).all() for grad in reg_grads)
    assert sum(grad.detach().abs().sum().item() for grad in reg_grads) > 0.0
