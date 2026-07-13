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


def stage_c_mtl_lite_config(
    variant="split",
    reconstruction=False,
    cross_correlation=False,
    calibration_only=False,
    reconstruction_weight=0.0,
    cross_correlation_weight=0.0,
):
    cfg = minimal_mtl_lite_config()
    cfg.MODEL = {
        "TASK_NUISANCE": {
            "ENABLE": True,
            "VARIANT": variant,
            "DEP_DIM": 4,
            "NUISANCE_DIM": 4,
            "RECONSTRUCTION_ENABLE": reconstruction,
            "CROSS_CORRELATION_ENABLE": cross_correlation,
            "AUXILIARY_CALIBRATION_ONLY": calibration_only,
            "CALIBRATION_STEPS": 100,
        }
    }
    cfg.LOSSES.RECONSTRUCTION_WEIGHT = reconstruction_weight
    cfg.LOSSES.CROSS_CORRELATION_WEIGHT = cross_correlation_weight
    return cfg


def build_dummy_mtl_lite_model(monkeypatch, cfg):
    import src.models.mtl_lite as mtl_lite

    def build_dummy_backbone(
        model_name, weight_path=None, timm_pretrained=False, img_size=112
    ):
        return DummyBackbone(output_dim=cfg.BACKBONE_OUT_DIMS[model_name])

    monkeypatch.setattr(mtl_lite, "build_feature_backbone", build_dummy_backbone)
    return mtl_lite.MTLLiteDepressionModel(cfg)


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
    assert outputs.h0_features is None
    assert outputs.z_dep_features is None
    assert outputs.z_nuisance_features is None
    assert outputs.reconstructed_h0 is None


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


def test_stage_c_bottleneck_routes_prediction_through_z_dep(monkeypatch):
    cfg = stage_c_mtl_lite_config(variant="bottleneck")
    model = build_dummy_mtl_lite_model(monkeypatch, cfg)
    model.eval()
    video = torch.randn(2, 4, 3, 16, 16)
    mask = torch.ones(2, 4, dtype=torch.bool)

    with torch.no_grad():
        outputs = model(
            video,
            mask,
            return_features=True,
            return_layer_features=True,
        )

    assert outputs.h0_features.shape == (2, 8)
    assert outputs.z_dep_features.shape == (2, 4)
    assert outputs.shared_features.shape == (2, 4)
    assert torch.equal(outputs.shared_features, outputs.z_dep_features)
    assert outputs.z_nuisance_features is None
    assert outputs.reconstructed_h0 is None
    assert outputs.layer_features["layer_temporal"].shape == (2, 8)
    assert outputs.layer_features["layer_shared"].shape == (2, 4)
    assert model.reg_task_head[0].in_features == 4


def test_stage_c_split_returns_all_candidate_representations(monkeypatch):
    cfg = stage_c_mtl_lite_config(
        reconstruction=True,
        cross_correlation=True,
        reconstruction_weight=0.01,
        cross_correlation_weight=0.01,
    )
    model = build_dummy_mtl_lite_model(monkeypatch, cfg)
    model.eval()
    video = torch.randn(3, 4, 3, 16, 16)
    mask = torch.ones(3, 4, dtype=torch.bool)

    with torch.no_grad():
        outputs = model(video, mask, return_features=True)

    assert outputs.h0_features.shape == (3, 8)
    assert outputs.z_dep_features.shape == (3, 4)
    assert outputs.z_nuisance_features.shape == (3, 4)
    assert outputs.reconstructed_h0.shape == (3, 8)
    assert torch.equal(outputs.shared_features, outputs.z_dep_features)


def test_explicit_stage_c_disable_preserves_state_dict_and_outputs(monkeypatch):
    cfg_default = minimal_mtl_lite_config()
    cfg_disabled = minimal_mtl_lite_config()
    cfg_disabled.MODEL = {
        "TASK_NUISANCE": {
            "ENABLE": False,
            "VARIANT": "split",
            "DEP_DIM": 4,
            "NUISANCE_DIM": 4,
            "RECONSTRUCTION_ENABLE": False,
            "CROSS_CORRELATION_ENABLE": False,
            "AUXILIARY_CALIBRATION_ONLY": False,
            "CALIBRATION_STEPS": 100,
        }
    }
    cfg_disabled.LOSSES.RECONSTRUCTION_WEIGHT = 0.0
    cfg_disabled.LOSSES.CROSS_CORRELATION_WEIGHT = 0.0

    torch.manual_seed(17)
    default_model = build_dummy_mtl_lite_model(monkeypatch, cfg_default)
    torch.manual_seed(17)
    disabled_model = build_dummy_mtl_lite_model(monkeypatch, cfg_disabled)

    assert default_model.state_dict().keys() == disabled_model.state_dict().keys()
    disabled_model.load_state_dict(default_model.state_dict(), strict=True)
    default_model.eval()
    disabled_model.eval()
    video = torch.randn(2, 4, 3, 16, 16)
    mask = torch.ones(2, 4, dtype=torch.bool)
    with torch.no_grad():
        default_output = default_model(video, mask, return_features=True)
        disabled_output = disabled_model(video, mask, return_features=True)

    assert torch.equal(default_output.bdi_pred, disabled_output.bdi_pred)
    assert torch.equal(
        default_output.shared_features, disabled_output.shared_features
    )
