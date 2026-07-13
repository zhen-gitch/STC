import pytest
import torch
from omegaconf import OmegaConf

from src.models.task_nuisance import (
    TaskNuisanceBlock,
    cross_correlation_penalty,
    reconstruction_mse,
    resolve_task_nuisance_config,
)


def _config(
    *,
    enabled=True,
    variant="split",
    dep_dim=4,
    nuisance_dim=4,
    reconstruction=False,
    cross_correlation=False,
    calibration_only=False,
    reconstruction_weight=0.0,
    cross_correlation_weight=0.0,
):
    return OmegaConf.create(
        {
            "MODEL": {
                "TASK_NUISANCE": {
                    "ENABLE": enabled,
                    "VARIANT": variant,
                    "DEP_DIM": dep_dim,
                    "NUISANCE_DIM": nuisance_dim,
                    "RECONSTRUCTION_ENABLE": reconstruction,
                    "CROSS_CORRELATION_ENABLE": cross_correlation,
                    "AUXILIARY_CALIBRATION_ONLY": calibration_only,
                    "CALIBRATION_STEPS": 100,
                }
            },
            "LOSSES": {
                "RECONSTRUCTION_WEIGHT": reconstruction_weight,
                "CROSS_CORRELATION_WEIGHT": cross_correlation_weight,
            },
        }
    )


def test_task_nuisance_defaults_to_disabled_when_section_is_missing():
    policy = resolve_task_nuisance_config(OmegaConf.create({}), hidden_dim=8)

    assert policy.enabled is False
    assert policy.dep_dim == 4
    assert policy.nuisance_dim == 4
    assert policy.reconstruction_weight == 0.0
    assert policy.cross_correlation_weight == 0.0
    assert policy.calibration_steps == 100


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (_config(variant="invalid"), "VARIANT"),
        (_config(dep_dim=5, nuisance_dim=4), "must equal"),
        (_config(reconstruction=False), "split variant requires"),
        (
            _config(reconstruction=True, reconstruction_weight=0.0),
            "RECONSTRUCTION_ENABLE",
        ),
        (
            _config(
                reconstruction=True,
                reconstruction_weight=0.01,
                cross_correlation=False,
                cross_correlation_weight=0.01,
            ),
            "CROSS_CORRELATION_ENABLE",
        ),
        (
            _config(
                calibration_only=True,
                reconstruction=True,
                cross_correlation=False,
            ),
            "AUXILIARY_CALIBRATION_ONLY",
        ),
        (
            _config(
                calibration_only=True,
                reconstruction=True,
                cross_correlation=True,
                reconstruction_weight=0.01,
            ),
            "requires both auxiliary weights to be 0",
        ),
        (
            _config(
                variant="bottleneck",
                reconstruction=True,
                reconstruction_weight=0.01,
            ),
            "bottleneck variant",
        ),
    ],
)
def test_invalid_task_nuisance_contract_is_rejected(cfg, message):
    with pytest.raises(ValueError, match=message):
        resolve_task_nuisance_config(cfg, hidden_dim=8)


def test_unknown_task_nuisance_field_is_rejected():
    cfg = _config()
    cfg.MODEL.TASK_NUISANCE.UNKNOWN_FIELD = True

    with pytest.raises(ValueError, match="Unknown MODEL.TASK_NUISANCE"):
        resolve_task_nuisance_config(cfg, hidden_dim=8)


def test_bottleneck_block_only_emits_prediction_features():
    block = TaskNuisanceBlock(8, 4, 4, variant="bottleneck")
    output = block(torch.randn(3, 8), reconstruct=True)

    assert output.z_dep.shape == (3, 4)
    assert output.z_nuisance is None
    assert output.reconstructed_h0 is None
    assert block.nuisance_encoder is None
    assert block.reconstructor is None


def test_split_block_emits_equal_width_outlets_and_reconstruction():
    block = TaskNuisanceBlock(8, 4, 4, variant="split")
    output = block(torch.randn(3, 8), reconstruct=True)

    assert output.z_dep.shape == (3, 4)
    assert output.z_nuisance.shape == (3, 4)
    assert output.reconstructed_h0.shape == (3, 8)
    assert block.nuisance_encoder is not None
    assert block.reconstructor is not None


def test_reconstruction_target_is_detached_and_loss_is_float32():
    h0 = torch.randn(3, 8, requires_grad=True)
    reconstruction = torch.randn(3, 8, requires_grad=True)

    loss = reconstruction_mse(reconstruction, h0)
    loss.backward()

    assert loss.dtype == torch.float32
    assert h0.grad is None
    assert reconstruction.grad is not None
    assert reconstruction.grad.abs().sum() > 0


def test_cross_correlation_is_float32_finite_and_differentiable():
    z_dep = torch.randn(5, 4, dtype=torch.float16, requires_grad=True)
    z_nuisance = torch.randn(5, 3, dtype=torch.float16, requires_grad=True)

    loss = cross_correlation_penalty(z_dep, z_nuisance)
    loss.backward()

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert z_dep.grad is not None and torch.isfinite(z_dep.grad).all()
    assert z_nuisance.grad is not None and torch.isfinite(z_nuisance.grad).all()


def test_cross_correlation_batch_size_one_returns_differentiable_zero():
    z_dep = torch.randn(1, 4, requires_grad=True)
    z_nuisance = torch.randn(1, 4, requires_grad=True)

    loss = cross_correlation_penalty(z_dep, z_nuisance)
    loss.backward()

    assert loss.item() == 0.0
    assert z_dep.grad is not None
    assert z_nuisance.grad is not None
    assert torch.count_nonzero(z_dep.grad) == 0
    assert torch.count_nonzero(z_nuisance.grad) == 0
