import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_NUISANCE_CONFIG_KEYS = {
    "ENABLE",
    "VARIANT",
    "DEP_DIM",
    "NUISANCE_DIM",
    "RECONSTRUCTION_ENABLE",
    "CROSS_CORRELATION_ENABLE",
    "AUXILIARY_CALIBRATION_ONLY",
    "CALIBRATION_STEPS",
}


@dataclass(frozen=True)
class TaskNuisanceConfig:
    enabled: bool
    variant: str
    dep_dim: int
    nuisance_dim: int
    reconstruction_enabled: bool
    cross_correlation_enabled: bool
    calibration_only: bool
    calibration_steps: int
    reconstruction_weight: float
    cross_correlation_weight: float


@dataclass
class TaskNuisanceOutput:
    z_dep: torch.Tensor
    z_nuisance: Optional[torch.Tensor] = None
    reconstructed_h0: Optional[torch.Tensor] = None


def _config_section(configs, section_name, nested_name=None):
    section = getattr(configs, section_name, None)
    if section is None or nested_name is None:
        return section
    return getattr(section, nested_name, None)


def _section_value(section, key, default):
    if section is None:
        return default
    return getattr(section, key, default)


def _require_bool(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean, got: {value!r}")
    return value


def _require_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got: {value!r}")
    return int(value)


def _require_non_negative_float(name, value):
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(
            f"{name} must be a finite non-negative number, got: {value!r}"
        )
    return float(value)


def resolve_task_nuisance_config(configs, hidden_dim):
    """Resolve and strictly validate the Stage C model/loss contract."""
    hidden_dim = _require_positive_int("PROCESS_TEMPORAL.HIDDEN_DIM", hidden_dim)
    section = _config_section(configs, "MODEL", "TASK_NUISANCE")
    if section is not None:
        if not hasattr(section, "keys"):
            raise ValueError("MODEL.TASK_NUISANCE must be a mapping")
        unknown = set(section.keys()) - TASK_NUISANCE_CONFIG_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown MODEL.TASK_NUISANCE field(s): {names}")

    losses = _config_section(configs, "LOSSES")
    enabled = _require_bool(
        "MODEL.TASK_NUISANCE.ENABLE",
        _section_value(section, "ENABLE", False),
    )
    variant = _section_value(section, "VARIANT", "split")
    if not isinstance(variant, str) or variant not in {"bottleneck", "split"}:
        raise ValueError(
            "MODEL.TASK_NUISANCE.VARIANT must be 'bottleneck' or 'split', "
            f"got: {variant!r}"
        )

    default_dep_dim = max(1, hidden_dim // 2)
    dep_dim = _require_positive_int(
        "MODEL.TASK_NUISANCE.DEP_DIM",
        _section_value(section, "DEP_DIM", default_dep_dim),
    )
    nuisance_dim = _require_positive_int(
        "MODEL.TASK_NUISANCE.NUISANCE_DIM",
        _section_value(
            section,
            "NUISANCE_DIM",
            max(1, hidden_dim - default_dep_dim),
        ),
    )
    reconstruction_enabled = _require_bool(
        "MODEL.TASK_NUISANCE.RECONSTRUCTION_ENABLE",
        _section_value(section, "RECONSTRUCTION_ENABLE", False),
    )
    cross_correlation_enabled = _require_bool(
        "MODEL.TASK_NUISANCE.CROSS_CORRELATION_ENABLE",
        _section_value(section, "CROSS_CORRELATION_ENABLE", False),
    )
    calibration_only = _require_bool(
        "MODEL.TASK_NUISANCE.AUXILIARY_CALIBRATION_ONLY",
        _section_value(section, "AUXILIARY_CALIBRATION_ONLY", False),
    )
    calibration_steps = _require_positive_int(
        "MODEL.TASK_NUISANCE.CALIBRATION_STEPS",
        _section_value(section, "CALIBRATION_STEPS", 100),
    )
    reconstruction_weight = _require_non_negative_float(
        "LOSSES.RECONSTRUCTION_WEIGHT",
        _section_value(losses, "RECONSTRUCTION_WEIGHT", 0.0),
    )
    cross_correlation_weight = _require_non_negative_float(
        "LOSSES.CROSS_CORRELATION_WEIGHT",
        _section_value(losses, "CROSS_CORRELATION_WEIGHT", 0.0),
    )

    if not enabled:
        if reconstruction_enabled or cross_correlation_enabled or calibration_only:
            raise ValueError(
                "MODEL.TASK_NUISANCE auxiliary switches require ENABLE=True"
            )
        if reconstruction_weight != 0.0 or cross_correlation_weight != 0.0:
            raise ValueError(
                "Stage C auxiliary loss weights must be 0 when "
                "MODEL.TASK_NUISANCE.ENABLE=False"
            )
    elif variant == "bottleneck":
        if reconstruction_enabled or cross_correlation_enabled or calibration_only:
            raise ValueError(
                "The bottleneck variant cannot enable split-only auxiliary losses"
            )
        if reconstruction_weight != 0.0 or cross_correlation_weight != 0.0:
            raise ValueError("The bottleneck variant requires auxiliary weights of 0")
    else:
        if dep_dim + nuisance_dim != hidden_dim:
            raise ValueError(
                "MODEL.TASK_NUISANCE.DEP_DIM + NUISANCE_DIM must equal "
                f"PROCESS_TEMPORAL.HIDDEN_DIM ({hidden_dim}), got "
                f"{dep_dim} + {nuisance_dim}"
            )
        if not reconstruction_enabled:
            raise ValueError(
                "The split variant requires RECONSTRUCTION_ENABLE=True; use the "
                "bottleneck variant for a prediction-only control"
            )
        if calibration_only:
            if not reconstruction_enabled or not cross_correlation_enabled:
                raise ValueError(
                    "AUXILIARY_CALIBRATION_ONLY requires reconstruction and "
                    "cross-correlation to be enabled"
                )
            if reconstruction_weight != 0.0 or cross_correlation_weight != 0.0:
                raise ValueError(
                    "AUXILIARY_CALIBRATION_ONLY requires both auxiliary weights to be 0"
                )
        else:
            _validate_loss_switch(
                "RECONSTRUCTION",
                reconstruction_enabled,
                reconstruction_weight,
            )
            _validate_loss_switch(
                "CROSS_CORRELATION",
                cross_correlation_enabled,
                cross_correlation_weight,
            )

    return TaskNuisanceConfig(
        enabled=enabled,
        variant=variant,
        dep_dim=dep_dim,
        nuisance_dim=nuisance_dim,
        reconstruction_enabled=reconstruction_enabled,
        cross_correlation_enabled=cross_correlation_enabled,
        calibration_only=calibration_only,
        calibration_steps=calibration_steps,
        reconstruction_weight=reconstruction_weight,
        cross_correlation_weight=cross_correlation_weight,
    )


def _validate_loss_switch(name, enabled, weight):
    if enabled and weight <= 0.0:
        raise ValueError(f"{name}_ENABLE=True requires a positive loss weight")
    if not enabled and weight != 0.0:
        raise ValueError(f"{name}_ENABLE=False requires a loss weight of 0")


class TaskNuisanceBlock(nn.Module):
    """Minimal single-level Stage C task/nuisance projection block."""

    def __init__(self, h0_dim, dep_dim, nuisance_dim, variant):
        super().__init__()
        if variant not in {"bottleneck", "split"}:
            raise ValueError(f"Unsupported TaskNuisanceBlock variant: {variant!r}")
        self.variant = variant
        self.dep_encoder = self._build_encoder(h0_dim, dep_dim)
        self.nuisance_encoder = None
        self.reconstructor = None
        if variant == "split":
            self.nuisance_encoder = self._build_encoder(h0_dim, nuisance_dim)
            self.reconstructor = nn.Linear(dep_dim + nuisance_dim, h0_dim)

    @staticmethod
    def _build_encoder(input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, h0_features, reconstruct=False):
        z_dep = self.dep_encoder(h0_features)
        if self.variant == "bottleneck":
            return TaskNuisanceOutput(z_dep=z_dep)

        z_nuisance = self.nuisance_encoder(h0_features)
        reconstructed_h0 = None
        if reconstruct:
            reconstructed_h0 = self.reconstructor(
                torch.cat([z_dep, z_nuisance], dim=-1)
            )
        return TaskNuisanceOutput(
            z_dep=z_dep,
            z_nuisance=z_nuisance,
            reconstructed_h0=reconstructed_h0,
        )


def reconstruction_mse(reconstructed_h0, h0_features):
    """Float32 reconstruction loss with a stop-gradient H0 target."""
    return F.mse_loss(reconstructed_h0.float(), h0_features.detach().float())


def cross_correlation_penalty(z_dep, z_nuisance, eps=1e-6):
    """Mean squared normalized cross-correlation, computed in float32."""
    if z_dep.ndim != 2 or z_nuisance.ndim != 2:
        raise ValueError("Cross-correlation inputs must both have shape (batch, dim)")
    if z_dep.size(0) != z_nuisance.size(0):
        raise ValueError("Cross-correlation inputs must have the same batch size")
    if z_dep.size(0) < 2:
        return (z_dep.float().sum() + z_nuisance.float().sum()) * 0.0

    dep = z_dep.float()
    nuisance = z_nuisance.float()
    dep = dep - dep.mean(dim=0, keepdim=True)
    nuisance = nuisance - nuisance.mean(dim=0, keepdim=True)
    dep = dep / dep.square().mean(dim=0, keepdim=True).add(eps).sqrt()
    nuisance = nuisance / nuisance.square().mean(dim=0, keepdim=True).add(eps).sqrt()
    correlation = dep.transpose(0, 1).matmul(nuisance) / float(dep.size(0))
    return correlation.square().mean()
