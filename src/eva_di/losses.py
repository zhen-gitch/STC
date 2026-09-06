"""Losses for the DI mechanism (severity-weighted regression + masked CE).

Semantics pinned by docs/DUAL_LEVEL_IDENTITY_ADVERSARIAL_PLAN.md sections 2-3:

* T2 severity table is built from the *train* split only and constructed in
  float32 (never under autocast) to dodge bf16 quantization of weights;
* T1 frame CE only over ``frame_mask=True`` frames;
* ``subject == -1`` excludes the *sample* (never the whole batch);
* GRL sign handling lives in :class:`src.models.gradient_reversal.GradientReversalLayer`
  (reused, not re-implemented); this module only multiplies by ``w_k(e)``;
* empty valid selections return a differentiable zero so a batch never NaNs.

SINGLE warmup (user decision 2026-09-05, resolves audit R2-P1-10/scale²): the
ramp w_k(e) multiplies the CE weight HERE only (plan L61 reads as the
loss-weight schedule); the trainer keeps the GRL lambda CONSTANT at ``lam``,
so identity strength is exactly ``w_k(e) * lam * CE`` -- linear in the ramp,
not ``lam*w²``.  Note also that only ``lam * weight`` is identifiable on the
shared-feature path (GRL passes -lambda grad of the scaled CE; AdamW is
scale-invariant per parameter), so sweeping weight and lam remains a 1-D sweep.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.eva_di.contracts import EvaDiSchemaError
from src.eva_di.metrics import severity_group


def build_severity_weight_table(scores, *, epsilon: float = 1e-6) -> tuple[np.ndarray, dict[str, float]]:
    """Inverse group frequency weights, fixed float32 (train scores only)."""
    groups = [severity_group(float(s)) for s in scores]
    present = [g for g in groups if g != "unknown"]
    if not present:
        raise EvaDiSchemaError("severity table needs at least one known train score")
    counts: dict[str, int] = {}
    for group in present:
        counts[group] = counts.get(group, 0) + 1
    inv = {g: 1.0 / (c + epsilon) for g, c in counts.items()}
    norm = float(np.mean(list(inv.values())))
    weights = {g: v / norm for g, v in inv.items()}
    table = np.asarray(
        [weights.get(severity_group(float(s)), 1.0) for s in scores], dtype=np.float32
    )
    return table, weights


def severity_weighted_l2(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-sample weighted MSE in float32 math regardless of autocast state.

    All three tensors are flattened BEFORE the shape guard: comparing
    ``[B,1]`` pred against ``[B]`` weight only on ``shape[0]`` allowed a silent
    ``[B,B]`` broadcast (audit R2-P2 loss #9).
    """
    pred32 = pred.float().reshape(-1)  # float() is a no-op for float32 and keeps the graph
    target32 = target.float().reshape(-1)
    weight32 = weight.float().reshape(-1)
    if pred32.shape != target32.shape or weight32.shape != pred32.shape:
        raise EvaDiSchemaError(
            f"severity l2 shape mismatch: pred{tuple(pred.shape)} "
            f"target{tuple(target.shape)} weight{tuple(weight.shape)}"
        )
    return (weight32 * (pred32 - target32).square()).mean()


def mse_norm(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain float32 MSE on the normalized target scale (EVA-DI-REGNORM-v1).

    Protocol change 2026-09-06 (user): L_reg drops the severity-inverse-
    frequency weighting.  The target stays the train-only z-scored scale
    (dataset.TargetStats), so the regression term is O(1) end to end.
    Same flatten-before-guard discipline as severity_weighted_l2
    (audit R2-P2 loss #9).
    """
    pred32 = pred.float().reshape(-1)
    target32 = target.float().reshape(-1)
    if pred32.shape != target32.shape:
        raise EvaDiSchemaError(
            f"mse_norm shape mismatch: pred{tuple(pred.shape)} "
            f"target{tuple(target.shape)}")
    return (pred32 - target32).square().mean()


def ce_masked(
    logits: torch.Tensor,
    subject: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
    label_smoothing: float = 0.1,
) -> torch.Tensor:
    """CE over valid frames (T1: logits [B,T,S]) or videos (T3: [B,S]).

    ``subject == -1`` rows are dropped at sample level; with nothing left the
    loss is a differentiable zero built from ``logits`` so the graph survives.
    """
    if logits.dim() == 2:
        subject_rows = subject
        selected_logits = logits
        if bool((subject < 0).any()):
            keep = subject >= 0
            selected_logits = logits[keep]
            subject_rows = subject[keep]
        if selected_logits.shape[0] == 0:
            return logits.sum() * 0.0
        _check_label_range(subject_rows, logits.shape[-1])
        return F.cross_entropy(selected_logits, subject_rows, label_smoothing=label_smoothing)

    if logits.dim() != 3 or frame_mask is None:
        raise EvaDiSchemaError("frame CE expects logits [B,T,S] and a frame_mask")
    keep = subject >= 0
    if not bool(keep.any()):
        return logits.sum() * 0.0
    logits = logits[keep]
    mask = frame_mask[keep] & (subject[keep] >= 0).unsqueeze(-1)
    labels = subject[keep].unsqueeze(-1).expand(-1, logits.shape[1])  # [B',T]
    if logits.device != mask.device:
        raise EvaDiSchemaError("logits/mask device mismatch")
    flat_logits = logits[mask]
    flat_labels = labels[mask]
    if flat_logits.numel() == 0:
        return logits.sum() * 0.0
    _check_label_range(flat_labels, logits.shape[-1])
    return F.cross_entropy(flat_logits, flat_labels, label_smoothing=label_smoothing)


def _check_label_range(labels: torch.Tensor, n_classes: int) -> None:
    """Out-of-range CE targets are an async CUDA memory error otherwise
    (subject_table guarantees 0..S-1; this pins the contract, audit R2 loss #14)."""
    if int(labels.numel()) and (int(labels.min()) < 0 or int(labels.max()) >= n_classes):
        raise EvaDiSchemaError(
            f"CE label outside [0, {n_classes}): min {int(labels.min())} "
            f"max {int(labels.max())}")


def lambda_scale(epoch: int, warmup_epochs: int) -> float:
    """w_k(e) = min(1, e / warmup) with e counted from 1; no warmup -> 1.0."""
    if epoch < 1:
        raise EvaDiSchemaError(f"epoch must be >= 1, got {epoch}")
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch) / float(warmup_epochs))


def total_loss_terms(
    *,
    reg_loss: torch.Tensor,
    frame_ce: torch.Tensor,
    video_ce: torch.Tensor,
    epoch: int,
    w1: float,
    w3: float,
    warmup_epochs: int,
) -> dict[str, torch.Tensor]:
    scale = lambda_scale(epoch, warmup_epochs)
    total = reg_loss + (w1 * scale) * frame_ce + (w3 * scale) * video_ce
    if not bool(torch.isfinite(total)):
        raise EvaDiSchemaError(f"non-finite total loss at epoch {epoch}")
    return {
        "total": total,
        "reg": reg_loss,
        "frame_ce": frame_ce,
        "video_ce": video_ce,
        "warmup_scale": torch.tensor(lambda_scale(epoch, warmup_epochs)),
    }
