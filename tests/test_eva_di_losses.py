"""losses: severity table, weighted L2, masked CE exclusion rules, lambda schedule."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.eva_di.contracts import EvaDiSchemaError
from src.eva_di.losses import (
    build_severity_weight_table,
    ce_masked,
    lambda_scale,
    severity_weighted_l2,
    total_loss_terms,
)


def test_severity_table_inverse_frequency_and_unknown():
    scores = [10.0, 11.0, 20.0, float("nan")]  # mild x2, severe x1, unknown x1
    table, weights = build_severity_weight_table(scores)
    assert table.dtype == np.float32 and table.shape == (4,)
    assert np.isfinite(table).all()
    assert table[0] == table[1]                      # same group -> same weight
    assert table[0] < table[2]                       # rare group weighs more
    assert table[3] == pytest.approx(1.0)            # unknown falls to neutral 1.0
    assert "unknown" not in weights                  # unknown excluded from counts
    assert np.mean(list(weights.values())) == pytest.approx(1.0)
    with pytest.raises(EvaDiSchemaError):
        build_severity_weight_table([float("nan"), float("nan")])  # only unknowns


def test_severity_weighted_l2_matches_manual_and_keeps_graph():
    pred = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.tensor([0.0, 0.0])
    weight = torch.tensor([2.0, 0.5])
    loss = severity_weighted_l2(pred, target, weight)
    assert loss.item() == pytest.approx((2.0 * 1.0 + 0.5 * 4.0) / 2)
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    with pytest.raises(EvaDiSchemaError):
        severity_weighted_l2(torch.zeros(3), torch.zeros(3), torch.zeros(2))


def test_ce_2d_drops_unmapped_samples_only():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [1.0, 1.0]])
    subject = torch.tensor([0, -1, 1])
    loss = ce_masked(logits, subject, label_smoothing=0.0)
    ref = F.cross_entropy(torch.stack([logits[0], logits[2]]), torch.tensor([0, 1]),
                          label_smoothing=0.0)
    assert torch.allclose(loss, ref)


def test_ce_all_unmapped_is_differentiable_zero():
    logits = torch.randn(3, 5, requires_grad=True)
    loss = ce_masked(logits, torch.full((3,), -1, dtype=torch.long))
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_ce_3d_masks_frames_and_whole_unmapped_samples():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 3)
    subject = torch.tensor([1, -1])
    frame_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    loss = ce_masked(logits, subject, frame_mask, label_smoothing=0.0)
    ref = F.cross_entropy(logits[0, :2], torch.tensor([1, 1]), label_smoothing=0.0)
    assert torch.allclose(loss, ref)
    # padded/invalid frames contribute nothing: changing them leaves loss identical
    other = logits.clone()
    other[:, 2:] += 50.0
    assert torch.allclose(ce_masked(other, subject, frame_mask, 0.0), loss)
    with pytest.raises(EvaDiSchemaError):
        ce_masked(logits, subject)  # 3-d logits without mask


def test_ce_3d_all_masked_is_differentiable_zero():
    logits = torch.randn(2, 3, 4, requires_grad=True)
    subject = torch.tensor([0, 1])
    loss = ce_masked(logits, subject, torch.zeros(2, 3, dtype=torch.bool))
    assert loss.item() == 0.0 and loss.requires_grad


def test_lambda_scale_schedule():
    assert lambda_scale(1, 4) == pytest.approx(0.25)
    assert lambda_scale(4, 4) == 1.0 and lambda_scale(9, 4) == 1.0
    assert lambda_scale(3, 0) == 1.0
    with pytest.raises(EvaDiSchemaError):
        lambda_scale(0, 4)


def test_total_loss_terms_arithmetic_and_finiteness_guard():
    reg = torch.tensor(2.0, requires_grad=True)
    f, v = torch.tensor(1.0), torch.tensor(4.0)
    terms = total_loss_terms(reg_loss=reg, frame_ce=f, video_ce=v, epoch=2,
                             w1=1.0, w3=0.5, warmup_epochs=4)
    assert terms["total"].item() == pytest.approx(2.0 + 0.5 * 1.0 + 0.25 * 4.0)
    assert terms["warmup_scale"].item() == pytest.approx(0.5)
    terms["total"].backward()
    assert reg.grad is not None
    with pytest.raises(EvaDiSchemaError, match="non-finite"):
        total_loss_terms(reg_loss=torch.tensor(float("nan")), frame_ce=f, video_ce=v,
                         epoch=1, w1=1.0, w3=1.0, warmup_epochs=0)


def test_weighted_l2_broadcast_trap_is_neutralised():
    # R2 loss#9: [B,1] pred/target with [B] weight used to broadcast to [B,B]
    # and silently 3x the loss.  The guard now flattens FIRST: the [B,1]
    # spelling computes the identical elementwise result, while genuinely
    # mis-aligned lengths (post-flatten) are refused.
    flat = severity_weighted_l2(torch.zeros(3), torch.ones(3), torch.full((3,), 2.0))
    column = severity_weighted_l2(torch.zeros((3, 1)), torch.ones((3, 1)),
                                  torch.full((3,), 2.0))
    assert float(column) == pytest.approx(float(flat))
    with pytest.raises(EvaDiSchemaError, match="shape mismatch"):
        severity_weighted_l2(torch.zeros((3, 2)), torch.ones(3), torch.ones(3))


def test_ce_rejects_out_of_range_labels():
    logits = torch.zeros((2, 4), requires_grad=True)
    with pytest.raises(EvaDiSchemaError, match="outside"):
        ce_masked(logits, torch.tensor([0, 4]))
