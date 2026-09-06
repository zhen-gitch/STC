"""model: exit shapes, masked mean, train-gated heads, GRL sign, ablation zeroing."""

import dataclasses

import numpy as np
import pytest
import torch
import torch.autograd as autograd
import torch.nn as nn

from src.eva_di.batching import collate
from src.eva_di.config import ModelCfg
from src.eva_di.contracts import EVA_FEATURE_DIM, PROJ_DIM, EvaDiSchemaError
from src.eva_di.dataset import Recording
from src.eva_di.model import DualStreamDI
from src.models.gradient_reversal import GradientReversalLayer


def _cfg(**over):
    base = dict(use_gap=False, proj_dim=PROJ_DIM, gru_hidden=PROJ_DIM, gru_layers=1,
                behavior_mlp_dim=64, behavior_dropout=0.1, proj_dropout=0.1)
    base.update(over)
    return ModelCfg(**base)


def _rec(n):
    rng = np.random.default_rng(n)
    return Recording(
        sample_id=f"v{n}", split="train", subject=0, bdi_score=20.0, bdi_norm=0.5,
        cls=rng.normal(0, 1, (n, EVA_FEATURE_DIM)).astype(np.float32),
        gap=rng.normal(0, 1, (n, EVA_FEATURE_DIM)).astype(np.float32),
        behavior_norm=rng.normal(0, 1, (n, 206)).astype(np.float32),
        frame_mask=np.ones(n, dtype=bool), n_selected=n, n_dropped_tail=0,
        n_behavior_invalid=0,
    )


def _batch(**kw):
    torch.manual_seed(0)
    return collate([_rec(4), _rec(2)], **kw)


def test_forward_shapes_and_masked_mean():
    model = DualStreamDI(_cfg())
    model.eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch)
    assert out.p.shape == (2, 4, PROJ_DIM) and out.h0.shape == (2, PROJ_DIM)
    assert out.pred.shape == (2,) and out.pred is out.bdi_pred
    # h0 is the *masked* temporal mean of GRU outputs, not h_T:
    h_t, _ = model.gru(out.p)
    mask = batch.frame_mask.unsqueeze(-1).float()
    manual = (h_t * mask).sum(1) / mask.sum(1)
    assert torch.allclose(out.h0, manual, atol=1e-6)
    n_min = int(batch.frame_mask.sum(1).min())
    assert not torch.allclose(out.h0[:, 0], h_t[:, -1, 0], atol=1e-4)  # != h_T shortcut
    assert n_min == 2


def test_heads_are_train_gated_and_lazy():
    model = DualStreamDI(_cfg(), n_subjects=3, t1_enable=True, t3_enable=True)
    batch = _batch()
    out = model(batch)  # training mode by construction
    assert model.frame_head is None
    frame_logits, video_logits = model.identity_logits(out, batch)
    assert model.frame_head is not None  # created on demand
    assert frame_logits.shape == (2, 4, 3) and video_logits.shape == (2, 3)
    assert next(model.frame_head.parameters()).device == batch.visual.device
    model.eval()
    with torch.no_grad():
        out_eval = model(batch)
    assert model.identity_logits(out_eval, batch) == (None, None)  # eval sees no heads


def test_materialize_identity_heads_enables_strict_checkpoint_load():
    """Export-side pin: lazy heads must exist BEFORE strict load_state_dict."""
    model = DualStreamDI(_cfg(), n_subjects=3, t1_enable=True, t3_enable=True)
    batch = _batch()
    out = model(batch)  # training path; identity_logits materializes the heads
    model.identity_logits(out, batch)
    state = model.state_dict()
    assert "frame_head.weight" in state and "video_head.weight" in state
    fresh = DualStreamDI(_cfg(), n_subjects=3, t1_enable=True, t3_enable=True)
    with pytest.raises(RuntimeError, match="frame_head"):
        fresh.load_state_dict(state)  # the caught export bug
    fresh = DualStreamDI(_cfg(), n_subjects=3, t1_enable=True, t3_enable=True)
    fresh.materialize_identity_heads()
    fresh.load_state_dict(state)  # strict, pinned export path
    assert torch.allclose(fresh.frame_head.weight, model.frame_head.weight)


def test_heads_disabled_stay_none_and_need_subject_count():
    model = DualStreamDI(_cfg(), t1_enable=False, t3_enable=False)
    out = model(_batch())
    assert model.identity_logits(out, out) == (None, None)
    assert model.frame_head is None and model.video_head is None
    armed = DualStreamDI(_cfg(), t1_enable=True)  # n_subjects missing
    armed_batch = _batch()
    out = armed(armed_batch)
    with pytest.raises(EvaDiSchemaError, match="subject count"):
        armed.identity_logits(out, armed_batch)


def test_grl_reverses_gradient_sign_and_scale():
    p = torch.randn(2, 3, PROJ_DIM, requires_grad=True)
    head = nn.Linear(PROJ_DIM, 4)
    grl = GradientReversalLayer(lambda_=0.5)
    assert torch.equal(grl(p), p)  # forward is identity
    reversed_grad = autograd.grad(head(grl(p)).sum(), p, retain_graph=True)[0]
    plain_grad = autograd.grad(head(p).sum(), p)[0]
    assert torch.allclose(reversed_grad, -0.5 * plain_grad, atol=1e-6)


def test_lambda_scheduling_targets_each_grl():
    model = DualStreamDI(_cfg(), lambda1=0.05, lambda3=0.05)
    model.set_lambda(lambda1=0.5, lambda3=0.25)
    assert model.grl_frame.lambda_ == 0.5 and model.grl_video.lambda_ == 0.25


@pytest.mark.parametrize("mode,zero_key", [("eva_only", "behavior"), ("of3_only", "visual")])
def test_ablations_zero_input_streams_with_identical_graph(mode, zero_key):
    torch.manual_seed(7)
    model = DualStreamDI(_cfg(), input_mode=mode).eval()
    batch = _batch()
    zeroed = dataclasses.replace(
        batch, **{zero_key: torch.zeros_like(getattr(batch, zero_key))})
    with torch.no_grad():
        assert torch.equal(model(batch).p, model(zeroed).p)
    with torch.no_grad():
        dual = DualStreamDI(_cfg(), input_mode="dual").eval()(batch)
    assert not torch.allclose(model(batch).p, dual.p, atol=1e-6)  # actually differs


def test_rejects_non_float32_and_unknown_mode():
    batch = _batch()
    with pytest.raises(EvaDiSchemaError, match="float32"):
        DualStreamDI(_cfg()).eval()(
            dataclasses.replace(batch, visual=batch.visual.double()))
    with pytest.raises(EvaDiSchemaError, match="input_mode"):
        DualStreamDI(_cfg(), input_mode="gap_only")


def test_backward_updates_backbone_of_model_and_identity_gradient_flows():
    torch.manual_seed(1)
    model = DualStreamDI(_cfg(), n_subjects=2, t1_enable=True, t3_enable=True,
                         lambda1=0.5, lambda3=0.5)
    batch = _batch()
    out = model(batch)
    frame_logits, video_logits = model.identity_logits(out, batch)
    loss = out.pred.square().mean() + frame_logits.square().mean() \
        + video_logits.square().mean()
    loss.backward()
    proj_weight = model.projector[0].weight.grad
    assert proj_weight is not None and torch.isfinite(proj_weight).all()
    assert proj_weight.abs().sum() > 0
    assert model.frame_head.weight.grad is not None
    assert model.behavior_branch.net[1].weight.grad is not None
