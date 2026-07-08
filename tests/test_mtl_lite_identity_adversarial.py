"""Stage B1 identity-adversarial MTL tests.

Covers the B1 minimal closed loop (docs/MTL_LITE_DESIGN.md section 13.4 /
docs/TODO.md B1-code-minimal):

- import + config-default checks (switch off by default, head/loss stay None);
- dummy one-batch forward/loss with the switch on and a train-only subject
  index injected: identity_logits shape, finite identity loss, gradients reach
  the subject_id_head;
- the GRL switch does NOT perturb the default baseline (disabled total is
  bit-identical to a model with no identity branch);
- val/test (or unseen-subject) batches produce no identity loss even when the
  head exists;
- the subject index is built train-only and unseen subjects map to None.
"""

import math

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from tests.test_mtl_lite_forward import DummyBackbone, minimal_mtl_lite_config


def _identity_adversarial_config(lambda_id=0.05):
    cfg = minimal_mtl_lite_config()
    cfg.MODEL = {
        "IDENTITY_ADVERSARIAL": {
            "ENABLE": True,
            "LAMBDA_ID": lambda_id,
            "NUM_SUBJECT_CLASSES": 0,  # placeholder; injected from train split
        }
    }
    return cfg


def _build_model(monkeypatch, cfg):
    import src.models.mtl_lite as mtl_lite

    def build_dummy_backbone(model_name, weight_path=None, timm_pretrained=False, img_size=112):
        return DummyBackbone(output_dim=cfg.BACKBONE_OUT_DIMS[model_name])

    monkeypatch.setattr(mtl_lite, "build_feature_backbone", build_dummy_backbone)
    return mtl_lite.MTLLiteDepressionModel(cfg)


def _dummy_batch(subject_ids):
    video = torch.randn(len(subject_ids), 4, 3, 16, 16)
    mask = torch.ones(len(subject_ids), 4, dtype=torch.bool)
    labels = {
        "bdi_score": torch.tensor([12.0] * len(subject_ids), dtype=torch.float32),
        "class_label": torch.tensor([1] * len(subject_ids), dtype=torch.long),
        "subject_id": list(subject_ids),
    }
    return video, mask, labels


# ---------------------------------------------------------------------------
# import / config-default checks
# ---------------------------------------------------------------------------

def test_identity_adversarial_defaults_to_off(monkeypatch):
    cfg = minimal_mtl_lite_config()  # no MODEL.IDENTITY_ADVERSARIAL section
    model = _build_model(monkeypatch, cfg)

    assert model.identity_adversarial is False
    assert model.grl is None
    assert model.subject_id_head is None
    assert model.subject_id_to_index == {}


def test_identity_adversarial_enabled_but_no_index_is_inert(monkeypatch):
    cfg = _identity_adversarial_config()
    model = _build_model(monkeypatch, cfg)

    # Switch on, but set_subject_index not called yet: head stays None and the
    # forward path emits no identity_logits, so behaviour is safe.
    assert model.identity_adversarial is True
    assert model.grl is not None
    assert model.subject_id_head is None

    video, mask, _ = _dummy_batch(["203_1", "204_1"])
    with torch.no_grad():
        outputs = model(video, mask)
    assert outputs.identity_logits is None


# ---------------------------------------------------------------------------
# forward / loss / gradients with the switch on and index injected
# ---------------------------------------------------------------------------

def test_identity_branch_produces_logits_and_loss(monkeypatch):
    cfg = _identity_adversarial_config()
    model = _build_model(monkeypatch, cfg)
    subject_index = {"203_1": 0, "204_1": 1, "205_1": 2}
    model.set_subject_index(subject_index)

    assert model.subject_id_head is not None
    assert len(model.subject_id_to_index) == 3

    video, mask, labels = _dummy_batch(["203_1", "204_1"])
    model.train()
    outputs = model(video, mask)
    losses = model.compute_losses(outputs, labels, stage="train")

    assert outputs.identity_logits.shape == (2, 3)
    assert torch.isfinite(outputs.identity_logits).all()
    assert losses.identity is not None and torch.isfinite(losses.identity)
    # total = reg + ccc*w + ordinal*w + identity (lambda already in GRL)
    expected = (
        losses.regression
        + model.ccc_loss_weight * losses.ccc
        + model.ordinal_weight * losses.ordinal
        + losses.identity
    )
    assert torch.allclose(losses.total, expected)

    # Gradients must reach the subject-id head despite the GRL reversal.
    model.zero_grad(set_to_none=True)
    losses.total.backward()
    head_grads = [p.grad for p in model.subject_id_head.parameters() if p.grad is not None]
    assert head_grads
    assert all(torch.isfinite(g).all() for g in head_grads)
    assert sum(g.detach().abs().sum().item() for g in head_grads) > 0.0


# ---------------------------------------------------------------------------
# default baseline is bit-identical when the switch is off
# ---------------------------------------------------------------------------

def test_disabled_switch_matches_baseline_total(monkeypatch):
    cfg_off = minimal_mtl_lite_config()
    cfg_on = _identity_adversarial_config()
    torch.manual_seed(0)
    video, mask, labels = _dummy_batch(["203_1", "204_1"])

    model_off = _build_model(monkeypatch, cfg_off)
    model_off.eval()
    with torch.no_grad():
        out_off = model_off(video, mask)
        loss_off = model_off.compute_losses(out_off, labels, stage="train")

    # A switch-on model with NO index injected has grl (no params) and no
    # subject_id_head, so its state_dict keys equal the switch-off model's.
    # Copy weights so any output difference can ONLY come from the identity
    # branch -- which is dormant (head is None), so outputs must match exactly.
    model_on = _build_model(monkeypatch, cfg_on)
    model_on.load_state_dict(model_off.state_dict(), strict=True)
    model_on.eval()
    with torch.no_grad():
        out_on = model_on(video, mask)
        loss_on = model_on.compute_losses(out_on, labels, stage="train")

    assert out_on.identity_logits is None
    assert torch.allclose(out_off.bdi_pred, out_on.bdi_pred)
    assert torch.allclose(out_off.ordinal_logits, out_on.ordinal_logits)
    assert torch.allclose(loss_off.regression, loss_on.regression)
    assert torch.allclose(loss_off.ccc, loss_on.ccc)
    assert torch.allclose(loss_off.total, loss_on.total)


# ---------------------------------------------------------------------------
# val/test (and unseen-subject train) batches must not trigger identity loss
# ---------------------------------------------------------------------------

def test_val_stage_skips_identity_loss(monkeypatch):
    cfg = _identity_adversarial_config()
    model = _build_model(monkeypatch, cfg)
    model.set_subject_index({"203_1": 0, "204_1": 1})

    video, mask, labels = _dummy_batch(["203_1", "204_1"])
    model.train()
    outputs = model(video, mask)
    # Even though all subjects are known, val stage must not add identity loss.
    losses_val = model.compute_losses(outputs, labels, stage="val")
    assert losses_val.identity is None
    assert torch.allclose(
        losses_val.total,
        losses_val.regression + model.ccc_loss_weight * losses_val.ccc + model.ordinal_weight * losses_val.ordinal,
    )


def test_unseen_subject_skips_identity_loss(monkeypatch):
    cfg = _identity_adversarial_config()
    model = _build_model(monkeypatch, cfg)
    model.set_subject_index({"203_1": 0, "204_1": 1})  # 999_1 is unseen

    video, mask, labels = _dummy_batch(["203_1", "999_1"])
    model.train()
    outputs = model(video, mask)
    assert outputs.identity_logits is not None  # head ran on the whole batch

    losses = model.compute_losses(outputs, labels, stage="train")
    # Any unmapped subject -> the whole batch skips identity loss (no partial CE).
    assert losses.identity is None
    assert torch.allclose(
        losses.total,
        losses.regression + model.ccc_loss_weight * losses.ccc + model.ordinal_weight * losses.ordinal,
    )


# ---------------------------------------------------------------------------
# subject index mapping helpers
# ---------------------------------------------------------------------------

def test_subject_index_mapping_train_only(monkeypatch):
    cfg = _identity_adversarial_config()
    model = _build_model(monkeypatch, cfg)
    model.set_subject_index({"203_1": 0, "204_1": 1, "205_1": 2})

    mapped = model._map_subjects_to_index(["203_1", "205_1", "204_1"])
    assert mapped is not None
    assert mapped.tolist() == [0, 2, 1]

    # Unseen subject -> None for the whole batch.
    assert model._map_subjects_to_index(["203_1", "999_1"]) is None
    # Empty table -> None.
    model.subject_id_to_index = {}
    assert model._map_subjects_to_index(["203_1"]) is None
