"""batching: tail padding, prefix-mask invariant, zero-valid refusal, gap concat."""

import dataclasses

import numpy as np
import pytest
import torch

from src.eva_di.batching import collate
from src.eva_di.contracts import EvaDiSchemaError
from src.eva_di.dataset import Recording


def _rec(sample_id="a", n=4, *, bdi=20.0, bdi_norm=0.5, subject=1, mask=None):
    mask = np.ones(n, dtype=bool) if mask is None else mask
    return Recording(
        sample_id=sample_id, split="train", subject=subject, bdi_score=bdi,
        bdi_norm=bdi_norm,
        cls=np.full((n, 384), 0.1, dtype=np.float32),
        gap=np.full((n, 384), 0.2, dtype=np.float32),
        behavior_norm=np.full((n, 206), 0.3, dtype=np.float32),
        frame_mask=mask, n_selected=n, n_dropped_tail=0, n_behavior_invalid=0,
    )


def test_ragged_batch_pads_tail_only():
    batch = collate([_rec("a", n=5), _rec("b", n=2)])
    assert batch.visual.shape == (2, 5, 384)
    assert batch.behavior.shape == (2, 5, 206)
    assert batch.frame_mask.dtype == torch.bool
    assert batch.frame_mask.tolist() == [[True] * 5, [True, True, False, False, False]]
    assert torch.equal(batch.visual[1, 2:], torch.zeros(3, 384))  # padding is zeros
    assert torch.equal(batch.visual[0], torch.full((5, 384), 0.1))
    assert batch.sample_ids == ("a", "b")


def test_use_gap_concatenates_cls_and_gap():
    batch = collate([_rec(n=3)], use_gap=True)
    assert batch.visual.shape == (1, 3, 768)
    assert torch.allclose(batch.visual[0, :, :384], torch.full((3, 384), 0.1))
    assert torch.allclose(batch.visual[0, :, 384:], torch.full((3, 384), 0.2))


def test_targets_and_subjects():
    batch = collate([_rec("a", bdi=20.0, bdi_norm=0.5, subject=2),
                     _rec("b", subject=-1)])
    assert torch.equal(batch.bdi, torch.tensor([0.5, 0.5]))
    assert torch.equal(batch.bdi_raw, torch.tensor([20.0, 20.0]))
    assert torch.equal(batch.subject, torch.tensor([2, -1]))


def test_empty_batch_and_zero_valid_refused_by_name():
    with pytest.raises(EvaDiSchemaError, match="empty batch"):
        collate([])
    dead = _rec("zombie", n=3, mask=np.zeros(3, dtype=bool))
    with pytest.raises(EvaDiSchemaError, match="zombie"):
        collate([_rec("ok", n=2), dead])


def test_non_prefix_mask_rejected():
    broken = _rec("jagged", n=4, mask=np.array([True, False, True, False]))
    with pytest.raises(EvaDiSchemaError, match="jagged"):
        collate([broken])


def test_device_argument_is_noop_on_cpu():
    batch = collate([_rec(n=2)], device=torch.device("cpu"))
    assert batch.visual.device.type == "cpu" and batch.subject.dtype == torch.long
    assert dataclasses.is_dataclass(batch)
