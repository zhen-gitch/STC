"""frame_selection: uniform512_v1 golden values and invariants."""

import numpy as np
import pytest

from src.eva_di.frame_selection import uniform_select


def test_small_video_keeps_all_frames():
    assert uniform_select(5).tolist() == [0, 1, 2, 3, 4]
    assert uniform_select(512).tolist() == list(range(512))


def test_golden_downsample():
    assert uniform_select(7, budget=3).tolist() == [0, 3, 6]
    assert uniform_select(1000, budget=10).tolist() == [
        0, 111, 222, 333, 444, 555, 666, 777, 888, 999,
    ]


def test_invariants_at_real_scale():
    sel = uniform_select(1644)
    assert sel.dtype == np.int64
    assert sel[0] == 0 and sel[-1] == 1643
    assert np.all(np.diff(sel) > 0)
    assert 512 // 2 <= sel.size <= 512


def test_zero_frames():
    sel = uniform_select(0)
    assert sel.size == 0 and sel.dtype == np.int64


def test_negative_and_bad_budget_rejected():
    with pytest.raises(ValueError):
        uniform_select(-1)
    with pytest.raises(ValueError):
        uniform_select(10, budget=0)


def test_selection_is_deterministic_across_calls():
    a, b = uniform_select(2000), uniform_select(2000)
    assert np.array_equal(a, b)
