"""metrics: ccc/mae/rmse contracts, pinned severity bounds, worst-group MAE."""

import numpy as np
import pytest

from src.eva_di.metrics import ccc, mae, rmse, severity_group, severity_worst_group_mae


def test_basic_regression_values():
    y = np.array([10.0, 20.0, 30.0])
    y_hat = np.array([12.0, 18.0, 36.0])
    assert mae(y, y_hat) == pytest.approx((2 + 2 + 6) / 3)
    assert rmse(y, y_hat) == pytest.approx(np.sqrt((4 + 4 + 36) / 3))
    perfect = ccc(y, y + 1e-9 * np.arange(3))
    assert 0.99999 < perfect <= 1.0
    assert ccc(y, 40.0 - y) < 0.0  # anti-correlated


def test_ccc_degenerate_cases_are_nan_not_crash():
    assert np.isnan(ccc(np.array([1.0]), np.array([2.0])))            # n<2
    assert np.isnan(ccc(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0])))
    assert np.isnan(ccc(np.array([7.0, 7.0]), np.array([1.0, 2.0])))  # constant target


def test_shape_and_empty_guards():
    with pytest.raises(ValueError):
        mae(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        rmse(np.array([]), np.array([]))


@pytest.mark.parametrize("score,group", [
    (0, "mild"), (12.9, "mild"),
    (13, "moderate"), (18.5, "moderate"),
    (19, "severe"), (27.9, "severe"),
    (28, "very_severe"), (63, "very_severe"),
    (float("nan"), "unknown"), (None, "unknown"),
])
def test_severity_boundaries_and_unknown(score, group):
    assert severity_group(score) == group


def test_worst_group_mae_and_counts():
    y = np.array([10.0, 11.0, 20.0, float("nan")])
    y_hat = np.array([11.0, 12.0, 30.0, 25.0])
    worst, value, counts = severity_worst_group_mae(y, y_hat)
    assert worst == "severe" and value == pytest.approx(10.0)
    assert counts == {"mild": 2, "severe": 1, "unknown": 1}
