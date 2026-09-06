"""Regression metrics and severity grouping (numpy-only, decision-safe)."""

from __future__ import annotations

import math

import numpy as np

from src.eva_di.contracts import SEVERITY_BOUNDS, SEVERITY_GROUP_NAMES


def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    y, y_hat = _pair(y, y_hat)
    return float(np.abs(y - y_hat).mean())


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    y, y_hat = _pair(y, y_hat)
    return float(np.sqrt(np.square(y - y_hat).mean()))


def ccc(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Lin's concordance correlation; NaN when either side is degenerate."""
    y, y_hat = _pair(y, y_hat)
    if y.size < 2:
        return float("nan")
    var_y, var_hat = y.var(), y_hat.var()
    if var_y <= 0.0 or var_hat <= 0.0:
        return float("nan")
    cov = ((y - y.mean()) * (y_hat - y_hat.mean())).mean()
    denominator = var_y + var_hat + (y.mean() - y_hat.mean()) ** 2
    if denominator <= 0.0 or math.isnan(denominator):
        return float("nan")
    return float(2.0 * cov / denominator)


def pcc(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Pearson product-moment correlation; NaN when either side is degenerate.

    Same NaN-on-degenerate contract as ccc (a constant prediction stream carries
    no linear association to reward; it must not read as 0.0 correlation).
    """
    y, y_hat = _pair(y, y_hat)
    if y.size < 2:
        return float("nan")
    yc, hc = y - y.mean(), y_hat - y_hat.mean()
    denom = math.sqrt(float((yc * yc).sum()) * float((hc * hc).sum()))
    if denom <= 0.0 or math.isnan(denom):
        return float("nan")
    return float((yc * hc).sum() / denom)


def severity_group(score) -> str:
    """Pinned bounds (13, 19, 28); anything non-convertible or non-finite ->
    'unknown'.

    ``float()`` conversion covers np.float32/float16, numeric strings and
    Decimal NaN -- the old ``isinstance(score, float) and isnan`` test let
    np.float32 NaN fall through to 'very_severe' (audit R2-P2 metrics #15,
    reviving exactly the legacy bug the docstring claimed fixed).
    """
    if score is None:
        return "unknown"
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(value):
        return "unknown"
    low, mid, high = SEVERITY_BOUNDS
    if value < low:
        return "mild"
    if value < mid:
        return "moderate"
    if value < high:
        return "severe"
    return "very_severe"


def severity_worst_group_mae(
    y: np.ndarray, y_hat: np.ndarray
) -> tuple[str, float, dict[str, int]]:
    y, y_hat = _pair(y, y_hat)
    groups = [severity_group(v) for v in y]
    stats: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for group, target, pred in zip(groups, y.tolist(), y_hat.tolist()):
        stats.setdefault(group, []).append(abs(target - pred))
        counts[group] = counts.get(group, 0) + 1
    per_group = {g: float(np.mean(v)) for g, v in stats.items()}
    # worst group excludes 'unknown' (no-label bucket) and non-finite entries,
    # and breaks ties by the pinned SEVERITY_GROUP_NAMES order so repeated
    # runs of the same numbers always name the same group (audit R2-P3 #17).
    candidates = {
        g: v for g, v in per_group.items()
        if g != "unknown" and math.isfinite(v)
    }
    if not candidates:
        # every label unknown (or degenerate): fall back deterministically to
        # the first bucket present in SEVERITY_GROUP_NAMES order.
        ordered = [g for g in SEVERITY_GROUP_NAMES if g in per_group]
        worst = ordered[0]
    else:
        worst = max(
            candidates,
            key=lambda g: (candidates[g], -SEVERITY_GROUP_NAMES.index(g)),
        )
    return worst, per_group[worst], counts


def _pair(y, y_hat) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=np.float64)
    y_hat_arr = np.asarray(y_hat, dtype=np.float64)
    # compare SHAPES before flattening: [N,1] vs [N] or [a,b] vs [b,a] used to
    # pass and then compare element-wise in C order (audit R2-P3 #16).
    if y_arr.shape != y_hat_arr.shape:
        raise ValueError(
            f"metric shape mismatch: {y_arr.shape} vs {y_hat_arr.shape}")
    y_flat, y_hat_flat = y_arr.reshape(-1), y_hat_arr.reshape(-1)
    if y_flat.size == 0:
        raise ValueError("metrics require at least one sample")
    return y_flat, y_hat_flat
