"""Deterministic uniform frame-selection policy ``uniform512_v1``.

Pure function, no IO.  Pinned algorithm (design doc section 5):

    n <= budget        -> np.arange(n)
    otherwise          -> np.unique(np.rint(np.linspace(0, n-1, budget)).astype(np.int64))

Invariants raised on every call: strictly increasing, first==0, last==n-1
(endpoints included), dtype int64, and EXACT length min(n, budget) (audit
R2-P3: the old ``>= budget//2`` check could never fire and gave false
comfort).  ``budget == 1`` cannot preserve both endpoints and is refused.
"""

from __future__ import annotations

import numpy as np

from src.eva_di.contracts import FRAME_BUDGET, SELECTION_POLICY


def uniform_select(n_frames: int, budget: int = FRAME_BUDGET) -> np.ndarray:
    if isinstance(n_frames, bool) or not isinstance(n_frames, (int, np.integer)):
        raise TypeError(f"n_frames must be an integer, got {type(n_frames).__name__}")
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise TypeError(f"budget must be an integer, got {type(budget).__name__}")
    if int(budget) <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    n = int(n_frames)
    if n < 0:
        raise ValueError(f"n_frames must be non-negative, got {n}")
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n <= int(budget):
        selected = np.arange(n, dtype=np.int64)
        expected_size = n
    else:
        if int(budget) == 1:
            raise ValueError(
                "budget=1 cannot include both endpoints of a downsampled "
                "sequence; uniform512_v1 requires budget >= 2")
        selected = np.unique(
            np.rint(np.linspace(0, n - 1, int(budget))).astype(np.int64)
        )
        expected_size = int(budget)
    if selected.size != expected_size:
        raise AssertionError(
            f"uniform_select produced {selected.size} != min(n,budget)={expected_size} "
            f"for n={n}")
    if selected.size and (selected[0] != 0 or selected[-1] != n - 1):
        raise AssertionError(f"uniform_select lost endpoints for n={n}")
    if np.any(np.diff(selected) <= 0):
        raise AssertionError(f"uniform_select not strictly increasing for n={n}")
    return selected.astype(np.int64)


def selection_policy_id(budget: int = FRAME_BUDGET) -> str:
    return f"{SELECTION_POLICY}@{int(budget)}"
