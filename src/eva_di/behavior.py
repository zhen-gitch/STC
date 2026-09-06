"""Train-only behavior normalization for the 206-d OpenFace3 axis.

Design contract (EVA_DI_MODULE_DESIGN.md section 5): reuse the pinned, frozen
train-only statistics shipped with the OpenFace3 formal outputs
(``behavior_normalization_train_v1/technical.json``).  In v1 BOTH config modes
(``reuse`` and ``recompute``) feed training from the pinned json; ``recompute``
additionally runs the independent recompute-and-compare guard as an AUDIT lever
(user decision 2026-09-05: the guard must not slow the production ``reuse`` run
path, so ``reuse`` skips it and instead hash-pins the accepted stats into
provenance through :func:`load_pinned_stats`; run ``recompute`` to audit that
val/test frames could never have shifted a mean).

Axis order is locked to ``contracts.expected_behavior_axis_names()`` and
cross-checked against the json ``feature_names`` field-by-field.  Non-finite
behavior values are masked upstream (of3_registry) and never reach statistics
or normalization.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.eva_di.contracts import (
    BEHAVIOR_DIM,
    BEHAVIOR_NORM_FILE,
    BEHAVIOR_NORM_POLICY,
    BEHAVIOR_NORM_SCHEMA,
    SCHEMA_BEHAVIOR,
    EvaDiFingerprintError,
    EvaDiPathError,
    EvaDiSchemaError,
    expected_behavior_axis_names,
)
from src.eva_di.of3_registry import VideoRecord

_STD_FLOOR = 1e-6
# Tolerances calibrated against the REAL pinned artifact (E2E, 2026-09-05):
# our csv-only recompute sees 160559 rows while the generator fitted 160494
# (its quality policy drops 65 frames via a manifest/exclusion registry that
# is not reconstructible from features.csv alone).  Residuals of that
# population gap measure mean<=1.3e-4, std-ratio<=1.3e-2 in the pinned
# normalized space; the guard below those bounds still rejects every gross
# contamination (val/test inclusion shifts stats by O(1) sigma) while
# tolerating the irreducible 65-frame difference.  Leakage safety itself is
# structural: stats are only ever fit over ``split_ids["train"]``.
_ABS_TOL = 1e-3
_REL_TOL = 5e-2


@dataclass(frozen=True)
class BehaviorStats:
    mean: np.ndarray  # [206] float32
    std: np.ndarray  # [206] float32, already floored at _STD_FLOOR
    count: int  # valid train frames behind the stats
    source_sha256: str  # sha256 of the pinned json
    valid_frame_count: int
    feature_names: tuple[str, ...]

    def normalize(self, matrix: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Standardize rows where ``valid``; invalid rows stay zero (masked)."""
        if matrix.ndim != 2 or matrix.shape[1] != BEHAVIOR_DIM:
            raise EvaDiSchemaError(f"behavior matrix must be [N,{BEHAVIOR_DIM}]")
        if matrix.shape[0] != valid.shape[0]:
            raise EvaDiSchemaError("behavior/valid length mismatch")
        out = np.zeros_like(matrix, dtype=np.float32)
        if valid.any():
            out[valid] = (matrix[valid] - self.mean) / self.std
        return out


def load_pinned_stats(norm_root: Path, *, source_sha256: str = "") -> BehaviorStats:
    path = Path(norm_root) / BEHAVIOR_NORM_FILE
    if not path.is_file():
        raise EvaDiPathError(f"pinned behavior stats missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("behavior_schema", "normalization_schema", "policy", "split",
                "feature_names", "mean", "std", "count", "valid_frame_count"):
        if key not in payload:
            raise EvaDiSchemaError(f"pinned behavior stats missing key {key!r}: {path}")
    if payload["behavior_schema"] != SCHEMA_BEHAVIOR:
        raise EvaDiFingerprintError(
            f"behavior_schema {payload['behavior_schema']!r} != {SCHEMA_BEHAVIOR!r}"
        )
    if payload["normalization_schema"] != BEHAVIOR_NORM_SCHEMA:
        raise EvaDiFingerprintError(
            f"normalization_schema {payload['normalization_schema']!r} "
            f"!= {BEHAVIOR_NORM_SCHEMA!r}"
        )
    if payload["policy"] != BEHAVIOR_NORM_POLICY or payload["split"] != "train":
        raise EvaDiFingerprintError(
            f"stats must be policy={BEHAVIOR_NORM_POLICY}/split=train, got "
            f"{payload['policy']!r}/{payload['split']!r}"
        )
    names = tuple(payload["feature_names"])
    expected_names = expected_behavior_axis_names()
    if names != expected_names:
        diffs = [
            (i, got, want)
            for i, (got, want) in enumerate(zip(names, expected_names))
            if got != want
        ]
        raise EvaDiFingerprintError(f"feature_names axis drift at {diffs[:5]}")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if mean.shape != (BEHAVIOR_DIM,) or std.shape != (BEHAVIOR_DIM,):
        raise EvaDiSchemaError(f"stats mean/std must be length {BEHAVIOR_DIM}")
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise EvaDiSchemaError("pinned behavior stats contain non-finite values")
    if float(std.min()) <= 0.0:
        raise EvaDiSchemaError("pinned behavior stats contain non-positive std")
    from src.eva_di.paths import sha256_file

    return BehaviorStats(
        mean=mean,
        std=np.maximum(std, float(payload.get("std_floor", _STD_FLOOR))),
        count=int(payload["count"][0]) if isinstance(payload["count"], list) else int(payload["count"]),
        source_sha256=source_sha256 or sha256_file(path),
        valid_frame_count=int(payload["valid_frame_count"]),
        feature_names=names,
    )


def recompute_train_stats(train_records: list[VideoRecord]) -> np.ndarray:
    """Fit [2, 206] mean/std from train-split valid behavior rows only."""
    if not train_records:
        raise EvaDiSchemaError("cannot recompute behavior stats with zero train records")
    rows = [
        record.behavior[record.behavior_valid]
        for record in train_records
        if record.behavior_valid.any()
    ]
    if not rows:
        raise EvaDiSchemaError(
            "no valid behavior rows across the train records; refusing to fit "
            "stats from an empty population")
    stacked = np.concatenate(rows, axis=0).astype(np.float32)
    if not np.isfinite(stacked).all():
        raise EvaDiSchemaError(
            "recompute received non-finite rows despite valid mask; masked rows "
            "must never enter statistics"
        )
    mean = stacked.mean(axis=0)
    std = np.maximum(stacked.std(axis=0), _STD_FLOOR)
    return np.stack([mean, std])


def assert_stats_consistency(
    stats: BehaviorStats,
    train_records: list[VideoRecord],
    *,
    abs_tol: float = _ABS_TOL,
    rel_tol: float = _REL_TOL,
) -> None:
    """Recompute from train rows only; any tolerance breach stops the run."""
    fitted = recompute_train_stats(train_records)
    mean_delta = np.abs(fitted[0] - stats.mean)
    ratio = fitted[1] / stats.std
    mean_bad = int(np.argmax(mean_delta > abs_tol)) if bool((mean_delta > abs_tol).any()) else -1
    std_bad = int(np.argmax(np.abs(ratio - 1.0) > rel_tol)) if bool((np.abs(ratio - 1.0) > rel_tol).any()) else -1
    if mean_bad >= 0 or std_bad >= 0:
        raise EvaDiFingerprintError(
            "train-recomputed behavior stats disagree with pinned stats "
            f"(worst mean col {mean_bad} delta {float(mean_delta.max()):.3e}; "
            f"worst std col {std_bad} ratio {float(ratio[np.argmax(np.abs(ratio - 1.0))]):.6f})"
        )
