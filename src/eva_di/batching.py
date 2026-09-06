"""Main-process vectorized batching (resident arrays; num_workers=0 design).

Semantics (design doc section 5): tail padding, ``frame_mask`` must be a
prefix of True followed by False (GRU-without-pack precondition, mechanism doc
section 2), zero-valid recordings are refused by name.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.eva_di.contracts import BEHAVIOR_DIM, EVA_FEATURE_DIM, EvaDiSchemaError


@dataclass(frozen=True)
class DIBatch:
    visual: torch.Tensor  # [B,T,Dv] float32 (384 or 768 per use_gap)
    behavior: torch.Tensor  # [B,T,206] float32 (already train-only normalized)
    frame_mask: torch.Tensor  # [B,T] bool, prefix-True
    bdi: torch.Tensor  # [B] float32 (normalized target scale)
    bdi_raw: torch.Tensor  # [B] float32 original 0..63 scale for metrics
    subject: torch.Tensor  # [B] long, -1 unmapped
    sample_ids: tuple[str, ...]


def collate(
    recordings,
    *,
    use_gap: bool = False,
    device: torch.device | str | None = None,
) -> DIBatch:
    if not recordings:
        raise EvaDiSchemaError("empty batch")
    lengths = [int(rec.frame_mask.sum()) for rec in recordings]
    if min(lengths) == 0:
        offenders = [
            rec.sample_id for rec, n in zip(recordings, lengths) if n == 0
        ]
        raise EvaDiSchemaError(f"zero-valid-frame recordings refused: {offenders}")
    batch = len(recordings)
    t_max = max(lengths)
    d_visual = (2 if use_gap else 1) * EVA_FEATURE_DIM  # single source (R2-C8)
    visual = torch.zeros((batch, t_max, d_visual), dtype=torch.float32)
    behavior = torch.zeros((batch, t_max, BEHAVIOR_DIM), dtype=torch.float32)
    frame_mask = torch.zeros((batch, t_max), dtype=torch.bool)
    bdi = torch.zeros(batch, dtype=torch.float32)
    bdi_raw = torch.zeros(batch, dtype=torch.float32)
    subject = torch.full((batch,), -1, dtype=torch.long)
    for i, rec in enumerate(recordings):
        n = lengths[i]
        own = np.asarray(rec.frame_mask)
        if not np.array_equal(own[:n], np.ones(n, dtype=bool)) or bool(own[n:].any()):
            raise EvaDiSchemaError(
                f"frame_mask is not a True-prefix for {rec.sample_id!r}; "
                "recordings must arrive prefix-masked (dataset prefix-run rule)"
            )
        if use_gap:
            frame_visual = np.concatenate([rec.cls[:n], rec.gap[:n]], axis=1)
        else:
            frame_visual = rec.cls[:n]
        if frame_visual.shape[0] != n or rec.behavior_norm.shape[0] < n:
            raise EvaDiSchemaError(
                f"frame/behavior length drift for {rec.sample_id!r}: "
                f"visual {frame_visual.shape[0]} vs mask {n} vs behavior "
                f"{rec.behavior_norm.shape[0]}")
        visual[i, :n] = torch.from_numpy(np.ascontiguousarray(frame_visual))
        behavior[i, :n] = torch.from_numpy(np.ascontiguousarray(rec.behavior_norm[:n]))
        # copy the recording's own mask (do NOT rebuild from lengths, which would
        # silently "repair" a non-prefix mask instead of refusing it)
        frame_mask[i, :n] = torch.from_numpy(np.ascontiguousarray(rec.frame_mask[:n]))
        bdi[i] = float(rec.bdi_norm)
        bdi_raw[i] = float(rec.bdi_score)
        subject[i] = int(rec.subject)
    _assert_prefix_masks(frame_mask, [rec.sample_id for rec in recordings])
    if device is not None:
        visual, behavior = visual.to(device), behavior.to(device)
        frame_mask, bdi, bdi_raw, subject = (
            frame_mask.to(device), bdi.to(device), bdi_raw.to(device), subject.to(device)
        )
    return DIBatch(
        visual=visual, behavior=behavior, frame_mask=frame_mask, bdi=bdi,
        bdi_raw=bdi_raw, subject=subject,
        sample_ids=tuple(str(rec.sample_id) for rec in recordings),
    )


def _assert_prefix_masks(frame_mask: torch.Tensor, sample_ids: list[str]) -> None:
    for i in range(frame_mask.shape[0]):
        mask = frame_mask[i]
        n = int(mask.sum())
        if not bool(mask[:n].all()) or bool(mask[n:].any()):
            raise EvaDiSchemaError(
                f"frame_mask is not a True-prefix for {sample_ids[i]!r}; "
                "padding must be at the tail only"
            )
