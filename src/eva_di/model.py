"""Dual-stream DI model: projector (exit P) -> GRU -> masked mean (exit V).

Mechanism (docs/DUAL_LEVEL_IDENTITY_ADVERSARIAL_PLAN.md sections 2-3):
T1 frame-level GRL subject head on exit P, T3 video-level GRL subject head on
exit V, T2 regression on exit V.  The GRL itself is *reused* from
``src.models.gradient_reversal`` (forward identity, backward -lambda).  Identity
heads are constructed lazily with the train-only subject count, are gated by
``self.training`` **and** their config flags, and the lazy head is moved to the
input device immediately after creation (fix list item 3).  Since audit R2 each
enabled head is created INDIVIDUALLY (flag -> key-set), so a strict
``load_state_dict`` can detect t1/t3 flag drift between training and export
(R2-C6).  The trainer must call :meth:`DualStreamDI.materialize_identity_heads`
BEFORE snapshotting ``model.parameters()`` for the optimizer, otherwise the
lazy heads would never enter the optimizer (audit R2-P0: heads stuck at init,
all DI arms invalid).  Ablations zero the projector *input streams* so the
graph shape stays identical across ``dual|eva_only|of3_only``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.eva_di.batching import DIBatch
from src.eva_di.config import ModelCfg
from src.eva_di.contracts import (
    BEHAVIOR_DIM,
    EVA_FEATURE_DIM,
    EvaDiSchemaError,
)
from src.models.gradient_reversal import GradientReversalLayer


@dataclass(frozen=True)
class DIForward:
    p: torch.Tensor  # [B,T,proj_dim]  exit P (frame level)
    h0: torch.Tensor  # [B,proj_dim]   exit V (video level, masked mean of GRU)
    bdi_pred: torch.Tensor  # [B]

    @property
    def pred(self) -> torch.Tensor:
        return self.bdi_pred


class BehaviorBranch(nn.Module):
    def __init__(self, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(BEHAVIOR_DIM),
            nn.Linear(BEHAVIOR_DIM, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualStreamDI(nn.Module):
    def __init__(
        self,
        cfg: ModelCfg,
        *,
        input_mode: str = "dual",
        n_subjects: int | None = None,
        t1_enable: bool = False,
        t3_enable: bool = False,
        lambda1: float = 0.05,
        lambda3: float = 0.05,
    ):
        super().__init__()
        if input_mode not in ("dual", "eva_only", "of3_only"):
            raise EvaDiSchemaError(f"unknown input_mode {input_mode!r}")
        if cfg.gru_hidden != cfg.proj_dim:
            # the masked mean + reg head assume hidden==proj; validated at
            # config parse since 2026-09-05, kept here as defense in depth
            # (``assert`` was stripped by `python -O`, audit R2-P2).
            raise EvaDiSchemaError(
                f"model.gru_hidden({cfg.gru_hidden}) must equal proj_dim({cfg.proj_dim})")
        self.cfg = cfg
        self.input_mode = input_mode
        self.t1_enable = bool(t1_enable)
        self.t3_enable = bool(t3_enable)
        visual_dim = 2 * EVA_FEATURE_DIM if cfg.use_gap else EVA_FEATURE_DIM
        self.behavior_branch = BehaviorBranch(cfg.behavior_mlp_dim, cfg.behavior_dropout)
        self.projector = nn.Sequential(
            nn.Linear(visual_dim + cfg.behavior_mlp_dim, cfg.proj_dim),
            nn.LayerNorm(cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(cfg.proj_dropout),
        )
        self.gru = nn.GRU(cfg.proj_dim, cfg.gru_hidden, num_layers=cfg.gru_layers,
                          batch_first=True)
        self.reg_head = nn.Linear(cfg.proj_dim, 1)
        self.frame_head: nn.Module | None = None
        self.video_head: nn.Module | None = None
        self.grl_frame = GradientReversalLayer(lambda_=lambda1)
        self.grl_video = GradientReversalLayer(lambda_=lambda3)
        self._n_subjects = n_subjects

    # -- lambda scheduling (runner calls per epoch with w-scaled lambda) -----
    def set_lambda(self, *, lambda1: float, lambda3: float) -> None:
        self.grl_frame.lambda_ = float(lambda1)
        self.grl_video.lambda_ = float(lambda3)

    def materialize_identity_heads(self) -> None:
        """The trainer calls this BEFORE building the optimizer's parameter
        list; checkpoint loaders call it before a strict ``load_state_dict``
        against a training-armed state_dict.  No-op when both heads are
        disabled (R2-P3: a flags-off model must not demand a subject count)."""
        self._ensure_heads(next(self.parameters()).device)

    def _ensure_heads(self, device: torch.device) -> None:
        if not (self.t1_enable or self.t3_enable):
            return
        if self._n_subjects is None or self._n_subjects <= 0:
            raise EvaDiSchemaError(
                "identity heads need a positive train-only subject count"
            )
        # per-flag creation: the state_dict key set mirrors the enabled flags,
        # so strict loading catches t1/t3 drift (R2-C6).
        if self.t1_enable and self.frame_head is None:
            self.frame_head = nn.Linear(self.cfg.proj_dim, self._n_subjects).to(device)
        if self.t3_enable and self.video_head is None:
            self.video_head = nn.Linear(self.cfg.proj_dim, self._n_subjects).to(device)

    def _streams(self, batch: DIBatch) -> tuple[torch.Tensor, torch.Tensor]:
        visual = batch.visual
        behavior = batch.behavior
        if self.input_mode == "eva_only":
            behavior = torch.zeros_like(behavior)
        elif self.input_mode == "of3_only":
            visual = torch.zeros_like(visual)
        return visual, behavior

    def forward(self, batch: DIBatch) -> DIForward:
        if batch.visual.dtype != torch.float32 or batch.behavior.dtype != torch.float32:
            raise EvaDiSchemaError("batches arrive in float32; autocast is the runner's job")
        visual, behavior = self._streams(batch)
        b = self.behavior_branch(behavior)
        p = self.projector(torch.cat([visual, b], dim=-1))  # [B,T,proj]
        h_t, _ = self.gru(p)
        mask = batch.frame_mask.unsqueeze(-1).to(h_t.dtype)  # [B,T,1]
        counts = mask.sum(dim=1)  # [B,1] broadcast over channels
        if bool((counts <= 0).any()):
            # a zero-valid recording would silently become an all-zero h0 ->
            # constant prediction (audit R2-P2; collate/dataset already refuse
            # this, so reaching here means a new unvalidated path).
            raise EvaDiSchemaError(
                "recording with zero valid frames reached the masked mean; "
                "this must have been refused by collate/dataset")
        h0 = (h_t * mask).sum(dim=1) / counts  # masked temporal mean [B,proj]
        bdi_pred = self.reg_head(h0).squeeze(-1)
        return DIForward(p=p, h0=h0, bdi_pred=bdi_pred)

    def identity_logits(self, out: DIForward, batch: DIBatch):
        """Returns (frame_logits [B,Tv,S] | None, video_logits [B,S] | None).

        Eval/val runs must see ``(None, None)`` -- heads are training-only and
        config-gated (fix list item 2).  ``batch`` is currently unused by the
        head math (masking lives in losses.ce_masked); the parameter is kept
        for the documented module interface (design §4).
        """
        if not self.training or not (self.t1_enable or self.t3_enable):
            return None, None
        self._ensure_heads(out.p.device)
        frame_logits = None
        video_logits = None
        if self.t1_enable and self.frame_head is not None:
            frame_logits = self.frame_head(self.grl_frame(out.p))
        if self.t3_enable and self.video_head is not None:
            video_logits = self.video_head(self.grl_video(out.h0))
        return frame_logits, video_logits
