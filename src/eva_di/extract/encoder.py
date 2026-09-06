"""Frozen EVA02-small encoder with pinned, verified local weights.

The 112->224 bicubic + ``pretrained_cfg`` mean/std transform is an equivalent
re-implementation of the read-only reference
``trans/src/ribformer/models/image.py`` (:300-333), written with provenance
attribution and *not* imported from that repository (trans AGENTS section 2
excludes this research direction; zero runtime coupling).  Equivalence is
proven at extraction time by ``oracle_equivalence.py`` (cos >= 0.999, evidence
under ``cache_root/_equivalence`` only) -- NOTE: that harness is NOT
implemented yet (unclosed item), so no cached feature may claim proven
equivalence.

Weight loading is offline-only with byte-level SHA/size pinning; there is no
fallback path (``HF_HUB_OFFLINE`` is FORCED, not setdefault'd -- an inherited
``HF_HUB_OFFLINE=0`` used to silently enable network fetches, audit R2-P2).
timm is imported lazily so pure-CI environments can import the rest of
``eva_di`` without it.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

from src.eva_di.contracts import (
    EVA_BACKBONE,
    EVA_FEATURE_DIM,
    EVA_INPUT_SIZE,
    EVA_SOURCE_SIZE,
    EVA_TOKEN_COUNT,
    EVA_WEIGHT_SHA256,
    EVA_WEIGHT_SIZE_BYTES,
    EvaDiError,
    EvaDiFingerprintError,
    EvaDiPathError,
    EvaDiSchemaError,
)
from src.eva_di.paths import sha256_file


def verify_weight_file(weight_path: Path) -> dict:
    path = Path(weight_path)
    if not path.is_file():
        raise EvaDiPathError(f"pinned EVA02 weight file missing: {path}")
    size = path.stat().st_size
    if size != EVA_WEIGHT_SIZE_BYTES:
        raise EvaDiFingerprintError(f"weight size {size} != {EVA_WEIGHT_SIZE_BYTES}")
    digest = sha256_file(path)
    if digest != EVA_WEIGHT_SHA256:
        raise EvaDiFingerprintError(f"weight sha256 {digest} != {EVA_WEIGHT_SHA256}")
    return {"path": str(path), "size_bytes": size, "sha256": digest}


class Eva02FrozenEncoder:
    def __init__(
        self,
        weight_path: Path,
        *,
        device: str = "cuda",
        chunk: int = 128,
        autocast: str = "bf16",
    ) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        if isinstance(chunk, bool) or int(chunk) <= 0:
            raise ValueError("chunk must be positive")
        if autocast not in ("bf16", "none"):
            raise ValueError("autocast must be bf16|none")
        self.weight_artifact = verify_weight_file(weight_path)
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            # fail fast: a silent CPU fallback would mislabel provenance and
            # silently wreck throughput expectations (design doc rule: pinned
            # device, no fallback path).
            raise EvaDiError(
                f"device {device!r} requested but CUDA is unavailable; "
                "pass device='cpu' explicitly if that is intended")
        self.device = torch.device(device)
        self.chunk = int(chunk)
        self.autocast = autocast
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError("Eva02FrozenEncoder requires timm (conda env light)") from exc
        self._timm_version = getattr(timm, "__version__", "unknown")
        self.model = timm.create_model(
            EVA_BACKBONE,
            pretrained=True,
            pretrained_cfg_overlay={"file": self.weight_artifact["path"]},
            img_size=EVA_INPUT_SIZE,
            num_classes=0,
        )
        cfg = self.model.pretrained_cfg
        mean = torch.tensor([float(v) for v in cfg["mean"]], dtype=torch.float32)
        std = torch.tensor([float(v) for v in cfg["std"]], dtype=torch.float32)
        if mean.numel() != 3 or std.numel() != 3 or bool((std <= 0).any()):
            raise EvaDiSchemaError("pretrained normalization stats are invalid")
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._mean = mean.view(1, 3, 1, 1).to(self.device)
        self._std = std.view(1, 3, 1, 1).to(self.device)
        # timm pretrained_cfg["input_size"] is (C, H, W); the side length is the
        # LAST dim (an earlier [0] read the channel count -- caught by the E2E run).
        cfg_input = tuple(cfg.get("input_size", (3, EVA_INPUT_SIZE, EVA_INPUT_SIZE)))
        self._input_size = int(cfg_input[-1])
        if self._input_size != EVA_INPUT_SIZE:
            raise EvaDiSchemaError(f"unexpected cfg input size {cfg_input}")
        self._mean_std = ([float(v) for v in mean.tolist()], [float(v) for v in std.tolist()])

    @torch.no_grad()
    def encode(self, frames_uint8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """frames_uint8 [N,3,112,112] -> (cls [N,384], gap [N,384]) float32 on CPU."""
        if frames_uint8.ndim != 4 or tuple(frames_uint8.shape[1:]) != (3, EVA_SOURCE_SIZE, EVA_SOURCE_SIZE):
            raise EvaDiSchemaError(
                f"expected uint8 frames [N,3,{EVA_SOURCE_SIZE},{EVA_SOURCE_SIZE}], "
                f"got {tuple(frames_uint8.shape)}"
            )
        if frames_uint8.dtype != torch.uint8:
            # a pre-normalized float input would pass the shape check, get
            # divided by 255 AGAIN, and silently produce collapsed features
            # that the sha-pinned cache would cement (audit R2 encoder P2-21).
            raise EvaDiSchemaError(
                f"encode expects uint8 pixels, got dtype {frames_uint8.dtype}")
        outs_cls, outs_gap = [], []
        for start in range(0, frames_uint8.shape[0], self.chunk):
            batch = frames_uint8[start : start + self.chunk].to(self.device)
            x = batch.to(torch.float32) / 255.0
            x = F.interpolate(x, size=(EVA_INPUT_SIZE, EVA_INPUT_SIZE),
                              mode="bicubic", align_corners=False)
            x = (x - self._mean) / self._std
            if self.autocast == "bf16" and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    tokens = self.model.forward_features(x)
            else:
                tokens = self.model.forward_features(x)
            if not isinstance(tokens, torch.Tensor):
                raise EvaDiSchemaError("forward_features must return a Tensor")
            if tuple(tokens.shape[1:]) != (EVA_TOKEN_COUNT, EVA_FEATURE_DIM):
                raise EvaDiSchemaError(
                    f"token structure changed: expected [{EVA_TOKEN_COUNT},{EVA_FEATURE_DIM}], "
                    f"got {tuple(tokens.shape[1:])}"
                )
            tokens = tokens.float()
            outs_cls.append(tokens[:, 0].cpu())
            outs_gap.append(tokens[:, 1:].mean(dim=1).cpu())
        if not outs_cls:
            empty_cls = torch.zeros((0, EVA_FEATURE_DIM), dtype=torch.float32)
            empty_gap = torch.zeros((0, EVA_FEATURE_DIM), dtype=torch.float32)
            return empty_cls, empty_gap  # two DISTINCT tensors (aliasing trap, R2)
        return torch.cat(outs_cls, 0), torch.cat(outs_gap, 0)

    def metadata(self) -> dict:
        # pooling/norm identity pins WHICH feature the cached cls/gap are
        # (forward_features output is pre-fc_norm when timm resolves
        # global_pool='avg'; recorded so manifest drift is visible, R2-P2-20).
        return {
            "backbone": EVA_BACKBONE,
            "input_size": self._input_size,
            "source_input_size": EVA_SOURCE_SIZE,
            "feature_dim": EVA_FEATURE_DIM,
            "token_count": EVA_TOKEN_COUNT,
            "weight": dict(self.weight_artifact),
            "timm_version": self._timm_version,
            "torch_version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "device": str(self.device),
            "autocast": self.autocast,
            "chunk": self.chunk,
            "mean_std": {"mean": self._mean_std[0], "std": self._mean_std[1]},
            "global_pool": str(getattr(self.model, "global_pool", None)),
            "final_norm": type(getattr(self.model, "norm", None)).__name__,
            "fc_norm": type(getattr(self.model, "fc_norm", None)).__name__,
            "num_prefix_tokens": int(getattr(self.model, "num_prefix_tokens", 1)),
            "num_reg_tokens": int(getattr(self.model, "num_reg_tokens", 0)),
        }
