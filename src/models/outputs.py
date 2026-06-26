from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class MTLLiteOutput:
    """Outputs returned by the lightweight multi-task BDI model."""

    bdi_pred: torch.Tensor
    ordinal_logits: Optional[torch.Tensor] = None
    shared_features: Optional[torch.Tensor] = None
    # RPDF Stage A1: per-layer video-level embeddings, keyed by layer name.
    # Only populated when ``forward(..., return_layer_features=True)`` is used
    # by the layer-wise identity probe; the training path leaves this as None.
    layer_features: Optional[Dict[str, torch.Tensor]] = None


@dataclass
class MTLLiteLosses:
    """Named loss components for MTL-Lite training and logging."""

    total: torch.Tensor
    regression: Optional[torch.Tensor] = None
    ordinal: Optional[torch.Tensor] = None
    ccc: Optional[torch.Tensor] = None
