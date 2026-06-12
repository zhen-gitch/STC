from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MTLLiteOutput:
    """Outputs returned by the lightweight multi-task BDI model."""

    bdi_pred: torch.Tensor
    ordinal_logits: Optional[torch.Tensor] = None
    shared_features: Optional[torch.Tensor] = None


@dataclass
class MTLLiteLosses:
    """Named loss components for MTL-Lite training and logging."""

    total: torch.Tensor
    regression: torch.Tensor
    ordinal: Optional[torch.Tensor] = None
    ccc: Optional[torch.Tensor] = None
