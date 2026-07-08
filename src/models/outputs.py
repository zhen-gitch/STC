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
    # Stage B1: subject-id head logits, computed from the GRL-adapted
    # ``shared_features``.  Populated only when ``identity_adversarial`` is
    # enabled AND a train-only subject index has been injected; otherwise
    # ``None`` so the default baseline path is unchanged.
    identity_logits: Optional[torch.Tensor] = None


@dataclass
class MTLLiteLosses:
    """Named loss components for MTL-Lite training and logging."""

    total: torch.Tensor
    regression: Optional[torch.Tensor] = None
    ordinal: Optional[torch.Tensor] = None
    ccc: Optional[torch.Tensor] = None
    # Stage B1: subject-id CE loss.  Only non-None during training when
    # ``identity_adversarial`` is on and every subject in the batch maps to the
    # train-only subject index.  Default ``None`` keeps existing callers and
    # the default baseline total loss bit-identical.
    identity: Optional[torch.Tensor] = None
