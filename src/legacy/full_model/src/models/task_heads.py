import torch
import torch.nn as nn
import torch.nn.functional as F


def build_regression_task_head(input_dim: int, hidden_dim: int, output_dim: int):
    """Build a normalized BDI regression head.

    The head returns an unconstrained normalized score. Losses are computed
    against ``BDI / max_score``; metrics clamp the normalized prediction to
    ``[0, 1]`` and then restore the real BDI scale.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def build_classification_task_head(input_dim: int, hidden_dim: int, output_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim - 1),
    )


def build_contrastive_task_head(input_dim: int, hidden_dim: int, output_dim: int):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def get_coral_levels(labels: torch.Tensor, num_classes: int):
    levels = torch.zeros(labels.size(0), num_classes - 1, device=labels.device)
    for i in range(labels.size(0)):
        levels[i, : labels[i]] = 1
    return levels


def coral_loss(logits: torch.Tensor, levels: torch.Tensor):
    val = -torch.sum(
        F.logsigmoid(logits) * levels
        + (F.logsigmoid(logits) - logits) * (1 - levels),
        dim=1
    )
    return torch.mean(val)
