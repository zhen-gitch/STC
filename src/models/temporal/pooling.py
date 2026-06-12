import torch


def _expand_mask(mask, values):
    return mask.to(device=values.device, dtype=values.dtype).unsqueeze(-1)


def masked_mean_pool(values, mask, eps=1e-8):
    """Pool sequence features with a boolean valid-frame mask.

    Args:
        values: Tensor with shape [batch, seq_len, dim].
        mask: Tensor with shape [batch, seq_len], where true/1 marks valid frames.
        eps: Small value used to avoid division by zero.

    Returns:
        Tensor with shape [batch, dim]. Rows with no valid frames become zeros.
    """
    mask_expanded = _expand_mask(mask, values)
    counts = mask_expanded.sum(dim=1).clamp_min(eps)
    pooled = (values * mask_expanded).sum(dim=1) / counts
    empty_rows = mask_expanded.sum(dim=1) <= 0
    return torch.where(empty_rows, torch.zeros_like(pooled), pooled)


def masked_mean_std_pool(values, mask, eps=1e-8):
    """Return mean + std pooled sequence features under a valid-frame mask."""
    mask_expanded = _expand_mask(mask, values)
    mean = masked_mean_pool(values, mask, eps=eps)
    counts = mask_expanded.sum(dim=1).clamp_min(eps)
    variance = (((values - mean.unsqueeze(1)) * mask_expanded) ** 2).sum(dim=1) / counts
    pooled = mean + torch.sqrt(variance + eps)
    empty_rows = mask_expanded.sum(dim=1) <= 0
    return torch.where(empty_rows, torch.zeros_like(pooled), pooled)
