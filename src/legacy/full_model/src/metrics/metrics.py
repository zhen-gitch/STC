import torch
import torchmetrics


def concordance_correlation_coefficient(preds, targets, eps=1e-8):
    preds = preds.float().view(-1)
    targets = targets.float().view(-1)

    valid = torch.isfinite(preds) & torch.isfinite(targets)
    preds = preds[valid]
    targets = targets[valid]

    if preds.numel() < 2:
        return preds.sum() * 0.0

    pred_mean = preds.mean()
    target_mean = targets.mean()
    pred_var = torch.mean((preds - pred_mean) ** 2)
    target_var = torch.mean((targets - target_mean) ** 2)
    covariance = torch.mean((preds - pred_mean) * (targets - target_mean))

    return (2.0 * covariance) / (
        pred_var + target_var + (pred_mean - target_mean) ** 2 + eps
    )


def concordance_ccc_loss(preds, targets, eps=1e-8):
    preds = preds.float().view(-1)
    targets = targets.float().view(-1)

    valid = torch.isfinite(preds) & torch.isfinite(targets)
    preds = preds[valid]
    targets = targets[valid]

    if preds.numel() < 2:
        return preds.sum() * 0.0

    return 1.0 - concordance_correlation_coefficient(preds, targets, eps=eps)


class ConcordanceCorrCoefMetric(torchmetrics.Metric):
    full_state_update = False

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_x", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_x2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_y2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_xy", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        preds = preds.detach().float().view(-1)
        targets = targets.detach().float().view(-1)

        valid = torch.isfinite(preds) & torch.isfinite(targets)
        preds = preds[valid]
        targets = targets[valid]

        self.n += preds.numel()
        self.sum_x += preds.sum()
        self.sum_y += targets.sum()
        self.sum_x2 += (preds ** 2).sum()
        self.sum_y2 += (targets ** 2).sum()
        self.sum_xy += (preds * targets).sum()

    def compute(self):
        n = torch.clamp(self.n, min=1.0)
        mean_x = self.sum_x / n
        mean_y = self.sum_y / n
        var_x = self.sum_x2 / n - mean_x ** 2
        var_y = self.sum_y2 / n - mean_y ** 2
        cov_xy = self.sum_xy / n - mean_x * mean_y

        ccc = (2.0 * cov_xy) / (
            var_x + var_y + (mean_x - mean_y) ** 2 + self.eps
        )
        return torch.where(self.n > 1.0, ccc, torch.zeros_like(ccc))
