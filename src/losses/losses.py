import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSubjectContinuousContrastiveLoss(nn.Module):
    """
    跨受试者连续流形对比损失。

    Batch 内 BDI 分数越接近的样本互为更强正样本，同时联合三视角嵌入
    约束模型学习与抑郁程度连续变化更一致的表征空间。
    """

    def __init__(self, temperature=0.07, bdi_sigma=3.0):
        super().__init__()
        self.temperature = temperature
        self.bdi_sigma = bdi_sigma

    def forward(self, embeds_orig, embeds_v1, embeds_v2, bdi_scores):
        z_orig = F.normalize(embeds_orig, p=2, dim=-1)
        z_v1 = F.normalize(embeds_v1, p=2, dim=-1)
        z_v2 = F.normalize(embeds_v2, p=2, dim=-1)

        all_embeds = torch.cat([z_orig, z_v1, z_v2], dim=0)
        extended_bdi = bdi_scores.repeat(3)
        bdi_diff = torch.abs(extended_bdi.unsqueeze(0) - extended_bdi.unsqueeze(1))
        similarity_mask = torch.exp(- (bdi_diff ** 2) / (2 * (self.bdi_sigma ** 2)))

        sim_matrix = torch.matmul(all_embeds, all_embeds.T) / self.temperature
        sim_max, _ = torch.max(sim_matrix, dim=-1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        exp_logits = torch.exp(logits)
        diag_mask = torch.eye(exp_logits.size(0), device=exp_logits.device)
        exp_logits = exp_logits * (1 - diag_mask)

        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True) + 1e-8
        log_prob = logits - torch.log(sum_exp_logits)

        valid_mask = similarity_mask * (1 - diag_mask)
        loss_info_nce = - (valid_mask * log_prob).sum(dim=-1) / torch.clamp(
            valid_mask.sum(dim=-1),
            min=1.0
        )

        all_embeds_norm = all_embeds - all_embeds.mean(dim=0, keepdim=True)
        corr_matrix = torch.matmul(all_embeds_norm.T, all_embeds_norm) / (all_embeds.size(0) - 1)
        diag = torch.diagonal(corr_matrix)
        corr_loss = (corr_matrix ** 2).sum() - (diag ** 2).sum()

        return loss_info_nce.mean() + 0.001 * corr_loss
