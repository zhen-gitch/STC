import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CGC(nn.Module):
    """
    CGC - Customized Gate Control 架构
    解决多任务学习中的特征冲突干扰
    """
    def __init__(self, input_dim, expert_dim, num_shared, num_specific, dropout):
        super(CGC, self).__init__()
        # 任务特定专家 Task-Specific Experts
        self.reg_experts = nn.ModuleList(Expert(input_dim, expert_dim, dropout) for _ in range(num_specific))
        self.cls_experts = nn.ModuleList(Expert(input_dim, expert_dim, dropout) for _ in range(num_specific))

        # 共享专家 Shared Experts
        self.shared_experts = nn.ModuleList(Expert(input_dim, expert_dim, dropout) for _ in range(num_shared))

        # 门控网络 Gates
        # 回归门： 选择 回归专家 + 共享专家
        self.reg_gate = nn.Sequential(nn.Linear(input_dim, num_specific + num_shared),nn.Softmax(dim=-1))
        # 分类门： 选择分类专家 + 共享专家
        self.cls_gate = nn.Sequential(nn.Linear(input_dim, num_specific + num_shared),nn.Softmax(dim=-1))
        # 共享门： 用于对比学习的特征聚合
        self.shared_gate = nn.Sequential(nn.Linear(input_dim, num_shared),nn.Softmax(dim=-1))

    def forward(self, x):
        # 计算所有专家的输出
        reg_expert_outs = [expert(x).unsqueeze(1) for expert in self.reg_experts]
        cls_expert_outs = [expert(x).unsqueeze(1) for expert in self.cls_experts]
        shared_expert_outs = [expert(x).unsqueeze(1) for expert in self.shared_experts]

        # 回归任务聚合
        # 拼接形状 [Batch, num_specific + num_shared, expert_dim]
        reg_all_outs = torch.cat(reg_expert_outs + shared_expert_outs, dim=1)
        reg_gate_weights = self.reg_gate(x).unsqueeze(-1)   # [Batch, num_specific + num_shared, 1]
        reg_features = torch.sum(reg_all_outs * reg_gate_weights, dim=1) # 加权求和

        # 分类任务聚合
        cls_all_outs = torch.cat(cls_expert_outs + shared_expert_outs, dim=1)
        cls_gate_weights = self.cls_gate(x).unsqueeze(-1)
        cls_features = torch.sum(cls_all_outs * cls_gate_weights, dim=1)

        # 共享特征聚合，用于对比学习任务
        shared_only_outs = torch.cat(shared_expert_outs, dim=1)
        shared_gate_weights = self.shared_gate(x).unsqueeze(-1)
        shared_features = torch.sum(shared_only_outs * shared_gate_weights, dim=1)

        return reg_features, cls_features, shared_features


class SupervisedContrastiveLoss(nn.Module):
    """
        监督对比学习损失 (SupCon)
        将同类(同等抑郁级别)的样本在特征空间拉近，异类推开
        """
    def __init__(self, temperature=0.7):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: [Batch, embed_dim], labels: [Batch]
        device = features.device
        batch_size = features.shape[0]

        # 进行 L2 归一化，映射到超球面上
        features = F.normalize(features, p=2, dim=1)

        # 计算相似度矩阵 [Batch, Batch]
        similarity_matrix = torch.div(torch.matmul(features, features.T), self.temperature)

        # 创建掩码，找同类的正样本对
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 排除与自己的相似度对角线
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # 减去最大值，保证数值稳定性
        sim_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - sim_max.detach()

        # 计算 SupCon Loss
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # 计算每个样本的正样本平均 log 似然
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -mean_log_prob_pos.mean()
        return loss
