"""
file:   /src/models/temporal_blocks/mtl_blocks.py
"""
import torch
import torch.nn as nn

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
    def __init__(self, input_dim, expert_dim, num_shared_gen, num_shared_con, num_specific, dropout):
        super(CGC, self).__init__()
        # 任务特定专家 Task-Specific Experts
        self.reg_experts = nn.ModuleList(Expert(input_dim, expert_dim, dropout) for _ in range(num_specific))
        self.cls_experts = nn.ModuleList(Expert(input_dim, expert_dim, dropout) for _ in range(num_specific))

        # 双共享专家 (Dual Shared Experts)
        # 通用共享专家 (不受对比学习直接约束，仅受 Reg/Cls 联合回传的梯度更新)
        self.shared_experts_gen = nn.ModuleList(Expert(input_dim, expert_dim, dropout) for _ in range(num_shared_gen))
        # 对比共享专家 (受对比学习强烈约束)
        self.shared_experts_con = nn.ModuleList(Expert(input_dim, expert_dim, dropout) for _ in range(num_shared_con))

        # 门控网络 Gates
        total_experts = num_shared_gen + num_shared_con + num_specific
        # 回归门： 选择 回归专家 + 共享专家
        self.reg_gate = nn.Sequential(nn.Linear(input_dim, total_experts),nn.Softmax(dim=-1))
        # 分类门： 选择分类专家 + 共享专家
        self.cls_gate = nn.Sequential(nn.Linear(input_dim, total_experts),nn.Softmax(dim=-1))
        # 共享门： 用于对比学习的特征聚合
        self.con_gate = nn.Sequential(nn.Linear(input_dim, num_shared_con),nn.Softmax(dim=-1))

    def forward(self, x):
        # 计算所有专家的输出
        reg_expert_outs = [expert(x).unsqueeze(1) for expert in self.reg_experts]
        cls_expert_outs = [expert(x).unsqueeze(1) for expert in self.cls_experts]
        shared_gen_outs = [expert(x).unsqueeze(1) for expert in self.shared_experts_gen]
        shared_con_outs = [expert(x).unsqueeze(1) for expert in self.shared_experts_con]

        # ----------------------------------------------------
        # 回归任务聚合 (特定 + 通用共享 + 对比共享)
        # ----------------------------------------------------
        # 拼接形状 [Batch, num_specific + num_shared, expert_dim]
        reg_all_outs = torch.cat(reg_expert_outs + shared_gen_outs + shared_con_outs, dim=1)
        reg_gate_weights = self.reg_gate(x).unsqueeze(-1)   # [Batch, num_specific + num_shared, 1]
        reg_features = torch.sum(reg_all_outs * reg_gate_weights, dim=1) # 加权求和

        # ----------------------------------------------------
        # 分类任务聚合 (特定 + 通用共享 + 对比共享)
        # ----------------------------------------------------
        cls_all_outs = torch.cat(cls_expert_outs + shared_gen_outs + shared_con_outs, dim=1)
        cls_gate_weights = self.cls_gate(x).unsqueeze(-1)
        cls_features = torch.sum(cls_all_outs * cls_gate_weights, dim=1)

        # ----------------------------------------------------
        # 对比学习特征聚合 (仅来自对比共享专家)
        # ----------------------------------------------------
        con_only_outs = torch.cat(shared_con_outs, dim=1)
        con_gate_weights = self.con_gate(x).unsqueeze(-1)
        con_features = torch.sum(con_only_outs * con_gate_weights, dim=1)

        return reg_features, cls_features, con_features
