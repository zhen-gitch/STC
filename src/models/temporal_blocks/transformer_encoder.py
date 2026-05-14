import torch
import torch.nn as nn
# import torch.nn.functional.scaled_dot_product_attention
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)    # [1, max_len, d_model] 方便和 batch 数据相加
        self.register_buffer('pe', pe)  # 注册为 buffer，不会被优化器更新，但会随模型保存

    def forward(self, x):
        # x 形状: [Batch, Seq_len, hidden_dim]
        seq_len = x.size(1)
        # 将位置编码加到特征上
        # x = x + self.pe[:seq_len, :]
        # 切片[:, :seq_len, :]！
        # 从预设的 5000 长度里，精确截取前 seq_len (400) 个位置
        x = x + self.pe[:, :seq_len, :]
        return x

class TransformerEncoderBlock(nn.Module):
    """进阶模型：Transformer Encoder (为之后预留)"""
    def __init__(self, input_dim, hidden_dim, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 实例化位置编码
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim)

        # 定义单层 Transformer 结构
        # dim_feedforward 通常设置为 hidden_dim 的 4 倍
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,    # 强制要求输入输出都是 [Batch, Seq, Dim]
            norm_first=True
        )

        # 堆叠多层
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x, mask=None):
        # 注入位置信息
        x = self.pos_encoder(x)

        # 处理掩码 (Mask)
        if mask is not None:
            # PyTorch 规定 True 的地方是不参与计算的 (被忽略的)
            # 先前在 dataset 中定义的 mask 是: 有效帧=1, 补零帧=0。所以要反转它，把 0 变成 True
            src_key_padding_mask = (mask == 0).bool()
        else:
            src_key_padding_mask = None

        # 送入 Transformer (x: [B, S, D], src_key_padding_mask: [B, S])
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        return out # 输出: [Batch, Seq, hidden_dim], 对接后续池化层
