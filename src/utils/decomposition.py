import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalGaussianFilter(nn.Module):
    """
    一维时序高斯低通滤波器：在时间轴上平滑特征，消除高频突变噪点。
    """

    def __init__(self, channels, kernel_size=5, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels

        x = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        self.register_buffer('weight', kernel_1d.view(1, 1, -1).repeat(channels, 1, 1))
        self.padding = kernel_size // 2

    def forward(self, x):
        if self.kernel_size <= 1:
            return x

        x = x.transpose(1, 2)
        x_filtered = F.conv1d(x, self.weight, padding=self.padding, groups=self.channels)
        return x_filtered.transpose(1, 2)


class DeepDecompositionEncoder(nn.Module):
    """
    多尺度深度时序分解编码器。

    每个尺度分别提取 trend 与 seasonal 分量，再用动态门控融合不同尺度。
    """

    def __init__(self, in_dim, out_dim, moving_avg_kernels=None, gate_temperature=2.0):
        super().__init__()
        moving_avg_kernels = moving_avg_kernels or [7, 15, 31]
        self.kernels = [k if k % 2 != 0 else k + 1 for k in moving_avg_kernels]
        self.num_scales = len(self.kernels)
        self.gate_temperature = float(gate_temperature)

        self.moving_avgs = nn.ModuleList([
            nn.AvgPool1d(kernel_size=k, stride=1, padding=k // 2, count_include_pad=False)
            for k in self.kernels
        ])

        self.trend_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
            for _ in range(self.num_scales)
        ])
        self.seasonal_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
            for _ in range(self.num_scales)
        ])

        self.trend_gate = nn.Linear(out_dim * self.num_scales, self.num_scales)
        self.seasonal_gate = nn.Linear(out_dim * self.num_scales, self.num_scales)

    def forward(self, x, mask=None):
        _, seq_len, _ = x.shape

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = x * mask_expanded

        x_trans = x.transpose(1, 2)
        trend_outs = []
        seasonal_outs = []

        for i, moving_avg in enumerate(self.moving_avgs):
            trend_feat = moving_avg(x_trans).transpose(1, 2)
            if trend_feat.shape[1] > seq_len:
                trend_feat = trend_feat[:, :seq_len, :]

            seasonal_feat = x - trend_feat
            trend_outs.append(self.trend_layers[i](trend_feat))
            seasonal_outs.append(self.seasonal_layers[i](seasonal_feat))

        gamma = max(self.gate_temperature, 1e-6)

        concat_trends = torch.cat(trend_outs, dim=-1)
        trend_weights = F.softmax(self.trend_gate(concat_trends) / gamma, dim=-1).unsqueeze(-1)
        fused_trend = torch.sum(torch.stack(trend_outs, dim=2) * trend_weights, dim=2)

        concat_seasonals = torch.cat(seasonal_outs, dim=-1)
        seasonal_weights = F.softmax(self.seasonal_gate(concat_seasonals) / gamma, dim=-1).unsqueeze(-1)
        fused_seasonal = torch.sum(torch.stack(seasonal_outs, dim=2) * seasonal_weights, dim=2)

        out = fused_trend + fused_seasonal

        self.last_trend_weights = trend_weights.squeeze(-1).detach()
        self.last_seasonal_weights = seasonal_weights.squeeze(-1).detach()

        if mask is not None:
            out = out * mask_expanded

        return out
