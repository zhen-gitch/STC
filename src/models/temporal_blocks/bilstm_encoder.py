import torch
import torch.nn as nn

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, x, mask=None):
        # x: [Batch, Seq, input_dim]
        # 对于 LSTM，可以不显式传入 mask，后续的掩码池化会处理掉无用帧
        out, _ = self.lstm(x)
        return out  # 输出: [Batch, Seq, hidden_dim]