import torch
import torch.nn as nn
import torchaudio
import math

class VideoTransformer(nn.Module):
    """基于Transformer的时序建模"""
    def __init__(self, input_dim=2048, num_classes=5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=2048
            ),
            num_layers=4
        )
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [B, T, D]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.classifier(x[:, -1, :])

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:x.size(1), :]