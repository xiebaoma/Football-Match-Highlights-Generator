import torch
import torch.nn as nn
import torchaudio

class MultiModalFusion(nn.Module):
    """多模态特征融合模块"""
    def __init__(self, video_dim=2048, audio_dim=512, hidden_dim=1024):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(video_dim + audio_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, video_feat, audio_feat):
        """基于注意力机制的特征融合"""
        combined = torch.cat([video_feat, audio_feat], dim=1)
        attention_weights = torch.softmax(self.attention(combined), dim=1)
        return attention_weights * video_feat + (1 - attention_weights) * audio_feat