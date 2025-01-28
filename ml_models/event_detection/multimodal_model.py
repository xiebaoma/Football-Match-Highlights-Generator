import torch
import torch.nn as nn
import torchaudio
import feature_extraction as fe 

class MultiModalDetector(nn.Module):
    """融合视频+音频+比赛数据的事件检测"""
    def __init__(self):
        super().__init__()
        self.video_net = fe.VideoFeatureExtractor()
        self.audio_net = fe.AudioFeatureExtractor()
        self.fusion = fe.MultiModalFusion()
        
        self.temporal_model = nn.LSTM(
            input_size=1024,
            hidden_size=512,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )

    def forward(self, video_input, audio_input):
        video_feat = self.video_net(video_input)  # [B, T, 2048]
        audio_feat = self.audio_net(audio_input)  # [B, 512]
        
        # 扩展音频特征
        audio_feat = audio_feat.unsqueeze(1).repeat(1, video_feat.size(1), 1)
        
        # 特征融合
        fused = self.fusion(video_feat, audio_feat)
        
        # 时序建模
        temporal_out, _ = self.temporal_model(fused)
        return self.classifier(temporal_out[:, -1, :])