import torch
import torch.nn as nn
import feature_extraction.video_models as video_models

class SlowFastDetector(nn.Module):
    """结合SlowFast和时序建模的事件检测模型"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = video_models.VideoFeatureExtractor(model_type='slowfast')
        self.temporal_model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2304,
                nhead=8,
                dim_feedforward=4096
            ),
            num_layers=3
        )
        self.classifier = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, video_clips):
        # video_clips: (fast_path, slow_path)
        features = self.backbone(video_clips)  # [B, T, 2304]
        temporal_feat = self.temporal_model(features)
        return self.classifier(temporal_feat.mean(dim=1))