import torch
import torch.nn as nn
import torchaudio

class AudioFeatureExtractor(nn.Module):
    """音频特征提取器"""
    def __init__(self, sample_rate=16000, n_mels=128):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512
        )
        self.resnet = resnet50(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.resnet.fc = nn.Identity()

    def forward(self, waveform):
        # 输入形状: [batch, time]
        # 生成梅尔频谱图
        spec = self.mel_spec(waveform)  # [B, n_mels, time]
        spec = spec.unsqueeze(1)  # [B, 1, n_mels, time]
        
        # 用CNN提取特征
        features = self.resnet(spec)  # [B, 2048]
        return features