import torch
import torch.nn as nn
from torchvision.models import resnet50, r3d_18
from torch.hub import load_state_dict_from_url

class VideoFeatureExtractor(nn.Module):
    """支持多种视频特征提取架构"""
    def __init__(self, model_type='resnet3d', pretrained=True):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'resnet3d':
            self.model = r3d_18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_type == 'slowfast':
            self.model = SlowFastBackbone(pretrained)
            self.feature_dim = 2304
        else:
            raise ValueError("Unsupported model type")

        # 冻结所有层
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        """输入形状：
        - 3D CNN: [batch, channels, depth, height, width]
        - SlowFast: tuple(fast_path, slow_path)
        """
        return self.model(x)

class SlowFastBackbone(nn.Module):
    """简化版SlowFast网络"""
    def __init__(self, pretrained=True):
        super().__init__()
        self.fast_path = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        )
        
        self.slow_path = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(5,7,7), stride=(1,2,2), padding=(2,3,3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        )
        
        if pretrained:
            self.load_pretrained()

    def load_pretrained(self):
        try:
            state_dict = load_state_dict_from_url(
                "https://example.com/slowfast.pth")  # 替换实际URL
            self.load_state_dict(state_dict)
        except:
            print("Warning: Failed to load pretrained weights")

    def forward(self, x):
        fast_input = x[0]  # [B, C, T, H, W]
        slow_input = x[1]  # [B, C, T/4, H, W]
        
        fast_feat = self.fast_path(fast_input)
        slow_feat = self.slow_path(slow_input)
        
        # 特征融合
        return torch.cat([
            fast_feat.mean(dim=2), 
            slow_feat.mean(dim=2)
        ], dim=1)