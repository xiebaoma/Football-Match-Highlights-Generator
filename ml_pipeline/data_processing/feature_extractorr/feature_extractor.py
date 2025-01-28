import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

class FeatureExtractor:
    def __init__(self, model_name='resnet50'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_name)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self, model_name):
        model = models.__dict__[model_name](pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model = model.to(self.device).eval()
        return model
    
    def extract_frame_features(self, frame_path: str) -> np.ndarray:
        """提取单帧特征"""
        img = Image.open(frame_path).convert('RGB')
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
        
        return features.squeeze().cpu().numpy()
    
    def extract_video_features(self, video_dir: str):
        """提取视频序列特征"""
        frame_files = sorted(Path(video_dir).glob("*.jpg"))
        features = []
        
        for frame_path in frame_files:
            frame_feat = self.extract_frame_features(str(frame_path))
            features.append(frame_feat)
        
        return np.stack(features)  # [seq_len, feature_dim]