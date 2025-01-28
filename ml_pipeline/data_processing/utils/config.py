VIDEO_CONFIG = {
    'frame_interval': 10,      # 关键帧采样间隔
    'clip_duration': 5,        # 视频片段时长（秒）
    'resize_dim': (224, 224)   # 视频缩放尺寸
}

MODEL_CONFIG = {
    'feature_dim': 2048,       # 特征维度（ResNet50）
    'lstm_hidden_dim': 512,
    'num_classes': 5
}

TRAIN_CONFIG = {
    'batch_size': 32,
    'lr': 1e-4,
    'epochs': 20
}