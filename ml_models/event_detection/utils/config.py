# feature_extraction/utils/config.py
FEATURE_CONFIG = {
    'video_model': 'slowfast',
    'audio_model': 'resnet',
    'fusion_dim': 1024
}

# event_detection/utils/config.py
DETECTION_CONFIG = {
    'temporal_model': 'transformer',
    'num_classes': 5,
    'hidden_dim': 512
}