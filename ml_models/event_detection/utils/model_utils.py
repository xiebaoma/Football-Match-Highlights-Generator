import torch
import torch.nn as nn
import torchaudio

def save_model(model, path):
    """保存完整模型架构和参数"""
    torch.save({
        'model_state': model.state_dict(),
        'config': model.init_params
    }, path)

def load_model(path, model_class):
    """加载完整模型"""
    checkpoint = torch.load(path)
    model = model_class(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state'])
    return model

def print_model_summary(model, input_shape):
    """打印模型结构摘要"""
    from torchsummary import summary
    summary(model, input_shape)