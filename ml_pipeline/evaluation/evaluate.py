from sklearn.metrics import classification_report
import numpy as np
import torch

class ModelEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
    
    def generate_report(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return classification_report(
            all_labels, all_preds, 
            target_names=['goal', 'save', 'dribble', 'shot', 'other']
        )

# 使用示例
if __name__ == "__main__":
    from model_training.models.highlight_model import HighlightLSTM
    from model_training.train import HighlightDataset
    from torch.utils.data import DataLoader
    
    # 加载测试数据
    test_features = np.load("datasets/test/features.npy")
    test_labels = np.load("datasets/test/labels.npy")
    test_dataset = HighlightDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 加载模型
    model = HighlightLSTM(input_dim=2048, hidden_dim=512, num_classes=5)
    model.load_state_dict(torch.load("best_model.pth"))
    
    evaluator = ModelEvaluator(model)
    print(evaluator.generate_report(test_loader))