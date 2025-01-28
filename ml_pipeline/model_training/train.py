import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

class HighlightDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]), 
            torch.LongTensor(self.labels[idx])
        )

class HighlightTrainer:
    def __init__(self, model, config):
        self.model = model.to(config['device'])
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['lr']
        )
        self.config = config
        
    def train(self, train_loader, val_loader):
        best_acc = 0.0
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            for batch in train_loader:
                inputs, labels = batch
                inputs = inputs.to(self.config['device'])
                labels = labels.to(self.config['device'])
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            val_acc = self.evaluate(val_loader)
            if val_acc > best_acc:
                torch.save(self.model.state_dict(), 'best_model.pth')
                best_acc = val_acc
            
            print(f"Epoch {epoch+1} | Val Acc: {val_acc:.4f}")

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch
                inputs = inputs.to(self.config['device'])
                labels = labels.to(self.config['device'])
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total

# 使用示例
if __name__ == "__main__":
    from models.highlight_model import HighlightLSTM
    
    # 加载特征和标签
    features = np.load("datasets/processed/features.npy")
    labels = np.load("datasets/labels/annotations.npy")
    
    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # 创建数据加载器
    train_dataset = HighlightDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = HighlightDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 初始化模型和训练器
    model = HighlightLSTM(input_dim=2048, hidden_dim=512, num_classes=5)
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-4,
        'epochs': 20
    }
    
    trainer = HighlightTrainer(model, config)
    trainer.train(train_loader, val_loader)