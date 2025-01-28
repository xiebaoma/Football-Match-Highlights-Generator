import torch
import torch.nn as nn

class HighlightLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        out, _ = self.lstm(x)
        # 取最后一个时间步
        out = out[:, -1, :]  
        return self.fc(out)