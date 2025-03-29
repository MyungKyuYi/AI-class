
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
df = pd.read_csv("your_feature_data.csv")  # RED + IR feature DataFrame
X = df.drop(columns=["SBP", "DBP"]).values
y_sbp = df["SBP"].values
y_dbp = df["DBP"].values

# 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 텐서 변환
X_tensor = torch.tensor(X, dtype=torch.float32)
y_sbp_tensor = torch.tensor(y_sbp, dtype=torch.float32).view(-1, 1)
y_dbp_tensor = torch.tensor(y_dbp, dtype=torch.float32).view(-1, 1)

# 데이터로더
train_dataset = TensorDataset(X_tensor, y_sbp_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# ResNet 블록
class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)

# 전체 모델
class ResNetTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_resnet_blocks=2, num_transformer_layers=2):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, hidden_dim)

        # ResNet blocks 반복
        self.resnet_blocks = nn.Sequential(*[
            ResNetBlock(1) for _ in range(num_resnet_blocks)
        ])

        # Transformer blocks 반복
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc0(x)             # (B, F) → (B, H)
        x = x.unsqueeze(1)          # (B, H) → (B, 1, H)
        x = x.transpose(1, 2)       # (B, 1, H) → (B, H, 1)
        x = self.resnet_blocks(x)   # ResNet blocks
        x = x.transpose(1, 2)       # (B, H, 1) → (B, 1, H)
        x = self.transformer(x)     # Transformer blocks
        x = x.squeeze(1)            # (B, 1, H) → (B, H)
        x = self.fc_out(x)          # (B, H) → (B, 1)
        return x

# 모델 생성 (예: ResNet 3개, Transformer 2개)
model = ResNetTransformer(
    input_dim=X.shape[1],
    hidden_dim=64,
    output_dim=1,
    num_resnet_blocks=3,
    num_transformer_layers=2
)

# 손실 함수와 최적화
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 함수
def train(model, loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

# 학습 실행
train(model, train_loader, criterion, optimizer)
