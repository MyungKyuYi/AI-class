
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
X = df.drop(columns=["SBP", "DBP"]).values  # 입력 피처
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)

# 전체 모델
class ResNetTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, hidden_dim)
        self.resnet = ResNetBlock(1, 1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc0(x)
        x = x.unsqueeze(1)
        x = self.resnet(x)
        x = x.squeeze(1).unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        x = self.fc_out(x)
        return x

# 모델, 손실, 최적화
model = ResNetTransformer(input_dim=X.shape[1], hidden_dim=64, output_dim=1)
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
