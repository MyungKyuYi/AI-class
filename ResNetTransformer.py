import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Multi-Head Self-Attention
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, D = x.shape
        H = self.num_heads
        d = self.head_dim
        
        Q = self.query(x).view(B, N, H, d).transpose(1, 2)
        K = self.key(x).view(B, N, H, d).transpose(1, 2)
        V = self.value(x).view(B, N, H, d).transpose(1, 2)
        
        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
        attn_weight = F.softmax(attn_score, dim=-1)
        attn_out = torch.matmul(attn_weight, V)
        
        out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)

# -----------------------------
# Transformer Encoder Layer
# -----------------------------
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads=4, ff_dim=128, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# -----------------------------
# ResNet Block
# -----------------------------
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

# -----------------------------
# 전체 모델: ResNet + Transformer + 분류기
# -----------------------------
class ResNetTransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes,
                 num_resnet_blocks=2, num_transformer_layers=2,
                 num_heads=4, ff_dim=128):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, hidden_dim)

        self.resnet_blocks = nn.Sequential(*[
            ResNetBlock(1) for _ in range(num_resnet_blocks)
        ])

        self.transformer_layers = nn.Sequential(*[
            CustomTransformerEncoderLayer(hidden_dim, num_heads, ff_dim)
            for _ in range(num_transformer_layers)
        ])

        # Classification head
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc0(x)             # (B, F) → (B, H)
        x = x.unsqueeze(1)          # (B, H) → (B, 1, H)
        x = x.transpose(1, 2)       # (B, 1, H) → (B, H, 1)
        x = self.resnet_blocks(x)
        x = x.transpose(1, 2)       # (B, H, 1) → (B, 1, H)
        x = self.transformer_layers(x)
        x = x.squeeze(1)
        logits = self.fc_out(x)
        return logits, F.softmax(logits, dim=-1)
