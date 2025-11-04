import tensorflow as tf
from tensorflow.keras import layers, models, activations

# -----------------------------
# Multi-Head Self-Attention
# -----------------------------
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.query = layers.Dense(dim)
        self.key = layers.Dense(dim)
        self.value = layers.Dense(dim)
        self.out = layers.Dense(dim)

    def call(self, x):
        B = tf.shape(x)[0]
        N = tf.shape(x)[1]
        D = x.shape[-1]
        H = self.num_heads
        d = self.head_dim

        # Linear projections
        Q = tf.reshape(self.query(x), (B, N, H, d))
        K = tf.reshape(self.key(x), (B, N, H, d))
        V = tf.reshape(self.value(x), (B, N, H, d))
        
        Q = tf.transpose(Q, perm=[0, 2, 1, 3])  # (B, H, N, d)
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])

        # Scaled Dot-Product Attention
        attn_score = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(d, tf.float32))
        attn_weight = tf.nn.softmax(attn_score, axis=-1)
        attn_out = tf.matmul(attn_weight, V)  # (B, H, N, d)
        
        # Concat heads
        attn_out = tf.transpose(attn_out, perm=[0, 2, 1, 3])  # (B, N, H, d)
        attn_out = tf.reshape(attn_out, (B, N, D))  # (B, N, D)
        return self.out(attn_out)

# -----------------------------
# Transformer Encoder Layer
# -----------------------------
class CustomTransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=4, ff_dim=128, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ff = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(dim),
        ])
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out, training=training))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out, training=training))
        return x

# -----------------------------
# ResNet Block
# -----------------------------
class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = layers.Conv1D(channels, 3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(channels, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, x, training=False):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = out + residual
        return self.relu(out)

# -----------------------------
# 전체 모델: ResNet + Transformer + 분류기
# -----------------------------
class ResNetTransformerClassifier(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, num_classes,
                 num_resnet_blocks=2, num_transformer_layers=2,
                 num_heads=4, ff_dim=128):
        super().__init__()
        self.fc0 = layers.Dense(hidden_dim)

        self.resnet_blocks = [ResNetBlock(1) for _ in range(num_resnet_blocks)]

        self.transformer_layers = [
            CustomTransformerEncoderLayer(hidden_dim, num_heads, ff_dim)
            for _ in range(num_transformer_layers)
        ]

        self.fc_out = layers.Dense(num_classes, activation=None)

    def call(self, x, training=False):
        # (B, F) → (B, H)
        x = self.fc0(x)
        # (B, H) → (B, 1, H)
        x = tf.expand_dims(x, axis=1)
        # (B, 1, H) → (B, H, 1)
        x = tf.transpose(x, perm=[0, 2, 1])

        for block in self.resnet_blocks:
            x = block(x, training=training)

        # (B, H, 1) → (B, 1, H)
        x = tf.transpose(x, perm=[0, 2, 1])
        for layer in self.transformer_layers:
            x = layer(x, training=training)

        x = tf.squeeze(x, axis=1)
        logits = self.fc_out(x)
        probs = tf.nn.softmax(logits, axis=-1)
        return logits, probs

# -----------------------------
# ✅ 모델 테스트
# -----------------------------
if __name__ == "__main__":
    model = ResNetTransformerClassifier(
        input_dim=128,    # 예시 입력 feature 수
        hidden_dim=64,
        num_classes=5,    # 예시: 5-class 분류
        num_resnet_blocks=2,
        num_transformer_layers=2
    )

    dummy = tf.random.normal((16, 128))  # (batch, feature)
    logits, probs = model(dummy)
    print("logits:", logits.shape)
    print("probs:", probs.shape)
