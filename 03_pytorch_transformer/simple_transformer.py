import torch
import torch.nn as nn

# ----------------------------
# 0. デバイス設定
# ----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("使用デバイス:", device)

# ----------------------------
# 1. Transformerモデル定義
# ----------------------------
class SimpleTransformerClassifier(nn.Module):
    def __init__(self, embed_dim=16, num_heads=2, num_layers=2, num_classes=2):
        super().__init__()
        # TransformerEncoderの構成
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        out = self.transformer(x)          # 出力も (batch, seq_len, embed_dim)
        out = out.mean(dim=1)              # 各トークンの平均をとる (global pooling)
        return self.fc(out)                # (batch, num_classes)


# 短い「文」を入力して、2クラス分類（例：ポジティブ or ネガティブ）を行う。
# 実際の単語ではなく「埋め込み（数値ベクトル）」を使って簡略化します。
# ----------------------------
# 2. ダミーデータ準備
# ----------------------------
batch_size = 4   # 1回に4文まとめて処理（バッチ学習）
seq_len = 5      # 各文は5単語で構成されている
embed_dim = 16   # 各単語を16次元ベクトルで表す


x = torch.randn(batch_size, seq_len, embed_dim, device=device)  # 埋め込みベクトル
y = torch.tensor([0, 1, 0, 1], device=device)    # ラベル (2クラス) ４バッチの正解ラベル

print("dummy data")
print("input x:",x)
print("label y:",y)

# ----------------------------
# 3. モデルと学習設定
# ----------------------------
model = SimpleTransformerClassifier(embed_dim=embed_dim, num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ----------------------------
# 4. 学習ループ（超簡略）
# ----------------------------
epochs = 300
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        with torch.no_grad():
            preds = out.argmax(dim=1)
            acc = (preds == y).float().mean().item()
        print(f"Epoch {epoch+1:3d} | loss {loss.item():.4f} | train acc {acc:.2f}")

# ----------------------------
# 5. 推論
# ----------------------------
# inputのxを予想してみる。
model.eval()
with torch.no_grad():
    preds = model(x).argmax(dim=1)
print("予測:", preds.tolist())
