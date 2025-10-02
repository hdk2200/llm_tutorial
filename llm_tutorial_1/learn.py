import torch
import torch.nn as nn
import torch.optim as optim

# ダミーデータ（入力x: 100サンプル×3次元, 出力y: 100サンプル×1次元）
x = torch.randn(100, 3)
y = torch.randn(100, 1)

print("入力:", x)  # (サンプル数, 特徴量数)
print("出力(target):", y)  # (サンプル数, 1)

# モデル: 3次元 → 1次元
model = nn.Linear(3, 1)

# 損失関数（回帰なので MSELoss）
loss_fn = nn.MSELoss()

# 最適化手法（確率的勾配降下法）
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 学習ループ
for epoch in range(100):   # 10エポック
    # 1. forward（予測）
    pred = model(x)

    # 2. 損失を計算
    loss = loss_fn(pred, y)

    # 3. 勾配をリセット
    optimizer.zero_grad()

    # 4. backward（勾配計算）
    loss.backward()

    # 5. パラメータ更新
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
