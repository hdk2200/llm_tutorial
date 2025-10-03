import torch
import torch.nn as nn
import torch.optim as optim

# ダミーデータ (特徴量2つ)
# 犬(0): 小さめの値, 猫(1): 大きめの値
x = torch.tensor([[1.0, 1.0],   # 犬
                  [1.5, 2.0],   # 犬
                  [3.0, 3.5],   # 猫
                  [4.0, 5.0]])  # 猫
y = torch.tensor([0, 0, 1, 1])  # 正解ラベル

# モデル: 入力2次元 → 出力2クラス
model = nn.Linear(2, 2)  

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
x, y = x.to(device), y.to(device)

# 損失関数（分類なので CrossEntropyLoss）
loss_fn = nn.CrossEntropyLoss()

# 最適化手法
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 学習ループ
for epoch in range(20):
    # 1. 予測 (ロジット)
    logits = model(x)

    # 2. 損失計算
    loss = loss_fn(logits, y)

    # 3. 勾配リセット → 逆伝播 → 更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

## 推論（予測）
# 新しいデータを入力
x_new = torch.tensor([[2.0, 2.5]]).to(device)  # 新しい動物



logits = model(x_new)               # ロジット
probs = torch.softmax(logits, dim=1) # 確率に変換
pred_class = torch.argmax(probs, dim=1)

print("ロジット:", logits)
print("確率:", probs)
print("予測クラス:", pred_class.item())  # 0=犬, 1=猫


# 新しいサンプルをまとめて入力
x_test = torch.tensor([
    [1.0, 1.2],   # 犬っぽい
    [2.0, 2.5],   # 中間
    [3.5, 4.0],   # 猫っぽい
    [5.0, 5.5]    # 完全に猫
]).to(device)

# モデルで予測
logits = model(x_test)                # ロジット（生スコア）
probs = torch.softmax(logits, dim=1)  # 確率に変換
preds = torch.argmax(probs, dim=1)    # クラス番号を取得

print("ロジット:\n", logits)
print("確率:\n", probs)
print("予測クラス:\n", preds)
