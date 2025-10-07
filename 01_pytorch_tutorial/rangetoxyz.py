import torch
import torch.nn as nn
import torch.optim as optim



# ネットワーク定義
class RangeToXYZNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 16)   # 線形回帰（Linear Regression） 入力1次元 → N個の特徴
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)   # 線形回帰（Linear Regression） 中間N → 出力3 (X,Y,Z)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# モデルを作る
model = RangeToXYZNet()

# 損失関数と最適化 ---------------------
criterion = nn.CrossEntropyLoss()

# SGD（確率的勾配降下法）
# optimizer = optim.SGD(model.parameters(), lr=0.05)

# Adamで最適化 Adam（Adaptive Moment Estimation）
optimizer = optim.Adam(model.parameters(), lr=0.01)

# データ準備 ----------------------------
# 入力データ（0〜29）
x = torch.arange(0, 30, dtype=torch.float32).unsqueeze(1)
x = x / 29.0

# 正解ラベル（0: X, 1: Y, 2: Z）

label={0:"X",1:"Y",2:"Z"}

y = torch.zeros(30, dtype=torch.long)
y[10:20] = 1
y[20:30] = 2



# 学習ループ ----------------------------
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss = {loss.item():.4f}")

# 結果確認 ------------------------------
with torch.no_grad():   # 勾配計算をしないモード 推論のみのとき
    for value in range(30):
        x_single = torch.tensor([[float(value)]])
        pred = model(x_single).argmax(dim=1).item()
        print(f"x={value:2d} → {label[pred]}")

with torch.no_grad():
    pred = model(x).argmax(dim=1)        # 予測結果（0,1,2）
    pred_labels = [label[int(p)] for p in pred]

    # 正解数をカウント
    correct = (pred == y).sum().item()
    total = len(y)
    accuracy = correct / total
    error_rate = 1 - accuracy

    # 出力
    print("\n入力値:", x.squeeze().tolist())
    print("予測クラス:", pred_labels)
    print(f"\n正解数: {correct}/{total}")
    print(f"正解率: {accuracy*100:.2f}%")
    print(f"間違い率: {error_rate*100:.2f}%")