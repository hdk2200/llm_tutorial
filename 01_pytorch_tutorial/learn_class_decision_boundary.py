import torch
print(torch.backends.mps.is_available())

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# MPSが利用可能なら使う、なければCPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print("使用デバイス:", device)


# 1. ダミーデータ
x = torch.tensor([[1.0, 1.0],
                  [1.5, 2.0],
                  [3.0, 3.5],
                  [4.0, 5.0]])

y = torch.tensor([0, 0, 1, 1])  # 犬=0, 猫=1

# 2. モデル定義 (2次元入力→2クラス)
model = nn.Linear(2, 2)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
x, y = x.to(device), y.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 3. 学習ループ
for epoch in range(100):
    logits = model(x)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 4. 可視化のためのグリッドデータを作成
xx, yy = torch.meshgrid(
    torch.linspace(0, 6, 100),
    torch.linspace(0, 6, 100),
    indexing="xy"   # 警告回避
)

grid = torch.cat([xx.reshape(-1,1), yy.reshape(-1,1)], dim=1)

# デバイスに移動
grid = grid.to(device)

# モデルで予測
with torch.no_grad():
    logits = model(grid)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)


# 5. 描画
plt.contourf(xx.cpu(), yy.cpu(), preds.reshape(xx.shape).cpu(),
             alpha=0.3, cmap="coolwarm")  # 領域を色分け

plt.scatter(x[:,0].cpu(), x[:,1].cpu(), c=y.cpu(),
            s=100, cmap="coolwarm", edgecolors="k")   # データ点

plt.xlabel("特徴量1")
plt.ylabel("特徴量2")
plt.title("犬(0) vs 猫(1) の分類境界")
plt.show()
