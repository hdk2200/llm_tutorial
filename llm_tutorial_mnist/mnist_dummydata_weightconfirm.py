import torch
import torch.nn as nn
import torch.optim as optim

# シンプルなモデル (784→100→10)
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデル作成
model = SimpleNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ダミーデータ
x = torch.randn(16, 1, 28, 28)
y = torch.randint(0, 10, (16,))

# 学習ループ
for epoch in range(1, 301):  # 300エポック
    logits = model(x)
    loss = loss_fn(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # fc1.weight の平均を表示
    print(f"Epoch {epoch:02d}: loss={loss.item():.4f}, fc1.weight.mean={model.fc1.weight.mean().item():.6f}")
