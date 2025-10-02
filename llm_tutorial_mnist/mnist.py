import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------------
# 0. デバイス設定 (MPS or CPU)
# -------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("使用デバイス:", device)

# -------------------------
# 1. データ準備
# -------------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# -------------------------
# 2. モデル定義 (MLP)
# -------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)   # 28x28画像 → ベクトル
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet().to(device)  # モデルをMPSに移動

# -------------------------
# 3. 損失関数 & 最適化
# -------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# -------------------------
# 4. 学習ループ
# -------------------------
for epoch in range(3):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)  # データをMPSへ
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------
# 5. テスト
# -------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)  # テストデータもMPSへ
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

print(f"Test Accuracy: {correct/total:.4f}")
