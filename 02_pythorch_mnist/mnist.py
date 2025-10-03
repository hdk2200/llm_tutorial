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

print("訓練データ:", len(train_dataset))  
print("テストデータ:", len(test_dataset)) 

image, label = train_dataset[0]
print("1枚の画像サイズ:", image.shape)  # torch.Size([1, 28, 28])
print("ラベル:", label)                  # 例: 5

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# -------------------------
# 2. モデル定義  MLP（多層パーセプトロン, Multi-Layer Perceptron） を使った方法
# -------------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 100)  
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # nn.Linear の線形変換 を使って
        # 784 (ピクセル) → 100 (隠れ特徴) → 10 (クラススコア)
        # に変換するモデル
        # nn.Linearの

        # print ("入力サイズ:", x.shape)  # 例: torch.Size([64, 1, 28, 28]) 
        x = x.view(-1, 28*28)   # 28x28画像 → ベクトル。画像の2次元構造（28×28）は無視して、28×28 の画像を 1次元ベクトル（784要素の並び） に変える
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet().to(device)  # モデルをMPSに移動

# -------------------------
# 3. 損失関数 & 最適化
# -------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("model.parameters:",model.parameters())


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


# -------------------------
# 6. モデルの保存
# -------------------------
torch.save(model.state_dict(), "mnist_mlp.pth")
print("学習済みモデルを保存しました: mnist_mlp.pth")

