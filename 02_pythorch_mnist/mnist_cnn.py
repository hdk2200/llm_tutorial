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
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

print("訓練データ:", len(train_dataset))
print("テストデータ:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# -------------------------
# 2. モデル定義  CNNバージョン
# -------------------------
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # | 引数                | 意味                                      |
            # | ----------------- | --------------------------------------- |
            # | `in_channels=1`   | 入力は1チャンネル（白黒画像）                         |
            # | `out_channels=32` | 出力は32チャンネル（＝32個のフィルター）                  |
            # | `kernel_size=3`   | 各フィルターは 3×3 の領域を見て特徴を抽出                 |
            # | `padding=1`       | 端の情報を保つために周囲を1ピクセル分0で囲む（出力サイズが入力と同じになる） |
            

            # Conv2d 「画像を、複数の“意味ある特徴”に分解する層」
            
            # 1x28x28 -> 32x28x28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # ０以上を抽出（負の値を除去し、反応を強調）

            # 32x28x28 -> 32x14x14
            nn.MaxPool2d(2), # 2×2の範囲で最大値を取る → 画像を1/2に縮小 計算量を減らす  特徴の「要約」：位置のずれに強くする 全体の中で一番目立つ部分だけを見て、細かい部分は忘れる
            
            # 32x14x14 -> 64x14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),# ０以上を抽出（負の値を除去し、反応を強調）

            # 64x14x14 -> 64x7x7
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),# 64×7×7 → 3136  CNNで抽出した64枚の特徴マップを1本の長いベクトルに並べる。
            nn.Linear(64 * 7 * 7, 128),      # 3136 → 128 画像全体の特徴を128次元に圧縮。重要な特徴だけを残し、次段に渡す。
            nn.ReLU(),                     # 負の値を0にして、重要な特徴を強調。
            nn.Linear(128, 10),             # 128 → 10 最終出力。10クラス（数字 0〜9）それぞれのスコア（確率前）を出力。
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = ConvNet().to(device)


# -------------------------
# 3. 損失関数 & 最適化
# -------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# -------------------------
# 4. 学習ループ
# -------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


# -------------------------
# 5. テスト
# -------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")


# -------------------------
# 6. モデルの保存
# -------------------------
torch.save(model.state_dict(), "mnist_cnn.pth")
print("学習済みモデルを保存しました: mnist_cnn.pth")

