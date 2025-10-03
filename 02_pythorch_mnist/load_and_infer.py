import torch
import torch.nn as nn
from torchvision import datasets, transforms

# -------------------------
# 0. デバイス設定
# -------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("使用デバイス:", device)

# -------------------------
# 1. モデル定義（学習時と同じ構造）
# -------------------------
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

model = SimpleNet().to(device)

# -------------------------
# 2. モデルのロード
# -------------------------
model.load_state_dict(torch.load("mnist_mlp.pth", map_location=device))
model.eval()
print("学習済みモデルをロードしました")

# -------------------------
# 3. データ準備（テストデータ）
# -------------------------
transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# -------------------------
# サンプルを複数取り出す
# -------------------------
import matplotlib.pyplot as plt

# 先頭の10枚を取得
samples = [test_dataset[i] for i in range(20)]

model.eval()
fig, axes = plt.subplots(1, 20, figsize=(15, 2))

for idx, (image, label) in enumerate(samples):
    with torch.no_grad():
        # 画像を (1,1,28,28) に変形してデバイスへ
        img_input = image.unsqueeze(0).to(device)
        output = model(img_input)
        pred = output.argmax(dim=1).item()

    # 結果を表示
    axes[idx].imshow(image.squeeze(0), cmap="gray")
    axes[idx].set_title(f"T:{label}\nP:{pred}", fontsize=10)  # T=正解, P=予測
    axes[idx].axis("off")

plt.show()
