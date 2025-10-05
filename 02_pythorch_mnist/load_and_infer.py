import argparse
import math

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load MNIST model and run inference on test images")
    parser.add_argument("-s", "--start-index", type=int, default=0, help="予測を開始するテストデータのインデックス")
    parser.add_argument("-n", "--count", type=int, default=20, help="表示する画像の枚数")
    return parser.parse_args()


args = parse_args()

if args.count <= 0:
    raise ValueError("--count は 1 以上を指定する必要があります")

if args.start_index < 0 or args.start_index >= len(test_dataset):
    raise ValueError("--start-index がデータセットの範囲外です")

end_index = min(args.start_index + args.count, len(test_dataset))
indices = list(range(args.start_index, end_index))
samples = [(test_dataset[i], i) for i in indices]

model.eval()

cols = 10
rows = math.ceil(len(samples) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.8))

if rows == 1:
    axes = axes.reshape(-1)
else:
    axes = axes.flatten()

for ax in axes[len(samples):]:
    ax.axis("off")

correct = 0
incorrect = 0

for idx, (data, img_index) in enumerate(samples):
    image, label = data
    with torch.no_grad():
        # 画像を (1,1,28,28) に変形してデバイスへ
        img_input = image.unsqueeze(0).to(device)
        output = model(img_input)
        pred = output.argmax(dim=1).item()

    # 結果を表示
    axes[idx].imshow(image.squeeze(0), cmap="gray")
    if pred == label:
        correct += 1
        title_color = "royalblue"
    else:
        incorrect += 1
        title_color = "crimson"
    axes[idx].set_title(f"#{img_index} T:{label}\nP:{pred}", fontsize=9, color=title_color)
    axes[idx].axis("off")

summary = f"Correct: {correct}  Incorrect: {incorrect}"
fig.suptitle(summary, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
print(summary)

plt.show()
