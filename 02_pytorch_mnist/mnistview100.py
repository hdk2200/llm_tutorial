import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# データセット読み込み
transform = transforms.ToTensor()
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 全部取り出し
images = torch.stack([img for img, _ in testset])  # 形 (10000, 1, 28, 28)

# 例：最初の100枚を表示する
grid = make_grid(images[:100], nrow=10, padding=2)  # 10x10 に並べる

plt.figure(figsize=(10,10))
plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
plt.axis("off")
plt.show()
