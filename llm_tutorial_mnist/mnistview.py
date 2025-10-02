import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# 前処理（Tensor変換のみ）
transform = transforms.ToTensor()

# MNIST テストデータセット
testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# DataLoaderで取り出す
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)


# バッチからデータを取り出す
images, labels = next(iter(testloader))

# 4枚まとめて表示
fig, axes = plt.subplots(1, 4, figsize=(8, 2))
for i in range(4):
    axes[i].imshow(images[i].squeeze(), cmap="gray")  # 1chなのでsqueeze()
    axes[i].set_title(f"Label: {labels[i].item()}")
    axes[i].axis("off")

plt.show()

