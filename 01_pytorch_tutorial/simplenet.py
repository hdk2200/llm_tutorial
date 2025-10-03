import torch
import torch.nn as nn
import torch.nn.functional as F

# ネットワーク定義
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 部品（レイヤー）を用意
        self.fc1 = nn.Linear(10, 5)  # 入力10次元 → 出力5次元 fc: 全結合層（Fully Connected Layer, FC層）

    def forward(self, x):
        # forward: 入力から出力をどう計算するか
        x = self.fc1(x)      # 全結合層に通す
        x = F.relu(x)        # 活性化関数 ReLU(max(0,x)) を適用。 ReLU   https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html#relu
        return x

# モデルを作る
model = SimpleNet()

# ダミーのデータ（2サンプル, 10次元）
data = torch.randn(2, 10)
print("input:",data)

# 順伝播してみる
output = model(data)
print(output)


print("入力サイズ:", data.shape)   # torch.Size([2, 10])
print("出力サイズ:", output.shape) # torch.Size([2, 5])


print("\n------------------")
print("Linear処理の確認")
# 全結合層の重みとバイアスの形状確認
# PyTorch は Xavier 初期化（Kaiming uniform）に基づいてランダムに値を入れる
# 要するにWとbはランダム
layer = nn.Linear(3, 2)  # 入力3次元 → 出力2次元
print("W の形状:", layer.weight.shape)
print("W の値:\n", layer.weight)
print("bias の形状:", layer.bias.shape)
print("bias の値:\n", layer.bias)
print("layer :", layer)

sample_input = torch.randn(4, 3)  # 4サンプル, 3次元
sample_output = layer(sample_input)
print("sample_input :", sample_input)
print("sample_output:", sample_output)
print("sample_input size :", sample_input.shape)   # torch.Size([4, 3])
print("sample_output size:", sample_output.shape)  # torch.Size([4, 2
print("\n")

# Linear処理の確認
# W の形状: torch.Size([2, 3])
# W の値:
#  Parameter containing:
# tensor([[ 0.2008, -0.0412,  0.2203],
#         [-0.4407, -0.0926,  0.4977]], requires_grad=True)
# bias の形状: torch.Size([2])
# bias の値:
#  Parameter containing:
# tensor([-0.3489, -0.1077], requires_grad=True)
# layer : Linear(in_features=3, out_features=2, bias=True)
# sample_input : tensor([[-0.4326,  0.3903,  0.7557],
#         [-0.0172,  0.5180, -2.1547],
#         [-0.0298, -0.7512,  0.3226],
#         [ 0.0854,  0.4420,  0.6635]])
# sample_output: tensor([[-0.2853,  0.4229],
#         [-0.8484, -1.2205],
#         [-0.2529,  0.1355],
#         [-0.2038,  0.1439]], grad_fn=<AddmmBackward0>)
# sample_input size : torch.Size([4, 3])
# sample_output size: torch.Size([4, 2])
