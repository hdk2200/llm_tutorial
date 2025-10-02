import torch
import torch.nn as nn
import torch.nn.functional as F

# ハイパーパラメータ
vocab_size = 1000    # 語彙数
embed_dim = 64       # 埋め込み次元
num_heads = 4        # Attentionのヘッド数
num_layers = 2       # Transformerブロック数
seq_len = 32         # 最大系列長

# シンプルトランスフォーマーブロック

# PyTorchのニューラルネットワーク用モジュール
class TransformerBlock(nn.Module):
    # TODO:引数説明コメント torch.nn PyTorch ニューラルネットワーク用モジュール
    
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

# GPTライクのモデル
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        x = self.embed(idx).transpose(0, 1)  # (seq, batch, dim)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits.transpose(0, 1)  # (batch, seq, vocab)

# モデルとダミーデータ
model = MiniGPT()

# TODO:入力説明コメント 
# torch.randint — PyTorch 2.8 documentation
# https://docs.pytorch.org/docs/stable/generated/torch.randint.html
x = torch.randint(0, vocab_size, (2, seq_len))  # ダミー入力

logits = model(x)

print("出力の形状:", logits.shape)  # (バッチ, シーケンス長, 語彙数)
