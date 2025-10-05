import torch
import torch.nn as nn
import math

import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 0. デバイス設定
# -------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("使用デバイス:", device)

# =========================
# 1. 単語辞書と文章データ
# =========================
vocab = {
    "今日は": 0,
    "いい": 1,
    "天気": 2,
    "です": 3,
    "最悪": 4,
    "ひどい": 5,
    "ね": 6,
    "！": 7,
    "気分": 8,
    "が": 9,
    "な": 10,
    "の": 11,     
    "日": 12,     
    "最高": 13,   
}
vocab_size = len(vocab)
pad_id = vocab_size   # padding用ID
vocab_size += 1       # PADを含めた語彙サイズ

# =========================
# データセット（ポジティブ / ネガティブ）
# =========================
sentences = [
    # --- ポジティブ ---
    ["今日は", "いい", "天気", "です"],      # 1
    ["いい", "天気", "！"],                 # 2
    ["いい", "天気", "です", "ね"],          # 3
    ["今日は", "気分", "が", "いい"],        # 4
    ["最高", "の", "天気", "です"],          # 5
    ["いい", "気分", "です"],                # 6

    # --- ネガティブ ---
    ["最悪", "です", "ね"],                  # 7
    ["ひどい", "です"],                      # 8
    ["今日は", "最悪", "です"],              # 9
    ["ひどい", "天気", "です"],              # 10
    ["今日は", "最悪", "な", "天気", "です"], # 11
    ["ひどい", "気分", "です"],              # 12
    ["今日は", "気分", "が", "最悪"],        # 13
    ["今日は", "ひどい", "日", "です"],      # 14
]

# =========================
# ラベル: 1 = ポジティブ, 0 = ネガティブ
# =========================
labels = torch.tensor([
    1, 1, 1, 1, 1, 1,   # ← ポジティブ6文
    0, 0, 0, 0, 0, 0, 0, 0  # ← ネガティブ8文
], device=device)

max_len = max(len(s) for s in sentences)
input_ids = []
for s in sentences:
    ids = [vocab[w] for w in s]
    ids += [pad_id] * (max_len - len(ids))  # padding
    input_ids.append(ids)

x = torch.tensor(input_ids, device=device)
print("入力形状:", x.shape)

# =========================
# 2. Positional Encoding定義
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, D)
        x = x + self.pe[:, :x.size(1)]
        return x

# =========================
# 3. モデル定義
# =========================
class TransformerTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_heads=4, num_layers=2, num_classes=2, pad_id=None):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embed(x)               # (B, L, D)
        emb = self.pos_enc(emb)           # 位置エンコーディングを加算
        mask = (x == self.pad_id)         # PAD位置をマスク
        out = self.encoder(emb, src_key_padding_mask=mask)
        out = out.mean(dim=1)             # 平均プーリングで文全体の特徴
        return self.fc(out)

# =========================
# 4. 学習
# =========================
model = TransformerTextClassifier(vocab_size, pad_id=pad_id).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        preds = out.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        print(f"Epoch {epoch+1:3d} | loss {loss.item():.4f} | acc {acc:.2f}")

torch.save(model.state_dict(), "model_weights.pth")

# =========================
# 5. 推論（新しい文章を分類）
# =========================
def encode_sentence(sentence, vocab, pad_id, max_len):
    ids = [vocab.get(w, pad_id) for w in sentence]
    ids += [pad_id] * (max_len - len(ids))
    return torch.tensor(ids).unsqueeze(0).to(device)

model.eval()
new_sentences = [
    ["今日は", "いい", "天気", "！"],       # ポジティブ
    ["今日は", "ひどい", "天気", "です"],   # ネガティブ
    ["最悪", "です", "！"],                 # ネガティブ
    ["気分", "が", "いい"],                 # ポジティブ
]

with torch.no_grad():
    for s in new_sentences:
        x_new = encode_sentence(s, vocab, pad_id, max_len)
        pred = model(x_new).argmax(dim=1).item()
        label = "ポジティブ" if pred == 1 else "ネガティブ"
        print(f"{' '.join(s)} → 予測: {label}")



# =========================
# Self-Attentionを抽出できるよう改良
# =========================
class InspectableEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        # Transformerの内部attentionをhookして保存
        x, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False
        )
        self.attn_weights = attn_weights.detach().cpu()  # shape: [num_heads, batch, seq_len, seq_len]
        return self.dropout1(x)

# =========================
# モデルを差し替え（Encoder層をInspectable版に）
# =========================
def make_inspectable_model():
    embed_dim = 32
    num_heads = 4
    num_layers = 2
    model = TransformerTextClassifier(vocab_size, embed_dim=embed_dim, num_heads=num_heads,
                                      num_layers=num_layers, pad_id=pad_id)
    # Encoder層を差し替え
    model.encoder.layers = nn.ModuleList([
        InspectableEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                dim_feedforward=128, dropout=0.1, batch_first=True)
        for _ in range(num_layers)
    ])
    return model.to(device)

# あなたの既存のモデルを読み替える
model = make_inspectable_model()
model.load_state_dict(torch.load("model_weights.pth", map_location=device))  # 学習済み重みをロード
model.eval()

# =========================
# Attentionの可視化
# =========================
def visualize_attention(sentence_tokens):
    x_new = encode_sentence(sentence_tokens, vocab, pad_id, max_len)
    with torch.no_grad():
        _ = model(x_new)  # 推論実行（内部でattentionを保存）
        attn = model.encoder.layers[0].attn_weights  # 1層目のattention取得 [num_heads, batch, seq, seq]
        attn = attn.mean(dim=0)[0].numpy()  # 平均して1枚にする

    tokens = sentence_tokens + ["PAD"] * (max_len - len(sentence_tokens))
    plt.figure(figsize=(5,5))
    plt.imshow(attn, cmap="YlOrRd")
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.colorbar()
    plt.title("Self-Attention Heatmap")
    plt.show()

# =========================
# 実際に可視化してみる
# =========================
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'  # ← macOSの日本語フォント
matplotlib.rcParams['axes.unicode_minus'] = False     # マイナス記号の表示対策

visualize_attention(["今日は", "ひどい", "天気", "です"])


def visualize_multihead_attention(sentence_tokens):
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Hiragino Sans'  # mac日本語フォント
    matplotlib.rcParams['axes.unicode_minus'] = False

    x_new = encode_sentence(sentence_tokens, vocab, pad_id, max_len)
    with torch.no_grad():
        _ = model(x_new)
        attn = model.encoder.layers[0].attn_weights  # shape: [num_heads, batch, seq, seq]
        attn = attn[:, 0].numpy()  # 各headごとに1文の重みを取り出す

    tokens = sentence_tokens + ["PAD"] * (max_len - len(sentence_tokens))
    num_heads = attn.shape[0]

    plt.figure(figsize=(4 * num_heads, 4))
    for h in range(num_heads):
        plt.subplot(1, num_heads, h + 1)
        plt.imshow(attn[h], cmap="YlOrRd")
        plt.xticks(range(len(tokens)), tokens, rotation=45)
        plt.yticks(range(len(tokens)), tokens)
        plt.title(f"Head {h+1}")
    plt.tight_layout()
    plt.show()


visualize_multihead_attention(["今日は", "ひどい", "天気", "です"])
