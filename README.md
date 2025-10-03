# LLM Tutorial

PyTorch を使った小さな機械学習/LLM 実験をまとめたリポジトリです。線形回帰から簡易的な Transformer ブロックまで、最小限のコードで確認できるサンプルを収録しています。

## プロジェクト構成
- `01_pytorch_tutorial/`
  - `learn.py`: 3 次元入力に対する線形回帰タスクを `nn.Linear` + `MSELoss` で学習するデモ。
  - `learn_classify.py`: 犬/猫の特徴量を使った 2 クラス分類。Apple Silicon の MPS バックエンドがあれば自動で利用。
  - `learn_class_decision_boundary.py`: 上記分類器の決定境界をメッシュグリッドで可視化し、`images/learn_class_decision_boundary.png` を生成。
  - `minllm.py`: 変換ブロックを組み合わせた小型 GPT 風モデルを定義し、ランダムトークンに対する出力テンソル形状を確認。
  - `simplenet.py`: 単一の全結合層 + ReLU でランダム入力の流れと追加線形層の重み/バイアスを観察。
- `02_pythorch_mnist/`
  - `mnist.py`: MNIST データセットを使って手書き数字分類器を学習し、損失と精度をレポート。
- `images/`: チュートリアルの可視化結果を保存するディレクトリ。
- `requirements.txt`: PyTorch と Matplotlib を含む最小構成の依存パッケージ一覧。

## セットアップ
1. 仮想環境を作成して有効化します。
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. 依存関係をインストールします。
   ```bash
   pip install -r requirements.txt
   ```

## 実行方法
各スクリプトは自己完結型なので、目的のサブフォルダに移動して `python <script>` を実行するだけで、出力や可視化を再現できます。

### 実行例
```bash
cd 01_pytorch_tutorial
python learn_classify.py
```

![犬と猫の分類境界](images/learn_class_decision_boundary.png)

## 付録: `nn.Linear` のおさらい
`nn.Linear` は入力ベクトルに対して線形変換（全結合層）を適用する層です。パラメータは学習によって更新され、非線形活性化関数と組み合わせることで多層パーセプトロンなどを構成できます。

```python
layer = nn.Linear(in_features, out_features)
```

- `in_features`: 入力の次元数
- `out_features`: 出力の次元数
- 出力テンソルは \(y = xW^T + b\) で計算されます。

MNIST 例では次のように使用しています。

```python
self.fc1 = nn.Linear(28 * 28, 100)
self.fc2 = nn.Linear(100, 10)
```

1. 画像 (28×28) を 784 次元ベクトルにフラット化
2. `fc1` で特徴抽出（784 → 100）
3. ReLU で非線形性を付与
4. `fc2` でクラスごとのロジット（100 → 10）を計算

`fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias` は誤差逆伝播によって自動的に更新されます。`torch.softmax` を用いればロジットを確率として解釈できます。
