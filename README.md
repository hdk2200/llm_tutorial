# LLM Tutorial

This repository collects small PyTorch experiments ranging from linear models to miniature Transformer blocks.

## プロジェクト構成
- `llm_tutorial_1/`
  - `learn.py`: 3 次元入力に対する線形回帰タスクを `nn.Linear` + `MSELoss` で学習するデモ。
  - `learn_classify.py`: 犬/猫の特徴量を使った 2 クラス分類。Apple Silicon の MPS バックエンドがあれば自動で利用。
  - `learn_class_decision_boundary.py`: 上記分類器の決定境界をメッシュグリッドで可視化し、`images/learn_class_decision_boundary.png` を生成。
  - `minllm.py`: 変換ブロックを組み合わせた小型 GPT 風モデルを定義し、ランダムトークンに対する出力テンソル形状を確認。
  - `simplenet.py`: 単一の全結合層 + ReLU でランダム入力の流れと追加線形層の重み/バイアスを観察。
- `llm_tutorial_mnist/`
  - `mnist.py`: MNIST データセットを使って手書き数字分類器を学習し、損失と精度をレポート。
- `images/`: チュートリアルの可視化結果を保存するディレクトリ。
- `requirements.txt`: PyTorch トリオと Matplotlib を含む最小構成の依存パッケージ一覧。

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

### 例
```bash
cd llm_tutorial_1
python learn_classify.py
```

![犬と猫の分類境界](images/learn_class_decision_boundary.png)




# MNIST

# PyTorch `nn.Linear` まとめ

## 役割
- **線形変換（全結合層, Fully Connected Layer）** を行う層  
- 数式：
  \[
  y = xW^T + b
  \]  
  - \(x\)：入力ベクトル  
  - \(W\)：重み行列（学習で更新される）  
  - \(b\)：バイアス（学習で更新される）  

---

## 定義方法
```python
layer = nn.Linear(in_features, out_features)

in_features : 入力の次元数
out_features : 出力の次元数

```

使用例

```python
self.fc1 = nn.Linear(28*28, 100)  # 784次元 → 100次元
self.fc2 = nn.Linear(100, 10)     # 100次元 → 10クラス

処理の流れ

画像 (28×28) → 1次元ベクトル (784要素) に変換

fc1 : 784 → 100 に圧縮（特徴抽出のイメージ）

活性化関数 (ReLU) を通す → 非線形性を加える

fc2 : 100 → 10 に変換（クラスごとのスコアを出力）

学習で変わるもの

fc1.weight, fc1.bias

fc2.weight, fc2.bias
→ 誤差逆伝播により自動的に更新される

出力の解釈

fc2 の出力 = ロジット（生のスコア）

torch.softmax を適用すると確率として扱える
```


### まとめ

nn.Linear は「入力次元 → 出力次元」の線形変換を行う

内部のパラメータ (W, b) は学習で自動調整される

複数積み重ねて「特徴抽出 → クラス分類」を実現