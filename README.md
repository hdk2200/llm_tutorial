# LLM Tutorial

This repository collects small PyTorch experiments ranging from linear models to miniature Transformer blocks.

## Getting Started
- `llm_tutorial_1/requirements.txt` lists the minimal Python packages (PyTorch trio + Matplotlib).
- Set up a virtual environment before running the samples:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r llm_tutorial_1/requirements.txt
```
- Each script is self-contained; run it with `python <script>` to reproduce the printed outputs or plots.

## Experiments

### llm_tutorial_1/minllm.py
- Defines a tiny GPT-like model composed of Transformer blocks with multi-head self-attention and MLP layers.
- Sends random token indices through the network and prints the resulting `(batch, seq, vocab)` tensor shape.

### llm_tutorial_1/simplenet.py
- Implements a single fully connected layer followed by ReLU and inspects how random inputs propagate.
- Logs input/output tensor shapes and samples the weight/bias values of an extra linear layer for intuition.

### llm_tutorial_1/learn.py
- Demonstrates supervised learning for a regression task using `nn.Linear` and `MSELoss`.
- Trains on random 3D inputs with stochastic gradient descent and reports the epoch-wise loss.

### llm_tutorial_1/learn_classify.py
- Builds a minimal 2-class classifier that can leverage Apple Silicon's MPS backend when available.
- Trains on toy "dog vs cat" features, then evaluates single and batched samples while printing logits, probabilities, and predicted classes.

### llm_tutorial_1/learn_class_decision_boundary.py
- Reuses the binary classifier to visualize its decision regions over a mesh grid.
- Saves/uses the following plot to illustrate how the learned boundary separates the toy dataset.

![犬と猫の分類境界](images/learn_class_decision_boundary.png)
