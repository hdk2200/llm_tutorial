# LLM Tutorial

This repository collects small PyTorch experiments.

## llm_tutorial_1/minllm.py
- Defines a tiny GPT-like model composed of Transformer blocks.
- Each block stacks multi-head self-attention, a position-wise feed-forward MLP, and layer normalization.
- Runs a forward pass on dummy token indices and prints the resulting tensor shape.

## llm_tutorial_1/simplenet.py
- Implements a minimal fully connected network with one linear layer followed by ReLU.
- Executes a forward pass on random input and reports input and output tensor shapes.
- Logs the randomly initialized weights and bias of an extra linear layer and shows sample outputs.
