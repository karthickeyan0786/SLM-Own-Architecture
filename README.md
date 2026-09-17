# SLM-Own-Architecture
# GPT-Style Small Language Model (SLM) from Scratch

A decoder-only Transformer-based Small Language Model (SLM) implemented entirely from scratch using PyTorch.

This project was built to understand the internal workings of modern Large Language Models such as GPT, LLaMA, Mistral, and Phi-3 by implementing the core Transformer architecture without relying on high-level Hugging Face model classes.

---

## Project Overview

This project implements a GPT-style autoregressive language model capable of learning language patterns through next-token prediction.

The model includes:

- GPT-2 Byte Pair Encoding (BPE) Tokenizer
- Token Embeddings
- Positional Embeddings
- Multi-Head Causal Self-Attention
- Feed Forward Neural Networks (MLP)
- Layer Normalization
- Residual Connections
- Dropout Regularization
- Mixed Precision Training
- Learning Rate Warmup
- Gradient Accumulation
- Autoregressive Text Generation

The objective of this project was educational and research-focused, helping understand how modern Transformer-based language models function internally.

---

## Architecture

Input Text
↓
GPT-2 Tokenizer
↓
Token IDs
↓
Token Embedding Layer
↓
Positional Embedding Layer
↓
Transformer Block × 6
    ├── LayerNorm
    ├── Multi-Head Self Attention
    ├── Residual Connection
    ├── LayerNorm
    ├── Feed Forward Network
    └── Residual Connection
↓
Final LayerNorm
↓
Language Modeling Head
↓
Next Token Prediction

---

## Model Configuration

```python
GPTConfig(
    vocab_size = 50257,
    block_size = 128,
    n_layer = 6,
    n_head = 6,
    n_embd = 384,
    dropout = 0.1,
    bias = True
)
```

### Model Statistics

| Component | Value |
|------------|--------|
| Vocabulary Size | 50,257 |
| Context Length | 128 Tokens |
| Transformer Blocks | 6 |
| Attention Heads | 6 |
| Embedding Dimension | 384 |
| Dropout | 0.1 |
| Parameters | ~30 Million |

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Tiktoken
- Hugging Face Datasets
- Google Colab
- CUDA (T4 GPU)

---

## Features Implemented

### Tokenization

Uses GPT-2 Byte Pair Encoding (BPE) tokenizer through Tiktoken.

```python
enc = tiktoken.get_encoding("gpt2")
```

---

### Multi-Head Causal Self Attention

Implemented from scratch using:

- Query (Q)
- Key (K)
- Value (V)

with causal masking to prevent future token leakage.

---

### Feed Forward Network (MLP)

```python
Linear
↓
GELU
↓
Linear
↓
Dropout
```

Used after every attention layer for feature transformation.

---

### Layer Normalization

Implemented custom LayerNorm module for training stability and gradient flow.

---

### Residual Connections

Skip connections were used throughout the Transformer architecture to improve optimization and prevent gradient degradation.

---

### Mixed Precision Training

Implemented Automatic Mixed Precision (AMP) for faster GPU training and lower memory consumption.

---

### Gradient Accumulation

Supports large effective batch sizes without requiring excessive GPU memory.

---

## Training Configuration

```python
learning_rate = 1e-4

max_iters = 40000

warmup_steps = 1000

batch_size = 32

block_size = 128

gradient_accumulation_steps = 32
```

---

## Training Objective

The model is trained using Next Token Prediction.

Example:

Input:

```text
The crop yield depends on
```

Target:

```text
rainfall
```

The model learns language patterns by predicting the next token in a sequence.

---

## Dataset

The training dataset was converted into GPT-2 token IDs and stored in binary format:

```text
train.bin
validation.bin
```

### Dataset Availability

The dataset files are not included in this repository because they exceed GitHub's file size limitations.

Required files:

```text
train.bin
validation.bin
```

Users can create their own datasets following the preprocessing pipeline provided in the notebook.

---

## Repository Structure

```text
.
├── model.py
├── train.py
├── generate.py
├── tokenizer.py
├── train.bin
├── validation.bin
├── best_model_params.pt
├── README.md
└── notebooks/
```

---

## Training

Run:

```bash
python train.py
```

The best model checkpoint is automatically saved as:

```text
best_model_params.pt
```

---

## Text Generation

Example:

```python
prompt = "Artificial Intelligence"

generated_text = model.generate(
    prompt,
    max_new_tokens=100
)
```

---

## Concepts Learned

This project helped understand:

- Transformer Architecture
- Self Attention Mechanism
- Multi Head Attention
- Causal Masking
- Positional Encoding
- Layer Normalization
- Residual Connections
- Feed Forward Networks
- Cross Entropy Loss
- Gradient Accumulation
- Learning Rate Scheduling
- Mixed Precision Training
- GPT-style Text Generation

---

## Future Improvements

Planned enhancements:

- Rotary Positional Embeddings (RoPE)
- Flash Attention
- SwiGLU Feed Forward Layers
- Larger Context Windows
- Agriculture Domain Fine-Tuning
- Crop Yield Explanation Generation
- Instruction Tuning
- Quantization for Deployment

---

## Author

M. Karthickeyan

AI & Data Science Undergraduate  
Chennai Institute of Technology

GitHub:
https://github.com/karthickeyan0786

LinkedIn:
https://linkedin.com/in/karthickeyan-m-51052037b

---

## Disclaimer

This project was created for educational and research purposes to understand the internal implementation of GPT-style Transformer models. It is not intended to compete with production-scale models such as GPT-4, LLaMA, Mistral, or Phi-3.
