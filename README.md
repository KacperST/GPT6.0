# GPT6.0

A PyTorch implementation of a **GPT-inspired Language Model** featuring self-attention mechanisms. This project demonstrates fundamental concepts of modern neural language modeling, including tokenization, self-attention blocks, multi-head attention, and text generation.

## Features

- 🎯 **GPT Architecture** - transformer-based language model with self-attention
- 🧠 **Self-Attention Mechanism** - single-head attention for context understanding
- 👥 **Multi-Head Attention** - parallel attention heads for diverse feature extraction
- 🔤 **Character-level Tokenizer** - tokenization at the individual character level
- 🎲 **Text Generation** - creating new text sequences based on the trained model

## Project Structure

```
src/
├── main.py                      # Entry point – main pipeline
├── data/
│   ├── dataset.py               # Data loading and 
│   └── tokenizer.py             # Encoder/decoder
├── models/
│   ├── gpt.py                   # GPT Language Model architecture
│   ├── self_attention.py         # Single-head self-attention mechanism
│   └── multi_head_attention.py   # Multi-head attention blocks
└── train/
    ├── trainer.py               # Training loop
    └── evaluate.py              # Model evaluation
```

## Requirements

- Python ≥ 3.12
- PyTorch ≥ 2.10.0
- NumPy ≥ 2.4.3

## Installation

Create a virtual environment and install dependencies using `uv`:

```bash
# Clone or download the project
cd GPT6

# Create virtual environment with uv
uv venv

# Activate the environment
source .venv/bin/activate

# Sync dependencies
uv sync
```

## Usage

Run the main script to:
1. Load training data
2. Build tokenizer
3. Train the model
4. Generate new text

```bash
uv run src/main.py
```
## How It Works

1. **Tokenization**: Text is converted to a sequence of character indices
2. **Embedding**: Tokens are embedded into a continuous vector space with positional encodings
3. **Self-Attention**: Single-head attention layer captures relationships between tokens
4. **Multi-Head Attention**: Multiple attention heads learn different attention patterns in parallel
5. **Output Projection**: Attention output is projected to vocabulary size for next token prediction
6. **Loss**: Cross-entropy loss computed between predictions and actual tokens
7. **Backprop**: Gradients update model weights
8. **Generation**: Model autoregressively generates new tokens based on previous ones using attention context

## Notes

- This is a **simplified** implementation for educational purposes
- Features self-attention mechanisms inspired by the Transformer architecture
- Multi-head attention improves model capacity by learning diverse attention patterns

