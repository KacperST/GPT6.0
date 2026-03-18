# GPT6.0

A compact implementation of a **Bigram Language Model** in PyTorch. This project demonstrates fundamental concepts of natural language modeling, including tokenization, training neural language models, and text generation.

## Features

- 🎯 **Bigram Language Model** - a simple yet effective model for predicting the next token based on the previous one
- 🔤 **Character-level Tokenizer** - tokenization at the individual character level
- 🚂 **Training with AdamW** - model optimization using a modern optimizer
- 📊 **Model Evaluation** - performance assessment on a test set
- 🎲 **Text Generation** - creating new text sequences based on the trained model

## Project Structure

```
src/
├── main.py                # Entry point – main pipeline
├── data/
│   ├── dataset.py         # Data loading and preparation
│   └── tokenizer.py       # Encoder/decoder
├── models/
│   └── bigram.py          # Bigram Language Model architecture
└── train/
    ├── trainer.py         # Training loop
    └── evaluate.py        # Model evaluation
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
2. **Forward Pass**: Model takes token sequence and predicts next token
3. **Loss**: Cross-entropy loss computed between predictions and actual tokens
4. **Backprop**: Gradients update model weights
5. **Generation**: Model autoregressively generates new tokens based on previous ones

## Notes

- This is a **simplified** implementation for educational purposes
- Bigram model represents the simplest form of attention-free language modeling
- Will be extended to attention mechanisms in near future versions

