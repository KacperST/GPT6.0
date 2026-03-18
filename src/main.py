import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from data.dataset import fetch_raw_data, get_batch, tokenize_data
from data.tokenizer import CharTokenizer
from models.bigram import BigramLanguageModel
from train.trainer import train_model
from train.evaluate import evaluate_model
from models.gpt import GPT

BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 5000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-3
MAX_NEW_TOKENS = 100
EMBEDING_DIM = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)


def main():
    raw_data = fetch_raw_data()
    vocabulary = sorted(set(raw_data))
    tokenizer = CharTokenizer(vocabulary)
    data = tokenize_data(data=raw_data, tokenizer=tokenizer, to_tensor=True).to(DEVICE)
    model = GPT(len(vocabulary), EMBEDING_DIM, BLOCK_SIZE)
    model = model.to(device=DEVICE)
    train_model(model, data, get_batch, MAX_ITERS, evaluate_model, learning_rate=1e-3)
    idx_first = torch.zeros(1, 1, dtype=torch.long).to(DEVICE)
    generated_tokens = model.generate(idx_first, max_new_tokens=MAX_NEW_TOKENS)
    output = tokenizer.decode(generated_tokens.tolist()[0])
    print(f"\nGenerated sequence:{output}")


if __name__ == "__main__":
    main()
"""
OUTPUT
Step: 0, loss: 4.286160469055176
Step: 500, loss: 3.2832372188568115
Step: 1000, loss: 3.1075868606567383
Step: 1500, loss: 2.955082416534424
Step: 2000, loss: 2.749258041381836
Step: 2500, loss: 2.6402783393859863
Step: 3000, loss: 2.6137008666992188
Step: 3500, loss: 2.571695566177368
Step: 4000, loss: 2.5662286281585693
Step: 4500, loss: 2.571314811706543
Step: 4999, loss: 2.5537493228912354

Generated sequence:
SI:
Thutour; pe thy, wresmyo ff h

Aghe carveldro:
An mof KVEEGOLE:
We QSwhir, IOULroun qs imie wham
"""
