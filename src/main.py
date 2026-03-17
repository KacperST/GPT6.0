import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from data.dataset import fetch_raw_data, get_batch, tokenize_data
from data.tokenizer import CharTokenizer
from models.bigram import BigramLanguageModel
from train.trainer import train_model
from train.evaluate import evaluate_model


BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 1000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-3
MAX_NEW_TOKENS = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
torch.manual_seed(1337)

def main():
    raw_data = fetch_raw_data()
    vocabulary = sorted(set(raw_data))
    tokenizer = CharTokenizer(vocabulary)
    data = tokenize_data(data=raw_data, tokenizer=tokenizer, to_tensor=True)
    model = BigramLanguageModel(len(vocabulary))
    train_model(model, data, get_batch, MAX_ITERS,evaluate_model, learning_rate=1e-3)
    idx_first = torch.zeros(1,1, dtype=torch.long)
    generated_tokens = model.generate(idx_first,max_new_tokens=MAX_NEW_TOKENS)
    output = tokenizer.decode(generated_tokens.tolist()[0])
    print(f"\nGenerated sequence:{output}")

if __name__ == "__main__":
    main()


 


