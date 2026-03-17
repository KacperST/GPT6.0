import requests
from typing import List, Tuple
import torch
from .tokenizer import Tokenizer, CharTokenizer

def fetch_raw_data(url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def save_raw_data(data: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(data)

def tokenize_data(data:str, tokenizer: Tokenizer, to_tensor: bool = True):
    tokens = tokenizer.encode(data)
    return  torch.tensor(tokens, dtype=torch.long) if to_tensor else tokens

def get_batch(data, batch_size: int = 4, block_size: int = 8) -> Tuple:
    random_ints = torch.randint(0, len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in random_ints])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_ints])
    return x, y

