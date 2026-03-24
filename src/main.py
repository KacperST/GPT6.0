import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from data.dataset import fetch_raw_data, get_batch, tokenize_data
from data.tokenizer import CharTokenizer, GPTTokenizer
from models.bigram import BigramLanguageModel
from train.trainer import train_model
from train.evaluate import evaluate_model
from models.gpt import GPT

BATCH_SIZE = 64
BLOCK_SIZE = 128
MAX_ITERS = 4000
EVAL_INTERVAL = 50
LEARNING_RATE = 3e-4
MAX_NEW_TOKENS = 100
EMBEDING_DIM = 128
VOCAB_SIZE = 50257
NUM_HEADS = 8
NUM_DECODER_BLOCKS = 6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(1337)


def main():
    raw_data = fetch_raw_data()
    tokenizer = GPTTokenizer()
    data = tokenize_data(data=raw_data, tokenizer=tokenizer, to_tensor=True)
    n = int(0.8 * len(data))
    train_data, val_data = data[:n], data[n:]
    model = GPT(VOCAB_SIZE, EMBEDING_DIM, BLOCK_SIZE, num_heads=NUM_HEADS, num_decoder_blocks=NUM_DECODER_BLOCKS)
    model = model.to(device=DEVICE)
    train_model(model, train_data, val_data, get_batch, MAX_ITERS, evaluate_model, learning_rate=LEARNING_RATE)
    idx_first = torch.zeros(1, 1, dtype=torch.long).to(DEVICE)
    generated_tokens = model.generate(idx_first, max_new_tokens=MAX_NEW_TOKENS)
    output = tokenizer.decode(generated_tokens.tolist()[0])
    print(f"\nGenerated sequence:{output}")


if __name__ == "__main__":
    main()
"""
OUTPUT - SELF ATTENTION (single head)
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

"""
OUTPUT - SELF ATTENTION (8 heads)
Step: 0, loss: 4.179450511932373
Step: 500, loss: 2.9266843795776367
Step: 1000, loss: 2.6438372135162354
Step: 1500, loss: 2.6479485034942627
Step: 2000, loss: 2.535628080368042
Step: 2500, loss: 2.5052449703216553
Step: 3000, loss: 2.5269439220428467
Step: 3500, loss: 2.4310178756713867
Step: 4000, loss: 2.4090628623962402
Step: 4500, loss: 2.4429306983947754
Step: 4999, loss: 2.407313346862793

Generated sequence:
MLO:
AThe ban lome to thas shet than,
Yofst's,
Nt;
NI cer shrouteor yinth his ceish eatrens o to Puls soem

Pane thir,

MOTORIY:
JI
WUn:
Na'er bngaur Youinr momly will
Lhaige by te.



B! ist.

Whing,
&in fone tharl bur, teny hay wheti n yoiatw mysi thak ghou srad of haren eie shive beglen'd:
SiR CA
"""
