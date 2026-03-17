from models.bigram import BigramLanguageModel
from torch.optim import AdamW
import torch.nn as nn

def train_model(model: nn.Module, data, get_batch_fn: callable, max_iters, eval_fn, learning_rate: float= 1e-4):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    for step in range(max_iters):
        model.train()
        train_x, train_y = get_batch_fn(data)
        _, loss = model(train_x, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            loss = eval_fn(model, data, get_batch_fn)
            print(f"Step: {step}, loss: {loss}")

