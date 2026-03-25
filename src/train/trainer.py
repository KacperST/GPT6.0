from collections.abc import Callable
from models.bigram import BigramLanguageModel
from torch.optim import AdamW
import torch
import torch.nn as nn


def train_model(
    model: nn.Module,
    train_data: torch.Tensor,
    eval_data: torch.Tensor,
    get_batch_fn: Callable,
    max_iters: int,
    eval_fn: Callable,
    learning_rate: float = 1e-4,
    batch_size: int = 64,
    block_size: int = 128,
) -> None:
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    for step in range(max_iters):
        model.train()
        train_x, train_y = get_batch_fn(train_data, batch_size=batch_size, block_size=block_size)
        _, loss = model(train_x, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0 or step == max_iters - 1:
            loss = eval_fn(model, eval_data, get_batch_fn, batch_size=batch_size, block_size=block_size)
            print(f"Step: {step}, loss: {loss}")
