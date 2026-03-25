import torch
import torch.nn as nn
from collections.abc import Callable


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: torch.Tensor,
    get_batch_fn: Callable,
    max_eval_steps: int = 100,
    batch_size: int = 64,
    block_size: int = 128,
) -> torch.Tensor:
    model.eval()
    losses = []
    for _ in range(max_eval_steps):
        X, y = get_batch_fn(data, batch_size=batch_size, block_size=block_size)
        _, loss = model(X, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)
