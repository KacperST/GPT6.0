import torch
import torch.nn as nn

@torch.no_grad()
def evaluate_model(model: nn.Module, data, get_batch_fn: callable, max_eval_steps: int = 100):
    model.eval()
    losses = []
    for step in range(max_eval_steps):
        X, y = get_batch_fn(data)
        _, loss = model(X, y)
        losses.append(loss)
    model.train()
    return sum(losses) / len(losses)

    