import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNormalization(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(embed_dim))
        self.b = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5

    def forward(self, x):
        layer_mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        logits = (self.g / (std + self.eps)) * (x - layer_mean) + self.b
        return logits
