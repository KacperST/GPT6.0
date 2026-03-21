import torch.nn as nn
import torch.nn.functional as F
import torch
from models.head import Head


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, block_size):
        super().__init__()
        self.head_size = embed_dim // num_heads
        self.wo = nn.Linear(embed_dim, embed_dim)
        self.heads = nn.ModuleList(
            [Head(embed_dim=embed_dim, head_size=self.head_size, block_size=block_size) for _ in range(num_heads)]
        )

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.wo(x)
