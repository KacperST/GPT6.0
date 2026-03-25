import torch
import torch.nn as nn
from torch.nn import functional as F
from models.feedforward import FeedForward
from models.normalization import LayerNormalization
from models.multi_head_attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int):
        super().__init__()
        self.sa_attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, block_size=block_size)
        self.ln1 = LayerNormalization(embed_dim=embed_dim)
        self.ff = FeedForward(embed_dim=embed_dim)
        self.ln2 = LayerNormalization(embed_dim=embed_dim)

    def forward(self, x):
        x = x + self.sa_attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
