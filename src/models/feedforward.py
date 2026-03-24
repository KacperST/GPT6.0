import torch
import torch.nn as nn
from torch.nn import functional as F


class FeedForward(nn.Module):

    def __init__(self, embed_dim: int):
        super().__init__()
        self.w1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.w2 = nn.Linear(4 * embed_dim, embed_dim)
        self.w2.SCALE_INIT = 1
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.w1(x)
        x = F.gelu(x)
        logits = self.w2(x)
        return self.dropout(logits)
