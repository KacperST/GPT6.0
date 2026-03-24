import torch
import torch.nn as nn
from torch.nn import functional as F

class FeedForward(nn.Module):

  def __init__(self, embed_dim: int):
    super().__init__()
    self.w1 = nn.Linear(embed_dim, 4 * embed_dim)
    self.w2 = nn.Linear(4 * embed_dim, embed_dim)

  def forward(self, x):
    x = self.w1(x)
    x = F.relu(x)
    logits = self.w2(x)
    return logits