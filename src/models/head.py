import torch.nn as nn
import torch.nn.functional as F
import torch


class Head(nn.Module):

    def __init__(self, embed_dim: int, head_size: int, block_size: int) -> None:
        super().__init__()
        self.wk = nn.Linear(embed_dim, head_size, bias=False)
        self.wq = nn.Linear(embed_dim, head_size, bias=False)
        self.wv = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        head_dim = Q.shape[-1]
        wei = Q @ K.transpose(-2, -1) * head_dim**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = F.dropout(wei, p=0.1)
        return wei @ V
