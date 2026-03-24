import torch
import torch.nn as nn
from torch.nn import functional as F
from models.head import Head
from models.multi_head_attention import MultiHeadAttention
from models.block import DecoderBlock


class GPT(nn.Module):

    def __init__(self, vocab_size: int, embed_dim: int, block_size: int, num_heads: int = 8):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.positions = nn.Embedding(block_size, embed_dim)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(embed_dim, num_heads, block_size) for _ in range(12)])
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor, target=None):
        B, T = idx.shape
        token_embeddings = self.embeddings(idx)  # B,T,C
        device = idx.device
        position_embeddings = self.positions(torch.arange(0, T, device=device))  # T, C
        x = token_embeddings + position_embeddings

        for block in self.decoder_blocks:
            x = block(x)

        logits = self.lm_head(x)
        B, T, C = logits.shape

        if target is None:
            return logits, None

        logits = logits.view(B * T, C)
        target = target.view(B * T)
        loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int = 500):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
