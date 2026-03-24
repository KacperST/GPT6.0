import torch
import torch.nn as nn
from torch.nn import functional as F
from models.block import DecoderBlock
from models.normalization import LayerNormalization


class GPT(nn.Module):

    def __init__(
        self, vocab_size: int, embed_dim: int, block_size: int, num_heads: int = 8, num_decored_blocks: int = 12
    ):
        super().__init__()
        self.num_decoded_blocks = num_decored_blocks
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.positions = nn.Embedding(block_size, embed_dim)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(embed_dim, num_heads, block_size) for _ in range(num_decored_blocks)]
        )
        self.ln1 = LayerNormalization(embed_dim=embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.block_size = block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # Pre-init scale for residual connections
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.num_decoder_blocks) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, target=None):
        B, T = idx.shape
        token_embeddings = self.embeddings(idx)  # B,T,C
        device = idx.device
        position_embeddings = self.positions(torch.arange(0, T, device=device))  # T, C
        x = token_embeddings + position_embeddings

        for block in self.decoder_blocks:
            x = block(x)
        x = self.ln1(x)
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
