import torch
import torch.nn as nn
from torch.nn import functional as F
from data.dataset import fetch_raw_data, get_batch, tokenize_data
from data.tokenizer import CharTokenizer


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embeddings(
            idx
        )  # idx (pozycja tokenu w look-up tably/embedding), time (position in block), channel (embedding row correspodning to position)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


data = fetch_raw_data()
vocabulary = sorted(set(data))
tokenizer = CharTokenizer(vocabulary)
tokens = tokenize_data(data, tokenizer)
train_x, train_y = get_batch(tokens)
model = BigramLanguageModel(len(vocabulary))
logits, loss = model(train_x, train_y)
