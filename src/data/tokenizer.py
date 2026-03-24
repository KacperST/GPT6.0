from typing import List
from abc import ABC, abstractmethod
import tiktoken


class Tokenizer(ABC):

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass


class CharTokenizer(Tokenizer):
    def __init__(self, vocabulary: List[str]):
        self.vocab = vocabulary
        self.char_to_int = {c: i for i, c in enumerate(vocabulary)}
        self.int_to_char = {i: c for i, c in enumerate(vocabulary)}

    def encode(self, text: str) -> List[int]:
        return [self.char_to_int[c] for c in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.int_to_char[t] for t in tokens])


class GPTTokenizer(Tokenizer):
    def __init__(
        self,
    ):
        self.enc = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, tokens: List[int]):
        return self.enc.decode(tokens)
