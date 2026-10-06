"""Configuration contract for Ron's first native causal language model."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RonModelConfig:
    vocab_size: int = 4096
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    max_sequence_length: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("vocab_size", "hidden_size", "num_layers", "num_heads", "max_sequence_length"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
