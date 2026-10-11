"""Validated parameter-scale presets for Ron's native subword Transformer."""
from __future__ import annotations

from ron.scalable_model import ScalableRonConfig

RON_100M_PARAMETERS = 99_959_808
RON_100M_VOCAB_SIZE = 8_000


def build_scale_config(
    preset: str,
    *,
    vocab_size: int,
    sequence_length: int,
    hidden_size: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    num_kv_heads: int = 2,
) -> ScalableRonConfig:
    """Build a pilot config or a fixed, testable ~100M Ron architecture.

    The 100M preset deliberately fixes the vocabulary size and model dimensions;
    changing any of them changes the parameter count and invalidates this preset.
    """
    if preset == "pilot":
        return ScalableRonConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_sequence_length=sequence_length,
        )
    if preset == "ron-100m":
        if vocab_size != RON_100M_VOCAB_SIZE:
            raise ValueError(
                "ron-100m requires an 8,000-token tokenizer; train the tokenizer "
                "on a sufficiently large, reviewed corpus with --vocab-size 8000"
            )
        config = ScalableRonConfig(
            vocab_size=RON_100M_VOCAB_SIZE,
            hidden_size=1024,
            num_layers=8,
            num_heads=16,
            num_kv_heads=4,
            max_sequence_length=sequence_length,
            intermediate_size=2880,
        )
        return config
    raise ValueError(f"unknown scale preset: {preset}")
