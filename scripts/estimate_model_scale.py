"""Estimate RonCausalLM parameter count and rough memory needs without allocating a model."""
from __future__ import annotations

import argparse
import json


def estimate(vocab_size: int, hidden_size: int, num_layers: int, num_heads: int, sequence_length: int) -> dict:
    if min(vocab_size, hidden_size, num_layers, num_heads, sequence_length) < 1:
        raise ValueError("all dimensions must be positive")
    if hidden_size % num_heads:
        raise ValueError("hidden_size must be divisible by num_heads")

    # Mirrors the current model: tied token embedding/lm_head, learned positions,
    # fused QKV + output projection, 4x GELU MLP, LayerNorms and linear biases.
    embedding = vocab_size * hidden_size
    position_embedding = sequence_length * hidden_size
    per_layer = (
        4 * hidden_size * hidden_size + 4 * hidden_size
        + 8 * hidden_size * hidden_size + 5 * hidden_size
        + 4 * hidden_size
    )
    final_norm = 2 * hidden_size
    parameters = embedding + position_embedding + num_layers * per_layer + final_norm
    return {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "sequence_length": sequence_length,
        "estimated_parameters": parameters,
        "estimated_parameters_millions": round(parameters / 1_000_000, 3),
        "fp32_inference_weights_gib": round(parameters * 4 / (1024 ** 3), 3),
        "bf16_inference_weights_gib": round(parameters * 2 / (1024 ** 3), 3),
        "adamw_fp32_weights_gradients_moments_gib": round(parameters * 16 / (1024 ** 3), 3),
        "memory_note": "Training estimate excludes activations, temporary buffers, fragmentation, and checkpoint copies; actual peak memory is higher.",
        "architecture_note": "This estimate matches the current character-token model only. Billion-scale language modeling should first adopt a subword tokenizer and benchmarked scalable attention/normalization design.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(estimate(args.vocab_size, args.hidden_size, args.layers, args.heads, args.sequence_length), indent=2))


if __name__ == "__main__":
    main()
