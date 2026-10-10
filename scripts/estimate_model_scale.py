"""Estimate parameter count and rough memory needs for Ron model families without allocation."""
from __future__ import annotations
import argparse
import json

def estimate(vocab_size: int, hidden_size: int, num_layers: int, num_heads: int,
             sequence_length: int, architecture: str = "legacy",
             num_kv_heads: int | None = None, tied_embeddings: bool = True) -> dict:
    if min(vocab_size, hidden_size, num_layers, num_heads, sequence_length) < 1:
        raise ValueError("all dimensions must be positive")
    if hidden_size % num_heads:
        raise ValueError("hidden_size must be divisible by num_heads")
    if architecture not in {"legacy", "scalable"}:
        raise ValueError("architecture must be legacy or scalable")
    if architecture == "legacy":
        embedding = vocab_size * hidden_size
        position_embedding = sequence_length * hidden_size
        per_layer = (4 * hidden_size * hidden_size + 4 * hidden_size
                     + 8 * hidden_size * hidden_size + 5 * hidden_size + 4 * hidden_size)
        parameters = embedding + position_embedding + num_layers * per_layer + 2 * hidden_size
        architecture_note = "Legacy character-token RonCausalLM estimate: learned positions, LayerNorm, MHA and GELU MLP."
    else:
        kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        if kv_heads < 1 or num_heads % kv_heads:
            raise ValueError("num_heads must be divisible by positive num_kv_heads")
        head_dim = hidden_size // num_heads
        if head_dim % 2:
            raise ValueError("scalable RoPE head dimension must be even")
        kv_width = kv_heads * head_dim
        ff_size = ((int(8 * hidden_size / 3 + 63) // 64) * 64)
        embedding = vocab_size * hidden_size * (1 if tied_embeddings else 2)
        position_embedding = 0
        per_layer = 2 * hidden_size * hidden_size + 2 * hidden_size * kv_width + 3 * hidden_size * ff_size + 2 * hidden_size
        parameters = embedding + num_layers * per_layer + hidden_size
        architecture_note = "Scalable subword Ron estimate: RoPE, RMSNorm, SwiGLU and grouped-query attention; assumes no linear biases."
    return {
        "architecture": architecture, "vocab_size": vocab_size, "hidden_size": hidden_size,
        "num_layers": num_layers, "num_heads": num_heads,
        "num_kv_heads": num_kv_heads if architecture == "scalable" else num_heads,
        "sequence_length": sequence_length, "tied_embeddings": tied_embeddings,
        "estimated_parameters": parameters,
        "estimated_parameters_millions": round(parameters / 1_000_000, 3),
        "fp32_inference_weights_gib": round(parameters * 4 / (1024 ** 3), 3),
        "bf16_inference_weights_gib": round(parameters * 2 / (1024 ** 3), 3),
        "adamw_fp32_weights_gradients_moments_gib": round(parameters * 16 / (1024 ** 3), 3),
        "memory_note": "Training estimate excludes activations, temporary buffers, fragmentation, and checkpoint copies; actual peak memory is higher.",
        "architecture_note": architecture_note,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("legacy", "scalable"), default="legacy")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--kv-heads", type=int, help="Scalable architecture only; defaults to --heads")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--untied-embeddings", action="store_true")
    args = parser.parse_args()
    print(json.dumps(estimate(args.vocab_size, args.hidden_size, args.layers, args.heads,
        args.sequence_length, architecture=args.architecture, num_kv_heads=args.kv_heads,
        tied_embeddings=not args.untied_embeddings), indent=2))

if __name__ == "__main__":
    main()
