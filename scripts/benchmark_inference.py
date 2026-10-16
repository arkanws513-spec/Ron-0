#!/usr/bin/env python3
"""Compare cached and reference decoding for Ron's native Transformer.

By default this benchmarks a randomly initialized model to measure implementation
overhead only. Pass --checkpoint to benchmark real checkpoint weights. Results
are hardware- and workload-specific and must not be generalized to other setups.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from ron.model import RonCausalLM
from ron.model_config import RonModelConfig


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[RonCausalLM, str]:
    if args.checkpoint is not None:
        payload: dict[str, Any] = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True
        )
        config = RonModelConfig(**payload["config"])
        model = RonCausalLM(config)
        state = payload.get("best_state_dict", payload.get("state_dict"))
        if state is None:
            raise ValueError("Checkpoint contains neither best_state_dict nor state_dict")
        model.load_state_dict(state)
        source = str(args.checkpoint)
    else:
        config = RonModelConfig(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            num_heads=args.heads,
            max_sequence_length=args.context,
            dropout=0.0,
        )
        torch.manual_seed(args.seed)
        model = RonCausalLM(config)
        source = "randomly_initialized_synthetic_model"
    model.to(device).eval()
    return model, source


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def measure(
    model: RonCausalLM,
    prompt: torch.Tensor,
    *,
    new_tokens: int,
    repeats: int,
    warmup: int,
    use_cache: bool,
    device: torch.device,
) -> dict[str, float | int | list[float]]:
    for _ in range(warmup):
        model.generate(prompt, max_new_tokens=new_tokens, use_cache=use_cache)
    synchronize(device)

    durations: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        generated = model.generate(prompt, max_new_tokens=new_tokens, use_cache=use_cache)
        synchronize(device)
        durations.append(time.perf_counter() - started)
        if generated.shape[1] != prompt.shape[1] + new_tokens:
            raise RuntimeError(
                f"Expected {new_tokens} new tokens, got {generated.shape[1] - prompt.shape[1]}"
            )
    median_seconds = statistics.median(durations)
    return {
        "median_seconds": median_seconds,
        "median_new_tokens_per_second": new_tokens / median_seconds if median_seconds else 0.0,
        "durations_seconds": durations,
        "repeats": repeats,
        "warmup_runs": warmup,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--context", type=int, default=256)
    args = parser.parse_args()

    if args.prompt_tokens < 1 or args.new_tokens < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("prompt/new token counts and repeats must be positive; warmup cannot be negative")
    if args.prompt_tokens > args.context or args.new_tokens > args.context:
        parser.error("prompt-tokens and new-tokens must each be <= context")
    if args.hidden_size % args.heads:
        parser.error("hidden-size must be divisible by heads")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    model, source = load_model(args, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    prompt = torch.randint(
        low=0, high=model.config.vocab_size, size=(1, args.prompt_tokens), generator=generator
    ).to(device)

    # Check semantic equivalence before timing, so a faster incorrect path cannot pass.
    cached_output = model.generate(prompt, max_new_tokens=args.new_tokens, use_cache=True)
    reference_output = model.generate(prompt, max_new_tokens=args.new_tokens, use_cache=False)
    equivalent = torch.equal(cached_output, reference_output)
    if not equivalent:
        raise RuntimeError("Cached and reference greedy generation differ; refusing to report speedup")

    cached = measure(
        model, prompt, new_tokens=args.new_tokens, repeats=args.repeats,
        warmup=args.warmup, use_cache=True, device=device,
    )
    reference = measure(
        model, prompt, new_tokens=args.new_tokens, repeats=args.repeats,
        warmup=args.warmup, use_cache=False, device=device,
    )
    speedup = (
        reference["median_seconds"] / cached["median_seconds"]
        if cached["median_seconds"] else 0.0
    )
    report = {
        "benchmark": "ron_native_kv_cache_v1",
        "model_source": source,
        "weights_are_trained": args.checkpoint is not None,
        "device": str(device),
        "torch_version": torch.__version__,
        "model_config": {
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_layers,
            "num_heads": model.config.num_heads,
            "max_sequence_length": model.config.max_sequence_length,
        },
        "workload": {
            "batch_size": 1,
            "prompt_tokens": args.prompt_tokens,
            "new_tokens": args.new_tokens,
            "seed": args.seed,
        },
        "cached": cached,
        "reference": reference,
        "speedup_ratio_reference_over_cached": speedup,
        "cached_greedy_output_matches_reference": equivalent,
        "interpretation": (
            "A single hardware/workload result is not a universal speedup claim. "
            "Use a trained checkpoint and deployment hardware before deciding whether to enable caching."
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
