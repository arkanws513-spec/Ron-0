"""Train or continue training Ron's native character-level Transformer."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from torch import nn

from ron.model import RonCausalLM
from ron.model_config import RonModelConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "native-baseline"
LIVE_CHECKPOINT = ROOT / "ron" / "checkpoints" / "ron_native_baseline.pt"
SEED, STEPS, BATCH, LENGTH = 1701, 1800, 8, 96
HIDDEN, LAYERS, HEADS = 64, 2, 4


def batch(tokens: torch.Tensor, size: int, length: int):
    top = len(tokens) - length - 1
    if top < 1:
        raise ValueError(f"corpus split has {len(tokens)} tokens; needs at least {length + 2}")
    starts = torch.randint(0, top, (size,))
    return (
        torch.stack([tokens[s:s + length] for s in starts.tolist()]),
        torch.stack([tokens[s + 1:s + length + 1] for s in starts.tolist()]),
    )


@torch.no_grad()
def evaluate(model, tokens: torch.Tensor, count: int = 12) -> float:
    model.eval()
    values = [float(model(*batch(tokens, min(4, max(1, len(tokens) // (LENGTH + 2))), LENGTH)).loss.item())
              for _ in range(count)]
    model.train()
    return sum(values) / len(values)


def load_or_initialize(corpus: str):
    """Continue from Ron's live checkpoint when compatible; otherwise initialize Ron's own model."""
    if LIVE_CHECKPOINT.is_file():
        payload = torch.load(LIVE_CHECKPOINT, map_location="cpu", weights_only=True)
        old_vocab = payload["vocab"]
        vocab = dict(old_vocab)  # Preserve existing token IDs so old weights keep their meaning.
        for char in sorted(set(corpus) - set(vocab)):
            vocab[char] = len(vocab)

        old_cfg = payload["config"]
        cfg = RonModelConfig(
            vocab_size=len(vocab),
            hidden_size=int(old_cfg["hidden_size"]),
            num_layers=int(old_cfg["num_layers"]),
            num_heads=int(old_cfg["num_heads"]),
            max_sequence_length=int(old_cfg["max_sequence_length"]),
            dropout=float(old_cfg.get("dropout", 0.0)),
        )
        if cfg.hidden_size != HIDDEN or cfg.num_layers != LAYERS or cfg.num_heads != HEADS or cfg.max_sequence_length != LENGTH:
            raise RuntimeError("Existing Ron checkpoint architecture differs from this training recipe; refusing to overwrite it.")
        model = RonCausalLM(cfg)
        target = model.state_dict()
        old_state = payload["state_dict"]
        copied, expanded = 0, False
        for name, current in target.items():
            previous = old_state.get(name)
            if previous is None:
                continue
            if previous.shape == current.shape:
                current.copy_(previous)
                copied += 1
            elif name in {"token_embedding.weight", "lm_head.weight"} and previous.ndim == 2 and previous.shape[1] == current.shape[1]:
                rows = min(previous.shape[0], current.shape[0])
                current[:rows].copy_(previous[:rows])
                copied += 1
                expanded = True
        model.load_state_dict(target)
        if copied == 0:
            raise RuntimeError("Existing checkpoint contained no compatible weights; refusing silent reinitialization.")
        return model, vocab, "continued_existing_ron_checkpoint", int(payload.get("selected_step", 0)), expanded

    vocab = {char: index for index, char in enumerate(sorted(set(corpus)))}
    cfg = RonModelConfig(vocab_size=len(vocab), hidden_size=HIDDEN, num_layers=LAYERS,
                         num_heads=HEADS, max_sequence_length=LENGTH)
    return RonCausalLM(cfg), vocab, "initialized_ron_native_model_no_checkpoint_found", 0, False


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    corpus = (ROOT / "training" / "seed_corpus.txt").read_text(encoding="utf-8")
    model, vocab, checkpoint_source, prior_step, vocab_expanded = load_or_initialize(corpus)
    ids = torch.tensor([vocab[char] for char in corpus], dtype=torch.long)
    split = int(len(ids) * 0.9)
    train, val = ids[:split], ids[split:]
    if len(train) < LENGTH + 2 or len(val) < LENGTH + 2:
        raise ValueError("Training corpus is too small for the configured sequence length; expand training/seed_corpus.txt.")

    start_train, start_val = evaluate(model, train), evaluate(model, val)
    best_val, best_step = start_val, 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3 if checkpoint_source.startswith("continued") else 3e-3)
    started = time.time()
    history = []
    model.train()
    for step in range(1, STEPS + 1):
        x, y = batch(train, BATCH, LENGTH)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0:
            train_loss, validation_loss = evaluate(model, train), evaluate(model, val, count=16)
            history.append({"step": step, "train_loss": train_loss, "validation_loss": validation_loss})
            print(f"step={step} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}", flush=True)
            if validation_loss < best_val:
                best_val, best_step = validation_loss, step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": model.config.__dict__,
        "vocab": vocab,
        "seed": SEED,
        "selected_step": best_step,
        "training_steps_this_run": STEPS,
        "checkpoint_source": checkpoint_source,
        "prior_selected_step": prior_step,
    }
    torch.save(checkpoint, OUT / "ron_native_baseline.pt")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    metrics = {
        "experiment": "ron0-native-character-lm",
        "status": "completed",
        "checkpoint_source": checkpoint_source,
        "prior_selected_step": prior_step,
        "vocabulary_expanded": vocab_expanded,
        "seed": SEED,
        "training_steps_this_run": STEPS,
        "selected_checkpoint_step_this_run": best_step,
        "parameters": parameters,
        "vocab_size": len(vocab),
        "corpus_characters": len(ids),
        "train_characters": len(train),
        "validation_characters": len(val),
        "initial_train_loss": start_train,
        "initial_validation_loss": start_val,
        "best_validation_loss": best_val,
        "last_train_loss": history[-1]["train_loss"] if history else start_train,
        "last_validation_loss": history[-1]["validation_loss"] if history else start_val,
        "train_loss_reduction_percent": 100 * (start_train - (history[-1]["train_loss"] if history else start_train)) / max(start_train, 1e-9),
        "validation_loss_reduction_percent": 100 * (start_val - best_val) / max(start_val, 1e-9),
        "elapsed_seconds": round(time.time() - started, 2),
        "history": history,
        "limitations": [
            "This is a small character-level prototype trained on a hand-curated corpus, not a general-purpose large language model.",
            "Validation is a held-out tail segment of the same corpus, not an independent benchmark.",
            "The selected checkpoint is the best validation checkpoint, including the starting weights, to avoid promoting a regression.",
            "No external inference model or paid API is used.",
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
