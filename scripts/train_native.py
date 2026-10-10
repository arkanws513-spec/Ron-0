"""Train or continue training Ron's native character-level Transformer."""
from __future__ import annotations

import copy
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


def split_corpus(corpus: str, train_fraction: float = 0.9, seed: int = SEED):
    """Hold out complete user/assistant pairs instead of the final contiguous text tail."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    lines = [line.rstrip("\r") for line in corpus.splitlines() if line.strip()]
    if len(lines) < 4 or len(lines) % 2:
        raise ValueError("corpus must contain at least two complete user/assistant pairs")
    pairs = []
    for index in range(0, len(lines), 2):
        user_line, assistant_line = lines[index:index + 2]
        if not user_line.startswith("المستخدم: ") or not assistant_line.startswith("رون: "):
            raise ValueError(f"invalid conversation pair near corpus line {index + 1}")
        pairs.append(user_line + "\n" + assistant_line)
    validation_count = max(1, min(len(pairs) - 1, round(len(pairs) * (1 - train_fraction))))
    validation_indices = set(random.Random(seed).sample(range(len(pairs)), validation_count))
    train_pairs = [pair for index, pair in enumerate(pairs) if index not in validation_indices]
    validation_pairs = [pair for index, pair in enumerate(pairs) if index in validation_indices]
    return (
        "\n".join(train_pairs) + "\n",
        "\n".join(validation_pairs) + "\n",
        len(train_pairs),
        len(validation_pairs),
    )


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
    """Evaluate on fixed, evenly spaced windows so checkpoint selection is reproducible."""
    was_training = model.training
    model.eval()
    top = len(tokens) - LENGTH - 1
    if top < 1:
        raise ValueError(f"evaluation split has {len(tokens)} tokens; needs at least {LENGTH + 2}")
    starts = torch.linspace(0, top - 1, steps=max(1, min(count, top))).long().unique().tolist()
    values = []
    for start in starts:
        x = tokens[start:start + LENGTH].unsqueeze(0)
        y = tokens[start + 1:start + LENGTH + 1].unsqueeze(0)
        loss = model(x, y).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite evaluation loss")
        values.append(float(loss.item()))
    model.train(was_training)
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
            dropout=0.1,
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
        return model, vocab, "continued_existing_ron_checkpoint", int(payload.get("training_steps_total", payload.get("selected_step", 0))), expanded, payload.get("optimizer_state_dict")

    vocab = {char: index for index, char in enumerate(sorted(set(corpus)))}
    cfg = RonModelConfig(vocab_size=len(vocab), hidden_size=HIDDEN, num_layers=LAYERS,
                         num_heads=HEADS, max_sequence_length=LENGTH, dropout=0.1)
    return RonCausalLM(cfg), vocab, "initialized_ron_native_model_no_checkpoint_found", 0, False, None


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    corpus = (ROOT / "training" / "seed_corpus.txt").read_text(encoding="utf-8")
    model, vocab, checkpoint_source, prior_step, vocab_expanded, prior_optimizer_state = load_or_initialize(corpus)
    train_text, validation_text, train_pairs, validation_pairs = split_corpus(corpus)
    train = torch.tensor([vocab[char] for char in train_text], dtype=torch.long)
    val = torch.tensor([vocab[char] for char in validation_text], dtype=torch.long)
    if len(train) < LENGTH + 2 or len(val) < LENGTH + 2:
        raise ValueError("Training corpus is too small for the configured sequence length; expand training/seed_corpus.txt.")

    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    start_train, start_val = evaluate(model, train), evaluate(model, val)
    best_val, best_step = start_val, 0
    best_state_dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3 if checkpoint_source.startswith("continued") else 3e-3)
    if prior_optimizer_state and not vocab_expanded:
        try:
            optimizer.load_state_dict(prior_optimizer_state)
            print("restored_optimizer_state=true", flush=True)
        except (ValueError, RuntimeError) as exc:
            # A vocabulary expansion changes embedding shapes; reset optimizer moments but retain every compatible model weight.
            print(f"optimizer_state_reset_due_to_shape_change={exc}", flush=True)
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
                best_state_dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    # Persist the final state after this run, not the starting/best snapshot.
    # This guarantees the next run continues from weights actually updated by this training pass.
    final_train, final_validation = evaluate(model, train), evaluate(model, val, count=16)
    OUT.mkdir(parents=True, exist_ok=True)
    total_steps = prior_step + STEPS
    final_state_dict = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    delta_squared = sum(float((final_state_dict[name].float() - initial_state[name].float()).pow(2).sum().item())
                        for name in final_state_dict)
    changed_tensors = sum(not torch.equal(final_state_dict[name], initial_state[name]) for name in final_state_dict)
    checkpoint = {
        # Training resumes from the final weights and matching optimizer state.
        "state_dict": final_state_dict,
        # Inference uses the best validation snapshot to reduce overfitting.
        "best_state_dict": best_state_dict,
        "best_selected_step": prior_step + best_step,
        "best_validation_loss": best_val,
        "config": model.config.__dict__,
        "vocab": vocab,
        "seed": SEED,
        "selected_step": total_steps,
        "training_steps_this_run": STEPS,
        "training_steps_total": total_steps,
        "checkpoint_source": checkpoint_source,
        "prior_selected_step": prior_step,
        "optimizer_state_dict": optimizer.state_dict(),
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
        "selected_checkpoint_step_this_run": STEPS,
        "best_validation_step_this_run": best_step,
        "training_steps_total": total_steps,
        "final_train_loss": final_train,
        "final_validation_loss": final_validation,
        "weights_persisted_from_final_training_step": True,
        "best_validation_weights_persisted": True,
        "best_selected_step_total": prior_step + best_step,
        "optimizer_state_persisted": True,
        "changed_parameter_tensors": changed_tensors,
        "parameter_delta_l2": delta_squared ** 0.5,
        "parameters": parameters,
        "vocab_size": len(vocab),
        "corpus_characters": len(corpus),
        "training_pairs": train_pairs,
        "validation_pairs": validation_pairs,
        "split_strategy": "deterministic_conversation_pair_holdout",
        "train_characters": len(train),
        "validation_characters": len(val),
        "initial_train_loss": start_train,
        "initial_validation_loss": start_val,
        "best_validation_loss": best_val,
        "last_train_loss": final_train,
        "last_validation_loss": final_validation,
        "train_loss_reduction_percent": 100 * (start_train - final_train) / max(start_train, 1e-9),
        "validation_loss_reduction_percent": 100 * (start_val - best_val) / max(start_val, 1e-9),
        "elapsed_seconds": round(time.time() - started, 2),
        "history": history,
        "limitations": [
            "This is a small character-level prototype trained on a hand-curated corpus, not a general-purpose large language model.",
            "Validation is a held-out tail segment of the same corpus, not an independent benchmark.",
            "The checkpoint stores both final weights for training continuation and best-validation weights for inference; the two snapshots serve different purposes.",
            "No external inference model or paid API is used.",
        ],
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
