"""Supervised fine-tuning for Ron-10M on reviewed English dialogue examples.

This is a character-level SFT stage: loss is applied only to Ron's response characters,
not to the user's prompt characters. It requires a pretraining candidate checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from ron.model import RonCausalLM
from ron.model_config import RonModelConfig

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "training" / "corpus" / "oasst1" / "english_dialogues.txt"
ARABIC_CORPUS = ROOT / "training" / "arabic_sft_seed.txt"
PRETRAINED = ROOT / "artifacts" / "native-10m" / "ron_native_10m.pt"
LIVE_CHECKPOINT = ROOT / "ron" / "checkpoints" / "ron_native_10m.pt"
OUT = ROOT / "artifacts" / "native-10m"
SEED = 1702
STEPS = int(os.environ.get("RON_SFT_STEPS") or "5000")
BATCH = int(os.environ.get("RON_SFT_BATCH_SIZE") or "2")
LENGTH = int(os.environ.get("RON_SFT_SEQUENCE_LENGTH") or "256")
ARABIC_BATCH_FRACTION = float(os.environ.get("RON_ARABIC_BATCH_FRACTION") or "0.65")
if STEPS < 1 or BATCH < 1 or LENGTH < 8:
    raise ValueError("SFT steps and batch size must be positive and sequence length at least 8")
if not 0 < ARABIC_BATCH_FRACTION < 1:
    raise ValueError("RON_ARABIC_BATCH_FRACTION must be between 0 and 1")


def split_examples(corpus: str, seed: int = SEED, validation_fraction: float = 0.1, min_examples: int = 100):
    examples = [part.strip() for part in corpus.replace("\r\n", "\n").split("\n\n") if part.strip()]
    if len(examples) < min_examples:
        raise ValueError(f"Expected at least {min_examples} dialogue examples; found {len(examples)}")
    validation_count = max(1, min(len(examples) - 1, round(len(examples) * validation_fraction)))
    validation_ids = set(random.Random(seed).sample(range(len(examples)), validation_count))
    train = [example for index, example in enumerate(examples) if index not in validation_ids]
    validation = [example for index, example in enumerate(examples) if index in validation_ids]
    return train, validation


def encode_examples(examples: list[str], vocab: dict[str, int]):
    """Encode conversations and mark only Ron's response characters for next-token loss."""
    unknown_id = vocab.get(" ", next(iter(vocab.values())))
    tokens: list[int] = []
    response_mask: list[bool] = []
    unknown_characters = 0
    total_characters = 0
    for example in examples:
        lines = [line.strip() for line in example.splitlines() if line.strip()]
        if not lines:
            continue
        for line in lines:
            if ":" not in line:
                continue
            role, content = line.split(":", 1)
            role = role.strip()
            if role not in {"User", "Ron", "المستخدم", "رون"}:
                continue
            normalized = f"{role}:{content}"
            assistant_role = role in {"Ron", "رون"}
            response_start = len(role) + 1 + (1 if content.startswith(" ") else 0)
            for position, char in enumerate(normalized):
                total_characters += 1
                token_id = vocab.get(char, unknown_id)
                unknown_characters += char not in vocab
                tokens.append(token_id)
                # Keep the role label and separator out of the assistant-only loss.
                response_mask.append(assistant_role and position >= response_start)
            newline_id = vocab.get("\n", unknown_id)
            tokens.append(newline_id)
            response_mask.append(False)
        # Blank-line separators are not prediction targets.
        tokens.append(vocab.get("\n", unknown_id))
        response_mask.append(False)
    if not tokens:
        raise ValueError("No valid User/Ron dialogue lines found")
    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(response_mask, dtype=torch.bool),
        unknown_characters,
        total_characters,
    )


def candidate_starts(response_mask: torch.Tensor, length: int) -> torch.Tensor:
    top = len(response_mask) - length - 1
    if top < 1:
        raise ValueError(f"Dialogue split has {len(response_mask)} characters; needs at least {length + 2}")
    prefix = torch.cat((torch.zeros(1, dtype=torch.long), response_mask.long().cumsum(0)))
    starts = torch.arange(top, dtype=torch.long)
    target_counts = prefix[starts + length + 1] - prefix[starts + 1]
    valid = starts[target_counts > 0]
    if valid.numel() == 0:
        raise ValueError("No sequence window contains any assistant-response targets")
    return valid


def make_batch(tokens, response_mask, valid_starts, size: int, length: int):
    picks = torch.randint(0, len(valid_starts), (size,))
    starts = valid_starts[picks].tolist()
    inputs = torch.stack([tokens[start:start + length] for start in starts])
    targets = torch.stack([tokens[start + 1:start + length + 1].clone() for start in starts])
    masks = torch.stack([response_mask[start + 1:start + length + 1] for start in starts])
    targets[~masks] = -100
    return inputs, targets


def make_mixed_batch(english, arabic, size: int, length: int, arabic_fraction: float):
    """Sample languages with the requested long-run fraction, including small batches."""
    if size < 1:
        raise ValueError("batch size must be positive")
    if not 0 < arabic_fraction < 1:
        raise ValueError("Arabic fraction must be between 0 and 1")

    # Stochastic rounding preserves the target fraction when batch size is small.
    # For size=2 and fraction=0.65, this selects two Arabic rows 30% of the
    # time and one Arabic row 70% of the time: E[Arabic rows] = 1.3.
    expected_arabic = size * arabic_fraction
    arabic_size = int(math.floor(expected_arabic))
    if random.random() < expected_arabic - arabic_size:
        arabic_size += 1
    arabic_size = max(0, min(size, arabic_size))
    english_size = size - arabic_size

    if arabic_size == 0:
        return make_batch(*english, size, length)
    if english_size == 0:
        return make_batch(*arabic, size, length)

    x_en, y_en = make_batch(*english, english_size, length)
    x_ar, y_ar = make_batch(*arabic, arabic_size, length)
    return torch.cat((x_en, x_ar), dim=0), torch.cat((y_en, y_ar), dim=0)


@torch.no_grad()
def evaluate(model, tokens, response_mask, valid_starts, count: int = 24) -> float:
    was_training = model.training
    model.eval()
    picks = torch.linspace(0, len(valid_starts) - 1, steps=min(count, len(valid_starts))).long().unique()
    values = []
    for pick in picks.tolist():
        start = int(valid_starts[pick])
        x = tokens[start:start + LENGTH].unsqueeze(0)
        y = tokens[start + 1:start + LENGTH + 1].clone().unsqueeze(0)
        mask = response_mask[start + 1:start + LENGTH + 1].unsqueeze(0)
        y[~mask] = -100
        loss = model(x, y).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("non-finite assistant-only validation loss")
        values.append(float(loss.item()))
    model.train(was_training)
    return sum(values) / len(values)


def atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    if not CORPUS.is_file():
        raise FileNotFoundError("Missing OASST1 dialogue data; run scripts/download_oasst1_english.py first.")
    checkpoint_path = PRETRAINED if PRETRAINED.is_file() else LIVE_CHECKPOINT
    if not checkpoint_path.is_file():
        raise FileNotFoundError("No Ron-10M pretraining checkpoint found; run scripts/train_native_10m.py first.")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = RonModelConfig(**payload["config"])
    vocab: dict[str, int] = payload["vocab"]
    model = RonCausalLM(config)
    model.load_state_dict(payload.get("best_state_dict", payload["state_dict"]))
    model.eval()

    english_dialogues = CORPUS.read_text(encoding="utf-8")
    arabic_dialogues = ARABIC_CORPUS.read_text(encoding="utf-8") if ARABIC_CORPUS.is_file() else ""
    # Mix reviewed English dialogues with the curated Arabic conversation curriculum.
    dialogue_text = english_dialogues.rstrip() + ("\n\n" + arabic_dialogues.strip() if arabic_dialogues.strip() else "") + "\n"
    old_vocab = dict(vocab)
    for char in sorted(set(dialogue_text) - set(vocab)):
        vocab[char] = len(vocab)
    if len(vocab) != len(old_vocab):
        # Extend the native character vocabulary without changing existing token IDs.
        old_state = payload.get("best_state_dict", payload["state_dict"])
        expanded_config = dict(payload["config"])
        expanded_config["vocab_size"] = len(vocab)
        expanded_model = RonCausalLM(RonModelConfig(**expanded_config))
        expanded_state = expanded_model.state_dict()
        for name, current in expanded_state.items():
            previous = old_state.get(name)
            if previous is None:
                raise ValueError(f"Checkpoint is missing model tensor {name!r}")
            if previous.shape == current.shape:
                current.copy_(previous)
            elif name in {"token_embedding.weight", "lm_head.weight"} and previous.ndim == 2 and previous.shape[1] == current.shape[1] and previous.shape[0] < current.shape[0]:
                current[:previous.shape[0]].copy_(previous)
            else:
                raise ValueError(f"Cannot safely expand checkpoint tensor {name!r}: {tuple(previous.shape)} -> {tuple(current.shape)}")
        expanded_model.load_state_dict(expanded_state)
        model = expanded_model
        config = model.config
    if not arabic_dialogues.strip():
        raise ValueError("Arabic SFT curriculum is required; refusing English-only instruction tuning.")
    english_train, english_val = split_examples(english_dialogues, seed=SEED)
    arabic_train, arabic_val = split_examples(arabic_dialogues, seed=SEED + 1, min_examples=10)
    train_examples = english_train + arabic_train
    validation_examples = english_val + arabic_val

    train_tokens_en, train_mask_en, train_unknown_en, train_total_en = encode_examples(english_train, vocab)
    train_tokens_ar, train_mask_ar, train_unknown_ar, train_total_ar = encode_examples(arabic_train, vocab)
    val_tokens_en, val_mask_en, val_unknown_en, val_total_en = encode_examples(english_val, vocab)
    val_tokens_ar, val_mask_ar, val_unknown_ar, val_total_ar = encode_examples(arabic_val, vocab)
    unknown_rates = (
        train_unknown_en / max(1, train_total_en),
        train_unknown_ar / max(1, train_total_ar),
        val_unknown_en / max(1, val_total_en),
        val_unknown_ar / max(1, val_total_ar),
    )
    if max(unknown_rates) > 0.05:
        raise ValueError("More than 5% of dialogue characters are missing from the pretraining vocabulary; rebuild with a compatible tokenizer.")

    train_en = (train_tokens_en, train_mask_en, candidate_starts(train_mask_en, LENGTH))
    train_ar = (train_tokens_ar, train_mask_ar, candidate_starts(train_mask_ar, LENGTH))
    val_starts_en = candidate_starts(val_mask_en, LENGTH)
    val_starts_ar = candidate_starts(val_mask_ar, LENGTH)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    start_val_en = evaluate(model, val_tokens_en, val_mask_en, val_starts_en)
    start_val_ar = evaluate(model, val_tokens_ar, val_mask_ar, val_starts_ar)
    start_val = (start_val_en + start_val_ar) / 2
    best_val = start_val
    best_val_en = start_val_en
    best_val_ar = start_val_ar
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_step = 0

    base_lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), weight_decay=0.1)
    started = time.time()
    history = []
    stale = 0
    early_stopped = False
    warmup_steps = min(100, max(1, STEPS // 10))
    steps_completed = 0
    model.train()
    for step in range(1, STEPS + 1):
        steps_completed = step
        warmup = min(1.0, step / warmup_steps)
        progress = (step - 1) / max(1, STEPS - 1)
        lr = base_lr * warmup * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = make_mixed_batch(train_en, train_ar, BATCH, LENGTH, ARABIC_BATCH_FRACTION)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"non-finite SFT loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0:
            val_loss_en = evaluate(model, val_tokens_en, val_mask_en, val_starts_en)
            val_loss_ar = evaluate(model, val_tokens_ar, val_mask_ar, val_starts_ar)
            val_loss = (val_loss_en + val_loss_ar) / 2
            history.append({
                "step": step,
                "validation_response_loss": val_loss,
                "validation_english_loss": val_loss_en,
                "validation_arabic_loss": val_loss_ar,
                "learning_rate": lr,
            })
            print(
                f"sft_step={step} balanced_validation_loss={val_loss:.4f} "
                f"english={val_loss_en:.4f} arabic={val_loss_ar:.4f}",
                flush=True,
            )
            if val_loss < best_val:
                best_val = val_loss
                best_val_en = val_loss_en
                best_val_ar = val_loss_ar
                best_step = step
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 10:
                early_stopped = True
                break

    final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    changed_tensors = sum(not torch.equal(final_state[name], initial_state[name]) for name in final_state)
    corpus_text = dialogue_text
    train_tokens_total = int(train_tokens_en.numel() + train_tokens_ar.numel())
    val_tokens_total = int(val_tokens_en.numel() + val_tokens_ar.numel())
    train_unknown = train_unknown_en + train_unknown_ar
    train_total = train_total_en + train_total_ar
    val_unknown = val_unknown_en + val_unknown_ar
    val_total = val_total_en + val_total_ar
    checkpoint = {
        "state_dict": final_state,
        "best_state_dict": best_state,
        "best_selected_step": best_step,
        "best_validation_loss": best_val,
        "config": model.config.__dict__,
        "vocab": vocab,
        "seed": SEED,
        "selected_step": int(payload.get("selected_step", 0)) + steps_completed,
        "training_steps_this_run": steps_completed,
        "training_steps_total": int(payload.get("training_steps_total", payload.get("selected_step", 0))) + steps_completed,
        "corpus_sha256": hashlib.sha256(corpus_text.encode("utf-8")).hexdigest(),
        "checkpoint_source": f"sft_from_{checkpoint_path.name}",
        "optimizer_state_dict": optimizer.state_dict(),
        "training_stage": "assistant_only_supervised_fine_tuning",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    candidate_path = OUT / "ron_native_10m_sft.pt"
    atomic_torch_save(checkpoint, candidate_path)
    validation_improved = bool(best_val < start_val and best_step > 0 and changed_tensors > 0 and math.isfinite(best_val))
    # Do not replace the live checkpoint on response loss alone. Promotion is a
    # separate action after the blind conversational gate passes.
    promoted = False

    metrics = {
        "experiment": "ron0-native-10m-sft",
        "status": "completed",
        "training_stage": "assistant_only_supervised_fine_tuning",
        "checkpoint_source": str(checkpoint_path.relative_to(ROOT)),
        "candidate_checkpoint": str(candidate_path.relative_to(ROOT)),
        "checkpoint_promoted": promoted,
        "validation_loss_improved": validation_improved,
        "promotion_reason": "awaiting_blind_conversational_evaluation" if validation_improved else "assistant_validation_gate_not_passed",
        "live_checkpoint": str(LIVE_CHECKPOINT.relative_to(ROOT)) if promoted else None,
        "training_examples": len(train_examples),
        "training_examples_english": len(english_train),
        "training_examples_arabic": len(arabic_train),
        "validation_examples": len(validation_examples),
        "validation_examples_english": len(english_val),
        "validation_examples_arabic": len(arabic_val),
        "arabic_batch_fraction": ARABIC_BATCH_FRACTION,
        "train_characters": train_tokens_total,
        "validation_characters": val_tokens_total,
        "train_unknown_character_rate": train_unknown / max(1, train_total),
        "validation_unknown_character_rate": val_unknown / max(1, val_total),
        "initial_validation_response_loss": start_val,
        "initial_validation_english_loss": start_val_en,
        "initial_validation_arabic_loss": start_val_ar,
        "best_validation_response_loss": best_val,
        "best_validation_english_loss": best_val_en,
        "best_validation_arabic_loss": best_val_ar,
        "best_step": best_step,
        "training_steps": steps_completed,
        "early_stopped": early_stopped,
        "changed_parameter_tensors": changed_tensors,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": round(time.time() - started, 2),
        "history": history,
        "limitations": [
            "This is character-level supervised fine-tuning; a subword tokenizer remains a priority.",
            "OASST1 is human-generated but not a guarantee of factual correctness; dataset filtering is imperfect.",
            "A lower response loss does not establish general intelligence or factual accuracy.",
        ],
    }
    (OUT / "sft_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
