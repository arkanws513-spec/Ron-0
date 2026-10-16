"""Assistant-only BPE fine-tuning for Ron's isolated candidate checkpoint.

The script requires a compatible byte-BPE pretraining checkpoint and writes a
new candidate artifact. It never changes live checkpoint weights.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import torch

from ron.model import RonCausalLM
from ron.model_config import RonModelConfig
from ron.tokenizer import RonTokenizer

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = Path(os.environ.get("RON_BPE_PRETRAINED") or ROOT / "artifacts" / "native-bpe" / "ron_native_bpe_candidate.pt")
ENGLISH = ROOT / "training" / "corpus" / "oasst1" / "english_dialogues.txt"
ARABIC = ROOT / "training" / "arabic_sft_seed.txt"
OUT = ROOT / "artifacts" / "native-bpe"
STEPS = int(os.environ.get("RON_BPE_SFT_STEPS", "2000"))
BATCH = int(os.environ.get("RON_BPE_SFT_BATCH_SIZE", "2"))
LENGTH = int(os.environ.get("RON_BPE_SFT_SEQUENCE_LENGTH", "256"))
SEED = 2702


def split_examples(text: str, min_examples: int = 10):
    examples = [item.strip() for item in text.replace("\r\n", "\n").split("\n\n") if item.strip()]
    if len(examples) < min_examples:
        raise ValueError(f"Expected at least {min_examples} dialogue examples, found {len(examples)}")
    random.shuffle(examples)
    n_val = max(1, round(len(examples) * 0.1))
    return examples[n_val:], examples[:n_val]


def encode_examples(examples: list[str], tokenizer: RonTokenizer):
    tokens: list[int] = []
    response_mask: list[bool] = []
    for example in examples:
        for line in example.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            role, content = line.split(":", 1)
            role = role.strip()
            if role not in {"User", "Ron", "المستخدم", "رون"}:
                continue
            prefix = f"{role}:{' ' if content.startswith(' ') else ''}"
            # Encoding the prefix separately keeps the assistant target boundary explicit.
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            content_ids = tokenizer.encode(content.lstrip() if content.startswith(" ") else content,
                                           add_special_tokens=False)
            tokens.extend(prefix_ids)
            response_mask.extend([False] * len(prefix_ids))
            is_assistant = role in {"Ron", "رون"}
            tokens.extend(content_ids)
            response_mask.extend([is_assistant] * len(content_ids))
            newline = tokenizer.encode("\n", add_special_tokens=False)
            tokens.extend(newline)
            response_mask.extend([False] * len(newline))
        blank = tokenizer.encode("\n", add_special_tokens=False)
        tokens.extend(blank)
        response_mask.extend([False] * len(blank))
    if len(tokens) < LENGTH + 2:
        raise ValueError("Encoded dialogue corpus is too short for the configured sequence length.")
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(response_mask, dtype=torch.bool)


def valid_starts(mask: torch.Tensor):
    top = len(mask) - LENGTH - 1
    starts = torch.arange(max(0, top), dtype=torch.long)
    prefix = torch.cat((torch.zeros(1, dtype=torch.long), mask.long().cumsum(0)))
    counts = prefix[starts + LENGTH + 1] - prefix[starts + 1]
    valid = starts[counts > 0]
    if not valid.numel():
        raise ValueError("No training window contains assistant response tokens.")
    return valid


def batch(data, size: int):
    tokens, mask, starts = data
    chosen = starts[torch.randint(0, len(starts), (size,))].tolist()
    x = torch.stack([tokens[s:s + LENGTH] for s in chosen])
    y = torch.stack([tokens[s + 1:s + LENGTH + 1].clone() for s in chosen])
    target_mask = torch.stack([mask[s + 1:s + LENGTH + 1] for s in chosen])
    y[~target_mask] = -100
    return x, y


@torch.no_grad()
def evaluate(model, data, count: int = 24):
    tokens, mask, starts = data
    was_training = model.training
    model.eval()
    picks = torch.linspace(0, len(starts) - 1, steps=min(count, len(starts))).long().unique()
    losses = []
    for pick in picks.tolist():
        start = int(starts[pick])
        x = tokens[start:start + LENGTH].unsqueeze(0)
        y = tokens[start + 1:start + LENGTH + 1].clone().unsqueeze(0)
        y[~mask[start + 1:start + LENGTH + 1].unsqueeze(0)] = -100
        loss = model(x, y).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Non-finite assistant-only BPE validation loss.")
        losses.append(float(loss.item()))
    model.train(was_training)
    return sum(losses) / len(losses)


def main():
    if STEPS < 1 or BATCH < 1 or LENGTH < 8:
        raise ValueError("Steps and batch size must be positive; sequence length must be at least 8.")
    if not PRETRAINED.is_file():
        raise FileNotFoundError(f"Missing BPE pretraining candidate: {PRETRAINED}; run scripts/train_native_bpe.py first.")
    if not ENGLISH.is_file() or not ARABIC.is_file():
        raise FileNotFoundError("Both English OASST1 dialogue data and training/arabic_sft_seed.txt are required.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    payload = torch.load(PRETRAINED, map_location="cpu", weights_only=True)
    if payload.get("tokenizer_type") != "byte_bpe" or not payload.get("tokenizer_json"):
        raise ValueError("Refusing to fine-tune a non-BPE checkpoint.")
    tokenizer = RonTokenizer.from_json(payload["tokenizer_json"])
    config = RonModelConfig(**payload["config"])
    if config.vocab_size != tokenizer.backend.get_vocab_size():
        raise ValueError("Pretraining checkpoint model/tokenizer vocabulary mismatch.")
    model = RonCausalLM(config)
    model.load_state_dict(payload.get("best_state_dict", payload["state_dict"]))
    english_train, english_val = split_examples(ENGLISH.read_text(encoding="utf-8"))
    arabic_train, arabic_val = split_examples(ARABIC.read_text(encoding="utf-8"))
    train_data_en = (*encode_examples(english_train, tokenizer),)
    train_data_ar = (*encode_examples(arabic_train, tokenizer),)
    val_data_en = (*encode_examples(english_val, tokenizer),)
    val_data_ar = (*encode_examples(arabic_val, tokenizer),)
    train_en = (*train_data_en, valid_starts(train_data_en[1]))
    train_ar = (*train_data_ar, valid_starts(train_data_ar[1]))
    val_en = (*val_data_en, valid_starts(val_data_en[1]))
    val_ar = (*val_data_ar, valid_starts(val_data_ar[1]))
    start_en, start_ar = evaluate(model, val_en), evaluate(model, val_ar)
    start_val = (start_en + start_ar) / 2
    best_val, best_step = start_val, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    started = time.time()
    history = []
    for step in range(1, STEPS + 1):
        x, y = batch(train_data_ar if random.random() < 0.65 else train_data_en, BATCH)
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, y).loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite BPE SFT loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == STEPS:
            val_en, val_ar = evaluate(model, val_en), evaluate(model, val_ar)
            val = (val_en + val_ar) / 2
            history.append({"step": step, "validation_loss": val, "english_loss": val_en, "arabic_loss": val_ar})
            print(f"bpe_sft_step={step} validation_loss={val:.4f} en={val_en:.4f} ar={val_ar:.4f}", flush=True)
            if val < best_val:
                best_val, best_step = val, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    OUT.mkdir(parents=True, exist_ok=True)
    candidate = OUT / "ron_native_bpe_sft_candidate.pt"
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    result = {
        **payload,
        "checkpoint_format_version": 2,
        "tokenizer_type": "byte_bpe",
        "tokenizer_json": tokenizer.to_json(),
        "state_dict": final_state,
        "best_state_dict": best_state,
        "best_selected_step": best_step,
        "best_validation_loss": best_val,
        "selected_step": int(payload.get("selected_step", 0)) + STEPS,
        "training_steps_this_run": STEPS,
        "training_steps_total": int(payload.get("training_steps_total", 0)) + STEPS,
        "checkpoint_source": f"bpe_sft_from_{PRETRAINED.name}",
        "training_stage": "byte_bpe_assistant_only_sft",
        "optimizer_state_dict": optimizer.state_dict(),
    }
    temporary = candidate.with_suffix(".tmp.pt")
    torch.save(result, temporary)
    os.replace(temporary, candidate)
    metrics = {
        "status": "candidate_created_not_promoted",
        "tokenizer_type": "byte_bpe",
        "pretrained_checkpoint": str(PRETRAINED.relative_to(ROOT)) if PRETRAINED.is_relative_to(ROOT) else str(PRETRAINED),
        "candidate_checkpoint": str(candidate.relative_to(ROOT)),
        "initial_validation_loss": start_val,
        "initial_english_loss": start_en,
        "initial_arabic_loss": start_ar,
        "best_validation_loss": best_val,
        "best_step": best_step,
        "training_steps": STEPS,
        "elapsed_seconds": round(time.time() - started, 2),
        "live_checkpoint_modified": False,
        "history": history,
        "note": "Validation loss is not a substitute for blind conversational quality evaluation.",
    }
    (OUT / "bpe_sft_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
