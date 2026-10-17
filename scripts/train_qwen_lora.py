"""LoRA fine-tuning for Ron's deployed Qwen2.5-0.5B-Instruct base model.

This creates an experimental adapter only. It never edits the live GGUF or promotes
weights automatically. Run on a CUDA GPU; use the blind evaluation and explicit export
steps before considering deployment.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT = ROOT / "artifacts" / "qwen-lora-candidate"
ARABIC_SEED = ROOT / "training" / "arabic_sft_seed.txt"
ENGLISH_CORPUS = ROOT / "training" / "corpus" / "oasst1" / "english_dialogues.txt"
ARABIC_CORPUS = ROOT / "training" / "corpus" / "oasst1" / "arabic_dialogues.txt"
ROLE_MAP = {
    "user": "user",
    "المستخدم": "user",
    "assistant": "assistant",
    "ron": "assistant",
    "رون": "assistant",
}


def is_arabic(text: str) -> bool:
    return sum("\u0600" <= char <= "\u06ff" for char in text) >= sum(
        char.isascii() and char.isalpha() for char in text
    )


def parse_dialogue_blocks(text: str) -> list[list[dict[str, str]]]:
    """Parse exported User/Ron or المستخدم/رون transcripts into chat messages."""
    examples: list[list[dict[str, str]]] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        messages: list[dict[str, str]] = []
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            label, content = line.split(":", 1)
            role = ROLE_MAP.get(label.strip().casefold())
            content = " ".join(content.split())
            if role and content:
                messages.append({"role": role, "content": content})
        if any(m["role"] == "user" for m in messages) and any(
            m["role"] == "assistant" for m in messages
        ):
            examples.append(messages)
    return examples


def build_turn_examples(transcripts: list[str], max_history: int = 8) -> list[dict[str, Any]]:
    """Create one supervised example per assistant turn, preserving conversation context."""
    unique: dict[str, dict[str, Any]] = {}
    for transcript in transcripts:
        for conversation in parse_dialogue_blocks(transcript):
            history: list[dict[str, str]] = []
            for message in conversation:
                if message["role"] == "assistant" and any(m["role"] == "user" for m in history):
                    prompt = history[-max_history:]
                    key = json.dumps([prompt, message], ensure_ascii=False, sort_keys=True)
                    unique.setdefault(key, {
                        "messages": prompt + [message],
                        "language": "ar" if is_arabic(" ".join(
                            m["content"] for m in prompt if m["role"] == "user"
                        )) else "en",
                    })
                history.append(message)
    return list(unique.values())


def normalize_prompt(text: str) -> str:
    value = str(text or "").casefold()
    value = value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    value = value.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"[^\w\u0600-\u06ff]+", " ", value).strip()


def exclude_evaluation_overlap(
    examples: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Remove exact evaluation prompts from training to reduce avoidable leakage."""
    held_out = set()
    for case in cases:
        if str(case.get("prompt", "")).strip():
            held_out.add(normalize_prompt(case["prompt"]))
        for message in case.get("messages") or []:
            if isinstance(message, dict) and message.get("role") == "user":
                held_out.add(normalize_prompt(message.get("content", "")))
    kept = []
    removed = 0
    for example in examples:
        user_prompts = [
            normalize_prompt(message["content"])
            for message in example["messages"] if message["role"] == "user"
        ]
        if any(prompt and prompt in held_out for prompt in user_prompts):
            removed += 1
        else:
            kept.append(example)
    return kept, removed


def split_by_language(
    examples: list[dict[str, Any]], seed: int = 42, validation_fraction: float = 0.1
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic per-language split so both languages appear in validation."""
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for language in ("ar", "en"):
        group = [item for item in examples if item.get("language") == language]
        rng.shuffle(group)
        if len(group) < 2:
            train.extend(group)
            continue
        count = max(1, min(len(group) - 1, round(len(group) * validation_fraction)))
        validation.extend(group[:count])
        train.extend(group[count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def encode_completion(tokenizer, messages: list[dict[str, str]], max_length: int) -> dict[str, list[int]]:
    """Mask prompt tokens so the supervised loss is applied only to the assistant turn."""
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    full_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    if not full_ids[:len(prompt_ids)] == prompt_ids:
        raise ValueError("Tokenizer chat template does not preserve the prompt prefix; refusing unsafe label masking.")
    prompt_length = len(prompt_ids)
    if len(full_ids) > max_length:
        excess = len(full_ids) - max_length
        full_ids = full_ids[excess:]
        prompt_length = max(0, prompt_length - excess)
    if prompt_length >= len(full_ids):
        raise ValueError("Assistant completion was truncated away; increase max sequence length.")
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    return {"input_ids": full_ids, "labels": labels}


def load_transcripts() -> tuple[list[str], dict[str, int]]:
    paths = [ARABIC_SEED, ENGLISH_CORPUS, ARABIC_CORPUS]
    transcripts: list[str] = []
    source_counts: dict[str, int] = {}
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            transcripts.append(text)
            source_counts[str(path.relative_to(ROOT))] = len(parse_dialogue_blocks(text))
    if not transcripts:
        raise FileNotFoundError(
            "No training transcripts found. Add training/arabic_sft_seed.txt and run "
            "python scripts/download_oasst1_english.py for reviewed English/Arabic dialogue."
        )
    return transcripts, source_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow slow CPU smoke runs; not recommended for real training.")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.gradient_accumulation, args.eval_every) < 1:
        parser.error("steps, batch-size, gradient-accumulation, and eval-every must be positive")
    if args.max_sequence_length < 64:
        parser.error("max-sequence-length must be at least 64")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA GPU not detected. Refusing to start a slow/underpowered training run; "
            "use a GPU runtime or pass --allow-cpu only for a smoke test."
        )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device.type == "cuda" else torch.float32
    )

    transcripts, source_counts = load_transcripts()
    examples = build_turn_examples(transcripts)
    eval_cases_path = ROOT / "training" / "qwen_eval_cases.json"
    excluded_eval_examples = 0
    if eval_cases_path.is_file():
        eval_cases = json.loads(eval_cases_path.read_text(encoding="utf-8"))
        examples, excluded_eval_examples = exclude_evaluation_overlap(examples, eval_cases)
    counts = {language: sum(item["language"] == language for item in examples) for language in ("ar", "en")}
    if counts["ar"] < 100 or counts["en"] < 100:
        raise SystemExit(
            f"Insufficient balanced dialogue data: Arabic={counts['ar']}, English={counts['en']}; "
            "need at least 100 assistant-turn examples per language."
        )
    train_examples, val_examples = split_by_language(examples, seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora).to(device)
    model.print_trainable_parameters()

    def encode_all(items):
        encoded = []
        for item in items:
            try:
                sample = encode_completion(tokenizer, item["messages"], args.max_sequence_length)
            except ValueError:
                continue
            if any(label != -100 for label in sample["labels"]):
                encoded.append(sample)
        return encoded

    train_data, val_data = encode_all(train_examples), encode_all(val_examples)
    if len(train_data) < 100 or len(val_data) < 10:
        raise SystemExit(f"Encoded dataset too small: train={len(train_data)}, validation={len(val_data)}")

    def collate(items):
        width = max(len(item["input_ids"]) for item in items)
        input_ids, labels, masks = [], [], []
        for item in items:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            masks.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
        }

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stale_evals = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.time()

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for start_index in range(0, len(val_data), args.batch_size):
            batch = collate(val_data[start_index:start_index + args.batch_size])
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                output = model(**batch)
            if torch.isfinite(output.loss):
                losses.append(float(output.loss.float().item()))
        model.train()
        if not losses:
            raise RuntimeError("Validation produced no finite losses.")
        return sum(losses) / len(losses)

    model.train()
    for step in range(1, args.steps + 1):
        batch_items = random.choices(train_data, k=args.batch_size)
        batch = collate(batch_items)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            loss = model(**batch).loss / args.gradient_accumulation
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        loss.backward()
        if step % args.gradient_accumulation == 0 or step == args.steps:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate()
            print(json.dumps({"step": step, "validation_loss": val_loss}, ensure_ascii=False), flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                stale_evals = 0
                model.save_pretrained(args.output / "adapter")
                tokenizer.save_pretrained(args.output / "adapter")
                (args.output / "best_metrics.json").write_text(
                    json.dumps({"step": step, "validation_loss": val_loss}, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                stale_evals += 1
            if stale_evals >= 4:
                print("Early stopping: validation loss has not improved for four evaluations.", flush=True)
                break

    report = {
        "base_model": args.model,
        "training_method": "LoRA supervised fine-tuning",
        "adapter_only": True,
        "promoted": False,
        "device": str(device),
        "dtype": str(dtype),
        "train_examples": len(train_data),
        "validation_examples": len(val_data),
        "language_examples": counts,
        "source_dialogue_blocks": source_counts,
        "exact_evaluation_prompts_excluded": excluded_eval_examples,
        "best_validation_loss": best_loss,
        "elapsed_seconds": round(time.time() - start, 2),
        "note": "Experimental adapter only. It has not been blind-evaluated, merged, converted to GGUF, or deployed.",
    }
    (args.output / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
