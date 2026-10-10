"""Train a Ron-specific LoRA adapter from reviewed examples, with held-out evaluation.

The adapter remains separate from its base model. Saving an adapter alone does
not make it a standalone model; the base weights are still required for inference.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def validate_training_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and deduplicate user/assistant chat records before any model load."""
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} must be a JSON object")
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"record {index} must contain at least two messages")
        normalized = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"record {index} contains an invalid message")
            role = message.get("role")
            text = message.get("content")
            if role not in {"user", "assistant"} or not isinstance(text, str) or not text.strip():
                raise ValueError(f"record {index} messages must have non-empty user/assistant content")
            normalized.append({"role": role, "content": text.strip()})
        if normalized[0]["role"] != "user" or normalized[-1]["role"] != "assistant":
            raise ValueError(f"record {index} must start with user and end with assistant")
        signature = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        if signature not in seen:
            seen.add(signature)
            clean.append({"messages": normalized})
    if len(clean) < 2:
        raise ValueError("at least two distinct approved examples are required for train/validation split")
    return clean


def load_approved_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"training data not found: {source}")
    records = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
    return validate_training_records(records)


def split_records(
    records: list[dict[str, Any]],
    validation_split: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create a reproducible holdout split without sharing exact records."""
    if not 0 < validation_split < 0.5:
        raise ValueError("validation_split must be greater than 0 and less than 0.5")
    if len(records) < 2:
        raise ValueError("at least two records are required")
    validation_count = max(1, min(len(records) - 1, round(len(records) * validation_split)))
    validation_indices = set(random.Random(seed).sample(range(len(records)), validation_count))
    train = [record for index, record in enumerate(records) if index not in validation_indices]
    validation = [record for index, record in enumerate(records) if index in validation_indices]
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts/ron-qwen-lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        records = load_approved_jsonl(args.data)
        train_records, validation_records = split_records(
            records, validation_split=args.validation_split, seed=args.seed
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Heavy optional dependencies are imported only when a training run is requested.
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    config = SFTConfig(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=1e-4,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        max_length=args.max_length,
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=Dataset.from_list(train_records),
        eval_dataset=Dataset.from_list(validation_records),
        processing_class=tokenizer,
        peft_config=lora,
    )

    trainer.train()
    evaluation_metrics = trainer.evaluate()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))

    metrics = {
        key: float(value) if isinstance(value, (int, float)) else str(value)
        for key, value in evaluation_metrics.items()
    }
    manifest = {
        "base_model": args.model,
        "method": "LoRA SFT",
        "dataset": str(args.data),
        "approved_examples": len(records),
        "training_examples": len(train_records),
        "validation_examples": len(validation_records),
        "validation_split": args.validation_split,
        "seed": args.seed,
        "evaluation_metrics": metrics,
        "best_validation_loss": trainer.state.best_metric,
        "adapter_only": True,
        "base_model_required_for_adapter_inference": True,
        "qwen_required_at_runtime": args.model.lower().startswith("qwen/"),
        "merged_model_exported": False,
        "ron_identity_owned_by_project": True,
    }
    (output / "ron-model-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
