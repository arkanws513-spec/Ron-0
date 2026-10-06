"""Train a Ron-specific LoRA adapter from approved Qwen examples.

This script follows the current TRL SFT API and keeps Qwen base weights
separate from Ron-specific adapter weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts/ron-qwen-lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"training data not found: {data_path}")

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    if "messages" not in dataset.column_names:
        raise SystemExit("dataset must contain a messages column")
    if len(dataset) == 0:
        raise SystemExit("training dataset is empty")

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
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        max_length=args.max_length,
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )

    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))

    manifest = {
        "base_model": args.model,
        "method": "LoRA SFT",
        "dataset": str(data_path),
        "examples": len(dataset),
        "independent_identity": True,
        "qwen_required_at_runtime": False,
    }
    (output / "ron-model-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
