"""Merge a trained LoRA adapter into the original Qwen base for GGUF export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/qwen-lora-candidate/adapter"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/qwen-merged"))
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload(safe_merge=True)
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)
    report = {
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "merged_output": str(args.output),
        "merged": True,
        "quantized": False,
        "deployed": False,
        "note": "Convert this merged HF checkpoint to GGUF and run blind evaluation before changing Railway model settings.",
    }
    (args.output / "merge_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
