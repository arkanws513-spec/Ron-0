"""Generate explicitly labeled Pythia-160M teacher examples for Ron distillation.

Pythia-160M is a base language model, not an instruction-tuned chatbot. Its outputs
are synthetic and unreviewed; they are not labeled as approved facts.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch


def extract_prompts(corpus: str, limit: int) -> list[str]:
    prompts: list[str] = []
    for line in corpus.splitlines():
        line = line.strip()
        for prefix in ("المستخدم:", "User:", "Question:"):
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                if value and value not in prompts:
                    prompts.append(value)
                break
        if len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError("No user/question prompts found in the supplied corpus")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="EleutherAI/pythia-160m")
    parser.add_argument("--max-prompts", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    if not args.corpus.is_file():
        parser.error("--corpus must point to an existing UTF-8 text file")
    if args.max_prompts < 1 or args.max_new_tokens < 1:
        parser.error("max-prompts and max-new-tokens must be positive")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install distillation dependencies with: pip install -e '.[distill]'") from exc

    torch.manual_seed(args.seed)
    prompts = extract_prompts(args.corpus.read_text(encoding="utf-8"), args.max_prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as out, torch.inference_mode():
        for prompt in prompts:
            teacher_prompt = f"Question: {prompt}\nAnswer:"
            inputs = tokenizer(teacher_prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if not completion:
                continue
            row = {
                "prompt": prompt,
                "response": completion,
                "review_status": "teacher_generated",
                "source": "pythia-160m",
                "teacher_model": args.model,
                "human_reviewed": False,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    if written == 0:
        raise RuntimeError("Teacher generated no usable completions")
    print(json.dumps({
        "status": "teacher_dataset_generated_unreviewed",
        "teacher_model": args.model,
        "prompts_seen": len(prompts),
        "records_written": written,
        "device": str(device),
        "output": str(args.output),
        "warning": "Pythia-160M is a base model; outputs may be low quality and must not be treated as verified facts.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
