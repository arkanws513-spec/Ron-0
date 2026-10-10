"""Generate a reviewed-ready text corpus from a locally run Qwen2.5 teacher.

This script does not update Ron's weights. It creates JSONL records and a plain-text
corpus that can be used by Ron's native training recipe after human/data-quality review.
The teacher runs through Transformers locally; no hosted inference API is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED_PROMPTS = [
    "Explain the difference between a fact and an opinion, with one example.",
    "Give a careful, step-by-step explanation of why the seasons occur.",
    "Write a Python function that checks whether an integer is prime, and explain it.",
    "Explain what a neural network learns during training.",
    "How can a reader verify whether an online claim is reliable?",
    "Solve 17 * 23 and show a short verification.",
    "Explain the difference between training loss and validation loss.",
    "Summarize the water cycle for a middle-school student.",
    "Translate 'reliable evidence' into Arabic and explain the phrase.",
    "اشرح الفرق بين الحقيقة والرأي مع مثال بسيط.",
    "اشرح معنى التعلّم العميق بلغة عربية واضحة.",
    "كيف أتحقق من صحة خبر منتشر على الإنترنت؟",
    "حل المعادلة 3x + 7 = 22 واشرح خطوات الحل.",
    "اكتب دالة Python تجمع قائمة من الأعداد وتتحقق من أن القائمة غير فارغة.",
    "لخّص أهمية الاختبارات البرمجية في فقرة قصيرة.",
    "ما الفرق بين التدريب المسبق والضبط الدقيق؟",
    "اشرح دورة الماء لطفل في المرحلة الابتدائية.",
    "ترجم إلى الإنجليزية: يجب التحقق من المصدر قبل نشر الخبر.",
    "كيف يتعامل المساعد الجيد مع سؤال لا يعرف إجابته؟",
    "اكتب رسالة مهذبة للاعتذار عن التأخر في الرد.",
    "اشرح مفهوم التعميم في تعلم الآلة مع مثال.",
    "ما فائدة تقسيم البيانات إلى تدريب وتحقق واختبار؟",
    "كيف يمكن تنظيم خطة مذاكرة لمدة أسبوع؟",
    "اشرح لماذا لا يعني انخفاض خسارة التدريب بالضرورة أن النموذج أصبح أفضل.",
    "ما الفرق بين الذاكرة الخارجية وأوزان النموذج؟",
    "Give three safe ways to debug a Python program.",
    "Explain why model-generated text should be checked before becoming training data.",
    "Write a short, balanced comparison of solar and wind power.",
    "Explain the concept of probability using a fair six-sided die.",
    "What should a good assistant do when a request is ambiguous?",
]


def read_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(SEED_PROMPTS)
    if not path.is_file():
        raise FileNotFoundError(f"prompt file not found: {path}")
    prompts: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on prompt line {line_no}") from exc
            prompt = record.get("prompt") if isinstance(record, dict) else None
        else:
            prompt = line
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"line {line_no} must contain a non-empty prompt")
        prompts.append(prompt.strip())
    if not prompts:
        raise ValueError("prompt file contains no prompts")
    return prompts


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_outputs(records: list[dict], jsonl_path: Path, corpus_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Blank-line separated documents match scripts/train_native_10m.py's split logic.
    documents = [
        f"User: {item['prompt']}\nRon: {item['response']}"
        for item in records
    ]
    corpus_path.write_text("\n\n".join(documents) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local model directory")
    parser.add_argument("--prompts", type=Path, help="UTF-8 .txt prompts (one per line) or JSONL objects with a prompt field")
    parser.add_argument("--output-jsonl", type=Path, default=Path("training/distillation/qwen25_7b_teacher.jsonl"))
    parser.add_argument("--output-corpus", type=Path, default=Path("training/distillation/ron_distilled_corpus.txt"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, help="Process only the first N prompts (useful for a smoke test)")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--local-files-only", action="store_true", help="Do not download missing model/tokenizer files")
    parser.add_argument("--trust-remote-code", action="store_true", help="Explicitly allow model repository custom code")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("--temperature must be non-negative and --top-p must be in (0, 1]")
    random.seed(args.seed)
    prompts = read_prompts(args.prompts)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        prompts = prompts[:args.limit]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Distillation dependencies are missing. Install with: "
            "pip install -e '.[distill]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto",
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    records: list[dict] = []
    seen_pairs: set[str] = set()
    for index, prompt in enumerate(prompts, 1):
        messages = [
            {"role": "system", "content": "You are a careful teacher. Answer accurately, clearly, and in the language of the question. Do not invent sources. If uncertain, say so."},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        generation = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation.update(temperature=args.temperature, top_p=args.top_p)
        with torch.inference_mode():
            output_ids = model.generate(input_ids, **generation)
        response = tokenizer.decode(
            output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True
        )
        response = normalize(response)
        if not response:
            print(f"skip_empty_response index={index}")
            continue
        fingerprint = hashlib.sha256((normalize(prompt).casefold() + "\n" + response.casefold()).encode("utf-8")).hexdigest()
        if fingerprint in seen_pairs:
            continue
        seen_pairs.add(fingerprint)
        records.append({
            "prompt": normalize(prompt),
            "response": response,
            "teacher_model": args.model,
            "teacher_index": index,
            "seed": args.seed,
            "review_status": "needs_human_review",
        })
        if index % 10 == 0 or index == len(prompts):
            print(f"processed={index}/{len(prompts)} accepted={len(records)}", flush=True)

    if not records:
        raise RuntimeError("teacher produced no usable records; no output corpus was written")
    write_outputs(records, args.output_jsonl, args.output_corpus)
    print(json.dumps({
        "status": "data_generated_not_model_trained",
        "teacher_model": args.model,
        "prompts_seen": len(prompts),
        "records_written": len(records),
        "jsonl_path": str(args.output_jsonl),
        "corpus_path": str(args.output_corpus),
        "human_review_required": True,
        "native_training_required": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
