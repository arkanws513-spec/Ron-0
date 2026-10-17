"""Blind bilingual generation evaluation for a Qwen + LoRA candidate."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def normalize(text: str) -> str:
    value = str(text or "").casefold()
    value = value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    value = value.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", value).strip()


def classify_language(case: dict[str, Any]) -> str:
    explicit = str(case.get("language", "")).strip().lower()
    if explicit in {"ar", "arabic"}:
        return "ar"
    if explicit in {"en", "english"}:
        return "en"
    prompt = str(case.get("prompt", ""))
    if not prompt and case.get("messages"):
        prompt = " ".join(
            str(item.get("content", "")) for item in case["messages"]
            if isinstance(item, dict) and item.get("role") == "user"
        )
    for prefix in ("أجب بدقة وباختصار عن السؤال التالي:", "اقرأ السؤال جيدًا ثم أعط إجابة مباشرة:"):
        if prompt.startswith(prefix):
            prompt = prompt[len(prefix):].lstrip(" :：")
            break
    arabic = sum("\u0600" <= char <= "\u06ff" for char in prompt)
    latin = sum(char.isascii() and char.isalpha() for char in prompt)
    return "ar" if arabic >= latin else "en"


def assess(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    actual = normalize(answer)
    missing = [str(item) for item in case.get("contains", []) if normalize(item) not in actual]
    forbidden = [str(item) for item in case.get("forbidden", []) if normalize(item) in actual]
    words = actual.split()
    degenerate = not actual or len(actual) < 2 or len(set(words)) <= 1
    if len(words) >= 12 and len(set(zip(words, words[1:], words[2:]))) <= 2:
        degenerate = True
    return {
        "id": case.get("id", case.get("prompt", "unnamed")),
        "language": classify_language(case),
        "critical": bool(case.get("critical", False)),
        "passed": not missing and not forbidden and not degenerate,
        "missing_required": missing,
        "forbidden_found": forbidden,
        "degenerate": degenerate,
        "answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", type=Path, default=ROOT / "artifacts/qwen-lora-candidate/adapter")
    parser.add_argument("--cases", type=Path, default=ROOT / "training/qwen_eval_cases.json")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts/qwen-lora-candidate/evaluation.json")
    parser.add_argument("--min-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-language-pass-rate", type=float, default=0.80)
    parser.add_argument("--expected-case-count", type=int, default=300)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != args.expected_case_count:
        raise SystemExit(f"Expected exactly {args.expected_case_count} evaluation cases; found {len(cases) if isinstance(cases, list) else 'non-list'}.")
    if sum(bool(case.get("critical")) for case in cases) < 40:
        raise SystemExit("Evaluation set must include at least 40 critical cases.")
    if not args.adapter.is_dir():
        raise SystemExit(f"Adapter directory not found: {args.adapter}")
    if not torch.cuda.is_available():
        raise SystemExit("Blind evaluation requires a CUDA GPU so the same evaluation can be run reproducibly without an excessive CPU wait.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).cuda()
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    results = []
    for index, case in enumerate(cases, start=1):
        raw_messages = case.get("messages")
        messages = (
            [{"role": item["role"], "content": item["content"]} for item in raw_messages]
            if raw_messages else [{"role": "user", "content": str(case.get("prompt", ""))}]
        )
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results.append(assess(answer, case))
        if index % 25 == 0:
            print(json.dumps({"evaluated": index, "total": len(cases)}, ensure_ascii=False), flush=True)

    total = len(results)
    passed = sum(item["passed"] for item in results)
    language_metrics = {}
    for language in ("ar", "en"):
        group = [item for item in results if item["language"] == language]
        language_metrics[language] = {
            "case_count": len(group),
            "passed": sum(item["passed"] for item in group),
            "pass_rate": sum(item["passed"] for item in group) / max(1, len(group)),
        }
    critical = [item for item in results if item["critical"]]
    critical_rate = sum(item["passed"] for item in critical) / max(1, len(critical))
    passed_gate = (
        passed / total >= args.min_pass_rate
        and all(metric["case_count"] > 0 and metric["pass_rate"] >= args.min_language_pass_rate for metric in language_metrics.values())
        and critical_rate >= args.min_pass_rate
    )
    report = {
        "passed": passed_gate,
        "case_count": total,
        "passed_count": passed,
        "pass_rate": passed / total,
        "critical_case_count": len(critical),
        "critical_pass_rate": critical_rate,
        "min_pass_rate": args.min_pass_rate,
        "min_language_pass_rate": args.min_language_pass_rate,
        "language_metrics": language_metrics,
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "evaluation_mode": "qwen_lora_neural_only",
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "passed", "case_count", "passed_count", "pass_rate", "critical_case_count",
        "critical_pass_rate", "language_metrics", "evaluation_mode"
    )}, ensure_ascii=False, indent=2))
    if not passed_gate:
        raise SystemExit("Blind bilingual generation gate failed; do not merge or deploy this candidate.")


if __name__ == "__main__":
    main()
