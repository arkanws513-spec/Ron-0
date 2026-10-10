"""Run blind, out-of-training-corpus probes against Ron-0's native checkpoint.

This is an evaluation harness, not training data. Prompts are intentionally kept
out of training/seed_corpus.txt and are never fed back into optimizer updates.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from ron.contracts import Message, ModelRequest
from ron.native_provider import NativeCheckpointProvider
from ron.generation_quality import is_degenerate_text

ROOT = Path(__file__).resolve().parents[1]
BLIND_PROMPTS = (
    "لماذا يتغير طول الظل خلال اليوم؟",
    "اشرح الفرق بين السبب والنتيجة بمثال جديد.",
    "إذا كانت كل الصناديق مغلقة ورأينا صندوقًا مفتوحًا، فهل تنطبق القاعدة عليه؟",
    "ماذا ينبغي أن تقول عندما لا تعرف الإجابة بثقة؟",
    "حوّل هذه الفكرة إلى سؤال: النباتات تحتاج إلى الضوء.",
    "هل كلمة «ربما» تعبّر عن اليقين أم الاحتمال؟",
    "هل العبارة «اثنان زائد اثنين يساوي خمسة» صحيحة؟ وضّح باختصار.",
    "اكتب ردًا قصيرًا يشرح معنى التعاون لطالب جديد.",
    "ما الفرق بين ملاحظة حدث واستنتاج سببه؟",
    "أعد صياغة السؤال التالي دون تغيير معناه: كيف تتكوّن السحب؟",
)


def is_degenerate_answer(answer: str) -> bool:
    return is_degenerate_text(answer)


def evaluate(checkpoint: Path | None = None) -> dict:
    corpus = (ROOT / "training" / "seed_corpus.txt").read_text(encoding="utf-8")
    leaked = [prompt for prompt in BLIND_PROMPTS if prompt in corpus]
    if leaked:
        raise RuntimeError(f"Blind prompts found verbatim in training corpus: {leaked!r}")

    provider = NativeCheckpointProvider(checkpoint)
    rows = []
    for index, prompt in enumerate(BLIND_PROMPTS, start=1):
        request = ModelRequest(
            messages=(Message(role="user", content=prompt),),
            metadata={"evaluation": "blind-surprise-v1", "case": index},
        )
        response = provider.generate(request)
        answer = response.content.strip()
        rows.append({
            "case": index,
            "prompt": prompt,
            "answer": answer,
            "non_empty": bool(answer),
            "not_exact_echo": answer != prompt,
            "answer_length": len(answer),
            "degenerate": is_degenerate_answer(answer) or bool(response.metadata.get("native_generated_answer_rejected")),
            "model": response.model,
            "native_weights_loaded": bool(response.metadata.get("native_weights_loaded")),
            "external_model_used": bool(response.metadata.get("external_model_used", False)),
            "checkpoint": response.metadata.get("inference_checkpoint"),
            "selected_training_step": response.metadata.get("selected_training_step"),
            "generation_quality_fallback": bool(response.metadata.get("generation_quality_fallback", False)),
        })

    non_empty = sum(row["non_empty"] for row in rows)
    not_echo = sum(row["not_exact_echo"] for row in rows)
    unique_answers = len({row["answer"] for row in rows})
    stable_answers = sum(row["non_empty"] and row["not_exact_echo"] and not row["degenerate"] for row in rows)
    anti_degeneration_gate_passed = bool(rows) and stable_answers / len(rows) >= 0.8
    return {
        "suite": "ron0-blind-surprise-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_corpus_exact_prompt_leaks": len(leaked),
        "case_count": len(rows),
        "non_empty_answers": non_empty,
        "non_echo_answers": not_echo,
        "unique_answer_count": unique_answers,
        "degenerate_answers": sum(row["degenerate"] for row in rows),
        "stable_answers": stable_answers,
        "anti_degeneration_gate_threshold": 0.8,
        "anti_degeneration_gate_passed": anti_degeneration_gate_passed,
        "native_checkpoint_loaded": all(row["native_weights_loaded"] for row in rows),
        "external_model_used": any(row["external_model_used"] for row in rows),
        "cases": rows,
        "interpretation": (
            "These checks measure runtime robustness and prompt coverage, not factual "
            "accuracy or general intelligence. The anti-degeneration gate only catches obvious repetition; "
            "passing it is not evidence of factual correctness or general intelligence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ron" / "checkpoints" / "ron_native_baseline.pt")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(args.checkpoint)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["training_corpus_exact_prompt_leaks"] or not report["native_checkpoint_loaded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
