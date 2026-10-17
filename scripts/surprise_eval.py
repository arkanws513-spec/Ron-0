"""Run blind, out-of-training-corpus probes against Ron-0's native checkpoint.

This is an evaluation harness, not training data. Prompts are intentionally kept
out of training/seed_corpus.txt and are never fed back into optimizer updates.
"""
from __future__ import annotations

import argparse
import json
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
    "Describe how to verify a conclusion from evidence.",
    "What should Ron do if a request is ambiguous?",
    "Explain why a model can overfit its training examples.",
    "Rewrite this statement as a question: water freezes at zero degrees Celsius.",
    "Give a short example of a cause and its effect.",
)

# Lightweight answer checks: these are deliberately separate from training data.
# They check whether each response contains evidence of the requested concept;
# they do not claim full semantic understanding.
ANSWER_SIGNALS = (
    ("الشمس", "زاوية", "ضوء", "sun", "angle"),
    ("سبب", "نتيجة", "cause", "effect"),
    ("يخالف", "تناقض", "لا تنطبق", "استثناء", "contradict", "exception"),
    ("لا أعرف", "لا اعرف", "مصدر", "uncertain", "don't know"),
    ("هل", "ماذا", "لماذا", "كيف", "what", "how", "do plants"),
    ("احتمال", "غير مؤكد", "ربما", "possibility", "uncertain"),
    ("لا", "خطأ", "ليس", "false", "incorrect"),
    ("تعاون", "مساعدة", "معًا", "teamwork", "help"),
    ("ملاحظة", "استنتاج", "سبب", "evidence", "inference"),
    ("كيف", "تتكون", "تتشكل", "form", "cloud"),
    ("evidence", "source", "compare", "مصدر", "أدلة"),
    ("توضيح", "استفسر", "سؤال", "clarify", "clarification"),
    ("training", "new", "examples", "تدريب", "أمثلة", "تعميم"),
    ("what", "how", "does", "is water", "هل", "كيف", "ماذا"),
    ("because", "cause", "effect", "سبب", "نتيجة"),
)
CRITICAL_CASES = {3, 4, 7, 11, 12}


def is_degenerate_answer(answer: str) -> bool:
    return is_degenerate_text(answer)


def evaluate(checkpoint: Path | None = None, *, neural_only: bool = False) -> dict:
    corpus = (ROOT / "training" / "seed_corpus.txt").read_text(encoding="utf-8")
    leaked = [prompt for prompt in BLIND_PROMPTS if prompt in corpus]
    if leaked:
        raise RuntimeError(f"Blind prompts found verbatim in training corpus: {leaked!r}")

    provider = NativeCheckpointProvider(checkpoint, neural_only=neural_only)
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
            "deterministic_core_response": bool(response.metadata.get("deterministic_core_response", False)),
            "curated_retrieval_used": bool(response.metadata.get("curated_retrieval_used", False)),
            "neural_only_generation": bool(response.metadata.get("neural_only_generation", False)),
            "external_model_used": bool(response.metadata.get("external_model_used", False)),
            "checkpoint": response.metadata.get("inference_checkpoint"),
            "selected_training_step": response.metadata.get("selected_training_step"),
            "generation_quality_fallback": bool(response.metadata.get("generation_quality_fallback", False)),
            "expected_signals": list(ANSWER_SIGNALS[index - 1]),
            "answer_check_passed": any(signal.casefold() in answer.casefold() for signal in ANSWER_SIGNALS[index - 1]),
            "critical": index in CRITICAL_CASES,
        })

    non_empty = sum(row["non_empty"] for row in rows)
    not_echo = sum(row["not_exact_echo"] for row in rows)
    unique_answers = len({row["answer"] for row in rows})
    stable_answers = sum(row["non_empty"] and row["not_exact_echo"] and not row["degenerate"] for row in rows)
    answer_checks_passed = sum(row["answer_check_passed"] for row in rows)
    critical_rows = [row for row in rows if row["critical"]]
    critical_checks_passed = bool(critical_rows) and all(row["answer_check_passed"] for row in critical_rows)
    answer_accuracy = answer_checks_passed / len(rows) if rows else 0.0
    anti_degeneration_gate_passed = bool(rows) and stable_answers / len(rows) >= 0.8
    quality_gate_passed = anti_degeneration_gate_passed and answer_accuracy >= 0.8 and critical_checks_passed
    return {
        "suite": "ron0-blind-surprise-v1",
        "inference_mode": "neural_only" if neural_only else "full_provider",
        "neural_only_requested": neural_only,
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
        "answer_checks_passed": answer_checks_passed,
        "answer_check_accuracy": answer_accuracy,
        "answer_check_threshold": 0.8,
        "critical_checks_passed": critical_checks_passed,
        "critical_cases": sorted(CRITICAL_CASES),
        "quality_gate_passed": quality_gate_passed,
        "native_checkpoint_loaded": all(row["native_weights_loaded"] for row in rows),
        "external_model_used": any(row["external_model_used"] for row in rows),
        "cases": rows,
        "interpretation": (
            "This tougher gate combines anti-degeneration checks with keyword-based answer signals and mandatory critical cases. "
            "Signal checks are approximate and can miss valid paraphrases or accept shallow keyword matches; manual review is still required. "
            "Passing is not evidence of factual correctness or general intelligence; keyword checks can miss valid paraphrases or accept shallow matches, so manual review is still required."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ron" / "checkpoints" / "ron_native_baseline.pt")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--neural-only", action="store_true", help="Bypass deterministic answers and curated retrieval to evaluate the native neural generator only.")
    args = parser.parse_args()
    report = evaluate(args.checkpoint, neural_only=args.neural_only)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["training_corpus_exact_prompt_leaks"] or not report["native_checkpoint_loaded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
