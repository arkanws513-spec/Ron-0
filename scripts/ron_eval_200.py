"""Blind 200-case robustness exam for Ron's native checkpoint.

The exam is held out from optimizer updates. Its 70% gate measures usable,
non-echoing, non-degenerate output and native-checkpoint provenance; it is not
a factual-accuracy benchmark.
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
PASS_RATE = 0.70

BASE_PROMPTS = (
    "لماذا يتغير طول الظل خلال اليوم؟",
    "اشرح الفرق بين السبب والنتيجة بمثال جديد.",
    "كيف تتحقق من صحة معلومة قبل نشرها؟",
    "ماذا تقول عندما لا تعرف الإجابة بثقة؟",
    "حوّل العبارة النباتات تحتاج إلى الضوء إلى سؤال.",
    "هل كلمة ربما تعبّر عن اليقين أم الاحتمال؟",
    "هل اثنان زائد اثنين يساوي خمسة؟ وضح بإيجاز.",
    "اكتب ردًا قصيرًا يشرح معنى التعاون لطالب جديد.",
    "ما الفرق بين ملاحظة حدث واستنتاج سببه؟",
    "أعد صياغة سؤال كيف تتكون السحب دون تغيير معناه.",
    "كيف يمكن مقارنة خيارين باستخدام أدلة واضحة؟",
    "لماذا يجب التمييز بين الرأي والحقيقة؟",
    "ما الخطوة الأولى لحل مسألة غير واضحة؟",
    "اشرح مفهوم الطاقة بمثال من الحياة اليومية.",
    "كيف تؤثر درجة الحرارة في تبخر الماء؟",
    "لماذا لا يثبت الارتباط وحده وجود علاقة سببية؟",
    "كيف تلخص نصًا طويلًا دون فقد فكرته الأساسية؟",
    "ما فائدة طرح سؤال توضيحي عند الغموض؟",
    "كيف تتعامل مع معلومتين متعارضتين؟",
    "اذكر مثالًا على استنتاج لا تدعمه الأدلة.",
    "ما الفرق بين البيانات والتفسير؟",
    "كيف تشرح فكرة صعبة لطفل بطريقة بسيطة؟",
    "لماذا نحتاج إلى مصادر متعددة عند البحث؟",
    "كيف تصحح إجابة سابقة إذا اكتشفت خطأ؟",
    "ما معنى أن تكون النتيجة غير مؤكدة؟",
    "Explain how to verify a claim before sharing it.",
    "What is the difference between a cause and a correlation?",
    "What should an assistant do when a question is ambiguous?",
    "Give an example of a fair comparison between two choices.",
    "Why is it important to admit uncertainty?",
    "Explain how plants use sunlight in simple language.",
    "How can you distinguish an observation from an inference?",
    "Describe one limitation of a small language model.",
    "How would you summarize a long explanation accurately?",
    "What should you do when two sources disagree?",
    "Give a simple example of cause and effect.",
    "Why does a model need held-out evaluation data?",
    "How can a mistaken answer be corrected responsibly?",
    "Explain why repetition does not prove that a claim is true.",
    "What information would you ask for to clarify a vague request?",
)
TASK_FRAMES = (
    "أجب مباشرة وباختصار عن الطلب التالي: {prompt}",
    "اشرح الطلب التالي بمثال مناسب، ولا تخترع حقائق: {prompt}",
    "قدّم إجابة من جملتين، واذكر حدود اليقين عند الحاجة: {prompt}",
    "حلل المطلوب ثم أجب بوضوح وبأسلوب مساعد: {prompt}",
    "أجب عن الطلب التالي بلغة بسيطة، وتجنب تكرار السؤال: {prompt}",
)
EXAM = tuple(
    {"case": i + 1, "prompt": frame.format(prompt=base)}
    for i, (base, frame) in enumerate(
        ( (base, frame) for frame in TASK_FRAMES for base in BASE_PROMPTS )
    )
)


def corpus_texts() -> list[str]:
    candidates = [
        ROOT / "training" / "seed_corpus.txt",
        ROOT / "training" / "arabic_sft_seed.txt",
        ROOT / "training" / "corpus" / "oasst1" / "english_dialogues.txt",
    ]
    return [path.read_text(encoding="utf-8") for path in candidates if path.is_file()]


def evaluate(checkpoint: Path) -> dict:
    if len(EXAM) != 200:
        raise RuntimeError(f"Exam must contain exactly 200 cases; found {len(EXAM)}")
    corpora = corpus_texts()
    leaked = [item["case"] for item in EXAM if any(item["prompt"] in corpus for corpus in corpora)]
    if leaked:
        raise RuntimeError(f"Blind prompts leaked into training text: cases {leaked}")

    provider = NativeCheckpointProvider(checkpoint)
    rows = []
    for item in EXAM:
        request = ModelRequest(
            messages=(Message(role="user", content=item["prompt"]),),
            metadata={"evaluation": "ron-blind-200-v1", "case": item["case"]},
        )
        response = provider.generate(request)
        answer = response.content.strip()
        degenerate = is_degenerate_text(answer) or bool(
            response.metadata.get("native_generated_answer_rejected", False)
        )
        stable = bool(answer) and answer != item["prompt"] and not degenerate
        rows.append({
            "case": item["case"],
            "prompt": item["prompt"],
            "answer": answer,
            "non_empty": bool(answer),
            "not_exact_echo": answer != item["prompt"],
            "degenerate": degenerate,
            "stable_usable_answer": stable,
            "native_weights_loaded": bool(response.metadata.get("native_weights_loaded")),
            "external_model_used": bool(response.metadata.get("external_model_used", False)),
            "checkpoint_source": response.metadata.get("checkpoint_source"),
        })

    passed = sum(row["stable_usable_answer"] for row in rows)
    rate = passed / len(rows)
    native_loaded = all(row["native_weights_loaded"] for row in rows)
    external_used = any(row["external_model_used"] for row in rows)
    gate_passed = rate >= PASS_RATE and native_loaded and not external_used and not leaked
    return {
        "suite": "ron-blind-200-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(rows),
        "passed_cases": passed,
        "failed_cases": len(rows) - passed,
        "pass_rate": round(rate, 4),
        "required_pass_rate": PASS_RATE,
        "minimum_passes_required": int(len(rows) * PASS_RATE),
        "degenerate_answers": sum(row["degenerate"] for row in rows),
        "training_prompt_leaks": len(leaked),
        "native_checkpoint_loaded_for_all_cases": native_loaded,
        "external_model_used": external_used,
        "gate_passed": gate_passed,
        "cases": rows,
        "scope_note": (
            "This is a generation-stability and provenance gate, not a semantic "
            "correctness score. Passing does not establish factual accuracy or general intelligence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
