"""Blind conversational evaluation for Ron-native checkpoint candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ron.contracts import Message, ModelRequest
from ron.generation_quality import is_degenerate_text
from ron.native_provider import NativeCheckpointProvider

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "training" / "eval_cases.json"
DEFAULT_REPORT = ROOT / "artifacts" / "native-10m" / "conversational_eval.json"


def normalize(text: str) -> str:
    value = str(text or "").casefold()
    value = value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    value = value.replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", value).strip()


MIN_EVAL_CASES = 20
MIN_CRITICAL_CASES = 8
HARD_EVAL_CASE_COUNT = 300
HARD_CRITICAL_CASES = 40
ARABIC_INSTRUCTIONS = (
    "أجب بدقة وباختصار عن السؤال التالي:",
    "اقرأ السؤال جيدًا ثم أعط إجابة مباشرة:",
)


def classify_language(case: dict[str, Any]) -> str:
    """Classify the actual question, not Arabic wrapper instructions around English questions."""
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
    prompt = prompt.strip()
    for prefix in ARABIC_INSTRUCTIONS:
        if prompt.startswith(prefix):
            prompt = prompt[len(prefix):].lstrip(" :：")
            break
    arabic_count = sum("\u0600" <= char <= "\u06ff" for char in prompt)
    latin_count = sum(char.isascii() and char.isalpha() for char in prompt)
    return "ar" if arabic_count >= latin_count else "en"


def validate_cases(cases: Any, min_cases: int = MIN_EVAL_CASES, min_critical: int = MIN_CRITICAL_CASES) -> list[dict[str, Any]]:
    if not isinstance(cases, list) or len(cases) < min_cases:
        raise ValueError(f"evaluation requires at least {min_cases} cases")
    if any(not isinstance(case, dict) for case in cases):
        raise ValueError("every evaluation case must be an object")
    ids = [str(case.get("id", case.get("prompt", ""))).strip() for case in cases]
    if any(not case_id for case_id in ids):
        raise ValueError("every evaluation case needs a non-empty id or prompt")
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case IDs must be unique")
    critical_count = sum(bool(case.get("critical", False)) for case in cases)
    if critical_count < min_critical:
        raise ValueError(f"evaluation requires at least {min_critical} critical cases")
    for case in cases:
        if not case.get("messages") and not str(case.get("prompt", "")).strip():
            raise ValueError(f"evaluation case {case.get('id', '<unknown>')} needs a prompt or messages")
        if not case.get("contains"):
            raise ValueError(f"evaluation case {case.get('id', '<unknown>')} needs expected answer fragments")
    return cases


def assess_case(answer: str, case: dict[str, Any]) -> dict[str, Any]:
    actual = normalize(answer)
    required = [str(item) for item in case.get("contains", [])]
    forbidden = [str(item) for item in case.get("forbidden", [])]
    missing = [item for item in required if normalize(item) not in actual]
    found_forbidden = [item for item in forbidden if normalize(item) in actual]
    empty = not actual
    degenerate = is_degenerate_text(answer)
    rejected = "لم اتمكن من صياغه اجابه موثوقه" in actual or "generation_quality_fallback" in actual
    passed = not (missing or found_forbidden or empty or degenerate or rejected)
    return {
        "id": case.get("id", case.get("prompt", "unnamed")),
        "prompt": case.get("prompt", ""),
        "critical": bool(case.get("critical", False)),
        "language": classify_language(case),
        "passed": passed,
        "missing_required": missing,
        "forbidden_found": found_forbidden,
        "empty": empty,
        "degenerate": degenerate,
        "answer": str(answer),
    }


def evaluate_checkpoint(
    checkpoint_path: Path,
    cases_path: Path,
    report_path: Path,
    min_pass_rate: float,
    expected_case_count: int = HARD_EVAL_CASE_COUNT,
    min_language_pass_rate: float = 0.80,
    neural_only: bool = False,
) -> dict[str, Any]:
    loaded_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_cases, list) or len(loaded_cases) != expected_case_count:
        actual_count = len(loaded_cases) if isinstance(loaded_cases, list) else "non-list"
        raise ValueError(f"hard evaluation requires exactly {expected_case_count} cases; found {actual_count}")
    cases = validate_cases(loaded_cases, min_cases=expected_case_count, min_critical=HARD_CRITICAL_CASES)
    provider = NativeCheckpointProvider(checkpoint_path, neural_only=neural_only)
    parameter_count = sum(parameter.numel() for parameter in provider.model.parameters())
    results = []
    for case in cases:
        raw_messages = case.get("messages")
        if raw_messages:
            messages = tuple(
                Message(role=item["role"], content=item["content"])
                for item in raw_messages
            )
        else:
            messages = (Message(role="user", content=str(case["prompt"])),)
        response = provider.generate(ModelRequest(messages=messages, metadata={"evaluation": True}))
        results.append(assess_case(response.content, case) | {
            "model_metadata": response.metadata,
        })

    passed_count = sum(bool(item["passed"]) for item in results)
    pass_rate = passed_count / len(results)
    critical_failures = [
        item["id"] for item in results if item["critical"] and not item["passed"]
    ]
    language_metrics = {}
    for language in ("ar", "en"):
        subset = [item for item in results if item["language"] == language]
        language_passed = sum(bool(item["passed"]) for item in subset)
        language_metrics[language] = {
            "case_count": len(subset),
            "passed_count": language_passed,
            "pass_rate": language_passed / len(subset) if subset else 0.0,
            "required_pass_rate": min_language_pass_rate,
            "failed_ids": [item["id"] for item in subset if not item["passed"]],
        }
    both_languages_passed = all(
        metric["case_count"] > 0 and metric["pass_rate"] >= min_language_pass_rate
        for metric in language_metrics.values()
    )
    with checkpoint_path.open("rb") as checkpoint_file:
        checkpoint_hash = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    report = {
        "status": "completed",
        "passed": (
            pass_rate >= min_pass_rate
            and both_languages_passed
            and not critical_failures
            and len(results) == expected_case_count
            and sum(bool(case.get("critical", False)) for case in cases) >= HARD_CRITICAL_CASES
        ),
        "evaluation_mode": "neural_only" if neural_only else "full_stack",
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_hash,
        "parameter_count": parameter_count,
        "case_count": len(results),
        "critical_case_count": sum(bool(case.get("critical", False)) for case in cases),
        "passed_count": passed_count,
        "pass_rate": pass_rate,
        "required_pass_rate": min_pass_rate,
        "min_language_pass_rate": min_language_pass_rate,
        "language_metrics": language_metrics,
        "critical_failures": critical_failures,
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-pass-rate", type=float, default=0.80)
    parser.add_argument("--min-language-pass-rate", type=float, default=0.80)
    parser.add_argument("--expected-case-count", type=int, default=HARD_EVAL_CASE_COUNT)
    parser.add_argument(
        "--neural-only",
        action="store_true",
        help="Bypass curated rules and retrieval so the checkpoint's own generation is tested.",
    )
    args = parser.parse_args()
    if not 0 < args.min_pass_rate <= 1:
        raise ValueError("--min-pass-rate must be in (0, 1]")
    if not 0 < args.min_language_pass_rate <= 1:
        raise ValueError("--min-language-pass-rate must be in (0, 1]")
    report = evaluate_checkpoint(
        args.checkpoint,
        args.cases,
        args.report,
        args.min_pass_rate,
        args.expected_case_count,
        args.min_language_pass_rate,
        args.neural_only,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
