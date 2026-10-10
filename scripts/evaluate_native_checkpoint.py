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


def validate_cases(cases: Any) -> list[dict[str, Any]]:
    if not isinstance(cases, list) or len(cases) < MIN_EVAL_CASES:
        raise ValueError(f"evaluation requires at least {MIN_EVAL_CASES} cases")
    if any(not isinstance(case, dict) for case in cases):
        raise ValueError("every evaluation case must be an object")
    ids = [str(case.get("id", case.get("prompt", ""))).strip() for case in cases]
    if any(not case_id for case_id in ids):
        raise ValueError("every evaluation case needs a non-empty id or prompt")
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case IDs must be unique")
    critical_count = sum(bool(case.get("critical", False)) for case in cases)
    if critical_count < MIN_CRITICAL_CASES:
        raise ValueError(f"evaluation requires at least {MIN_CRITICAL_CASES} critical cases")
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
        "passed": passed,
        "missing_required": missing,
        "forbidden_found": found_forbidden,
        "empty": empty,
        "degenerate": degenerate,
        "answer": str(answer),
    }


def evaluate_checkpoint(checkpoint_path: Path, cases_path: Path, report_path: Path, min_pass_rate: float) -> dict[str, Any]:
    cases = validate_cases(json.loads(cases_path.read_text(encoding="utf-8")))
    provider = NativeCheckpointProvider(checkpoint_path)
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
    with checkpoint_path.open("rb") as checkpoint_file:
        checkpoint_hash = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    report = {
        "status": "completed",
        "passed": pass_rate >= min_pass_rate and not critical_failures,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_hash,
        "parameter_count": parameter_count,
        "case_count": len(results),
        "passed_count": passed_count,
        "pass_rate": pass_rate,
        "required_pass_rate": min_pass_rate,
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
    parser.add_argument("--min-pass-rate", type=float, default=0.70)
    args = parser.parse_args()
    if not 0 < args.min_pass_rate <= 1:
        raise ValueError("--min-pass-rate must be in (0, 1]")
    report = evaluate_checkpoint(args.checkpoint, args.cases, args.report, args.min_pass_rate)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
