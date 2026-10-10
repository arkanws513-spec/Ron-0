"""Run Ron against the versioned 25-case dialogue benchmark using Ron's native checkpoint."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ron.core import RonCore
from ron.intent import normalize_arabic
from ron.native_provider import NativeCheckpointProvider


def normalize_match(text: str) -> str:
    value = normalize_arabic(text)
    return re.sub(r"[\W_]+", " ", value, flags=re.UNICODE).strip()


def run_benchmark(benchmark_path: Path) -> dict:
    benchmark_path = benchmark_path.resolve()
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    bootstrap = RonCore()
    provider = bootstrap.provider
    if not isinstance(provider, NativeCheckpointProvider):
        raise RuntimeError(
            "Native checkpoint is not loaded; refusing to measure fallback behavior as model intelligence. "
            f"Provider selected: {type(provider).__name__}; warnings={bootstrap.provider_startup_warnings}"
        )
    parameter_count = sum(parameter.numel() for parameter in provider.model.parameters())
    sessions = {"default": bootstrap}
    results = []
    for case in benchmark["cases"]:
        session = case.get("session", "default")
        core = sessions.get(session)
        if core is None:
            core = RonCore(provider=provider)
            sessions[session] = core
        response = core.respond(case["prompt"])
        answer = response.content or ""
        normalized = normalize_match(answer)
        any_expected = case.get("contains_any", [])
        all_expected = case.get("contains_all", [])
        any_ok = not any_expected or any(normalize_match(token) in normalized for token in any_expected)
        all_ok = all(normalize_match(token) in normalized for token in all_expected)
        quality_guard_rejected = bool(
            (response.metadata or {}).get("generation_quality_fallback")
            or (response.metadata or {}).get("native_generated_answer_rejected")
            or (response.metadata or {}).get("fallback")
        )
        passed = bool(answer.strip()) and not quality_guard_rejected and any_ok and all_ok
        results.append({
            "id": case["id"],
            "session": session,
            "prompt": case["prompt"],
            "answer": answer,
            "model": response.model,
            "critical": bool(case.get("critical", False)),
            "passed": passed,
            "quality_guard_rejected": quality_guard_rejected,
            "expected_any": any_expected,
            "expected_all": all_expected,
            "metadata": response.metadata,
        })
    total = len(results)
    critical = [item for item in results if item["critical"]]
    passed_count = sum(item["passed"] for item in results)
    critical_passed = sum(item["passed"] for item in critical)
    native_generation = [item for item in results if item["model"] == "ron-native-checkpoint"]
    native_generation_passed = sum(item["passed"] for item in native_generation)
    curated_retrieval = [item for item in results if item["model"] == "ron-curated-knowledge"]
    deterministic = [item for item in results if item["model"] in {"ron-arithmetic", "ron-algebra", "ron-memory", "ron-reasoning"}]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(benchmark_path.relative_to(ROOT)),
        "runtime": "ron-0-python-core",
        "native_weights_loaded": True,
        "external_model_used": False,
        "checkpoint": str(provider.checkpoint_path.relative_to(ROOT)),
        "checkpoint_source": provider.checkpoint_source,
        "inference_checkpoint": provider.inference_checkpoint,
        "selected_training_step": provider.selected_step,
        "training_steps_total": provider.training_steps_total,
        "parameters": parameter_count,
        "total_cases": total,
        "passed": passed_count,
        "score": round(passed_count / max(1, total), 4),
        "critical_cases": len(critical),
        "critical_passed": critical_passed,
        "critical_score": round(critical_passed / max(1, len(critical)), 4),
        "native_generation_cases": len(native_generation),
        "native_generation_passed": native_generation_passed,
        "native_generation_score": round(native_generation_passed / max(1, len(native_generation)), 4) if native_generation else None,
        "curated_retrieval_cases": len(curated_retrieval),
        "deterministic_tool_cases": len(deterministic),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "training" / "conversation_benchmark.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "conversation-eval-report.json")
    parser.add_argument("--strict", action="store_true", help="exit non-zero when either score is below the configured thresholds")
    parser.add_argument("--min-score", type=float, default=0.8)
    parser.add_argument("--min-critical-score", type=float, default=0.9)
    args = parser.parse_args()
    report = run_benchmark(args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("runtime", "native_weights_loaded", "checkpoint", "inference_checkpoint", "selected_training_step", "parameters", "total_cases", "passed", "score", "critical_cases", "critical_passed", "critical_score", "native_generation_cases", "native_generation_passed", "native_generation_score", "curated_retrieval_cases", "deterministic_tool_cases")}, ensure_ascii=False, indent=2))
    for item in report["results"]:
        mark = "PASS" if item["passed"] else "FAIL"
        print(f"{mark} [{item['id']}] {item['prompt']} -> {item['answer']}")
    if args.strict and (report["score"] < args.min_score or report["critical_score"] < args.min_critical_score):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
