import hashlib
import json

from scripts.evaluate_native_checkpoint import assess_case, validate_cases
from scripts.promote_native_checkpoint import validate_promotion


def test_conversation_case_requires_expected_answer_and_rejects_irrelevant_biography():
    good = assess_case(
        "عاصمة مصر هي القاهرة.",
        {"id": "egypt", "contains": ["القاهرة"], "forbidden": ["كريستا ثيريت"], "critical": True},
    )
    bad = assess_case(
        "كريستا ثيريت ممثلة فرنسية.",
        {"id": "egypt", "contains": ["القاهرة"], "forbidden": ["كريستا ثيريت"], "critical": True},
    )
    assert good["passed"] is True
    assert bad["passed"] is False
    assert bad["missing_required"] == ["القاهرة"]
    assert bad["forbidden_found"] == ["كريستا ثيريت"]


def test_promotion_requires_blind_eval_and_validation_loss_improvement(tmp_path):
    candidate = tmp_path / "candidate.pt"
    report_path = tmp_path / "eval.json"
    metrics_path = tmp_path / "metrics.json"
    candidate.write_bytes(b"reviewed-candidate")
    report = {
        "passed": False,
        "checkpoint_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "parameter_count": 11_000_000,
    }
    metrics = {"validation_loss_improved": True}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    allowed, reason = validate_promotion(candidate, report_path, metrics_path)
    assert allowed is False
    assert reason == "blind_conversational_gate_failed"

    report["passed"] = True
    metrics["validation_loss_improved"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    allowed, reason = validate_promotion(candidate, report_path, metrics_path)
    assert allowed is False
    assert reason == "validation_loss_gate_failed"


def test_promotion_requires_the_exact_evaluated_checkpoint_hash(tmp_path):
    candidate = tmp_path / "candidate.pt"
    report_path = tmp_path / "eval.json"
    metrics_path = tmp_path / "metrics.json"
    candidate.write_bytes(b"candidate")
    report_path.write_text(json.dumps({
        "passed": True,
        "checkpoint_sha256": "wrong-hash",
        "parameter_count": 11_000_000,
    }), encoding="utf-8")
    metrics_path.write_text(json.dumps({"validation_loss_improved": True}), encoding="utf-8")
    allowed, reason = validate_promotion(candidate, report_path, metrics_path)
    assert allowed is False
    assert reason == "candidate_hash_does_not_match_evaluated_checkpoint"


def test_promotion_accepts_only_a_matching_candidate_that_passed_both_gates(tmp_path):
    candidate = tmp_path / "candidate.pt"
    report_path = tmp_path / "eval.json"
    metrics_path = tmp_path / "metrics.json"
    candidate.write_bytes(b"candidate")
    report_path.write_text(json.dumps({
        "passed": True,
        "checkpoint_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "parameter_count": 11_000_000,
    }), encoding="utf-8")
    metrics_path.write_text(json.dumps({"validation_loss_improved": True}), encoding="utf-8")
    allowed, reason = validate_promotion(candidate, report_path, metrics_path)
    assert allowed is True
    assert reason == "all_promotion_gates_passed"


def test_blind_evaluation_requires_enough_cases_and_critical_cases():
    too_small = [
        {"id": str(i), "prompt": "question", "contains": ["answer"], "critical": i < 2}
        for i in range(4)
    ]
    try:
        validate_cases(too_small)
    except ValueError as exc:
        assert "at least 20 cases" in str(exc)
    else:
        raise AssertionError("small evaluation set must be rejected")

    enough = [
        {"id": str(i), "prompt": "question", "contains": ["answer"], "critical": i < 8}
        for i in range(20)
    ]
    assert len(validate_cases(enough)) == 20

    not_enough_critical = [
        {"id": str(i), "prompt": "question", "contains": ["answer"], "critical": i < 7}
        for i in range(20)
    ]
    try:
        validate_cases(not_enough_critical)
    except ValueError as exc:
        assert "at least 8 critical cases" in str(exc)
    else:
        raise AssertionError("evaluation set with too few critical cases must be rejected")


def test_repository_blind_evaluation_set_has_unique_ids_and_required_coverage():
    from pathlib import Path

    cases_path = Path(__file__).resolve().parents[1] / "training" / "eval_cases.json"
    cases = validate_cases(json.loads(cases_path.read_text(encoding="utf-8")))
    ids = {case["id"] for case in cases}
    assert {"egypt_capital_short", "egypt_capital_variant", "france_capital", "context_followup"} <= ids
    assert sum(bool(case.get("critical")) for case in cases) >= 8
