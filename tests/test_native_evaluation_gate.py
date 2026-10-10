import hashlib
import json

from scripts.evaluate_native_checkpoint import assess_case
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
