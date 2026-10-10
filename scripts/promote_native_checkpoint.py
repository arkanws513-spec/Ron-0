"""Promote a Ron-native checkpoint only after loss and blind conversation gates pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION = ROOT / "ron" / "checkpoints" / "ron_native_10m.pt"


def validate_promotion(candidate: Path, report_path: Path, metrics_path: Path) -> tuple[bool, str]:
    if not candidate.is_file():
        return False, "candidate_checkpoint_missing"
    if not report_path.is_file() or not metrics_path.is_file():
        return False, "evaluation_report_or_training_metrics_missing"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        return False, "blind_conversational_gate_failed"
    if metrics.get("validation_loss_improved") is not True:
        return False, "validation_loss_gate_failed"
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest != report.get("checkpoint_sha256"):
        return False, "candidate_hash_does_not_match_evaluated_checkpoint"
    if int(report.get("parameter_count", 0)) < 1_000_000:
        return False, "candidate_below_1m_parameter_minimum"
    return True, "all_promotion_gates_passed"


def promote(candidate: Path, report_path: Path, metrics_path: Path, destination: Path) -> None:
    allowed, reason = validate_promotion(candidate, report_path, metrics_path)
    if not allowed:
        raise SystemExit(f"checkpoint promotion denied: {reason}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as handle:
            temp_path = Path(handle.name)
        temp_path.write_bytes(candidate.read_bytes())
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    print(json.dumps({"promoted": True, "reason": reason, "destination": str(destination)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    promote(args.candidate, args.report, args.metrics, args.destination)


if __name__ == "__main__":
    main()
