"""Evaluation contracts for native Ron model candidates."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence

@dataclass(frozen=True)
class EvalCase:
    prompt: str
    expected: str

@dataclass(frozen=True)
class EvalReport:
    score: float
    passed: int
    total: int

    @property
    def passed_all(self) -> bool:
        return self.total > 0 and self.passed == self.total

def evaluate_exact(cases: Sequence[EvalCase], responder: Callable[[str], str]) -> EvalReport:
    passed = sum(responder(case.prompt).strip() == case.expected.strip() for case in cases)
    total = len(cases)
    return EvalReport(score=(passed / total if total else 0.0), passed=passed, total=total)
