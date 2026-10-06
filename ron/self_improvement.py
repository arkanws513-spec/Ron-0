"""Auditable, bounded self-improvement primitives for Ron.

Ron can learn from recorded experience and promote a proposed skill only when
verification shows an improvement. This module never overwrites production
code or the base model by itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping


@dataclass(frozen=True)
class Experience:
    """A verified observation that can be turned into a reusable skill."""

    task: str
    outcome: str
    lesson: str
    success: bool = True


@dataclass(frozen=True)
class Skill:
    name: str
    instruction: str
    version: int = 1
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ImprovementCandidate:
    """A proposed replacement; it is inert until verification succeeds."""

    skill: Skill
    reason: str


@dataclass(frozen=True)
class Evaluation:
    baseline_score: float
    candidate_score: float
    tests_passed: bool

    @property
    def improved(self) -> bool:
        return self.tests_passed and self.candidate_score > self.baseline_score


@dataclass(frozen=True)
class PromotionRecord:
    skill_name: str
    version: int
    evaluation: Evaluation
    status: str


class SkillRegistry:
    """Versioned skill store with explicit promotion and rollback."""

    def __init__(self) -> None:
        self._active: dict[str, Skill] = {}
        self._history: dict[str, list[Skill | None]] = {}

    def get(self, name: str) -> Skill | None:
        return self._active.get(name)

    def promote(self, skill: Skill) -> None:
        previous = self._active.get(skill.name)
        self._history.setdefault(skill.name, []).append(previous)
        self._active[skill.name] = skill

    def rollback(self, name: str) -> Skill | None:
        history = self._history.get(name, [])
        if not history:
            return self._active.get(name)
        previous = history.pop()
        if previous is None:
            self._active.pop(name, None)
        else:
            self._active[name] = previous
        return self._active.get(name)

    def snapshot(self) -> dict[str, Skill]:
        return dict(self._active)


class SelfImprovementEngine:
    """Learn -> propose -> verify -> promote, with rollback on demand."""

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        max_experiences: int = 500,
    ) -> None:
        if max_experiences < 1:
            raise ValueError("max_experiences must be positive")
        self.registry = registry or SkillRegistry()
        self.experiences: list[Experience] = []
        self.max_experiences = max_experiences
        self.audit_log: list[PromotionRecord] = []

    def learn(self, experience: Experience) -> None:
        """Record bounded experience; failed experiences remain useful evidence."""
        self.experiences.append(experience)
        del self.experiences[:-self.max_experiences]

    def propose(
        self,
        experience: Experience,
        skill_name: str,
        proposer: Callable[[Experience], ImprovementCandidate],
    ) -> ImprovementCandidate:
        if not skill_name.strip():
            raise ValueError("skill_name cannot be empty")
        candidate = proposer(experience)
        if candidate.skill.name != skill_name:
            raise ValueError("proposer returned a different skill name")
        return candidate

    def verify_and_promote(
        self,
        candidate: ImprovementCandidate,
        baseline_score: float,
        candidate_score: float,
        tests_passed: bool,
    ) -> Evaluation:
        evaluation = Evaluation(
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            tests_passed=tests_passed,
        )
        status = "promoted" if evaluation.improved else "rejected"
        if evaluation.improved:
            self.registry.promote(candidate.skill)
        self.audit_log.append(
            PromotionRecord(
                skill_name=candidate.skill.name,
                version=candidate.skill.version,
                evaluation=evaluation,
                status=status,
            )
        )
        return evaluation

    def rollback(self, skill_name: str) -> Skill | None:
        return self.registry.rollback(skill_name)
