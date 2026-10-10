"""Advanced reasoning primitives for Ron: weighted hypotheses, causal chains and counterfactual checks."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import Iterable

class ReasoningMode(str, Enum):
    DEDUCTIVE="deductive"
    INDUCTIVE="inductive"
    ABDUCTIVE="abductive"
    CAUSAL="causal"
    COUNTERFACTUAL="counterfactual"

@dataclass(frozen=True)
class WeightedEvidence:
    statement: str
    support: float
    reliability: float = 0.8
    source: str = "unknown"

    @property
    def strength(self) -> float:
        return max(0.0, min(1.0, self.support * self.reliability))

@dataclass(frozen=True)
class HypothesisScore:
    hypothesis: str
    support: float
    opposition: float
    confidence: float
    evidence: tuple[str, ...] = ()

@dataclass(frozen=True)
class CausalLink:
    cause: str
    effect: str
    strength: float = 0.7
    evidence: tuple[str, ...] = ()

@dataclass(frozen=True)
class CounterfactualResult:
    assumption: str
    expected_changes: tuple[str, ...]
    confidence: float
    blocked_by: tuple[str, ...] = ()

class AdvancedReasoner:
    """Deterministic evidence aggregation; no hidden chain-of-thought is stored."""

    def score_hypotheses(
        self,
        hypotheses: Iterable[str],
        evidence: Iterable[WeightedEvidence],
    ) -> tuple[HypothesisScore, ...]:
        ev=list(evidence)
        results=[]
        negation_prefixes = ("not ", "not:", "not-", "ليس ", "ليست ", "لا ", "لم ", "لن ")
        for hypothesis in hypotheses:
            h=hypothesis.strip()
            matching = [e for e in ev if h.casefold() in e.statement.casefold()]
            negative_evidence = [e for e in matching if e.statement.strip().casefold().startswith(negation_prefixes)]
            positive_evidence = [e for e in matching if e not in negative_evidence]
            positive=sum(e.strength for e in positive_evidence)
            negative=sum(e.strength for e in negative_evidence)
            total=positive+negative
            confidence=positive/total if total else 0.0
            results.append(HypothesisScore(h,min(1.0,positive),min(1.0,negative),min(1.0,confidence),tuple(e.statement for e in matching)))
        return tuple(sorted(results,key=lambda x:(x.confidence,x.support),reverse=True))

    @staticmethod
    def combine_independent_confidences(values: Iterable[float]) -> float:
        vals=[max(0.0,min(1.0,float(v))) for v in values]
        return 1.0-prod(1.0-v for v in vals) if vals else 0.0

    def causal_reason(self, links: Iterable[CausalLink], start: str, target: str) -> tuple[bool,float,tuple[str,...]]:
        graph={}
        for link in links:
            graph.setdefault(link.cause,[]).append(link)
        queue=[(start,1.0,(start,))]
        seen=set()
        while queue:
            node,confidence,path=queue.pop(0)
            if node==target:
                return True,confidence,path
            if node in seen: continue
            seen.add(node)
            for link in graph.get(node,()):
                queue.append((link.effect,confidence*max(0.0,min(1.0,link.strength)),path+(link.effect,)))
        return False,0.0,()

    def counterfactual(self, assumption: str, links: Iterable[CausalLink], known_effects: Iterable[str]) -> CounterfactualResult:
        effects=set(known_effects)
        predicted=[]
        blocked=[]
        for link in links:
            if link.cause.lower()==assumption.lower():
                if link.effect in effects:
                    predicted.append(link.effect)
                else:
                    predicted.append(link.effect+" (متوقع)")
        if not predicted:
            blocked.append("لا توجد علاقة سببية معروفة تربط الفرضية بالنتيجة.")
        confidence=max((l.strength for l in links if l.cause.lower()==assumption.lower()),default=0.2)
        return CounterfactualResult(assumption,tuple(predicted),confidence,tuple(blocked))
