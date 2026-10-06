"""Small, deterministic natural-language fact layer for Ron.

This is intentionally conservative: it extracts only common first-person
facts that can be stored safely and reversibly. A real language model can
replace or extend this layer later without changing the memory contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import MemoryItem, MemoryStore

_NAME_PATTERNS = (
    re.compile(r"^\s*(?:اسمي|انا اسمي|أنا اسمي)\s+(.+?)\s*$", re.I),
    re.compile(r"^\s*اريدك(?: أن| ان)?\s+تعلم\s+أن\s+اسمي\s+هو\s+(.+?)\s*$", re.I),
)
_PREFERENCE_PATTERNS = (
    re.compile(r"^\s*(?:انا|أنا)\s+(?:احب|أحب)\s+(.+?)\s*$", re.I),
    re.compile(r"^\s*(?:انا|أنا)\s+(?:لا احب|لا أحب)\s+(.+?)\s*$", re.I),
)


@dataclass(frozen=True)
class ExtractedFact:
    key: str
    value: str
    kind: str

    @property
    def memory(self) -> MemoryItem:
        return MemoryItem(
            key=self.key,
            content=self.value,
            metadata={"kind": self.kind, "source": "natural-language"},
        )


def extract_fact(text: str) -> ExtractedFact | None:
    for pattern in _NAME_PATTERNS:
        match = pattern.match(text)
        if match:
            value = match.group(1).strip(" .،,؛;")
            if value:
                return ExtractedFact("user.name", value, "name")

    for pattern in _PREFERENCE_PATTERNS:
        match = pattern.match(text)
        if match:
            value = match.group(1).strip(" .،,؛;")
            if value:
                return ExtractedFact("user.preference", value, "preference")

    return None


def answer_fact_question(text: str, memory: MemoryStore) -> str | None:
    normalized = text.lower().strip()
    if "ما اسمي" in normalized or "ايه اسمي" in normalized or "إيه اسمي" in text:
        hits = memory.recall("user.name", limit=1)
        if hits:
            return f"اسمك {hits[0].content}."
        return "لم تخبرني باسمك بعد."

    if "ماذا تحب" in normalized or "ايه اللي بتحبه" in text or "ما الذي تحبه" in text:
        hits = memory.recall("user.preference", limit=5)
        if hits:
            return "أتذكر أنك تحب: " + "، ".join(item.content for item in hits) + "."
        return "لم تخبرني بتفضيلاتك بعد."

    return None
