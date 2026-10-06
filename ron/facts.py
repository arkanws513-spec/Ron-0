"""Conservative natural-language fact extraction and recall for Ron."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .contracts import MemoryItem, MemoryStore

_ARABIC_DIACRITICS = re.compile(r"[ًٌٍَُِّْـ]")
_SPACES = re.compile(r"\s+")

_NAME_PATTERNS = (
    re.compile(r"^\s*(?:اسمي|انا اسمي)\s+(.+?)\s*$", re.I),
    re.compile(r"^\s*اريدك(?: أن| ان)?\s+تعلم\s+(?:أن|ان)\s+اسمي\s+هو\s+(.+?)\s*$", re.I),
)
_PREFERENCE_PATTERNS = (
    re.compile(r"^\s*(?:انا|أنا)\s+(?:احب|أحب)\s+(.+?)\s*$", re.I),
    re.compile(r"^\s*(?:انا|أنا)\s+(?:لا احب|لا أحب)\s+(.+?)\s*$", re.I),
)
_AGE_PATTERNS = (
    re.compile(r"^\s*(?:انا|أنا)\s+(?:عمري|سني)\s+(\d{1,3})\s*(?:عام|سنة|سنين)?\s*$", re.I),
    re.compile(r"^\s*(?:عمري|سني)\s+(\d{1,3})\s*(?:عام|سنة|سنين)?\s*$", re.I),
    re.compile(r"^\s*(?:انا|أنا)\s+(\d{1,3})\s*(?:عام|سنة|سنين)\s*$", re.I),
)


def normalize_arabic(text: str) -> str:
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    return _SPACES.sub(" ", text.lower()).strip()


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


def extract_facts(text: str) -> list[ExtractedFact]:
    normalized = normalize_arabic(text)

    correction = re.match(
        r"^اسمي\s+(.+?)\s+فقط\s+(?:اما|لكن)\s+(\d{1,3})\s*(?:عام|سنة|سنين)?(?:\s+فهذا\s+عمري)?\s*[.،,؛;؟?]*$",
        normalized,
        re.I,
    )
    if correction:
        name = correction.group(1).strip(" .،,؛;؟؟")
        age = int(correction.group(2))
        facts: list[ExtractedFact] = []
        if name:
            facts.append(ExtractedFact("user.name", name, "name"))
        if 1 <= age <= 120:
            facts.append(ExtractedFact("user.age", str(age), "age"))
        return facts

    combined = re.match(
        r"^انا\s+(.+?)\s+وعمري\s+(\d{1,3})\s*(?:عام|سنة|سنين)?\s*[.،,؛;؟?]*$",
        normalized,
        re.I,
    )
    if combined:
        name = combined.group(1).strip(" .،,؛;؟؟")
        age = int(combined.group(2))
        facts = []
        if name and not re.match(r"^(?:عمري|سني|احب|لا احب)\b", name):
            facts.append(ExtractedFact("user.name", name, "name"))
        if 1 <= age <= 120:
            facts.append(ExtractedFact("user.age", str(age), "age"))
        return facts

    named_age = re.match(
        r"^اسمي\s+(.+?)\s+(\d{1,3})\s*(?:عام|سنة|سنين)\s*[.،,؛;؟?]*$",
        normalized,
        re.I,
    )
    if named_age:
        name = named_age.group(1).strip(" .،,؛;؟؟")
        age = int(named_age.group(2))
        facts = []
        if name:
            facts.append(ExtractedFact("user.name", name, "name"))
        if 1 <= age <= 120:
            facts.append(ExtractedFact("user.age", str(age), "age"))
        return facts

    for pattern in _AGE_PATTERNS:
        match = pattern.match(normalized)
        if match:
            age = int(match.group(1))
            if 1 <= age <= 120:
                return [ExtractedFact("user.age", str(age), "age")]

    for pattern in _PREFERENCE_PATTERNS:
        match = pattern.match(normalized)
        if match:
            value = match.group(1).strip(" .،,؛;؟؟")
            if value:
                return [ExtractedFact("user.preference", value, "preference")]

    for pattern in _NAME_PATTERNS:
        match = pattern.match(normalized)
        if match:
            value = match.group(1).strip(" .،,؛;؟؟")
            if value:
                return [ExtractedFact("user.name", value, "name")]

    natural_name = re.match(r"^انا\s+(.+?)$", normalized, re.I)
    if natural_name and not re.match(r"^(?:عمري|سني|احب|لا احب)\b", natural_name.group(1)):
        value = natural_name.group(1).strip(" .،,؛;؟؟")
        if value:
            return [ExtractedFact("user.name", value, "name")]
    return []


def extract_fact(text: str) -> ExtractedFact | None:
    facts = extract_facts(text)
    return facts[0] if facts else None


def answer_fact_question(text: str, memory: MemoryStore) -> str | None:
    normalized = normalize_arabic(text)
    name_q = any(p in normalized for p in ("ما اسمي", "ايه اسمي", "اي اسمي", "ما هو اسمي", "هل تتذكر اسمي"))
    age_q = any(p in normalized for p in ("كم عمري", "ما عمري", "ما هو عمري", "عندي كام سنة", "هل تتذكر عمري"))
    if name_q and age_q:
        names, ages = memory.recall("user.name", 1), memory.recall("user.age", 1)
        name, age = (names[0].content if names else None), (ages[0].content if ages else None)
        if name and age:
            return f"اسمك {name}، وعمرك {age} سنة."
        if name:
            return f"اسمك {name}، ولم تخبرني بعمرك بعد."
        if age:
            return f"عمرك {age} سنة، ولم تخبرني باسمك بعد."
        return "لم تخبرني باسمك أو عمرك بعد."
    if name_q:
        hits = memory.recall("user.name", 1)
        return f"اسمك {hits[0].content}." if hits else "لم تخبرني باسمك بعد."
    if age_q:
        hits = memory.recall("user.age", 1)
        return f"عمرك {hits[0].content} سنة." if hits else "لم تخبرني بعمرك بعد."
    if any(p in normalized for p in ("ماذا احب", "ما الذي احبه", "ايه اللي بحبه", "هل تتذكر ما احب")):
        hits = memory.recall("user.preference", 5)
        return "أتذكر أنك تحب: " + "، ".join(x.content for x in hits) + "." if hits else "لم تخبرني بتفضيلاتك بعد."
    return None
