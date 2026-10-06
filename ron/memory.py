"""Deterministic in-process memory for the Ron core."""
from __future__ import annotations
from dataclasses import dataclass, field
from .contracts import MemoryItem

@dataclass
class InMemoryStore:
    items: dict[str, MemoryItem] = field(default_factory=dict)

    def remember(self, item: MemoryItem) -> None:
        self.items[item.key] = item

    def recall(self, query: str, limit: int = 5) -> list[MemoryItem]:
        terms = {t.lower() for t in query.split() if t.strip()}
        ranked = []
        for item in self.items.values():
            haystack = f"{item.key} {item.content}".lower()
            score = sum(term in haystack for term in terms)
            if score:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked[:limit]]