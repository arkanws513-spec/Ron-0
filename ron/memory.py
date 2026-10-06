"""Deterministic memory with simple token-aware retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from .contracts import MemoryItem

_TOKEN_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if len(token) > 1}


@dataclass
class InMemoryStore:
    items: dict[str, MemoryItem] = field(default_factory=dict)

    def remember(self, item: MemoryItem) -> None:
        if not item.key.strip():
            raise ValueError("memory key cannot be empty")
        if not item.content.strip():
            raise ValueError("memory content cannot be empty")
        self.items[item.key] = item

    def recall(self, query: str, limit: int = 5) -> list[MemoryItem]:
        if limit < 1:
            return []
        query_terms = _tokens(query)
        if not query_terms:
            return []

        ranked: list[tuple[int, MemoryItem]] = []
        for item in self.items.values():
            haystack_terms = _tokens(f"{item.key} {item.content}")
            score = len(query_terms & haystack_terms)
            if score:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in ranked[:limit]]
