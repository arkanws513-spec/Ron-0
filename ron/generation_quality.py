"""Lightweight generation-stability guard for Ron's native character model."""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)


def is_degenerate_text(answer: str) -> bool:
    """Detect obvious token loops; this does not assess factual correctness."""
    tokens = [token.casefold() for token in _TOKEN_RE.findall(str(answer))]
    if len(tokens) < 6:
        return False
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    dominant_ratio = max(counts.values()) / len(tokens)
    unique_ratio = len(counts) / len(tokens)
    return dominant_ratio >= 0.45 or unique_ratio <= 0.35


def guard_generated_text(answer: str, fallback: str) -> tuple[str, bool]:
    """Return generated text unless empty or obviously repetitive; also report rejection."""
    candidate = str(answer or "").strip()
    if not candidate or is_degenerate_text(candidate):
        return fallback, True
    return candidate, False
