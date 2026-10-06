"""Stable provider-agnostic interfaces for Ron-0."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

@dataclass(frozen=True)
class Message:
    role: str
    content: str

@dataclass(frozen=True)
class ModelRequest:
    messages: Sequence[Message]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ModelResponse:
    content: str
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class MemoryItem:
    key: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

class ModelProvider(Protocol):
    def generate(self, request: ModelRequest) -> ModelResponse: ...

class MemoryStore(Protocol):
    def remember(self, item: MemoryItem) -> None: ...
    def recall(self, query: str, limit: int = 5) -> list[MemoryItem]: ...

class Retriever(Protocol):
    def retrieve(self, query: str, limit: int = 5) -> list[MemoryItem]: ...

class Tool(Protocol):
    name: str
    def describe(self) -> dict[str, Any]: ...
    def execute(self, arguments: dict[str, Any]) -> dict[str, Any]: ...

class Planner(Protocol):
    def plan(self, request: ModelRequest) -> list[dict[str, Any]]: ...
