"""Small, explicit tool registry with no implicit execution."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class ToolRegistry:
    handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = field(default_factory=dict)

    def register(self, name: str, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        if not name.strip():
            raise ValueError("tool name cannot be empty")
        self.handlers[name] = handler

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": name} for name in sorted(self.handlers)]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self.handlers:
            raise KeyError(f"unknown tool: {name}")
        return self.handlers[name](arguments)