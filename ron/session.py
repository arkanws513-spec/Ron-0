"""Conversation/session state independent from any model vendor."""
from __future__ import annotations
from dataclasses import dataclass, field
from .contracts import Message

@dataclass
class Conversation:
    session_id: str
    messages: list[Message] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        if not content.strip():
            raise ValueError("message content cannot be empty")
        self.messages.append(Message(role=role, content=content))

    def history(self) -> tuple[Message, ...]:
        return tuple(self.messages)

    def clear(self) -> None:
        self.messages.clear()
