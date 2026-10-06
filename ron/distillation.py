"""Controlled knowledge transfer from Qwen into Ron's future model dataset.

Qwen outputs are candidates, not trusted memory. Ron records provenance,
approval state, and evaluation so a future fine-tuning job can consume only
approved examples.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Iterable


@dataclass(frozen=True)
class TrainingExample:
    user: str
    assistant: str
    context: str = ""
    source: str = "qwen3"
    status: str = "candidate"

    def to_messages(self) -> dict:
        return {
            "messages": [
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ]
        }


class KnowledgeDistiller:
    """Collect, approve, and export Qwen examples without polluting memory."""

    def __init__(self, max_examples: int = 1000) -> None:
        if max_examples < 1:
            raise ValueError("max_examples must be positive")
        self.max_examples = max_examples
        self.examples: list[TrainingExample] = []

    def add(self, example: TrainingExample) -> None:
        if not example.user.strip() or not example.assistant.strip():
            raise ValueError("training example must contain user and assistant text")
        self.examples = [
            x for x in self.examples
            if not (x.user == example.user and x.assistant == example.assistant)
        ]
        self.examples.append(example)
        self.examples = self.examples[-self.max_examples :]

    def approve(self, user: str, assistant: str) -> bool:
        for index, example in enumerate(self.examples):
            if example.user == user and example.assistant == assistant:
                self.examples[index] = TrainingExample(
                    user=example.user,
                    assistant=example.assistant,
                    context=example.context,
                    source=example.source,
                    status="approved",
                )
                return True
        return False

    def approved(self) -> tuple[TrainingExample, ...]:
        return tuple(x for x in self.examples if x.status == "approved")

    def export_jsonl(self) -> str:
        return "\n".join(
            json.dumps(x.to_messages(), ensure_ascii=False)
            for x in self.approved()
        )

    def export_records(self) -> list[dict]:
        return [asdict(x) for x in self.examples]
