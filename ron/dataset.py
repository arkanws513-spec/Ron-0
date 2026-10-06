"""Training-data contracts for Ron's native model."""
from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class TrainingExample:
    input_text:str
    target_text:str
    source:str="unknown"
@dataclass
class TrainingDataset:
    examples:list[TrainingExample]
    def __len__(self): return len(self.examples)
    def add(self,example):
        if not example.input_text.strip() or not example.target_text.strip(): raise ValueError("training text cannot be empty")
        self.examples.append(example)
