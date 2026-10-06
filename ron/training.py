"""CPU-friendly training and checkpoint utilities for Ron's native model."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import torch
from torch import Tensor
from .model import RonCausalLM

@dataclass(frozen=True)
class TrainResult:
    initial_loss: float
    final_loss: float
    steps: int

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

def make_next_token_batch(token_ids: list[int], sequence_length: int, batch_size: int, *, device: str = "cpu") -> tuple[Tensor, Tensor]:
    if sequence_length < 2 or batch_size < 1:
        raise ValueError("invalid training dimensions")
    if len(token_ids) < sequence_length + 1:
        raise ValueError("not enough tokens for one training example")
    starts = torch.randint(0, len(token_ids) - sequence_length, (batch_size,), device=device)
    data = torch.tensor(token_ids, dtype=torch.long, device=device)
    return (
        torch.stack([data[start:start + sequence_length] for start in starts]),
        torch.stack([data[start + 1:start + sequence_length + 1] for start in starts]),
    )

def train_steps(model: RonCausalLM, token_ids: list[int], *, steps: int = 20, sequence_length: int = 32, batch_size: int = 4, learning_rate: float = 3e-3, device: str = "cpu") -> TrainResult:
    if steps < 1:
        raise ValueError("steps must be positive")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    initial_loss = final_loss = 0.0
    for step in range(steps):
        inputs, targets = make_next_token_batch(token_ids, sequence_length, batch_size, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs, targets)
        assert output.loss is not None
        if step == 0:
            initial_loss = float(output.loss.detach())
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(output.loss.detach())
    return TrainResult(initial_loss, final_loss, steps)

def save_checkpoint(model: RonCausalLM, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config.__dict__, "state_dict": model.state_dict()}, target)

def load_checkpoint(model: RonCausalLM, path: str | Path) -> RonCausalLM:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["config"] != model.config.__dict__:
        raise ValueError("checkpoint configuration does not match model")
    model.load_state_dict(payload["state_dict"])
    return model
