"""training utilities: lr schedule, ema, checkpoints, parameter counts."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


def get_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
) -> float:
    """linear warmup into cosine decay, pinned at ``min_lr`` afterward."""
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_span = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_span
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


class EMA:
    """exponential moving average for display metrics."""

    def __init__(self, beta: float = 0.99) -> None:
        self.beta = beta
        self._value: float | None = None

    def update(self, value: float) -> None:
        v = float(value)
        if self._value is None:
            self._value = v
        else:
            self._value = self.beta * self._value + (1.0 - self.beta) * v

    @property
    def value(self) -> float:
        return 0.0 if self._value is None else self._value


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler_state,
    step: int,
    config: dict,
    metrics: dict | None = None,
) -> None:
    """persist model + optimizer + run metadata to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler_state,
        "step": step,
        "config": config,
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    """restore model state and optionally optimizer state."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def count_params(model: nn.Module) -> int:
    """total parameters, trainable and frozen."""
    return sum(p.numel() for p in model.parameters())
