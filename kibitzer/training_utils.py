"""Utilities for the training loop: LR schedule, EMA, checkpoint I/O, param counting."""

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
    """Linear warmup → cosine decay schedule.

    * ``step < warmup_steps``: linear from 0 to ``peak_lr``.
    * ``warmup_steps <= step <= total_steps``: cosine decay from ``peak_lr``
      to ``min_lr``.
    * ``step > total_steps``: pinned at ``min_lr``.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    decay_span = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_span
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


class EMA:
    """Exponential moving average for streaming scalar metrics (e.g. loss).

    Initializes on the first update so the first reported value is the input
    itself (no warm-up bias). ``value`` returns 0.0 before any update so a
    fresh tracker reads cleanly.
    """

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
    """Persist model + optimizer + run metadata to ``path``."""
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
    """Restore model state (and optionally optimizer); return the full payload."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


def count_params(model: nn.Module) -> int:
    """Total number of parameters in the model (trainable and frozen)."""
    return sum(p.numel() for p in model.parameters())
