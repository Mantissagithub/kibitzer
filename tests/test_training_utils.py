"""Tests for kibitzer.training_utils."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from kibitzer.training_utils import (
    EMA,
    count_params,
    get_lr,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# get_lr
# ---------------------------------------------------------------------------


def test_lr_zero_at_step_zero() -> None:
    assert get_lr(0, warmup_steps=100, total_steps=1000,
                  peak_lr=1e-3, min_lr=1e-5) == 0.0


def test_lr_peak_at_warmup_end() -> None:
    lr = get_lr(100, 100, 1000, 1e-3, 1e-5)
    assert math.isclose(lr, 1e-3, rel_tol=1e-9)


def test_lr_min_at_total_steps() -> None:
    lr = get_lr(1000, 100, 1000, 1e-3, 1e-5)
    assert math.isclose(lr, 1e-5, rel_tol=1e-9)


def test_lr_warmup_is_linear() -> None:
    # Halfway through warmup → half of peak_lr.
    lr = get_lr(50, 100, 1000, 1e-3, 1e-5)
    assert math.isclose(lr, 5e-4, rel_tol=1e-9)


def test_lr_cosine_decay_monotonic() -> None:
    prev = float("inf")
    for step in range(100, 1001, 25):
        lr = get_lr(step, 100, 1000, 1e-3, 1e-5)
        assert lr <= prev + 1e-12, f"non-monotonic at step={step}"
        prev = lr


def test_lr_past_total_steps_pinned() -> None:
    assert math.isclose(
        get_lr(2000, 100, 1000, 1e-3, 1e-5), 1e-5, rel_tol=1e-9
    )


def test_lr_no_warmup() -> None:
    # warmup_steps=0 → step 0 starts at peak_lr (cosine progress=0 → factor=1).
    assert math.isclose(get_lr(0, 0, 1000, 1e-3, 1e-5), 1e-3, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def test_ema_value_before_update_is_zero() -> None:
    assert EMA().value == 0.0


def test_ema_first_update_takes_value() -> None:
    ema = EMA(beta=0.99)
    ema.update(7.5)
    assert ema.value == 7.5


def test_ema_converges_to_constant() -> None:
    ema = EMA(beta=0.9)
    ema.update(0.0)  # seed away from target
    for _ in range(200):
        ema.update(5.0)
    assert math.isclose(ema.value, 5.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# save / load checkpoint
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model_a = nn.Linear(4, 2)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    # Take a real step so the optimizer accumulates state.
    loss = model_a(torch.randn(3, 4)).sum()
    loss.backward()
    opt_a.step()

    path = tmp_path / "ckpt" / "model.pt"
    config = {"hidden": 16, "lr": 1e-3}
    save_checkpoint(
        path, model_a, opt_a,
        scheduler_state={"last_epoch": 1},
        step=42,
        config=config,
        metrics={"loss": 0.5},
    )

    torch.manual_seed(99)  # ensure model_b's init differs from model_a
    model_b = nn.Linear(4, 2)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    ckpt = load_checkpoint(path, model_b, optimizer=opt_b)

    for (n_a, p_a), (n_b, p_b) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert torch.allclose(p_a, p_b), f"param {n_a} mismatch after load"
    assert ckpt["step"] == 42
    assert ckpt["config"] == config
    assert ckpt["metrics"] == {"loss": 0.5}
    assert ckpt["scheduler_state"] == {"last_epoch": 1}


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    deep = tmp_path / "a" / "b" / "c" / "ckpt.pt"
    save_checkpoint(deep, model, opt, scheduler_state=None, step=0, config={})
    assert deep.exists()


def test_load_without_optimizer(tmp_path: Path) -> None:
    model_a = nn.Linear(3, 3)
    opt_a = torch.optim.SGD(model_a.parameters(), lr=1e-2)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model_a, opt_a, scheduler_state=None, step=7, config={})

    model_b = nn.Linear(3, 3)
    ckpt = load_checkpoint(path, model_b)  # optimizer omitted
    assert ckpt["step"] == 7


# ---------------------------------------------------------------------------
# count_params
# ---------------------------------------------------------------------------


def test_count_params_linear() -> None:
    # Linear(10, 5): weight 10*5 + bias 5 = 55.
    assert count_params(nn.Linear(10, 5)) == 55


def test_count_params_sequential() -> None:
    # Linear(3,4) + Linear(4,2) = (12+4) + (8+2) = 26.
    seq = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    assert count_params(seq) == 26
