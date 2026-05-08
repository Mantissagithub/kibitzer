"""Tests for kibitzer.eval.evaluate_checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from kibitzer.eval import evaluate_checkpoint
from kibitzer.model import Kibitzer


def _save_random_checkpoint(tmp_path: Path) -> str:
    torch.manual_seed(0)
    ckpt = tmp_path / "rand.pt"
    torch.save(Kibitzer().state_dict(), ckpt)
    return str(ckpt)


def test_random_eval(tmp_path: Path) -> None:
    ckpt = _save_random_checkpoint(tmp_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = evaluate_checkpoint(
        ckpt,
        opponent="random",
        n_games=6,
        device=device,
        max_plies=80,
    )
    assert result["wins"] + result["losses"] + result["draws"] == 6
    assert 0.0 <= result["win_rate"] <= 1.0
    assert result["avg_value_pred"] >= 0.0
    assert result["avg_plies"] > 0.0


def test_eval_returns_expected_keys(tmp_path: Path) -> None:
    ckpt = _save_random_checkpoint(tmp_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = evaluate_checkpoint(
        ckpt,
        opponent="random",
        n_games=2,
        device=device,
        max_plies=20,
    )
    expected = {
        "checkpoint",
        "opponent",
        "n_games",
        "wins",
        "losses",
        "draws",
        "win_rate",
        "approx_elo_diff",
        "avg_plies",
        "avg_value_pred",
    }
    assert set(result.keys()) == expected
