from __future__ import annotations

import pytest

from kibitzer.diagnostics import (
    bounded_value,
    paired_bootstrap_interval,
    summarize_move_regret,
    summarize_value_predictions,
    value_bin,
)


def test_value_bins_use_raw_centipawn_magnitude() -> None:
    assert value_bin(20) == "quiet"
    assert value_bin(-120) == "edge"
    assert value_bin(350) == "decisive"
    assert value_bin(-800) == "won"
    assert bounded_value(1500) == 1.0


def test_value_summary_is_stratified() -> None:
    summary = summarize_value_predictions(
        [25, 100, 300, 700],
        [0.0, 0.1, -0.2, 0.9],
    )

    assert summary["quiet"]["count"] == 1
    assert summary["decisive"]["sign_accuracy"] == 0.0
    assert summary["won"]["mae"] == pytest.approx(0.2)


def test_regret_summary_and_paired_interval() -> None:
    metrics = summarize_move_regret([0.0, 20.0, 80.0, 100.0])
    low, high = paired_bootstrap_interval([10.0, 20.0, 30.0], seed=42)

    assert metrics["near_best_accuracy"] == 0.5
    assert metrics["p90_cp"] == pytest.approx(94.0)
    assert low > 0.0
    assert high > low
