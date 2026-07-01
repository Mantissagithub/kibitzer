"""Metrics for deterministic policy/value/search diagnostics."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


VALUE_BINS = ("quiet", "edge", "decisive", "won")


def value_bin(centipawns: int) -> str:
    magnitude = abs(centipawns)
    if magnitude < 50:
        return "quiet"
    if magnitude < 200:
        return "edge"
    if magnitude < 500:
        return "decisive"
    return "won"


def bounded_value(centipawns: int) -> float:
    return max(-1.0, min(1.0, centipawns / 1000.0))


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    seed: int,
    samples: int = 2000,
) -> tuple[float, float]:
    if not differences:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    count = len(differences)
    means = []
    for _ in range(samples):
        means.append(
            sum(differences[rng.randrange(count)] for _ in range(count)) / count
        )
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize_value_predictions(
    centipawns: Sequence[int],
    predictions: Sequence[float],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[tuple[float, float]]] = {name: [] for name in VALUE_BINS}
    for cp, prediction in zip(centipawns, predictions, strict=True):
        grouped[value_bin(cp)].append((bounded_value(cp), prediction))

    summary: dict[str, dict[str, float]] = {}
    for name, pairs in grouped.items():
        if not pairs:
            summary[name] = {"count": 0, "mae": float("nan"), "sign_accuracy": float("nan")}
            continue
        absolute_errors = [abs(prediction - target) for target, prediction in pairs]
        sign_matches = [
            (prediction >= 0.0) == (target >= 0.0)
            for target, prediction in pairs
            if target != 0.0
        ]
        summary[name] = {
            "count": len(pairs),
            "mae": sum(absolute_errors) / len(absolute_errors),
            "sign_accuracy": (
                sum(sign_matches) / len(sign_matches) if sign_matches else float("nan")
            ),
        }
    return summary


def summarize_move_regret(
    regrets: Sequence[float],
    *,
    near_best_cp: int = 50,
) -> dict[str, float]:
    if not regrets:
        raise ValueError("at least one regret is required")
    return {
        "count": len(regrets),
        "mean_cp": sum(regrets) / len(regrets),
        "p90_cp": percentile(regrets, 0.90),
        "p95_cp": percentile(regrets, 0.95),
        "near_best_accuracy": sum(regret <= near_best_cp for regret in regrets)
        / len(regrets),
    }
