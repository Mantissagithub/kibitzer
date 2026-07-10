from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _summary(rows: list[dict]) -> tuple[int, int, int, float]:
    scores = [float(row["score"]) for row in rows]
    wins = sum(score == 1.0 for score in scores)
    draws = sum(score == 0.5 for score in scores)
    losses = sum(score == 0.0 for score in scores)
    return wins, draws, losses, sum(scores) / len(scores)


def _elo(score: float, opponent: int) -> float:
    if score <= 0.0:
        return float("-inf")
    if score >= 1.0:
        return float("inf")
    return opponent + 400.0 * math.log10(score / (1.0 - score))


def _paired_interval(deltas: list[float], seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(10_000):
        means.append(sum(rng.choice(deltas) for _ in deltas) / len(deltas))
    means.sort()
    return means[250], means[9749]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puct", type=Path, required=True)
    parser.add_argument("--gumbel", type=Path, required=True)
    parser.add_argument("--opponent-elo", type=int, default=2700)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    puct = _read(args.puct)
    gumbel = _read(args.gumbel)
    if not puct or len(puct) != len(gumbel):
        raise ValueError("both gates must contain the same non-zero number of games")
    for left, right in zip(puct, gumbel, strict=True):
        if left.get("opening_fen") != right.get("opening_fen"):
            raise ValueError("gate openings do not match")
        if left.get("network_white") != right.get("network_white"):
            raise ValueError("gate colors do not match")

    puct_w, puct_d, puct_l, puct_rate = _summary(puct)
    g_w, g_d, g_l, g_rate = _summary(gumbel)
    deltas = [float(right["score"]) - float(left["score"]) for left, right in zip(puct, gumbel, strict=True)]
    delta = sum(deltas) / len(deltas)
    low, high = _paired_interval(deltas)
    if delta >= 0.03:
        verdict = "PROMISING: run the 80-game confirmation"
    elif delta <= -0.03:
        verdict = "REGRESSION: stop here, do not train from it"
    else:
        verdict = "NO SIGNAL: stop here unless the regret diagnostic improves"

    print("============================================================")
    print(" GUMBEL SEARCH DECISION")
    print("============================================================")
    print(f"PUCT:    {puct_w}W/{puct_d}D/{puct_l}L  rate={puct_rate:.3f}  elo={_elo(puct_rate, args.opponent_elo):.0f}")
    print(f"Gumbel:  {g_w}W/{g_d}D/{g_l}L  rate={g_rate:.3f}  elo={_elo(g_rate, args.opponent_elo):.0f}")
    print(f"Delta:   {delta:+.3f} score rate  paired 95% CI [{low:+.3f}, {high:+.3f}]")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
