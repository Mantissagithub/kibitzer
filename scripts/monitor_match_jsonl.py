from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MatchSummary:
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    last_result: str | None

    @property
    def score_rate(self) -> float:
        if self.games == 0:
            return 0.0
        return self.score / self.games


def _result_bucket(row: dict[str, Any]) -> str:
    score = row.get("score")
    if score == 1.0:
        return "win"
    if score == 0.5:
        return "draw"
    if score == 0.0:
        return "loss"
    return "unknown"


def summarize_match(path: Path) -> MatchSummary:
    wins = 0
    draws = 0
    losses = 0
    score = 0.0
    last_result: str | None = None
    if not path.exists():
        return MatchSummary(0, 0, 0, 0, 0.0, None)

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            bucket = _result_bucket(row)
            if bucket == "win":
                wins += 1
            elif bucket == "draw":
                draws += 1
            elif bucket == "loss":
                losses += 1
            score += float(row.get("score", 0.0))
            last_result = str(row.get("result", "?"))
    games = wins + draws + losses
    return MatchSummary(games, wins, draws, losses, score, last_result)


def elo_delta_from_score(score_rate: float) -> float:
    if score_rate <= 0.0:
        return float("-inf")
    if score_rate >= 1.0:
        return float("inf")
    return 400.0 * math.log10(score_rate / (1.0 - score_rate))


def _format_elo(value: float) -> str:
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:.0f}"


def format_summary(summary: MatchSummary, expected_games: int | None, opponent_elo: int | None = 2700) -> str:
    target = f"/{expected_games}" if expected_games else ""
    last = summary.last_result or "none"
    elo_delta = elo_delta_from_score(summary.score_rate)
    elo_text = f" elo_delta={_format_elo(elo_delta)}"
    if opponent_elo is not None:
        elo = opponent_elo + elo_delta
        elo_text += f" elo={_format_elo(elo)}"
    return (
        f"{summary.games}{target} games: "
        f"{summary.wins}W/{summary.draws}D/{summary.losses}L "
        f"score={summary.score:.1f} rate={summary.score_rate:.3f} "
        f"last={last}{elo_text}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, default=20)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--opponent-elo", type=int, default=2700)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        summary = summarize_match(args.path)
        print(format_summary(summary, args.expected_games, args.opponent_elo), flush=True)
        if args.once:
            return
        if args.expected_games and summary.games >= args.expected_games:
            return
        # this is intentionally dumb: the eval writer owns the file, we just
        # keep re-reading it so partial writes do not need a side channel.
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
