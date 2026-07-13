from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_to_delta(score: float) -> float:
    clipped = min(max(score, 1e-6), 1.0 - 1e-6)
    return 400.0 * math.log10(clipped / (1.0 - clipped))


def summarize(rows: list[dict]) -> dict:
    wins = sum(row["score"] == 1.0 for row in rows)
    draws = sum(row["score"] == 0.5 for row in rows)
    losses = sum(row["score"] == 0.0 for row in rows)
    score = sum(row["score"] for row in rows) / len(rows)
    moves = sum(row["search"]["moves"] for row in rows)
    simulations = sum(row["search"]["total_simulations"] for row in rows)
    seconds = sum(row["search"]["seconds"] for row in rows)
    stages: dict[str, int] = {}
    for row in rows:
        for stage, count in row["search"]["stage_counts"].items():
            stages[stage] = stages.get(stage, 0) + count
    return {
        "games": len(rows),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "moves": moves,
        "avg_simulations": simulations / max(moves, 1),
        "seconds_per_move": seconds / max(moves, 1),
        "stage_counts": dict(sorted(stages.items(), key=lambda item: int(item[0]))),
    }


def paired_interval(fixed: list[dict], adaptive: list[dict], *, samples: int, seed: int) -> tuple[float, float]:
    differences = [candidate["score"] - control["score"] for control, candidate in zip(fixed, adaptive)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(rng.choice(differences) for _ in differences) / len(differences))
    estimates.sort()
    return estimates[int(0.025 * samples)], estimates[min(int(0.975 * samples), samples - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--max-average-sims", type=float, default=512.0)
    parser.add_argument("--score-margin", type=float, default=0.03)
    parser.add_argument("--match-tolerance", type=float, default=0.02)
    parser.add_argument("--time-ratio", type=float, default=0.75)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    fixed_rows = load_rows(args.fixed)
    adaptive_rows = load_rows(args.adaptive)
    if not fixed_rows or len(fixed_rows) != len(adaptive_rows):
        raise SystemExit("fixed and adaptive JSONL files must contain the same non-zero game count")
    fixed_keys = [(row["pair"], row["network_white"], row["maia_elo"]) for row in fixed_rows]
    adaptive_keys = [(row["pair"], row["network_white"], row["maia_elo"]) for row in adaptive_rows]
    if fixed_keys != adaptive_keys:
        raise SystemExit("fixed and adaptive games are not paired by opening, color, and opponent")

    fixed = summarize(fixed_rows)
    adaptive = summarize(adaptive_rows)
    score_delta = adaptive["score"] - fixed["score"]
    ci_low, ci_high = paired_interval(
        fixed_rows,
        adaptive_rows,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    elo_delta = score_to_delta(adaptive["score"]) - score_to_delta(fixed["score"])
    time_ratio = adaptive["seconds_per_move"] / max(fixed["seconds_per_move"], 1e-9)
    budget_ok = adaptive["avg_simulations"] <= args.max_average_sims
    strength_win = score_delta >= args.score_margin
    efficiency_win = score_delta >= -args.match_tolerance and time_ratio <= args.time_ratio
    accepted = budget_ok and (strength_win or efficiency_win)

    report = {
        "fixed": fixed,
        "adaptive": adaptive,
        "score_delta": score_delta,
        "paired_score_ci95": [ci_low, ci_high],
        "paired_elo_delta": elo_delta,
        "time_ratio": time_ratio,
        "budget_ok": budget_ok,
        "strength_win": strength_win,
        "efficiency_win": efficiency_win,
        "accepted": accepted,
    }
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("============================================================")
    print(" ADAPTIVE SEARCH DECISION")
    print("============================================================")
    print(
        f"fixed:    {fixed['wins']}W/{fixed['draws']}D/{fixed['losses']}L "
        f"score={fixed['score']:.3f} avg={fixed['avg_simulations']:.1f}sims "
        f"time={fixed['seconds_per_move']:.3f}s/move"
    )
    print(
        f"adaptive: {adaptive['wins']}W/{adaptive['draws']}D/{adaptive['losses']}L "
        f"score={adaptive['score']:.3f} avg={adaptive['avg_simulations']:.1f}sims "
        f"time={adaptive['seconds_per_move']:.3f}s/move"
    )
    print(f"stages:   {adaptive['stage_counts']}")
    print(
        f"delta:    score={score_delta:+.3f} paired95=[{ci_low:+.3f}, {ci_high:+.3f}] "
        f"elo={elo_delta:+.0f} time_ratio={time_ratio:.3f}"
    )
    print(f"budget:   {'PASS' if budget_ok else 'FAIL'} <= {args.max_average_sims:.0f} average sims")
    print(f"verdict:  {'ADOPT' if accepted else 'REJECT'}")

    raise SystemExit(0 if accepted else 1)


if __name__ == "__main__":
    main()
