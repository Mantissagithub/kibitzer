from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess.pgn


ALLOWED_TERMINATIONS = {"normal", "adjudication"}


def validate_pgn(path: Path, *, expected_games: int, allow_unterminated: bool = False) -> dict:
    games = 0
    terminations: dict[str, int] = {}
    errors: list[str] = []

    with path.open(encoding="utf-8", errors="replace") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            games += 1
            termination = game.headers.get("Termination", "missing").strip().lower()
            terminations[termination] = terminations.get(termination, 0) + 1

            allowed = set(ALLOWED_TERMINATIONS)
            if allow_unterminated:
                allowed.add("unterminated")
            if termination not in allowed:
                errors.append(f"game {games}: rejected termination '{termination}'")
            if game.headers.get("Result") not in {"1-0", "0-1", "1/2-1/2"}:
                errors.append(f"game {games}: invalid result '{game.headers.get('Result', 'missing')}'")
            if game.errors:
                errors.append(f"game {games}: PGN parser errors: {game.errors}")

    if games != expected_games:
        errors.append(f"game count mismatch: expected {expected_games}, found {games}")

    return {
        "pgn": str(path),
        "expected_games": expected_games,
        "games": games,
        "terminations": dict(sorted(terminations.items())),
        "valid": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, required=True)
    parser.add_argument("--allow-unterminated", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = validate_pgn(
        args.pgn,
        expected_games=args.expected_games,
        allow_unterminated=args.allow_unterminated,
    )
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("============================================================")
    print(" TOURNAMENT PGN VALIDATION")
    print("============================================================")
    print(f"games:        {report['games']} / {report['expected_games']}")
    print(f"terminations: {report['terminations']}")
    print(f"status:       {'CLEAN' if report['valid'] else 'REJECTED'}")
    for error in report["errors"]:
        print(f"  - {error}")

    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
