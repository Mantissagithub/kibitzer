from __future__ import annotations

from pathlib import Path

from scripts.validate_tournament_pgn import validate_pgn


def write_pgn(path: Path, termination: str) -> None:
    path.write_text(
        "\n".join([
            '[Event "test"]',
            '[Site "?"]',
            '[Date "2026.07.13"]',
            '[Round "1"]',
            '[White "a"]',
            '[Black "b"]',
            '[Result "1-0"]',
            f'[Termination "{termination}"]',
            "",
            "1. e4 e5 2. Nf3 Nc6 1-0",
            "",
        ]),
        encoding="utf-8",
    )


def test_validator_accepts_clean_adjudication(tmp_path: Path) -> None:
    path = tmp_path / "clean.pgn"
    write_pgn(path, "adjudication")

    report = validate_pgn(path, expected_games=1)

    assert report["valid"]
    assert report["terminations"] == {"adjudication": 1}


def test_validator_rejects_time_forfeit(tmp_path: Path) -> None:
    path = tmp_path / "time.pgn"
    write_pgn(path, "time forfeit")

    report = validate_pgn(path, expected_games=1)

    assert not report["valid"]
    assert "time forfeit" in report["errors"][0]


def test_validator_rejects_missing_games(tmp_path: Path) -> None:
    path = tmp_path / "short.pgn"
    write_pgn(path, "normal")

    report = validate_pgn(path, expected_games=2)

    assert not report["valid"]
    assert "game count mismatch" in report["errors"][-1]
