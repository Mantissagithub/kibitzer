"""Tests for kibitzer.cutechess_runner.

The argv-construction test runs everywhere; the integration test is gated on
having both ``cutechess-cli`` and ``stockfish`` on PATH and validates the
wrapper end-to-end without depending on any Kibitzer training.
"""

from __future__ import annotations

import shutil

import pytest

from kibitzer.cutechess_runner import _engine_argv, run_match


def test_engine_argv_construction() -> None:
    plain = _engine_argv("stockfish", "B", None)
    assert plain == ["cmd=stockfish", "name=B", "proto=uci"]

    multi = _engine_argv("python /tmp/uci.py --checkpoint x.pt", "A", None)
    assert multi == [
        "cmd=python",
        "arg=/tmp/uci.py",
        "arg=--checkpoint",
        "arg=x.pt",
        "name=A",
        "proto=uci",
    ]

    with_options = _engine_argv("stockfish", "B", {"Skill Level": "0"})
    # The space inside the option name must survive as a single argv token —
    # otherwise cutechess will see two args and reject the option.
    assert "option.Skill Level=0" in with_options

    with pytest.raises(ValueError):
        _engine_argv("", "X", None)
    with pytest.raises(ValueError):
        _engine_argv("   ", "X", None)


@pytest.mark.skipif(
    shutil.which("cutechess-cli") is None or shutil.which("stockfish") is None,
    reason="requires cutechess-cli and stockfish on PATH",
)
def test_higher_skill_wins_more(tmp_path) -> None:
    pgn = tmp_path / "match.pgn"
    result = run_match(
        engine_a_cmd="stockfish",
        engine_b_cmd="stockfish",
        engine_a_options={"Skill Level": "0"},
        engine_b_options={"Skill Level": "5"},
        n_games=2,
        time_per_move_ms=100,
        pgn_output=str(pgn),
    )

    assert result["wins"] + result["losses"] + result["draws"] == 2
    assert pgn.exists()
    # A is the weaker engine (skill 0); over 2 games it should not win more
    # often than it loses. Forgiving form so a 1-1 split or a sweep both pass.
    assert result["losses"] >= result["wins"], (
        f"expected weaker engine to lose at least as often as it wins: {result}"
    )
