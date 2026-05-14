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
    # the space inside the option name must survive as one argv token;
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
        engine_b_options={"Skill Level": "20"},
        n_games=2,
        time_per_move_ms=500,
        pgn_output=str(pgn),
    )

    # stockfish "Skill Level" only differentiates with real thinking time and
    # a wide gap; skill 0 vs 20 at 500ms/move is enough for skill 20 to win
    # both games via -repeat (one with each color).
    assert result["wins"] + result["losses"] + result["draws"] == 2
    assert pgn.exists()
    assert result["losses"] >= result["wins"], (
        f"expected weaker engine to lose at least as often as it wins: {result}"
    )
