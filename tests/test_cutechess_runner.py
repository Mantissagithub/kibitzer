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

    with_options = _engine_argv("stockfish", "B", {"UCI_LimitStrength": "true"})
    assert "option.UCI_LimitStrength=true" in with_options

    spaced_option = _engine_argv("stockfish", "B", {"Debug Log File": "/tmp/sf.log"})
    # the space inside the option name must survive as one argv token;
    # otherwise cutechess will see two args and reject the option.
    assert "option.Debug Log File=/tmp/sf.log" in spaced_option

    with pytest.raises(ValueError):
        _engine_argv("", "X", None)
    with pytest.raises(ValueError):
        _engine_argv("   ", "X", None)


@pytest.mark.skipif(
    shutil.which("cutechess-cli") is None or shutil.which("stockfish") is None,
    reason="requires cutechess-cli and stockfish on PATH",
)
def test_full_stockfish_beats_limited_stockfish(tmp_path) -> None:
    pgn = tmp_path / "match.pgn"
    result = run_match(
        engine_a_cmd="stockfish",
        engine_b_cmd="stockfish",
        engine_a_options={"UCI_LimitStrength": "true", "UCI_Elo": "1320"},
        engine_b_options=None,
        n_games=2,
        time_per_move_ms=500,
        pgn_output=str(pgn),
    )

    # UCI_LimitStrength + UCI_Elo is the rated-limited Stockfish mode; this is
    # the baseline style used by Kibitzer evals.
    assert result["wins"] + result["losses"] + result["draws"] == 2
    assert pgn.exists()
    assert result["losses"] >= result["wins"], (
        f"expected rated-limited engine to lose at least as often as it wins: {result}"
    )
