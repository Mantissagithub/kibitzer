"""Run cutechess-cli matches between two UCI engines and parse the results.

The in-Python ``kibitzer.match`` runner plays both sides itself; this module
shells out to ``cutechess-cli`` so we can run real tournament-style matches
against external binaries (Stockfish, lc0, prior Kibitzer checkpoints, …)
with cutechess handling time enforcement, crash recovery, color alternation,
and PGN output.

The single public entry point is :func:`run_match`; everything else is a
private helper exposed only for testing.

Example
-------
    from kibitzer.cutechess_runner import run_match

    result = run_match(
        engine_a_cmd="python scripts/uci.py --checkpoint runs/best.pt",
        engine_b_cmd="stockfish",
        engine_b_options={"Skill Level": "0"},
        n_games=20,
        time_per_move_ms=200,
    )
    # -> {"wins": 3, "losses": 15, "draws": 2,
    #     "elo_diff": -195.4, "elo_err": 84.2, "pgn_path": "match.pgn"}
"""

from __future__ import annotations

import logging
import math
import re
import shlex
import shutil
import subprocess
from typing import Mapping

logger = logging.getLogger(__name__)


_SCORE_RE = re.compile(
    r"^Score of \S+ vs \S+: (\d+) - (\d+) - (\d+)\s+\["
)
_ELO_RE = re.compile(r"^Elo difference: (\S+) \+/- (\S+)")


def _safe_float(s: str) -> float:
    """Parse a cutechess number, mapping ``inf``/``-inf``/``nan`` correctly."""
    try:
        return float(s.rstrip(","))
    except ValueError:
        return math.nan


def _engine_argv(
    cmd: str, name: str, options: Mapping[str, str] | None
) -> list[str]:
    """Build the argv tokens that follow ``-engine`` for one engine.

    ``cmd`` is shlex-split so callers can pass a full command line
    (``"python /path/to/uci.py --checkpoint x.pt"``); the first token becomes
    cutechess's ``cmd=`` and the rest are emitted as repeated ``arg=...``.
    Option names with spaces (``Skill Level``) survive because each
    ``option.NAME=VALUE`` is a single argv entry — subprocess passes argv
    directly with no shell parsing.
    """
    parts = shlex.split(cmd)
    if not parts:
        raise ValueError("engine command is empty")
    out = [f"cmd={parts[0]}"]
    for a in parts[1:]:
        out.append(f"arg={a}")
    out.append(f"name={name}")
    out.append("proto=uci")
    if options:
        for k, v in options.items():
            out.append(f"option.{k}={v}")
    return out


def run_match(
    engine_a_cmd: str,
    engine_b_cmd: str,
    engine_a_options: Mapping[str, str] | None = None,
    engine_b_options: Mapping[str, str] | None = None,
    n_games: int = 20,
    time_per_move_ms: int = 1000,
    tc: str | None = None,
    opening_book: str | None = None,
    pgn_output: str = "match.pgn",
    concurrency: int = 1,
    cutechess_path: str | None = None,
) -> dict:
    """Run a cutechess-cli match between engines A and B.

    Returns a dict with W/L/D counts (from A's perspective), Elo difference
    and its standard error, and the PGN output path.
    """
    cli = cutechess_path or shutil.which("cutechess-cli")
    if cli is None:
        raise RuntimeError(
            "cutechess-cli not found on PATH; install it or pass cutechess_path"
        )

    argv: list[str] = [cli]
    argv += ["-engine", *_engine_argv(engine_a_cmd, "A", engine_a_options)]
    argv += ["-engine", *_engine_argv(engine_b_cmd, "B", engine_b_options)]

    if tc is not None:
        argv += ["-each", f"tc={tc}"]
    else:
        argv += ["-each", f"st={time_per_move_ms / 1000}"]

    argv += ["-games", str(n_games), "-repeat", "-recover"]
    argv += ["-pgnout", str(pgn_output)]
    if concurrency > 1:
        argv += ["-concurrency", str(concurrency)]
    if opening_book:
        fmt = "pgn" if opening_book.endswith(".pgn") else "epd"
        argv += [
            "-openings",
            f"file={opening_book}",
            f"format={fmt}",
            "order=random",
        ]

    logger.info("running: %s", " ".join(shlex.quote(a) for a in argv))

    last_score: re.Match | None = None
    elo_diff = 0.0
    elo_err = 0.0

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            logger.info("[cutechess] %s", line)
            m = _SCORE_RE.match(line)
            if m:
                last_score = m
                continue
            m = _ELO_RE.match(line)
            if m:
                elo_diff = _safe_float(m.group(1))
                elo_err = _safe_float(m.group(2))
    finally:
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"cutechess-cli exited with code {rc}")
    if last_score is None:
        raise RuntimeError(
            "could not parse cutechess output (no Score line found)"
        )

    return {
        "wins": int(last_score.group(1)),
        "losses": int(last_score.group(2)),
        "draws": int(last_score.group(3)),
        "elo_diff": elo_diff,
        "elo_err": elo_err,
        "pgn_path": str(pgn_output),
    }
