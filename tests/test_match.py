"""Tests for kibitzer.match.play_match."""

from __future__ import annotations

import random

import chess

from kibitzer.match import play_match


def random_engine(seed: int):
    rng = random.Random(seed)

    def f(board: chess.Board) -> chess.Move:
        return rng.choice(list(board.legal_moves))

    return f


def first_legal(board: chess.Board) -> chess.Move:
    return next(iter(board.legal_moves))


def lex_min(board: chess.Board) -> chess.Move:
    return min(board.legal_moves, key=lambda m: m.uci())


def lex_max(board: chess.Board) -> chess.Move:
    return max(board.legal_moves, key=lambda m: m.uci())


def test_random_vs_random() -> None:
    a = random_engine(seed=1)
    b = random_engine(seed=2)
    out = play_match(a, b, n_games=4, max_plies=400)
    assert out["wins_a"] + out["wins_b"] + out["draws"] == 4
    assert len(out["games"]) == 4
    for g in out["games"]:
        assert g["result"] in {"1-0", "0-1", "1/2-1/2"}
        assert g["plies"] >= 1
        assert g["termination"] in {
            "checkmate",
            "stalemate",
            "insufficient material",
            "threefold repetition",
            "fifty-move rule",
            "max plies",
        }


def test_color_swap() -> None:
    out = play_match(
        engine_white=lex_min,
        engine_black=lex_max,
        n_games=2,
        max_plies=2,
        swap_colors=True,
    )

    # Game 1: A=white, B=black. White's first move = lex-min legal = a2a3 (SAN "a3").
    # Game 2: B=white, A=black. White's first move = lex-max legal = h2h4 (SAN "h4").
    pgn0 = out["games"][0]["pgn"]
    pgn1 = out["games"][1]["pgn"]

    assert "[White \"A\"]" in pgn0 and "[Black \"B\"]" in pgn0
    assert "[White \"B\"]" in pgn1 and "[Black \"A\"]" in pgn1
    assert "1. a3" in pgn0, f"expected 'a3' as first move in game 1, got: {pgn0}"
    assert "1. h4" in pgn1, f"expected 'h4' as first move in game 2, got: {pgn1}"


def test_starting_fens() -> None:
    fen1 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    fen2 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    out = play_match(
        engine_white=first_legal,
        engine_black=first_legal,
        n_games=4,
        max_plies=2,
        starting_fens=[fen1, fen2],
        swap_colors=False,
    )
    assert len(out["games"]) == 4

    fen1_count = sum(1 for g in out["games"] if fen1 in g["pgn"])
    fen2_count = sum(1 for g in out["games"] if fen2 in g["pgn"])
    assert fen1_count == 2, f"expected fen1 in 2 games, got {fen1_count}"
    assert fen2_count == 2, f"expected fen2 in 2 games, got {fen2_count}"

    for g in out["games"]:
        assert "[FEN" in g["pgn"]


def test_max_plies_draw() -> None:
    out = play_match(
        engine_white=first_legal,
        engine_black=first_legal,
        n_games=1,
        max_plies=8,
    )
    g = out["games"][0]
    assert g["result"] == "1/2-1/2"
    assert g["termination"] == "max plies"
    assert g["plies"] == 8
    assert out["draws"] == 1
