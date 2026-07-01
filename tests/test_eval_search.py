from __future__ import annotations

import chess

from scripts.eval_search_vs_stockfish import network_result, starting_board


def test_openings_are_legal_and_paired() -> None:
    for game_index in range(0, 10, 2):
        white_board = starting_board(game_index)
        black_board = starting_board(game_index + 1)
        assert white_board.fen() == black_board.fen()
        assert not white_board.is_game_over()


def test_network_result_accounts_for_color() -> None:
    assert network_result("1-0", network_is_white=True) == "win"
    assert network_result("1-0", network_is_white=False) == "loss"
    assert network_result("0-1", network_is_white=False) == "win"
    assert network_result("1/2-1/2", network_is_white=True) == "draw"
