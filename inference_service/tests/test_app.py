from __future__ import annotations

import chess
from fastapi.testclient import TestClient

import inference_service.app as service
from kibitzer.inference import PositionEvaluation


class UniformEvaluator:
    model = type("Model", (), {"num_params": lambda self: 15_223_105})()

    def evaluate(self, board: chess.Board) -> PositionEvaluation:
        moves = list(board.legal_moves)
        probability = 1 / len(moves)
        return PositionEvaluation(
            priors={move: probability for move in moves},
            value=0.0,
        )

    def evaluate_batch(self, boards: list[chess.Board]) -> list[PositionEvaluation]:
        return [self.evaluate(board) for board in boards]


def test_build_board_replays_uci_moves() -> None:
    board = service.build_board(None, ["e2e4", "e7e5", "g1f3"])
    assert board.turn == chess.BLACK
    assert board.piece_at(chess.F3) == chess.Piece(chess.KNIGHT, chess.WHITE)


def test_build_board_rejects_illegal_move() -> None:
    try:
        service.build_board(None, ["e2e5"])
    except ValueError as error:
        assert "illegal" in str(error)
    else:
        raise AssertionError("illegal move was accepted")


def test_move_endpoint_returns_a_legal_search_result(monkeypatch) -> None:
    evaluator = UniformEvaluator()
    monkeypatch.setattr(service, "_evaluator", evaluator)

    with TestClient(service.app) as client:
        response = client.post("/move", json={"moves": ["e2e4"], "simulations": 64})

    assert response.status_code == 200
    payload = response.json()
    board = service.build_board(None, ["e2e4"])
    assert chess.Move.from_uci(payload["move"]) in board.legal_moves
    assert payload["simulations"] == 64
    assert payload["top_moves"]


def test_move_endpoint_rejects_an_unsupported_budget(monkeypatch) -> None:
    monkeypatch.setattr(service, "_evaluator", UniformEvaluator())

    with TestClient(service.app) as client:
        response = client.post("/move", json={"moves": [], "simulations": 32})

    assert response.status_code == 422
