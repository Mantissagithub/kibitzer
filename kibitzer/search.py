"""PUCT search over policy priors and side-to-move value estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import chess

from kibitzer.inference import PositionEvaluation


class PositionEvaluator(Protocol):
    def evaluate(self, board: chess.Board) -> PositionEvaluation: ...


@dataclass
class SearchNode:
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, SearchNode] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move
    root_value: float
    visits: dict[chess.Move, int]


def terminal_value(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        raise ValueError("board is not terminal")
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def _expand(node: SearchNode, evaluation: PositionEvaluation) -> None:
    node.children = {
        move: SearchNode(prior=prior) for move, prior in evaluation.priors.items()
    }


def _select_child(
    node: SearchNode,
    c_puct: float,
    value_scale: float,
) -> tuple[chess.Move, SearchNode]:
    parent_scale = math.sqrt(max(1, node.visit_count))

    def score(item: tuple[chess.Move, SearchNode]) -> float:
        _, child = item
        exploitation = -value_scale * child.mean_value
        exploration = c_puct * child.prior * parent_scale / (1 + child.visit_count)
        return exploitation + exploration

    return max(node.children.items(), key=score)


def puct_search(
    board: chess.Board,
    evaluator: PositionEvaluator,
    *,
    simulations: int,
    c_puct: float = 1.5,
    value_scale: float = 1.0,
) -> SearchResult:
    if simulations < 1:
        raise ValueError("simulations must be at least 1")
    if value_scale < 0.0:
        raise ValueError("value_scale must be non-negative")
    if board.is_game_over(claim_draw=True):
        raise ValueError("cannot search a terminal board")

    root = SearchNode()
    _expand(root, evaluator.evaluate(board))

    for _ in range(simulations):
        simulation_board = board.copy(stack=True)
        node = root
        path = [node]

        while node.children:
            move, node = _select_child(node, c_puct, value_scale)
            simulation_board.push(move)
            path.append(node)
            if simulation_board.is_game_over(claim_draw=True):
                break

        if simulation_board.is_game_over(claim_draw=True):
            value = terminal_value(simulation_board)
        else:
            evaluation = evaluator.evaluate(simulation_board)
            _expand(node, evaluation)
            value = evaluation.value

        for visited in reversed(path):
            visited.visit_count += 1
            visited.value_sum += value
            value = -value

    move, _ = max(root.children.items(), key=lambda item: item[1].visit_count)
    return SearchResult(
        move=move,
        root_value=root.mean_value,
        visits={candidate: child.visit_count for candidate, child in root.children.items()},
    )
