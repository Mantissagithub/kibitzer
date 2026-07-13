"""PUCT search over policy priors and side-to-move value estimates."""

from __future__ import annotations

import math
import random
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
    simulations: int
    normalized_entropy: float
    top2_ratio: float
    value_delta: float = 0.0
    stop_reason: str = "fixed"


def terminal_value(board: chess.Board, *, claim_draw: bool = True) -> float:
    outcome = board.outcome(claim_draw=claim_draw)
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


def _root_metrics(root: SearchNode) -> tuple[float, float]:
    counts = sorted((child.visit_count for child in root.children.values()), reverse=True)
    total = sum(counts)
    if total == 0:
        return 0.0, 0.0

    entropy = 0.0
    for count in counts:
        if count > 0:
            probability = count / total
            entropy -= probability * math.log(probability)
    normalizer = math.log(len(counts)) if len(counts) > 1 else 1.0
    normalized_entropy = entropy / normalizer
    top2_ratio = counts[1] / counts[0] if len(counts) > 1 and counts[0] > 0 else 0.0
    return normalized_entropy, top2_ratio


def _run_simulations(
    root: SearchNode,
    board: chess.Board,
    evaluator: PositionEvaluator,
    *,
    simulations: int,
    c_puct: float,
    value_scale: float,
    claim_draw: bool,
) -> None:
    for _ in range(simulations):
        simulation_board = board.copy(stack=True)
        node = root
        path = [node]

        while node.children:
            move, node = _select_child(node, c_puct, value_scale)
            simulation_board.push(move)
            path.append(node)
            if simulation_board.is_game_over(claim_draw=claim_draw):
                break

        if simulation_board.is_game_over(claim_draw=claim_draw):
            value = terminal_value(simulation_board, claim_draw=claim_draw)
        else:
            evaluation = evaluator.evaluate(simulation_board)
            _expand(node, evaluation)
            value = evaluation.value

        for visited in reversed(path):
            visited.visit_count += 1
            visited.value_sum += value
            value = -value


def _search_result(
    root: SearchNode,
    *,
    simulations: int,
    value_delta: float = 0.0,
    stop_reason: str = "fixed",
) -> SearchResult:
    move, _ = max(root.children.items(), key=lambda item: item[1].visit_count)
    normalized_entropy, top2_ratio = _root_metrics(root)
    return SearchResult(
        move=move,
        root_value=root.mean_value,
        visits={candidate: child.visit_count for candidate, child in root.children.items()},
        simulations=simulations,
        normalized_entropy=normalized_entropy,
        top2_ratio=top2_ratio,
        value_delta=value_delta,
        stop_reason=stop_reason,
    )


def puct_search(
    board: chess.Board,
    evaluator: PositionEvaluator,
    *,
    simulations: int,
    c_puct: float = 1.5,
    value_scale: float = 1.0,
    dirichlet_alpha: float = 0.0,
    dirichlet_epsilon: float = 0.0,
    claim_draw: bool = True,
) -> SearchResult:
    if simulations < 1:
        raise ValueError("simulations must be at least 1")
    if value_scale < 0.0:
        raise ValueError("value_scale must be non-negative")
    if board.is_game_over(claim_draw=claim_draw):
        raise ValueError("cannot search a terminal board")

    root = SearchNode()
    _expand(root, evaluator.evaluate(board))

    # alphazero root exploration: mix dirichlet noise into the root priors so
    # self-play games diverge. off by default (epsilon=0) => inference unchanged.
    if dirichlet_epsilon > 0.0 and dirichlet_alpha > 0.0 and root.children:
        moves = list(root.children)
        gammas = [random.gammavariate(dirichlet_alpha, 1.0) for _ in moves]
        total = sum(gammas) or 1.0
        for move, g in zip(moves, gammas):
            child = root.children[move]
            child.prior = (1.0 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * (g / total)

    _run_simulations(
        root,
        board,
        evaluator,
        simulations=simulations,
        c_puct=c_puct,
        value_scale=value_scale,
        claim_draw=claim_draw,
    )
    return _search_result(root, simulations=simulations)


def adaptive_puct_search(
    board: chess.Board,
    evaluator: PositionEvaluator,
    *,
    stages: tuple[int, ...] = (128, 256, 512, 1024),
    entropy_threshold: float = 0.55,
    top2_ratio_threshold: float = 0.75,
    value_delta_threshold: float = 0.10,
    c_puct: float = 1.5,
    value_scale: float = 1.0,
    claim_draw: bool = True,
) -> SearchResult:
    if not stages or stages[0] < 1 or any(a >= b for a, b in zip(stages, stages[1:])):
        raise ValueError("stages must be strictly increasing positive simulation counts")
    if not 0.0 <= entropy_threshold <= 1.0:
        raise ValueError("entropy_threshold must be in [0, 1]")
    if not 0.0 <= top2_ratio_threshold <= 1.0:
        raise ValueError("top2_ratio_threshold must be in [0, 1]")
    if value_delta_threshold < 0.0:
        raise ValueError("value_delta_threshold must be non-negative")
    if value_scale < 0.0:
        raise ValueError("value_scale must be non-negative")
    if board.is_game_over(claim_draw=claim_draw):
        raise ValueError("cannot search a terminal board")

    root = SearchNode()
    _expand(root, evaluator.evaluate(board))
    previous_value: float | None = None
    value_delta = 0.0

    for index, target in enumerate(stages):
        already_run = stages[index - 1] if index > 0 else 0
        _run_simulations(
            root,
            board,
            evaluator,
            simulations=target - already_run,
            c_puct=c_puct,
            value_scale=value_scale,
            claim_draw=claim_draw,
        )
        normalized_entropy, top2_ratio = _root_metrics(root)
        value_delta = abs(root.mean_value - previous_value) if previous_value is not None else 0.0
        previous_value = root.mean_value

        uncertain = (
            normalized_entropy >= entropy_threshold
            or top2_ratio >= top2_ratio_threshold
            or (index > 0 and value_delta >= value_delta_threshold)
        )
        if not uncertain:
            return _search_result(
                root,
                simulations=target,
                value_delta=value_delta,
                stop_reason="stable",
            )

    return _search_result(
        root,
        simulations=stages[-1],
        value_delta=value_delta,
        stop_reason="max",
    )
