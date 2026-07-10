from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import chess

from kibitzer.inference import PositionEvaluation


@dataclass
class _Node:
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    raw_value: float = 0.0
    children: dict[chess.Move, _Node] = field(default_factory=dict)

    @property
    def q(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


@dataclass(frozen=True)
class GumbelSearchResult:
    move: chess.Move
    root_value: float
    visits: dict[chess.Move, int]


def _terminal_value(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        raise ValueError("board is not terminal")
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0


def _expand(node: _Node, evaluation: PositionEvaluation) -> None:
    node.raw_value = evaluation.value
    node.children = {
        move: _Node(prior=prior) for move, prior in evaluation.priors.items()
    }


# this follows deepmind mctx's completed-by-mix-value transform. unvisited moves
# get the mixed node value, then q is rescaled before it changes the policy logits.
def _completed_qvalues(
    node: _Node,
    *,
    value_scale: float,
    maxvisit_init: float,
) -> dict[chess.Move, float]:
    visited = [child for child in node.children.values() if child.visits > 0]
    total_visits = sum(child.visits for child in node.children.values())
    visited_prior = sum(child.prior for child in visited)
    weighted_q = 0.0
    if visited_prior > 0.0:
        weighted_q = sum(
            child.prior * -child.q / visited_prior for child in visited
        )
    mixed_value = (node.raw_value + total_visits * weighted_q) / (total_visits + 1)
    completed = {
        move: (-child.q if child.visits > 0 else mixed_value)
        for move, child in node.children.items()
    }
    low = min(completed.values())
    high = max(completed.values())
    width = high - low
    if width <= 1e-8:
        normalized = {move: 0.0 for move in completed}
    else:
        normalized = {move: (value - low) / width for move, value in completed.items()}
    max_visits = max(child.visits for child in node.children.values())
    scale = (maxvisit_init + max_visits) * value_scale
    return {move: scale * value for move, value in normalized.items()}


# sequential halving spends equally on every surviving move, halves the field,
# then repeats. the values are target visit counts for each simulation at the root.
def sequential_halving_schedule(num_actions: int, simulations: int) -> tuple[int, ...]:
    if num_actions < 1:
        raise ValueError("num_actions must be at least 1")
    if simulations < 1:
        raise ValueError("simulations must be at least 1")
    if num_actions == 1:
        return tuple(range(simulations))
    phases = math.ceil(math.log2(num_actions))
    schedule: list[int] = []
    visits = [0] * num_actions
    considered = num_actions
    while len(schedule) < simulations:
        extra_visits = max(1, simulations // (phases * considered))
        for _ in range(extra_visits):
            schedule.extend(visits[:considered])
            for index in range(considered):
                visits[index] += 1
        considered = max(2, considered // 2)
    return tuple(schedule[:simulations])


def _improved_probabilities(
    node: _Node,
    *,
    value_scale: float,
    maxvisit_init: float,
) -> dict[chess.Move, float]:
    completed_q = _completed_qvalues(
        node,
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
    )
    logits = {
        move: math.log(max(child.prior, 1e-30)) + completed_q[move]
        for move, child in node.children.items()
    }
    max_logit = max(logits.values())
    weights = {move: math.exp(logit - max_logit) for move, logit in logits.items()}
    total = sum(weights.values()) or 1.0
    return {move: weight / total for move, weight in weights.items()}


def _interior_child(
    node: _Node,
    *,
    value_scale: float,
    maxvisit_init: float,
) -> tuple[chess.Move, _Node]:
    improved = _improved_probabilities(
        node,
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
    )
    total_visits = sum(child.visits for child in node.children.values())

    def score(item: tuple[chess.Move, _Node]) -> float:
        move, child = item
        return improved[move] - child.visits / (1 + total_visits)

    return max(node.children.items(), key=score)


def _sample_gumbel(rng: random.Random) -> float:
    uniform = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
    return -math.log(-math.log(uniform))


def gumbel_search(
    board: chess.Board,
    evaluator,
    *,
    simulations: int,
    max_num_considered_actions: int = 16,
    gumbel_scale: float = 0.0,
    value_scale: float = 0.1,
    maxvisit_init: float = 50.0,
    rng: random.Random | None = None,
) -> GumbelSearchResult:
    if simulations < 1:
        raise ValueError("simulations must be at least 1")
    if max_num_considered_actions < 1:
        raise ValueError("max_num_considered_actions must be at least 1")
    if gumbel_scale < 0.0:
        raise ValueError("gumbel_scale must be non-negative")
    if board.is_game_over(claim_draw=True):
        raise ValueError("cannot search a terminal board")

    search_rng = rng if rng is not None else random.Random()
    root = _Node()
    _expand(root, evaluator.evaluate(board))
    prior_logits = {
        move: math.log(max(child.prior, 1e-30))
        for move, child in root.children.items()
    }
    gumbels = {
        move: gumbel_scale * _sample_gumbel(search_rng) for move in root.children
    }
    considered = min(max_num_considered_actions, len(root.children))
    schedule = sequential_halving_schedule(considered, simulations)

    for target_visits in schedule:
        completed_q = _completed_qvalues(
            root,
            value_scale=value_scale,
            maxvisit_init=maxvisit_init,
        )
        candidates = [
            (move, child)
            for move, child in root.children.items()
            if child.visits == target_visits
        ]
        if not candidates:
            candidates = list(root.children.items())
        root_move, node = max(
            candidates,
            key=lambda item: gumbels[item[0]] + prior_logits[item[0]] + completed_q[item[0]],
        )

        simulation_board = board.copy(stack=True)
        simulation_board.push(root_move)
        path = [root, node]
        while node.children and not simulation_board.is_game_over(claim_draw=True):
            move, node = _interior_child(
                node,
                value_scale=value_scale,
                maxvisit_init=maxvisit_init,
            )
            simulation_board.push(move)
            path.append(node)

        if simulation_board.is_game_over(claim_draw=True):
            value = _terminal_value(simulation_board)
        else:
            evaluation = evaluator.evaluate(simulation_board)
            _expand(node, evaluation)
            value = evaluation.value

        for visited in reversed(path):
            visited.visits += 1
            visited.value_sum += value
            value = -value

    completed_q = _completed_qvalues(
        root,
        value_scale=value_scale,
        maxvisit_init=maxvisit_init,
    )
    final_visits = max(child.visits for child in root.children.values())
    finalists = [
        (move, child)
        for move, child in root.children.items()
        if child.visits == final_visits
    ]
    move, _ = max(
        finalists,
        key=lambda item: gumbels[item[0]] + prior_logits[item[0]] + completed_q[item[0]],
    )
    return GumbelSearchResult(
        move=move,
        root_value=root.q,
        visits={candidate: child.visits for candidate, child in root.children.items()},
    )
