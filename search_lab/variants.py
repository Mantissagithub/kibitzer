# search-variant lab: each variant is a picker(board, ev, budget) -> chess.Move,
# blending a different search algorithm over the SAME policy/value net so we can
# compare their cap at equal compute. "budget" = net evaluations per move, so
# mcts sims and alpha-beta nodes are compared on the same footing. see the search
# axis table / LOGBOOK D50.

from __future__ import annotations

import math

import chess

from kibitzer.search import puct_search

INF = float("inf")


# counts every net evaluation and holds the per-move budget so the alpha-beta
# variants can stop searching once they've spent as much compute as mcts would.
class CountingEvaluator:
    def __init__(self, base) -> None:
        self._base = base
        self.count = 0
        self.budget = 1 << 30

    def evaluate(self, board):
        self.count += 1
        return self._base.evaluate(board)


def _terminal(board: chess.Board) -> float:
    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        return 0.0
    return 1.0 if o.winner == board.turn else -1.0


# ---- puct family (local reimpl so we can toggle fpu / cpuct(s) / prior pruning) ----

class _Node:
    __slots__ = ("prior", "visits", "value_sum", "children")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children = None

    @property
    def q(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


def _cpuct(n: int, c_init: float, c_base: float) -> float:
    return math.log((n + c_base + 1) / c_base) + c_init


def _expand(node: _Node, board: chess.Board, ev, prune: float) -> float:
    result = ev.evaluate(board)
    priors = result.priors
    if prune > 0.0 and priors:
        top = max(priors.values())
        kept = {m: p for m, p in priors.items() if p >= prune * top}
        tot = sum(kept.values()) or 1.0
        priors = {m: p / tot for m, p in kept.items()}
    node.children = {m: _Node(p) for m, p in priors.items()}
    return result.value


def _select(node: _Node, c_init: float, c_base: float, fpu: float):
    scale = math.sqrt(max(1, node.visits))
    cp = _cpuct(node.visits, c_init, c_base)

    def score(item):
        _, ch = item
        # child q is from the child's pov, so negate for the parent; unvisited
        # children take the fpu value (0 = optimistic az, negative = leela-style reduction)
        q = -ch.q if ch.visits > 0 else fpu
        u = cp * ch.prior * scale / (1 + ch.visits)
        return q + u

    return max(node.children.items(), key=score)


def _local_puct(board, ev, sims, *, c_init=1.25, c_base=19652.0, fpu=0.0, prune=0.0) -> chess.Move:
    root = _Node(1.0)
    _expand(root, board, ev, prune)
    for _ in range(sims):
        b = board.copy(stack=True)
        node = root
        path = [node]
        while node.children:
            move, node = _select(node, c_init, c_base, fpu)
            b.push(move)
            path.append(node)
            if b.is_game_over(claim_draw=True):
                break
        value = _terminal(b) if b.is_game_over(claim_draw=True) else _expand(node, b, ev, prune)
        for nd in reversed(path):
            nd.visits += 1
            nd.value_sum += value
            value = -value
    return max(root.children.items(), key=lambda kv: kv[1].visits)[0]


# ---- alpha-beta family (depth-d negamax, value-net leaf, policy move ordering) ----

def _quiesce(board, ev, alpha, beta):
    # only trust the value net at quiet positions: extend on captures to dodge
    # the horizon effect (the net's known tactical blind spot).
    if board.is_game_over(claim_draw=True):
        return _terminal(board)
    if ev.count >= ev.budget:
        return ev.evaluate(board).value
    stand = ev.evaluate(board).value
    if stand >= beta:
        return beta
    if stand > alpha:
        alpha = stand
    for m in board.legal_moves:
        if not board.is_capture(m):
            continue
        board.push(m)
        v = -_quiesce(board, ev, -beta, -alpha)
        board.pop()
        if v >= beta:
            return beta
        if v > alpha:
            alpha = v
    return alpha


def _alphabeta(board, ev, depth, alpha, beta, quiescence):
    if board.is_game_over(claim_draw=True):
        return _terminal(board)
    if ev.count >= ev.budget:
        return ev.evaluate(board).value  # out of budget -> static (anytime cutoff)
    if depth <= 0:
        return _quiesce(board, ev, alpha, beta) if quiescence else ev.evaluate(board).value
    result = ev.evaluate(board)  # value discarded here, priors used for ordering
    moves = sorted(board.legal_moves, key=lambda m: result.priors.get(m, 0.0), reverse=True)
    best = -INF
    for m in moves:
        board.push(m)
        v = -_alphabeta(board, ev, depth - 1, -beta, -alpha, quiescence)
        board.pop()
        if v > best:
            best = v
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def _iterative_alphabeta(board, ev, budget, quiescence) -> chess.Move:
    ev.budget = budget
    best_move = None
    depth = 1
    while ev.count < budget and depth <= 40:
        result = ev.evaluate(board)
        moves = sorted(board.legal_moves, key=lambda m: result.priors.get(m, 0.0), reverse=True)
        alpha, bv, local_best = -INF, -INF, None
        for m in moves:
            board.push(m)
            v = -_alphabeta(board, ev, depth - 1, -INF, INF, quiescence)
            board.pop()
            if v > bv:
                bv, local_best = v, m
            if bv > alpha:
                alpha = bv
            if ev.count >= budget:
                break
        if local_best is not None:
            best_move = local_best
        depth += 1
    return best_move or next(iter(board.legal_moves))


# ---- variant registry ----

def baseline_puct(board, ev, budget):
    return puct_search(board, ev, simulations=budget).move


def puct_fpu(board, ev, budget):
    return _local_puct(board, ev, budget, c_init=1.25, c_base=19652.0, fpu=-0.2)


def puct_prune(board, ev, budget):
    return _local_puct(board, ev, budget, prune=0.15)


def alphabeta(board, ev, budget):
    return _iterative_alphabeta(board, ev, budget, quiescence=False)


def alphabeta_quiescence(board, ev, budget):
    return _iterative_alphabeta(board, ev, budget, quiescence=True)


# the two winners stacked: fpu + cpuct(s) scaling + prior-threshold pruning, the
# best search axis from the first sweep (D50).
def puct_stacked(board, ev, budget):
    return _local_puct(board, ev, budget, c_init=1.25, c_base=19652.0, fpu=-0.2, prune=0.15)


VARIANTS = {
    "baseline_puct": baseline_puct,
    "puct_fpu": puct_fpu,
    "puct_prune": puct_prune,
    "alphabeta": alphabeta,
    "alphabeta_quiescence": alphabeta_quiescence,
    "puct_stacked": puct_stacked,
}
