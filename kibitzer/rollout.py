# parallel game generation for the grpo loop. the model plays a fixed external
# opponent (strength-limited stockfish) and the reward is the verified game
# outcome -- NOT self-play visit targets. this is the whole point: an external
# reward can't be style-exploited against a sibling checkpoint (the trap that
# sank every prior fine-tune, D48-D54).
#
# moves are SELECTED by puct search (dirichlet root noise + a temperature schedule
# for group diversity) so the model plays at its real ~2500 searched strength and
# the outcome reflects real decisions -- the raw policy sampled at temp 1.0 hangs
# pieces into a 1600 and the reward becomes blunder-noise. we still record the RAW
# policy mu at each state (not the visit dist), so the dppo trust region fences the
# raw prior toward the base while grpo pushes it toward search-validated winners.
# sims=0 keeps a fast searchless raw path for smoke/ablation only.

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import chess
import chess.engine

from kibitzer.inference import ModelEvaluator
from kibitzer.search import puct_search


# open stockfish as a strength-capped opponent. UCI_Elo floors at 1320 in
# stockfish, so callers keep the ladder above that.
def open_stockfish(path: str, elo: int) -> chess.engine.SimpleEngine:
    engine = chess.engine.SimpleEngine.popen_uci(path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(elo)})
    return engine


# sample a move from the raw legal-move distribution with temperature. temp<=0
# is greedy argmax; temp=1 samples proportional to policy mass (honest importance
# ratios for the dppo trust region).
def sample_move(priors: dict[chess.Move, float], rng: random.Random, temp: float) -> chess.Move:
    moves = list(priors)
    if temp <= 1e-3:
        return max(moves, key=lambda m: priors[m])
    weights = [priors[m] ** (1.0 / temp) for m in moves]
    total = sum(weights) or 1.0
    r = rng.random() * total
    acc = 0.0
    for m, w in zip(moves, weights):
        acc += w
        if r <= acc:
            return m
    return moves[-1]


# reward from the model's point of view. a game hitting the ply cap has no
# outcome -> scored as a draw, which also removes the incentive to stall.
def outcome_reward(board: chess.Board, model_white: bool) -> float:
    o = board.outcome(claim_draw=True)
    if o is None or o.winner is None:
        return 0.5
    return 1.0 if (o.winner == chess.WHITE) == model_white else 0.0


@dataclass
class _Game:
    board: chess.Board
    model_white: bool
    group_id: int
    game_id: int
    plies: int = 0
    records: list[tuple[str, str, dict[str, float]]] = field(default_factory=list)


# specs: one (opening_ucis, model_white, group_id, game_id) per game. games in
# the same group share opening+color+opponent so their reward z-score is a clean
# baseline. returns flat per-ply records tagged with the game's final reward.
def generate(
    evaluator: ModelEvaluator,
    engine: chess.engine.SimpleEngine,
    specs: list[tuple[str, bool, int, int]],
    *,
    sims: int,
    max_plies: int,
    rng: random.Random,
    temp: float = 0.8,
    temp_plies: int = 16,
    temp_late: float = 0.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    engine_time: float = 0.01,
    log_prefix: str | None = None,
) -> list[dict]:
    games: list[_Game] = []
    for opening, model_white, group_id, game_id in specs:
        board = chess.Board()
        for uci in opening.split():
            board.push_uci(uci)
        games.append(_Game(board, model_white, group_id, game_id))

    def still_active(g: _Game) -> bool:
        return g.plies < max_plies and not g.board.is_game_over(claim_draw=True)

    # temperature schedule: spread the opening (group diversity so grpo has
    # informative reward variance), then near-greedy so the rest is played at
    # strength and the outcome credits real decisions.
    def ply_temp(board: chess.Board) -> float:
        return temp if board.ply() < temp_plies else temp_late

    total = len(games)
    start = time.monotonic()
    last_log = start
    active = [g for g in games if still_active(g)]
    while active:
        model_games = [g for g in active if (g.board.turn == chess.WHITE) == g.model_white]
        engine_games = [g for g in active if (g.board.turn == chess.WHITE) != g.model_white]
        if model_games and sims > 0:
            # search move selection; mu is the raw prior (one extra forward, tiny
            # next to `sims` leaf evals). no cross-game batching -- search is
            # sequential per position.
            for g in model_games:
                res = puct_search(g.board, evaluator, simulations=sims,
                                  dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon)
                mu = {m.uci(): float(p) for m, p in evaluator.evaluate(g.board).priors.items()}
                move = sample_move(res.visits, rng, ply_temp(g.board))
                g.records.append((g.board.fen(), move.uci(), mu))
                g.board.push(move)
                g.plies += 1
        elif model_games:
            # searchless raw path (smoke/ablation): one batched forward for all games.
            evals = evaluator.evaluate_batch([g.board for g in model_games])
            for g, ev in zip(model_games, evals):
                move = sample_move(ev.priors, rng, ply_temp(g.board))
                mu = {m.uci(): float(p) for m, p in ev.priors.items()}
                g.records.append((g.board.fen(), move.uci(), mu))
                g.board.push(move)
                g.plies += 1
        # opponent plies, sequential on cpu
        for g in engine_games:
            result = engine.play(g.board, chess.engine.Limit(time=engine_time))
            if result.move is None:
                break
            g.board.push(result.move)
            g.plies += 1
        active = [g for g in games if still_active(g)]
        # live progress so a 20-min rollout isn't a silent wait: finished games,
        # running score, and a linear-extrapolation eta. throttled to ~20s.
        if log_prefix is not None:
            now = time.monotonic()
            finished = total - len(active)
            if finished > 0 and (now - last_log > 20.0 or not active):
                active_ids = {id(g) for g in active}
                w = d = l = 0
                for g in games:
                    if id(g) in active_ids:
                        continue
                    r = outcome_reward(g.board, g.model_white)
                    w, d, l = (w + (r == 1.0), d + (r == 0.5), l + (r == 0.0))
                pos = sum(len(g.records) for g in games)
                elapsed = now - start
                eta = elapsed / finished * (total - finished)
                print(f"  [{log_prefix}] {finished}/{total} games  {pos} pos  "
                      f"{w}W/{d}D/{l}L  score={(w + 0.5 * d) / max(1, finished):.3f}  "
                      f"{elapsed / 60:.1f}m elapsed  ~{eta / 60:.1f}m left", flush=True)
                last_log = now

    out: list[dict] = []
    for g in games:
        reward = outcome_reward(g.board, g.model_white)
        for fen, action, mu in g.records:
            out.append({
                "fen": fen, "action": action, "mu": mu,
                "reward": reward, "group_id": g.group_id, "game_id": g.game_id,
            })
    return out


# per-game win/draw/loss tally over a record buffer, for the adaptive ladder and
# metrics. reward is stored per-ply so dedupe by game_id first.
def game_results(records: list[dict]) -> tuple[int, int, int]:
    by_game: dict[int, float] = {}
    for r in records:
        by_game[r["game_id"]] = r["reward"]
    w = sum(1 for v in by_game.values() if v == 1.0)
    d = sum(1 for v in by_game.values() if v == 0.5)
    l = sum(1 for v in by_game.values() if v == 0.0)
    return w, d, l


# fraction of groups whose games did not all share one result. a group with a
# single result gives zero grpo advantage, so this is the live "how much of the
# rollout actually taught anything" signal the ladder is trying to keep high.
def informative_group_fraction(records: list[dict]) -> float:
    by_group: dict[int, set[float]] = {}
    game_seen: set[int] = set()
    for r in records:
        if r["game_id"] in game_seen:
            continue
        game_seen.add(r["game_id"])
        by_group.setdefault(r["group_id"], set()).add(r["reward"])
    if not by_group:
        return 0.0
    informative = sum(1 for results in by_group.values() if len(results) > 1)
    return informative / len(by_group)
