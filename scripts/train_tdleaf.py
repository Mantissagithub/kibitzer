# tdleaf(lambda): the value head learns from its own search leaves, online vs
# stockfish. adapts knightcap's tdleaf to the puct/mcts model -- the search-backed
# root value from puct_search (mean over simulations, side-to-move pov) is the
# analog of knightcap's minimaxed principal-variation leaf eval. over the model's
# own-move sequence v_0..v_{t-1} (model pov) with terminal result z we form
# td(lambda) targets and regress the raw value head onto them. encoder, trunk and
# policy head stay frozen (puct leans on the sft priors); only the value head and
# the final rmsnorm train, so the priors that drive search stay intact. novelty
# vs the d30-d35 value-repair campaign: value *learns from* its own search leaves
# online rather than searching at inference over a statically trained head. see
# decisions d40.

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import chess
import chess.engine
import torch

from kibitzer.encoding import board_to_tensor
from kibitzer.inference import ModelEvaluator
from kibitzer.search import puct_search

LEVELS = [1320, 1500, 1700, 1900, 2100]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TDLeaf(lambda) value repair vs Stockfish.")
    p.add_argument("--checkpoint", type=Path, default=Path("runs/scaling/S2.pt"))
    p.add_argument("--out", type=Path, default=Path("runs/tdleaf/tdleaf.pt"))
    p.add_argument("--log", type=Path, default=Path("reports/tdleaf/tdleaf_log.jsonl"))
    p.add_argument("--games", type=int, default=200, help="total self-play games")
    p.add_argument("--games-per-update", type=int, default=4)
    p.add_argument("--simulations", type=int, default=64)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--value-scale", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.7, help="TD(lambda) decay")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-steps", type=int, default=1, help="grad steps per update batch")
    p.add_argument("--stockfish-path", default="stockfish")
    p.add_argument("--stockfish-elo", type=int, default=1320)
    p.add_argument("--stockfish-time", type=float, default=0.05)
    p.add_argument("--max-plies", type=int, default=160)
    p.add_argument("--opening-random-plies", type=int, default=6)
    p.add_argument("--curriculum-window", type=int, default=20)
    p.add_argument("--curriculum-threshold", type=float, default=0.6)
    p.add_argument("--save-every", type=int, default=40, help="save checkpoint every N games")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# diversify games with a few random legal plies (stockfish+puct are ~deterministic).
def random_opening(rng: random.Random, plies: int) -> chess.Board:
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over(claim_draw=True):
            break
        board.push(rng.choice(moves))
    return board


# forward-view td(lambda) targets. values = model-pov search root values;
# terminal bootstrap v_t = z (game result, model pov). no intermediate reward.
def td_lambda_targets(values: list[float], z: float, lam: float) -> list[float]:
    ext = values + [z]
    deltas = [ext[t + 1] - ext[t] for t in range(len(values))]
    targets = [0.0] * len(values)
    running = 0.0
    for t in range(len(values) - 1, -1, -1):
        running = deltas[t] + lam * running
        targets[t] = values[t] + running
    return targets


def play_training_game(
    *,
    board: chess.Board,
    network_color: bool,
    evaluator: ModelEvaluator,
    engine: chess.engine.SimpleEngine,
    simulations: int,
    c_puct: float,
    value_scale: float,
    stockfish_time: float,
    max_plies: int,
) -> tuple[list[chess.Board], list[float], float]:
    # returns (model-move boards, search root values at those boards, model-pov result z).
    boards: list[chess.Board] = []
    values: list[float] = []
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == network_color:
            result = puct_search(
                board,
                evaluator,
                simulations=simulations,
                c_puct=c_puct,
                value_scale=value_scale,
            )
            boards.append(board.copy(stack=False))
            values.append(result.root_value)  # side-to-move POV == model POV here
            move = result.move
        else:
            played = engine.play(board, chess.engine.Limit(time=stockfish_time))
            if played.move is None:
                raise RuntimeError("Stockfish returned no move for a non-terminal board")
            move = played.move
        board.push(move)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        z = 0.0
    else:
        z = 1.0 if outcome.winner == network_color else -1.0
    return boards, values, z


def value_grad_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    boards: list[chess.Board],
    targets: list[float],
    device: str,
    grad_steps: int,
) -> float:
    piece = torch.stack([board_to_tensor(b)["piece_idx"] for b in boards]).unsqueeze(1).to(device)
    aux = torch.stack([board_to_tensor(b)["aux"] for b in boards]).unsqueeze(1).to(device)
    tgt = torch.tensor(targets, dtype=torch.float32, device=device)
    last = 0.0
    for _ in range(grad_steps):
        _, value = model(piece, aux)
        pred = value[:, -1, 0]
        loss = torch.nn.functional.mse_loss(pred, tgt)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last = float(loss.item())
    return last


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    model = evaluator.model  # shared object: grad steps update the same weights search uses

    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.value_head.parameters():
        p.requires_grad_(True)
    for p in model.norm.parameters():
        p.requires_grad_(True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    n_trainable = sum(p.numel() for p in trainable)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = args.log.open("w", encoding="utf-8")

    level_idx = 0
    recent = deque(maxlen=args.curriculum_window)
    batch_boards: list[chess.Board] = []
    batch_targets: list[float] = []
    games_played = 0
    updates = 0
    start = time.time()

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    try:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": LEVELS[level_idx]})
        print(
            f"TDLeaf: base={args.checkpoint} trainable_params={n_trainable} "
            f"start_elo={LEVELS[level_idx]} sims={args.simulations} lam={args.lam}"
        )
        while games_played < args.games:
            board = random_opening(rng, args.opening_random_plies)
            if board.is_game_over(claim_draw=True):
                continue
            network_color = chess.WHITE if games_played % 2 == 0 else chess.BLACK
            t0 = time.time()
            boards, values, z = play_training_game(
                board=board,
                network_color=network_color,
                evaluator=evaluator,
                engine=engine,
                simulations=args.simulations,
                c_puct=args.c_puct,
                value_scale=args.value_scale,
                stockfish_time=args.stockfish_time,
                max_plies=args.max_plies,
            )
            game_s = time.time() - t0
            games_played += 1
            recent.append(0.5 * (z + 1.0))  # 1 win / 0.5 draw / 0 loss

            if boards:
                targets = td_lambda_targets(values, z, args.lam)
                batch_boards.extend(boards)
                batch_targets.extend(targets)

            loss = None
            if games_played % args.games_per_update == 0 and batch_boards:
                loss = value_grad_step(
                    model, optimizer, batch_boards, batch_targets, args.device, args.grad_steps
                )
                batch_boards, batch_targets = [], []
                updates += 1

            rolling = sum(recent) / len(recent)
            record = {
                "game": games_played,
                "elo": LEVELS[level_idx],
                "z": z,
                "model_moves": len(boards),
                "rolling_score": round(rolling, 3),
                "loss": None if loss is None else round(loss, 5),
                "game_s": round(game_s, 2),
                "updates": updates,
            }
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(
                f"g{games_played:04d} elo{LEVELS[level_idx]} z={z:+.0f} "
                f"mv={len(boards):2d} roll={rolling:.2f} "
                f"loss={'-' if loss is None else f'{loss:.4f}'} {game_s:.1f}s"
            )

            if (
                len(recent) == args.curriculum_window
                and rolling >= args.curriculum_threshold
                and level_idx < len(LEVELS) - 1
            ):
                level_idx += 1
                engine.configure({"UCI_LimitStrength": True, "UCI_Elo": LEVELS[level_idx]})
                recent.clear()
                print(f"  curriculum: advance to Stockfish {LEVELS[level_idx]}")

            if games_played % args.save_every == 0:
                torch.save({"model": model.state_dict(), "config": model.config}, args.out)
    finally:
        engine.quit()
        log_file.close()

    torch.save({"model": model.state_dict(), "config": model.config}, args.out)
    elapsed = time.time() - start
    print(
        f"done: {games_played} games, {updates} updates, {elapsed / 60:.1f} min, "
        f"{elapsed / max(1, games_played):.1f}s/game -> {args.out}"
    )


if __name__ == "__main__":
    main()
