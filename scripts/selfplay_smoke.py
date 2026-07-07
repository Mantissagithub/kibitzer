# minimal alphazero-lite self-play smoke: does self-play from the strong 100M base
# improve on itself (or at least not collapse like the old weak-base attempts)?
# gen: model plays itself with temperature exploration for game diversity, and we
# record (position, the search-improved best move, game outcome z). train: behavioral
# clone toward those search-improved moves + value toward z (reuses model.loss).
# match: new model vs the base, puct both, head-to-head score. if new beats base,
# the self-play direction is validated. see the strength-ceiling investigation.

from __future__ import annotations

import argparse
import glob
import json
import math
import random
from pathlib import Path

import chess
import torch
from torch.utils.data import DataLoader

from kibitzer.data import PositionSample, collate_positions, encode_position_sample, result_to_value
from kibitzer.inference import ModelEvaluator
from kibitzer.model import Kibitzer
from kibitzer.search import puct_search

OPENING_BOOK = [
    "e2e4 e7e5 g1f3 b8c6", "e2e4 c7c5 g1f3 d7d6", "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5", "d2d4 d7d5 c2c4 e7e6", "d2d4 g8f6 c2c4 g7g6",
    "d2d4 g8f6 c2c4 e7e6", "c2c4 e7e5 b1c3 g8f6", "g1f3 d7d5 d2d4 g8f6",
    "e2e4 e7e5 g1f3 g8f6", "d2d4 d7d5 g1f3 g8f6", "e2e4 g7g6 d2d4 f8g7",
]


def book_board(rng: random.Random) -> chess.Board:
    board = chess.Board()
    for uci in rng.choice(OPENING_BOOK).split():
        board.push_uci(uci)
    return board


# sample a move from the visit counts with temperature (exploration); temp->0 = argmax.
def sample_move(visits: dict[chess.Move, int], rng: random.Random, temp: float) -> chess.Move:
    moves = list(visits)
    counts = [visits[m] for m in moves]
    if temp <= 1e-3 or sum(counts) == 0:
        return moves[max(range(len(moves)), key=lambda i: counts[i])]
    weights = [c ** (1.0 / temp) for c in counts]
    total = sum(weights) or 1.0
    r = rng.random() * total
    acc = 0.0
    for m, w in zip(moves, weights):
        acc += w
        if r <= acc:
            return m
    return moves[-1]


def gen(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out = args.out_jsonl.open("w", encoding="utf-8")
    for _ in range(args.games):
        board = book_board(rng)
        records = []  # (fen, best_move_uci, turn)
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < args.max_plies:
            res = puct_search(board, evaluator, simulations=args.sims)
            # policy target = the search-improved best move (argmax visits)
            best = max(res.visits.items(), key=lambda kv: kv[1])[0]
            records.append((board.fen(), best.uci(), board.turn))
            # play with temperature exploration early for game diversity
            temp = 1.0 if plies < args.temp_plies else 0.0
            board.push(sample_move(res.visits, rng, temp))
            plies += 1
        outcome = board.outcome(claim_draw=True)
        result = outcome.result() if outcome is not None else "1/2-1/2"
        for fen, move, turn in records:
            out.write(json.dumps({"fen": fen, "move": move, "value": result_to_value(result, turn)}) + "\n")
        out.flush()
    out.close()


def train(args: argparse.Namespace) -> None:
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Kibitzer(payload["config"]).to(args.device)
    model.load_state_dict(payload["model"])
    samples = []
    for f in sorted(glob.glob(args.data)):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                samples.append(PositionSample(fen=d["fen"], move_uci=d["move"], value=float(d["value"])))
    print(f"training on {len(samples)} self-play positions")
    encoded = [encode_position_sample(s) for s in samples]
    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, collate_fn=collate_positions)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    for epoch in range(args.epochs):
        tot = 0.0
        for batch in loader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, _ = model.loss(**batch, value_weight=args.value_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
        print(f"epoch {epoch+1}: mean loss {tot/max(1,len(loader)):.4f}")
    torch.save({"model": model.state_dict(), "config": model.config, "training_objective": "selfplay_smoke"}, args.out)
    print(f"saved {args.out}")


def match(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    a = ModelEvaluator.from_checkpoint(args.model_a, device=args.device)
    b = ModelEvaluator.from_checkpoint(args.model_b, device=args.device)
    w = d = l = 0
    for i in range(args.games):
        board = book_board(rng)
        a_white = i % 2 == 0
        while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
            ev = a if (board.turn == chess.WHITE) == a_white else b
            board.push(puct_search(board, ev, simulations=args.sims).move)
        o = board.outcome(claim_draw=True)
        if o is None or o.winner is None:
            d += 1
        elif (o.winner == chess.WHITE) == a_white:
            w += 1
        else:
            l += 1
        print(f"game {i+1}: A {w}W/{d}D/{l}L  score={(w+0.5*d)/(w+d+l):.3f}", flush=True)
    score = (w + 0.5 * d) / max(1, args.games)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"model_a": str(args.model_a), "model_b": str(args.model_b),
                                         "w": w, "d": d, "l": l, "a_score": score}, indent=2))
    print(f"A (new) vs B (base): {w}W/{d}D/{l}L  A_score={score:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--games", type=int, default=25)
    g.add_argument("--sims", type=int, default=64)
    g.add_argument("--temp-plies", type=int, default=16)
    g.add_argument("--max-plies", type=int, default=160)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out-jsonl", type=Path, required=True)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t = sub.add_parser("train")
    t.add_argument("--checkpoint", type=Path, required=True)
    t.add_argument("--data", required=True, help="glob of gen jsonl")
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--value-weight", type=float, default=0.5)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    m = sub.add_parser("match")
    m.add_argument("--model-a", type=Path, required=True)
    m.add_argument("--model-b", type=Path, required=True)
    m.add_argument("--games", type=int, default=40)
    m.add_argument("--sims", type=int, default=64)
    m.add_argument("--max-plies", type=int, default=200)
    m.add_argument("--seed", type=int, default=100)
    m.add_argument("--out-json", type=Path, required=True)
    m.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    {"gen": gen, "train": train, "match": match}[args.mode](args)


if __name__ == "__main__":
    main()
