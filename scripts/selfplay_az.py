# proper alphazero, one iteration: self-play with dirichlet root noise, train the
# policy toward the full mcts VISIT DISTRIBUTION (soft cross-entropy, not the hard
# argmax move) and the value toward the game outcome, then head-to-head vs the base.
# this is the real az objective, unlike the earlier bc-on-best-move smoke (D48).

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from kibitzer.data import result_to_value
from kibitzer.encoding import ACTION_SIZE, board_to_tensor, legal_move_mask, move_to_index
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


# sample a move from visit counts with temperature; temp<=0 => argmax.
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
    import time

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    total_positions = 0
    with args.out_jsonl.open("w", encoding="utf-8") as out:
        for g in range(args.games):
            board = book_board(rng)
            records = []
            plies = 0
            while not board.is_game_over(claim_draw=True) and plies < args.max_plies:
                res = puct_search(
                    board, evaluator, simulations=args.sims,
                    dirichlet_alpha=args.dirichlet_alpha, dirichlet_epsilon=args.dirichlet_epsilon,
                )
                visits = {m.uci(): c for m, c in res.visits.items()}
                records.append((board.fen(), visits, board.turn))
                temp = 1.0 if plies < args.temp_plies else 0.0
                board.push(sample_move(res.visits, rng, temp))
                plies += 1
            outcome = board.outcome(claim_draw=True)
            result = outcome.result() if outcome is not None else "1/2-1/2"
            for fen, visits, turn in records:
                out.write(json.dumps({"fen": fen, "visits": visits, "value": result_to_value(result, turn)}) + "\n")
            out.flush()
            total_positions += len(records)
            elapsed = time.monotonic() - start
            eta = (elapsed / (g + 1)) * (args.games - g - 1)
            print(f"[gen {g+1}/{args.games}] {plies} plies  {len(records)} pos  total {total_positions} pos  result {result}  {elapsed/60:.1f}m elapsed  ~{eta/60:.1f}m left", flush=True)
    elapsed = time.monotonic() - start
    print(f"[gen] done: {args.games} games, {total_positions} positions, {elapsed/60:.1f} min")


# az targets: dense visit-count distribution over the action space + value.
class AZDataset(Dataset):
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        board = chess.Board(s["fen"])
        enc = board_to_tensor(board)
        total = sum(s["visits"].values()) or 1
        idxs, probs = [], []
        for uci, count in s["visits"].items():
            idxs.append(move_to_index(chess.Move.from_uci(uci), board))
            probs.append(count / total)
        return {
            "piece_idx": enc["piece_idx"],
            "aux": enc["aux"],
            "legal_mask": legal_move_mask(board),
            "target_idx": torch.tensor(idxs, dtype=torch.long),
            "target_prob": torch.tensor(probs, dtype=torch.float32),
            "value": torch.tensor(s["value"], dtype=torch.float32),
        }


def az_collate(batch: list[dict]) -> dict:
    b = len(batch)
    piece_idx = torch.stack([x["piece_idx"] for x in batch])
    aux = torch.stack([x["aux"] for x in batch])
    legal = torch.stack([x["legal_mask"] for x in batch])
    value = torch.stack([x["value"] for x in batch])
    target = torch.zeros(b, ACTION_SIZE, dtype=torch.float32)
    for i, x in enumerate(batch):
        target[i, x["target_idx"]] = x["target_prob"]
    return {"piece_idx": piece_idx, "aux": aux, "legal_mask": legal, "target_policy": target, "value": value}


def train(args: argparse.Namespace) -> None:
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Kibitzer(payload["config"]).to(args.device)
    model.load_state_dict(payload["model"])
    samples = []
    for f in sorted(glob.glob(args.data)):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                samples.append(json.loads(line))
    print(f"training on {len(samples)} self-play positions (soft visit-distribution targets)")
    loader = DataLoader(AZDataset(samples), batch_size=args.batch_size, shuffle=True, collate_fn=az_collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()
    for epoch in range(args.epochs):
        tp = tv = 0.0
        for batch in loader:
            piece = batch["piece_idx"].unsqueeze(1).to(args.device)  # [B,1,64]
            aux = batch["aux"].unsqueeze(1).to(args.device)
            legal = batch["legal_mask"].to(args.device)
            target = batch["target_policy"].to(args.device)
            value_t = batch["value"].to(args.device)
            logits, value = model(piece, aux)
            logits = logits[:, -1, :].masked_fill(~legal, -1e9)
            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(target * logp).sum(dim=-1).mean()  # soft cross-entropy to visit dist
            value_loss = F.mse_loss(value[:, -1, 0], value_t)
            loss = policy_loss + args.value_weight * value_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tp += float(policy_loss.item()); tv += float(value_loss.item())
        n = max(1, len(loader))
        print(f"epoch {epoch+1}: policy {tp/n:.4f}  value {tv/n:.4f}")
    torch.save({"model": model.state_dict(), "config": model.config, "training_objective": "selfplay_az"}, args.out)
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
        print(f"game {i+1}: A {w}W/{d}D/{l}L score={(w+0.5*d)/(w+d+l):.3f}", flush=True)
    score = (w + 0.5 * d) / max(1, args.games)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"model_a": str(args.model_a), "model_b": str(args.model_b),
                                         "w": w, "d": d, "l": l, "a_score": score}, indent=2))
    print(f"A (az) vs B (base): {w}W/{d}D/{l}L  A_score={score:.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--games", type=int, default=25)
    g.add_argument("--sims", type=int, default=128)
    g.add_argument("--temp-plies", type=int, default=20)
    g.add_argument("--max-plies", type=int, default=160)
    g.add_argument("--dirichlet-alpha", type=float, default=0.3)
    g.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out-jsonl", type=Path, required=True)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t = sub.add_parser("train")
    t.add_argument("--checkpoint", type=Path, required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--value-weight", type=float, default=1.0)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    m = sub.add_parser("match")
    m.add_argument("--model-a", type=Path, required=True)
    m.add_argument("--model-b", type=Path, required=True)
    m.add_argument("--games", type=int, default=40)
    m.add_argument("--sims", type=int, default=128)
    m.add_argument("--max-plies", type=int, default=200)
    m.add_argument("--seed", type=int, default=100)
    m.add_argument("--out-json", type=Path, required=True)
    m.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    {"gen": gen, "train": train, "match": match}[args.mode](args)


if __name__ == "__main__":
    main()
