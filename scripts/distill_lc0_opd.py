# on-policy distillation from lc0 (leela) as a stronger teacher. the student plays
# its OWN games (on-policy positions), lc0 grades each position with its search
# visit-distribution over legal moves (~2900 at a few hundred nodes), and the student
# is trained with REVERSE kl toward that distribution (mode-seeking). unlike the old
# off-policy chessbot policy-distill (sub-SFT), this fixes the distribution mismatch
# and distills a demonstrably stronger teacher (lc0 nodes=1 beats our search 0.78).
#
# gen:
#   uv run python scripts/distill_lc0_opd.py gen \
#     --checkpoint runs/scaling_shaw_comp/S2_shaw_142M_comp.pt \
#     --lc0-path data/leela/lc0 --weights data/leela/t1-256x10-distilled.pb.gz \
#     --teacher-nodes 400 --games 60 --out-jsonl runs/opd/data.jsonl
# train:
#   uv run python scripts/distill_lc0_opd.py train \
#     --checkpoint runs/scaling_shaw_comp/S2_shaw_142M_comp.pt \
#     --data runs/opd/data.jsonl --out runs/opd/opd_v1.pt
# eval (vs base + vs 2700 = lc0 nodes=1) uses selfplay_az.py match and maia_gauntlet.py.

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from kibitzer.encoding import ACTION_SIZE, board_to_tensor, legal_move_mask, move_to_index
from kibitzer.inference import ModelEvaluator
from kibitzer.model import Kibitzer

OPENING_BOOK = [
    "e2e4 e7e5 g1f3 b8c6", "e2e4 c7c5 g1f3 d7d6", "e2e4 e7e6 d2d4 d7d5",
    "e2e4 c7c6 d2d4 d7d5", "d2d4 d7d5 c2c4 e7e6", "d2d4 g8f6 c2c4 g7g6",
    "d2d4 g8f6 c2c4 e7e6", "c2c4 e7e5 b1c3 g8f6", "g1f3 d7d5 d2d4 g8f6",
    "e2e4 e7e5 g1f3 g8f6", "d2d4 d7d5 g1f3 g8f6", "e2e4 g7g6 d2d4 f8g7",
]

_STAT = re.compile(r'^([a-h][1-8][a-h][1-8][qrbn]?)\s.*N:\s*(\d+).*P:\s*([0-9.]+)%')


def book_board(rng: random.Random) -> chess.Board:
    board = chess.Board()
    for uci in rng.choice(OPENING_BOOK).split():
        board.push_uci(uci)
    return board


def open_lc0(lc0_path: str, weights: str, backend: str) -> chess.engine.SimpleEngine:
    return chess.engine.SimpleEngine.popen_uci(
        [lc0_path, f"--weights={weights}", f"--backend={backend}", "--verbose-move-stats=true"]
    )


# teacher target: lc0 visit distribution over legal moves at `nodes`. floored with a
# small epsilon (visits are sparse; reverse kl needs non-zero teacher support) and
# renormalized. falls back to the policy prior P if no visits were recorded.
def lc0_policy(engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int, eps: float) -> dict[str, float]:
    stats: dict[str, tuple[int, float]] = {}
    with engine.analysis(board, chess.engine.Limit(nodes=nodes)) as an:
        for info in an:
            s = info.get("string")
            if not s:
                continue
            m = _STAT.search(s)
            if m:
                stats[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    legal = [mv.uci() for mv in board.legal_moves]
    visits = {u: stats.get(u, (0, 0.0))[0] for u in legal}
    total_v = sum(visits.values())
    if total_v > 0:
        base = {u: visits[u] / total_v for u in legal}
    else:
        priors = {u: stats.get(u, (0, 0.0))[1] for u in legal}
        tot = sum(priors.values()) or 1.0
        base = {u: priors[u] / tot for u in legal}
    floored = {u: base[u] + eps for u in legal}
    z = sum(floored.values())
    return {u: floored[u] / z for u in legal}


def gen(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    evaluator = ModelEvaluator.from_checkpoint(args.checkpoint, device=args.device)
    engine = open_lc0(args.lc0_path, str(args.weights), args.backend)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out = args.out_jsonl.open("w", encoding="utf-8")
    written = 0
    for g in range(args.games):
        board = book_board(rng)
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < args.max_plies:
            teacher = lc0_policy(engine, board, args.teacher_nodes, args.eps)
            out.write(json.dumps({"fen": board.fen(), "teacher": teacher}) + "\n")
            written += 1
            # on-policy: the student picks the move by sampling its own policy with
            # temperature, so recorded positions come from the student's distribution.
            priors = evaluator.evaluate(board).priors
            moves = list(priors)
            weights = [priors[m] ** (1.0 / args.temperature) for m in moves]
            board.push(rng.choices(moves, weights=weights)[0])
            plies += 1
        out.flush()
        print(f"game {g+1}/{args.games}: {written} positions labeled", flush=True)
    out.close()
    engine.quit()
    print(f"done: {written} on-policy positions -> {args.out_jsonl}")


class OPDDataset(Dataset):
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        board = chess.Board(s["fen"])
        enc = board_to_tensor(board)
        idxs, probs = [], []
        for uci, p in s["teacher"].items():
            idxs.append(move_to_index(chess.Move.from_uci(uci), board))
            probs.append(p)
        return {
            "piece_idx": enc["piece_idx"],
            "aux": enc["aux"],
            "legal_mask": legal_move_mask(board),
            "t_idx": torch.tensor(idxs, dtype=torch.long),
            "t_prob": torch.tensor(probs, dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict:
    b = len(batch)
    target = torch.zeros(b, ACTION_SIZE, dtype=torch.float32)
    for i, x in enumerate(batch):
        target[i, x["t_idx"]] = x["t_prob"]
    return {
        "piece_idx": torch.stack([x["piece_idx"] for x in batch]),
        "aux": torch.stack([x["aux"] for x in batch]),
        "legal_mask": torch.stack([x["legal_mask"] for x in batch]),
        "teacher": target,
    }


def train(args: argparse.Namespace) -> None:
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Kibitzer(payload["config"]).to(args.device)
    model.load_state_dict(payload["model"])
    # frozen base copy: anchor the student to it so the distill can't forget (D3 rule).
    base = Kibitzer(payload["config"]).to(args.device)
    base.load_state_dict(payload["model"])
    base.eval()
    base.requires_grad_(False)
    # freeze trunk/encoder/value; train only the policy head (+ final norm). the D53
    # full-model lr-2e-4 run nuked the base representations.
    trainable = list(model.policy_head.parameters())
    if args.freeze_trunk:
        for pm in model.parameters():
            pm.requires_grad_(False)
        for pm in model.policy_head.parameters():
            pm.requires_grad_(True)
        trainable = list(model.policy_head.parameters())
        for pm in model.norm.parameters():
            pm.requires_grad_(True)
        trainable += list(model.norm.parameters())
    else:
        trainable = list(model.parameters())
    samples = []
    for f in sorted(glob.glob(args.data)):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                samples.append(json.loads(line))
    tp = sum(p.numel() for p in trainable)
    print(f"on-policy distillation: {len(samples)} positions, reverse-KL to lc0 + base anchor "
          f"(w={args.anchor_weight}); trainable {tp:,} params (frozen_trunk={args.freeze_trunk})")
    loader = DataLoader(OPDDataset(samples), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    model.train()
    for epoch in range(args.epochs):
        trk = tan = 0.0
        n = 0
        for batch in loader:
            piece = batch["piece_idx"].unsqueeze(1).to(args.device)
            aux = batch["aux"].unsqueeze(1).to(args.device)
            legal = batch["legal_mask"].to(args.device)
            teacher = batch["teacher"].to(args.device)
            logits, _ = model(piece, aux)
            logits = logits[:, -1, :].masked_fill(~legal, -1e9)
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            # reverse kl(student || teacher) = sum p_s (log p_s - log p_t), over legal moves.
            logt = (teacher + 1e-9).log()
            rkl = (p * (logp - logt) * legal).sum(dim=-1).mean()
            # anchor: forward kl(base || student) keeps the student covering the base's
            # policy so it can't catastrophically forget while chasing the teacher.
            with torch.no_grad():
                blogits, _ = base(piece, aux)
                blogits = blogits[:, -1, :].masked_fill(~legal, -1e9)
                blogp = F.log_softmax(blogits, dim=-1)
                bp = blogp.exp()
            anchor = (bp * (blogp - logp) * legal).sum(dim=-1).mean()
            loss = rkl + args.anchor_weight * anchor
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            trk += float(rkl.item()); tan += float(anchor.item()); n += 1
        print(f"epoch {epoch+1}/{args.epochs}: reverse_kl {trk/max(1,n):.4f}  base_anchor_kl {tan/max(1,n):.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.config,
                "training_objective": "lc0_opd_reverse_kl_anchored"}, args.out)
    print(f"saved {args.out}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--lc0-path", required=True)
    g.add_argument("--weights", type=Path, required=True)
    g.add_argument("--backend", default="cuda")
    g.add_argument("--teacher-nodes", type=int, default=400)
    g.add_argument("--games", type=int, default=60)
    g.add_argument("--max-plies", type=int, default=160)
    # low temp keeps the on-policy positions near real play; temp 1.0 samples junk (D53).
    g.add_argument("--temperature", type=float, default=0.3)
    g.add_argument("--eps", type=float, default=1e-3)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out-jsonl", type=Path, required=True)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t = sub.add_parser("train")
    t.add_argument("--checkpoint", type=Path, required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--freeze-trunk", action="store_true", help="train only the policy head + final norm")
    t.add_argument("--anchor-weight", type=float, default=0.5, help="KL(base||student) anti-forgetting weight")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    {"gen": gen, "train": train}[args.mode](args)


if __name__ == "__main__":
    main()
