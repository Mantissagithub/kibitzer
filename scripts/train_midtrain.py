# tactical mid-training: continue the 100M shaw base on a mix of lichess puzzle
# positions (sharp tactics the game corpus underweights) and general game
# positions (an anti-forgetting anchor). policy-focused, low lr, short pass.
# gate: held-out top-1 must not regress from the base, and sf-1900 play (run
# separately) must beat the 0.900 baseline. see the tactical-midtrain plan.

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Iterator

import chess
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import random

from kibitzer.data import (
    PositionSample,
    collate_positions,
    encode_position_sample,
    iter_pgn_samples,
)
from kibitzer.model import Kibitzer

# lichess puzzle csv columns:
# PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
# convention: FEN is the position *before* the setup move; apply Moves[0] to reach
# the position the solver faces, then Moves[1],3,5,... are the solver's winning
# moves and Moves[2],4,... the forced opponent replies. we train only on the
# solver-move positions, value=+1 (side to move is winning the tactic).
def iter_puzzle_samples(
    csv_path: Path,
    *,
    rating_min: int,
    rating_max: int,
    worker_id: int = 0,
    num_workers: int = 1,
) -> Iterator[PositionSample]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row_idx, row in enumerate(reader):
            # shard by row across workers (single csv, so file-sharding won't work)
            if row_idx % num_workers != worker_id or len(row) < 3:
                continue
            fen, moves = row[1], row[2].split()
            try:
                rating = int(row[3])
            except (IndexError, ValueError):
                rating = 1500
            if rating < rating_min or rating > rating_max or len(moves) < 2:
                continue
            board = chess.Board(fen)
            try:
                board.push_uci(moves[0])  # setup move -> solver to move
                for i in range(1, len(moves), 2):
                    yield PositionSample(fen=board.fen(), move_uci=moves[i], value=1.0)
                    board.push_uci(moves[i])
                    if i + 1 < len(moves):
                        board.push_uci(moves[i + 1])  # forced opponent reply
            except (ValueError, AssertionError, IndexError):
                continue  # skip malformed puzzle lines


# interleave puzzle + game samples per-draw at mix_ratio, capping each worker at
# max_positions/num_workers. if one source drains, the other carries the rest.
class MixedDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        puzzle_csv: Path,
        game_paths: list[Path],
        *,
        max_positions: int,
        mix_ratio: float,
        rating_min: int,
        rating_max: int,
        seed: int,
    ) -> None:
        self.puzzle_csv = puzzle_csv
        self.game_paths = game_paths
        self.max_positions = max_positions
        self.mix_ratio = mix_ratio
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        wid = 0 if worker is None else worker.id
        nw = 1 if worker is None else worker.num_workers
        cap = math.ceil(self.max_positions / nw)
        rng = random.Random(self.seed + wid)

        pz = iter_puzzle_samples(
            self.puzzle_csv,
            rating_min=self.rating_min,
            rating_max=self.rating_max,
            worker_id=wid,
            num_workers=nw,
        )
        gm = iter_pgn_samples([self.game_paths[i] for i in range(wid, len(self.game_paths), nw)])

        n = 0
        while n < cap:
            src = pz if rng.random() < self.mix_ratio else gm
            try:
                sample = next(src)
            except StopIteration:
                other = gm if src is pz else pz
                try:
                    sample = next(other)
                except StopIteration:
                    break  # both drained
            yield encode_position_sample(sample)
            n += 1


def lr_lambda(step: int, total_steps: int, warmup_frac: float) -> float:
    warmup = max(1, int(total_steps * warmup_frac))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return 0.1 + 0.9 * (0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def eval_top1(model: Kibitzer, eval_path: Path, *, positions: int, batch_size: int, device: str) -> dict[str, float]:
    # deterministic held-out slice; same measurement as the scaling sweep so the
    # top-1 is directly comparable to the base's 49.45%.
    from kibitzer.data import StreamingPositionDataset

    ds = StreamingPositionDataset([eval_path], max_positions=positions, shuffle_buffer_size=1, seed=0)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_positions, num_workers=0)
    model.eval()
    top1 = 0
    vse = 0.0
    seen = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, value = model(batch["piece_idx"], batch["aux"])
        logits = logits.masked_fill(~batch["legal_mask"], -1e9)
        top1 += int((logits.argmax(dim=-1) == batch["policy_target"]).sum().item())
        vse += float(((value.squeeze(-1) - batch["value_target"]) ** 2).sum().item())
        seen += batch["policy_target"].numel()
    return {"top1": top1 / seen, "value_mse": vse / seen}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tactical mid-training on Lichess puzzles + games.")
    p.add_argument("--checkpoint", type=Path, default=Path("runs/scaling_shaw_data/checkpoints/S2_shaw_100M.pt"))
    p.add_argument("--puzzle-csv", type=Path, required=True)
    p.add_argument("--game-pgn", action="append", required=True, help="general game PGN; repeatable")
    p.add_argument("--eval-pgn", required=True, help="held-out PGN for the top-1 gate")
    p.add_argument("--out", type=Path, default=Path("runs/scaling_shaw_data/checkpoints/S2_shaw_100M_tactical.pt"))
    p.add_argument("--max-positions", type=int, default=10_000_000)
    p.add_argument("--mix-ratio", type=float, default=0.3, help="fraction of samples drawn from puzzles")
    p.add_argument("--rating-min", type=int, default=1200)
    p.add_argument("--rating-max", type=int, default=2400)
    p.add_argument("--lr", type=float, default=5e-5, help="low, this is a refine not a retrain")
    p.add_argument("--value-weight", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-positions", type=int, default=50_000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    game_paths = [Path(p) for p in args.game_pgn]
    for pth in [args.checkpoint, args.puzzle_csv, Path(args.eval_pgn), *game_paths]:
        if not pth.is_file():
            raise SystemExit(f"missing: {pth}")

    print("[1/5] TACTICAL REPAIR CONFIG", flush=True)
    print(f"  checkpoint:      {args.checkpoint}")
    print(f"  puzzle csv:      {args.puzzle_csv}")
    print(f"  game pgns:       {len(game_paths)}")
    print(f"  eval pgn:        {args.eval_pgn}")
    print(f"  max positions:   {args.max_positions:,}")
    print(f"  mix ratio:       {args.mix_ratio:g} puzzle / {1.0 - args.mix_ratio:g} game")
    print(f"  puzzle rating:   {args.rating_min}-{args.rating_max}")
    print(f"  lr/value weight: {args.lr:g} / {args.value_weight:g}")
    print(f"  batch/workers:   {args.batch_size} / {args.num_workers}")
    print(f"  output:          {args.out}")

    # rebuild the base from its saved config, then load its weights.
    print("\n[2/5] LOAD CHECKPOINT", flush=True)
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Kibitzer(payload["config"]).to(args.device)
    model.load_state_dict(payload["model"])
    print(f"  params:          {model.num_params():,}")
    print(f"  objective:       {payload.get('training_objective', 'unknown')}")
    if payload.get("eval_metrics"):
        print(f"  prior eval:      {payload['eval_metrics']}")

    # baseline top-1 before touching anything, so the gate is honest.
    print("\n[3/5] BASELINE HELD-OUT GATE", flush=True)
    base_metrics = eval_top1(
        model, Path(args.eval_pgn), positions=args.eval_positions, batch_size=args.batch_size, device=args.device
    )
    print(f"  positions:       {args.eval_positions:,}")
    print(f"  base top1:       {base_metrics['top1']:.4f}")
    print(f"  base value mse:  {base_metrics['value_mse']:.4f}")

    print("\n[4/5] TRAIN", flush=True)
    dataset = MixedDataset(
        args.puzzle_csv,
        game_paths,
        max_positions=args.max_positions,
        mix_ratio=args.mix_ratio,
        rating_min=args.rating_min,
        rating_max=args.rating_max,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_positions, num_workers=args.num_workers)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(args.max_positions / args.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: lr_lambda(s, total_steps, args.warmup_frac))

    model.train()
    start = time.perf_counter()
    from tqdm import tqdm

    progress = tqdm(loader, total=total_steps, desc="midtrain")
    policy_sum = loss_sum = 0.0
    for step, batch in enumerate(progress, start=1):
        batch = {k: v.to(args.device) for k, v in batch.items()}
        loss, metrics = model.loss(**batch, value_weight=args.value_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        loss_sum += float(metrics["loss"].item())
        policy_sum += float(metrics["policy_loss"].item())
        progress.set_postfix(loss=f"{metrics['loss'].item():.3f}", policy=f"{metrics['policy_loss'].item():.3f}")
        if step == 1 or step % 500 == 0:
            print(
                f"  step {step:,}/{total_steps:,}: "
                f"avg_loss={loss_sum/step:.4f} avg_policy={policy_sum/step:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
        )
        if args.save_every and step % args.save_every == 0:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model.config,
                    "step": step,
                    "training_objective": "tactical_midtrain_partial",
                },
                args.out,
            )
    elapsed = time.perf_counter() - start

    # gate: held-out top-1 must not regress (play eval is a separate command).
    print("\n[5/5] FINAL HELD-OUT GATE", flush=True)
    post = eval_top1(
        model, Path(args.eval_pgn), positions=args.eval_positions, batch_size=args.batch_size, device=args.device
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config,
            "training_objective": "tactical_midtrain",
            "base_eval_metrics": base_metrics,
            "eval_metrics": post,
            "training_config": {
                "checkpoint": str(args.checkpoint),
                "puzzle_csv": str(args.puzzle_csv),
                "game_pgn": [str(path) for path in game_paths],
                "eval_pgn": args.eval_pgn,
                "max_positions": args.max_positions,
                "mix_ratio": args.mix_ratio,
                "rating_min": args.rating_min,
                "rating_max": args.rating_max,
                "lr": args.lr,
                "value_weight": args.value_weight,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "eval_positions": args.eval_positions,
            },
        },
        args.out,
    )
    print(f"\ndone in {elapsed/60:.1f} min -> {args.out}")
    print(f"base  : top1={base_metrics['top1']:.4f} value_mse={base_metrics['value_mse']:.4f}")
    print(f"tactic: top1={post['top1']:.4f} value_mse={post['value_mse']:.4f}")
    print(f"delta : top1={post['top1']-base_metrics['top1']:+.4f} value_mse={post['value_mse']-base_metrics['value_mse']:+.4f}")
    if post["top1"] < base_metrics["top1"] - 0.005:
        print("WARNING: top-1 regressed >0.5pp -> forgetting; lower mix-ratio or lr, or shorten the pass.")
        print("GATE: REJECT_FOR_FORGETTING")
    else:
        print("GATE: PASS_HELDOUT_TOP1")
        print("NEXT: run Leela/Maia proxy eval before promoting.")


if __name__ == "__main__":
    main()
