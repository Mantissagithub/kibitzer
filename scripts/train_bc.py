from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kibitzer.data import PositionDataset, collate_positions, iter_pgn_samples
from kibitzer.model import Kibitzer, KibitzerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: human move cloning.")
    parser.add_argument("--pgn", action="append", required=True, help="PGN path; repeatable.")
    parser.add_argument("--out", required=True, help="Checkpoint output path.")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = list(
        iter_pgn_samples(
            [Path(p) for p in args.pgn],
            max_games=args.max_games,
            max_positions=args.max_positions,
        )
    )
    if not samples:
        raise SystemExit("no training samples found")

    dataset = PositionDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_positions,
        num_workers=0,
    )
    model = Kibitzer(KibitzerConfig()).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, metrics = model.loss(**batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            progress.set_postfix(
                loss=f"{metrics['loss'].item():.3f}",
                policy=f"{metrics['policy_loss'].item():.3f}",
                value=f"{metrics['value_loss'].item():.3f}",
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": model.config}, out)


if __name__ == "__main__":
    main()
