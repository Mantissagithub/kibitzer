from __future__ import annotations

import argparse
from pathlib import Path

import chess
import chess.engine
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kibitzer.data import (
    PositionSample,
    collate_positions,
    dense_policy_from_scores,
    iter_pgn_samples,
)
from kibitzer.encoding import board_to_tensor, legal_move_mask
from kibitzer.model import Kibitzer, KibitzerConfig
from kibitzer.stockfish import analyze_actions


class DistillDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        samples: list[PositionSample],
        labels: list[tuple[dict[int, float], float]],
        temperature: float,
    ) -> None:
        self.samples = samples
        self.labels = labels
        self.temperature = temperature

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        board = chess.Board(self.samples[idx].fen)
        action_scores, value = self.labels[idx]
        encoded = board_to_tensor(board)
        return {
            "piece_idx": encoded["piece_idx"],
            "aux": encoded["aux"],
            "policy_target": dense_policy_from_scores(action_scores, self.temperature),
            "value_target": torch.tensor(value, dtype=torch.float32),
            "legal_mask": legal_move_mask(board),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: Stockfish action-value distillation.")
    parser.add_argument("--pgn", action="append", required=True, help="PGN path; repeatable.")
    parser.add_argument("--out", required=True, help="Checkpoint output path.")
    parser.add_argument("--init")
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-positions", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
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
        raise SystemExit("no distillation samples found")

    labels: list[tuple[dict[int, float], float]] = []
    with chess.engine.SimpleEngine.popen_uci(args.stockfish_path) as engine:
        for sample in tqdm(samples, desc="label stockfish"):
            board = chess.Board(sample.fen)
            labels.append(
                analyze_actions(
                    engine,
                    board,
                    depth=args.depth,
                    multipv=args.multipv,
                )
            )

    dataset = DistillDataset(samples, labels, args.temperature)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_positions,
        num_workers=0,
    )
    model = Kibitzer(KibitzerConfig()).to(args.device)
    if args.init:
        payload = torch.load(args.init, map_location=args.device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state)
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
