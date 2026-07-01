from __future__ import annotations

import argparse
import multiprocessing
import random
from dataclasses import replace
from pathlib import Path

import chess
import chess.engine
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kibitzer.data import (
    PositionSample,
    collate_positions,
    dense_policy_from_scores,
    iter_pgn_samples,
)
from kibitzer.encoding import board_to_tensor, legal_move_mask
from kibitzer.hf_utils import parse_bool, push_checkpoint_to_hf, validate_hf_push
from kibitzer.model import Kibitzer, KibitzerConfig
from kibitzer.stockfish import analyze_actions


_worker_engine: chess.engine.SimpleEngine | None = None
_worker_depth = 0
_worker_multipv = 1


def _start_stockfish_worker(path: str, depth: int, multipv: int) -> None:
    global _worker_engine, _worker_depth, _worker_multipv
    _worker_engine = chess.engine.SimpleEngine.popen_uci(path)
    _worker_depth = depth
    _worker_multipv = multipv


def _label_fen(fen: str) -> tuple[dict[int, float], float]:
    if _worker_engine is None:
        raise RuntimeError("Stockfish worker was not initialized")
    return analyze_actions(
        _worker_engine,
        chess.Board(fen),
        depth=_worker_depth,
        multipv=_worker_multipv,
    )


def label_samples(
    samples: list[PositionSample],
    *,
    stockfish_path: str,
    depth: int,
    multipv: int,
    workers: int,
) -> list[tuple[dict[int, float], float]]:
    if workers == 1:
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            return [
                analyze_actions(
                    engine,
                    chess.Board(sample.fen),
                    depth=depth,
                    multipv=multipv,
                )
                for sample in tqdm(samples, desc="label stockfish")
            ]

    context = multiprocessing.get_context("spawn")
    with context.Pool(
        workers,
        initializer=_start_stockfish_worker,
        initargs=(stockfish_path, depth, multipv),
    ) as pool:
        labels = pool.imap(_label_fen, (sample.fen for sample in samples), chunksize=4)
        return list(tqdm(labels, total=len(samples), desc="label stockfish"))


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
    parser.add_argument(
        "--value-only",
        action="store_true",
        help="Freeze the backbone and policy head; train only value_head. Requires --init.",
    )
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument(
        "--stockfish-workers",
        type=int,
        default=min(8, multiprocessing.cpu_count()),
    )
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-positions", type=int, default=1000)
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="Fraction of complete games reserved for Stockfish-labeled evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf-push", type=parse_bool, default=False)
    parser.add_argument("--hf-repo")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.value_only and not args.init:
        raise SystemExit("--value-only requires --init with a trained clean-rebuild checkpoint")
    if not 0.0 < args.eval_fraction < 1.0:
        raise SystemExit("--eval-fraction must be between 0 and 1")
    if getattr(args, "stockfish_workers", 1) < 1:
        raise SystemExit("--stockfish-workers must be at least 1")


def split_train_eval_by_game(
    samples: list[PositionSample],
    labels: list[tuple[dict[int, float], float]],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[
    list[PositionSample],
    list[tuple[dict[int, float], float]],
    list[PositionSample],
    list[tuple[dict[int, float], float]],
]:
    game_ids = sorted({sample.game_id for sample in samples})
    if len(game_ids) < 2:
        raise SystemExit("value evaluation requires positions from at least two games")

    rng = random.Random(seed)
    rng.shuffle(game_ids)
    eval_games = min(len(game_ids) - 1, max(1, round(len(game_ids) * eval_fraction)))
    eval_ids = set(game_ids[:eval_games])

    train_samples: list[PositionSample] = []
    train_labels: list[tuple[dict[int, float], float]] = []
    eval_samples: list[PositionSample] = []
    eval_labels: list[tuple[dict[int, float], float]] = []
    for sample, label in zip(samples, labels, strict=True):
        if sample.game_id in eval_ids:
            eval_samples.append(sample)
            eval_labels.append(label)
        else:
            train_samples.append(sample)
            train_labels.append(label)
    return train_samples, train_labels, eval_samples, eval_labels


def collect_balanced_samples(
    paths: list[Path],
    *,
    max_games: int | None,
    max_positions: int | None,
    seed: int,
) -> list[PositionSample]:
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    positions_per_file = None
    if max_positions is not None:
        positions_per_file = max(1, (max_positions + len(paths) - 1) // len(paths))
    games_per_file = None
    if max_games is not None:
        games_per_file = max(1, (max_games + len(paths) - 1) // len(paths))

    samples: list[PositionSample] = []
    for file_index, path in enumerate(paths):
        for sample in iter_pgn_samples(
            [path],
            max_games=games_per_file,
            max_positions=positions_per_file,
        ):
            samples.append(replace(sample, game_id=file_index * 1_000_000_000 + sample.game_id))
    if max_positions is not None:
        samples = samples[:max_positions]
    return samples


def configure_trainable_parameters(
    model: Kibitzer,
    *,
    value_only: bool,
) -> list[torch.nn.Parameter]:
    if not value_only:
        return list(model.parameters())

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.value_head.parameters():
        parameter.requires_grad_(True)
    return list(model.value_head.parameters())


def training_loss(
    model: Kibitzer,
    batch: dict[str, torch.Tensor],
    *,
    value_only: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not value_only:
        return model.loss(**batch)

    _, value = model(batch["piece_idx"], batch["aux"])
    value_loss = F.mse_loss(value.squeeze(-1), batch["value_target"])
    return value_loss, {
        "loss": value_loss.detach(),
        "value_loss": value_loss.detach(),
    }


def calculate_value_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    predictions = predictions.float().flatten()
    targets = targets.float().flatten()
    errors = predictions - targets
    mse = errors.square().mean()
    mae = errors.abs().mean()

    pred_centered = predictions - predictions.mean()
    target_centered = targets - targets.mean()
    pearson_denom = pred_centered.square().sum().sqrt() * target_centered.square().sum().sqrt()
    pearson = (pred_centered * target_centered).sum() / pearson_denom.clamp_min(1e-12)

    nonzero = targets != 0
    if nonzero.any():
        sign_accuracy = (predictions[nonzero].sign() == targets[nonzero].sign()).float().mean()
    else:
        sign_accuracy = torch.tensor(float("nan"))

    baseline_mse = target_centered.square().mean()
    r2 = 1.0 - mse / baseline_mse.clamp_min(1e-12)
    return {
        "mse": float(mse),
        "mae": float(mae),
        "pearson": float(pearson),
        "sign_accuracy": float(sign_accuracy),
        "r2": float(r2),
    }


@torch.no_grad()
def evaluate_value_head(
    model: Kibitzer,
    loader: DataLoader,
    *,
    device: str,
) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        _, value = model(batch["piece_idx"], batch["aux"])
        predictions.append(value.squeeze(-1).cpu())
        targets.append(batch["value_target"].cpu())
    return calculate_value_metrics(torch.cat(predictions), torch.cat(targets))


def main() -> None:
    args = parse_args()
    validate_args(args)
    validate_hf_push(enabled=args.hf_push, repo_id=args.hf_repo)
    paths = [Path(path) for path in args.pgn]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing PGN files: {missing}")
    samples = collect_balanced_samples(
        paths,
        max_games=args.max_games,
        max_positions=args.max_positions,
        seed=args.seed,
    )
    if not samples:
        raise SystemExit("no distillation samples found")

    labels = label_samples(
        samples,
        stockfish_path=args.stockfish_path,
        depth=args.depth,
        multipv=1 if args.value_only else args.multipv,
        workers=args.stockfish_workers,
    )

    train_samples, train_labels, eval_samples, eval_labels = split_train_eval_by_game(
        samples,
        labels,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    train_dataset = DistillDataset(train_samples, train_labels, args.temperature)
    eval_dataset = DistillDataset(eval_samples, eval_labels, args.temperature)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_positions,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_positions,
        num_workers=0,
    )
    print(
        f"game-level split: train_positions={len(train_dataset)} "
        f"eval_positions={len(eval_dataset)}"
    )
    model = Kibitzer(KibitzerConfig()).to(args.device)
    if args.init:
        payload = torch.load(args.init, map_location=args.device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state)
    trainable_parameters = configure_trainable_parameters(
        model,
        value_only=args.value_only,
    )
    opt = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=0.01)

    eval_metrics: dict[str, float] = {}
    for epoch in range(args.epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, metrics = training_loss(model, batch, value_only=args.value_only)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            opt.step()
            if args.value_only:
                progress.set_postfix(value=f"{metrics['value_loss'].item():.3f}")
            else:
                progress.set_postfix(
                    loss=f"{metrics['loss'].item():.3f}",
                    policy=f"{metrics['policy_loss'].item():.3f}",
                    value=f"{metrics['value_loss'].item():.3f}",
                )
        eval_metrics = evaluate_value_head(model, eval_loader, device=args.device)
        print(
            f"value eval epoch={epoch + 1} "
            f"mse={eval_metrics['mse']:.4f} "
            f"mae={eval_metrics['mae']:.4f} "
            f"pearson={eval_metrics['pearson']:.4f} "
            f"sign_acc={eval_metrics['sign_accuracy']:.4f} "
            f"r2={eval_metrics['r2']:.4f}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    training_objective = "value_only" if args.value_only else "policy_value"
    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config,
            "eval_metrics": eval_metrics,
            "training_objective": training_objective,
        },
        out,
    )
    if args.hf_push:
        push_checkpoint_to_hf(
            out,
            repo_id=args.hf_repo,
            training_objective=training_objective,
            metadata={
                "depth": args.depth,
                "epochs": args.epochs,
                "eval_fraction": args.eval_fraction,
                "eval_metrics": eval_metrics,
                "learning_rate": args.lr,
                "max_games": args.max_games,
                "max_positions": args.max_positions,
                "multipv": 1 if args.value_only else args.multipv,
                "temperature": args.temperature,
            },
        )
        print(f"pushed checkpoint to https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
