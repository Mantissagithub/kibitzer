from __future__ import annotations

import argparse
import multiprocessing
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    parser.add_argument("--label-cache", type=Path)
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
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument(
        "--unfreeze-last-trunk-blocks",
        type=int,
        help="Freeze the encoder and early trunk; train both heads, norm, and the last N blocks.",
    )
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
    if getattr(args, "epochs", 1) < 1:
        raise SystemExit("--epochs must be at least 1")
    if getattr(args, "value_weight", 0.25) < 0.0:
        raise SystemExit("--value-weight must be non-negative")
    unfreeze_last_trunk_blocks = getattr(args, "unfreeze_last_trunk_blocks", None)
    if args.value_only and unfreeze_last_trunk_blocks is not None:
        raise SystemExit("--unfreeze-last-trunk-blocks cannot be used with --value-only")
    if unfreeze_last_trunk_blocks is not None and unfreeze_last_trunk_blocks < 1:
        raise SystemExit("--unfreeze-last-trunk-blocks must be at least 1")


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


def label_cache_signature(
    paths: list[Path],
    *,
    depth: int,
    multipv: int,
    max_games: int | None,
    max_positions: int | None,
    seed: int,
) -> dict[str, Any]:
    return {
        "depth": depth,
        "multipv": multipv,
        "max_games": max_games,
        "max_positions": max_positions,
        "seed": seed,
        "pgns": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        ],
    }


def load_or_create_labels(
    paths: list[Path],
    *,
    cache_path: Path | None,
    stockfish_path: str,
    workers: int,
    depth: int,
    multipv: int,
    max_games: int | None,
    max_positions: int | None,
    seed: int,
) -> tuple[list[PositionSample], list[tuple[dict[int, float], float]], bool]:
    signature = label_cache_signature(
        paths,
        depth=depth,
        multipv=multipv,
        max_games=max_games,
        max_positions=max_positions,
        seed=seed,
    )
    if cache_path is not None and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("signature") != signature:
            raise SystemExit(
                f"label cache settings do not match: {cache_path}; "
                "use a different --label-cache path"
            )
        return payload["samples"], payload["labels"], True

    samples = collect_balanced_samples(
        paths,
        max_games=max_games,
        max_positions=max_positions,
        seed=seed,
    )
    if not samples:
        raise SystemExit("no distillation samples found")
    labels = label_samples(
        samples,
        stockfish_path=stockfish_path,
        depth=depth,
        multipv=multipv,
        workers=workers,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".part")
        torch.save(
            {"signature": signature, "samples": samples, "labels": labels},
            temporary_path,
        )
        temporary_path.replace(cache_path)
    return samples, labels, False


def configure_trainable_parameters(
    model: Kibitzer,
    *,
    value_only: bool,
    unfreeze_last_trunk_blocks: int | None = None,
) -> list[torch.nn.Parameter]:
    if not value_only and unfreeze_last_trunk_blocks is None:
        return list(model.parameters())

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules: list[torch.nn.Module] = [model.value_head]
    if not value_only:
        if unfreeze_last_trunk_blocks is None:
            raise ValueError("joint partial fine-tuning requires a trunk block count")
        if unfreeze_last_trunk_blocks > len(model.trunk):
            raise SystemExit(
                f"cannot unfreeze {unfreeze_last_trunk_blocks} trunk blocks; "
                f"model has {len(model.trunk)}"
            )
        modules.extend([model.policy_head, model.norm])
        modules.extend(model.trunk[-unfreeze_last_trunk_blocks:])
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def training_loss(
    model: Kibitzer,
    batch: dict[str, torch.Tensor],
    *,
    value_only: bool,
    value_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not value_only:
        return model.loss(**batch, value_weight=value_weight)

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


@torch.no_grad()
def evaluate_joint_model(
    model: Kibitzer,
    loader: DataLoader,
    *,
    device: str,
) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []
    policy_cross_entropy = 0.0
    policy_top1_matches = 0
    policy_teacher_coverage = 0
    positions = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits, value = model(batch["piece_idx"], batch["aux"])
        logits = logits.masked_fill(~batch["legal_mask"], -1e9)
        policy_targets = batch["policy_target"]
        log_probabilities = F.log_softmax(logits, dim=-1)
        policy_cross_entropy += float(
            -(policy_targets * log_probabilities).sum(dim=-1).sum().item()
        )
        predicted_moves = logits.argmax(dim=-1)
        teacher_moves = policy_targets.argmax(dim=-1)
        policy_top1_matches += int((predicted_moves == teacher_moves).sum().item())
        predicted_teacher_mass = policy_targets.gather(
            dim=-1,
            index=predicted_moves.unsqueeze(-1),
        ).squeeze(-1)
        policy_teacher_coverage += int((predicted_teacher_mass > 0).sum().item())
        positions += policy_targets.shape[0] * policy_targets.shape[1]
        predictions.append(value.squeeze(-1).cpu())
        targets.append(batch["value_target"].cpu())

    metrics = calculate_value_metrics(torch.cat(predictions), torch.cat(targets))
    metrics.update(
        {
            "policy_cross_entropy": policy_cross_entropy / positions,
            "policy_top1_accuracy": policy_top1_matches / positions,
            "policy_teacher_coverage": policy_teacher_coverage / positions,
        }
    )
    return metrics


def format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {remaining_seconds}s"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def save_checkpoint(
    path: Path,
    *,
    model: Kibitzer,
    eval_metrics: dict[str, float],
    training_objective: str,
    best_epoch: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config,
            "eval_metrics": eval_metrics,
            "training_objective": training_objective,
            "best_epoch": best_epoch,
        },
        path,
    )


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    validate_args(args)
    validate_hf_push(enabled=args.hf_push, repo_id=args.hf_repo)
    paths = [Path(path) for path in args.pgn]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing PGN files: {missing}")

    effective_multipv = 1 if args.value_only else args.multipv
    print("\n[1/5] DISTILLATION CONFIGURATION", flush=True)
    print(f"  objective:       {'value head only' if args.value_only else 'joint policy + value'}")
    print(f"  source files:    {len(paths)}")
    print(f"  positions:       {args.max_positions or 'all available'}")
    print(f"  Stockfish:       depth {args.depth}, MultiPV {effective_multipv}")
    print(f"  label workers:   {args.stockfish_workers}")
    print(f"  epochs:          {args.epochs}")
    print(f"  device:          {args.device}")

    print("\n[2/5] STOCKFISH TEACHER LABELS", flush=True)
    label_started = time.perf_counter()
    samples, labels, cache_hit = load_or_create_labels(
        paths,
        cache_path=args.label_cache,
        stockfish_path=args.stockfish_path,
        workers=args.stockfish_workers,
        depth=args.depth,
        multipv=effective_multipv,
        max_games=args.max_games,
        max_positions=args.max_positions,
        seed=args.seed,
    )
    label_duration = time.perf_counter() - label_started
    if cache_hit:
        print(f"  cache hit:       {args.label_cache}")
        print(f"  labels loaded:   {len(labels):,} in {format_duration(label_duration)}")
    else:
        print(f"  labels created:  {len(labels):,} in {format_duration(label_duration)}")
        if args.label_cache is not None:
            print(f"  cache saved:     {args.label_cache}")

    print("\n[3/5] GAME-DISJOINT DATA SPLIT", flush=True)
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
    print(f"  training:        {len(train_dataset):,} positions")
    print(f"  held-out eval:   {len(eval_dataset):,} positions")
    print("  leakage guard:   complete games stay in exactly one split")

    print("\n[4/5] MODEL INITIALIZATION", flush=True)
    model = Kibitzer(KibitzerConfig()).to(args.device)
    if args.init:
        payload = torch.load(args.init, map_location=args.device, weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state)
        print(f"  initialized from: {args.init}")
    else:
        print("  initialized from: random weights")
    trainable_parameters = configure_trainable_parameters(
        model,
        value_only=args.value_only,
        unfreeze_last_trunk_blocks=args.unfreeze_last_trunk_blocks,
    )
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    print(f"  total params:     {model.num_params():,}")
    print(
        f"  trainable params: {trainable_count:,} "
        f"({100 * trainable_count / model.num_params():.1f}%)"
    )
    if args.unfreeze_last_trunk_blocks is not None:
        print(
            "  trainable scope:  policy head, value head, final norm, "
            f"last {args.unfreeze_last_trunk_blocks} trunk blocks"
        )
    opt = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=0.01)

    print("\n[5/5] TRAINING AND HELD-OUT EVALUATION", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    training_objective = "value_only" if args.value_only else "joint_policy_value"
    eval_metrics: dict[str, float] = {}
    best_eval_score = float("inf")
    best_epoch = 0
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        progress = tqdm(
            train_loader,
            desc=f"train epoch {epoch + 1}/{args.epochs}",
            unit="batch",
        )
        metric_sums: dict[str, float] = {}
        batches = 0
        for batch in progress:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, metrics = training_loss(
                model,
                batch,
                value_only=args.value_only,
                value_weight=args.value_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            opt.step()
            batches += 1
            for name, value in metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value.item())
            if args.value_only:
                progress.set_postfix(
                    avg_value=f"{metric_sums['value_loss'] / batches:.4f}"
                )
            else:
                progress.set_postfix(
                    avg_policy=f"{metric_sums['policy_loss'] / batches:.4f}",
                    avg_value=f"{metric_sums['value_loss'] / batches:.4f}",
                )
        if args.value_only:
            eval_metrics = evaluate_value_head(model, eval_loader, device=args.device)
            eval_score = eval_metrics["mse"]
        else:
            eval_metrics = evaluate_joint_model(model, eval_loader, device=args.device)
            eval_score = (
                eval_metrics["policy_cross_entropy"]
                + args.value_weight * eval_metrics["mse"]
            )

        print(f"\n  epoch {epoch + 1}/{args.epochs} summary")
        print(f"    duration:              {format_duration(time.perf_counter() - epoch_started)}")
        if not args.value_only:
            print(f"    train policy loss:     {metric_sums['policy_loss'] / batches:.4f}")
        print(f"    train value MSE:       {metric_sums['value_loss'] / batches:.4f}")
        if not args.value_only:
            print(f"    eval policy CE:        {eval_metrics['policy_cross_entropy']:.4f} (lower is better)")
            print(f"    eval teacher top-1:    {100 * eval_metrics['policy_top1_accuracy']:.2f}%")
            print(f"    eval teacher coverage: {100 * eval_metrics['policy_teacher_coverage']:.2f}%")
        print(f"    eval value MSE:        {eval_metrics['mse']:.4f} (lower is better)")
        print(f"    eval value MAE:        {eval_metrics['mae']:.4f} (lower is better)")
        print(f"    eval value Pearson:    {eval_metrics['pearson']:.4f} (higher is better)")
        print(f"    eval value sign:       {100 * eval_metrics['sign_accuracy']:.2f}%")
        print(f"    eval value R2:         {eval_metrics['r2']:.4f} (higher is better)")
        if eval_score < best_eval_score:
            best_eval_score = eval_score
            best_epoch = epoch + 1
            save_checkpoint(
                out,
                model=model,
                eval_metrics=eval_metrics,
                training_objective=training_objective,
                best_epoch=best_epoch,
            )
            print(f"    checkpoint:            saved new best -> {out}")
        else:
            print(f"    checkpoint:            kept best from epoch {best_epoch}")

    print("\nTRAINING COMPLETE", flush=True)
    print(f"  best epoch:      {best_epoch}/{args.epochs}")
    print(f"  checkpoint:      {out}")
    print(f"  total duration:  {format_duration(time.perf_counter() - run_started)}")
    best_payload = torch.load(out, map_location="cpu", weights_only=False)
    best_eval_metrics = best_payload["eval_metrics"]
    if args.hf_push:
        print(f"\nUPLOADING BEST CHECKPOINT TO HUGGING FACE: {args.hf_repo}", flush=True)
        push_checkpoint_to_hf(
            out,
            repo_id=args.hf_repo,
            training_objective=training_objective,
            metadata={
                "depth": args.depth,
                "epochs": args.epochs,
                "eval_fraction": args.eval_fraction,
                "best_epoch": best_epoch,
                "eval_metrics": best_eval_metrics,
                "learning_rate": args.lr,
                "max_games": args.max_games,
                "max_positions": args.max_positions,
                "multipv": effective_multipv,
                "temperature": args.temperature,
                "unfreeze_last_trunk_blocks": args.unfreeze_last_trunk_blocks,
                "value_weight": args.value_weight,
            },
        )
        print(f"  upload complete: https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
