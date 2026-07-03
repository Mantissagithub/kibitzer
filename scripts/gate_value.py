from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kibitzer.data import collate_positions
from kibitzer.model import Kibitzer, KibitzerConfig
from scripts.distill_stockfish import (
    DistillDataset,
    evaluate_value_head,
    load_or_create_labels,
    split_train_eval_by_game,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare value heads on the held-out Stockfish split. The first "
            "--checkpoint is the baseline; the rest are reported as deltas from it."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named checkpoint, e.g. baseline=runs/value/value_final.pt; repeatable.",
    )
    parser.add_argument(
        "--label-cache",
        type=Path,
        default=Path("data/stockfish/joint_d14_mpv8_250000.pt"),
        help="Cached Stockfish labels; must already exist (no relabeling here).",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_checkpoints(raw: list[str]) -> list[tuple[str, Path]]:
    checkpoints: list[tuple[str, Path]] = []
    for entry in raw:
        if "=" not in entry:
            raise SystemExit(f"--checkpoint must be NAME=PATH, got: {entry!r}")
        name, path = entry.split("=", 1)
        path = Path(path)
        if not path.is_file():
            raise SystemExit(f"checkpoint does not exist: {path}")
        checkpoints.append((name, path))
    return checkpoints


def load_model(path: Path, device: str) -> Kibitzer:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = Kibitzer(KibitzerConfig()).to(device)
    model.load_state_dict(state)
    return model


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    if not args.label_cache.is_file():
        raise SystemExit(
            f"label cache does not exist: {args.label_cache}; "
            "run the Stockfish value stage first to create it"
        )

    payload = torch.load(args.label_cache, map_location="cpu", weights_only=False)
    samples, labels = payload["samples"], payload["labels"]
    _, _, eval_samples, eval_labels = split_train_eval_by_game(
        samples,
        labels,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    eval_loader = DataLoader(
        DistillDataset(eval_samples, eval_labels, args.temperature),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_positions,
        num_workers=0,
    )
    print(f"held-out eval positions: {len(eval_samples):,} (Stockfish depth-14 targets)")
    print(
        "note: Pearson and sign accuracy are scale-invariant, so they compare cleanly\n"
        "      across value heads trained on different target scales (game-result vs cp).\n"
        "      MSE/MAE are NOT comparable if a model was trained on a different target."
    )

    baseline_metrics: dict[str, float] | None = None
    for name, path in checkpoints:
        model = load_model(path, args.device)
        metrics = evaluate_value_head(model, eval_loader, device=args.device)
        print(f"\n{name}  ({path})")
        print(f"  Pearson:        {metrics['pearson']:.4f}")
        print(f"  R2:             {metrics['r2']:.4f}")
        print(f"  sign accuracy:  {100 * metrics['sign_accuracy']:.2f}%")
        print(
            f"  decisive:       sign={100 * metrics['decisive_sign_accuracy']:.2f}% "
            f"mae={metrics['decisive_mae']:.4f} n={int(metrics['decisive_count'])}"
        )
        print(
            f"  won:            sign={100 * metrics['won_sign_accuracy']:.2f}% "
            f"mae={metrics['won_mae']:.4f} n={int(metrics['won_count'])}"
        )
        print(f"  MSE / MAE:      {metrics['mse']:.4f} / {metrics['mae']:.4f}")
        if baseline_metrics is None:
            baseline_metrics = metrics
        else:
            print("  vs baseline:")
            print(f"    Pearson:      {metrics['pearson'] - baseline_metrics['pearson']:+.4f}")
            print(
                f"    decisive sign:{100 * (metrics['decisive_sign_accuracy'] - baseline_metrics['decisive_sign_accuracy']):+.2f}pp"
            )
            print(
                f"    won sign:     {100 * (metrics['won_sign_accuracy'] - baseline_metrics['won_sign_accuracy']):+.2f}pp"
            )


if __name__ == "__main__":
    main()
