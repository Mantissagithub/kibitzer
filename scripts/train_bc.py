from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kibitzer.data import StreamingPositionDataset, collate_positions
from kibitzer.hf_utils import parse_bool, push_checkpoint_to_hf, validate_hf_push
from kibitzer.model import Kibitzer, KibitzerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: human move cloning.")
    parser.add_argument("--pgn", action="append", required=True, help="PGN path; repeatable.")
    parser.add_argument("--out", required=True, help="Checkpoint output path.")
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="Freeze value_head and train only the shared backbone plus policy head.",
    )
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle-buffer-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf-push", type=parse_bool, default=False)
    parser.add_argument("--hf-repo")
    return parser.parse_args()


def configure_trainable_parameters(
    model: Kibitzer,
    *,
    policy_only: bool,
) -> list[torch.nn.Parameter]:
    if not policy_only:
        return list(model.parameters())

    for parameter in model.value_head.parameters():
        parameter.requires_grad_(False)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def main() -> None:
    args = parse_args()
    validate_hf_push(enabled=args.hf_push, repo_id=args.hf_repo)
    if args.max_games is not None:
        raise SystemExit("--max-games is not supported by the streaming policy trainer")
    paths = [Path(path) for path in args.pgn]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing PGN files: {missing}")

    dataset = StreamingPositionDataset(
        paths,
        max_positions=args.max_positions,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_positions,
        num_workers=args.num_workers,
    )
    model = Kibitzer(KibitzerConfig()).to(args.device)
    trainable_parameters = configure_trainable_parameters(
        model,
        policy_only=args.policy_only,
    )
    opt = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        dataset.epoch = epoch
        model.train()
        total = None
        if args.max_positions is not None:
            total = math.ceil(args.max_positions / args.batch_size)
        progress = tqdm(loader, total=total, desc=f"epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss, metrics = model.loss(
                **batch,
                value_weight=0.0 if args.policy_only else 0.25,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            opt.step()
            if args.policy_only:
                progress.set_postfix(policy=f"{metrics['policy_loss'].item():.3f}")
            else:
                progress.set_postfix(
                    loss=f"{metrics['loss'].item():.3f}",
                    policy=f"{metrics['policy_loss'].item():.3f}",
                    value=f"{metrics['value_loss'].item():.3f}",
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    training_objective = "policy_only" if args.policy_only else "policy_value"
    torch.save(
        {
            "model": model.state_dict(),
            "config": model.config,
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
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "max_games": args.max_games,
                "max_positions": args.max_positions,
                "num_workers": args.num_workers,
                "shuffle_buffer_size": args.shuffle_buffer_size,
            },
        )
        print(f"pushed checkpoint to https://huggingface.co/{args.hf_repo}")


if __name__ == "__main__":
    main()
