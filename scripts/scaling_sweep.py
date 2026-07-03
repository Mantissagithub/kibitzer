"""Scaling-law sweep trainer: train a ladder of attention-first model sizes and
log a policy-loss-vs-params scaling curve.

See docs/scaling_study/design.md (fixed choices, ladder, LR transfer) and
DECISIONS.md D36 for the full rationale. This drives Kibitzer training directly
(mirroring scripts/train_bc.py's streaming loop) rather than shelling out, so
that per-rung LR schedules and eval can be inserted between rungs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kibitzer.data import StreamingPositionDataset, collate_positions
from kibitzer.model import Kibitzer, KibitzerConfig

# Attention-first ladder (design.md): attention_every=1 forced below, so every
# trunk block is CausalAttentionBlock. encoder_heads stays 8 (divides every
# d_model here); n_heads is chosen to divide its own d_model.
SIZES: dict[str, dict[str, int]] = {
    "S0": {"d_model": 128, "trunk_layers": 6, "n_heads": 4},
    "S1": {"d_model": 192, "trunk_layers": 8, "n_heads": 6},
    "S2": {"d_model": 256, "trunk_layers": 10, "n_heads": 8},
    "S3": {"d_model": 320, "trunk_layers": 10, "n_heads": 8},
    "S4": {"d_model": 448, "trunk_layers": 12, "n_heads": 8},
}


def build_config(tag: str) -> KibitzerConfig:
    spec = SIZES[tag]
    return KibitzerConfig(
        d_model=spec["d_model"],
        trunk_layers=spec["trunk_layers"],
        n_heads=spec["n_heads"],
        encoder_heads=8,
        attention_every=1,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scaling-law sweep over attention-first Kibitzer sizes.")
    parser.add_argument("--sizes", default="S0,S1,S2,S3", help="Comma-separated ladder tags to run.")
    parser.add_argument("--train-pgn", action="append", required=True, help="PGN path; repeatable.")
    parser.add_argument("--eval-pgn", required=True, help="Held-out PGN, disjoint from --train-pgn.")
    parser.add_argument("--max-positions", type=int, default=5_000_000, help="Positions trained per rung.")
    parser.add_argument("--eval-positions", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--base-lr", type=float, default=3e-4, help="LR for the smallest selected rung.")
    parser.add_argument("--lr-transfer", choices=["mup", "constant"], default="mup")
    parser.add_argument("--warmup-frac", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="reports/scaling_law/results.json")
    parser.add_argument("--ckpt-dir", default="runs/scaling")
    return parser.parse_args()


def lr_for_rung(tag: str, sizes: list[str], *, base_lr: float, lr_transfer: str) -> float:
    if lr_transfer == "constant":
        return base_lr
    smallest_d_model = min(SIZES[t]["d_model"] for t in sizes)
    return base_lr * smallest_d_model / SIZES[tag]["d_model"]


def lr_lambda(step: int, total_steps: int, warmup_frac: float) -> float:
    """Linear warmup over warmup_frac of total_steps, then cosine decay to 10% of peak."""
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return 0.1 + 0.9 * cosine


def train_rung(
    tag: str,
    *,
    train_paths: list[Path],
    max_positions: int,
    batch_size: int,
    lr: float,
    warmup_frac: float,
    num_workers: int,
    device: str,
) -> tuple[Kibitzer, int, float]:
    """Trains one rung for a single pass over max_positions. Returns (model, params, wall_clock_s)."""
    cfg = build_config(tag)
    model = Kibitzer(cfg).to(device)
    params = model.num_params()

    dataset = StreamingPositionDataset(
        train_paths,
        max_positions=max_positions,
        shuffle_buffer_size=8192,
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_positions, num_workers=num_workers)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = math.ceil(max_positions / batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda step: lr_lambda(step, total_steps, warmup_frac)
    )

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    start = time.perf_counter()
    progress = tqdm(loader, total=total_steps, desc=f"train {tag}")
    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, metrics = model.loss(**batch, value_weight=0.25)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        progress.set_postfix(
            loss=f"{metrics['loss'].item():.3f}",
            policy=f"{metrics['policy_loss'].item():.3f}",
            value=f"{metrics['value_loss'].item():.3f}",
        )
    wall_clock_s = time.perf_counter() - start
    return model, params, wall_clock_s


@torch.no_grad()
def evaluate_rung(
    model: Kibitzer,
    *,
    eval_path: Path,
    eval_positions: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    """Fixed, non-shuffled slice: shuffle_buffer_size=1 makes ordering deterministic."""
    dataset = StreamingPositionDataset(
        [eval_path],
        max_positions=eval_positions,
        shuffle_buffer_size=1,
        seed=0,
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_positions, num_workers=0)

    model.eval()
    policy_ce_sum = 0.0
    value_se_sum = 0.0
    top1_matches = 0
    positions = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits, value = model(batch["piece_idx"], batch["aux"])
        logits = logits.masked_fill(~batch["legal_mask"], -1e9)
        policy_target = batch["policy_target"]

        log_probs = torch.log_softmax(logits, dim=-1)
        target_logp = log_probs.gather(dim=-1, index=policy_target.unsqueeze(-1)).squeeze(-1)
        policy_ce_sum += float(-target_logp.sum().item())

        predicted_moves = logits.argmax(dim=-1)
        top1_matches += int((predicted_moves == policy_target).sum().item())

        value_error = value.squeeze(-1) - batch["value_target"]
        value_se_sum += float((value_error**2).sum().item())

        positions += policy_target.numel()

    return {
        "eval_policy_ce": policy_ce_sum / positions,
        "eval_value_mse": value_se_sum / positions,
        "eval_top1": top1_matches / positions,
    }


def load_results(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"runs": []}


def upsert_run(results: dict[str, Any], record: dict[str, Any]) -> None:
    runs = results["runs"]
    for i, run in enumerate(runs):
        if run["tag"] == record["tag"]:
            runs[i] = record
            return
    runs.append(record)


def main() -> None:
    args = parse_args()
    sizes = args.sizes.split(",")
    unknown = [tag for tag in sizes if tag not in SIZES]
    if unknown:
        raise SystemExit(f"unknown size tags: {unknown}; choose from {list(SIZES)}")

    train_paths = [Path(p) for p in args.train_pgn]
    missing = [str(p) for p in train_paths if not p.is_file()]
    if missing:
        raise SystemExit(f"missing train PGN files: {missing}")
    eval_path = Path(args.eval_pgn)
    if not eval_path.is_file():
        raise SystemExit(f"missing eval PGN file: {eval_path}")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for tag in sizes:
        lr = lr_for_rung(tag, sizes, base_lr=args.base_lr, lr_transfer=args.lr_transfer)
        print(f"=== {tag}: {SIZES[tag]} lr={lr:.2e} ===")

        model, params, wall_clock_s = train_rung(
            tag,
            train_paths=train_paths,
            max_positions=args.max_positions,
            batch_size=args.batch_size,
            lr=lr,
            warmup_frac=args.warmup_frac,
            num_workers=args.num_workers,
            device=args.device,
        )

        ckpt_path = ckpt_dir / f"{tag}.pt"
        torch.save(
            {"model": model.state_dict(), "config": model.config, "training_objective": "policy_value"},
            ckpt_path,
        )

        eval_metrics = evaluate_rung(
            model,
            eval_path=eval_path,
            eval_positions=args.eval_positions,
            batch_size=args.batch_size,
            device=args.device,
        )

        peak_vram_gb = 0.0
        if args.device.startswith("cuda"):
            peak_vram_gb = torch.cuda.max_memory_allocated(args.device) / 1e9

        record = {
            "tag": tag,
            "d_model": SIZES[tag]["d_model"],
            "trunk_layers": SIZES[tag]["trunk_layers"],
            "n_heads": SIZES[tag]["n_heads"],
            "attention_every": 1,
            "params": params,
            "train_positions": args.max_positions,
            "lr": lr,
            **eval_metrics,
            "wall_clock_s": wall_clock_s,
            "peak_vram_gb": peak_vram_gb,
        }
        results = load_results(out_path)
        upsert_run(results, record)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

        print(
            f"{tag}: params={params:,} eval_policy_ce={eval_metrics['eval_policy_ce']:.4f} "
            f"eval_top1={eval_metrics['eval_top1']:.4f} wall_clock_s={wall_clock_s:.1f} "
            f"peak_vram_gb={peak_vram_gb:.2f}"
        )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
