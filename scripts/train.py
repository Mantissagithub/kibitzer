"""supervised pretraining loop for kibitzer.

streams lichess elite pgns through ``lichessgamedataset``, trains the model
with policy + value losses, runs periodic cutechess gauntlets vs stockfish for
a real elo signal, saves checkpoints (the uci adapter loads them at eval time).

use ``--config <path>.yaml`` to pin defaults; any field is also a cli flag
(``--peak-lr``, ``--batch-size``, …) that overrides yaml and the dataclass
default.

``--dry-run`` runs 5 steps and exits — for verifying the loop works without
committing to a long run. ``--resume path`` continues from a saved checkpoint.
``--no-tui`` swaps the rich layout for plain key=value log lines (auto when
stdout isn't a tty).
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterator

import chess
import torch
import yaml
from rich.live import Live
from torch.utils.data import DataLoader
from tqdm import tqdm

from kibitzer import tui
from kibitzer.data import LichessGameDataset, collate_games
from kibitzer.encoding import index_to_move
from kibitzer.eval import evaluate_checkpoint
from kibitzer.hf_utils import HFPushConfig, prepare_hf_push, push_checkpoint_to_hf
from kibitzer.loss import combined_loss
from kibitzer.model import Kibitzer, KibitzerConfig
from kibitzer.training_utils import (
    EMA,
    count_params,
    get_lr,
    load_checkpoint,
    save_checkpoint,
)


@dataclass
class TrainConfig:
    data_dir: str = "data/raw"
    val_data_dir: str | None = None  # reserved; not consumed in v1
    max_seq_len: int = 256
    batch_size: int = 16
    grad_accum_steps: int = 2
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    total_steps: int = 200_000
    value_loss_weight: float = 1.0
    dtype: str = "bfloat16"
    log_every: int = 20
    eval_every: int = 2000
    eval_n_games: int = 10
    eval_opponent: str = "stockfish-elo-1320"
    eval_time_per_move_ms: int = 200
    checkpoint_every: int = 5000
    checkpoint_dir: str = "runs"
    max_grad_norm: float = 1.0
    num_workers: int = 4
    shuffle_buffer_size: int = 1024
    seed: int = 42
    run_name: str | None = None
    wandb: bool = False
    auto_download_data: bool = True
    data_download_year: int = 2024
    data_download_sample: bool = False
    hf_push: bool = True
    hf_private: bool = False
    hf_repo_prefix: str = "kibitzer-sft"


def _add_config_args(p: argparse.ArgumentParser) -> None:
    for f in fields(TrainConfig):
        flag = f"--{f.name.replace('_', '-')}"
        if f.type is bool or f.type == "bool":
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        elif f.type is int or f.type == "int":
            p.add_argument(flag, type=int, default=None)
        elif f.type is float or f.type == "float":
            p.add_argument(flag, type=float, default=None)
        else:
            p.add_argument(flag, type=str, default=None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default=None, help="YAML config path")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--dry-run", action="store_true",
                   help="run 5 steps then exit")
    p.add_argument("--no-tui", action="store_true",
                   help="disable rich TUI; auto-disabled when stdout isn't a TTY")
    _add_config_args(p)
    return p.parse_args()


def _load_config(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig()
    if args.config:
        with open(args.config) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    for f in fields(TrainConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _move_to(batch: dict, device: str) -> dict:
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def _forever(loader: DataLoader) -> Iterator[dict]:
    while True:
        for batch in loader:
            yield batch


def _list_pgns(data_dir: str) -> list[str]:
    p = Path(data_dir)
    if not p.exists():
        return []
    return sorted(str(x) for x in p.glob("*.pgn"))


def _download_missing_data(cfg: TrainConfig, use_tui: bool) -> list[str]:
    """download lichess elite pgns if the data dir is empty."""
    pgn_paths = _list_pgns(cfg.data_dir)
    if pgn_paths or not cfg.auto_download_data:
        return pgn_paths

    try:
        from scripts import lichess_download
    except ImportError as e:
        tui.console.print(f"[error]could not import downloader: {e}[/]")
        return []

    output_dir = Path(cfg.data_dir)
    msg = (
        f"no PGNs under {cfg.data_dir}; downloading "
        f"Lichess Elite {cfg.data_download_year}"
    )
    if cfg.data_download_sample:
        msg += " sample month"
    tui.console.print(f"[muted]{msg}[/]")

    available = lichess_download.discover_year(cfg.data_download_year)
    if not available:
        tui.console.print(
            f"[error]no downloadable months found for {cfg.data_download_year}[/]"
        )
        return []

    chosen = sorted(available.keys())
    if cfg.data_download_sample:
        chosen = chosen[-1:]

    output_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    for month in chosen:
        existing = lichess_download.existing_pgn(output_dir, cfg.data_download_year, month)
        if existing is not None:
            successes += 1
            continue
        url = lichess_download.BASE_URL.format(
            year=cfg.data_download_year, month=month
        )
        zip_path = output_dir / f"lichess_elite_{cfg.data_download_year:04d}-{month:02d}.zip"
        size = available[month]
        try:
            lichess_download.download_one(url, zip_path, size, ui=use_tui)
            lichess_download.extract_zip(zip_path, output_dir)
            successes += 1
        except Exception as e:  # noqa: BLE001
            tui.console.print(f"[error]download failed for {url}: {e}[/]")

    if successes:
        tui.console.print(f"[success]downloaded/prepared {successes} PGN month(s)[/]")
    return _list_pgns(cfg.data_dir)





def _value_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    err = (pred.float() - target.float()).abs()
    valid = mask.float()
    denom = valid.sum().clamp(min=1.0)
    return float((err * valid).sum() / denom)


def _sample_overlay(
    model: Kibitzer,
    batch: dict,
    device: str,
    autocast_dtype: torch.dtype | None,
) -> tuple[chess.Board, list[tuple[chess.Move, float]], float]:
    """top moves and value for the first position in ``batch``."""
    board = chess.Board()
    if autocast_dtype is not None:
        amp_ctx = torch.autocast(device_type=device, dtype=autocast_dtype)
    else:
        amp_ctx = contextlib.nullcontext()
    with torch.no_grad(), amp_ctx:
        piece_idx = batch["piece_idx"][:1]
        aux = batch["aux"][:1]
        loss_mask = batch["loss_mask"][:1]
        legal_mask = batch["legal_mask"][:1]
        policy_logits, value_pred = model(piece_idx, aux, ~loss_mask)
        legal0 = legal_mask[0, 0]
        masked = policy_logits[0, 0].masked_fill(~legal0, float("-inf"))
        probs = torch.softmax(masked.float(), dim=-1)
        n_top = min(5, int(legal0.sum().item()))
        if n_top == 0:
            return board, [], 0.0
        top = torch.topk(probs, k=n_top)
        top_moves = []
        for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
            try:
                mv = index_to_move(idx, board)
            except ValueError:
                continue
            top_moves.append((mv, float(prob)))
        v = float(value_pred[0, 0, 0])
    return board, top_moves, v


def main() -> int:
    args = parse_args()
    cfg = _load_config(args)
    cfg.run_name = cfg.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    use_tui = not args.no_tui and tui.is_tty()

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    requested_dtype = _DTYPES.get(cfg.dtype, torch.float32)
    if device == "cpu":
        param_dtype = torch.float32
        autocast_dtype = None
    else:
        param_dtype = requested_dtype
        autocast_dtype = (
            requested_dtype if requested_dtype != torch.float32 else None
        )

    model_cfg = KibitzerConfig(max_seq_len=cfg.max_seq_len)
    model = Kibitzer(model_cfg).to(device)
    if param_dtype != torch.float32:
        model = model.to(param_dtype)
    n_params = count_params(model)
    tui.console.print(
        f"[muted]model: {n_params:,} params · device={device} · "
        f"dtype={cfg.dtype} (param={param_dtype})[/]"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.peak_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, map_location=device)
        start_step = int(ckpt.get("step", 0))
        tui.console.print(
            f"[muted]resumed from {args.resume} at step {start_step}[/]"
        )

    wandb_run = None
    if cfg.wandb:
        try:
            import wandb  # type: ignore
        except ImportError:
            tui.console.print("[error]--wandb requested but `wandb` not installed[/]")
            return 1
        wandb_run = wandb.init(
            project="kibitzer", name=cfg.run_name, config=asdict(cfg)
        )

    hf_push = prepare_hf_push(cfg, use_tui)

    pgn_paths = _download_missing_data(cfg, use_tui)
    if not pgn_paths:
        tui.console.print(f"[error]no PGN files found under {cfg.data_dir}[/]")
        return 1
    tui.console.print(f"[muted]found {len(pgn_paths)} PGN file(s) in {cfg.data_dir}[/]")

    dataset = LichessGameDataset(
        pgn_paths,
        max_plies=cfg.max_seq_len,
        min_elo=2400,
        min_plies=10,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        seed=cfg.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=collate_games,
        persistent_workers=cfg.num_workers > 0,
    )
    it = _forever(loader)

    first_batch: dict | None = None
    if use_tui:
        with tqdm(
            total=1,
            desc="Preparing DataLoader first batch",
            unit="batch",
            leave=True,
        ) as bar:
            first_batch = next(it)
            bar.update(1)

    run_dir = Path(cfg.checkpoint_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    ema_loss = EMA(beta=0.99)
    ema_acc = EMA(beta=0.99)
    state = tui.TrainState(run_name=cfg.run_name, total_steps=cfg.total_steps)
    live_holder: dict = {"live": None}
    last_eval_elo: float | None = None

    def log_line(msg: str) -> None:
        state.log_tail_lines.append(msg)
        state.log_tail_lines = state.log_tail_lines[-12:]
        if not use_tui:
            print(msg, flush=True)

    def refresh() -> None:
        live = live_holder["live"]
        if live is not None:
            live.update(tui.train_layout(state))

    def save_and_maybe_eval(step: int, do_eval: bool) -> None:
        nonlocal last_eval_elo
        ckpt_path = run_dir / f"step_{step:06d}.pt"
        metrics = {"ema_loss": ema_loss.value, "ema_acc": ema_acc.value}
        save_checkpoint(
            ckpt_path, model, optimizer,
            scheduler_state=None, step=step, config=asdict(cfg),
            metrics=metrics,
        )
        latest = run_dir / "latest.pt"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(ckpt_path.name)
        except OSError:
            pass
        log_line(f"[ckpt] saved {ckpt_path}")
        refresh()

        if do_eval:
            log_line(
                f"[eval] cutechess vs {cfg.eval_opponent} "
                f"({cfg.eval_n_games} games)…"
            )
            refresh()
            try:
                r = evaluate_checkpoint(
                    str(ckpt_path),
                    opponent=cfg.eval_opponent,
                    n_games=cfg.eval_n_games,
                    time_per_move_ms=cfg.eval_time_per_move_ms,
                )
                state.elo_history.append((step, r["elo_diff"]))
                last_eval_elo = float(r["elo_diff"])
                metrics.update({
                    "eval_score": r["score"],
                    "eval_n_games": r["n_games"],
                    "eval_elo_diff": r["elo_diff"],
                    "eval_elo_err": r["elo_err"],
                    "eval_win_rate": r["win_rate"],
                })
                log_line(
                    f"[eval] step={step} score={r['score']:.1f}/{r['n_games']} "
                    f"elo={r['elo_diff']:+.1f}±{r['elo_err']:.1f}"
                )
                if wandb_run is not None:
                    try:
                        wandb_run.log({
                            "step": step,
                            "eval/elo_diff": r["elo_diff"],
                            "eval/elo_err": r["elo_err"],
                            "eval/win_rate": r["win_rate"],
                        })
                    except Exception:
                        pass
            except Exception as e:
                log_line(f"[eval] failed: {type(e).__name__}: {e}")
            refresh()

        if last_eval_elo is not None:
            metrics["last_eval_elo_diff"] = last_eval_elo
        pushed = push_checkpoint_to_hf(
            hf_push, ckpt_path, step, cfg, metrics, last_eval_elo,
            eval_opponent=cfg.eval_opponent,
        )
        if pushed is not None:
            if "/" in pushed and not pushed.startswith(("ImportError", "RuntimeError")):
                log_line(f"[hf] pushed {pushed}")
            else:
                log_line(f"[hf] push failed: {pushed}")
            refresh()

    def loop_body() -> int:
        nonlocal first_batch, start_step
        max_steps = cfg.total_steps
        if args.dry_run:
            max_steps = start_step + 5

        t0 = time.time()
        plies_seen = 0

        for step in range(start_step, max_steps):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            acc_sum = 0.0
            vmae_sum = 0.0
            last_value_pred: torch.Tensor | None = None
            last_batch: dict | None = None

            for _ in range(cfg.grad_accum_steps):
                if first_batch is not None:
                    batch = first_batch
                    first_batch = None
                else:
                    batch = next(it)
                batch = _move_to(batch, device)
                last_batch = batch
                if autocast_dtype is not None:
                    ctx = torch.autocast(device_type=device, dtype=autocast_dtype)
                else:
                    ctx = contextlib.nullcontext()
                with ctx:
                    policy_logits, value_pred = model(
                        batch["piece_idx"], batch["aux"], ~batch["loss_mask"]
                    )
                    value_pred = value_pred.squeeze(-1)
                    loss_dict = combined_loss(
                        {"policy_logits": policy_logits, "value": value_pred},
                        batch,
                        value_weight=cfg.value_loss_weight,
                    )
                (loss_dict["loss"] / cfg.grad_accum_steps).backward()
                loss_sum += float(loss_dict["loss"].item())
                acc_sum += float(loss_dict["policy_acc"].item())
                vmae_sum += _value_mae(
                    value_pred.detach(), batch["value_target"], batch["loss_mask"]
                )
                last_value_pred = value_pred.detach()
                plies_seen += int(batch["loss_mask"].sum().item())

            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.max_grad_norm
            ))
            lr = get_lr(
                step, cfg.warmup_steps, cfg.total_steps, cfg.peak_lr, cfg.min_lr
            )
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.step()

            avg_loss = loss_sum / cfg.grad_accum_steps
            avg_acc = acc_sum / cfg.grad_accum_steps
            avg_vmae = vmae_sum / cfg.grad_accum_steps
            ema_loss.update(avg_loss)
            ema_acc.update(avg_acc)

            if step % cfg.log_every == 0 or step == max_steps - 1:
                wall = time.time() - t0
                pps = plies_seen / max(1.0, wall)
                steps_done = step - start_step + 1
                eta = (
                    (cfg.total_steps - step - 1) * (wall / max(1, steps_done))
                )

                state.step = step
                state.wall_seconds = wall
                state.eta_seconds = eta
                state.ema_loss = ema_loss.value
                state.ema_acc = ema_acc.value
                state.last_lr = lr
                state.last_grad_norm = grad_norm
                state.plies_per_sec = pps
                state.loss_history.append((step, ema_loss.value))

                if last_batch is not None:
                    try:
                        b, top, v = _sample_overlay(
                            model, last_batch, device, autocast_dtype
                        )
                        state.sample_board = b
                        state.sample_top_moves = top
                        state.sample_value = v
                    except Exception as e:  # noqa: BLE001
                        log_line(f"[overlay] {type(e).__name__}: {e}")

                log_line(
                    f"step={step} loss={ema_loss.value:.4f} "
                    f"acc={ema_acc.value*100:.2f}% vmae={avg_vmae:.3f} "
                    f"lr={lr:.2e} gnorm={grad_norm:.3f} plies/s={pps:.0f}"
                )

                if wandb_run is not None:
                    try:
                        wandb_run.log({
                            "step": step,
                            "loss/ema": ema_loss.value,
                            "loss/policy": float(loss_dict["policy_loss"].item()),
                            "loss/value": float(loss_dict["value_loss"].item()),
                            "policy_acc": ema_acc.value,
                            "value_mae": avg_vmae,
                            "lr": lr,
                            "grad_norm": grad_norm,
                            "plies_per_sec": pps,
                        })
                    except Exception:
                        pass

                refresh()

            do_ckpt = step > 0 and step % cfg.checkpoint_every == 0
            do_eval = step > 0 and step % cfg.eval_every == 0
            if do_ckpt or do_eval:
                save_and_maybe_eval(step, do_eval)

        # final save, with eval unless this is a dry run.
        save_and_maybe_eval(max_steps, do_eval=not args.dry_run)
        return 0

    if use_tui:
        with Live(
            tui.train_layout(state),
            console=tui.console,
            refresh_per_second=1,
            screen=True,
            redirect_stdout=False,
            redirect_stderr=False,
        ) as live:
            live_holder["live"] = live
            return loop_body()
    else:
        return loop_body()


if __name__ == "__main__":
    sys.exit(main())
