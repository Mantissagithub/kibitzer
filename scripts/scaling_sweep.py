# scaling-law sweep trainer: train a ladder of attention-first model sizes and
# log a policy-loss-vs-params curve. drives kibitzer training directly (mirroring
# scripts/train_bc.py's streaming loop) rather than shelling out, so per-rung lr
# schedules and eval sit inline. see docs/scaling_study/design.md and decisions
# d36 for the ladder/lr rationale, d42 for the in-loop stockfish early-stopping eval.

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import chess
import chess.engine
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from kibitzer.data import StreamingPositionDataset, collate_positions
from kibitzer.inference import ModelEvaluator
from kibitzer.model import Kibitzer, KibitzerConfig
from kibitzer.search import puct_search

# attention-first ladder: attention_every=1 forces every trunk block to be a
# CausalAttentionBlock. encoder_heads stays 8 (divides every d_model here);
# n_heads is chosen to divide its own d_model.
SIZES: dict[str, dict[str, int]] = {
    "S0": {"d_model": 128, "trunk_layers": 6, "n_heads": 4},
    "S1": {"d_model": 192, "trunk_layers": 8, "n_heads": 6},
    "S2": {"d_model": 256, "trunk_layers": 10, "n_heads": 8},
    "S3": {"d_model": 320, "trunk_layers": 10, "n_heads": 8},
    "S4": {"d_model": 448, "trunk_layers": 12, "n_heads": 8},
}

# fixed opening book for the play-eval, so the early-stop signal tracks the model
# and not opening variance. alternating colors cover both sides of each line.
EVAL_OPENINGS = [
    ("e2e4", "e7e5", "g1f3", "b8c6"),
    ("e2e4", "c7c5", "g1f3", "d7d6"),
    ("d2d4", "d7d5", "c2c4", "e7e6"),
    ("c2c4", "e7e5", "b1c3", "g8f6"),
    ("g1f3", "d7d5", "g2g3", "g8f6"),
]


def build_config(tag: str, pos_encoding: str = "absolute") -> KibitzerConfig:
    spec = SIZES[tag]
    return KibitzerConfig(
        d_model=spec["d_model"],
        trunk_layers=spec["trunk_layers"],
        n_heads=spec["n_heads"],
        encoder_heads=8,
        attention_every=1,
        pos_encoding=pos_encoding,
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
    parser.add_argument(
        "--pos-encoding",
        choices=["absolute", "shaw"],
        default="absolute",
        help="Position encoding for the position encoder (D39 A/B).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="reports/scaling_law/results.json")
    parser.add_argument("--ckpt-dir", default="runs/scaling")
    parser.add_argument("--save-every", type=int, default=0, help="Interim checkpoint every N steps (0=off).")
    # in-loop play eval vs stockfish with early stopping (d42). eval-every=0 -> off.
    parser.add_argument("--eval-every", type=int, default=0, help="Play-eval vs Stockfish every N positions (0=off).")
    parser.add_argument("--eval-elo", type=int, default=1900)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--eval-sims", type=int, default=64)
    parser.add_argument("--eval-c-puct", type=float, default=1.5)
    parser.add_argument("--eval-value-scale", type=float, default=1.0)
    parser.add_argument("--eval-max-plies", type=int, default=200)
    parser.add_argument("--eval-patience", type=int, default=2, help="Stop after this many non-improving evals.")
    parser.add_argument("--eval-min-delta", type=float, default=0.02, help="Score gain that counts as improvement.")
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--stockfish-time", type=float, default=0.05)
    return parser.parse_args()


def lr_for_rung(tag: str, sizes: list[str], *, base_lr: float, lr_transfer: str) -> float:
    if lr_transfer == "constant":
        return base_lr
    # mup-style transfer: lr scales inversely with width off the smallest rung.
    smallest_d_model = min(SIZES[t]["d_model"] for t in sizes)
    return base_lr * smallest_d_model / SIZES[tag]["d_model"]


# linear warmup over warmup_frac of total_steps, then cosine decay to 10% of peak.
def lr_lambda(step: int, total_steps: int, warmup_frac: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return 0.1 + 0.9 * cosine


# one eval game: model plays via puct from a fixed opening, stockfish replies at
# the configured elo. returns "win"/"draw"/"loss" from the model's pov.
def play_eval_game(
    *,
    game_index: int,
    evaluator: ModelEvaluator,
    engine: chess.engine.SimpleEngine,
    simulations: int,
    c_puct: float,
    value_scale: float,
    stockfish_time: float,
    max_plies: int,
) -> str:
    board = chess.Board()
    for move_uci in EVAL_OPENINGS[(game_index // 2) % len(EVAL_OPENINGS)]:
        board.push_uci(move_uci)
    network_color = chess.WHITE if game_index % 2 == 0 else chess.BLACK

    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == network_color:
            move = puct_search(
                board, evaluator, simulations=simulations, c_puct=c_puct, value_scale=value_scale
            ).move
        else:
            played = engine.play(board, chess.engine.Limit(time=stockfish_time))
            if played.move is None:
                raise RuntimeError("Stockfish returned no move for a non-terminal board")
            move = played.move
        board.push(move)
        plies += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "draw"
    return "win" if outcome.winner == network_color else "loss"


# play a short match and return the model's score in [0,1] (win=1, draw=0.5). the
# in-training model is wrapped in a ModelEvaluator, which flips it to eval mode;
# the caller restores train mode afterward. a fresh stockfish process per match
# keeps training decoupled from a long-lived engine handle.
def play_vs_stockfish(
    model: Kibitzer,
    *,
    device: str,
    games: int,
    simulations: int,
    c_puct: float,
    value_scale: float,
    stockfish_path: str,
    stockfish_elo: int,
    stockfish_time: float,
    max_plies: int,
) -> float:
    evaluator = ModelEvaluator(model, device=device)
    counts = {"win": 0, "draw": 0, "loss": 0}
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
        for game_index in range(games):
            result = play_eval_game(
                game_index=game_index,
                evaluator=evaluator,
                engine=engine,
                simulations=simulations,
                c_puct=c_puct,
                value_scale=value_scale,
                stockfish_time=stockfish_time,
                max_plies=max_plies,
            )
            counts[result] += 1
    return (counts["win"] + 0.5 * counts["draw"]) / games


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
    pos_encoding: str = "absolute",
    ckpt_path: Path | None = None,
    save_every: int = 0,
    eval_cfg: dict[str, Any] | None = None,
) -> tuple[Kibitzer, int, float, list[dict[str, float]], int]:
    # trains one rung for a single pass over max_positions. returns
    # (model, params, wall_clock_s, eval_history, positions_trained); the last
    # two differ from the defaults only when the play-eval early-stops the rung.
    cfg = build_config(tag, pos_encoding=pos_encoding)
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

    # play-eval bookkeeping. eval_step_interval converts the positions cadence
    # into a step count; best_score / no_improve drive the plateau early stop.
    eval_step_interval = 0
    if eval_cfg is not None and eval_cfg["every_positions"] > 0:
        eval_step_interval = max(1, eval_cfg["every_positions"] // batch_size)
    eval_history: list[dict[str, float]] = []
    best_score = -1.0
    no_improve = 0
    positions_trained = max_positions

    model.train()
    start = time.perf_counter()
    progress = tqdm(loader, total=total_steps, desc=f"train {tag}")
    for step, batch in enumerate(progress, start=1):
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

        # interim checkpoint so a long run survives an interrupt.
        if ckpt_path is not None and save_every > 0 and step % save_every == 0:
            torch.save(
                {"model": model.state_dict(), "config": model.config, "step": step},
                ckpt_path,
            )

        # periodic play-eval vs stockfish + plateau early stop. runs between
        # optimizer steps and never touches the optimizer, so it leaves training
        # numerics untouched. wrapped defensively: a flaky engine skips the eval
        # rather than killing hours of training.
        if eval_step_interval and step % eval_step_interval == 0:
            positions_seen = step * batch_size
            if ckpt_path is not None:
                torch.save(
                    {"model": model.state_dict(), "config": model.config, "step": step},
                    ckpt_path,
                )
            try:
                score = play_vs_stockfish(
                    model,
                    device=device,
                    games=eval_cfg["games"],
                    simulations=eval_cfg["sims"],
                    c_puct=eval_cfg["c_puct"],
                    value_scale=eval_cfg["value_scale"],
                    stockfish_path=eval_cfg["stockfish_path"],
                    stockfish_elo=eval_cfg["elo"],
                    stockfish_time=eval_cfg["stockfish_time"],
                    max_plies=eval_cfg["max_plies"],
                )
            except Exception as exc:  # engine hiccup shouldn't lose the run
                print(f"  [eval skipped @ {positions_seen:,} positions: {exc}]")
                model.train()
                continue
            model.train()  # ModelEvaluator flipped us to eval mode

            eval_history.append({"positions": positions_seen, "score": score})
            improved = score > best_score + eval_cfg["min_delta"]
            best_score = max(best_score, score)
            no_improve = 0 if improved else no_improve + 1
            print(
                f"  [play-eval @ {positions_seen:,} pos vs SF-{eval_cfg['elo']} "
                f"@ {eval_cfg['sims']} sims: score={score:.3f} best={best_score:.3f} "
                f"no_improve={no_improve}/{eval_cfg['patience']}]"
            )
            if no_improve >= eval_cfg["patience"]:
                positions_trained = positions_seen
                print(f"  [early stop: no play improvement for {eval_cfg['patience']} evals]")
                break

    wall_clock_s = time.perf_counter() - start
    return model, params, wall_clock_s, eval_history, positions_trained


# fixed, non-shuffled slice: shuffle_buffer_size=1 makes ordering deterministic.
@torch.no_grad()
def evaluate_rung(
    model: Kibitzer,
    *,
    eval_path: Path,
    eval_positions: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
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

    # bundle the play-eval knobs once; none disables in-loop eval entirely.
    eval_cfg = None
    if args.eval_every > 0:
        eval_cfg = {
            "every_positions": args.eval_every,
            "elo": args.eval_elo,
            "games": args.eval_games,
            "sims": args.eval_sims,
            "c_puct": args.eval_c_puct,
            "value_scale": args.eval_value_scale,
            "max_plies": args.eval_max_plies,
            "patience": args.eval_patience,
            "min_delta": args.eval_min_delta,
            "stockfish_path": args.stockfish_path,
            "stockfish_time": args.stockfish_time,
        }

    for tag in sizes:
        lr = lr_for_rung(tag, sizes, base_lr=args.base_lr, lr_transfer=args.lr_transfer)
        print(f"=== {tag}: {SIZES[tag]} lr={lr:.2e} ===")

        ckpt_path = ckpt_dir / f"{tag}.pt"
        model, params, wall_clock_s, eval_history, positions_trained = train_rung(
            tag,
            train_paths=train_paths,
            max_positions=args.max_positions,
            batch_size=args.batch_size,
            lr=lr,
            warmup_frac=args.warmup_frac,
            num_workers=args.num_workers,
            device=args.device,
            pos_encoding=args.pos_encoding,
            ckpt_path=ckpt_path,
            save_every=args.save_every,
            eval_cfg=eval_cfg,
        )

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
            "pos_encoding": args.pos_encoding,
            "params": params,
            # positions_trained reflects an early stop; falls back to max otherwise.
            "train_positions": positions_trained,
            "lr": lr,
            **eval_metrics,
            "play_eval_history": eval_history,
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
