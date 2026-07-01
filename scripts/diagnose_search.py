from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.engine
import chess.pgn
import torch
from tqdm import tqdm

from kibitzer.diagnostics import (
    VALUE_BINS,
    paired_bootstrap_interval,
    summarize_move_regret,
    summarize_value_predictions,
    value_bin,
)
from kibitzer.inference import ModelEvaluator
from kibitzer.search import puct_search


MATE_SCORE_CP = 10_000
REGRET_CAP_CP = 2_000
ORACLE_VERSION = 1
_engine: chess.engine.SimpleEngine | None = None
_engine_depth = 0


@dataclass(frozen=True)
class Candidate:
    fen: str
    game_key: str
    split: str


def _start_engine(path: str, depth: int) -> None:
    global _engine, _engine_depth
    _engine = chess.engine.SimpleEngine.popen_uci(path)
    _engine_depth = depth


def _analyse_task(task: tuple[str, str | None]) -> tuple[int, str | None]:
    if _engine is None:
        raise RuntimeError("Stockfish worker was not initialized")
    fen, move_uci = task
    board = chess.Board(fen)
    perspective = board.turn
    if move_uci is not None:
        board.push_uci(move_uci)
    info = _engine.analyse(board, chess.engine.Limit(depth=_engine_depth))
    score = info["score"].pov(perspective).score(mate_score=MATE_SCORE_CP)
    if score is None:
        score = 0
    best_move = None
    if move_uci is None and info.get("pv"):
        best_move = info["pv"][0].uci()
    return int(score), best_move


class StockfishPool:
    def __init__(self, *, path: str, depth: int, workers: int) -> None:
        context = multiprocessing.get_context("spawn")
        self.pool = context.Pool(
            workers,
            initializer=_start_engine,
            initargs=(path, depth),
        )

    def analyse(
        self,
        tasks: list[tuple[str, str | None]],
        *,
        description: str,
    ) -> list[tuple[int, str | None]]:
        results = self.pool.imap(_analyse_task, tasks, chunksize=4)
        return list(tqdm(results, total=len(tasks), desc=description, unit="pos"))

    def close(self) -> None:
        self.pool.terminate()
        self.pool.join()

    def __enter__(self) -> StockfishPool:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def game_split(game_key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{game_key}".encode()).digest()
    return "validation" if digest[0] % 2 == 0 else "test"


def iter_real_positions(
    paths: list[Path],
    *,
    min_ply: int,
    stride: int,
    seed: int,
) -> Iterator[Candidate]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            game_index = 0
            while game := chess.pgn.read_game(handle):
                game_index += 1
                game_key = f"{path.resolve()}:{game_index}"
                split = game_split(game_key, seed)
                board = game.board()
                for ply, move in enumerate(game.mainline_moves()):
                    if ply >= min_ply and (ply - min_ply) % stride == 0:
                        yield Candidate(board.fen(), game_key, split)
                    board.push(move)


def counts_by_split_and_bin(records: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    return {
        split: {
            name: sum(
                record["split"] == split and record["value_bin"] == name
                for record in records
            )
            for name in VALUE_BINS
        }
        for split in ("validation", "test")
    }


def build_oracle(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit(f"oracle already exists: {args.output}")
    paths = [Path(path) for path in args.pgn]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing PGNs: {missing}")

    target_candidates = args.positions_per_bin * args.oversample
    accepted: list[dict[str, object]] = []
    accepted_keys: set[tuple[str, str, str]] = set()
    accepted_counts = {
        split: {name: 0 for name in VALUE_BINS}
        for split in ("validation", "test")
    }
    natural_counts = {name: 0 for name in VALUE_BINS}
    batch: list[Candidate] = []

    def consume_prescan_results(
        candidates: list[Candidate],
        results: list[tuple[int, str | None]],
    ) -> None:
        for item, (centipawns, _) in zip(candidates, results, strict=True):
            name = value_bin(centipawns)
            natural_counts[name] += 1
            key = (item.split, name, item.game_key)
            if accepted_counts[item.split][name] < target_candidates and key not in accepted_keys:
                accepted.append(
                    {
                        "fen": item.fen,
                        "game_key": item.game_key,
                        "split": item.split,
                        "prescan_bin": name,
                    }
                )
                accepted_keys.add(key)
                accepted_counts[item.split][name] += 1

    print("[1/3] DEPTH-10 PRESCAN OF UNSEEN REAL-GAME POSITIONS", flush=True)
    print(f"  sources:          {len(paths)} PGN files")
    print(f"  target candidates:{target_candidates} per split/bin")
    with StockfishPool(
        path=args.stockfish_path,
        depth=args.prescan_depth,
        workers=args.stockfish_workers,
    ) as pool:
        for candidate in iter_real_positions(
            paths,
            min_ply=args.min_ply,
            stride=args.position_stride,
            seed=args.seed,
        ):
            batch.append(candidate)
            if len(batch) < args.prescan_batch_size:
                continue
            results = pool.analyse(
                [(item.fen, None) for item in batch],
                description="prescan",
            )
            consume_prescan_results(batch, results)
            batch.clear()
            if all(
                accepted_counts[split][name] >= target_candidates
                for split in accepted_counts
                for name in VALUE_BINS
            ):
                break
        if batch:
            results = pool.analyse(
                [(item.fen, None) for item in batch],
                description="prescan final",
            )
            consume_prescan_results(batch, results)

    print("[2/3] COMMON DEPTH-20 MULTIPV-1 ORACLE", flush=True)
    with StockfishPool(
        path=args.stockfish_path,
        depth=args.oracle_depth,
        workers=args.stockfish_workers,
    ) as pool:
        oracle_results = pool.analyse(
            [(str(record["fen"]), None) for record in accepted],
            description="oracle",
        )
    rebinned: list[dict[str, object]] = []
    final_game_bins: set[tuple[str, str, str]] = set()
    for record, (centipawns, best_move) in zip(accepted, oracle_results, strict=True):
        name = value_bin(centipawns)
        key = (str(record["split"]), name, str(record["game_key"]))
        if key in final_game_bins:
            continue
        current = sum(
            item["split"] == record["split"] and item["value_bin"] == name
            for item in rebinned
        )
        if current >= args.positions_per_bin:
            continue
        rebinned.append(
            {
                "fen": record["fen"],
                "game_key": record["game_key"],
                "split": record["split"],
                "centipawns": centipawns,
                "value_bin": name,
                "best_move": best_move,
            }
        )
        final_game_bins.add(key)

    final_counts = counts_by_split_and_bin(rebinned)
    if not all(
        final_counts[split][name] >= args.positions_per_bin
        for split in final_counts
        for name in VALUE_BINS
    ):
        raise SystemExit(
            "depth-20 re-binning did not fill every split/bin; add unseen PGN months "
            f"or increase --oversample. Counts: {final_counts}"
        )

    total_prescan = sum(natural_counts.values())
    natural_weights = {
        name: count / total_prescan for name, count in natural_counts.items()
    }
    payload = {
        "version": ORACLE_VERSION,
        "source_pgns": [str(path.resolve()) for path in paths],
        "prescan_depth": args.prescan_depth,
        "oracle_depth": args.oracle_depth,
        "mate_score_cp": MATE_SCORE_CP,
        "value_transform": "clip(cp / 1000, -1, 1)",
        "natural_bin_weights": natural_weights,
        "records": rebinned,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print("[3/3] ORACLE LOCKED", flush=True)
    print(f"  counts: {final_counts}")
    print(f"  output: {args.output}")


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit("--checkpoint must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"checkpoint does not exist: {path}")
        checkpoints[name] = path
    return checkpoints


def load_move_cache(path: Path, *, depth: int) -> dict[str, int]:
    if not path.is_file():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("depth") != depth:
        raise SystemExit(f"chosen-move cache depth does not match: {path}")
    return payload["scores"]


def save_move_cache(path: Path, *, depth: int, scores: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save({"depth": depth, "scores": scores}, temporary)
    temporary.replace(path)


def move_cache_key(fen: str, move_uci: str) -> str:
    return hashlib.sha256(f"{fen}|{move_uci}".encode()).hexdigest()


def load_tactics(path: Path | None) -> list[tuple[chess.Board, set[chess.Move]]]:
    if path is None:
        return []
    tactics = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        board = chess.Board()
        operations = board.set_epd(line)
        best_moves = set(operations.get("bm", []))
        if not best_moves:
            raise SystemExit(f"tactical EPD is missing bm moves: {line}")
        tactics.append((board, best_moves))
    return tactics


def evaluate(args: argparse.Namespace) -> None:
    test_lock = args.oracle.with_suffix(args.oracle.suffix + ".test.lock")
    if args.split == "test" and test_lock.exists():
        raise SystemExit(f"oracle test split was already consumed: {test_lock}")
    if args.split == "test" and args.output.exists():
        raise SystemExit(f"test output already exists and is locked: {args.output}")
    oracle = torch.load(args.oracle, map_location="cpu", weights_only=False)
    if oracle.get("version") != ORACLE_VERSION:
        raise SystemExit("unsupported oracle cache version")
    records = [
        record for record in oracle["records"] if record["split"] == args.split
    ]
    if not records:
        raise SystemExit(f"oracle has no {args.split} records")
    checkpoints = parse_checkpoints(args.checkpoint)
    value_scales = [float(value) for value in args.value_scales.split(",")]
    if args.split == "test" and len(value_scales) != 1:
        raise SystemExit("test evaluation accepts exactly one selected value scale")

    all_choices: dict[str, list[str]] = {}
    all_predictions: dict[str, list[float]] = {}
    tactical_results: dict[str, dict[str, float]] = {}
    tactics = load_tactics(args.tactics_epd)

    for checkpoint_name, checkpoint_path in checkpoints.items():
        print(f"evaluating checkpoint: {checkpoint_name} ({checkpoint_path})", flush=True)
        evaluator = ModelEvaluator.from_checkpoint(checkpoint_path, device=args.device)
        raw_moves = []
        predictions = []
        boards = [chess.Board(str(record["fen"])) for record in records]
        for board in tqdm(boards, desc=f"{checkpoint_name} raw", unit="pos"):
            evaluation = evaluator.evaluate(board)
            raw_moves.append(max(evaluation.priors, key=evaluation.priors.get).uci())
            predictions.append(evaluation.value)
        all_choices[f"{checkpoint_name}:raw"] = raw_moves
        all_predictions[checkpoint_name] = predictions

        for value_scale in value_scales:
            label = f"{checkpoint_name}:s{args.simulations}:v{value_scale:g}"
            moves = []
            for board in tqdm(boards, desc=label, unit="pos"):
                moves.append(
                    puct_search(
                        board,
                        evaluator,
                        simulations=args.simulations,
                        c_puct=args.c_puct,
                        value_scale=value_scale,
                    ).move.uci()
                )
            all_choices[label] = moves

        if tactics:
            strategy_scores: dict[str, float] = {}
            raw_solved = 0
            for board, best_moves in tactics:
                evaluation = evaluator.evaluate(board)
                raw_solved += max(evaluation.priors, key=evaluation.priors.get) in best_moves
            strategy_scores["raw"] = raw_solved / len(tactics)
            for value_scale in value_scales:
                solved = 0
                for board, best_moves in tactics:
                    move = puct_search(
                        board,
                        evaluator,
                        simulations=args.simulations,
                        c_puct=args.c_puct,
                        value_scale=value_scale,
                    ).move
                    solved += move in best_moves
                strategy_scores[f"s{args.simulations}:v{value_scale:g}"] = solved / len(tactics)
            tactical_results[checkpoint_name] = strategy_scores

    move_scores = load_move_cache(args.chosen_move_cache, depth=oracle["oracle_depth"])
    missing_tasks: dict[str, tuple[str, str | None]] = {}
    for choices in all_choices.values():
        for record, move_uci in zip(records, choices, strict=True):
            key = move_cache_key(str(record["fen"]), move_uci)
            if key not in move_scores:
                missing_tasks[key] = (str(record["fen"]), move_uci)
    if missing_tasks:
        print(f"evaluating {len(missing_tasks):,} uncached chosen moves with Stockfish", flush=True)
        keys = list(missing_tasks)
        with StockfishPool(
            path=args.stockfish_path,
            depth=oracle["oracle_depth"],
            workers=args.stockfish_workers,
        ) as pool:
            results = pool.analyse(
                [missing_tasks[key] for key in keys],
                description="chosen moves",
            )
        for key, (centipawns, _) in zip(keys, results, strict=True):
            move_scores[key] = centipawns
        save_move_cache(
            args.chosen_move_cache,
            depth=oracle["oracle_depth"],
            scores=move_scores,
        )

    centipawns = [int(record["centipawns"]) for record in records]
    report: dict[str, object] = {
        "split": args.split,
        "oracle": str(args.oracle),
        "positions": len(records),
        "checkpoints": {},
        "strategies": {},
        "tactics": tactical_results,
        "gates": {"value": {}, "search": {}, "tactical_non_regression": {}},
    }
    for checkpoint_name, predictions in all_predictions.items():
        value_by_bin = summarize_value_predictions(centipawns, predictions)
        natural_weights = oracle.get("natural_bin_weights", {})
        weighted_mae = sum(
            natural_weights.get(name, 0.0) * value_by_bin[name]["mae"]
            for name in VALUE_BINS
            if value_by_bin[name]["count"]
        )
        report["checkpoints"][checkpoint_name] = {
            "value_by_bin": value_by_bin,
            "natural_weighted_mae": weighted_mae,
        }
        report["gates"]["value"][checkpoint_name] = {
            "decisive_sign_at_least_90pct": value_by_bin["decisive"]["sign_accuracy"]
            >= 0.90,
            "won_sign_at_least_95pct": value_by_bin["won"]["sign_accuracy"] >= 0.95,
        }

    strategy_regrets: dict[str, list[float]] = {}
    for label, choices in all_choices.items():
        regrets = []
        exact_matches = 0
        for record, move_uci in zip(records, choices, strict=True):
            chosen_cp = move_scores[move_cache_key(str(record["fen"]), move_uci)]
            regret = min(
                REGRET_CAP_CP,
                max(0, int(record["centipawns"]) - chosen_cp),
            )
            regrets.append(float(regret))
            exact_matches += move_uci == record["best_move"]
        strategy_regrets[label] = regrets
        metrics = summarize_move_regret(regrets)
        metrics["best_move_accuracy"] = exact_matches / len(records)
        report["strategies"][label] = metrics

    comparisons = {}
    for label, regrets in strategy_regrets.items():
        checkpoint_name = label.split(":", 1)[0]
        raw_label = f"{checkpoint_name}:raw"
        if label == raw_label:
            continue
        raw_regrets = strategy_regrets[raw_label]
        regret_improvements = [
            raw - candidate for raw, candidate in zip(raw_regrets, regrets, strict=True)
        ]
        near_best_improvements = [
            float(candidate <= 50) - float(raw <= 50)
            for raw, candidate in zip(raw_regrets, regrets, strict=True)
        ]
        regret_ci = paired_bootstrap_interval(regret_improvements, seed=args.seed)
        near_best_ci = paired_bootstrap_interval(near_best_improvements, seed=args.seed + 1)
        raw_p90 = float(report["strategies"][raw_label]["p90_cp"])
        candidate_p90 = float(report["strategies"][label]["p90_cp"])
        comparisons[label] = {
            "mean_regret_improvement_cp": sum(regret_improvements) / len(regret_improvements),
            "mean_regret_improvement_ci95": regret_ci,
            "near_best_improvement": sum(near_best_improvements) / len(near_best_improvements),
            "near_best_improvement_ci95": near_best_ci,
            "p90_reduction_cp": raw_p90 - candidate_p90,
            "p90_reduction_fraction": (
                (raw_p90 - candidate_p90) / raw_p90 if raw_p90 else 0.0
            ),
        }
        report["gates"]["search"][label] = {
            "paired_regret_ci_above_zero": regret_ci[0] > 0.0,
            "paired_near_best_ci_above_zero": near_best_ci[0] > 0.0,
            "p90_reduction_at_least_15cp": raw_p90 - candidate_p90 >= 15.0,
            "p90_reduction_at_least_10pct": (
                (raw_p90 - candidate_p90) / raw_p90 >= 0.10 if raw_p90 else False
            ),
        }
    report["comparisons_to_raw"] = comparisons

    baseline_tactics = tactical_results.get("phase2", {})
    for checkpoint_name, scores in tactical_results.items():
        report["gates"]["tactical_non_regression"][checkpoint_name] = {
            strategy: score >= baseline_tactics.get(strategy, score)
            for strategy, score in scores.items()
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    if args.split == "test":
        test_lock.write_text(
            json.dumps(
                {
                    "output": str(args.output.resolve()),
                    "checkpoints": {name: str(path.resolve()) for name, path in checkpoints.items()},
                    "value_scales": value_scales,
                    "simulations": args.simulations,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, allow_nan=True))
    print(f"report: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate a locked search oracle.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build disjoint validation/test oracle caches.")
    build.add_argument("--pgn", action="append", required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--stockfish-path", default="stockfish")
    build.add_argument("--stockfish-workers", type=int, default=8)
    build.add_argument("--prescan-depth", type=int, default=10)
    build.add_argument("--oracle-depth", type=int, default=20)
    build.add_argument("--positions-per-bin", type=int, default=200)
    build.add_argument("--oversample", type=int, default=3)
    build.add_argument("--prescan-batch-size", type=int, default=512)
    build.add_argument("--min-ply", type=int, default=12)
    build.add_argument("--position-stride", type=int, default=4)
    build.add_argument("--seed", type=int, default=42)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate validation or locked test split.")
    evaluate_parser.add_argument("--oracle", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate_parser.add_argument("--checkpoint", action="append", required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--chosen-move-cache", type=Path, required=True)
    evaluate_parser.add_argument("--stockfish-path", default="stockfish")
    evaluate_parser.add_argument("--stockfish-workers", type=int, default=8)
    evaluate_parser.add_argument("--simulations", type=int, default=64)
    evaluate_parser.add_argument("--value-scales", default="0,0.5,1")
    evaluate_parser.add_argument("--c-puct", type=float, default=1.5)
    evaluate_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    evaluate_parser.add_argument("--tactics-epd", type=Path)
    evaluate_parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        build_oracle(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
