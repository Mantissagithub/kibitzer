from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import chess
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from kibitzer.inference import ModelEvaluator
from kibitzer.search import puct_search


LOGGER = logging.getLogger("uvicorn.error")
CHECKPOINT = Path(os.getenv("KIBITZER_CHECKPOINT", "runs/tactical/tactical_repair.pt"))
BATCH_SIZE = 32


class MoveRequest(BaseModel):
    moves: list[str] = Field(default_factory=list, max_length=512)
    initial_fen: str | None = None
    simulations: Literal[64, 128, 512] = 128


class VisitLine(BaseModel):
    move: str
    visits: int
    share: float


class MoveResponse(BaseModel):
    move: str
    san: str
    fen: str
    value: float
    simulations: int
    elapsed_ms: int
    top_moves: list[VisitLine]


_evaluator: ModelEvaluator | None = None
_load_lock = threading.Lock()
_search_lock = threading.Lock()


def build_board(initial_fen: str | None, moves: list[str]) -> chess.Board:
    try:
        board = chess.Board(initial_fen) if initial_fen else chess.Board()
    except ValueError as error:
        raise ValueError(f"invalid initial FEN: {error}") from error

    for index, uci in enumerate(moves, start=1):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as error:
            raise ValueError(f"move {index} is not valid UCI: {uci}") from error
        if move not in board.legal_moves:
            raise ValueError(f"move {index} is illegal in this position: {uci}")
        board.push(move)
    return board


def get_evaluator() -> ModelEvaluator:
    global _evaluator
    if _evaluator is not None:
        return _evaluator

    with _load_lock:
        if _evaluator is None:
            threads = max(1, int(os.getenv("KIBITZER_TORCH_THREADS", "1")))
            torch.set_num_threads(threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            LOGGER.info("loading checkpoint %s with %s torch thread(s)", CHECKPOINT, threads)
            _evaluator = ModelEvaluator.from_checkpoint(CHECKPOINT, device="cpu")
            _evaluator.evaluate(chess.Board())
            LOGGER.info("checkpoint ready with %s parameters", _evaluator.model.num_params())
    return _evaluator


def calculate_move(payload: MoveRequest, evaluator: ModelEvaluator) -> MoveResponse:
    board = build_board(payload.initial_fen, payload.moves)
    if board.is_game_over(claim_draw=False):
        raise ValueError("the supplied game is already over")

    started = time.perf_counter()
    with _search_lock:
        result = puct_search(
            board,
            evaluator,
            simulations=payload.simulations,
            batch_size=BATCH_SIZE,
            claim_draw=False,
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    san = board.san(result.move)
    total_visits = sum(result.visits.values()) or 1
    top_moves = [
        VisitLine(move=move.uci(), visits=visits, share=visits / total_visits)
        for move, visits in sorted(
            result.visits.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:8]
    ]
    LOGGER.info(
        "search_completed ply=%s simulations=%s move=%s elapsed_ms=%s value=%.4f",
        len(payload.moves) + 1,
        payload.simulations,
        result.move.uci(),
        elapsed_ms,
        result.root_value,
    )
    board.push(result.move)
    return MoveResponse(
        move=result.move.uci(),
        san=san,
        fen=board.fen(),
        value=result.root_value,
        simulations=payload.simulations,
        elapsed_ms=elapsed_ms,
        top_moves=top_moves,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_evaluator()
    yield


app = FastAPI(title="Kibitzer inference", version="1.0.0", lifespan=lifespan)
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "KIBITZER_ALLOWED_ORIGINS",
        "https://kibitzer.vercel.app,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str | int | bool]:
    evaluator = get_evaluator()
    return {
        "status": "ok",
        "model": "kibitzer-tactical-repair",
        "parameters": evaluator.model.num_params(),
        "ready": True,
    }


@app.post("/move", response_model=MoveResponse)
def move(payload: MoveRequest) -> MoveResponse:
    try:
        return calculate_move(payload, get_evaluator())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        LOGGER.exception("move search failed")
        raise HTTPException(status_code=500, detail="Kibitzer could not finish this search") from error
