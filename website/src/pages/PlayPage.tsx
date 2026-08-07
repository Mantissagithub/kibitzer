import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { Activity, Clock3, Cpu, Flag, Play, RotateCcw } from "lucide-react";
import { Chessboard } from "react-chessboard";
import type { Square } from "chess.js";
import { Button } from "@/components/ui/button";
import {
  inferenceApiRoot,
  requestModelMove,
  type ModelMove,
  type SearchBudget,
} from "@/lib/inference";
import {
  applyBoardMove,
  applyUciMove,
  checkmateResult,
  formatHumanClock,
  gameFromMoves,
  gameStatus,
  HUMAN_TIME_LIMIT_MS,
  isHumanLowTime,
  shouldRunHumanClock,
  turnFor,
  type HumanColor,
} from "@/lib/play";
import { cn } from "@/lib/utils";

const searchBudgets: Array<{
  value: SearchBudget;
  label: string;
  note: string;
}> = [
  { value: 64, label: "quick", note: "about 0.4s local" },
  { value: 128, label: "balanced", note: "about 0.7s local" },
  { value: 512, label: "full search", note: "2581 protocol" },
];

type TimeControl = "ten-minutes" | "unlimited";

const timeControls: Array<{ value: TimeControl; label: string }> = [
  { value: "ten-minutes", label: "10 minutes" },
  { value: "unlimited", label: "unlimited" },
];

function formatValue(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

type SearchRecord = ModelMove & {
  ply: number;
  positionFen: string;
};

export default function PlayPage() {
  const [moves, setMoves] = useState<string[]>([]);
  const [humanColor, setHumanColor] = useState<HumanColor>("white");
  const [simulations, setSimulations] = useState<SearchBudget>(128);
  const [timeControl, setTimeControl] = useState<TimeControl>("ten-minutes");
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchLog, setSearchLog] = useState<SearchRecord[]>([]);
  const [selectedSquare, setSelectedSquare] = useState<string | null>(null);
  const [humanTimeMs, setHumanTimeMs] = useState(HUMAN_TIME_LIMIT_MS);
  const [lostOnTime, setLostOnTime] = useState(false);
  const [gameStarted, setGameStarted] = useState(false);
  const [resigned, setResigned] = useState(false);
  const requestGeneration = useRef(0);
  const requestController = useRef<AbortController | null>(null);
  const humanTimeRef = useRef(HUMAN_TIME_LIMIT_MS);
  const humanDeadlineRef = useRef<number | null>(null);

  const game = useMemo(() => gameFromMoves(moves), [moves]);
  const history = game.history({ verbose: true });
  const lastMove = history.at(-1);
  const lastSearch = searchLog.at(-1) ?? null;
  const humanTurn = turnFor(humanColor);
  const timedGame = timeControl === "ten-minutes";
  const humanTurnActive = (
    gameStarted
    && !resigned
    && shouldRunHumanClock(game, humanColor, thinking, lostOnTime)
  );
  const clockRunning = timedGame && humanTurnActive;
  const lowTime = timedGame && gameStarted && !resigned && isHumanLowTime(humanTimeMs);
  const canMove = humanTurnActive && (!timedGame || humanTimeMs > 0);
  const status = resigned
    ? "You resigned"
    : !gameStarted
      ? "Ready when you are"
      : lostOnTime
    ? "You lost on time"
    : thinking
      ? "Kibitzer is searching"
      : gameStatus(game);
  const checkmate = checkmateResult(game, humanColor);
  const endNotice = resigned
    ? {
        heading: "You resigned.",
        detail: `Kibitzer wins as ${humanColor === "white" ? "Black" : "White"}.`,
      }
    : lostOnTime
    ? {
        heading: "Time expired.",
        detail: "You lost on time. Kibitzer is untimed.",
      }
    : checkmate
      ? {
          heading: "Checkmate.",
          detail: `${checkmate.title} as ${checkmate.winnerName}.`,
        }
      : null;

  useEffect(() => {
    if (!clockRunning) {
      humanDeadlineRef.current = null;
      return;
    }

    const deadline = Date.now() + humanTimeRef.current;
    humanDeadlineRef.current = deadline;
    const interval = window.setInterval(() => {
      if (humanDeadlineRef.current !== deadline) return;
      const remaining = Math.max(0, deadline - Date.now());
      humanTimeRef.current = remaining;
      setHumanTimeMs(remaining);
      if (remaining === 0) {
        humanDeadlineRef.current = null;
        setLostOnTime(true);
      }
    }, 250);

    return () => {
      window.clearInterval(interval);
      if (humanDeadlineRef.current === deadline) humanDeadlineRef.current = null;
    };
  }, [clockRunning]);

  const squareStyles: Record<string, CSSProperties> = {};
  const lastMoveArrows = lastMove
    ? [{
        startSquare: lastMove.from,
        endSquare: lastMove.to,
        color: "hsl(var(--kibitzer) / 0.72)",
      }]
    : [];
  if (lastMove) {
    squareStyles[lastMove.from] = {
      boxShadow: "inset 0 0 0 4px hsl(var(--kibitzer) / 0.9), inset 0 0 0 9999px hsl(var(--kibitzer) / 0.1)",
    };
    squareStyles[lastMove.to] = {
      boxShadow: "inset 0 0 0 9999px hsl(var(--kibitzer) / 0.34)",
    };
  }
  if (selectedSquare) {
    squareStyles[selectedSquare] = { boxShadow: "inset 0 0 0 3px hsl(var(--kibitzer))" };
  }

  function cancelPendingRequest() {
    requestGeneration.current += 1;
    requestController.current?.abort();
    requestController.current = null;
    setThinking(false);
  }

  function commitHumanClock() {
    const deadline = humanDeadlineRef.current;
    if (deadline === null) return humanTimeRef.current;

    const remaining = Math.max(0, deadline - Date.now());
    humanDeadlineRef.current = null;
    humanTimeRef.current = remaining;
    setHumanTimeMs(remaining);
    if (remaining === 0) setLostOnTime(true);
    return remaining;
  }

  async function askKibitzer(positionMoves: string[]) {
    const generation = requestGeneration.current + 1;
    requestGeneration.current = generation;
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    setThinking(true);
    setError(null);

    try {
      const result = await requestModelMove(positionMoves, simulations, controller.signal);
      if (requestGeneration.current !== generation) return;
      const applied = applyUciMove(positionMoves, result.move);
      if (!applied) throw new Error(`The model returned an illegal move: ${result.move}`);
      setMoves(applied.moves);
      setSearchLog((records) => [
        ...records,
        {
          ...result,
          ply: applied.moves.length,
          positionFen: gameFromMoves(positionMoves).fen(),
        },
      ]);
    } catch (requestError) {
      if (requestGeneration.current !== generation) return;
      if (requestError instanceof DOMException && requestError.name === "AbortError") return;
      setError(requestError instanceof Error ? requestError.message : String(requestError));
    } finally {
      if (requestGeneration.current === generation) {
        setThinking(false);
        requestController.current = null;
      }
    }
  }

  function resetGame() {
    cancelPendingRequest();
    humanDeadlineRef.current = null;
    humanTimeRef.current = HUMAN_TIME_LIMIT_MS;
    setMoves([]);
    setSelectedSquare(null);
    setError(null);
    setSearchLog([]);
    setHumanTimeMs(HUMAN_TIME_LIMIT_MS);
    setLostOnTime(false);
    setGameStarted(false);
    setResigned(false);
  }

  function startGame() {
    if (gameStarted) return;
    setGameStarted(true);
    setError(null);
    if (humanColor === "black") void askKibitzer([]);
  }

  function resignGame() {
    if (!gameStarted || resigned || lostOnTime || game.isGameOver()) return;
    cancelPendingRequest();
    humanDeadlineRef.current = null;
    setSelectedSquare(null);
    setResigned(true);
  }

  function chooseColor(color: HumanColor) {
    setHumanColor(color);
    resetGame();
  }

  function chooseTimeControl(control: TimeControl) {
    setTimeControl(control);
    resetGame();
  }

  function makeHumanMove(from: string, to: string) {
    if (!canMove) return false;
    const applied = applyBoardMove(moves, from, to);
    if (!applied) return false;
    if (timedGame && commitHumanClock() === 0) return false;

    setMoves(applied.moves);
    setSelectedSquare(null);
    setError(null);
    const nextGame = gameFromMoves(applied.moves);
    if (!nextGame.isGameOver()) void askKibitzer(applied.moves);
    return true;
  }

  function handleSquareClick(square: string) {
    if (!canMove) return;
    const piece = game.get(square as Square);
    if (!selectedSquare) {
      if (piece?.color === humanTurn) setSelectedSquare(square);
      return;
    }
    if (selectedSquare === square) {
      setSelectedSquare(null);
      return;
    }
    if (makeHumanMove(selectedSquare, square)) return;
    setSelectedSquare(piece?.color === humanTurn ? square : null);
  }

  return (
    <div className="pb-20 pt-10 lg:pt-14">
      <header className="page-shell mb-8 grid gap-6 border-b border-divider pb-8 lg:grid-cols-[0.72fr_1.45fr]">
        <p className="eyebrow">live policy · your board · measured search</p>
        <div>
          <h1 className="font-serif text-4xl font-semibold tracking-[-0.045em] sm:text-6xl">
            Play the model, not the rating card.
          </h1>
          <p className="mt-4 max-w-2xl text-sm leading-7 text-text-secondary">
            You make a legal move. Kibitzer rebuilds the position and spends the
            search budget you selected before answering. Full search matches the
            512-simulation tournament protocol. The faster modes trade that claim
            for response time. Choose a ten-minute clock or unlimited game time.
            Kibitzer is always untimed, so wall time can run longer.
          </p>
        </div>
      </header>

      <main className="page-shell grid min-w-0 gap-6 lg:grid-cols-[minmax(0,1.25fr)_360px] lg:items-start">
        <section className="min-w-0">
          <div className="border border-divider bg-surface/45 p-2 sm:p-3">
            <div className="mb-3 flex items-center justify-between px-1 font-mono text-[10px] uppercase tracking-[0.12em] text-text-tertiary">
              <span>Kibitzer</span>
              <span>{humanColor === "white" ? "black" : "white"}</span>
            </div>
            {lowTime ? (
              <div
                className="mb-2 flex items-center justify-between bg-kibitzer px-3 py-2 text-white"
                role="status"
                aria-live="polite"
                aria-atomic="true"
              >
                <span className="font-mono text-[9px] uppercase tracking-[0.14em]">
                  one minute left
                </span>
                <strong className="font-mono text-sm tabular-nums">
                  {formatHumanClock(humanTimeMs)}
                </strong>
              </div>
            ) : null}
            <div className="relative">
              <Chessboard
                options={{
                  id: "play-kibitzer",
                  position: game.fen(),
                  boardOrientation: humanColor,
                  animationDurationInMs: 180,
                  allowDrawingArrows: false,
                  allowDragging: canMove,
                  arrows: lastMoveArrows,
                  canDragPiece: ({ piece }) => canMove && piece.pieceType.startsWith(humanTurn),
                  onPieceDrop: ({ sourceSquare, targetSquare }) => (
                    targetSquare ? makeHumanMove(sourceSquare, targetSquare) : false
                  ),
                  onSquareClick: ({ square }) => handleSquareClick(square),
                  squareStyles,
                  darkSquareStyle: { backgroundColor: "#8f6f52" },
                  lightSquareStyle: { backgroundColor: "#eadfc8" },
                  boardStyle: { width: "100%", height: "auto", boxShadow: "none" },
                }}
              />

              {!gameStarted && !endNotice ? (
                <div className="absolute inset-x-2 bottom-2 z-20 border border-background/40 bg-foreground p-4 text-background shadow-[0_18px_50px_hsl(0_0%_0%/0.35)] sm:inset-x-4 sm:bottom-4 sm:p-5">
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                    <div>
                      <p className="font-mono text-[9px] uppercase tracking-[0.2em] opacity-60">
                        ready to play
                      </p>
                      <p className="mt-1 font-serif text-4xl font-semibold tracking-[-0.045em] sm:text-5xl">
                        Your move when you start.
                      </p>
                      <p className="mt-2 text-sm font-medium">
                        {timedGame
                          ? "The board and your 10:00 clock are waiting."
                          : "The board is ready with unlimited time."}
                      </p>
                    </div>
                    <button
                      type="button"
                      className="inline-flex items-center justify-center gap-2 border border-background bg-background px-4 py-2 font-mono text-[10px] uppercase tracking-[0.12em] text-foreground transition-opacity hover:opacity-85 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-background focus-visible:ring-offset-2 focus-visible:ring-offset-foreground"
                      onClick={startGame}
                    >
                      <Play className="size-3" fill="currentColor" />
                      start game
                    </button>
                  </div>
                </div>
              ) : endNotice ? (
                <div
                  className="absolute inset-x-2 bottom-2 z-20 border border-background/40 bg-foreground p-4 text-background shadow-[0_18px_50px_hsl(0_0%_0%/0.35)] sm:inset-x-4 sm:bottom-4 sm:p-5"
                  role="alert"
                  aria-live="assertive"
                  aria-atomic="true"
                >
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                    <div>
                      <p className="font-mono text-[9px] uppercase tracking-[0.2em] opacity-60">
                        game over
                      </p>
                      <p className="mt-1 font-serif text-4xl font-semibold tracking-[-0.045em] sm:text-5xl">
                        {endNotice.heading}
                      </p>
                      <p className="mt-2 text-sm font-medium">
                        {endNotice.detail}
                      </p>
                    </div>
                    <button
                      type="button"
                      className="border border-background bg-background px-4 py-2 font-mono text-[10px] uppercase tracking-[0.12em] text-foreground transition-opacity hover:opacity-85 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-background focus-visible:ring-offset-2 focus-visible:ring-offset-foreground"
                      onClick={() => resetGame()}
                    >
                      play again
                    </button>
                  </div>
                </div>
              ) : null}
            </div>
            <div className="mt-3 flex items-center justify-between px-1 font-mono text-[10px] uppercase tracking-[0.12em] text-text-tertiary">
              <span>you</span>
              <span className={cn(lowTime && "font-semibold text-kibitzer")}>
                {humanColor} · {timedGame ? formatHumanClock(humanTimeMs) : "unlimited"}
              </span>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-[1fr_auto] items-end border border-divider bg-surface/40 px-4 py-4">
            <div>
              <p className="eyebrow">latest board move</p>
              {lastMove ? (
                <p
                  className="mt-2 flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.08em]"
                  aria-label={`Previous move from ${lastMove.from} to ${lastMove.to}`}
                >
                  <span className="border border-kibitzer/50 px-2 py-1 text-text-secondary">
                    from <strong className="text-foreground">{lastMove.from}</strong>
                  </span>
                  <span aria-hidden="true" className="text-kibitzer">→</span>
                  <span className="bg-kibitzer px-2 py-1 text-white">
                    to <strong>{lastMove.to}</strong>
                  </span>
                  <span className="text-[9px] text-text-tertiary">ply {history.length}</span>
                </p>
              ) : (
                <p className="mt-2 font-mono text-[10px] text-text-tertiary">
                  the board is ready
                </p>
              )}
            </div>
            <strong className="font-serif text-3xl font-semibold tracking-[-0.04em] text-foreground">
              {lastMove?.san ?? "waiting"}
            </strong>
          </div>

          <div
            className={cn(
              "mt-4 grid grid-cols-[auto_1fr_auto] items-center gap-3 border-y border-divider px-1 py-4",
              thinking && "text-kibitzer",
            )}
            role="status"
            aria-live="polite"
          >
            <span className={cn("size-2 rounded-full bg-text-tertiary", thinking && "animate-pulse bg-kibitzer")} />
            <div>
              <p className="text-sm font-medium text-foreground">{status}</p>
              <p className="mt-1 font-mono text-[9px] uppercase tracking-[0.1em] text-text-tertiary">
                {thinking ? `${simulations} leaf evaluations in progress` : `${moves.length} plies recorded`}
              </p>
            </div>
            <span className="font-mono text-[10px] text-text-tertiary">
              {lastSearch ? `${lastSearch.elapsed_ms} ms` : "live"}
            </span>
          </div>

          {error ? (
            <div className="mt-4 border-l-2 border-kibitzer bg-surface px-4 py-3" role="alert">
              <p className="text-xs leading-6 text-text-secondary">{error}</p>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="mt-2 px-0 font-mono text-[10px] uppercase tracking-[0.1em]"
                onClick={() => void askKibitzer(moves)}
                disabled={!gameStarted || thinking || resigned || lostOnTime || game.isGameOver() || game.turn() === humanTurn}
              >
                retry this position
              </Button>
            </div>
          ) : null}

          <div className="mt-6 border border-divider">
            <div className="flex items-center justify-between border-b border-divider bg-surface/55 px-4 py-3">
              <p className="eyebrow">board move record</p>
              <span className="font-mono text-[9px] text-text-tertiary">{history.length} plies</span>
            </div>
            <div className="max-h-64 overflow-y-auto p-4">
              {history.length ? (
                <ol className="grid grid-cols-[2rem_1fr_1fr] gap-x-3 gap-y-2 font-mono text-[11px]">
                  {Array.from({ length: Math.ceil(history.length / 2) }, (_, index) => (
                    <li key={index} className="contents">
                      <span className="text-text-tertiary">{index + 1}.</span>
                      <span>{history[index * 2]?.san ?? ""}</span>
                      <span>{history[index * 2 + 1]?.san ?? ""}</span>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="text-xs leading-6 text-text-tertiary">Move a piece to begin the record.</p>
              )}
            </div>
          </div>
        </section>

        <aside className="border border-divider bg-surface/40 lg:sticky lg:top-20">
          <div className="border-b border-divider p-4">
            <div className="flex items-center justify-between gap-3">
              <p className="eyebrow">game control</p>
              <Button type="button" variant="ghost" size="sm" className="h-7 px-2" onClick={() => resetGame()}>
                <RotateCcw className="size-3" /> new game
              </Button>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2" aria-label="Choose your color">
              {(["white", "black"] as HumanColor[]).map((color) => (
                <button
                  key={color}
                  type="button"
                  onClick={() => chooseColor(color)}
                  disabled={gameStarted}
                  className={cn(
                    "border border-divider px-3 py-2 font-mono text-[9px] uppercase tracking-[0.1em] transition-colors hover:bg-surface-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-45",
                    humanColor === color && "border-foreground bg-foreground text-background",
                  )}
                  aria-pressed={humanColor === color}
                >
                  play {color}
                </button>
              ))}
            </div>

            <div className="mt-4">
              <p className="eyebrow">game time</p>
              <div className="mt-3 grid grid-cols-2 gap-2" aria-label="Choose game time">
                {timeControls.map((control) => (
                  <button
                    key={control.value}
                    type="button"
                    onClick={() => chooseTimeControl(control.value)}
                    disabled={gameStarted}
                    className={cn(
                      "border border-divider px-3 py-2 font-mono text-[9px] uppercase tracking-[0.1em] transition-colors hover:bg-surface-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-45",
                      timeControl === control.value && "border-foreground bg-foreground text-background hover:bg-foreground",
                    )}
                    aria-pressed={timeControl === control.value}
                  >
                    {control.label}
                  </button>
                ))}
              </div>
            </div>

            <Button
              type="button"
              variant="outline"
              className="mt-2 w-full rounded-none font-mono text-[9px] uppercase tracking-[0.12em]"
              onClick={resignGame}
              disabled={!gameStarted || resigned || lostOnTime || game.isGameOver()}
            >
              <Flag className="size-3" />
              resign
            </Button>

            <div
              className={cn(
                "mt-4 border-y border-divider py-4",
                lowTime && "border-kibitzer",
              )}
            >
              <div className="flex items-center justify-between gap-3">
                <p className="eyebrow">your clock</p>
                <span
                  className={cn(
                    "font-mono text-[9px] uppercase tracking-[0.1em] text-text-tertiary",
                    lowTime && "font-semibold text-kibitzer",
                  )}
                >
                  {resigned
                    ? "stopped"
                    : !gameStarted
                      ? "ready"
                      : !timedGame
                        ? "no limit"
                        : lostOnTime
                          ? "expired"
                          : clockRunning
                            ? "running"
                            : "paused"}
                </span>
              </div>
              <div className="mt-2 flex items-end justify-between gap-3">
                <strong
                  className={cn(
                    "font-mono font-medium tabular-nums tracking-[-0.05em] text-foreground",
                    timedGame ? "text-4xl" : "text-2xl uppercase tracking-[-0.03em]",
                    lowTime && "text-kibitzer",
                  )}
                  aria-label={timedGame ? `${formatHumanClock(humanTimeMs)} remaining` : "Unlimited time"}
                >
                  {timedGame ? formatHumanClock(humanTimeMs) : "unlimited"}
                </strong>
                <span className="pb-1 font-mono text-[8px] uppercase tracking-[0.12em] text-text-tertiary">
                  human only
                </span>
              </div>
              <p className="mt-2 text-[10px] leading-5 text-text-tertiary">
                {timedGame
                  ? "Runs only on your turn. Kibitzer has no clock, and its search never uses your time."
                  : "No countdown. Play each move at your own pace."}
              </p>
            </div>

            <div className="mt-4 flex items-center justify-between gap-3">
              <p className="eyebrow">search budget</p>
              <Cpu className="size-3.5 text-text-tertiary" />
            </div>
            <div className="mt-3 grid grid-cols-3 gap-2">
              {searchBudgets.map((budget) => (
                <button
                  key={budget.value}
                  type="button"
                  onClick={() => setSimulations(budget.value)}
                  disabled={thinking}
                  className={cn(
                    "border border-divider px-2 py-2 text-left transition-colors hover:bg-surface-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-45",
                    simulations === budget.value && "border-kibitzer bg-kibitzer/[0.07]",
                  )}
                  aria-pressed={simulations === budget.value}
                >
                  <span className="block font-mono text-xs font-semibold text-foreground">{budget.value}</span>
                  <span className="mt-1 block truncate text-[9px] text-text-tertiary">{budget.label}</span>
                </button>
              ))}
            </div>
            <p className="mt-2 font-mono text-[8px] text-text-tertiary">
              {searchBudgets.find((budget) => budget.value === simulations)?.note}
            </p>

            <div className="mt-4 grid grid-cols-2 gap-3 border-t border-divider pt-4 font-mono text-[10px]">
              <div>
                <span className="flex items-center gap-1.5 uppercase tracking-[0.1em] text-text-tertiary">
                  <Activity className="size-3" /> model view
                </span>
                <strong className="mt-2 block text-lg font-medium text-foreground">
                  {lastSearch ? formatValue(lastSearch.value) : "n/a"}
                </strong>
              </div>
              <div>
                <span className="flex items-center gap-1.5 uppercase tracking-[0.1em] text-text-tertiary">
                  <Clock3 className="size-3" /> last search
                </span>
                <strong className="mt-2 block text-lg font-medium text-foreground">
                  {lastSearch ? `${lastSearch.elapsed_ms}ms` : "n/a"}
                </strong>
              </div>
            </div>
          </div>

          <div className="border-b border-divider">
            <div className="flex items-center justify-between border-b border-divider bg-surface/55 px-5 py-3">
              <p className="eyebrow">model search log</p>
              <span className="font-mono text-[9px] text-text-tertiary">{searchLog.length} records</span>
            </div>
            <div className="max-h-[34rem] overflow-y-auto" role="log" aria-live="polite">
              {thinking ? (
                <div className="flex items-center gap-3 border-b border-divider px-5 py-4 text-kibitzer">
                  <span className="size-2 animate-pulse rounded-full bg-kibitzer" />
                  <span className="font-mono text-[10px] uppercase tracking-[0.1em]">
                    searching {simulations} simulations
                  </span>
                </div>
              ) : null}
              {searchLog.length ? searchLog.slice().reverse().map((record) => (
                <article key={record.ply} className="border-b border-divider px-5 py-4 last:border-b-0">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="font-mono text-[9px] uppercase tracking-[0.1em] text-text-tertiary">
                        ply {record.ply} · {record.simulations} sims
                      </p>
                      <p className="mt-1 font-serif text-2xl font-semibold tracking-[-0.035em] text-foreground">
                        {record.san}
                      </p>
                      <p className="mt-1 font-mono text-[9px] text-text-tertiary">{record.move}</p>
                    </div>
                    <div className="text-right font-mono">
                      <p className="text-xs font-medium text-foreground">{formatValue(record.value)}</p>
                      <p className="mt-1 text-[9px] text-text-tertiary">{record.elapsed_ms} ms</p>
                    </div>
                  </div>

                  <details className="mt-3 border-t border-divider pt-3">
                    <summary className="cursor-pointer font-mono text-[9px] uppercase tracking-[0.1em] text-text-tertiary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                      position and root visits
                    </summary>
                    <p className="mt-3 break-all font-mono text-[8px] leading-4 text-text-tertiary">
                      {record.positionFen}
                    </p>
                    <div className="mt-3 space-y-2">
                      {record.top_moves.slice(0, 6).map((candidate) => (
                        <div key={candidate.move} className="grid grid-cols-[3.2rem_1fr_2.5rem] items-center gap-2">
                          <span className="font-mono text-[9px] text-foreground">{candidate.move}</span>
                          <span className="h-1 overflow-hidden bg-surface-active">
                            <span
                              className="block h-full bg-kibitzer"
                              style={{ width: `${Math.max(2, candidate.share * 100)}%` }}
                            />
                          </span>
                          <span className="text-right font-mono text-[8px] text-text-tertiary">
                            {Math.round(candidate.share * 100)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </details>
                </article>
              )) : (
                <p className="px-5 py-6 text-xs leading-6 text-text-tertiary">
                  Each Kibitzer reply will leave its search record here.
                </p>
              )}
            </div>
          </div>

          <div className="p-5">
            <p className="eyebrow">connection</p>
            <p className="mt-3 break-all font-mono text-[9px] leading-5 text-text-tertiary">
              {inferenceApiRoot}
            </p>
          </div>
        </aside>
      </main>
    </div>
  );
}
