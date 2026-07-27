import {
  FlipHorizontal2,
  Pause,
  Play,
  SkipBack,
  SkipForward,
  StepBack,
  StepForward,
} from "lucide-react";
import { Chessboard } from "react-chessboard";
import type { GameTrace } from "@/lib/pgn";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";

const INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const speeds = [0.5, 1, 2, 4];

type GameBoardProps = {
  game: GameTrace;
  activePly: number;
  playing: boolean;
  speed: number;
  orientation: "white" | "black";
  onSeek: (index: number) => void;
  onPlayingChange: (playing: boolean) => void;
  onSpeedChange: (speed: number) => void;
  onOrientationChange: (orientation: "white" | "black") => void;
};

export function GameBoard(props: GameBoardProps) {
  const {
    game,
    activePly,
    playing,
    speed,
    orientation,
    onSeek,
    onPlayingChange,
    onSpeedChange,
    onOrientationChange,
  } = props;
  const ply = activePly >= 0 ? game.plies[activePly] : undefined;
  const position = ply?.fen ?? INITIAL_FEN;
  const atEnd = activePly >= game.plies.length - 1;
  const squareStyles = ply
    ? {
        [ply.from]: { boxShadow: "inset 0 0 0 9999px hsl(var(--kibitzer) / 0.18)" },
        [ply.to]: { boxShadow: "inset 0 0 0 9999px hsl(var(--kibitzer) / 0.32)" },
      }
    : {};

  return (
    <section className="min-w-0">
      <header className="mb-4 flex flex-wrap items-start justify-between gap-4 border-b border-divider pb-4">
        <div>
          <p className="font-serif text-base font-semibold">{game.headers.White ?? "?"}</p>
          <p className="mt-1 font-mono text-[9px] uppercase tracking-[0.1em] text-text-tertiary">white</p>
        </div>
        <div className="text-center">
          <p className="font-mono text-xl font-semibold text-kibitzer">{game.headers.Result ?? "*"}</p>
          <p className="mt-1 font-mono text-[9px] uppercase tracking-[0.1em] text-text-tertiary">
            round {game.headers.Round ?? "?"}
          </p>
        </div>
        <div className="text-right">
          <p className="font-serif text-base font-semibold">{game.headers.Black ?? "?"}</p>
          <p className="mt-1 font-mono text-[9px] uppercase tracking-[0.1em] text-text-tertiary">black</p>
        </div>
      </header>

      <div className="mx-auto w-full max-w-[620px] border border-divider bg-surface p-2 shadow-[0_22px_60px_rgba(0,0,0,0.08)]">
        <Chessboard
          options={{
            id: `trace-${game.id}`,
            position,
            boardOrientation: orientation,
            allowDragging: false,
            allowDrawingArrows: false,
            animationDurationInMs: 180,
            squareStyles,
            darkSquareStyle: { backgroundColor: "#8f6f52" },
            lightSquareStyle: { backgroundColor: "#eadfc8" },
            boardStyle: { width: "100%", height: "auto", boxShadow: "none" },
          }}
        />
      </div>

      <div className="mx-auto mt-5 max-w-[620px] border-y border-divider py-4">
        <div className="flex items-center gap-3">
          <span className="w-12 font-mono text-[9px] text-text-tertiary">
            {activePly < 0 ? "start" : `${activePly + 1}/${game.plies.length}`}
          </span>
          <Slider
            aria-label="Current move"
            min={-1}
            max={Math.max(-1, game.plies.length - 1)}
            step={1}
            value={[activePly]}
            onValueChange={([value]) => onSeek(value)}
          />
        </div>

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" aria-label="First move" onClick={() => onSeek(-1)} disabled={activePly < 0}>
              <SkipBack className="size-4" />
            </Button>
            <Button variant="ghost" size="icon" aria-label="Previous move" onClick={() => onSeek(Math.max(-1, activePly - 1))} disabled={activePly < 0}>
              <StepBack className="size-4" />
            </Button>
            <Button
              variant="accent"
              size="icon"
              aria-label={playing ? "Pause playback" : "Play game"}
              onClick={() => {
                if (!playing && atEnd) onSeek(-1);
                onPlayingChange(!playing);
              }}
              disabled={game.plies.length === 0}
            >
              {playing ? <Pause className="size-4" /> : <Play className="size-4" />}
            </Button>
            <Button variant="ghost" size="icon" aria-label="Next move" onClick={() => onSeek(Math.min(game.plies.length - 1, activePly + 1))} disabled={atEnd}>
              <StepForward className="size-4" />
            </Button>
            <Button variant="ghost" size="icon" aria-label="Last move" onClick={() => onSeek(game.plies.length - 1)} disabled={atEnd}>
              <SkipForward className="size-4" />
            </Button>
          </div>

          <div className="flex items-center gap-1">
            {speeds.map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => onSpeedChange(value)}
                className={cn(
                  "rounded px-2 py-1.5 font-mono text-[9px] text-text-tertiary hover:bg-surface-hover hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  speed === value && "bg-surface-active text-foreground",
                )}
              >
                {value}x
              </button>
            ))}
            <Button
              variant="ghost"
              size="icon"
              aria-label="Flip board"
              onClick={() => onOrientationChange(orientation === "white" ? "black" : "white")}
            >
              <FlipHorizontal2 className="size-4" />
            </Button>
          </div>
        </div>
      </div>

      <div className="mx-auto mt-4 grid max-w-[620px] gap-3 sm:grid-cols-[1fr_auto]">
        <p className="min-h-6 text-xs leading-6 text-text-secondary">
          {ply?.comment ?? "Select play or use the arrow keys to step through the game."}
        </p>
        <p className="font-mono text-[9px] uppercase tracking-[0.1em] text-text-tertiary">
          {ply ? `${ply.from} → ${ply.to}` : "initial position"}
        </p>
      </div>
    </section>
  );
}
