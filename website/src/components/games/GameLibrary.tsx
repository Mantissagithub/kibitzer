import { Search, Upload } from "lucide-react";
import type { DragEvent } from "react";
import type { GameTrace } from "@/lib/pgn";
import { cn } from "@/lib/utils";

type GameLibraryProps = {
  games: GameTrace[];
  selectedId: string | null;
  search: string;
  resultFilter: string;
  opponentFilter: string;
  opponents: string[];
  loading: boolean;
  onSearchChange: (value: string) => void;
  onResultFilterChange: (value: string) => void;
  onOpponentFilterChange: (value: string) => void;
  onSelect: (id: string) => void;
  onFiles: (files: FileList | File[]) => void;
  onDrop: (event: DragEvent<HTMLLabelElement>) => void;
};

function kibitzerResult(game: GameTrace) {
  const { White = "", Result = "*" } = game.headers;
  const kibitzerIsWhite = White.toLowerCase().includes("kibitzer");
  if (Result === "1/2-1/2") return "draw";
  if ((Result === "1-0" && kibitzerIsWhite) || (Result === "0-1" && !kibitzerIsWhite)) return "win";
  if (Result === "1-0" || Result === "0-1") return "loss";
  return "unknown";
}

export function GameLibrary(props: GameLibraryProps) {
  const {
    games,
    selectedId,
    search,
    resultFilter,
    opponentFilter,
    opponents,
    loading,
    onSearchChange,
    onResultFilterChange,
    onOpponentFilterChange,
    onSelect,
    onFiles,
    onDrop,
  } = props;

  return (
    <aside className="flex min-h-0 flex-col border border-divider bg-surface/45 lg:h-[calc(100vh-8.5rem)]">
      <div className="border-b border-divider p-4">
        <div className="flex items-center justify-between gap-3">
          <p className="eyebrow">trace library</p>
          <span className="font-mono text-[10px] text-text-tertiary">{games.length} games</span>
        </div>
        <label className="mt-4 flex items-center gap-2 border-b border-divider pb-2 focus-within:border-foreground">
          <Search className="size-3.5 text-text-tertiary" />
          <span className="sr-only">Search games</span>
          <input
            value={search}
            onChange={(event) => onSearchChange(event.target.value)}
            placeholder="player, opening, round"
            className="min-w-0 flex-1 bg-transparent font-mono text-[11px] outline-none placeholder:text-text-tertiary"
          />
        </label>
        <div className="mt-3 grid grid-cols-2 gap-2">
          <select
            aria-label="Filter by result"
            value={resultFilter}
            onChange={(event) => onResultFilterChange(event.target.value)}
            className="min-w-0 border border-divider bg-background px-2 py-2 font-mono text-[10px] outline-none focus:border-foreground"
          >
            <option value="all">all results</option>
            <option value="win">wins</option>
            <option value="draw">draws</option>
            <option value="loss">losses</option>
          </select>
          <select
            aria-label="Filter by opponent"
            value={opponentFilter}
            onChange={(event) => onOpponentFilterChange(event.target.value)}
            className="min-w-0 border border-divider bg-background px-2 py-2 font-mono text-[10px] outline-none focus:border-foreground"
          >
            <option value="all">all opponents</option>
            {opponents.map((opponent) => <option key={opponent}>{opponent}</option>)}
          </select>
        </div>
      </div>

      <div className="min-h-[260px] flex-1 overflow-y-auto" role="listbox" aria-label="Available games">
        {loading ? (
          <p className="p-5 font-mono text-[10px] uppercase tracking-[0.12em] text-text-tertiary">parsing tournament</p>
        ) : games.length === 0 ? (
          <p className="p-5 text-xs leading-6 text-text-tertiary">No games match these filters.</p>
        ) : (
          games.map((game, index) => {
            const state = kibitzerResult(game);
            const selected = game.id === selectedId;
            return (
              <button
                key={game.id}
                type="button"
                role="option"
                aria-selected={selected}
                onClick={() => onSelect(game.id)}
                className={cn(
                  "grid w-full grid-cols-[2.1rem_1fr_auto] gap-2 border-b border-divider px-3 py-3 text-left transition-colors hover:bg-surface-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
                  selected && "bg-surface-active",
                )}
              >
                <span className="pt-0.5 font-mono text-[9px] text-text-tertiary">{String(index + 1).padStart(3, "0")}</span>
                <span className="min-w-0">
                  <span className="block truncate text-[11px] font-medium text-foreground">
                    {game.headers.White ?? "?"} / {game.headers.Black ?? "?"}
                  </span>
                  <span className="mt-1 block truncate font-mono text-[9px] text-text-tertiary">
                    {game.headers.Opening ?? game.headers.Event ?? "Imported game"}
                  </span>
                </span>
                <span
                  className={cn(
                    "font-mono text-[9px] uppercase",
                    state === "win" && "text-kibitzer",
                    state === "draw" && "text-text-secondary",
                    state === "loss" && "text-text-tertiary",
                  )}
                >
                  {state}
                </span>
              </button>
            );
          })
        )}
      </div>

      <label
        onDragOver={(event) => event.preventDefault()}
        onDrop={onDrop}
        className="m-3 flex cursor-pointer items-center gap-3 border border-dashed border-divider bg-background p-3 text-text-tertiary transition-colors hover:border-foreground hover:text-foreground focus-within:ring-2 focus-within:ring-ring"
      >
        <Upload className="size-4" />
        <span className="text-[10px] leading-4">
          <strong className="block font-mono font-medium uppercase tracking-[0.08em]">load local PGN</strong>
          file or drag and drop, nothing leaves the browser
        </span>
        <input
          type="file"
          accept=".pgn,application/x-chess-pgn,text/plain"
          multiple
          className="sr-only"
          onChange={(event) => event.target.files && onFiles(event.target.files)}
        />
      </label>
    </aside>
  );
}

export { kibitzerResult };
