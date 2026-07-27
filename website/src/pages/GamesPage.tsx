import { useEffect, useMemo, useState, type DragEvent } from "react";
import { GameBoard } from "@/components/games/GameBoard";
import { GameLibrary, kibitzerResult } from "@/components/games/GameLibrary";
import { PastePgn } from "@/components/games/PastePgn";
import { TraceTimeline } from "@/components/games/TraceTimeline";
import { parsePgnDocument, type GameTrace, type PgnParseIssue } from "@/lib/pgn";

function opponentFor(game: GameTrace) {
  const white = game.headers.White ?? "?";
  const black = game.headers.Black ?? "?";
  return white.toLowerCase().includes("kibitzer") ? black : white;
}

export default function GamesPage() {
  const [builtInGames, setBuiltInGames] = useState<GameTrace[]>([]);
  const [importedGames, setImportedGames] = useState<GameTrace[]>([]);
  const [issues, setIssues] = useState<PgnParseIssue[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activePly, setActivePly] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [orientation, setOrientation] = useState<"white" | "black">("white");
  const [search, setSearch] = useState("");
  const [resultFilter, setResultFilter] = useState("all");
  const [opponentFilter, setOpponentFilter] = useState("all");

  const allGames = useMemo(() => [...importedGames, ...builtInGames], [builtInGames, importedGames]);
  const opponents = useMemo(
    () => [...new Set(allGames.map(opponentFor))].sort((a, b) => a.localeCompare(b)),
    [allGames],
  );
  const filteredGames = useMemo(() => {
    const term = search.trim().toLowerCase();
    return allGames.filter((game) => {
      if (resultFilter !== "all" && kibitzerResult(game) !== resultFilter) return false;
      if (opponentFilter !== "all" && opponentFor(game) !== opponentFilter) return false;
      if (!term) return true;
      return Object.values(game.headers).join(" ").toLowerCase().includes(term);
    });
  }, [allGames, opponentFilter, resultFilter, search]);
  const selectedGame = filteredGames.find((game) => game.id === selectedId) ?? filteredGames[0] ?? null;

  useEffect(() => {
    let cancelled = false;
    fetch("/generated/official-elo-clean.pgn")
      .then((response) => {
        if (!response.ok) throw new Error(`Could not load built-in PGN (${response.status})`);
        return response.text();
      })
      .then((text) => {
        if (cancelled) return;
        const parsed = parsePgnDocument(text, "built-in");
        setBuiltInGames(parsed.games);
        setIssues(parsed.issues);
        setSelectedId(parsed.games[0]?.id ?? null);
      })
      .catch((error: unknown) => {
        if (!cancelled) setIssues([{ gameIndex: 0, message: error instanceof Error ? error.message : String(error) }]);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!playing || !selectedGame) return;
    if (activePly >= selectedGame.plies.length - 1) return;
    const timer = window.setTimeout(
      () => setActivePly((current) => {
        const next = Math.min(selectedGame.plies.length - 1, current + 1);
        if (next >= selectedGame.plies.length - 1) setPlaying(false);
        return next;
      }),
      1000 / speed,
    );
    return () => window.clearTimeout(timer);
  }, [activePly, playing, selectedGame, speed]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target?.matches("input, textarea, select, button")) return;
      if (!selectedGame) return;
      if (event.key === "ArrowLeft") setActivePly((current) => Math.max(-1, current - 1));
      else if (event.key === "ArrowRight") setActivePly((current) => Math.min(selectedGame.plies.length - 1, current + 1));
      else if (event.key === "Home") setActivePly(-1);
      else if (event.key === "End") setActivePly(selectedGame.plies.length - 1);
      else if (event.key === " ") {
        if (!playing && activePly >= selectedGame.plies.length - 1) setActivePly(-1);
        setPlaying((current) => !current);
      }
      else return;
      event.preventDefault();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activePly, playing, selectedGame]);

  function resetPlayback() {
    setActivePly(-1);
    setPlaying(false);
  }

  function importText(text: string) {
    const parsed = parsePgnDocument(text, "imported");
    setIssues(parsed.issues);
    setImportedGames((current) => [...parsed.games, ...current]);
    if (parsed.games[0]) {
      setSelectedId(parsed.games[0].id);
      resetPlayback();
    }
  }

  async function importFiles(files: FileList | File[]) {
    const texts = await Promise.all([...files].map((file) => file.text()));
    importText(texts.join("\n\n"));
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    if (event.dataTransfer.files.length) void importFiles(event.dataTransfer.files);
  }

  return (
    <div className="py-10">
      <header className="page-shell mb-8 grid gap-6 border-b border-divider pb-8 lg:grid-cols-[0.72fr_1.45fr]">
        <p className="eyebrow">171 clean games · local imports</p>
        <div>
          <h1 className="font-serif text-4xl font-semibold tracking-[-0.045em] sm:text-6xl">Game traces</h1>
          <p className="mt-4 max-w-2xl text-sm leading-7 text-text-secondary">
            Select a tournament game and let it unfold at your pace. Every
            comment in the PGN stays attached to the move that produced it, so
            time, evaluation, book play, and adjudication remain visible.
          </p>
        </div>
      </header>

      {issues.length > 0 ? (
        <div className="page-shell mb-5 border-l-2 border-kibitzer bg-surface px-4 py-3 text-xs text-text-secondary" role="status">
          Parsed what was valid. {issues.length} game{issues.length === 1 ? "" : "s"} could not be read: {issues[0].message}
        </div>
      ) : null}

      <PastePgn onImport={importText} />

      <div className="page-shell grid min-w-0 gap-5 lg:grid-cols-[270px_minmax(420px,1fr)_285px]">
        <GameLibrary
          games={filteredGames}
          selectedId={selectedGame?.id ?? null}
          search={search}
          resultFilter={resultFilter}
          opponentFilter={opponentFilter}
          opponents={opponents}
          loading={loading}
          onSearchChange={(value) => { setSearch(value); resetPlayback(); }}
          onResultFilterChange={(value) => { setResultFilter(value); resetPlayback(); }}
          onOpponentFilterChange={(value) => { setOpponentFilter(value); resetPlayback(); }}
          onSelect={(id) => { setSelectedId(id); resetPlayback(); }}
          onFiles={(files) => void importFiles(files)}
          onDrop={handleDrop}
        />

        {selectedGame ? (
          <>
            <GameBoard
              game={selectedGame}
              activePly={activePly}
              playing={playing}
              speed={speed}
              orientation={orientation}
              onSeek={(index) => { setActivePly(index); setPlaying(false); }}
              onPlayingChange={setPlaying}
              onSpeedChange={setSpeed}
              onOrientationChange={setOrientation}
            />
            <TraceTimeline
              game={selectedGame}
              activePly={activePly}
              onSeek={(index) => { setActivePly(index); setPlaying(false); }}
            />
          </>
        ) : (
          <div className="grid min-h-[460px] place-items-center border border-divider lg:col-span-2">
            <p className="font-mono text-[10px] uppercase tracking-[0.12em] text-text-tertiary">load a PGN to begin</p>
          </div>
        )}
      </div>
    </div>
  );
}
