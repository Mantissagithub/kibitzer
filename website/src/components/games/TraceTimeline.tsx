import { useEffect, useRef } from "react";
import type { GameTrace, TracePly } from "@/lib/pgn";
import { cn } from "@/lib/utils";

type TraceTimelineProps = {
  game: GameTrace;
  activePly: number;
  onSeek: (index: number) => void;
};

type Signal = { evaluation?: string; depth?: string; time?: string; book: boolean };

function parseSignal(comment?: string): Signal {
  if (!comment) return { book: false };
  const book = /\bbook\b/i.test(comment);
  const evaluation = comment.match(/(?:^|\s)([+-]?(?:M\d+|\d+(?:\.\d+)?))\/(?:\d+)/i)?.[1];
  const depth = comment.match(/\/(\d+)/)?.[1];
  const time = comment.match(/(\d+(?:\.\d+)?)s\b/i)?.[1];
  return { evaluation, depth, time, book };
}

function PlyRow({ ply, active, onClick }: { ply: TracePly; active: boolean; onClick: () => void }) {
  const signal = parseSignal(ply.comment);
  return (
    <button
      type="button"
      onClick={onClick}
      data-active={active || undefined}
      className={cn(
        "grid w-full grid-cols-[2.6rem_2.8rem_1fr] items-start gap-2 border-b border-divider px-3 py-2.5 text-left hover:bg-surface-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
        active && "bg-surface-active",
      )}
    >
      <span className="font-mono text-[9px] text-text-tertiary">
        {ply.moveNumber}{ply.color === "w" ? "." : "..."}
      </span>
      <span className="font-mono text-[11px] font-medium text-foreground">{ply.san}</span>
      <span className="min-w-0">
        {signal.book ? <span className="font-mono text-[9px] uppercase text-text-tertiary">book</span> : null}
        {signal.evaluation ? (
          <span className="font-mono text-[9px] text-kibitzer">
            {signal.evaluation} {signal.depth ? <span className="text-text-tertiary">d{signal.depth}</span> : null}
          </span>
        ) : null}
        {signal.time ? <span className="ml-2 font-mono text-[9px] text-text-tertiary">{signal.time}s</span> : null}
        {ply.comment && !signal.book && !signal.evaluation && !signal.time ? (
          <span className="block truncate text-[9px] text-text-tertiary">{ply.comment}</span>
        ) : null}
      </span>
    </button>
  );
}

export function TraceTimeline({ game, activePly, onSeek }: TraceTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    const active = container?.querySelector<HTMLElement>("[data-active=true]");
    if (!container || !active) return;

    const activeTop = active.offsetTop;
    const activeBottom = activeTop + active.offsetHeight;
    const visibleBottom = container.scrollTop + container.clientHeight;

    if (activeTop < container.scrollTop) {
      container.scrollTo({ top: activeTop, behavior: "smooth" });
    } else if (activeBottom > visibleBottom) {
      container.scrollTo({ top: activeBottom - container.clientHeight, behavior: "smooth" });
    }
  }, [activePly]);

  return (
    <section className="flex min-h-[420px] flex-col border border-divider bg-surface/45 lg:h-[calc(100vh-8.5rem)] lg:min-h-0">
      <header className="border-b border-divider p-4">
        <p className="eyebrow">move trace</p>
        <div className="mt-3 flex items-baseline justify-between gap-3">
          <h2 className="truncate text-sm font-semibold">
            {game.headers.Opening ?? "Recorded game"}
          </h2>
          <span className="font-mono text-[10px] text-text-tertiary">{game.plies.length} ply</span>
        </div>
        {game.headers.Variation ? (
          <p className="mt-1 truncate text-[10px] text-text-tertiary">{game.headers.Variation}</p>
        ) : null}
      </header>
      <div ref={containerRef} className="min-h-0 flex-1 overflow-y-auto">
        <button
          type="button"
          onClick={() => onSeek(-1)}
          data-active={activePly === -1 || undefined}
          className={cn(
            "w-full border-b border-divider px-3 py-3 text-left font-mono text-[9px] uppercase tracking-[0.1em] text-text-tertiary hover:bg-surface-hover",
            activePly === -1 && "bg-surface-active text-foreground",
          )}
        >
          initial position
        </button>
        {game.plies.map((ply) => (
          <PlyRow
            key={`${ply.index}-${ply.san}`}
            ply={ply}
            active={ply.index === activePly}
            onClick={() => onSeek(ply.index)}
          />
        ))}
      </div>
    </section>
  );
}
