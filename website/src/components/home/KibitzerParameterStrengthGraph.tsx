import { useState } from "react";

type PoolId = "ccrl" | "paper" | "lichess" | "llm" | "kibitzer" | "stockfish-proxy";
type SearchMode = "none" | "search" | "successor";

type StrengthPoint = {
  id: string;
  name: string;
  rating: number;
  params?: number;
  error?: number;
  pool: PoolId;
  poolLabel: string;
  suite: string;
  search: string;
  mode: SearchMode;
  sourceLabel: string;
  sourceUrl: string;
  note?: string;
  extra?: string;
  label?: string;
  xOffset?: number;
};

const CHESSBENCH_URL = "https://arxiv.org/abs/2402.04494";
const CCRL_URL = "https://computerchess.org.uk/ccrl/404FRC/cgi/engine_details.cgi?eng=Stockfish+18&match_length=30&print=Details";
const LLM_BENCH_URL = "https://chessbenchllm.onrender.com/";
const KIBITZER_LOG_URL = "https://github.com/Mantissagithub/kibitzer/blob/main/LOGBOOK.md#d67-measure-a-clean-official-tournament-elo";
const KARVONEN_URL = "https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html";
const CHESS_LLAMA_URL = "https://huggingface.co/lazy-guy12/chess-llama";

const points: StrengthPoint[] = [
  {
    id: "stockfish-18",
    name: "Stockfish 18",
    rating: 4103,
    pool: "ccrl",
    poolLabel: "CCRL engine pool",
    suite: "CCRL 40/2 FRC, 4 CPU, February 8, 2026 snapshot",
    search: "NNUE with alpha-beta search",
    mode: "search",
    sourceLabel: "CCRL 40/2 FRC",
    sourceUrl: CCRL_URL,
    xOffset: -14,
  },
  {
    id: "torch-v4",
    name: "Torch v4",
    rating: 4087,
    pool: "ccrl",
    poolLabel: "CCRL engine pool",
    suite: "CCRL 40/2 FRC, 4 CPU, February 8, 2026 snapshot",
    search: "NNUE with alpha-beta search",
    mode: "search",
    sourceLabel: "CCRL 40/2 FRC",
    sourceUrl: CCRL_URL,
    xOffset: 0,
  },
  {
    id: "reckless",
    name: "Reckless 0.8.0",
    rating: 4078,
    pool: "ccrl",
    poolLabel: "CCRL engine pool",
    suite: "CCRL 40/2 FRC, 4 CPU, February 8, 2026 snapshot",
    search: "NNUE with alpha-beta search",
    mode: "search",
    sourceLabel: "CCRL 40/2 FRC",
    sourceUrl: CCRL_URL,
    xOffset: 14,
  },
  {
    id: "plentychess",
    name: "PlentyChess 7.0.0",
    rating: 4053,
    pool: "ccrl",
    poolLabel: "CCRL engine pool",
    suite: "CCRL 40/2 FRC, 4 CPU, February 8, 2026 snapshot",
    search: "NNUE with alpha-beta search",
    mode: "search",
    sourceLabel: "CCRL 40/2 FRC",
    sourceUrl: CCRL_URL,
    xOffset: 28,
  },
  {
    id: "berserk",
    name: "Berserk 13",
    rating: 4015,
    pool: "ccrl",
    poolLabel: "CCRL engine pool",
    suite: "CCRL 40/2 FRC, 4 CPU, February 8, 2026 snapshot",
    search: "NNUE with alpha-beta search",
    mode: "search",
    sourceLabel: "CCRL 40/2 FRC",
    sourceUrl: CCRL_URL,
    xOffset: 42,
  },
  {
    id: "dragon",
    name: "Dragon 3.3",
    rating: 3994,
    pool: "ccrl",
    poolLabel: "CCRL engine pool",
    suite: "CCRL 40/2 FRC, 4 CPU, February 8, 2026 snapshot",
    search: "NNUE with alpha-beta search",
    mode: "search",
    sourceLabel: "CCRL 40/2 FRC",
    sourceUrl: CCRL_URL,
    xOffset: 56,
  },
  {
    id: "ethereal",
    name: "Ethereal 14.25",
    rating: 3969,
    pool: "ccrl",
    poolLabel: "CCRL engine pool",
    suite: "CCRL 40/2 FRC, 4 CPU, February 8, 2026 snapshot",
    search: "NNUE with alpha-beta search",
    mode: "search",
    sourceLabel: "CCRL 40/2 FRC",
    sourceUrl: CCRL_URL,
    xOffset: 70,
  },
  {
    id: "searchless-270m",
    name: "ChessBench 270M",
    label: "270M searchless",
    params: 270,
    rating: 2895,
    pool: "lichess",
    poolLabel: "Lichess human pool",
    suite: "Lichess blitz games against humans",
    search: "No tree search",
    mode: "none",
    sourceLabel: "Grandmaster-Level Chess Without Search",
    sourceUrl: CHESSBENCH_URL,
    extra: "The same checkpoint scored 2299 against Lichess bots and 95.4% on the paper's puzzle suite.",
  },
  {
    id: "stockfish16-50ms",
    name: "Stockfish 16 at 50 ms",
    rating: 2713,
    pool: "lichess",
    poolLabel: "Lichess bot pool",
    suite: "Lichess blitz games against bots",
    search: "Alpha-beta search, 50 ms per move",
    mode: "search",
    sourceLabel: "Grandmaster-Level Chess Without Search",
    sourceUrl: CHESSBENCH_URL,
    xOffset: -18,
  },
  {
    id: "kibitzer",
    name: "Kibitzer",
    label: "Kibitzer 15.2M",
    params: 15.223,
    rating: 2581.6,
    error: 102.3,
    pool: "kibitzer",
    poolLabel: "Kibitzer's Stockfish ladder",
    suite: "Cute Chess CLI gauntlet, Ordo anchored at SF-2500 = 2500, 171 rated games",
    search: "Policy-value network with 512-simulation PUCT",
    mode: "search",
    sourceLabel: "Kibitzer Logbook, D67",
    sourceUrl: KIBITZER_LOG_URL,
    note: "This is the complete network-plus-search player, not the raw checkpoint.",
  },
  {
    id: "alphazero-search",
    name: "AlphaZero",
    label: "AlphaZero",
    params: 27.6,
    rating: 2470,
    error: 16,
    pool: "paper",
    poolLabel: "ChessBench paper tournament",
    suite: "22,000-game round robin, 400 games per pairing, BayesElo",
    search: "400 MCTS simulations",
    mode: "search",
    sourceLabel: "Grandmaster-Level Chess Without Search",
    sourceUrl: CHESSBENCH_URL,
    extra: "The policy-only and value-only variants scored 1777 and 1992 in the same tournament.",
  },
  {
    id: "lc0-search",
    name: "Lc0 T82",
    rating: 2858,
    error: 20,
    pool: "paper",
    poolLabel: "ChessBench paper tournament",
    suite: "22,000-game round robin, 400 games per pairing, BayesElo",
    search: "400 MCTS simulations",
    mode: "search",
    sourceLabel: "Grandmaster-Level Chess Without Search",
    sourceUrl: CHESSBENCH_URL,
    extra: "It scored 2620 against Lichess bots and 99.6% on the paper's puzzle suite.",
    xOffset: -4,
  },
  {
    id: "searchless-136m",
    name: "ChessBench 136M",
    label: "136M searchless",
    params: 136,
    rating: 2156,
    pool: "lichess",
    poolLabel: "Lichess bot pool",
    suite: "Lichess blitz games against bots",
    search: "No tree search",
    mode: "none",
    sourceLabel: "Grandmaster-Level Chess Without Search",
    sourceUrl: CHESSBENCH_URL,
    extra: "Its rating in the paper's internal tournament was 2259 +/- 16.",
  },
  {
    id: "searchless-9m",
    name: "ChessBench 9M",
    label: "9M searchless",
    params: 9,
    rating: 2054,
    pool: "lichess",
    poolLabel: "Lichess bot pool",
    suite: "Lichess blitz games against bots",
    search: "No tree search",
    mode: "none",
    sourceLabel: "Grandmaster-Level Chess Without Search",
    sourceUrl: CHESSBENCH_URL,
    extra: "Its rating in the paper's internal tournament was 2025 +/- 18.",
  },
  {
    id: "eubos",
    name: "Eubos",
    rating: 2211,
    pool: "llm",
    poolLabel: "LLM Chess Benchmark",
    suite: "Lichess classical rating, Glicko-2, 29 games",
    search: "Traditional engine search",
    mode: "search",
    sourceLabel: "LLM Chess Benchmark leaderboard",
    sourceUrl: LLM_BENCH_URL,
    xOffset: 8,
  },
  {
    id: "gemini-31-pro",
    name: "Gemini 3.1 Pro, high",
    rating: 2066,
    error: 85,
    pool: "llm",
    poolLabel: "LLM Chess Benchmark",
    suite: "Lichess classical rating, Glicko-2, 18 games",
    search: "No external chess search",
    mode: "none",
    sourceLabel: "LLM Chess Benchmark leaderboard",
    sourceUrl: LLM_BENCH_URL,
    xOffset: 22,
  },
  {
    id: "maia-1900",
    name: "Maia-1900",
    rating: 1816,
    pool: "llm",
    poolLabel: "LLM Chess Benchmark",
    suite: "Lichess classical anchor, 206 games",
    search: "Policy network without tree search",
    mode: "none",
    sourceLabel: "LLM Chess Benchmark leaderboard",
    sourceUrl: LLM_BENCH_URL,
    xOffset: 36,
  },
  {
    id: "gpt35-lichess",
    name: "GPT-3.5 Turbo Instruct",
    rating: 1755,
    pool: "lichess",
    poolLabel: "Lichess bot pool",
    suite: "Lichess blitz games against bots",
    search: "No external chess search",
    mode: "none",
    sourceLabel: "Grandmaster-Level Chess Without Search",
    sourceUrl: CHESSBENCH_URL,
    xOffset: 10,
  },
  {
    id: "gpt35-llm",
    name: "GPT-3.5 Turbo Instruct",
    rating: 1650,
    pool: "llm",
    poolLabel: "LLM Chess Benchmark",
    suite: "Lichess classical rating, Glicko-2, 116 games",
    search: "No external chess search",
    mode: "none",
    sourceLabel: "LLM Chess Benchmark leaderboard",
    sourceUrl: LLM_BENCH_URL,
    xOffset: 50,
  },
  {
    id: "maia-1100",
    name: "Maia-1100",
    rating: 1628,
    pool: "llm",
    poolLabel: "LLM Chess Benchmark",
    suite: "Lichess classical anchor, 523 games",
    search: "Policy network without tree search",
    mode: "none",
    sourceLabel: "LLM Chess Benchmark leaderboard",
    sourceUrl: LLM_BENCH_URL,
    xOffset: 64,
  },
  {
    id: "gpt56-sol",
    name: "GPT-5.6 Sol, xhigh",
    rating: 1612,
    error: 126,
    pool: "llm",
    poolLabel: "LLM Chess Benchmark",
    suite: "Lichess classical rating, Glicko-2, 8 games",
    search: "No external chess search",
    mode: "none",
    sourceLabel: "LLM Chess Benchmark leaderboard",
    sourceUrl: LLM_BENCH_URL,
    xOffset: 78,
  },
  {
    id: "karvonen-50m",
    name: "Karvonen Chess-GPT 50M",
    label: "Chess-GPT 50M",
    params: 50,
    rating: 1500,
    pool: "stockfish-proxy",
    poolLabel: "Stockfish 16 proxy",
    suite: "Games against Stockfish 16 level 0 at 0.1 seconds",
    search: "No tree search",
    mode: "none",
    sourceLabel: "Chess-GPT's Internal World Model",
    sourceUrl: KARVONEN_URL,
    note: "This is an approximate Stockfish-referenced estimate.",
  },
  {
    id: "claude-opus-46",
    name: "Claude Opus 4.6, medium",
    rating: 1403,
    error: 150,
    pool: "llm",
    poolLabel: "LLM Chess Benchmark",
    suite: "Lichess classical rating, Glicko-2, 5 games",
    search: "No external chess search",
    mode: "none",
    sourceLabel: "LLM Chess Benchmark leaderboard",
    sourceUrl: LLM_BENCH_URL,
    xOffset: 92,
  },
  {
    id: "chess-llama",
    name: "Chess-Llama",
    label: "Chess-Llama 23M",
    params: 23,
    rating: 1400,
    pool: "stockfish-proxy",
    poolLabel: "Stockfish 16 proxy",
    suite: "Estimated from games against Stockfish skill levels",
    search: "No tree search",
    mode: "none",
    sourceLabel: "Chess-Llama model card",
    sourceUrl: CHESS_LLAMA_URL,
    note: "The model card describes this as an estimated Elo, not a shared tournament rating.",
  },
  {
    id: "karvonen-25m",
    name: "Karvonen Chess-GPT 25M",
    label: "Chess-GPT 25M",
    params: 25,
    rating: 1300,
    pool: "stockfish-proxy",
    poolLabel: "Stockfish 16 proxy",
    suite: "Games against Stockfish 16 level 0 at 0.1 seconds",
    search: "No tree search",
    mode: "none",
    sourceLabel: "Chess-GPT's Internal World Model",
    sourceUrl: KARVONEN_URL,
    note: "This is an approximate Stockfish-referenced estimate.",
  },
  {
    id: "random-bot",
    name: "Random bot",
    rating: 400,
    pool: "llm",
    poolLabel: "LLM Chess Benchmark",
    suite: "Lichess classical anchor",
    search: "Random legal move selection",
    mode: "none",
    sourceLabel: "LLM Chess Benchmark leaderboard",
    sourceUrl: LLM_BENCH_URL,
    xOffset: 106,
  },
];

const poolStyles: Record<PoolId, { label: string; color: string; text: string }> = {
  ccrl: { label: "CCRL engines", color: "#707982", text: "text-[#59636d] dark:text-[#b2bcc6]" },
  paper: { label: "Paper tournament", color: "#4974bb", text: "text-[#3f65a5] dark:text-[#82a9ed]" },
  lichess: { label: "Lichess", color: "#8064ae", text: "text-[#70559f] dark:text-[#b99ae9]" },
  llm: { label: "LLM benchmark", color: "#9a6c2e", text: "text-[#8a5c20] dark:text-[#d6a45f]" },
  kibitzer: { label: "Kibitzer eval", color: "#d84b2a", text: "text-[#d84b2a] dark:text-[#ff765b]" },
  "stockfish-proxy": { label: "Stockfish proxy", color: "#347657", text: "text-[#347657] dark:text-[#68c99a]" },
};

const unknownLabelLayout: Record<string, { pointX: number; labelY: number; label: string }> = {
  "stockfish-18": { pointX: 854, labelY: 108, label: "Stockfish 18" },
  "torch-v4": { pointX: 862, labelY: 124, label: "Torch v4" },
  reckless: { pointX: 870, labelY: 140, label: "Reckless 0.8" },
  plentychess: { pointX: 878, labelY: 156, label: "PlentyChess 7" },
  berserk: { pointX: 886, labelY: 172, label: "Berserk 13" },
  dragon: { pointX: 894, labelY: 188, label: "Dragon 3.3" },
  ethereal: { pointX: 902, labelY: 204, label: "Ethereal 14.25" },
  "lc0-search": { pointX: 860, labelY: 250, label: "Lc0 T82" },
  "stockfish16-50ms": { pointX: 874, labelY: 268, label: "Stockfish 16, 50 ms" },
  eubos: { pointX: 860, labelY: 329, label: "Eubos" },
  "gemini-31-pro": { pointX: 874, labelY: 347, label: "Gemini 3.1 Pro" },
  "maia-1900": { pointX: 860, labelY: 365, label: "Maia-1900" },
  "gpt35-lichess": { pointX: 874, labelY: 383, label: "GPT-3.5, paper" },
  "gpt35-llm": { pointX: 860, labelY: 401, label: "GPT-3.5, LLM bench" },
  "maia-1100": { pointX: 874, labelY: 419, label: "Maia-1100" },
  "gpt56-sol": { pointX: 860, labelY: 437, label: "GPT-5.6 Sol" },
  "claude-opus-46": { pointX: 874, labelY: 455, label: "Claude Opus 4.6" },
  "random-bot": { pointX: 860, labelY: 545, label: "Random bot" },
};

const xTicks = [6, 10, 20, 50, 100, 200, 300];
const yTicks = [400, 1000, 1600, 2200, 2800, 3400, 4000];
const plot = { left: 76, right: 790, top: 94, bottom: 565 };
const unknownX = 914;
const xMin = Math.log10(4.8);
const xMax = Math.log10(330);
const yMin = 250;
const yMax = 4250;

function xScale(value: number) {
  return plot.left + ((Math.log10(value) - xMin) / (xMax - xMin)) * (plot.right - plot.left);
}

function yScale(value: number) {
  return plot.bottom - ((value - yMin) / (yMax - yMin)) * (plot.bottom - plot.top);
}

function pointX(point: StrengthPoint) {
  return point.params ? xScale(point.params) : unknownLabelLayout[point.id]?.pointX ?? unknownX + (point.xOffset ?? 0) - 46;
}

function Marker({ point, x, y, active }: { point: StrengthPoint; x: number; y: number; active: boolean }) {
  const size = point.id === "kibitzer" || active ? 7 : 5;

  if (point.mode === "search") {
    return (
      <path
        d={`M ${x} ${y - size} L ${x + size} ${y} L ${x} ${y + size} L ${x - size} ${y} Z`}
        fill={poolStyles[point.pool].color}
        stroke="currentColor"
        strokeWidth={active ? 2 : 1.2}
      />
    );
  }

  if (point.mode === "successor") {
    return (
      <rect
        x={x - size}
        y={y - size}
        width={size * 2}
        height={size * 2}
        className="fill-background"
        stroke={poolStyles[point.pool].color}
        strokeWidth={active ? 2.5 : 1.8}
      />
    );
  }

  return (
    <circle
      cx={x}
      cy={y}
      r={size}
      className="fill-background"
      stroke={poolStyles[point.pool].color}
      strokeWidth={active ? 2.8 : 2}
    />
  );
}

function DetailPopup({ point }: { point: StrengthPoint }) {
  const style = poolStyles[point.pool];

  return (
    <aside
      aria-live="polite"
      className={`absolute z-20 w-[310px] max-w-[calc(100%-2rem)] bg-background/95 p-4 shadow-[0_18px_50px_rgba(0,0,0,0.16)] backdrop-blur-md ${
        pointX(point) > 600 ? "left-20" : "right-4"
      } ${yScale(point.rating) < 310 ? "bottom-20" : "top-24"}`}
      style={{ borderLeft: `3px solid ${style.color}` }}
    >
      <p className={`mb-1 font-mono text-[9px] uppercase tracking-[0.16em] ${style.text}`}>{point.poolLabel}</p>
      <div className="flex items-baseline justify-between gap-4">
        <p className="m-0 font-serif text-[19px] font-semibold leading-tight text-foreground">{point.name}</p>
        <p className="m-0 whitespace-nowrap font-mono text-sm font-semibold text-foreground">
          {point.rating}{point.error ? ` +/- ${point.error}` : ""}
        </p>
      </div>
      <dl className="mt-3 grid grid-cols-[72px_1fr] gap-x-3 gap-y-2 text-[11px] leading-relaxed">
        <dt className="font-mono uppercase tracking-wide text-text-tertiary">Params</dt>
        <dd className="m-0 text-text-secondary">{point.params ? `${point.params.toFixed(point.params < 20 ? 1 : 0)}M` : "Not published"}</dd>
        <dt className="font-mono uppercase tracking-wide text-text-tertiary">Search</dt>
        <dd className="m-0 text-text-secondary">{point.search}</dd>
        <dt className="font-mono uppercase tracking-wide text-text-tertiary">Eval suite</dt>
        <dd className="m-0 text-text-secondary">{point.suite}</dd>
      </dl>
      {point.note ? <p className="mb-0 mt-3 text-[10.5px] leading-relaxed text-text-tertiary">{point.note}</p> : null}
      {point.extra ? <p className="mb-0 mt-3 text-[10.5px] leading-relaxed text-text-tertiary">{point.extra}</p> : null}
      <a
        href={point.sourceUrl}
        target="_blank"
        rel="noreferrer"
        className={`mt-3 inline-flex font-mono text-[10px] font-semibold no-underline hover:underline ${style.text}`}
      >
        Source: {point.sourceLabel} ↗
      </a>
    </aside>
  );
}

export function KibitzerParameterStrengthGraph() {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const activePoint = hoveredId ? points.find((point) => point.id === hoveredId) ?? null : null;

  return (
    <figure className="my-10 w-full text-foreground">
      <div
        className="relative left-1/2 w-[min(1160px,calc(100vw-2rem))] -translate-x-1/2 overflow-x-auto"
        onMouseLeave={() => setHoveredId(null)}
      >
        <div className="relative min-w-[820px]">
          <svg
            viewBox="0 0 1080 660"
            className="block h-auto w-full"
            role="img"
            aria-labelledby="kibitzer-strength-title kibitzer-strength-description"
          >
            <title id="kibitzer-strength-title">Parameter count versus reported chess strength</title>
            <desc id="kibitzer-strength-description">
              One interactive graph comparing published chess ratings. Colors separate evaluation pools. Models without a published parameter count appear in a shaded rail whose horizontal positions carry no meaning.
            </desc>

            <text x={plot.left} y={31} className="fill-foreground font-serif text-[22px] font-semibold">
              Parameter count vs reported chess strength
            </text>
            <text x={plot.left} y={52} className="fill-text-secondary font-mono text-[10px]">
              Hover or focus a point to isolate its pool, eval suite, and source.
            </text>

            <rect x={842} y={plot.top - 12} width={206} height={plot.bottom - plot.top + 24} rx={4} className="fill-muted/50" />
            <text x={945} y={plot.top - 17} textAnchor="middle" className="fill-text-tertiary font-mono text-[9px] uppercase tracking-[0.12em]">
              Parameters not published
            </text>
            <text x={945} y={plot.bottom + 24} textAnchor="middle" className="fill-text-tertiary font-mono text-[8.5px]">
              x placement has no meaning
            </text>

            {yTicks.map((tick) => {
              const y = yScale(tick);
              return (
                <g key={tick}>
                  <line x1={plot.left} x2={1048} y1={y} y2={y} className="stroke-divider" strokeWidth={1} />
                  <text x={plot.left - 13} y={y + 4} textAnchor="end" className="fill-text-tertiary font-mono text-[10px]">
                    {tick}
                  </text>
                </g>
              );
            })}

            {xTicks.map((tick) => {
              const x = xScale(tick);
              return (
                <g key={tick}>
                  <line x1={x} x2={x} y1={plot.top} y2={plot.bottom} className="stroke-divider" strokeWidth={1} />
                  <text x={x} y={plot.bottom + 24} textAnchor="middle" className="fill-text-tertiary font-mono text-[10px]">
                    {tick}M
                  </text>
                </g>
              );
            })}

            <text x={(plot.left + plot.right) / 2} y={623} textAnchor="middle" className="fill-text-secondary font-mono text-[11px]">
              Network parameters, logarithmic scale
            </text>
            <text
              x={20}
              y={(plot.top + plot.bottom) / 2}
              textAnchor="middle"
              transform={`rotate(-90 20 ${(plot.top + plot.bottom) / 2})`}
              className="fill-text-secondary font-mono text-[11px]"
            >
              Source-specific Elo or rating
            </text>

            {points.map((point) => {
              const x = pointX(point);
              const y = yScale(point.rating);
              const isActive = point.id === activePoint?.id;
              const isSamePool = point.pool === activePoint?.pool;
              const opacity = !activePoint ? 0.86 : isActive ? 1 : isSamePool ? 0.72 : 0.18;
              const yTop = point.error ? yScale(point.rating + point.error) : y;
              const yBottom = point.error ? yScale(point.rating - point.error) : y;

              return (
                <g
                  key={point.id}
                  tabIndex={0}
                  role="button"
                  aria-label={`${point.name}, ${point.rating}${point.error ? ` plus or minus ${point.error}` : ""}, ${point.poolLabel}`}
                  className="cursor-pointer outline-none transition-opacity duration-150 focus-visible:opacity-100"
                  opacity={opacity}
                  onMouseEnter={() => setHoveredId(point.id)}
                  onFocus={() => setHoveredId(point.id)}
                  onBlur={() => setHoveredId(null)}
                >
                  <title>{`${point.name}: ${point.rating}, ${point.suite}`}</title>
                  {isActive ? (
                    <line x1={plot.left} x2={1048} y1={y} y2={y} stroke={poolStyles[point.pool].color} strokeWidth={1} strokeDasharray="3 5" opacity={0.5} />
                  ) : null}
                  {point.error ? (
                    <g stroke={poolStyles[point.pool].color} strokeWidth={isActive ? 2 : 1.3}>
                      <line x1={x} x2={x} y1={yTop} y2={yBottom} />
                      <line x1={x - 4} x2={x + 4} y1={yTop} y2={yTop} />
                      <line x1={x - 4} x2={x + 4} y1={yBottom} y2={yBottom} />
                    </g>
                  ) : null}
                  <circle cx={x} cy={y} r={14} fill="transparent" />
                  <Marker point={point} x={x} y={y} active={isActive} />
                  {!point.params && unknownLabelLayout[point.id] ? (
                    <g pointerEvents="none">
                      <path
                        d={`M ${x + 7} ${y} L 910 ${unknownLabelLayout[point.id].labelY} L 916 ${unknownLabelLayout[point.id].labelY}`}
                        fill="none"
                        stroke={poolStyles[point.pool].color}
                        strokeWidth={isActive ? 1.4 : 0.8}
                        opacity={isActive ? 0.9 : 0.5}
                      />
                      <text
                        x={921}
                        y={unknownLabelLayout[point.id].labelY + 3}
                        fill={poolStyles[point.pool].color}
                        className={`font-mono text-[7.8px] ${isActive ? "font-semibold" : ""}`}
                      >
                        {unknownLabelLayout[point.id].label}
                      </text>
                    </g>
                  ) : null}
                  {point.label ? (
                    <text
                      x={x + (point.id === "searchless-270m" ? -10 : 10)}
                      y={y + (point.id === "searchless-270m" ? 22 : -11)}
                      textAnchor={point.id === "searchless-270m" ? "end" : "start"}
                      fill={poolStyles[point.pool].color}
                      className={`font-mono text-[9px] ${point.id === "kibitzer" ? "font-semibold" : ""}`}
                    >
                      {point.label}
                    </text>
                  ) : null}
                </g>
              );
            })}

            <g transform="translate(76 646)">
              {Object.entries(poolStyles).map(([id, style], index) => {
                const x = index * 175;
                const y = index < 3 ? -3 : -3;
                const samePool = activePoint?.pool === id;
                return (
                  <g key={id} transform={`translate(${x} ${y})`} opacity={!activePoint || samePool ? 1 : 0.35}>
                    <circle cx={0} cy={0} r={4} fill={style.color} />
                    <text x={11} y={3.5} className="fill-text-secondary font-mono text-[8.5px]">
                      {style.label}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>

          {activePoint ? <DetailPopup point={activePoint} /> : null}
        </div>
      </div>
      <figcaption className="mt-3 text-center font-mono text-[10px] leading-relaxed text-text-tertiary">
        Color identifies the evaluation pool. Ratings compare directly only within the same color.
      </figcaption>
    </figure>
  );
}
