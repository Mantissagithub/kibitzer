import { ArrowRight, ExternalLink } from "lucide-react";
import leaderboardData from "@/generated/leaderboard.json";
import { KibitzerParameterStrengthGraph } from "@/components/home/KibitzerParameterStrengthGraph";
import { Button } from "@/components/ui/button";
import { Link } from "@/lib/router";

type LeaderboardEntry = {
  rank: number;
  player: string;
  rating: number;
  error: number | null;
  points: number;
  played: number;
  percent: number;
};

const leaderboard = leaderboardData as LeaderboardEntry[];

const facts = [
  ["15.2M", "trained parameters"],
  ["142M", "lichess positions"],
  ["512", "PUCT simulations"],
  ["171", "rated clean games"],
];

export default function HomePage() {
  return (
    <>
      <section className="relative overflow-hidden border-b border-divider">
        <div className="paper-grid pointer-events-none absolute inset-0 opacity-70" />
        <div className="page-shell relative grid min-h-[680px] gap-12 py-20 lg:grid-cols-[minmax(0,1.55fr)_minmax(280px,0.65fr)] lg:items-end lg:py-28">
          <div className="max-w-[820px]">
            <p className="eyebrow mb-7">small model · one laptop · too many experiments</p>
            <h1 className="max-w-[780px] font-serif text-[clamp(3.25rem,8.5vw,7.9rem)] font-semibold leading-[0.9] tracking-[-0.065em]">
              A chess model with a very strong opinion.
            </h1>
            <p className="mt-9 max-w-[650px] text-base leading-8 text-text-secondary sm:text-lg">
              I trained Kibitzer on a laptop RTX 4060 to see how far a small
              policy and value model could go. The honest answer is 2581 Elo
              with deep PUCT search, plus a long record of metrics that looked
              useful right until the model had to play.
            </p>
            <div className="mt-9 flex flex-wrap gap-3">
              <Button asChild variant="accent">
                <Link to="/games">
                  watch the games <ArrowRight className="size-4" />
                </Link>
              </Button>
              <Button asChild variant="outline">
                <Link to="/logbook">open the full logbook</Link>
              </Button>
              <Button asChild variant="ghost">
                <a
                  href="https://www.pradheep.dev/blogs/goodharts-gambit"
                  target="_blank"
                  rel="noreferrer"
                >
                  Goodhart&apos;s Gambit <ExternalLink className="size-3.5" />
                </a>
              </Button>
            </div>
          </div>

          <aside className="border-l border-divider pl-5 lg:mb-2 lg:pl-7" aria-label="Experiment snapshot">
            <p className="eyebrow mb-5">current record / d67</p>
            <p className="font-mono text-[clamp(2.7rem,6vw,5.1rem)] font-medium leading-none tracking-[-0.06em] text-kibitzer">
              2581.6
            </p>
            <p className="mt-2 font-mono text-xs text-text-tertiary">± 102.3 Ordo Elo</p>
            <div className="mt-8 space-y-3 border-t border-divider pt-5 font-mono text-[10px] uppercase tracking-[0.1em] text-text-tertiary">
              <div className="flex justify-between gap-4"><span>hardware</span><span className="text-foreground">RTX 4060 8GB</span></div>
              <div className="flex justify-between gap-4"><span>checkpoint</span><span className="text-foreground">tactical repair r1</span></div>
              <div className="flex justify-between gap-4"><span>search</span><span className="text-foreground">PUCT s512</span></div>
            </div>
          </aside>
        </div>
      </section>

      <section className="border-b border-divider bg-surface/35">
        <div className="page-shell grid grid-cols-2 divide-x divide-y divide-divider border-x border-divider md:grid-cols-4 md:divide-y-0">
          {facts.map(([value, label]) => (
            <div key={label} className="px-5 py-7 sm:px-7">
              <p className="font-mono text-2xl font-medium tracking-tight">{value}</p>
              <p className="mt-2 font-mono text-[9px] uppercase tracking-[0.13em] text-text-tertiary">{label}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="page-shell py-20 lg:py-28">
        <div className="grid gap-12 lg:grid-cols-[0.72fr_1.45fr]">
          <div>
            <p className="eyebrow section-rule">§ 01 · what it is</p>
            <h2 className="mt-8 max-w-md font-serif text-3xl font-semibold leading-tight tracking-[-0.035em] sm:text-4xl">
              One board in. One move and one value out.
            </h2>
          </div>
          <div className="max-w-2xl space-y-5 text-[15px] leading-8 text-text-secondary">
            <p>
              Kibitzer is an attention-first chess policy and value network. It
              encodes the 64 squares with chess-relative attention, pools the
              position, then predicts over a fixed 4,672-move action space and
              estimates the side-to-move value.
            </p>
            <p>
              At play time those outputs guide PUCT, an AlphaZero-style search.
              The model is useful on its own, but the search is where the
              measured strength appears. Keeping those two claims separate is
              the point of the site, not a footnote.
            </p>
            <a
              href="https://github.com/Mantissagithub/kibitzer"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 border-b border-kibitzer pb-1 font-mono text-[10px] uppercase tracking-[0.12em] text-foreground"
            >
              inspect the implementation <ExternalLink className="size-3" />
            </a>
          </div>
        </div>
      </section>

      <section className="border-y border-divider bg-surface/40 py-20 lg:py-28">
        <div className="page-shell">
          <div className="grid gap-8 lg:grid-cols-[0.72fr_1.45fr]">
            <div>
              <p className="eyebrow section-rule">§ 02 · official result</p>
              <h2 className="mt-8 font-serif text-3xl font-semibold tracking-[-0.035em] sm:text-4xl">
                Fourth in its calibrated ladder.
              </h2>
              <p className="mt-5 max-w-sm text-sm leading-7 text-text-secondary">
                Ordo rated 171 clean games after removing tagged time forfeits.
                SF-2500 is the fixed 2500 anchor. This is the result to quote.
              </p>
            </div>

            <div className="min-w-0">
              <div className="overflow-x-auto border border-divider bg-background">
                <table className="w-full min-w-[640px] border-collapse text-left">
                  <thead className="bg-surface font-mono text-[9px] uppercase tracking-[0.12em] text-text-tertiary">
                    <tr>
                      <th className="px-4 py-3 font-medium">rank</th>
                      <th className="px-4 py-3 font-medium">player</th>
                      <th className="px-4 py-3 text-right font-medium">rating</th>
                      <th className="px-4 py-3 text-right font-medium">error</th>
                      <th className="px-4 py-3 text-right font-medium">games</th>
                      <th className="px-4 py-3 text-right font-medium">score</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono text-xs">
                    {leaderboard.map((entry) => {
                      const isKibitzer = entry.player.toLowerCase().includes("kibitzer");
                      return (
                        <tr
                          key={entry.player}
                          className={isKibitzer ? "bg-kibitzer/[0.08] text-foreground" : "border-t border-divider text-text-secondary"}
                        >
                          <td className="px-4 py-4 text-text-tertiary">{String(entry.rank).padStart(2, "0")}</td>
                          <td className="px-4 py-4 font-medium">{entry.player}</td>
                          <td className={isKibitzer ? "px-4 py-4 text-right font-semibold text-kibitzer" : "px-4 py-4 text-right"}>
                            {entry.rating.toFixed(1)}
                          </td>
                          <td className="px-4 py-4 text-right">{entry.error === null ? "anchor" : `± ${entry.error.toFixed(1)}`}</td>
                          <td className="px-4 py-4 text-right">{entry.played}</td>
                          <td className="px-4 py-4 text-right">{entry.percent}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

        </div>
      </section>

      <section className="page-shell py-20 lg:py-28">
        <div className="mb-12 grid gap-8 lg:grid-cols-[0.72fr_1.45fr]">
          <div>
            <p className="eyebrow section-rule">§ 03 · wider landscape</p>
          </div>
          <div>
            <h2 className="font-serif text-3xl font-semibold tracking-[-0.035em] sm:text-4xl">
              Similar number, different measuring stick.
            </h2>
            <p className="mt-5 max-w-2xl text-sm leading-7 text-text-secondary">
              The comparison below keeps every evaluation pool visible. Hover
              or focus a point for its method and source. Ratings of different
              colors are context, not one universal leaderboard.
            </p>
          </div>
        </div>
        <KibitzerParameterStrengthGraph />
      </section>

      <section className="border-t border-divider bg-foreground text-background">
        <div className="page-shell grid gap-12 py-20 lg:grid-cols-[1.25fr_0.75fr] lg:items-end lg:py-24">
          <div>
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] opacity-55">the useful artifact</p>
            <h2 className="mt-6 max-w-3xl font-serif text-4xl font-semibold leading-[1.06] tracking-[-0.045em] sm:text-6xl">
              The failures are part of the result.
            </h2>
            <p className="mt-6 max-w-2xl text-sm leading-7 opacity-70">
              The logbook keeps the decisions, regressions, misleading proxy
              wins, and the experiments that actually survived an external
              gate. The game traces let you inspect the final claim move by
              move.
            </p>
          </div>
          <div className="flex flex-wrap gap-3 lg:justify-end">
            <Button asChild variant="inverse">
              <Link to="/logbook">read the logbook</Link>
            </Button>
            <Button asChild variant="outline" className="border-background/30 text-background hover:bg-background/10">
              <Link to="/games">open 171 games</Link>
            </Button>
          </div>
        </div>
      </section>
    </>
  );
}
