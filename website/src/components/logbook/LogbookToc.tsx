import type { RefObject } from "react";

type Decision = {
  id: string;
  number: number;
  title: string;
  phase: string;
  slug: string;
};

type LogbookTocProps = {
  decisions: Decision[];
  activeId: string | null;
  scrollContainerRef: RefObject<HTMLElement | null>;
};

export function LogbookToc({ decisions, activeId }: LogbookTocProps) {
  const phases = decisions.reduce<Map<string, Decision[]>>((map, decision) => {
    const group = map.get(decision.phase) ?? [];
    group.push(decision);
    map.set(decision.phase, group);
    return map;
  }, new Map());

  const contents = (
    <div className="space-y-7">
      {[...phases.entries()].map(([phase, entries]) => (
        <section key={phase}>
          <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.12em] text-text-tertiary">
            {phase}
          </p>
          <ol className="space-y-0.5">
            {entries.map((decision) => (
              <li key={decision.id}>
                <a
                  href={`#${decision.slug}`}
                  className={`grid grid-cols-[2.3rem_1fr] gap-2 rounded px-1 py-1.5 text-[11px] leading-4 transition-colors ${
                    activeId === decision.id
                      ? "bg-surface-active text-foreground"
                      : "text-text-tertiary hover:bg-surface-hover hover:text-foreground"
                  }`}
                >
                  <span className="font-mono text-[9px] uppercase">{decision.id}</span>
                  <span>{decision.title}</span>
                </a>
              </li>
            ))}
          </ol>
        </section>
      ))}
    </div>
  );

  return (
    <>
      <aside className="sticky top-20 hidden max-h-[calc(100vh-6rem)] overflow-y-auto pr-4 lg:block">
        <p className="eyebrow mb-5">decision index</p>
        {contents}
      </aside>
      <details className="border border-divider bg-surface p-4 lg:hidden">
        <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-[0.12em]">
          browse all decisions
        </summary>
        <div className="mt-5 max-h-[55vh] overflow-y-auto pr-2">{contents}</div>
      </details>
    </>
  );
}
