import { useEffect, useRef, useState } from "react";
import LogbookContent from "@/generated/logbook.mdx";
import logbookIndexData from "@/generated/logbook-index.json";
import { LogbookDocument } from "@/components/logbook/LogbookDocument";
import { LogbookToc } from "@/components/logbook/LogbookToc";

type Decision = {
  id: string;
  number: number;
  title: string;
  phase: string;
  slug: string;
};

type LogbookIndex = {
  sourceChecksum: string;
  decisionCount: number;
  imageCount: number;
  imageDimensions: Record<string, { width: number; height: number }>;
  decisions: Decision[];
};

const logbookIndex = logbookIndexData as LogbookIndex;

export default function LogbookPage() {
  const articleRef = useRef<HTMLElement>(null);
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    const headings = [...document.querySelectorAll<HTMLElement>("[data-logbook-decision]")];
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.find((entry) => entry.isIntersecting);
        const id = visible?.target.getAttribute("data-logbook-decision");
        if (id) setActiveId(id);
      },
      { rootMargin: "-15% 0px -72% 0px", threshold: 0 },
    );
    headings.forEach((heading) => observer.observe(heading));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!window.location.hash) return;
    const id = window.location.hash.slice(1);
    requestAnimationFrame(() => document.getElementById(id)?.scrollIntoView());
  }, []);

  return (
    <div className="page-shell py-12 lg:py-16">
      <header className="mb-12 grid gap-7 border-b border-divider pb-10 lg:grid-cols-[250px_minmax(0,760px)] lg:gap-12">
        <p className="eyebrow">full experimental record</p>
        <div>
          <h1 className="font-serif text-4xl font-semibold tracking-[-0.045em] sm:text-6xl">
            The logbook, rendered as a record rather than a victory lap.
          </h1>
          <div className="mt-6 flex flex-wrap gap-x-6 gap-y-2 font-mono text-[10px] uppercase tracking-[0.12em] text-text-tertiary">
            <span>{logbookIndex.decisionCount} decisions</span>
            <span>{logbookIndex.imageCount} figures</span>
            <span>source: LOGBOOK.md</span>
          </div>
        </div>
      </header>

      <div className="grid min-w-0 gap-10 lg:grid-cols-[250px_minmax(0,760px)] lg:gap-12">
        <LogbookToc
          decisions={logbookIndex.decisions}
          activeId={activeId}
          scrollContainerRef={articleRef}
        />
        <section ref={articleRef} className="min-w-0">
          <LogbookDocument content={LogbookContent} index={logbookIndex} />
        </section>
      </div>
    </div>
  );
}
