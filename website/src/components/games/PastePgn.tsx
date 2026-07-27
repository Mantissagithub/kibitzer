import { useState } from "react";
import { Button } from "@/components/ui/button";

export function PastePgn({ onImport }: { onImport: (text: string) => void }) {
  const [value, setValue] = useState("");

  return (
    <details className="page-shell mb-8 border border-divider bg-surface/45 p-4">
      <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-[0.12em] text-text-secondary">
        paste a PGN instead
      </summary>
      <div className="mt-4 grid gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
        <label>
          <span className="sr-only">PGN text</span>
          <textarea
            value={value}
            onChange={(event) => setValue(event.target.value)}
            placeholder={'[Event "My game"]\n\n1. e4 e5 ...'}
            className="min-h-36 w-full resize-y border border-divider bg-background p-3 font-mono text-[11px] leading-5 outline-none focus:border-foreground"
          />
        </label>
        <Button
          type="button"
          variant="accent"
          disabled={!value.trim()}
          onClick={() => {
            onImport(value);
            setValue("");
          }}
        >
          load pasted games
        </Button>
      </div>
    </details>
  );
}
