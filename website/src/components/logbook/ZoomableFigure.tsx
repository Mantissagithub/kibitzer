import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";

type ZoomableFigureProps = {
  src?: string;
  alt?: string;
  width?: number;
  height?: number;
};

export function ZoomableFigure({ src = "", alt = "", width, height }: ZoomableFigureProps) {
  if (!src) return null;

  return (
    <Dialog>
      <figure className="my-8">
        <DialogTrigger asChild>
          <button
            type="button"
            className="block w-full cursor-zoom-in overflow-hidden border border-divider bg-surface p-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label={`Enlarge figure: ${alt || "Logbook figure"}`}
          >
            <img
              src={src}
              alt={alt}
              width={width}
              height={height}
              loading="lazy"
              className="h-auto w-full"
            />
          </button>
        </DialogTrigger>
        {alt ? (
          <figcaption className="mt-2 text-center font-mono text-[10px] leading-5 text-text-tertiary">
            {alt} · click to inspect
          </figcaption>
        ) : null}
      </figure>
      <DialogContent aria-describedby={undefined}>
        <img src={src} alt={alt} width={width} height={height} className="h-auto w-full" />
      </DialogContent>
    </Dialog>
  );
}
