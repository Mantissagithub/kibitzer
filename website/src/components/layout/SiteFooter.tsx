export function SiteFooter() {
  return (
    <footer className="border-t border-divider">
      <div className="mx-auto flex max-w-[1180px] flex-col gap-2 px-4 py-8 font-mono text-[10px] uppercase tracking-[0.12em] text-text-tertiary sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
        <p>Kibitzer · trained on one laptop GPU</p>
        <a
          href="https://github.com/Mantissagithub/kibitzer"
          target="_blank"
          rel="noreferrer"
          className="hover:text-foreground"
        >
          source on github ↗
        </a>
      </div>
    </footer>
  );
}
