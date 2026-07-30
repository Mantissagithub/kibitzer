import { ExternalLink, Moon, Sun } from "lucide-react";
import { useTheme } from "@/components/ThemeProvider";
import { Button } from "@/components/ui/button";
import { Link, NavLink } from "@/lib/router";
import { cn } from "@/lib/utils";

const navItems = [
  { to: "/", label: "overview", end: true },
  { to: "/play", label: "play" },
  { to: "/games", label: "game traces", compactLabel: "games" },
  { to: "/logbook", label: "logbook" },
];

export function SiteHeader() {
  const { theme, setTheme } = useTheme();

  return (
    <header className="sticky top-0 z-40 border-b border-divider/70 bg-background/88 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1180px] items-center justify-between gap-2 px-4 py-3 sm:gap-4 sm:px-6 lg:px-8">
        <Link to="/" className="group flex shrink-0 items-center gap-3" aria-label="Kibitzer home">
          <span className="grid size-8 place-items-center bg-black font-mono text-xs font-semibold text-white transition-transform group-hover:-rotate-3 dark:bg-white dark:text-black">
            K
          </span>
          <span className="hidden font-serif text-sm font-semibold tracking-tight sm:inline">kibitzer</span>
        </Link>

        <nav aria-label="Primary navigation" className="flex min-w-0 items-center gap-1 overflow-x-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                cn(
                  "whitespace-nowrap rounded-md px-2.5 py-2 font-mono text-[10px] uppercase tracking-[0.12em] text-text-tertiary transition-colors hover:bg-surface-hover hover:text-foreground sm:px-3",
                  isActive && "bg-surface text-foreground",
                )
              }
            >
              <span className={item.compactLabel ? "sm:hidden" : undefined}>{item.compactLabel ?? item.label}</span>
              {item.compactLabel ? <span className="hidden sm:inline">{item.label}</span> : null}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-1">
          <a
            href="https://www.pradheep.dev/blogs/goodharts-gambit"
            target="_blank"
            rel="noreferrer"
            className="hidden items-center gap-1 rounded-md px-2 py-2 font-mono text-[10px] uppercase tracking-[0.1em] text-text-tertiary hover:bg-surface-hover hover:text-foreground md:flex"
          >
            essay <ExternalLink className="size-3" />
          </a>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </Button>
        </div>
      </div>
    </header>
  );
}
