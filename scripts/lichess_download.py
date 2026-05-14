"""download the lichess elite database (one zip per month) into data/raw.

source: https://database.nikonoel.fr/  (curated by nikonoel; rated 2500+ on
both sides since dec 2021, 2400+ before that). files run from june 2020 to
the present month-or-two. each zip contains one pgn.

examples:
    # all of 2024 into ./data/raw (skip months already extracted):
    uv run python scripts/lichess_download.py --year 2024

    # just january and february:
    uv run python scripts/lichess_download.py --year 2024 --months "01,02"

    # one most-recent month for fast pipeline iteration:
    uv run python scripts/lichess_download.py --year 2024 --sample

    # see what would happen, no network writes:
    uv run python scripts/lichess_download.py --year 2024 --dry-run
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from rich.table import Table
from tqdm import tqdm

from kibitzer import tui


BASE_URL = "https://database.nikonoel.fr/lichess_elite_{year:04d}-{month:02d}.zip"
USER_AGENT = "kibitzer-lichess-download/0.1 (+pretraining)"
CHUNK = 1 << 20  # 1 MiB


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--months", default="all",
                   help='comma-separated months ("01,02,03") or "all"')
    p.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="skip months whose unzipped PGN already exists (default: on)")
    p.add_argument("--sample", action="store_true",
                   help="download only the latest available month for the year")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve plan and exit without downloading")
    p.add_argument("--no-tui", action="store_true",
                   help="disable rich/tqdm UI (auto-disabled when stdout isn't a TTY)")
    return p.parse_args()


def parse_months(s: str) -> list[int] | None:
    if s == "all":
        return None
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            m = int(part)
        except ValueError:
            raise SystemExit(f"unparseable month: {part!r}")
        if not 1 <= m <= 12:
            raise SystemExit(f"month out of range (1-12): {m}")
        out.append(m)
    return sorted(set(out))


def head_size(url: str) -> int | None:
    """return content-length; none if the file is absent or the probe fails.

    a transient network error (timeout, ssl, refused) is treated like a 404
    for the purposes of this run — we just skip the month rather than abort
    the whole probe loop.
    """
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"HEAD {url} → HTTP {e.code}; skipping", file=sys.stderr)
        return None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"HEAD {url} → {type(e).__name__}: {e}; skipping", file=sys.stderr)
        return None


def discover_year(year: int) -> dict[int, int]:
    """head-probe each month of `year`; return {month: size_bytes}."""
    found: dict[int, int] = {}
    for m in range(1, 13):
        url = BASE_URL.format(year=year, month=m)
        size = head_size(url)
        if size is not None:
            found[m] = size
    return found


def existing_pgn(output_dir: Path, year: int, month: int) -> Path | None:
    """return the already-extracted pgn path if present."""
    p = output_dir / f"lichess_elite_{year:04d}-{month:02d}.pgn"
    return p if p.exists() else None


def download_one(url: str, dest_zip: Path, expected: int, *, ui: bool) -> None:
    """stream `url` to `dest_zip` via a `.part` sidecar; verify size; rename."""
    part = dest_zip.with_suffix(dest_zip.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total = expected or int(r.headers.get("Content-Length", 0)) or None
        bar = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest_zip.name,
            disable=not ui,
            miniters=1,
            leave=True,
        )
        with bar:
            while True:
                chunk = r.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))

    actual = part.stat().st_size
    if expected and actual != expected:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"size mismatch for {url}: expected {expected}, got {actual}"
        )
    part.rename(dest_zip)


def extract_zip(zip_path: Path, output_dir: Path) -> list[Path]:
    """verify and extract; return paths of extracted entries."""
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt entry {bad!r} in {zip_path}")
        names = zf.namelist()
        zf.extractall(output_dir)
    return [output_dir / n for n in names]


def render_plan(year: int, plan: list[tuple[str, str, Path, int | None]]) -> None:
    if tui.is_tty():
        table = Table(title=f"plan (year={year})", title_style="header",
                      header_style="muted", show_lines=False)
        table.add_column("action", style="accent")
        table.add_column("month", justify="right")
        table.add_column("size", justify="right", style="muted")
        table.add_column("destination")
        for action, _url, dest, size in plan:
            month = dest.stem.split("-")[-1]
            size_s = f"{size / (1 << 20):.1f} MiB" if size else "—"
            table.add_row(action, month, size_s, str(dest))
        tui.console.print(table)
    else:
        for action, url, dest, size in plan:
            size_s = f"{size}" if size else "?"
            print(f"{action}\t{url}\t{dest}\t{size_s}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    use_ui = not args.no_tui and tui.is_tty()

    months_filter = parse_months(args.months)

    if use_ui:
        with tui.console.status(f"[muted]probing {args.year} months…[/]"):
            available = discover_year(args.year)
    else:
        available = discover_year(args.year)
    if not available:
        msg = f"no months available at {BASE_URL.format(year=args.year, month=1).rsplit('-', 1)[0]}-* (year {args.year})"
        if use_ui:
            tui.console.print(f"[error]{msg}[/]")
        else:
            print(msg, file=sys.stderr)
        return 1

    if months_filter is None:
        chosen = sorted(available.keys())
    else:
        chosen = [m for m in months_filter if m in available]
        missing = sorted(set(months_filter) - set(available))
        if missing:
            warn = f"requested but not available for {args.year}: {missing}"
            if use_ui:
                tui.console.print(f"[warning]{warn}[/]")
            else:
                print(warn, file=sys.stderr)

    if args.sample:
        chosen = chosen[-1:]

    if not chosen:
        if use_ui:
            tui.console.print("[error]nothing to download after filtering[/]")
        else:
            print("nothing to download after filtering", file=sys.stderr)
        return 1

    plan: list[tuple[str, str, Path, int | None]] = []
    for m in chosen:
        url = BASE_URL.format(year=args.year, month=m)
        zip_path = output_dir / f"lichess_elite_{args.year:04d}-{m:02d}.zip"
        if args.skip_existing and existing_pgn(output_dir, args.year, m):
            plan.append(("skip", url, existing_pgn(output_dir, args.year, m), None))
        else:
            plan.append(("get", url, zip_path, available[m]))

    if args.dry_run:
        render_plan(args.year, plan)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures = 0
    for action, url, dest, size in plan:
        if action == "skip":
            if use_ui:
                tui.console.print(f"[muted]skip {dest}[/]")
            else:
                print(f"skip {dest}", file=sys.stderr)
            successes += 1
            continue

        zip_path = dest
        if not use_ui:
            print(f"downloading {url} → {zip_path} ({size} bytes)", file=sys.stderr)
        try:
            download_one(url, zip_path, size or 0, ui=use_ui)
            if use_ui:
                with tui.console.status(f"[muted]extracting {zip_path.name}…[/]"):
                    extracted = extract_zip(zip_path, output_dir)
            else:
                extracted = extract_zip(zip_path, output_dir)
            if use_ui:
                tui.console.print(
                    f"[success]✓[/] {zip_path.name}  "
                    f"[muted]→ {len(extracted)} file(s)[/]"
                )
            else:
                print(f"done {zip_path} ({len(extracted)} file(s))", file=sys.stderr)
            successes += 1
        except Exception as e:  # noqa: BLE001
            failures += 1
            msg = f"failed {url}: {e}"
            if use_ui:
                tui.console.print(f"[error]✗[/] {msg}")
            else:
                print(msg, file=sys.stderr)

    if use_ui:
        tui.console.print(
            f"[header]done[/] [success]{successes}[/] ok, "
            f"[error]{failures}[/] failed"
        )
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
