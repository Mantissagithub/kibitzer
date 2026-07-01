from __future__ import annotations

import argparse
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm


BASE_URL = "https://database.nikonoel.fr/lichess_elite_{year:04d}-{month:02d}.zip"
USER_AGENT = "kibitzer-clean-rebuild/1.0"
CHUNK_SIZE = 1 << 20


def parse_months(value: str) -> list[int]:
    months = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not months or any(month < 1 or month > 12 for month in months):
        raise ValueError("months must be a comma-separated subset of 1..12")
    return months


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.headers.get("Content-Length", 0))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise SystemExit(f"dataset month is unavailable: {url}") from error
        raise


def download(url: str, destination: Path, expected_size: int) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
        with tqdm(
            total=expected_size or None,
            desc=destination.name,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            while chunk := response.read(CHUNK_SIZE):
                handle.write(chunk)
                progress.update(len(chunk))
    actual_size = partial.stat().st_size
    if expected_size and actual_size != expected_size:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"download size mismatch for {url}: expected {expected_size}, got {actual_size}"
        )
    partial.replace(destination)


def extract_pgn(archive: Path, output_dir: Path) -> Path:
    expected_name = archive.with_suffix(".pgn").name
    with zipfile.ZipFile(archive) as zip_file:
        corrupt = zip_file.testzip()
        if corrupt is not None:
            raise RuntimeError(f"corrupt zip member: {corrupt}")
        matching = [name for name in zip_file.namelist() if Path(name).name == expected_name]
        if len(matching) != 1:
            raise RuntimeError(f"expected exactly one {expected_name} in {archive}")
        member = matching[0]
        destination = output_dir / expected_name
        with zip_file.open(member) as source, destination.open("wb") as target:
            while chunk := source.read(CHUNK_SIZE):
                target.write(chunk)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download monthly Lichess Elite PGNs.")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--months", required=True, help="Comma-separated months, e.g. 06,07,08")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for month in parse_months(args.months):
        stem = f"lichess_elite_{args.year:04d}-{month:02d}"
        pgn_path = args.output_dir / f"{stem}.pgn"
        if pgn_path.exists() and pgn_path.stat().st_size > 0:
            print(f"cached: {pgn_path}")
            continue

        url = BASE_URL.format(year=args.year, month=month)
        archive = args.output_dir / f"{stem}.zip"
        expected_size = remote_size(url)
        print(f"downloading: {url}")
        download(url, archive, expected_size)
        extracted = extract_pgn(archive, args.output_dir)
        archive.unlink()
        print(f"ready: {extracted}")


if __name__ == "__main__":
    main()
