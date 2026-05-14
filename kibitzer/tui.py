"""shared rich-based tui helpers for kibitzer's user-facing scripts.

single console, single theme, a handful of renderables. scripts decide via
:func:`is_tty` whether to use the tui path or a plain-print fallback so
piped/automated invocations stay clean.

future training code should plug into the same primitives (live loss panels,
eval bars during cutechess gauntlets) — that's why this module is in the
package, not under ``scripts/``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import chess
from rich.align import Align
from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


THEME = Theme(
    {
        "header": "bold cyan",
        "accent": "bold magenta",
        "success": "bold green",
        "warning": "yellow",
        "error": "bold red",
        "dim": "grey50",
        "muted": "grey62",
        "board.light": "on grey78",
        "board.dark": "on grey42",
        "board.white_piece": "bright_white",
        "board.black_piece": "black",
        "board.highlight_light": "on yellow1",
        "board.highlight_dark": "on dark_orange3",
        "value.white": "bold green",
        "value.black": "bold red",
    }
)

console = Console(theme=THEME, highlight=False)


_PIECE_GLYPH = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}


def is_tty() -> bool:
    """true when stdout can draw a tui."""
    return sys.stdout.isatty()


def header(title: str, subtitle: str | None = None) -> Panel:
    body = Text(title, style="header")
    if subtitle:
        body.append("\n")
        body.append(subtitle, style="muted")
    return Panel(Align.center(body), border_style="header", padding=(0, 2))


def board_panel(
    board: chess.Board,
    last_move: chess.Move | None = None,
    perspective: chess.Color = chess.WHITE,
    title: str | None = None,
) -> Panel:
    """render a chess board with file/rank labels and last-move highlight."""
    grid = Table.grid(padding=0, collapse_padding=True)
    for _ in range(9):
        grid.add_column(no_wrap=True)

    files = "abcdefgh"
    ranks = range(7, -1, -1) if perspective == chess.WHITE else range(0, 8)
    file_order = range(0, 8) if perspective == chess.WHITE else range(7, -1, -1)

    file_header = [Text("  ", style="dim")]
    for f in file_order:
        file_header.append(Text(f" {files[f]} ", style="dim"))
    grid.add_row(*file_header)

    highlights = set()
    if last_move is not None:
        highlights = {last_move.from_square, last_move.to_square}
    check_sq = board.king(board.turn) if board.is_check() else None

    for rank in ranks:
        row = [Text(f" {rank + 1} ", style="dim")]
        for f in file_order:
            sq = chess.square(f, rank)
            piece = board.piece_at(sq)
            light_sq = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1

            if sq in highlights:
                bg = "board.highlight_light" if light_sq else "board.highlight_dark"
            else:
                bg = "board.light" if light_sq else "board.dark"

            if piece is None:
                cell = Text("   ", style=bg)
            else:
                glyph = _PIECE_GLYPH[piece.symbol()]
                fg = "board.white_piece" if piece.color == chess.WHITE else "board.black_piece"
                style = f"{bg} {fg}"
                if sq == check_sq:
                    style = f"on red {fg}"
                cell = Text(f" {glyph} ", style=style)
            row.append(cell)
        grid.add_row(*row)

    return Panel(
        Align.center(grid),
        title=title or board.fen().split()[0],
        title_align="left",
        border_style="muted",
        padding=(0, 1),
    )


def _value_bar(value: float, width: int = 24) -> Text:
    """horizontal bar from -1 (black) to +1 (white) with a marker at value."""
    value = max(-1.0, min(1.0, value))
    half = width // 2
    text = Text()
    pos = int(round((value + 1) * (width - 1) / 2))
    for i in range(width):
        if i == pos:
            text.append("█", style="value.white" if value >= 0 else "value.black")
        elif i < half:
            text.append("░", style="value.black")
        elif i > half:
            text.append("░", style="value.white")
        else:
            text.append("│", style="dim")
    return text


def eval_panel(
    value: float,
    top_moves: list[tuple[chess.Move, float]],
    board: chess.Board,
    title: str = "engine",
) -> Panel:
    bar = _value_bar(value)
    val_line = Text()
    val_line.append("value ", style="muted")
    sign_style = "value.white" if value >= 0 else "value.black"
    val_line.append(f"{value:+.3f}  ", style=sign_style)
    val_line.append(bar)

    moves_table = Table.grid(padding=(0, 1))
    moves_table.add_column(style="accent", no_wrap=True)
    moves_table.add_column(no_wrap=True)
    moves_table.add_column(style="muted", no_wrap=True)
    for mv, prob in top_moves[:3]:
        san = board.san(mv)
        bar_w = 18
        filled = int(round(prob * bar_w))
        bar_text = Text("█" * filled, style="success") + Text(
            "░" * (bar_w - filled), style="dim"
        )
        moves_table.add_row(san, bar_text, f"{prob * 100:5.1f}%")

    return Panel(
        Group(val_line, Text(""), moves_table),
        title=title,
        title_align="left",
        border_style="accent",
        padding=(0, 1),
    )


def move_history(board: chess.Board, max_pairs: int = 12) -> Panel:
    """last ``max_pairs`` move pairs in san."""
    moves = list(board.move_stack)
    replay = chess.Board()
    pairs: list[tuple[str, str]] = []
    current_pair: list[str] = []
    for mv in moves:
        san = replay.san(mv)
        replay.push(mv)
        current_pair.append(san)
        if len(current_pair) == 2:
            pairs.append((current_pair[0], current_pair[1]))
            current_pair = []
    if current_pair:
        pairs.append((current_pair[0], ""))

    visible = pairs[-max_pairs:]
    start_num = len(pairs) - len(visible) + 1

    table = Table.grid(padding=(0, 1))
    table.add_column(style="dim", no_wrap=True, justify="right")
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    for i, (w, b) in enumerate(visible):
        table.add_row(f"{start_num + i}.", w, b)

    return Panel(
        table or Text("(no moves yet)", style="dim"),
        title="moves",
        title_align="left",
        border_style="muted",
        padding=(0, 1),
    )


@dataclass
class MatchState:
    n_games: int
    label_a: str = "A"
    label_b: str = "B"
    wins: int = 0
    losses: int = 0
    draws: int = 0
    elo_diff: float = 0.0
    elo_err: float = 0.0
    completed_results: list[str] = field(default_factory=list)  # 'W'/'L'/'D'
    log_tail: list[str] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return self.wins + self.losses + self.draws

    def update(self, w: int, l: int, d: int) -> None:
        prev_total = self.completed
        self.wins, self.losses, self.draws = w, l, d
        new_total = self.completed
        if new_total <= prev_total:
            return
        # append new results in best-effort order from cumulative counts.
        delta = new_total - prev_total
        for _ in range(delta):
            if w > self.completed_results.count("W"):
                self.completed_results.append("W")
            elif l > self.completed_results.count("L"):
                self.completed_results.append("L")
            else:
                self.completed_results.append("D")


def match_progress(state: MatchState) -> RenderableType:
    """live body for an in-progress cutechess match."""
    bar = ProgressBar(total=state.n_games, completed=state.completed, width=40)

    counts = Table.grid(padding=(0, 2))
    counts.add_column()
    counts.add_column()
    counts.add_column()
    counts.add_row(
        Text(f" W {state.wins} ", style="success on grey15"),
        Text(f" L {state.losses} ", style="error on grey15"),
        Text(f" D {state.draws} ", style="warning on grey15"),
    )

    elo_line = Text()
    elo_line.append(f"{state.label_a} vs {state.label_b}    ", style="muted")
    elo_line.append("Elo Δ ", style="muted")
    if state.completed and state.elo_err == state.elo_err:  # not nan
        elo_line.append(f"{state.elo_diff:+.1f}", style="accent")
        elo_line.append(f" ± {state.elo_err:.1f}", style="muted")
    else:
        elo_line.append("(pending)", style="dim")

    strip = Text()
    for r in state.completed_results[-state.n_games:]:
        if r == "W":
            strip.append("■ ", style="success")
        elif r == "L":
            strip.append("■ ", style="error")
        else:
            strip.append("■ ", style="warning")

    body = Group(
        elo_line,
        Text(""),
        bar,
        Text(""),
        counts,
        Text(""),
        strip if strip else Text(""),
    )
    return Panel(body, title="match", title_align="left", border_style="header", padding=(1, 2))


def tail_log(lines: Iterable[str], max_lines: int = 6) -> Panel:
    snapshot = list(lines)[-max_lines:]
    body = Text("\n".join(snapshot) if snapshot else "(no output yet)", style="dim")
    return Panel(body, title="cutechess", title_align="left", border_style="muted", padding=(0, 1))


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: Sequence[float], width: int = 72) -> Text:
    """terminal sparkline, stable enough for live refreshes."""
    if not values:
        return Text("(no data yet)", style="dim")
    vals = list(values[-width:])
    lo = min(vals)
    hi = max(vals)
    span = max(hi - lo, 1e-12)
    line = Text()
    for v in vals:
        idx = int(round((v - lo) / span * (len(_SPARK_CHARS) - 1)))
        line.append(_SPARK_CHARS[idx], style="success")
    return line


def trend_panel(
    title: str,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    border_style: str = "muted",
) -> Panel:
    """compact scalar card without bitmap rendering."""
    if not xs or not ys:
        return Panel(
            Text("(no data yet)", style="dim"),
            title=title,
            title_align="left",
            border_style=border_style,
            padding=(0, 1),
        )
    vals = list(ys)
    current = vals[-1]
    best = min(vals) if "loss" in title.lower() else max(vals)
    meta = Text()
    meta.append("current ", style="muted")
    meta.append(f"{current:.4f}", style="accent")
    meta.append("   best ", style="muted")
    meta.append(f"{best:.4f}", style="success")
    meta.append("   points ", style="muted")
    meta.append(str(len(vals)), style="accent")
    body = Group(meta, Text(""), _sparkline(vals))
    return Panel(
        body,
        title=title,
        title_align="left",
        border_style=border_style,
        padding=(0, 1),
    )


def scalar_panel(label: str, value: str, *, border_style: str = "muted") -> Panel:
    """single-line label/value panel."""
    body = Text()
    body.append(f"{label}\n", style="muted")
    body.append(value, style="accent")
    return Panel(body, border_style=border_style, padding=(0, 1))


def policy_overlay_panel(
    board: chess.Board,
    top_moves: list[tuple[chess.Move, float]],
    value: float | None = None,
) -> Panel:
    """board plus top move bars."""
    board_table = Table.grid(padding=0, collapse_padding=True)
    for _ in range(9):
        board_table.add_column(no_wrap=True)
    files = "abcdefgh"
    file_header = [Text("  ", style="dim")] + [
        Text(f" {files[f]} ", style="dim") for f in range(8)
    ]
    board_table.add_row(*file_header)
    for rank in range(7, -1, -1):
        row = [Text(f" {rank + 1} ", style="dim")]
        for f in range(8):
            sq = chess.square(f, rank)
            piece = board.piece_at(sq)
            light_sq = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            bg = "board.light" if light_sq else "board.dark"
            if piece is None:
                row.append(Text("   ", style=bg))
            else:
                glyph = _PIECE_GLYPH[piece.symbol()]
                fg = (
                    "board.white_piece" if piece.color == chess.WHITE
                    else "board.black_piece"
                )
                row.append(Text(f" {glyph} ", style=f"{bg} {fg}"))
        board_table.add_row(*row)

    moves_table = Table.grid(padding=(0, 1))
    moves_table.add_column(style="accent", no_wrap=True)
    moves_table.add_column(no_wrap=True)
    moves_table.add_column(style="muted", no_wrap=True)
    for mv, prob in top_moves[:5]:
        try:
            san = board.san(mv)
        except Exception:
            san = mv.uci()
        bar_w = 12
        filled = int(round(prob * bar_w))
        bar_text = (
            Text("█" * filled, style="success")
            + Text("░" * (bar_w - filled), style="dim")
        )
        moves_table.add_row(san, bar_text, f"{prob * 100:5.1f}%")

    side = Group(
        Text(f"value {value:+.3f}" if value is not None else "", style="accent"),
        Text(""),
        moves_table,
    )

    inner = Table.grid(padding=(0, 2))
    inner.add_column()
    inner.add_column()
    inner.add_row(board_table, side)
    return Panel(inner, title="latest position", title_align="left",
                 border_style="accent", padding=(0, 1))


@dataclass
class TrainState:
    run_name: str
    step: int = 0
    total_steps: int = 0
    wall_seconds: float = 0.0
    eta_seconds: float = 0.0
    ema_loss: float = 0.0
    ema_acc: float = 0.0
    last_lr: float = 0.0
    last_grad_norm: float = 0.0
    plies_per_sec: float = 0.0
    loss_history: list[tuple[int, float]] = field(default_factory=list)
    elo_history: list[tuple[int, float]] = field(default_factory=list)
    log_tail_lines: list[str] = field(default_factory=list)
    sample_board: chess.Board | None = None
    sample_top_moves: list[tuple[chess.Move, float]] = field(default_factory=list)
    sample_value: float | None = None


def _fmt_seconds(s: float) -> str:
    s = int(max(0, s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}"


def train_header(state: TrainState) -> Panel:
    title = Text()
    title.append(state.run_name, style="header")
    title.append(f"  ·  step {state.step}/{state.total_steps}", style="muted")
    title.append(
        f"  ·  wall {_fmt_seconds(state.wall_seconds)}", style="muted"
    )
    title.append(f"  ·  ETA {_fmt_seconds(state.eta_seconds)}", style="muted")
    return Panel(Align.center(title), border_style="header", padding=(0, 2))


def train_layout(state: TrainState) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(train_header(state), name="header", size=3),
        Layout(name="body", ratio=1),
        Layout(name="footer", size=8),
    )

    layout["body"].split_column(
        Layout(name="upper", ratio=1),
        Layout(name="lower", ratio=1),
    )

    # upper row: loss plot | scalars
    layout["body"]["upper"].split_row(
        Layout(name="loss", ratio=2),
        Layout(name="scalars", ratio=1),
    )
    loss_xs = [s for s, _ in state.loss_history]
    loss_ys = [v for _, v in state.loss_history]
    layout["body"]["upper"]["loss"].update(
        trend_panel("loss EMA", loss_xs, loss_ys, border_style="success")
    )
    scalars = Group(
        scalar_panel("lr", f"{state.last_lr:.2e}"),
        scalar_panel("grad norm", f"{state.last_grad_norm:.3f}"),
        scalar_panel("plies/s", f"{state.plies_per_sec:.0f}"),
        scalar_panel("ema acc", f"{state.ema_acc * 100:.2f}%"),
    )
    layout["body"]["upper"]["scalars"].update(
        Panel(scalars, border_style="muted", padding=(0, 1), title="scalars",
              title_align="left")
    )

    # lower row: policy overlay | elo curve
    layout["body"]["lower"].split_row(
        Layout(name="board", ratio=1),
        Layout(name="elo", ratio=1),
    )
    if state.sample_board is not None:
        layout["body"]["lower"]["board"].update(
            policy_overlay_panel(
                state.sample_board, state.sample_top_moves, state.sample_value
            )
        )
    else:
        layout["body"]["lower"]["board"].update(
            Panel(Text("(awaiting first batch)", style="dim"),
                  title="latest position", title_align="left",
                  border_style="muted")
        )
    elo_xs = [s for s, _ in state.elo_history]
    elo_ys = [v for _, v in state.elo_history]
    layout["body"]["lower"]["elo"].update(
        trend_panel("Elo Δ vs stockfish-0", elo_xs, elo_ys, border_style="success")
    )

    layout["footer"].update(tail_log(state.log_tail_lines, max_lines=6))
    return layout


def result_table(result: dict) -> Panel:
    """final scorecard panel."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="muted", justify="right")
    table.add_column(style="accent")

    def _row(k: str, v: str) -> None:
        table.add_row(k, v)

    if "checkpoint" in result:
        _row("checkpoint", str(result["checkpoint"]))
    if "opponent" in result:
        _row("opponent", str(result["opponent"]))
    _row("games", str(result.get("n_games", result["wins"] + result["losses"] + result["draws"])))
    score_line = Text()
    score_line.append(f"{result['wins']}", style="success")
    score_line.append(" - ", style="dim")
    score_line.append(f"{result['losses']}", style="error")
    score_line.append(" - ", style="dim")
    score_line.append(f"{result['draws']}", style="warning")
    table.add_row("W-L-D", score_line)
    if "score" in result:
        _row("score", f"{result['score']:.1f}")
    if "win_rate" in result:
        _row("win rate", f"{result['win_rate'] * 100:.1f}%")
    elo = result.get("elo_diff", 0.0)
    err = result.get("elo_err", 0.0)
    if err == err:  # not nan
        _row("Elo Δ", f"{elo:+.1f} ± {err:.1f}")
    else:
        _row("Elo Δ", f"{elo:+.1f} (err n/a)")
    if "pgn_path" in result:
        _row("pgn", result["pgn_path"])

    return Panel(
        table,
        title="result",
        title_align="left",
        border_style="success",
        padding=(1, 2),
    )
