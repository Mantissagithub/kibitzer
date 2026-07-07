# render a pgn game to an mp4: draw each position with pillow (filled unicode
# piece glyphs distinguished by fill color + stroke so they read on any square,
# last-move highlight, caption bar), then encode the frames with ffmpeg.

from __future__ import annotations

import argparse
import io
import subprocess
import tempfile
from pathlib import Path

import chess
import chess.pgn
from PIL import Image, ImageDraw, ImageFont

FILLED = {"K": "♚", "Q": "♛", "R": "♜", "B": "♝", "N": "♞", "P": "♟"}
LIGHT = (237, 214, 176)
DARK = (181, 136, 99)
HL_LIGHT = (246, 236, 132)
HL_DARK = (214, 196, 96)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def render_board(board, last_move, *, title, subtitle, sq=104, margin=30, capbar=118):
    size = sq * 8
    w = size + 2 * margin
    h = size + 2 * margin + capbar
    img = Image.new("RGB", (w, h), (26, 27, 30))
    d = ImageDraw.Draw(img)
    pf = ImageFont.truetype(FONT_PATH, int(sq * 0.80))
    cf = ImageFont.truetype(FONT_PATH, 16)
    tf = ImageFont.truetype(FONT_PATH, 30)
    sf = ImageFont.truetype(FONT_PATH, 24)
    ox, oy = margin, margin

    for rank in range(8):
        for file in range(8):
            # white at the bottom -> rank 0 (row from top) is the 8th rank
            sqi = chess.square(file, 7 - rank)
            x0, y0 = ox + file * sq, oy + rank * sq
            light = (file + rank) % 2 == 0
            color = LIGHT if light else DARK
            if last_move is not None and sqi in (last_move.from_square, last_move.to_square):
                color = HL_LIGHT if light else HL_DARK
            d.rectangle([x0, y0, x0 + sq, y0 + sq], fill=color)
            piece = board.piece_at(sqi)
            if piece is not None:
                ch = FILLED[piece.symbol().upper()]
                fill = (248, 248, 248) if piece.color == chess.WHITE else (22, 22, 22)
                stroke = (22, 22, 22) if piece.color == chess.WHITE else (215, 215, 215)
                bb = d.textbbox((0, 0), ch, font=pf, stroke_width=3)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                d.text(
                    (x0 + sq / 2 - tw / 2 - bb[0], y0 + sq / 2 - th / 2 - bb[1]),
                    ch, font=pf, fill=fill, stroke_width=3, stroke_fill=stroke,
                )

    # rank/file coordinates in the margins
    for i in range(8):
        d.text((ox + i * sq + 4, oy + size + 2), "abcdefgh"[i], font=cf, fill=(150, 150, 150))
        d.text((ox - 16, oy + i * sq + 4), str(8 - i), font=cf, fill=(150, 150, 150))

    # caption bar
    by = oy + size + margin
    d.text((margin, by - 4), title, font=tf, fill=(238, 238, 238))
    d.text((margin, by + 34), subtitle, font=sf, fill=(180, 200, 235))
    return img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a PGN game to an mp4.")
    p.add_argument("--pgn", type=Path, required=True)
    p.add_argument("--game-index", type=int, default=1, help="1-based game to render")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="Kibitzer vs Stockfish")
    p.add_argument("--seconds-per-move", type=float, default=0.9)
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fh = io.StringIO(args.pgn.read_text(encoding="utf-8"))
    game = None
    for _ in range(args.game_index):
        game = chess.pgn.read_game(fh)
        if game is None:
            raise SystemExit("game index out of range")

    board = game.board()
    kib_white = game.headers.get("White") == "Kibitzer"
    side = "White" if kib_white else "Black"
    frames = []

    # opening frame (no move yet)
    frames.append(render_board(board, None, title=args.title, subtitle=f"Kibitzer plays {side}"))

    ply = 0
    for move in game.mainline_moves():
        mover_is_kib = (board.turn == chess.WHITE) == kib_white
        san = board.san(move)
        num = board.fullmove_number
        prefix = f"{num}." if board.turn == chess.WHITE else f"{num}..."
        who = "Kibitzer" if mover_is_kib else "Stockfish"
        board.push(move)
        ply += 1
        tag = ""
        if board.is_checkmate():
            tag = "  #  checkmate"
        elif board.is_check():
            tag = "  +  check"
        frames.append(render_board(board, move, title=args.title, subtitle=f"{prefix} {san}   ({who}){tag}"))

    result = game.headers.get("Result", "*")
    winner = "Kibitzer wins" if ((result == "1-0") == kib_white and result in ("1-0", "0-1")) else result
    # hold the final position with a result caption
    final = render_board(board, None, title=args.title, subtitle=f"{result}   {winner}")
    frames.append(final)

    # frame timing: repeat each rendered position for seconds-per-move; longer holds
    # on the opening and final frames so they read.
    hold = max(1, int(args.fps * args.seconds_per_move))
    per = [hold] * len(frames)
    per[0] = int(args.fps * 1.6)
    per[-1] = int(args.fps * 3.0)

    with tempfile.TemporaryDirectory() as td:
        idx = 0
        for frame, reps in zip(per and frames, per):
            for _ in range(reps):
                frame.save(Path(td) / f"f{idx:05d}.png")
                idx += 1
        args.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(Path(td) / "f%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)],
            check=True, capture_output=True,
        )
    print(f"wrote {args.out} ({len(frames)} positions, {idx} frames)")


if __name__ == "__main__":
    main()
