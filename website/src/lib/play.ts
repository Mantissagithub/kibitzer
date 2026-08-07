import { Chess, type Square } from "chess.js";

export type HumanColor = "white" | "black";

export const HUMAN_TIME_LIMIT_MS = 10 * 60 * 1000;
export const HUMAN_LOW_TIME_MS = 60 * 1000;

export function remainingHumanTime(deadlineMs: number, nowMs = Date.now()) {
  return Math.max(0, deadlineMs - nowMs);
}

function parseUci(uci: string) {
  const match = /^([a-h][1-8])([a-h][1-8])([qrbn])?$/.exec(uci);
  if (!match) throw new Error(`Invalid UCI move: ${uci}`);
  return {
    from: match[1] as Square,
    to: match[2] as Square,
    promotion: match[3] || undefined,
  };
}

export function gameFromMoves(moves: string[]) {
  const game = new Chess();
  for (const uci of moves) {
    game.move(parseUci(uci));
  }
  return game;
}

export function applyBoardMove(
  moves: string[],
  from: string,
  to: string,
  promotion?: string,
) {
  try {
    const game = gameFromMoves(moves);
    const move = game.move({ from: from as Square, to: to as Square, promotion });
    const uci = `${move.from}${move.to}${move.promotion ?? ""}`;
    return { moves: [...moves, uci], san: move.san, uci };
  } catch {
    return null;
  }
}

export function applyUciMove(moves: string[], uci: string) {
  const parsed = parseUci(uci);
  return applyBoardMove(moves, parsed.from, parsed.to, parsed.promotion);
}

export function turnFor(color: HumanColor) {
  return color === "white" ? "w" : "b";
}

export function formatHumanClock(timeMs: number) {
  const totalSeconds = Math.max(0, Math.ceil(timeMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = String(totalSeconds % 60).padStart(2, "0");
  return `${minutes}:${seconds}`;
}

export function isHumanLowTime(timeMs: number) {
  return timeMs > 0 && timeMs <= HUMAN_LOW_TIME_MS;
}

export function shouldRunHumanClock(
  game: Chess,
  humanColor: HumanColor,
  thinking: boolean,
  lostOnTime: boolean,
) {
  return (
    !thinking
    && !lostOnTime
    && !game.isGameOver()
    && game.turn() === turnFor(humanColor)
  );
}

export function checkmateResult(game: Chess, humanColor: HumanColor) {
  if (!game.isCheckmate()) return null;

  const winner: HumanColor = game.turn() === "w" ? "black" : "white";
  return {
    winner,
    winnerName: winner === "white" ? "White" : "Black",
    title: winner === humanColor ? "You win" : "Kibitzer wins",
  };
}

export function gameStatus(game: Chess) {
  if (game.isCheckmate()) {
    return game.turn() === "w" ? "Black wins by checkmate" : "White wins by checkmate";
  }
  if (game.isStalemate()) return "Draw by stalemate";
  if (game.isThreefoldRepetition()) return "Draw by repetition";
  if (game.isInsufficientMaterial()) return "Draw by insufficient material";
  if (game.isDraw()) return "Draw";
  const side = game.turn() === "w" ? "White" : "Black";
  return game.inCheck() ? `${side} is in check` : `${side} to move`;
}
