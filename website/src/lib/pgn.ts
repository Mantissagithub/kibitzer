import { Chess } from "chess.js";

export type GameSource = "built-in" | "imported";

export interface TracePly {
  index: number;
  moveNumber: number;
  color: "w" | "b";
  san: string;
  from: string;
  to: string;
  fen: string;
  comment?: string;
}

export interface GameTrace {
  id: string;
  source: GameSource;
  raw: string;
  headers: Record<string, string>;
  plies: TracePly[];
}

export interface PgnParseIssue {
  gameIndex: number;
  message: string;
}

export interface PgnParseResult {
  games: GameTrace[];
  issues: PgnParseIssue[];
}

const USEFUL_HEADERS = [
  "Event",
  "Site",
  "Date",
  "Round",
  "White",
  "Black",
  "Result",
  "ECO",
  "WhiteElo",
  "BlackElo",
  "TimeControl",
  "Opening",
  "Variation",
  "Termination",
  "PlyCount",
  "GameDuration",
  "WhiteTimeControl",
  "BlackTimeControl",
];

function pickHeaders(all: Record<string, string>): Record<string, string> {
  const picked: Record<string, string> = {};
  for (const key of USEFUL_HEADERS) {
    if (all[key] !== undefined) {
      picked[key] = all[key];
    }
  }
  return picked;
}

function hashString(input: string): string {
  let hash = 5381;
  for (let i = 0; i < input.length; i++) {
    hash = ((hash << 5) + hash + input.charCodeAt(i)) >>> 0;
  }
  return hash.toString(36);
}

function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function buildGameId(
  source: GameSource,
  gameIndex: number,
  headers: Record<string, string>,
  raw: string,
): string {
  const label = slugify(
    [headers.White, headers.Black, headers.Date, headers.Round]
      .filter(Boolean)
      .join("-"),
  );
  return [source, gameIndex, label, hashString(raw)].filter(Boolean).join("-");
}

function firstLine(message: string): string {
  return message.split("\n")[0].trim();
}

/**
 * Splits a PGN document into individual game blocks, split on line-start
 * Event headers. A single headerless PGN (no Event header at all) is
 * accepted as one game.
 */
export function splitPgnGames(doc: string): string[] {
  const normalized = doc.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const lines = normalized.split("\n");
  const eventLineIndices: number[] = [];
  lines.forEach((line, i) => {
    if (/^\[Event\s+"/.test(line)) {
      eventLineIndices.push(i);
    }
  });

  if (eventLineIndices.length === 0) {
    const trimmed = normalized.trim();
    return trimmed.length > 0 ? [trimmed] : [];
  }

  const chunks: string[] = [];
  for (let i = 0; i < eventLineIndices.length; i++) {
    const start = eventLineIndices[i];
    const end = i + 1 < eventLineIndices.length ? eventLineIndices[i + 1] : lines.length;
    const chunk = lines.slice(start, end).join("\n").trim();
    if (chunk.length > 0) {
      chunks.push(chunk);
    }
  }
  return chunks;
}

/**
 * Parses a single or multi-game PGN document into GameTrace objects,
 * parsing each game independently so a malformed game only produces an
 * issue while the remaining valid games are still returned.
 */
export function parsePgnDocument(
  doc: string,
  source: GameSource = "imported",
): PgnParseResult {
  const rawGames = splitPgnGames(doc);
  const games: GameTrace[] = [];
  const issues: PgnParseIssue[] = [];

  rawGames.forEach((raw, i) => {
    const gameIndex = i + 1;
    try {
      const chess = new Chess();
      chess.loadPgn(raw);

      const headers = pickHeaders(chess.getHeaders());
      const commentByFen = new Map(
        chess.getComments().map((c) => [c.fen, c.comment]),
      );
      const history = chess.history({ verbose: true });

      const plies: TracePly[] = history.map((move, index) => ({
        index,
        moveNumber: Math.floor(index / 2) + 1,
        color: move.color,
        san: move.san,
        from: move.from,
        to: move.to,
        fen: move.after,
        comment: commentByFen.get(move.after),
      }));

      games.push({
        id: buildGameId(source, gameIndex, headers, raw),
        source,
        raw,
        headers,
        plies,
      });
    } catch (err) {
      const message = err instanceof Error ? firstLine(err.message) : String(err);
      issues.push({ gameIndex, message });
    }
  });

  return { games, issues };
}
