import { describe, expect, it } from "vitest";
import { parsePgnDocument, splitPgnGames } from "../src/lib/pgn";

const GAME_A = `[Event "Test Open"]
[Site "Somewhere"]
[Date "2024.01.01"]
[Round "1"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0
`;

const GAME_B = `[Event "Test Open"]
[Site "Somewhere"]
[Date "2024.01.02"]
[Round "2"]
[White "Carol"]
[Black "Dave"]
[Result "0-1"]

1. d4 d5 2. c4 e6 0-1
`;

const GAME_WITH_COMMENTS = `[Event "Commented Game"]
[White "Alice"]
[Black "Bob"]
[Result "*"]

1. e4 {best by test} e5 2. Nf3 {developing} Nc6 *
`;

const HEADERLESS_GAME = `1. e4 e5 2. Nf3 Nc6 3. Bb5 a6`;

describe("splitPgnGames", () => {
  it("splits a multi-game document on line-start Event headers", () => {
    const chunks = splitPgnGames(`${GAME_A}\n${GAME_B}`);
    expect(chunks).toHaveLength(2);
    expect(chunks[0]).toContain('[White "Alice"]');
    expect(chunks[1]).toContain('[White "Carol"]');
  });

  it("accepts a single headerless PGN as one game", () => {
    const chunks = splitPgnGames(HEADERLESS_GAME);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toBe(HEADERLESS_GAME);
  });

  it("returns an empty list for blank input", () => {
    expect(splitPgnGames("   \n\n  ")).toEqual([]);
  });
});

describe("parsePgnDocument", () => {
  it("parses a multi-game document into separate game traces", () => {
    const result = parsePgnDocument(`${GAME_A}\n${GAME_B}`);
    expect(result.issues).toEqual([]);
    expect(result.games).toHaveLength(2);
    expect(result.games[0].headers.White).toBe("Alice");
    expect(result.games[1].headers.White).toBe("Carol");
  });

  it("attaches the comment for the position a move produces", () => {
    const result = parsePgnDocument(GAME_WITH_COMMENTS);
    expect(result.issues).toEqual([]);
    const [game] = result.games;
    expect(game.plies[0].san).toBe("e4");
    expect(game.plies[0].comment).toBe("best by test");
    expect(game.plies[1].san).toBe("e5");
    expect(game.plies[1].comment).toBeUndefined();
    expect(game.plies[2].san).toBe("Nf3");
    expect(game.plies[2].comment).toBe("developing");
  });

  it("reports an issue for a malformed game while keeping valid games", () => {
    const brokenGame = `[Event "Broken"]
[White "Eve"]
[Black "Mallory"]
[Result "*"]

1. e4 Zz9 *
`;
    const result = parsePgnDocument(`${GAME_A}\n${brokenGame}\n${GAME_B}`);
    expect(result.games).toHaveLength(2);
    expect(result.games[0].headers.White).toBe("Alice");
    expect(result.games[1].headers.White).toBe("Carol");
    expect(result.issues).toHaveLength(1);
    expect(result.issues[0].gameIndex).toBe(2);
    expect(result.issues[0].message.length).toBeGreaterThan(0);
  });

  it("parses a headerless PGN as a single game with default headers", () => {
    const result = parsePgnDocument(HEADERLESS_GAME);
    expect(result.issues).toEqual([]);
    expect(result.games).toHaveLength(1);
    const [game] = result.games;
    expect(game.headers).toEqual({
      Event: "?",
      Site: "?",
      Date: "????.??.??",
      Round: "?",
      White: "?",
      Black: "?",
      Result: "*",
    });
    expect(game.plies).toHaveLength(6);
    expect(game.plies[0]).toMatchObject({
      index: 0,
      moveNumber: 1,
      color: "w",
      san: "e4",
      from: "e2",
      to: "e4",
    });
    expect(game.plies[0].fen).toBe(
      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    );
  });

  it("produces stable ids and honors the requested source", () => {
    const first = parsePgnDocument(GAME_A, "built-in");
    const second = parsePgnDocument(GAME_A, "built-in");
    expect(first.games[0].id).toBe(second.games[0].id);
    expect(first.games[0].source).toBe("built-in");
    expect(first.games[0].raw).toContain('[Event "Test Open"]');
  });
});
