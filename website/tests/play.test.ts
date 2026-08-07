import { describe, expect, it } from "vitest";
import {
  applyBoardMove,
  applyUciMove,
  checkmateResult,
  formatHumanClock,
  gameFromMoves,
  gameStatus,
  HUMAN_TIME_LIMIT_MS,
  isHumanLowTime,
  remainingHumanTime,
  shouldRunHumanClock,
} from "@/lib/play";

describe("play game state", () => {
  it("rebuilds a game from UCI history", () => {
    const game = gameFromMoves(["e2e4", "e7e5", "g1f3"]);
    expect(game.fen()).toContain(" b ");
    expect(game.history()).toEqual(["e4", "e5", "Nf3"]);
  });

  it("accepts legal human and model moves", () => {
    const human = applyBoardMove([], "e2", "e4");
    expect(human?.uci).toBe("e2e4");
    const model = applyUciMove(human?.moves ?? [], "e7e5");
    expect(model?.moves).toEqual(["e2e4", "e7e5"]);
  });

  it("rejects illegal moves without changing the record", () => {
    expect(applyBoardMove([], "e2", "e5")).toBeNull();
  });

  it("requires an explicit piece for pawn promotion", () => {
    const moves = [
      "a2a4",
      "h7h6",
      "a4a5",
      "h6h5",
      "a5a6",
      "g7g6",
      "a6b7",
      "g6g5",
    ];

    expect(applyBoardMove(moves, "b7", "a8")).toBeNull();
    expect(applyBoardMove(moves, "b7", "a8", "n")?.uci).toBe("b7a8n");
  });

  it("reports whose turn it is", () => {
    expect(gameStatus(gameFromMoves([]))).toBe("White to move");
    expect(gameStatus(gameFromMoves(["e2e4"]))).toBe("Black to move");
  });

  it("formats and identifies the final minute of the human clock", () => {
    expect(formatHumanClock(HUMAN_TIME_LIMIT_MS)).toBe("10:00");
    expect(formatHumanClock(59_000)).toBe("0:59");
    expect(formatHumanClock(0)).toBe("0:00");
    expect(isHumanLowTime(60_000)).toBe(true);
    expect(isHumanLowTime(60_001)).toBe(false);
    expect(remainingHumanTime(61_000, 1_000)).toBe(60_000);
    expect(remainingHumanTime(61_000, 62_000)).toBe(0);
  });

  it("runs the clock only while the human is on move", () => {
    expect(shouldRunHumanClock(gameFromMoves([]), "white", false, false)).toBe(true);
    expect(shouldRunHumanClock(gameFromMoves(["e2e4"]), "white", false, false)).toBe(false);
    expect(shouldRunHumanClock(gameFromMoves([]), "black", false, false)).toBe(false);
    expect(shouldRunHumanClock(gameFromMoves(["e2e4"]), "black", false, false)).toBe(true);
    expect(shouldRunHumanClock(gameFromMoves([]), "white", true, false)).toBe(false);
    expect(shouldRunHumanClock(gameFromMoves([]), "white", false, true)).toBe(false);
  });

  it("identifies a Kibitzer checkmate against the player", () => {
    const game = gameFromMoves(["f2f3", "e7e5", "g2g4", "d8h4"]);

    expect(gameStatus(game)).toBe("Black wins by checkmate");
    expect(checkmateResult(game, "white")).toEqual({
      winner: "black",
      winnerName: "Black",
      title: "Kibitzer wins",
    });
  });

  it("identifies a player checkmate against Kibitzer", () => {
    const game = gameFromMoves([
      "e2e4",
      "e7e5",
      "f1c4",
      "b8c6",
      "d1h5",
      "g8f6",
      "h5f7",
    ]);

    expect(gameStatus(game)).toBe("White wins by checkmate");
    expect(checkmateResult(game, "white")).toEqual({
      winner: "white",
      winnerName: "White",
      title: "You win",
    });
  });
});
