import { describe, expect, it } from "vitest";
import {
  applyBoardMove,
  applyUciMove,
  checkmateResult,
  gameFromMoves,
  gameStatus,
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

  it("reports whose turn it is", () => {
    expect(gameStatus(gameFromMoves([]))).toBe("White to move");
    expect(gameStatus(gameFromMoves(["e2e4"]))).toBe("Black to move");
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
