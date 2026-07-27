import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { GameBoard } from "../src/components/games/GameBoard";
import { kibitzerResult } from "../src/components/games/GameLibrary";
import type { GameTrace } from "../src/lib/pgn";

vi.mock("react-chessboard", () => ({
  Chessboard: ({ options }: { options: { position: string; boardOrientation: string } }) => (
    <div data-testid="board" data-position={options.position} data-orientation={options.boardOrientation} />
  ),
}));

const game: GameTrace = {
  id: "test-game",
  source: "built-in",
  raw: "1. e4 e5",
  headers: {
    White: "Kibitzer-s512",
    Black: "SF-2500",
    Result: "1-0",
    Round: "1",
    Opening: "King pawn",
  },
  plies: [
    {
      index: 0,
      moveNumber: 1,
      color: "w",
      san: "e4",
      from: "e2",
      to: "e4",
      fen: "after-e4",
      comment: "1.7s",
    },
    {
      index: 1,
      moveNumber: 1,
      color: "b",
      san: "e5",
      from: "e7",
      to: "e5",
      fen: "after-e5",
    },
  ],
};

describe("kibitzerResult", () => {
  it("reads wins, losses, and draws from either color", () => {
    expect(kibitzerResult(game)).toBe("win");
    expect(kibitzerResult({ ...game, headers: { ...game.headers, Result: "0-1" } })).toBe("loss");
    expect(kibitzerResult({ ...game, headers: { ...game.headers, White: "SF", Black: "Kibitzer", Result: "0-1" } })).toBe("win");
    expect(kibitzerResult({ ...game, headers: { ...game.headers, Result: "1/2-1/2" } })).toBe("draw");
  });
});

describe("GameBoard", () => {
  it("renders the active position and calls playback controls", () => {
    const onSeek = vi.fn();
    const onPlayingChange = vi.fn();
    const onOrientationChange = vi.fn();

    render(
      <GameBoard
        game={game}
        activePly={0}
        playing={false}
        speed={1}
        orientation="white"
        onSeek={onSeek}
        onPlayingChange={onPlayingChange}
        onSpeedChange={vi.fn()}
        onOrientationChange={onOrientationChange}
      />,
    );

    expect(screen.getByTestId("board")).toHaveAttribute("data-position", "after-e4");
    fireEvent.click(screen.getByRole("button", { name: "Next move" }));
    expect(onSeek).toHaveBeenCalledWith(1);
    fireEvent.click(screen.getByRole("button", { name: "Play game" }));
    expect(onPlayingChange).toHaveBeenCalledWith(true);
    fireEvent.click(screen.getByRole("button", { name: "Flip board" }));
    expect(onOrientationChange).toHaveBeenCalledWith("black");
  });

  it("restarts before playing again at the final ply", () => {
    const onSeek = vi.fn();
    const onPlayingChange = vi.fn();
    render(
      <GameBoard
        game={game}
        activePly={1}
        playing={false}
        speed={1}
        orientation="white"
        onSeek={onSeek}
        onPlayingChange={onPlayingChange}
        onSpeedChange={vi.fn()}
        onOrientationChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Play game" }));
    expect(onSeek).toHaveBeenCalledWith(-1);
    expect(onPlayingChange).toHaveBeenCalledWith(true);
  });
});
