import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import PlayPage from "../src/pages/PlayPage";
import { requestModelMove, type ModelMove } from "../src/lib/inference";

type MockChessboardOptions = {
  onPieceDrop: (move: { sourceSquare: string; targetSquare: string | null }) => boolean;
  arrows: Array<{ startSquare: string; endSquare: string; color: string }>;
  position: string;
};

const chessboardOptions = vi.hoisted(() => ({
  current: null as MockChessboardOptions | null,
}));

vi.mock("react-chessboard", () => ({
  Chessboard: ({ options }: { options: MockChessboardOptions }) => {
    chessboardOptions.current = options;
    return <div data-testid="play-board" />;
  },
}));

vi.mock("../src/lib/inference", () => ({
  inferenceApiRoot: "/api",
  requestModelMove: vi.fn(),
}));

const requestModelMoveMock = vi.mocked(requestModelMove);

function modelReply(move: string, san: string): ModelMove {
  return {
    move,
    san,
    fen: "",
    value: 0,
    simulations: 128,
    elapsed_ms: 10,
    top_moves: [],
  };
}

async function playTurn(
  sourceSquare: string,
  targetSquare: string,
  replyFrom: string,
  replyTo: string,
) {
  act(() => {
    expect(chessboardOptions.current?.onPieceDrop({ sourceSquare, targetSquare })).toBe(true);
  });
  await waitFor(() => {
    expect(screen.getByLabelText(`Previous move from ${replyFrom} to ${replyTo}`)).toBeInTheDocument();
  });
}

describe("PlayPage game controls", () => {
  beforeEach(() => {
    requestModelMoveMock.mockReset();
    chessboardOptions.current = null;
    requestModelMoveMock.mockResolvedValue(modelReply("e7e5", "e5"));
  });

  it("waits for Start and ends an active game on resignation", () => {
    render(<PlayPage />);

    expect(screen.getByText("Ready when you are")).toBeInTheDocument();
    expect(screen.getByText("10:00")).toBeInTheDocument();
    expect(screen.getByText("white · 10:00")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "resign" })).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "start game" }));
    expect(screen.getByText("White to move")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "resign" })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: "resign" }));
    const result = screen.getByRole("alert");
    expect(within(result).getByText("You resigned.")).toBeInTheDocument();
    expect(within(result).getByText("Kibitzer wins as Black.")).toBeInTheDocument();
  });

  it("starts with unlimited game time when selected", () => {
    render(<PlayPage />);

    const unlimited = screen.getByRole("button", { name: "unlimited" });
    fireEvent.click(unlimited);

    expect(unlimited).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByLabelText("Unlimited time")).toHaveTextContent("unlimited");
    expect(screen.getByText("white · unlimited")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "start game" }));
    expect(screen.getByText("White to move")).toBeInTheDocument();
    expect(unlimited).toBeDisabled();
    expect(screen.getByText("no limit")).toBeInTheDocument();
  });

  it("shows the previous move from its origin to its destination", async () => {
    render(<PlayPage />);
    fireEvent.click(screen.getByRole("button", { name: "start game" }));

    act(() => {
      expect(chessboardOptions.current?.onPieceDrop({
        sourceSquare: "e2",
        targetSquare: "e4",
      })).toBe(true);
    });

    await waitFor(() => {
      expect(chessboardOptions.current?.arrows).toEqual([
        {
          startSquare: "e7",
          endSquare: "e5",
          color: "hsl(var(--kibitzer) / 0.72)",
        },
      ]);
    });
    expect(screen.getByLabelText("Previous move from e7 to e5")).toBeInTheDocument();
  });

  it("chooses a promotion piece before resuming the game", async () => {
    requestModelMoveMock
      .mockResolvedValueOnce(modelReply("h7h6", "h6"))
      .mockResolvedValueOnce(modelReply("h6h5", "h5"))
      .mockResolvedValueOnce(modelReply("g7g6", "g6"))
      .mockResolvedValueOnce(modelReply("g6g5", "g5"))
      .mockResolvedValueOnce(modelReply("b8c6", "Nc6"));

    render(<PlayPage />);
    fireEvent.click(screen.getByRole("button", { name: "start game" }));

    await playTurn("a2", "a4", "h7", "h6");
    await playTurn("a4", "a5", "h6", "h5");
    await playTurn("a5", "a6", "g7", "g6");
    await playTurn("a6", "b7", "g6", "g5");

    act(() => {
      expect(chessboardOptions.current?.onPieceDrop({
        sourceSquare: "b7",
        targetSquare: "a8",
      })).toBe(false);
    });

    const dialog = screen.getByRole("dialog", { name: "Choose promotion piece" });
    expect(within(dialog).getByText("b7 → a8")).toBeInTheDocument();
    expect(requestModelMoveMock).toHaveBeenCalledTimes(4);

    fireEvent.click(within(dialog).getByRole("button", { name: "Promote to knight" }));

    await waitFor(() => expect(requestModelMoveMock).toHaveBeenCalledTimes(5));
    expect(requestModelMoveMock).toHaveBeenLastCalledWith(
      [
        "a2a4",
        "h7h6",
        "a4a5",
        "h6h5",
        "a5a6",
        "g7g6",
        "a6b7",
        "g6g5",
        "b7a8n",
      ],
      128,
      expect.any(AbortSignal),
    );
    await waitFor(() => {
      expect(screen.getByLabelText("Previous move from b8 to c6")).toBeInTheDocument();
      expect(chessboardOptions.current?.position).toMatch(/^N1bqkbnr\//);
    });
  });

  it("does not request Kibitzer's opening move as Black before Start", async () => {
    render(<PlayPage />);

    fireEvent.click(screen.getByRole("button", { name: "play black" }));
    expect(requestModelMoveMock).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "start game" }));
    await waitFor(() => expect(requestModelMoveMock).toHaveBeenCalledTimes(1));
    expect(requestModelMoveMock).toHaveBeenCalledWith([], 128, expect.any(AbortSignal));
  });
});
