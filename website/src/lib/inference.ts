export type SearchBudget = 64 | 128 | 512;

export type SearchVisit = {
  move: string;
  visits: number;
  share: number;
};

export type ModelMove = {
  move: string;
  san: string;
  fen: string;
  value: number;
  simulations: SearchBudget;
  elapsed_ms: number;
  top_moves: SearchVisit[];
};

const configuredApiRoot = import.meta.env.VITE_KIBITZER_API_URL?.trim();
export const inferenceApiRoot = (configuredApiRoot || "/api").replace(/\/$/, "");

export async function requestModelMove(
  moves: string[],
  simulations: SearchBudget,
  signal?: AbortSignal,
): Promise<ModelMove> {
  const response = await fetch(`${inferenceApiRoot}/move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ moves, simulations }),
    signal,
  });

  if (!response.ok) {
    let message = `Inference request failed (${response.status})`;
    try {
      const payload = await response.json() as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(message);
  }

  return response.json() as Promise<ModelMove>;
}
