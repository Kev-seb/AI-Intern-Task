export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export type ToolCall = {
  name: string;
  args: Record<string, unknown>;
  result?: unknown;
};

export type ChatResponse = {
  reply: string;
  tool_calls: ToolCall[];
  session_id: string;
  state: string;
  model: string;
};

export type Health = {
  ok: boolean;
  model: string;
  gemini_key_configured: boolean;
  embeddings_loaded: boolean;
  dataset: {
    raw_records: number;
    usable_recipes: number;
    per_source: Record<string, { raw: number; complete: number }>;
    exact_duplicate_records: number;
    junk_records: number;
  };
};

export type Recipe = {
  id: string;
  title: string;
  source: string;
  n_ingredients: number;
  n_steps: number;
  ingredients: string[];
  instructions: string | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  chat: (message: string, session_id: string) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, session_id }),
    }),
  reset: (session_id: string) =>
    request<{ ok: boolean }>("/api/reset", {
      method: "POST",
      body: JSON.stringify({ message: "reset", session_id }),
    }),
  recipe: (id: string) => request<Recipe>(`/api/recipe/${encodeURIComponent(id).replace(/%3A/g, ":")}`),
};
