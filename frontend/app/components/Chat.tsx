"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, type ChatResponse, type Health, type ToolCall } from "../lib/api";
import RecipeDrawer from "./RecipeDrawer";

type Message =
  | { role: "user"; text: string }
  | { role: "assistant"; text: string; tools: ToolCall[]; error?: boolean };

const ID_RE = /^(ar|epi|fn):\S+$/;

const EXAMPLES: { task: string; prompts: string[] }[] = [
  {
    task: "1 · Search & stats",
    prompts: [
      "Find a penne pasta recipe.",
      "How many recipes contain both garlic and onion? What percentage of the dataset is that?",
      "Among recipes containing tomato, which five have the fewest instruction steps?",
    ],
  },
  {
    task: "2 · Ingredient matching",
    prompts: [
      "I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many as possible, no peanuts, at most six instruction steps. Prefer the fewest additional ingredients.",
      "Exclude eggs too.",
    ],
  },
  {
    task: "3 · Substitution",
    prompts: [
      "Take a banana bread recipe and remove eggs. What replacements have evidence from related baked recipes?",
      "Replace the cream in a penne pasta recipe without dairy, preferring examples from the database.",
    ],
  },
];

function newSessionId() {
  return crypto.randomUUID().replace(/-/g, "").slice(0, 12);
}

function ToolTrace({ tools }: { tools: ToolCall[] }) {
  const [open, setOpen] = useState(false);
  if (!tools.length) return null;
  return (
    <div className="mt-3 rounded-lg border border-stone-200 text-xs dark:border-stone-700">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-stone-600 hover:bg-stone-50 dark:text-stone-400 dark:hover:bg-stone-800"
      >
        <span>
          Evidence &amp; tool trace ·{" "}
          {tools.map((t) => t.name).join(" → ")}
        </span>
        <span>{open ? "Hide" : "Show"}</span>
      </button>
      {open && (
        <div className="space-y-3 border-t border-stone-200 p-3 dark:border-stone-700">
          {tools.map((t, i) => (
            <div key={i}>
              <p className="font-mono font-semibold">{t.name}</p>
              <pre className="mt-1 overflow-x-auto rounded bg-stone-100 p-2 dark:bg-stone-800">
                {JSON.stringify(t.args, null, 2)}
              </pre>
              <pre className="mt-1 max-h-72 overflow-auto rounded bg-stone-100 p-2 dark:bg-stone-800">
                {JSON.stringify(t.result, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Chat() {
  const [sessionId, setSessionId] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [state, setState] = useState("");
  const [openId, setOpenId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setSessionId(newSessionId());
    api
      .health()
      .then(setHealth)
      .catch(() => setHealthError("Cannot reach the backend. Start it with: uvicorn app.main:app --port 8000"));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const send = useCallback(
    async (text: string) => {
      const message = text.trim();
      if (!message || loading || !sessionId) return;
      setMessages((m) => [...m, { role: "user", text: message }]);
      setInput("");
      setLoading(true);
      try {
        const res: ChatResponse = await api.chat(message, sessionId);
        setMessages((m) => [...m, { role: "assistant", text: res.reply, tools: res.tool_calls }]);
        setState(res.state);
      } catch (e) {
        setMessages((m) => [
          ...m,
          { role: "assistant", text: `**Error:** ${(e as Error).message}`, tools: [], error: true },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [loading, sessionId],
  );

  const newChat = async () => {
    if (sessionId) await api.reset(sessionId).catch(() => undefined);
    setSessionId(newSessionId());
    setMessages([]);
    setState("");
  };

  return (
    <div className="mx-auto flex h-dvh w-full max-w-4xl flex-col px-4">
      <header className="flex items-center justify-between gap-3 border-b border-stone-200 py-4 dark:border-stone-800">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Recipe Box Chat</h1>
          <p className="text-xs text-stone-500">
            {health
              ? `${health.dataset.usable_recipes.toLocaleString()} usable recipes (of ${health.dataset.raw_records.toLocaleString()} raw records) · ${health.model}`
              : "Grounded answers from the Recipe Box dataset"}
          </p>
        </div>
        <button
          onClick={newChat}
          className="rounded-md border border-stone-300 px-3 py-1.5 text-sm hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800"
        >
          New chat
        </button>
      </header>

      {healthError && (
        <p className="mt-3 rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">{healthError}</p>
      )}
      {health && !health.gemini_key_configured && (
        <p className="mt-3 rounded-md bg-amber-50 p-3 text-sm text-amber-800 dark:bg-amber-950 dark:text-amber-200">
          No Gemini API key found. Copy <code>backend/.env.example</code> to <code>backend/.env</code>, add{" "}
          <code>GEMINI_API_KEY</code>, then restart the backend.
        </p>
      )}

      <main className="flex-1 space-y-5 overflow-y-auto py-5">
        {messages.length === 0 && (
          <div className="space-y-5">
            <p className="text-sm text-stone-600 dark:text-stone-400">
              Ask about recipes, compute dataset statistics, match what is in your fridge, or adapt a recipe. Every
              answer is built from retrieved records or exact computation; the trace under each reply shows the
              evidence. Click an example to start.
            </p>
            {EXAMPLES.map((g) => (
              <div key={g.task}>
                <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-stone-500">{g.task}</p>
                <div className="flex flex-wrap gap-2">
                  {g.prompts.map((p) => (
                    <button
                      key={p}
                      onClick={() => send(p)}
                      className="rounded-full border border-stone-300 px-3 py-1.5 text-left text-sm hover:border-orange-400 hover:bg-orange-50 dark:border-stone-700 dark:hover:bg-stone-800"
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-orange-600 px-4 py-2 text-white">{m.text}</div>
            </div>
          ) : (
            <div key={i} className="flex justify-start">
              <div
                className={`w-full max-w-[95%] rounded-2xl rounded-bl-sm border px-4 py-3 ${
                  m.error
                    ? "border-red-300 bg-red-50 dark:border-red-900 dark:bg-red-950"
                    : "border-stone-200 bg-white dark:border-stone-700 dark:bg-stone-900"
                }`}
              >
                <div className="prose-recipe text-sm leading-6">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      code({ children, className }) {
                        const text = String(children);
                        if (!className && ID_RE.test(text)) {
                          return (
                            <button
                              onClick={() => setOpenId(text)}
                              title="Open the original recipe record"
                              className="rounded bg-orange-100 px-1.5 py-0.5 font-mono text-[0.8em] text-orange-800 underline decoration-dotted hover:bg-orange-200 dark:bg-orange-950 dark:text-orange-200"
                            >
                              {text.length > 18 ? `${text.slice(0, 18)}…` : text}
                            </button>
                          );
                        }
                        return <code className={className}>{children}</code>;
                      },
                      table: ({ children }) => (
                        <div className="my-2 overflow-x-auto">
                          <table className="min-w-full border-collapse text-xs">{children}</table>
                        </div>
                      ),
                      th: ({ children }) => (
                        <th className="border border-stone-300 bg-stone-100 px-2 py-1 text-left dark:border-stone-700 dark:bg-stone-800">
                          {children}
                        </th>
                      ),
                      td: ({ children }) => (
                        <td className="border border-stone-300 px-2 py-1 align-top dark:border-stone-700">{children}</td>
                      ),
                      ul: ({ children }) => <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>,
                      ol: ({ children }) => <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>,
                      p: ({ children }) => <p className="my-2">{children}</p>,
                      h3: ({ children }) => <h3 className="mb-1 mt-3 font-semibold">{children}</h3>,
                      a: ({ children, href }) => (
                        <a href={href} target="_blank" rel="noreferrer" className="text-orange-700 underline">
                          {children}
                        </a>
                      ),
                    }}
                  >
                    {m.text}
                  </ReactMarkdown>
                </div>
                <ToolTrace tools={m.tools} />
              </div>
            </div>
          ),
        )}

        {loading && (
          <div className="flex justify-start">
            <div className="animate-pulse rounded-2xl border border-stone-200 px-4 py-2 text-sm text-stone-500 dark:border-stone-700">
              Searching the dataset…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      {state && state !== "No active search or match yet." && (
        <p className="whitespace-pre-line border-t border-stone-200 pt-2 text-[11px] text-stone-500 dark:border-stone-800">
          Session context · {state.replaceAll("\n", "  |  ")}
        </p>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex gap-2 py-3"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about recipes, stats, ingredients or substitutions…"
          className="flex-1 rounded-xl border border-stone-300 bg-white px-4 py-2.5 text-sm outline-none focus:border-orange-500 dark:border-stone-700 dark:bg-stone-900"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="rounded-xl bg-orange-600 px-5 py-2.5 text-sm font-medium text-white disabled:opacity-50"
        >
          Send
        </button>
      </form>

      {openId && <RecipeDrawer id={openId} onClose={() => setOpenId(null)} />}
    </div>
  );
}
