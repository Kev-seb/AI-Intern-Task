# Recipe Box Chat

A conversational chatbot that **searches, analyses, matches and adapts** recipes from the
[Recipe Box dataset](https://eightportions.com/datasets/Recipes/) (all three JSON files, ~125k records).
Answers are grounded in retrieved records; counts, rankings and averages are computed in code, never guessed by the LLM.

- **Frontend:** Next.js 16 (chat UI, clickable recipe ids, per-answer evidence/tool trace)
- **Backend:** Python / FastAPI + SQLite (FTS5 BM25) + local MiniLM embeddings (fastembed, no API quota)
- **LLM:** Gemini 2.5 Flash (default) **or** any OpenAI-compatible provider (Groq `openai/gpt-oss-120b` tested), selected in
  `backend/.env`. The LLM only understands the request, calls tools and phrases the answer.
- **Runs locally** (no deployment required by the brief; see `docs/DESIGN.md` for why Vercel is a poor fit for the backend)

## Run it

Prerequisites: Python 3.11+, Node 20+, a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/Kev-seb/AI-Intern-Task.git
cd AI-Intern-Task

# 1. backend: install, download dataset, build the index (one-off)
cd backend
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_data.py      # download + parse + embeddings (~30 min on CPU)
#   faster alternative (keyword search only, ~2 min):  python scripts/setup_data.py --skip-embeddings

# 2. add a key (either provider; see .env.example)
copy .env.example .env            # macOS/Linux: cp .env.example .env
#   Gemini: LLM_PROVIDER=gemini  GEMINI_API_KEY=...   (free key: aistudio.google.com/apikey)
#   Groq:   LLM_PROVIDER=groq    GROQ_API_KEY=...     (free key: console.groq.com/keys)

# 3. start the API
uvicorn app.main:app --port 8000

# 4. frontend (new terminal)
cd frontend
npm install
npm run dev                       # open http://localhost:3000
```

The backend never needs a key for tests, tools or the deterministic evaluation set; only the chat endpoint does.

**Free-tier note:** free LLM tiers are small (Gemini: ~20 requests/day for 2.5 Flash; Groq: 8,000 tokens/minute and
200,000 tokens/day). One chat turn makes 1-3 model calls, so replies can take 20-60 s and the daily quota covers a few
dozen turns. The backend retries automatically and shows a clear error if the quota is exhausted.

## Verify it

```bash
cd backend
python -m pytest -q                         # 31 tests: normaliser (8) + router (12) + 11 deterministic evaluation cases
python eval/run_eval.py --md ../docs/EVAL_RESULTS.md
python eval/run_llm_eval.py                 # live: agent vs one-shot LLM (needs a key) -> docs/EVAL_LLM.md
```

## What each task does

| Task | Entry point | Highlights |
|---|---|---|
| 1 Search + Q&A | `search_recipes`, `get_recipe`, `dataset_stats` | Hybrid BM25 + embedding search; the active search persists across turns ("I cannot have dairy" refines it); stats state filters, denominator, calculation, tie info and SQL |
| 2 Ingredient matching | `match_by_ingredients` | Explainable score (usage × completeness × optional steps), matched/missing ingredients, quantity checks only when units convert, hard constraints never relaxed silently |
| 3 Substitution | `adapt_recipe` | Role-aware candidates, local evidence tiers (direct / moderate / weak / none) with recipe ids, lift vs. recipes that keep the ingredient, updated ingredient list + revised instructions, programmatic re-verification, optional labelled web lookup |

Try: *"Find a penne pasta recipe."* → *"I cannot have dairy."* · *"How many recipes contain both garlic and onion? What percentage is that?"* ·
*"I have eggs, onions, tomatoes and rice… no peanuts, at most six steps"* → *"Exclude eggs too."* · *"Take a banana bread recipe and remove eggs…"*

## Repo layout

```
backend/app/        normalize.py (parsing, synonyms, allergen rules) · store.py (SQLite, filters, hybrid search)
                    stats.py · match.py · substitute.py · tools.py (LLM tools) · router.py (forces dataset_stats
                    for statistical questions) · session.py (per-conversation state) · llm.py (provider clients)
                    agent.py (Gemini loop) · agent_openai.py (OpenAI-compatible loop) · main.py (FastAPI)
backend/scripts/    setup_data.py, build_db.py, build_embeddings.py, ...
backend/eval/       cases.py (deterministic eval) · run_eval.py · run_llm_eval.py (live, vs one-shot LLM)
backend/tests/      pytest wrappers
frontend/           Next.js chat UI
docs/               DESIGN.md (decisions, limitations, AI disclosure) · EVAL_RESULTS.md (deterministic suite)
                    EVAL_LLM.md / .json (live agent vs one-shot, Groq) · EVAL_LLM_gemini_L2.md (L2 re-run on Gemini)
```

## Evaluation summary

| Suite | Cases | Result |
|---|---|---|
| Deterministic ([`docs/EVAL_RESULTS.md`](docs/EVAL_RESULTS.md)) | 11 (multi-turn search, aggregate stats, step ranking, missing attributes, matching + follow-up, no-match, quantities, weak-evidence substitution, cream→dairy-free, refused substitution, normalisation) | 11 / 11 pass |
| Live LLM vs one-shot ([`docs/EVAL_LLM.md`](docs/EVAL_LLM.md)) | 6 conversations (L1-L6) | 5 / 6 pass on Groq `gpt-oss-120b`; L2 failed (model browsed with `search_recipes` instead of computing) → fixed by `router.py`, passes on Gemini ([`docs/EVAL_LLM_gemini_L2.md`](docs/EVAL_LLM_gemini_L2.md)) |

In every live case the one-shot LLM (same model, no tools) either guesses, asks for the data, or invents dataset facts;
see the side-by-side answers in `docs/EVAL_LLM.md`.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the design decisions, data findings, what worked / failed, and the AI-tool disclosure.
