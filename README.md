# Recipe Box Chat

A conversational chatbot that **searches, analyses, matches and adapts** recipes from the
[Recipe Box dataset](https://eightportions.com/datasets/Recipes/): all three JSON files, about 125,000 records.

The focus is **reliable, grounded answers**. The LLM only interprets the request, chooses tools and writes the reply.
Every recipe id, ingredient line, count, percentage, ranking, score and substitution comes from deterministic Python/SQL
code over the dataset, and every answer shows the tool calls and evidence behind it.

---

## Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Tech stack](#tech-stack)
4. [Quick start](#quick-start)
5. [Configuration](#configuration)
6. [Using the chatbot](#using-the-chatbot)
7. [How each task is solved](#how-each-task-is-solved)
8. [Testing and evaluation](#testing-and-evaluation)
9. [API reference](#api-reference)
10. [Project structure](#project-structure)
11. [Troubleshooting](#troubleshooting)
12. [Limitations](#limitations)
13. [Further documentation](#further-documentation)

---

## Features

- **Search and Q&A (Task 1):** search by dish name, ingredient or natural language. Results show the title, a stable
  recipe id and, on request, the original ingredients and instructions. Follow-ups such as *"I cannot have dairy"* refine
  the **active** search instead of starting over.
- **Statistics computed in code:** counts, percentages, most common ingredients, average/median ingredient and step
  counts, and rankings by number of steps. Each answer states the filters, the number of recipes analysed, the
  denominator, the calculation and the SQL used.
- **Ingredient matching (Task 2):** give the ingredients you have, hard constraints (*no peanuts, at most six steps*) and
  preferences (*fewest missing ingredients*). You get a ranked list with an explainable score, matched / missing / extra
  ingredients and quantity checks. Hard constraints are never relaxed silently.
- **Evidence-based substitution (Task 3):** remove or replace an ingredient (*banana bread without eggs*, *pasta without
  cream*). Candidates are chosen by the ingredient's role, supported by recipe ids from the dataset, graded by evidence
  strength, and returned as an updated ingredient list with revised instructions that are re-checked against the
  restriction.
- **Honest refusals:** attributes the dataset doesn't contain (nutrition, cooking time, servings, price) are declined,
  not estimated. External facts are clearly labelled.
- **Transparent UI:** clickable recipe ids open the full original recipe, and each answer has an
  **"Evidence & tool trace"** panel showing exactly what the tools returned.

## Architecture

```
┌──────────────────┐   HTTP    ┌──────────────────────────────────────────────────────────┐
│  Next.js chat UI │ ────────▶ │  FastAPI backend                                         │
│  (localhost:3000)│ ◀──────── │                                                          │
└──────────────────┘   JSON    │  router.py ─ forces dataset_stats for statistical asks   │
                               │  agent.py / agent_openai.py ─ LLM tool-calling loop       │
                               │        │ (Gemini 2.5 Flash or OpenAI-compatible, e.g. Groq)│
                               │        ▼                                                 │
                               │  tools.py                                                │
                               │   ├─ search_recipes / get_recipe ─┐                      │
                               │   ├─ dataset_stats                ├─▶ SQLite (recipes,   │
                               │   ├─ match_by_ingredients         │   ingredients, FTS5) │
                               │   ├─ adapt_recipe                ─┘   + MiniLM embeddings│
                               │   └─ web_lookup (optional, labelled EXTERNAL)            │
                               │  session.py ─ active search / match / recipe per session │
                               └──────────────────────────────────────────────────────────┘
```

A server-side **session** stores the active search, the last results, the recipe being discussed and the active
ingredient match. A follow-up only needs to say what is new.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 16, React 19, Tailwind CSS 4, `react-markdown` |
| Backend | Python 3.11+, FastAPI, Uvicorn, Pydantic |
| Storage / search | SQLite with FTS5 (BM25 keyword search) |
| Embeddings | `all-MiniLM-L6-v2` via `fastembed`, run locally (no API key or quota) |
| LLM | Gemini 2.5 Flash (default) or any OpenAI-compatible endpoint (tested: Groq `openai/gpt-oss-120b`) |
| Tests | `pytest`, plus a deterministic evaluation harness and a live LLM evaluation |

## Quick start

### Prerequisites

- **Python 3.11+** and **Node.js 20+**
- A few hundred MB of free disk space for the dataset, database and embeddings
- An API key for **one** LLM provider (both have free tiers):
  - Gemini: <https://aistudio.google.com/apikey>
  - Groq: <https://console.groq.com/keys>

> The key is only needed for the chat endpoint. Building the data, the tests and the deterministic evaluation run without one.

### 1. Clone the repository

```bash
git clone https://github.com/Kev-seb/AI-Intern-Task.git
cd AI-Intern-Task
```

### 2. Set up the backend

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 3. Download the dataset (one-off)

The dataset host often refuses Python's scripted download (HTTP 403), so download the zip yourself first.
From the **repository root** (the `AI-Intern-Task` folder):

```bash
curl -O https://eightportions.com/recipes_raw.zip
```

Or download <https://eightportions.com/recipes_raw.zip> in a browser and move it to the repository root.
Then extract it into a `data/` folder there:

```bash
# macOS / Linux
unzip recipes_raw.zip -d data
```

```powershell
# Windows (PowerShell)
Expand-Archive recipes_raw.zip -DestinationPath data
```

You should now have `data/recipes_raw_nosource_ar.json`, `data/recipes_raw_nosource_epi.json` and
`data/recipes_raw_nosource_fn.json`.

### 4. Build the index (one-off)

From the `backend` folder, with the virtual environment active:

```bash
python scripts/setup_data.py
```

This runs three steps:

1. `download_data.py`: finds the three JSON files in `data/` (or tries to download them, and prints the manual steps if that is refused)
2. `build_db.py`: parses and normalises every recipe into `backend/storage/recipes.db`
3. `build_embeddings.py`: downloads the small MiniLM model on first use and computes the MiniLM embeddings for hybrid search (**about 25-30 minutes on a CPU**)

**Faster option (about 2 minutes):** skip the embeddings. Search then uses BM25 keyword ranking only, and everything
else works the same. You can run `python scripts/build_embeddings.py` later to enable hybrid search.

```bash
python scripts/setup_data.py --skip-embeddings
```

### 5. Add your API key

```bash
# Windows
copy .env.example .env
```

```bash
# macOS / Linux
cp .env.example .env
```

Then edit `backend/.env` and fill in **one** provider:

```ini
# Option A: Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here

# Option B: Groq
LLM_PROVIDER=groq
GROQ_API_KEY=your-key-here
```

### 6. Start the backend

From the `backend` folder, with the virtual environment active:

```bash
uvicorn app.main:app --port 8000
```

Check it at <http://127.0.0.1:8000/api/health>. It reports the model, whether a key is configured, whether embeddings
are loaded and the dataset counts.

### 7. Start the frontend (in a second terminal)

```bash
cd frontend
npm install
npm run dev
```

Open **<http://localhost:3000>** and start chatting.

## Configuration

All settings live in `backend/.env` (see `backend/.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `groq` (any OpenAI-compatible API) |
| `GEMINI_API_KEY` | (none) | Gemini key (`GOOGLE_API_KEY` also works) |
| `GROQ_API_KEY` | (none) | Key for the OpenAI-compatible provider (`LLM_API_KEY` also works) |
| `LLM_MODEL` | `gemini-2.5-flash` / `openai/gpt-oss-120b` | Override the model name |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1` for Ollama |
| `LLM_DEBUG` | (unset) | Set to any value to log LLM retries and errors |

Frontend: set `NEXT_PUBLIC_API_URL` if the backend is not on `http://127.0.0.1:8000`.

**Free-tier note:** free LLM tiers are small (Gemini 2.5 Flash: about 20 requests/day; Groq: 8,000 tokens/minute and
200,000 tokens/day). One chat turn makes 1-3 model calls, so a reply can take 20-60 seconds and a day's quota covers a
few dozen turns. The backend retries rate-limited calls automatically and shows a clear message once the quota is used up.

## Using the chatbot

Try these conversations (they mirror the assignment's examples):

| Task | You type | What happens |
|---|---|---|
| Search + follow-up | *"Find a penne pasta recipe."* then *"I cannot have dairy."* | The second turn refines the same search and shows dairy-free penne recipes, with ingredient lines as evidence |
| Statistics | *"How many recipes contain both garlic and onion? What percentage of the dataset is that?"* | 9,788 of 122,912 usable recipes (7.96 %) with exact names, 14,729 (11.98 %) including variants such as red onion; denominator and normalisation explained |
| Ranking | *"Among recipes containing tomato, which five have the fewest instruction steps?"* | Five recipe ids ranked by a defined step-counting rule, with ties reported |
| Missing data | *"How many calories does the first one have?"* | Declined: the dataset has no nutrition data |
| Matching | *"I have eggs, onions, tomatoes and rice. Suggest five recipes using as many as possible, no peanuts, at most six steps, fewest extra ingredients."* then *"Exclude eggs too."* | A ranked list with matched / missing ingredients, step counts and score breakdown; the follow-up updates it |
| Substitution | *"Take a banana bread recipe and remove eggs. What replacements have evidence from related baked recipes, and how might the texture change?"* | Candidates with evidence tier and supporting recipe ids, updated ingredients and instructions, expected effects; weak evidence is stated as weak |
| Substitution | *"Replace cream in this pasta recipe without dairy."* | Dairy-free alternatives backed by local recipes, adjusted instructions and trade-offs |

Click any recipe id to open the full original recipe. Open **"Evidence & tool trace"** under an answer to see the raw
tool output. Click **New chat** to start a fresh session.

## How each task is solved

### Task 1: Search and factual Q&A

- **Hybrid retrieval:** SQLite FTS5 BM25 (title weighted 6×) fused with cosine similarity over MiniLM embeddings of
  "title + canonical ingredients", using reciprocal-rank fusion (k = 60).
- **Filters before ranking:** restrictions (exclude dairy, include garlic, at most N steps, …) are applied in SQL to the
  *whole* dataset first, so a restriction can never be lost by the retriever.
- **Duplicates collapsed:** 1,528 exact duplicates are grouped in results (the other ids are listed); junk test records
  are excluded.
- **Statistics:** `dataset_stats` builds SQL from the same filter object used for search, so a count always agrees with
  the list behind it. Supported metrics: `count`, `percentage`, `top_ingredients`, `ingredient_count_stats`,
  `step_count_stats`, `rank_by_steps`, `rank_by_ingredients`, `source_breakdown`.
- **Step rule:** one step per sentence. Each non-empty line also starts a step, and fragments under 3 letters are
  ignored, because the dataset stores instructions as free text.

### Task 2: Ingredient matching

```
score = 0.5 × usage + 0.5 × completeness   (+ 0.15 × step_score when fewer steps are preferred)
usage        = share of YOUR ingredients the recipe uses
completeness = share of the recipe's required (non-pantry) ingredients you have
```

- "Fewest missing ingredients" shifts the weights to 0.4 / 0.6. Variant matches (e.g. *red onion* for *onion*) count 0.8.
- Salt, water and black pepper are treated as pantry items and reported as such.
- Quantities are compared only when units convert (volume, weight, count). Each ingredient is marked *sufficient on
  paper*, *INSUFFICIENT* or *not verified*.
- Hard constraints are SQL filters. If nothing qualifies the result is **no match**, and alternatives are produced by
  relaxing **one constraint at a time**, each labelled `VIOLATES: <constraint>`.

### Task 3: Evidence-based substitution

1. Locate the target ingredient lines (by name or by restricted category such as dairy).
2. Classify the ingredient's **role** (binder, fat, liquid, thickener, …).
3. Build candidates from a role lexicon plus "X or Y" alternatives mined from the dataset, dropping anything that breaks
   the restriction (including *possible* allergens).
4. Gather local evidence: **direct** ("or" lines, "instead of" sentences) and **indirect** (how often the candidate
   appears in similar recipes that lack the restricted item).
5. Control for popularity with **lift** against similar recipes that keep the ingredient, so common items like olive oil
   don't "win" by default.
6. Grade the evidence (**direct / moderate / weak / none**) and flag small samples and conflicting approaches.
7. Build the updated ingredient list and revised instructions, then **re-verify** them against the restriction. If
   nothing is supportable, the tool refuses and says which lines remain unchanged.

Rule-of-thumb amounts (e.g. ¼ cup applesauce per egg) are labelled *EXTERNAL, unverified*. Nothing is called
medically "safe".

### Normalisation and allergen rules

- Each ingredient line is parsed into quantity / unit / name. Preparation words are dropped, plurals singularised and
  synonyms merged (scallion = green onion).
- **Different products stay different:** tomato ≠ tomato sauce, butter ≠ peanut butter, milk ≠ coconut milk.
- Matching has two levels: *exact* and *variant* (harmless modifiers such as red / roma / unsalted).
- Allergen and diet groups use trigger words with exceptions (coconut milk and cream of tartar are not dairy). A separate
  *possible* list (margarine, chocolate, soy sauce, …) produces warnings.

## Testing and evaluation

From the `backend` folder:

```bash
python -m pytest -q
```

Runs **31 tests**: 8 normaliser tests, 12 router tests and the 11 deterministic evaluation cases (the evaluation cases
need the built database and are skipped without it).

```bash
python eval/run_eval.py --md ../docs/EVAL_RESULTS.md
```

Runs the deterministic evaluation set and writes the report. No LLM is involved: each case calls the same tools the
chatbot uses and checks them against an independent recomputation.

```bash
python eval/run_llm_eval.py
```

Runs the live evaluation (needs an API key): scripted conversations through the real agent, compared side by side with
a one-shot answer from the same model without tools. Writes `docs/EVAL_LLM.md`. Use `--only L2,L5` to re-run selected
cases.

### Results

| Suite | Cases | Result |
|---|---|---|
| Deterministic ([`docs/EVAL_RESULTS.md`](docs/EVAL_RESULTS.md)) | 11: multi-turn search, aggregate stats, step ranking, missing attributes, matching + follow-up, no-match, quantities, weak-evidence substitution, cream → dairy-free, refused substitution, normalisation | **11 / 11 pass** |
| Live LLM vs one-shot ([`docs/EVAL_LLM.md`](docs/EVAL_LLM.md)) | 6 conversations (L1-L6) | **5 / 6 pass** on Groq `gpt-oss-120b`. L2 failed (the model browsed search results instead of computing) and was fixed by `router.py`; the re-run passes ([`docs/EVAL_LLM_gemini_L2.md`](docs/EVAL_LLM_gemini_L2.md)) |

In every live case the one-shot LLM either guesses, asks for the data, or invents dataset facts. For example, asked how
many recipes contain garlic and onion, it cannot know; this system computes 9,788 (7.96 % of 122,912) and shows the SQL.

## API reference

The backend runs on `http://127.0.0.1:8000`. Interactive docs are at `/docs`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Model, key status, whether embeddings are loaded, dataset counts |
| `POST` | `/api/chat` | Body `{"session_id": "...", "message": "..."}`. Returns the reply plus the tool trace |
| `POST` | `/api/reset` | Body `{"session_id": "..."}`. Clears that session's context |
| `GET` | `/api/recipe/{recipe_id}` | Full original recipe (title, ingredients, instructions) |

Recipe ids have the form `<source>:<original key>`, e.g. `ar:23to8Xz…`. The source is `ar` (Allrecipes),
`epi` (Epicurious) or `fn` (Food Network).

## Project structure

```
AI-Intern-Task/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app and endpoints
│   │   ├── agent.py           Gemini tool-calling loop and system prompt
│   │   ├── agent_openai.py    OpenAI-compatible (Groq) tool-calling loop
│   │   ├── llm.py             provider settings, API clients, retries
│   │   ├── router.py          forces dataset_stats for clearly statistical questions
│   │   ├── session.py         per-conversation state (active search / match / recipe)
│   │   ├── tools.py           the tools the LLM can call
│   │   ├── store.py           SQLite access, filters, hybrid search
│   │   ├── normalize.py       ingredient parsing, synonyms, allergen rules, step counting
│   │   ├── stats.py           statistics (Task 1)
│   │   ├── match.py           ingredient matching and ranking (Task 2)
│   │   └── substitute.py      evidence-based substitution (Task 3)
│   ├── scripts/               setup_data.py, download_data.py, build_db.py, build_embeddings.py, …
│   ├── eval/                  cases.py, run_eval.py (deterministic), run_llm_eval.py (live)
│   ├── tests/                 pytest: normaliser, router, evaluation cases
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── app/                   page.tsx, components/Chat.tsx, components/RecipeDrawer.tsx, lib/api.ts
├── docs/
│   ├── DESIGN.md              design decisions, data findings, what worked / failed, limitations, AI disclosure
│   ├── EVAL_RESULTS.md        deterministic evaluation report
│   ├── EVAL_LLM.md / .json    live evaluation: agent vs one-shot LLM
│   └── EVAL_LLM_gemini_L2.md  re-run of case L2 on Gemini
└── README.md
```

Generated files (`data/`, `backend/storage/`, `.env`) are git-ignored and rebuilt by `scripts/setup_data.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| Chat returns *503 / missing key* | Create `backend/.env` from `.env.example`, set `LLM_PROVIDER` and the matching key, then restart Uvicorn |
| Chat returns *502 / LLM call failed* or a quota message | The free-tier limit is used up. Wait, or switch provider in `.env` |
| `/api/health` shows `embeddings_loaded: false` | Run `python scripts/build_embeddings.py` (search still works with BM25 only) |
| `setup_data.py` fails with *HTTP Error 403* | Download and extract the zip manually (Quick start, step 3), then run it again |
| Error `recipes.db not found` | The data hasn't been built yet: run `python scripts/setup_data.py` from the `backend` folder |
| Frontend shows a network / CORS error | Make sure the backend is running on port 8000, and open the UI at `localhost:3000` or `127.0.0.1:3000` |
| Evaluation tests are skipped | They need `backend/storage/recipes.db`; build the data first |

## Limitations

- Allergen and role knowledge is rule-based and finite (15 ingredient roles, about 12 allergen/diet groups). It is text
  analysis, **not allergy or medical advice**.
- Step counts are sentence-based approximations, and instruction quality varies by source.
- Quantity checks ignore serving size, which the dataset does not contain.
- Indirect substitution evidence is correlational (similar recipes that lack an ingredient), not tested swaps.
- Statistics count all usable records, including 1,528 exact duplicates, because the brief asks for dataset-level counts;
  search and matching collapse them.
- The LLM sometimes leaves out details the tools returned; the full output is always in the evidence panel.
- Runs locally only, as the brief requires. See `docs/DESIGN.md` for how it could be deployed.

## Further documentation

- [`docs/DESIGN.md`](docs/DESIGN.md): architecture, design decisions, data findings, what worked and what failed,
  known limitations, and the **AI tools disclosure**
- [`docs/EVAL_RESULTS.md`](docs/EVAL_RESULTS.md): deterministic evaluation, case by case
- [`docs/EVAL_LLM.md`](docs/EVAL_LLM.md): live agent vs one-shot LLM, side by side
