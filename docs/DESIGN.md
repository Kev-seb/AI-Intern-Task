# Design notes

## Architecture

```
Next.js chat UI ──HTTP──▶ FastAPI ──▶ Gemini 2.5 Flash (tool-calling loop)
                                         │ calls only these deterministic tools:
                                         ├─ search_recipes / get_recipe   ─┐
                                         ├─ dataset_stats                  ├─▶ SQLite (recipes, ingredients, flags, FTS5)
                                         ├─ match_by_ingredients           │   + MiniLM embedding matrix (numpy)
                                         ├─ adapt_recipe                  ─┘
                                         └─ web_lookup (optional, labelled EXTERNAL)
```

The LLM never produces a fact on its own: ids, ingredient lines, counts, percentages, rankings, scores and
adaptations all come from tool output, and the system prompt requires the model to quote them and state their provenance.
A server-side **session** keeps the active search, last result ids, last discussed recipe and the active ingredient match,
so follow-ups only need to state what is new.

**LLM providers.** `LLM_PROVIDER=gemini` uses the Gemini API with an explicit tool-calling loop (`agent.py`);
`LLM_PROVIDER=groq` uses any OpenAI-compatible endpoint (`agent_openai.py`, plain `httpx`). Both share the same tools and
system prompt. Because free tiers are tiny, the OpenAI-compatible path sends the model a *compacted* view of each tool result
(the UI trace still shows the full result), keeps tool descriptions short, and every model call retries on rate-limit and
network errors using the server's suggested delay. Smaller models (Groq `gpt-oss-20b`) were tried and rejected: they chose
the wrong tools and exhausted the step budget on the matching task; `gpt-oss-120b` handled it in one call.

## Data findings (all three files used)

| Fact | Value |
|---|---|
| Raw records (ar 39,802 · epi 25,323 · fn 60,039) | 125,164 |
| Usable (title + ≥1 parsed ingredient + ≥1 step) | 122,912 |
| Empty/partial shells | ~2,200 (missing fields; kept in the DB, never returned) |
| Exact duplicates (same title + ingredient list) | 1,534 copies |
| Junk test records (`Tst`, `Tstcgdgd…`) | 8 (flagged unusable) |

Stable id = `<source>:<original hash key>` (e.g. `ar:23to8Xz…`), because the hash keys are only unique within a file.
The original ingredient lines and instruction text are stored untouched (`ingredients_json`, `instructions`); display
copies only strip the scraper's `ADVERTISEMENT` token. Search and matching **collapse duplicates** (and list the other ids);
statistics count every usable record and say so.

## Decisions

**Retrieval.** SQLite FTS5 (BM25, title weighted 6×) fused with cosine similarity over `all-MiniLM-L6-v2` embeddings of
"title + canonical ingredients" using reciprocal-rank fusion (k = 60). Local embeddings were chosen over the Gemini
embedding API so building the index needs no quota or key. Filters are applied to the *complete* set in SQL; the
ranking pool is drawn from inside the filtered set, so a restriction can never be "forgotten" by the retriever.
Without the embedding file the system degrades gracefully to BM25 only.

**Normalisation** (`app/normalize.py`, rule based so every decision is explainable). Each ingredient line is parsed into
quantity / unit / name. Pure preparation words (chopped, softened…) are dropped, plurals singularised, synonyms merged
(scallion = green onion). **Product forms are kept distinct**: tomato ≠ tomato sauce, butter ≠ peanut butter, milk ≠ coconut
milk, chicken ≠ chicken fat. Two match levels: *exact* and *variant* (a harmless modifier such as red/roma/unsalted/dried;
flagged when the form differs). Statistics expose both (`match_mode` strict/exact); e.g. garlic AND onion = 9,788 recipes
(7.96 %) exact vs 14,729 (11.98 %) with variants.

**Allergen / diet rules.** Per-category trigger words with exceptions (coconut milk, cream of tartar, peanut butter ≠ dairy)
plus a separate *possible* list (margarine, chocolate, soy sauce…). *Definite* hits are excluded; *possible* hits are
surfaced as warnings in results and are never offered as a "safe" replacement. Exclusion of a named ingredient is
deliberately broad ("peanuts" also removes peanut butter/oil) because over-excluding is the safe error.

**Computation.** `dataset_stats` builds SQL from the same filter object used for search, so a count and the list behind it
always agree. Every statistic returns filters, `recipes_analyzed`, denominator, the calculation, the normalisation note,
tie information and the SQL. *Step rule:* one step per sentence (each non-empty line also starts a step; fragments < 3 letters
ignored), because the dataset stores instructions as free text.

**Ranking (Task 2).** `score = w_u·usage + w_c·completeness (+ w_s·step_score)` — usage = share of the user's ingredients used,
completeness = share of the recipe's required (non-pantry) ingredients the user has. Default weights 0.5/0.5; preferring
fewest-missing → 0.4/0.6; preferring few steps adds 0.15. Variant matches count 0.8. Salt, water and black pepper are assumed
available and reported as such. Quantities are compared only when units are convertible (volume, weight, count) and the
status is reported per ingredient ("sufficient on paper", "INSUFFICIENT", "not verified"). Hard constraints (exclusions,
categories, step/ingredient limits) are SQL filters; if nothing qualifies the result is `no_match` and the alternatives are
produced by relaxing **one constraint at a time**, each labelled `VIOLATES: <constraint>`.

**Substitution (Task 3).** (1) locate target lines by term and/or restricted category; (2) classify the role from a small
lexicon (binder, fat, liquid, thickener…); (3) candidates = lexicon pool ∪ vetted "X or Y" alternatives mined from the data,
dropping anything that violates the restriction (including *possible* allergens); (4) gather local evidence —
*direct* (explicit "or" lines / "instead of" sentences, precomputed into `or_lines` and `subst_sentences`) and *indirect*
(how often the candidate appears in similar recipes, found by dish-title phrase + hybrid search, that lack the restricted item);
(5) control for popularity with a **lift** against similar recipes that keep the ingredient, so olive oil does not "win" just
for being common; (6) tier the evidence (direct / moderate / weak / none), flag small samples and conflicting approaches;
(7) build the updated ingredient list and revised instructions deterministically, then **re-verify** the result against the
restriction. Rule-of-thumb amounts (¼ cup applesauce per egg) are labelled *EXTERNAL, unverified*. Adjunct items that only
fix part of a role (oil for cream, salt for cheese) are shown but never swapped in at the original amount. If nothing is
supportable the tool refuses and says which lines remain unchanged.

**Uncertainty and refusal.** No nutrition, cooking-time, serving-size or price fields exist; `get_recipe` states this and the
prompt forbids guessing. Evidence tiers, lift notes, small-sample warnings and `UNVERIFIED` caveats travel with every
suggestion. Nothing is called "safe"; every adaptation carries an explicit not-medical-advice note.

**External search.** `web_lookup` (Gemini Google-Search grounding) is only used when the user asks or agrees, runs as a separate
call, is labelled *EXTERNAL*, and returns links.

**Deployment.** Local only, as the brief requires. A Python backend holding ~125k recipes plus a 94 MB embedding matrix does
not fit Vercel's serverless size/cold-start limits; if hosting were required, the Next.js app would go on Vercel and the API
on a container host (Render/Fly/HF Spaces) with `NEXT_PUBLIC_API_URL` pointing at it.

## What worked, what failed (found while building)

- Naïve normalisation merged products ("peanut butter" matched "butter"); fixed with the exact/variant split and a harmless-modifier list.
- "chicken fat" matched "chicken" because `fat` was treated as harmless (low-fat); now only harmless alongside low/reduced/free.
- A correlated `NOT EXISTS` filter made a match query take ~20 s; rewritten as uncorrelated `IN/NOT IN` (→ ~1 s).
- Mining "X or Y" lines naïvely produced nonsense alternatives for egg ("large", "brioche"); now word-boundary matched,
  frequency-vetted, and restricted to role-plausible candidates.
- Dish-agnostic "similar recipes" made banana, baking soda and olive oil look like egg substitutes purely by popularity;
  fixed by anchoring on the dish title, excluding ingredients already in the recipe, and the lift control.
  Result: banana bread without eggs is now honestly reported as **weak, indirect** evidence (only 9 comparable egg-free recipes).
- Duplicate and junk records made ranked lists repeat themselves ("Tst" ×7); handled with `dup_of` and a junk filter.
- The evaluation originally passed while an unsupported swap ("Parmesan → ¼ cup salt") slipped through; the checks were
  tightened (adjunct handling, UNVERIFIED caveats) so that class of error now fails the suite.

## Known limitations

- Allergen and role knowledge is rule based and finite (15 ingredient roles, ~12 allergen/diet groups); outside the lexicon the
  substitution tool refuses or relies on mined "or" alternatives. It is text analysis, not allergy advice.
- Step counts are sentence-based approximations; instruction quality varies by source.
- Quantity comparison ignores yield/serving size (not in the data).
- Indirect substitution evidence is correlational (similar recipes that lack an ingredient), not tested swaps.
- Reported statistics count duplicate records (1,534) because the brief asks for dataset-level counts; search/match collapse them.
- The live-LLM evaluation (`eval/run_llm_eval.py`) needs an API key and its results depend on the model's behaviour (replies
  are phrased differently run to run, and free-tier quotas can interrupt it); the deterministic suite (`eval/run_eval.py`)
  is the reproducible part.
- LLM answers sometimes omit details the tools return (e.g. the score breakdown) or make extra tool calls; the full tool
  output is always available in the "Evidence & tool trace" panel.

## One-shot LLM vs this system

A plain LLM asked *"How many recipes contain both garlic and onion, and what percentage of the dataset is that?"* cannot
know the dataset and will produce a plausible number (and may invent a total). Here the same question calls `dataset_stats`,
which returns 9,788 (7.96 % of 122,912) with exact-name matching and 14,729 with variants, the denominator definition and the
SQL. Likewise *"How many calories does this recipe have?"* is answered from the missing-attribute statement, not an estimate.
`eval/run_llm_eval.py` records both answers side by side.

## AI tools used (disclosure)

Claude Code (Anthropic, Claude Sonnet) was used throughout this project: to scaffold the FastAPI/Next.js code, draft the
normaliser, search/stat/match/substitution modules and the evaluation harness, and to debug (profiling slow queries,
finding the normalisation and evidence-confound bugs listed above). Design choices — the exact/variant matching split,
the evidence tiers and lift control, hard-constraint handling, local-only deployment — were reviewed and can be explained
and modified in the walkthrough. *(Edit this paragraph to describe your own involvement before submitting.)*
