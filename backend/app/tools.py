"""Tools exposed to the LLM. Every factual answer comes from these deterministic functions.

`build_tools(session)` returns plain Python callables bound to a session; the google-genai SDK turns their type
hints + docstrings into function declarations. Docstrings are therefore part of the prompt: keep them precise.
"""
from __future__ import annotations

from typing import Optional

from .match import Prefs, match_recipes
from .session import Session
from .stats import METRICS, run_stats
from .store import Filters, get_store
from .substitute import build_adaptation, load_recipe, substitution_evidence

MAX_K = 10


def _norm_list(x) -> list[str]:
    return [str(i).strip() for i in (x or []) if str(i).strip()]


def _merge(f: Filters, include=None, exclude=None, exclude_categories=None, max_steps=None, min_steps=None,
           max_ingredients=None, min_ingredients=None, source=None, title_contains=None) -> Filters:
    g = f.copy()
    for lst, new in ((g.include, include), (g.exclude, exclude), (g.exclude_categories, exclude_categories),
                     (g.title_contains, title_contains)):
        for t in _norm_list(new):
            if t.lower() not in [x.lower() for x in lst]:
                lst.append(t)
    if max_steps is not None:
        g.max_steps = int(max_steps)
    if min_steps is not None:
        g.min_steps = int(min_steps)
    if max_ingredients is not None:
        g.max_ingredients = int(max_ingredients)
    if min_ingredients is not None:
        g.min_ingredients = int(min_ingredients)
    if source:
        s = source.strip().lower()
        s = {"allrecipes": "ar", "epicurious": "epi", "foodnetwork": "fn", "food network": "fn"}.get(s, s)
        if s in {"ar", "epi", "fn"} and s not in g.sources:
            g.sources.append(s)
    return g


def _drop(f: Filters, terms: list[str]) -> Filters:
    g = f.copy()
    low = {t.lower() for t in terms}
    g.include = [t for t in g.include if t.lower() not in low]
    g.exclude = [t for t in g.exclude if t.lower() not in low]
    g.exclude_categories = [t for t in g.exclude_categories if t.lower() not in low]
    g.title_contains = [t for t in g.title_contains if t.lower() not in low]
    if "max_steps" in low:
        g.max_steps = None
    if "min_steps" in low:
        g.min_steps = None
    if "max_ingredients" in low:
        g.max_ingredients = None
    return g


def _recipe_card(store, d: dict, f: Filters, show_ingredients: bool) -> dict:
    full = store.recipe(d["rid"], full=True) or {}
    ev = store.constraint_evidence(d["rid"], f)
    card = {"id": d["id"], "title": d["title"], "source": d["source"], "n_steps": d["n_steps"],
            "n_ingredients": d["n_ingredients"]}
    if ev["matched"]:
        card["required_ingredient_evidence"] = ev["matched"]
    if ev["warnings"]:
        card["warnings"] = ev["warnings"]
    if d.get("duplicate_records"):
        card["duplicate_records"] = d["duplicate_records"][:3]
    if show_ingredients or not f.is_empty():
        card["ingredients"] = [x.replace("ADVERTISEMENT", "").strip() for x in full.get("ingredients", [])
                               if x.replace("ADVERTISEMENT", "").strip()]
    return card


def build_tools(session: Session) -> list:
    store = get_store()

    def search_recipes(query: str = "", include: Optional[list[str]] = None, exclude: Optional[list[str]] = None,
                       exclude_categories: Optional[list[str]] = None, max_steps: Optional[int] = None,
                       min_steps: Optional[int] = None, max_ingredients: Optional[int] = None,
                       source: Optional[str] = None, remove_constraints: Optional[list[str]] = None,
                       reset: bool = False, k: int = 5, show_ingredients: bool = False) -> dict:
        """Search the recipe dataset (hybrid keyword + semantic) and refine the ACTIVE search across turns.

        The session remembers the previous query and all restrictions. Pass ONLY what is new: a follow-up such as
        "I cannot have dairy" -> exclude_categories=["dairy"] with an empty query keeps the earlier query and adds the
        restriction. Use reset=true when the user starts a different search. Use remove_constraints to drop a
        previously stated restriction (e.g. ["dairy"]) when the user changes their mind.

        Args:
            query: natural-language request or dish name, e.g. "penne pasta" (empty = keep previous query).
            include: ingredients every result must contain, e.g. ["chicken", "lemon"].
            exclude: specific ingredients that must be absent, e.g. ["mushroom"].
            exclude_categories: allergen/diet groups that must be absent: dairy, egg, peanut, tree_nut, gluten, soy,
                fish, shellfish, meat, pork, alcohol, vegetarian, vegan.
            max_steps: maximum instruction steps. min_steps similarly. max_ingredients: maximum ingredient count.
            source: restrict to ar (AllRecipes), epi (Epicurious) or fn (Food Network).
            remove_constraints: restrictions to drop from the active search.
            reset: start a fresh search (clears earlier query and restrictions).
            k: number of results (1-10).
            show_ingredients: include full ingredient lists.
        Returns:
            results (id, title, evidence), the active filters, and how many recipes satisfy them in total.
        """
        k = max(1, min(int(k), MAX_K))
        if reset:
            session.search_filters = Filters()
            session.search_query = ""
        f = session.search_filters
        if remove_constraints:
            f = _drop(f, _norm_list(remove_constraints))
        f = _merge(f, include, exclude, exclude_categories, max_steps, min_steps, max_ingredients, None, source, None)
        session.search_filters = f
        if query and query.strip():
            session.search_query = query.strip()
        res = store.search(session.search_query, f, k=k)
        cards = [_recipe_card(store, d, f, show_ingredients) for d in res["results"]]
        session.last_ids = [c["id"] for c in cards]
        if cards:
            session.last_recipe_id = cards[0]["id"]
        out = {"query": session.search_query, "active_filters": f.describe(),
               "recipes_satisfying_filters": res["filtered_total"], "retrieval": res["method"], "results": cards}
        if not cards:
            out["note"] = "No recipe satisfies all active restrictions. Nothing was relaxed."
        if f.exclude_categories or f.exclude:
            out["exclusion_note"] = ("Exclusions are applied on parsed ingredient names (definite allergen/diet matches). "
                                     "Packaged items may hide ingredients - this is not allergy advice.")
        return out

    def get_recipe(recipe_id: str, include_instructions: bool = True) -> dict:
        """Fetch one recipe exactly as stored (original title, ingredient lines and instructions) by its stable id.

        Args:
            recipe_id: id such as "ar:xxxx", "epi:xxxx" or "fn:xxxx" (as returned by search tools).
            include_instructions: include the full instruction text.
        """
        d = store.recipe(recipe_id, full=True)
        if not d:
            return {"error": f"no recipe with id '{recipe_id}'"}
        session.last_recipe_id = d["id"]
        out = {"id": d["id"], "title": d["title"], "source": d["source"], "n_ingredients": d["n_ingredients"],
               "n_steps": d["n_steps"], "ingredients": [x.replace("ADVERTISEMENT", "").strip() for x in d["ingredients"]
                                                        if x.replace("ADVERTISEMENT", "").strip()]}
        if include_instructions:
            out["instructions"] = d["instructions"]
        out["not_in_dataset"] = "nutrition values, cooking times and serving sizes are not available in this dataset"
        return out

    def dataset_stats(metric: str, include: Optional[list[str]] = None, exclude: Optional[list[str]] = None,
                      exclude_categories: Optional[list[str]] = None, max_steps: Optional[int] = None,
                      min_steps: Optional[int] = None, max_ingredients: Optional[int] = None,
                      min_ingredients: Optional[int] = None, source: Optional[str] = None,
                      title_contains: Optional[list[str]] = None, use_active_search: bool = False,
                      limit: int = 10, order: str = "asc", match_mode: str = "strict", skip_staples: bool = False,
                      denominator_include: Optional[list[str]] = None,
                      denominator_title_contains: Optional[list[str]] = None) -> dict:
        """Compute an exact statistic over the COMPLETE filtered set of recipes (programmatic, never estimated).

        Use for any count, percentage, ranking, average/median, most-common-ingredient or comparison question.
        metric must be one of: count, percentage, top_ingredients, ingredient_count_stats, step_count_stats,
        rank_by_steps, rank_by_ingredients, source_breakdown.
        "How many contain garlic and onion" -> metric=count, include=["garlic","onion"].
        "What percentage of the dataset" -> metric=percentage (denominator defaults to ALL usable recipes; use
        denominator_include / denominator_title_contains for a narrower base).
        "Five with the fewest steps among recipes with tomato" -> metric=rank_by_steps, include=["tomato"], limit=5,
        order="asc".
        The result states the filters, the number of recipes analysed, the calculation, tie information and the SQL.

        Args:
            metric: one of the metric names above.
            include / exclude / exclude_categories / title_contains / source / step and ingredient bounds: filters.
            use_active_search: also apply the restrictions of the active search from earlier turns.
            limit: rows for rankings/top lists (max 50). order: "asc" (fewest first) or "desc".
            match_mode: "strict" (exact name + harmless variants like red onion) or "exact" (identical name only).
            skip_staples: drop salt/water/pepper/sugar/flour/oil/butter from top_ingredients.
        """
        base = session.search_filters if use_active_search else Filters()
        f = _merge(base, include, exclude, exclude_categories, max_steps, min_steps, max_ingredients, min_ingredients,
                   source, title_contains)
        den = None
        if metric == "percentage" and (denominator_include or denominator_title_contains):
            den = _merge(Filters(), denominator_include, None, None, None, None, None, None, None, denominator_title_contains)
        out = run_stats(metric, f, limit=limit, order=order, match_mode=match_mode, denominator_filters=den,
                        skip_staples=skip_staples, store=store)
        out.pop("sql", None) if out.get("error") else None
        return out

    def match_by_ingredients(have: Optional[list[str]] = None, exclude: Optional[list[str]] = None,
                             exclude_categories: Optional[list[str]] = None, max_steps: Optional[int] = None,
                             max_ingredients: Optional[int] = None, remove_constraints: Optional[list[str]] = None,
                             prefer_fewest_missing: bool = True, prefer_fewest_steps: bool = False,
                             assume_pantry: bool = True, k: int = 5, reset: bool = False) -> dict:
        """Rank EXISTING recipes by how well they fit the ingredients the user has (never invents recipes).

        The session remembers the ingredient list and hard constraints, so a follow-up like "exclude eggs too" is
        match_by_ingredients(exclude=["eggs"]) with nothing else. Hard constraints are never relaxed silently; if
        nothing qualifies the result says so and lists alternatives that are explicitly marked as VIOLATING a
        constraint. Quantities (e.g. "2 cups rice") are compared only when units are convertible, otherwise the match is
        name-level and quantity sufficiency is reported as not verified.

        Args:
            have: ingredients the user has, optionally with quantities: ["eggs", "2 cups rice", "onions"]. Omit on
                follow-ups to reuse the previous list; provide to replace it.
            exclude: ingredients that must NOT appear in a recipe (hard constraint), e.g. ["peanuts"].
            exclude_categories: dairy, egg, peanut, tree_nut, gluten, soy, fish, shellfish, meat, pork, vegetarian, vegan...
            max_steps / max_ingredients: hard limits.
            remove_constraints: constraints to drop (e.g. ["eggs"]); use max_steps to drop with "max_steps".
            prefer_fewest_missing: rank recipes needing fewer extra ingredients higher (default true).
            prefer_fewest_steps: also reward fewer steps.
            assume_pantry: treat salt, water and black pepper as available (reported explicitly).
            k: number of results (1-10). reset: start a new match, clearing earlier ingredients and constraints.
        """
        if reset:
            session.match_have, session.match_filters = [], Filters()
            session.match_prefs = {"fewest_missing": True, "fewest_steps": False}
        if have:
            session.match_have = list(have)
        f = session.match_filters
        if remove_constraints:
            f = _drop(f, _norm_list(remove_constraints))
        f = _merge(f, None, exclude, exclude_categories, max_steps, None, max_ingredients, None, None, None)
        session.match_filters = f
        session.match_prefs = {"fewest_missing": prefer_fewest_missing, "fewest_steps": prefer_fewest_steps}
        if not session.match_have:
            return {"error": "no ingredients known yet - ask the user what they have"}
        res = match_recipes(session.match_have, f, Prefs(prefer_fewest_missing, prefer_fewest_steps, assume_pantry),
                            k=max(1, min(int(k), MAX_K)), store=store)
        ids = [r["id"] for r in res.get("results", [])]
        if ids:
            session.last_ids, session.last_recipe_id = ids, ids[0]
        return res

    def adapt_recipe(remove: list[str], restrict_categories: Optional[list[str]] = None, recipe_id: Optional[str] = None,
                     recipe_title: Optional[str] = None, recipe_ingredients: Optional[list[str]] = None,
                     recipe_instructions: Optional[str] = None, choose: Optional[str] = None) -> dict:
        """Adapt an existing recipe when an ingredient is unavailable or restricted, using LOCAL dataset evidence.

        Finds the ingredient's role, mines the dataset for supporting evidence (explicit "X or Y" lines, "instead of"
        sentences, and similar recipes that lack the restricted item), labels every substitute as direct / moderate /
        weak / none (external, unverified), then returns an updated ingredient list and revised instructions and
        re-checks them against the restriction. Evidence is cited by recipe id. Never claim a substitution is medically
        safe.

        Args:
            remove: ingredient names to remove/replace, e.g. ["eggs"] or ["cream"].
            restrict_categories: groups that must not remain, e.g. ["dairy"], ["egg"], ["gluten"].
            recipe_id: dataset recipe id to adapt. If omitted, the last discussed recipe is used.
            recipe_title: dataset recipe to look up by title when no id is known.
            recipe_ingredients / recipe_instructions: a recipe pasted by the user (use instead of an id).
            choose: force a specific substitute name from the candidates.
        """
        recipe = None
        if recipe_ingredients:
            recipe = load_recipe(store, recipe_text={"title": recipe_title, "ingredients": recipe_ingredients,
                                                     "instructions": recipe_instructions or ""})
        elif recipe_id:
            recipe = load_recipe(store, recipe_id=recipe_id)
        elif recipe_title:
            hit = store.search(recipe_title, Filters(), k=1)["results"]
            recipe = load_recipe(store, recipe_id=hit[0]["id"]) if hit else None
        elif session.last_recipe_id:
            recipe = load_recipe(store, recipe_id=session.last_recipe_id)
        if not recipe:
            return {"error": "recipe not found; give a recipe id, a title found in the dataset, or paste the recipe"}
        if recipe["id"]:
            session.last_recipe_id = recipe["id"]
        ev = substitution_evidence(recipe, _norm_list(remove), _norm_list(restrict_categories), store)
        choices = None
        if choose and ev.get("status") == "ok":
            choices = {p["target"]["index"]: choose for p in ev["plans"]}
        ad = build_adaptation(recipe, ev, choices)
        # compact evidence for the model: keep what is needed to cite, drop bulky fields
        compact = {k: v for k, v in ev.items() if k not in {"plans"}}
        compact["plans"] = []
        for p in ev.get("plans", []):
            compact["plans"].append({
                "target_line": p["target"]["raw"], "role": p["role"], "confidence": p["confidence"],
                "indirect_evidence_note": p["indirect_evidence_note"],
                "instead_of_mentions_in_dataset": p["instead_of_mentions"][:3],
                "other_or_alternatives_seen_unvetted": p.get("other_or_alternatives_seen_unvetted", [])[:3],
                "skipped_because_already_in_recipe": p.get("skipped_because_already_in_recipe", []),
                "candidates": [{k: c.get(k) for k in ("name", "evidence", "label", "adjunct", "lift", "lift_note", "direct_alternative_lines",
                                                      "direct_examples", "similar_recipes_using_it",
                                                      "of_similar_recipes_without_restricted", "supporting_recipe_ids",
                                                      "prep_in_local_recipe", "external_note", "ratio_per_original")}
                               for c in p["candidates"]]})
        return {"recipe": ev["recipe"], "original_ingredients": recipe["ingredients"],
                "original_instructions": recipe["instructions"], "evidence": compact, "adaptation": ad}

    def web_lookup(query: str) -> dict:
        """OPTIONAL external web search (Google Search grounding). Call ONLY when the user explicitly asks for web/
        internet evidence or agrees to it after local evidence was weak. Results are EXTERNAL and must be shown with
        their links and clearly separated from dataset evidence.

        Args:
            query: a focused question, e.g. "egg substitute in banana bread texture effects".
        """
        from .llm import grounded_search
        return grounded_search(query)

    return [search_recipes, get_recipe, dataset_stats, match_by_ingredients, adapt_recipe, web_lookup]


METRIC_HELP = METRICS
