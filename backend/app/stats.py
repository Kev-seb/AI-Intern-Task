"""Programmatic analytics over the *complete* filtered set (never sampled, never LLM-estimated).

Every result carries: the filters applied, the denominator/number of recipes analysed, the exact
calculation, the ingredient-normalisation mode, and the SQL used, so any answer is reproducible.
"""
from __future__ import annotations

import statistics

from .store import Filters, Store, get_store

STAPLES = {"salt", "water", "black pepper", "sugar", "all-purpose flour", "vegetable oil", "olive oil", "butter"}

METRICS = {
    "count": "number of recipes matching the filters",
    "percentage": "matching recipes as a share of a denominator set (default: all usable recipes)",
    "top_ingredients": "most common canonical ingredients across matching recipes (recipe-level counts)",
    "ingredient_count_stats": "mean / median / min / max / stdev of ingredients per recipe",
    "step_count_stats": "mean / median / min / max / stdev of instruction steps per recipe",
    "rank_by_steps": "recipes ranked by instruction-step count (asc = fewest first)",
    "rank_by_ingredients": "recipes ranked by number of ingredients (asc = fewest first)",
    "source_breakdown": "matching recipes per source file (ar / epi / fn)",
}

NORMALIZATION_NOTE = (
    "Ingredient lines are parsed into quantity / unit / name; preparation words (chopped, minced, softened...) "
    "are dropped, plurals are singularised, and synonyms are merged (scallion = green onion). Product forms are "
    "kept distinct (tomato sauce != tomato; peanut butter != butter). 'strict' matching counts the exact name plus "
    "harmless variants (red onion for onion); 'exact' matching counts only the identical canonical name."
)
STEP_RULE = (
    "A step is one sentence of the instructions (each non-empty line also starts a new step; fragments with fewer "
    "than 3 letters are ignored). The dataset stores instructions as free text, so step counts are an approximation."
)


def _describe_terms(store: Store, f: Filters, mode: str) -> dict:
    forms = {}
    for t in f.include:
        forms[t] = [c for c, _ in store.resolve_term(t, "strict")][:8] if mode == "strict" else [t]
    return forms


def run_stats(metric: str, filters: Filters | None = None, limit: int = 10, order: str = "asc",
              match_mode: str = "strict", denominator_filters: Filters | None = None,
              skip_staples: bool = False, store: Store | None = None) -> dict:
    store = store or get_store()
    f = filters or Filters()
    if metric not in METRICS:
        return {"error": f"unknown metric '{metric}'", "available": METRICS}
    limit = max(1, min(int(limit), 50))

    where, params = store._where(f, match_mode)
    total_usable = int(store.meta()["complete"])
    n = store.count(f, match_mode)
    out: dict = {
        "metric": metric,
        "filters": f.describe(),
        "match_mode": match_mode,
        "matched_forms": _describe_terms(store, f, match_mode),
        "recipes_analyzed": n,
        "dataset": {"raw_records": int(store.meta()["raw_records"]), "usable_recipes": total_usable,
                    "usable_definition": "has a title, >=1 parsed ingredient and >=1 instruction step"},
        "normalization": NORMALIZATION_NOTE,
    }
    if metric == "count":
        out["result"] = n
        out["calculation"] = f"COUNT(recipes WHERE {f.describe()}) over {total_usable} usable recipes"
        out["sql"] = f"SELECT COUNT(*) FROM recipes r WHERE {where}"
    elif metric == "percentage":
        df = denominator_filters or Filters()
        d = store.count(df, match_mode)
        out["denominator"] = {"filters": df.describe(), "recipes": d}
        out["result"] = {"count": n, "denominator": d, "percent": round(100.0 * n / d, 3) if d else None}
        out["calculation"] = f"{n} / {d} * 100 (numerator: {f.describe()}; denominator: {df.describe()})"
    elif metric == "top_ingredients":
        skip = tuple(STAPLES) if skip_staples else ()
        skip_sql = f" AND i.canon NOT IN ({','.join('?' * len(skip))})" if skip else ""
        sql = (f"SELECT i.canon, COUNT(DISTINCT i.rid) c FROM recipe_ingredients i WHERE i.rid IN "
               f"(SELECT r.rid FROM recipes r WHERE {where}){skip_sql} GROUP BY i.canon ORDER BY c DESC, i.canon LIMIT ?")
        rows = store.q(sql, [*params, *skip, limit])
        out["result"] = [{"ingredient": r[0], "recipes": r[1],
                          "percent_of_analyzed": round(100.0 * r[1] / n, 2) if n else None} for r in rows]
        out["calculation"] = (f"for each canonical ingredient: number of distinct recipes (of the {n} analysed) containing it; "
                              f"top {limit}" + ("; common staples excluded: " + ", ".join(sorted(STAPLES)) if skip_staples else ""))
        out["sql"] = sql
    elif metric in {"ingredient_count_stats", "step_count_stats"}:
        col = "n_ingredients" if metric == "ingredient_count_stats" else "n_steps"
        vals = [r[0] for r in store.q(f"SELECT r.{col} FROM recipes r WHERE {where}", params)]
        if vals:
            out["result"] = {"mean": round(statistics.fmean(vals), 3), "median": statistics.median(vals),
                             "min": min(vals), "max": max(vals),
                             "stdev": round(statistics.pstdev(vals), 3)}
        else:
            out["result"] = None
        out["calculation"] = f"mean/median/min/max/population-stdev of {col} over the {n} analysed recipes"
        if col == "n_steps":
            out["step_rule"] = STEP_RULE
        else:
            out["ingredient_rule"] = ("ingredient count = number of parsed ingredient lines (section headers such as "
                                      "'For the sauce:' are ignored; 'salt and pepper' counts as 2)")
        out["sql"] = f"SELECT r.{col} FROM recipes r WHERE {where}"
    elif metric in {"rank_by_steps", "rank_by_ingredients"}:
        col = "n_steps" if metric == "rank_by_steps" else "n_ingredients"
        direction = "ASC" if order.lower().startswith("asc") else "DESC"
        sql = (f"SELECT r.id, r.title, r.n_steps, r.n_ingredients, r.source FROM recipes r WHERE {where} "
               f"ORDER BY r.{col} {direction}, r.rid ASC LIMIT ?")
        rows = store.q(sql, [*params, limit])
        out["result"] = [{"rank": i + 1, "id": r["id"], "title": r["title"], "n_steps": r["n_steps"],
                          "n_ingredients": r["n_ingredients"], "source": r["source"]} for i, r in enumerate(rows)]
        if rows:
            cutoff = rows[-1][col]
            tied = store.q(f"SELECT COUNT(*) FROM recipes r WHERE {where} AND r.{col} = ?", [*params, cutoff])[0][0]
            out["ties_at_cutoff"] = {"value": cutoff, "recipes_with_that_value": tied}
        out["calculation"] = (f"ORDER BY {col} {direction}, then dataset order (rid) as a deterministic tie-break; "
                              f"first {limit} of {n} analysed recipes")
        if col == "n_steps":
            out["step_rule"] = STEP_RULE
        out["sql"] = sql
    elif metric == "source_breakdown":
        rows = store.q(f"SELECT r.source, COUNT(*) FROM recipes r WHERE {where} GROUP BY r.source ORDER BY 2 DESC", params)
        out["result"] = {r[0]: r[1] for r in rows}
        out["calculation"] = "COUNT grouped by source file (ar=AllRecipes, epi=Epicurious, fn=Food Network)"
    return out
