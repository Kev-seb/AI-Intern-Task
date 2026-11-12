"""Task 2: rank existing recipes against the ingredients a user has (never invents recipes).

Scoring (all components shown in every result):
    usage        = share of YOUR ingredients that the recipe uses            (use as many as possible)
    completeness = share of the recipe's required ingredients you already have (fewest missing)
    step_score   = 1 - min(steps / 20, 1)                                  (only if "fewest steps" preferred)
    score        = w_u*usage + w_c*completeness + w_s*step_score
Weights: default 0.5 / 0.5 / 0; prefer fewest missing -> 0.4 / 0.6 / 0; prefer fewest steps -> x0.85 and w_s = 0.15.
An exact name match counts 1.0, a harmless variant (roma tomato for tomato) counts 0.8 and is flagged.
Hard constraints are enforced in SQL and are never relaxed; alternatives that violate one are returned separately.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .normalize import normalize_term, parse_ingredients, variant_note
from .store import Filters, Store, get_store

PANTRY = {"salt", "water", "black pepper"}
VARIANT_CREDIT = 0.8
POOL = 400

VOL = {"tsp": 1, "teaspoon": 1, "tbsp": 3, "tablespoon": 3, "cup": 48, "fl oz": 6, "pint": 96, "pt": 96,
       "quart": 192, "qt": 192, "gallon": 768, "ml": 0.2029, "liter": 202.9, "litre": 202.9}
WT = {"oz": 1, "ounce": 1, "lb": 16, "pound": 16, "g": 0.03527, "gram": 0.03527, "kg": 35.27, "kilogram": 35.27}


def _unit_key(u: str | None) -> str | None:
    if not u:
        return None
    u = u.lower().strip(".")
    if u.endswith("s") and u[:-1] in {**VOL, **WT}:
        u = u[:-1]
    if u in {"cups", "tablespoons", "teaspoons", "ounces", "pounds", "grams", "liters", "litres"}:
        u = u[:-1]
    return u


def _to_base(qty: float | None, unit: str | None) -> tuple[str, float] | None:
    if qty is None:
        return None
    u = _unit_key(unit)
    if u is None:
        return ("count", qty)
    if u in VOL:
        return ("volume", qty * VOL[u])
    if u in WT:
        return ("weight", qty * WT[u])
    return (f"unit:{u}", qty)


def parse_have(items: list) -> list[dict]:
    """Accept ['eggs', '2 cups rice', {'name':'onion','qty':2,'unit':null}] -> uniform dicts."""
    out = []
    for it in items:
        if isinstance(it, dict):
            name = it.get("name") or ""
            canon = normalize_term(name)
            if canon:
                out.append({"name": name, "canon": canon, "qty": it.get("qty"), "unit": it.get("unit")})
        else:
            for ing in parse_ingredients(str(it)):
                out.append({"name": str(it).strip(), "canon": ing.canon, "qty": ing.qty, "unit": ing.unit})
    seen, uniq = set(), []
    for h in out:
        if h["canon"] not in seen:
            seen.add(h["canon"])
            uniq.append(h)
    return uniq


@dataclass
class Prefs:
    fewest_missing: bool = True
    fewest_steps: bool = False
    assume_pantry: bool = True

    def weights(self) -> tuple[float, float, float]:
        wu, wc, ws = (0.4, 0.6, 0.0) if self.fewest_missing else (0.5, 0.5, 0.0)
        if self.fewest_steps:
            wu, wc, ws = wu * 0.85, wc * 0.85, 0.15
        return wu, wc, ws


def _quantity_status(item: dict, lines: list[dict]) -> str:
    if item.get("qty") is None:
        return "not verified: no quantity given (name-level match only)"
    have = _to_base(item["qty"], item.get("unit"))
    need_total: dict[str, float] = defaultdict(float)
    for ln in lines:
        b = _to_base(ln["qty"], ln["unit"])
        if b is None:
            return "not verified: recipe line has no quantity"
        need_total[b[0]] += b[1]
    if have is None or len(need_total) != 1 or have[0] not in need_total:
        return "not verified: units are not comparable (e.g. count vs volume)"
    need = need_total[have[0]]
    if have[1] + 1e-9 >= need:
        return f"sufficient on paper (have {item['qty']} {item.get('unit') or ''}".strip() + ", recipe needs about the same or less)"
    return f"INSUFFICIENT on paper (have {item['qty']} {item.get('unit') or ''}".strip() + f", recipe needs more; ~{round(need / have[1], 2)}x what you have)"


def _score_recipe(store: Store, rid: int, have: list[dict], forms: dict[int, dict[str, str]], prefs: Prefs,
                  meta: dict) -> dict | None:
    ings = store.canon_ingredients(rid)
    if not ings:
        return None
    matched, missing, assumed = [], [], []
    item_lines: dict[int, list[dict]] = defaultdict(list)
    credit = 0.0
    seen_missing: set[str] = set()
    for row in ings:
        hit = None
        for idx, fm in forms.items():
            lvl = fm.get(row["canon"])
            if lvl and (hit is None or (lvl == "exact" and hit[1] != "exact")):
                hit = (idx, lvl)
        if hit:
            idx, lvl = hit
            item_lines[idx].append(row)
            credit += 1.0 if lvl == "exact" else VARIANT_CREDIT
            entry = {"you_have": have[idx]["name"], "recipe_line": row["raw"], "match": lvl}
            if lvl != "exact":
                entry["note"] = variant_note(have[idx]["canon"], row["canon"])
            matched.append((idx, entry))
        elif prefs.assume_pantry and row["canon"] in PANTRY:
            assumed.append(row["canon"])
        elif row["canon"] not in seen_missing:
            seen_missing.add(row["canon"])
            missing.append({"ingredient": row["canon"], "recipe_line": row["raw"]})
    n_required = len(matched) + len(missing)
    if not matched:
        return None
    completeness = credit / n_required if n_required else 1.0
    used_items = {idx for idx, _ in matched}
    best = {}
    for idx, e in matched:
        best[idx] = max(best.get(idx, 0), 1.0 if e["match"] == "exact" else VARIANT_CREDIT)
    usage = sum(best.values()) / max(len(have), 1)
    step_score = 1 - min(meta["n_steps"] / 20, 1)
    wu, wc, ws = prefs.weights()
    score = wu * usage + wc * completeness + ws * step_score
    for idx in used_items:
        status = _quantity_status(have[idx], item_lines[idx])
        for i, e in matched:
            if i == idx:
                e["quantity"] = status
    unused = [have[i]["name"] for i in range(len(have)) if i not in used_items]
    return {
        "id": meta["id"], "title": meta["title"], "source": meta["source"],
        "n_steps": meta["n_steps"], "n_ingredients": meta["n_ingredients"],
        "score": round(score, 4),
        "components": {"usage": round(usage, 3), "completeness": round(completeness, 3),
                       "step_score": round(step_score, 3) if ws else None,
                       "weights": {"usage": round(wu, 3), "completeness": round(wc, 3), "steps": round(ws, 3)}},
        "matched": [e for _, e in matched],
        "missing": missing,
        "n_missing": len(missing),
        "assumed_pantry": sorted(set(assumed)),
        "your_ingredients_unused": unused,
        "why": (f"uses {len(used_items)}/{len(have)} of your ingredients; you have {round(completeness * 100)}% of the "
                f"recipe's {n_required} required (non-pantry) ingredients; {len(missing)} missing; {meta['n_steps']} steps"),
    }


def _candidates(store: Store, have: list[dict], f: Filters):
    forms: dict[int, dict[str, str]] = {}
    for idx, h in enumerate(have):
        fm = {c: lvl for c, lvl in store.resolve_term(h["canon"], "strict")}
        forms[idx] = fm
    all_canons = sorted({c for fm in forms.values() for c in fm})
    if not all_canons:
        return forms, {}, 0
    rows = store.q(f"SELECT rid, canon FROM recipe_ingredients WHERE canon IN ({','.join('?' * len(all_canons))})", all_canons)
    allowed = None if f.is_empty() else set(store.filtered_rids(f))
    table = store.meta_table()
    canon_to_items: dict[str, list[int]] = defaultdict(list)
    for idx, fm in forms.items():
        for c in fm:
            canon_to_items[c].append(idx)
    by_rid: dict[int, dict] = {}
    for rid, canon in rows:
        if allowed is not None and rid not in allowed:
            continue
        m = table.get(rid)
        if m is None or m[5] != rid:          # incomplete, or a duplicate copy of an earlier recipe
            continue
        d = by_rid.get(rid)
        if d is None:
            d = by_rid[rid] = {"id": m[0], "title": m[1], "source": m[2], "n_steps": m[3],
                               "n_ingredients": m[4], "items": set()}
        d["items"].update(canon_to_items[canon])
    return forms, by_rid, len(by_rid)


def _rank(store: Store, have: list[dict], f: Filters, prefs: Prefs, k: int):
    forms, by_rid, n_cand = _candidates(store, have, f)
    pre = sorted(by_rid.items(), key=lambda kv: (-len(kv[1]["items"]), kv[1]["n_ingredients"], kv[1]["n_steps"], kv[0]))[:POOL]
    scored = []
    for rid, meta in pre:
        s = _score_recipe(store, rid, have, forms, prefs, meta)
        if s:
            scored.append(s)
    scored.sort(key=lambda s: (-s["score"], s["n_missing"], s["n_steps"], s["id"]))
    return scored[:k], n_cand


def _relaxations(f: Filters) -> list[tuple[str, Filters]]:
    out = []
    for field_name in ("exclude", "exclude_categories", "include", "title_contains", "sources"):
        for v in getattr(f, field_name):
            g = f.copy()
            getattr(g, field_name).remove(v)
            label = {"exclude": f"excluded ingredient '{v}'", "exclude_categories": f"'{v}'-free requirement",
                     "include": f"required ingredient '{v}'", "title_contains": f"title containing '{v}'",
                     "sources": f"source '{v}'"}[field_name]
            out.append((label, g))
    for attr, label in (("max_steps", "maximum step count"), ("min_steps", "minimum step count"),
                        ("max_ingredients", "maximum ingredient count"), ("min_ingredients", "minimum ingredient count")):
        if getattr(f, attr) is not None:
            g = f.copy()
            setattr(g, attr, None)
            out.append((label + f" ({getattr(f, attr)})", g))
    return out


def match_recipes(have_items: list, filters: Filters | None = None, prefs: Prefs | None = None, k: int = 5,
                  store: Store | None = None) -> dict:
    store = store or get_store()
    f = filters or Filters()
    prefs = prefs or Prefs()
    have = parse_have(have_items)
    notes: list[str] = []
    if not have:
        return {"error": "no ingredients provided"}

    # ingredients the user listed but that their own hard constraints forbid are dropped, with a note
    from .normalize import categories, resolve_category
    banned_cats: dict[str, str] = {}
    for label in [*f.exclude_categories, *f.exclude]:
        for c in resolve_category(label):
            banned_cats.setdefault(c, label)
    kept = []
    for h in have:
        forbidden = None
        for term in f.exclude:
            if h["canon"] in {c for c, _ in store.resolve_term(term, "broad")} or h["canon"] == normalize_term(term):
                forbidden = f"excluded ingredient '{term}'"
        definite, _ = categories(h["canon"])
        for c in sorted(definite & set(banned_cats)):
            forbidden = forbidden or f"'{banned_cats[c]}' restriction"
        if forbidden:
            notes.append(f"'{h['name']}' is on your list but conflicts with {forbidden}; it was not used for matching.")
        else:
            kept.append(h)
    have = kept
    unrecognised = [h["name"] for h in have if not store.resolve_term(h["canon"], "strict")]
    if unrecognised:
        notes.append("Not found in the dataset's ingredient vocabulary (ignored): " + ", ".join(unrecognised))
    have = [h for h in have if h["name"] not in unrecognised]
    if not have:
        return {"error": "none of the listed ingredients could be matched", "notes": notes, "no_match": True}

    results, n_cand = _rank(store, have, f, prefs, k)
    wu, wc, ws = prefs.weights()
    out: dict = {
        "you_have": [{"name": h["name"], "normalized": h["canon"], "quantity": h["qty"], "unit": h["unit"]} for h in have],
        "hard_constraints": f.describe(),
        "preferences": {"fewest_missing": prefs.fewest_missing, "fewest_steps": prefs.fewest_steps,
                        "pantry_assumed": sorted(PANTRY) if prefs.assume_pantry else []},
        "eligible_recipes_containing_any_ingredient": n_cand,
        "scoring": (f"score = {wu:.2f}*usage + {wc:.2f}*completeness" + (f" + {ws:.2f}*step_score" if ws else "")
                    + "; usage = share of your ingredients used; completeness = share of the recipe's required "
                      "ingredients you have; variant matches (e.g. roma tomato for tomato) count 0.8; ties -> fewer "
                      "missing, then fewer steps"),
        "results": results,
        "notes": notes,
        "no_match": not results,
    }
    if results and len(results) < k:
        notes.append(f"Only {len(results)} eligible recipes found (requested {k}); nothing was added to pad the list.")
    if not results:
        notes.append("No recipe satisfies ALL hard constraints. Constraints were NOT relaxed for the main result.")
        alts = []
        for label, g in _relaxations(f):
            res, _ = _rank(store, have, g, prefs, 3)
            if res:
                alts.append({"violates": label, "results": res})
        out["closest_violating_alternatives"] = alts
        out["alternatives_warning"] = ("Everything under closest_violating_alternatives VIOLATES the stated constraint "
                                       "shown in 'violates'; they are not valid matches.")
    return out
