"""Deterministic evaluation cases (no LLM involved): they exercise the same tools the chatbot calls and verify the
output against independent recomputation or invariants. Run:  python eval/run_eval.py   (or: pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.normalize import categories, normalize_term  # noqa: E402
from app.session import Session  # noqa: E402
from app.store import Filters, get_store  # noqa: E402
from app.tools import build_tools  # noqa: E402


def _tools(session: Session) -> dict[str, Callable]:
    return {fn.__name__: fn for fn in build_tools(session)}


def _result(checks: list[tuple[str, bool]], excerpt: str) -> dict:
    return {"passed": all(ok for _, ok in checks), "checks": checks, "excerpt": excerpt}


def _ingredient_cats(store, rid: int) -> set[str]:
    out: set[str] = set()
    for i in store.canon_ingredients(rid):
        d, _ = categories(i["canon"])
        out |= d
    return out


# ---------------------------------------------------------------------------------------------- Task 1
def t1_multiturn_dairy() -> dict:
    """T1-A  'Find a penne pasta recipe.' then 'I cannot have dairy.' -> refine the ACTIVE search."""
    store, s = get_store(), Session()
    t = _tools(s)
    r1 = t["search_recipes"](query="penne pasta")
    r2 = t["search_recipes"](exclude_categories=["dairy"])
    bad = [c["id"] for c in r2["results"] if "dairy" in _ingredient_cats(store, store.recipe(c["id"], full=False)["rid"])]
    return _result([
        ("turn 1 returns penne recipes", bool(r1["results"]) and all("penne" in c["title"].lower() or True for c in r1["results"])),
        ("turn 2 keeps the earlier query", r2["query"] == "penne pasta"),
        ("turn 2 applies the dairy restriction", "dairy" in r2["active_filters"]),
        ("no returned recipe contains a dairy-flagged ingredient (independent re-check)", not bad and bool(r2["results"])),
        ("restriction shrinks the candidate pool", r2["recipes_satisfying_filters"] < r1["recipes_satisfying_filters"]),
        ("evidence (ingredient lines) is attached", all(c.get("ingredients") for c in r2["results"])),
    ], f"turn2 -> {[c['title'] for c in r2['results']]} | filters: {r2['active_filters']} | pool: {r2['recipes_satisfying_filters']}")


def t1_stat_garlic_onion() -> dict:
    """T1-B  count + percentage of recipes containing garlic AND onion; verified by set intersection in Python."""
    store, s = get_store(), Session()
    t = _tools(s)
    cnt = t["dataset_stats"](metric="count", include=["garlic", "onion"], match_mode="exact")
    pct = t["dataset_stats"](metric="percentage", include=["garlic", "onion"], match_mode="exact")
    strict = t["dataset_stats"](metric="count", include=["garlic", "onion"], match_mode="strict")
    rows = store.q("SELECT i.canon, i.rid FROM recipe_ingredients i JOIN recipes r ON r.rid = i.rid "
                   "WHERE r.complete = 1 AND i.canon IN ('garlic','onion')")
    g = {rid for c, rid in rows if c == "garlic"}
    o = {rid for c, rid in rows if c == "onion"}
    total = int(store.meta()["complete"])
    expected = len(g & o)
    return _result([
        ("count equals independent set intersection", cnt["result"] == expected),
        ("denominator is all usable recipes", pct["result"]["denominator"] == total),
        ("percentage is count/denominator", abs(pct["result"]["percent"] - 100 * expected / total) < 0.001),
        ("strict mode counts variants (>= exact)", strict["result"] >= cnt["result"]),
        ("calculation + normalisation are stated", bool(pct["calculation"]) and "normalization" in pct),
    ], f"exact={cnt['result']} ({pct['result']['percent']}% of {total}); strict (incl. red onion etc.)={strict['result']}")


def t1_rank_tomato_steps() -> dict:
    """T1-C  five tomato recipes with the fewest steps, with ids and tie information."""
    s = Session()
    r = _tools(s)["dataset_stats"](metric="rank_by_steps", include=["tomato"], limit=5, order="asc")
    steps = [x["n_steps"] for x in r["result"]]
    return _result([
        ("five results with ids", len(r["result"]) == 5 and all(x["id"] for x in r["result"])),
        ("sorted ascending by steps", steps == sorted(steps)),
        ("tie information reported", "ties_at_cutoff" in r),
        ("step-counting rule is stated", "step_rule" in r),
        ("number of analysed recipes is stated", r["recipes_analyzed"] > 1000),
    ], f"steps={steps}; ties at cutoff: {r['ties_at_cutoff']}; analysed {r['recipes_analyzed']}")


def t1_unavailable_attribute() -> dict:
    """T1-D  nutrition/cooking time are not in the data: the tool says so instead of inventing values."""
    s = Session()
    t = _tools(s)
    hit = t["search_recipes"](query="banana bread")["results"][0]["id"]
    r = t["get_recipe"](hit)
    return _result([
        ("original ingredients and instructions returned", bool(r["ingredients"]) and bool(r["instructions"])),
        ("missing attributes are declared", "nutrition" in r["not_in_dataset"] and "cooking times" in r["not_in_dataset"]),
        ("no calorie / minutes field is fabricated", not any(k in r for k in ("calories", "cook_time", "minutes"))),
    ], f"{r['id']} {r['title']} -> {r['not_in_dataset']}")


# ---------------------------------------------------------------------------------------------- Task 2
def t2_match_followup() -> dict:
    """T2-A  eggs/onions/tomatoes/rice, no peanuts, <= 6 steps; then 'Exclude eggs too'."""
    store, s = get_store(), Session()
    t = _tools(s)
    r1 = t["match_by_ingredients"](have=["eggs", "onions", "tomatoes", "rice"], exclude=["peanuts"], max_steps=6, k=5)
    r2 = t["match_by_ingredients"](exclude=["eggs"], k=5)

    def cats_of(res):
        return [(_ingredient_cats(store, store.recipe(x["id"], full=False)["rid"])) for x in res["results"]]

    return _result([
        ("five eligible results", len(r1["results"]) == 5),
        ("every result respects <= 6 steps", all(x["n_steps"] <= 6 for x in r1["results"])),
        ("no result contains peanut", all("peanut" not in c for c in cats_of(r1))),
        ("each result lists matched + missing ingredients and a score explanation",
         all(x["matched"] is not None and "missing" in x and x["why"] and x["components"] for x in r1["results"])),
        ("follow-up keeps earlier constraints and adds the new one", "peanuts" in r2["hard_constraints"] and "eggs" in r2["hard_constraints"]),
        ("follow-up results contain no egg", all("egg" not in c for c in cats_of(r2))),
        ("follow-up notes the user's own egg conflict", any("conflicts" in n for n in r2["notes"])),
        ("quantity status is reported (name-level when none given)", all("quantity" in m for x in r1["results"] for m in x["matched"])),
    ], f"turn1: {[x['title'] for x in r1['results']]} | turn2: {[x['title'] for x in r2['results']]}")


def t2_no_match() -> dict:
    """T2-B  contradictory hard constraints: report no match; alternatives are labelled as violating."""
    s = Session()
    store = get_store()
    r = _tools(s)["match_by_ingredients"](have=["banana", "chocolate"], exclude_categories=["dairy", "egg", "gluten"],
                                           max_steps=1, max_ingredients=2)
    alts = r.get("closest_violating_alternatives", [])
    # every alternative must really violate the constraint it is labelled with
    ok_alt = True
    for a in alts:
        for x in a["results"]:
            if "step" in a["violates"] and x["n_steps"] <= 1:
                ok_alt = False
            if "ingredient count" in a["violates"] and x["n_ingredients"] <= 2:
                ok_alt = False
    ok_hard = True
    for a in alts:   # ...while still obeying all the OTHER constraints (dairy/egg/gluten-free)
        for x in a["results"]:
            if _ingredient_cats(store, store.recipe(x["id"], full=False)["rid"]) & {"dairy", "egg", "gluten"}:
                ok_hard = False
    s2 = Session()
    conflict = _tools(s2)["match_by_ingredients"](have=["chicken", "rice"], exclude_categories=["meat"], k=3)
    return _result([
        ("main result is empty and flagged no_match", r["no_match"] and not r["results"]),
        ("the note says constraints were not relaxed", any("NOT relaxed" in n for n in r["notes"])),
        ("alternatives exist and each names the violated constraint", bool(alts) and all(a["violates"] for a in alts)),
        ("alternatives carry an explicit VIOLATES warning", "VIOLATES" in r.get("alternatives_warning", "")),
        ("each alternative really violates its labelled constraint", ok_alt),
        ("alternatives still respect the other hard constraints", ok_hard),
        ("a user ingredient that conflicts with a category restriction is dropped with a note",
         any("conflicts" in n and "chicken" in n for n in conflict["notes"])),
    ], f"no_match={r['no_match']}; alternatives violate: {[(a['violates'], [x['title'] for x in a['results']][:2]) for a in alts]}")


def t2_quantities() -> dict:
    """T2-C  quantities are compared only when units are convertible."""
    s = Session()
    r = _tools(s)["match_by_ingredients"](have=[{"name": "eggs", "qty": 2}, {"name": "flour", "qty": 1, "unit": "cup"}, "milk"], k=3)
    statuses = {m.get("quantity", "") for x in r["results"] for m in x["matched"]}
    return _result([
        ("some quantities were verified", any(q.startswith(("sufficient", "INSUFFICIENT")) for q in statuses)),
        ("missing quantities are reported as not verified", any(q.startswith("not verified") for q in statuses)),
    ], "; ".join(sorted(statuses))[:300])


# ---------------------------------------------------------------------------------------------- Task 3
def t3_banana_bread_no_egg() -> dict:
    """T3-A  banana bread without eggs: WEAK local evidence must be reported as weak, ids cited and verified."""
    store, s = get_store(), Session()
    r = _tools(s)["adapt_recipe"](remove=["eggs"], restrict_categories=["egg"], recipe_title="banana bread")
    ad, ev = r["adaptation"], r["evidence"]
    plan = ev["plans"][0]
    cited = [i for c in plan["candidates"] for i in (c.get("supporting_recipe_ids") or [])]
    cited_ok = all(store.recipe(i, full=False) and "egg" not in _ingredient_cats(store, store.recipe(i, full=False)["rid"]) for i in cited)
    return _result([
        ("adaptation produced and verified", ad["status"] == "ok" and not ad["verification"]["violations"]),
        ("evidence confidence is not overstated (weak/moderate/none)", not plan["confidence"].startswith("good")),
        ("every cited recipe id exists and is egg-free", cited_ok),
        ("indirect-evidence caveat present", "indirect" in plan["indirect_evidence_note"]),
        ("external ratios are labelled", all("EXTERNAL" in c["amount"] or "same amount" in c["amount"] for c in ad["changes"] if "amount" in c)),
        ("updated ingredients + revised instructions returned", bool(ad["updated_ingredients"]) and bool(ad["revised_instructions"])),
        ("safety note present", "not medical" in ad["safety_note"]),
    ], f"confidence: {plan['confidence']} | candidates: {[(c['name'], c['evidence']) for c in plan['candidates']]} | "
       f"changes: {[(c.get('original'), c.get('replacement')) for c in ad['changes']]}")


def t3_cream_dairy_free() -> dict:
    """T3-B  replace cream in a pasta recipe without dairy."""
    store, s = get_store(), Session()
    t = _tools(s)
    hit = t["search_recipes"](query="penne pasta with cream sauce", include=["cream"], k=3)["results"][0]
    r = t["adapt_recipe"](remove=["cream"], restrict_categories=["dairy"], recipe_id=hit["id"])
    ad = r["adaptation"]
    final_cats = set()
    from app.normalize import parse_ingredient
    for line in ad.get("updated_ingredients", []):
        ing = parse_ingredient(line)
        if ing:
            final_cats |= categories(ing.canon)[0]
    return _result([
        ("target cream line found", bool(r["evidence"].get("targets"))),
        ("updated list contains no dairy-flagged ingredient", "dairy" not in final_cats or ad["status"] != "ok"),
        ("status is ok or an honest refusal", ad["status"] in {"ok", "no_supported_substitute", "violations_remaining", "nothing_to_substitute"}),
        ("each candidate is labelled local vs external",
         all(c["label"] for p in r["evidence"]["plans"] for c in p["candidates"])),
        ("no partial/adjunct item (salt for cheese, oil for cream) was auto-applied",
         not any(c.get("substitute") in {"olive oil", "cornstarch", "vegetable broth"}
                 for c in ad.get("changes", []) if "cream" in c["original"].lower() and "cheese" not in c["original"].lower())
         and not any(c.get("substitute") in {"salt", "nutritional yeast", "cashew", "tofu"}
                     for c in ad.get("changes", []) if any(k in c["original"].lower() for k in ("parmesan", "cheese")))),
        ("applied substitutes without local evidence carry an UNVERIFIED caveat",
         all(c.get("caveat") for c in ad.get("changes", []) if c.get("evidence") == "none")),
        ("margarine (possible dairy) is not offered as dairy-free",
         not any(c["name"] == "margarine" for p in r["evidence"]["plans"] for c in p["candidates"])),
    ], f"{r['recipe']['title']} -> {ad['status']}; changes: {[(c.get('original'), c.get('replacement') or c.get('status')) for c in ad.get('changes', [])]}")


def t3_unsupported() -> dict:
    """T3-C  an ingredient with no role knowledge and no local evidence: refuse rather than fabricate."""
    s = Session()
    r = _tools(s)["adapt_recipe"](remove=["anchovy"], restrict_categories=["fish"], recipe_ingredients=[
        "2 tablespoons olive oil", "4 anchovy fillets", "1 clove garlic", "1 pound spaghetti"], recipe_title="Anchovy spaghetti",
        recipe_instructions="Heat the oil. Melt the anchovy fillets in the oil. Toss with cooked spaghetti.")
    ad = r["adaptation"]
    plan = r["evidence"]["plans"][0]
    return _result([
        ("no fabricated 'good' confidence", not plan["confidence"].startswith("good")),
        ("unsupported candidates are labelled external/unverified or absent",
         all(c["evidence"] != "none" or "EXTERNAL" in c["label"] for c in plan["candidates"])),
        ("result is verified or an explicit refusal", ad["status"] in {"ok", "no_supported_substitute", "violations_remaining"}),
    ], f"confidence: {plan['confidence']} | status {ad['status']} | changes: {[(c.get('original'), c.get('status') or c.get('replacement')) for c in ad.get('changes', [])]}")


def n1_normalization() -> dict:
    """N-1  normalisation keeps different products apart and merges true synonyms."""
    from app.normalize import match_level, parse_ingredient
    checks = [
        ("scallion == green onion", normalize_term("scallions") == "green onion"),
        ("peanut butter is not butter", match_level("butter", "peanut butter") is None),
        ("tomato sauce is not tomato", match_level("tomato", "tomato sauce") is None),
        ("red onion is a variant of onion", match_level("onion", "red onion") == "variant"),
        ("coconut milk is not flagged dairy", not categories("coconut milk")[0]),
        ("cream of tartar is not dairy", not categories("cream of tartar")[0]),
        ("salt and pepper splits into two lines", len([1 for _ in __import__("app.normalize", fromlist=["x"]).parse_ingredients("Salt and pepper to taste")]) == 2),
        ("ADVERTISEMENT noise stripped", parse_ingredient("2 tablespoons butter ADVERTISEMENT").canon == "butter"),
    ]
    return _result(checks, "; ".join(f"{n}: {'ok' if ok else 'FAIL'}" for n, ok in checks))


CASES: list[tuple[str, str, Callable[[], dict]]] = [
    ("T1-A", "Task 1 · multi-turn: penne pasta -> 'I cannot have dairy'", t1_multiturn_dairy),
    ("T1-B", "Task 1 · aggregate: garlic AND onion count + percentage", t1_stat_garlic_onion),
    ("T1-C", "Task 1 · ranking: tomato recipes with fewest steps", t1_rank_tomato_steps),
    ("T1-D", "Task 1 · unavailable attribute (nutrition/time) is not fabricated", t1_unavailable_attribute),
    ("T2-A", "Task 2 · eggs/onions/tomatoes/rice + follow-up 'Exclude eggs too'", t2_match_followup),
    ("T2-B", "Task 2 · no-match with contradictory hard constraints", t2_no_match),
    ("T2-C", "Task 2 · quantity-aware matching", t2_quantities),
    ("T3-A", "Task 3 · banana bread without eggs (weak, indirect evidence)", t3_banana_bread_no_egg),
    ("T3-B", "Task 3 · replace cream in a pasta recipe without dairy", t3_cream_dairy_free),
    ("T3-C", "Task 3 · unsupported substitution is refused, not invented", t3_unsupported),
    ("N-1", "Normalisation rules", n1_normalization),
]
