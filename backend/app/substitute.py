"""Task 3: evidence-based ingredient substitution.

Pipeline
  1. find the target ingredient lines in the recipe (by term and/or restricted category)
  2. classify the ingredient's ROLE with a small lexicon (binder, fat, liquid, thickener ...)
  3. generate candidates = lexicon pool for that role  U  alternatives discovered in the dataset itself
     ("butter or margarine" style lines); candidates must satisfy the user's restriction
  4. gather LOCAL evidence for each candidate
       direct   : explicit "X or Y" ingredient lines and "instead of X / substitute" sentences in recipes
       indirect : how often the candidate appears in SIMILAR recipes that do not contain the restricted item
  5. label confidence (direct / moderate / weak / none / conflicting), cite recipe ids, and mark anything that
     comes only from general culinary knowledge as EXTERNAL and UNVERIFIED
  6. build the adapted ingredient list + revised instructions deterministically, then re-verify programmatically
     that no restricted ingredient/category remains
"""
from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict

from .normalize import _clean_name, categories, normalize_term, parse_ingredient, resolve_category
from .store import Filters, Store, get_store

SAFETY_NOTE = ("This is a cooking adaptation based on recipe text, not medical or allergy advice. Ingredient lists in "
               "the dataset cannot confirm that a product is free of an allergen (cross-contact, hidden ingredients). "
               "For allergies, check labels and consult a qualified professional.")

# role lexicon: key tokens -> roles + candidate pool. Notes are GENERAL CULINARY KNOWLEDGE (external, unverified).
LEX: dict[str, dict] = {
    "egg": {"roles": ["binder", "moisture", "structure/leavening"],
            "pool": [("applesauce", "binder+moisture; common in cakes/muffins", (0.25, "cup")),
                     ("banana", "binder+moisture; adds banana flavour", (0.25, "cup mashed")),
                     ("ground flaxseed", "binder when mixed with water", (1, "tablespoon", (3, "tablespoons", "water"))),
                     ("silken tofu", "binder/moisture; neutral flavour", (0.25, "cup")),
                     ("yogurt", "moisture+tenderness (dairy)", (0.25, "cup")),
                     ("buttermilk", "moisture+tenderness (dairy)", (0.25, "cup")),
                     ("pumpkin puree", "moisture+binder", (0.25, "cup")), ("vegetable oil", "moisture/fat only; weak binder"),
                     ("cornstarch", "binder/thickener")]},
    "butter": {"roles": ["fat", "flavour", "moisture"],
               "pool": [("olive oil", "fat; savoury/robust flavour"), ("vegetable oil", "fat; neutral"),
                        ("coconut oil", "solid fat; neutral to coconut flavour"), ("margarine", "closest texture (check label for milk)"),
                        ("shortening", "fat; no flavour"), ("applesauce", "partial replacement in baking"),
                        ("avocado", "fat+moisture in baking")]},
    "milk": {"roles": ["liquid", "moisture", "fat/richness"],
             "pool": [("soy milk", "closest protein/behaviour"), ("almond milk", "light, mildly nutty"),
                      ("oat milk", "creamy, neutral"), ("coconut milk", "richer, coconut note"),
                      ("rice milk", "thin, sweet"), ("water", "adds liquid only; less richness"),
                      ("vegetable broth", "savoury dishes only")]},
    "cream": {"roles": ["fat", "body/richness", "thickening"],
              "pool": [("coconut cream", "rich, thick; coconut flavour"), ("coconut milk", "thinner than cream"),
                       ("cashew", "blended cashew cream (tree nut)"), ("silken tofu", "blended for body"),
                       ("soy milk", "thin; add fat/starch"), ("olive oil", "fat only"),
                       ("cornstarch", "thickener to add body"), ("vegetable broth", "savoury liquid only")]},
    "sour cream": {"roles": ["acid", "creaminess", "moisture"],
                   "pool": [("coconut yogurt", "tangy, creamy"), ("silken tofu", "blend with lemon juice"),
                            ("lemon juice", "acidity"), ("yogurt", "dairy"), ("soy yogurt", "tangy")]},
    "cheese": {"roles": ["fat", "salt/umami", "melt/binding"],
               "pool": [("nutritional yeast", "umami/cheesy flavour"), ("tofu", "crumbled for texture"),
                        ("soy cheese", "melts"), ("cashew", "cashew-based cheese (tree nut)"), ("salt", "salt only")]},
    "flour": {"roles": ["structure", "thickening"],
              "pool": [("rice flour", "gluten-free structure"), ("cornstarch", "thickening"), ("almond flour", "tree nut; denser"),
                       ("oat flour", "check gluten cross-contact"), ("gluten-free flour", "blend"), ("potato starch", "thickening"),
                       ("tapioca flour", "chew/thickening")]},
    "honey": {"roles": ["sweetener", "moisture"],
              "pool": [("maple syrup", "closest plant-based swap"), ("agave nectar", "similar viscosity"), ("sugar", "dry sweetener; add liquid")]},
    "sugar": {"roles": ["sweetener", "moisture/browning"],
              "pool": [("honey", "liquid sweetener"), ("maple syrup", "liquid sweetener"), ("brown sugar", "more moisture")]},
    "peanut butter": {"roles": ["fat", "binder", "flavour"],
                      "pool": [("sunflower seed butter", "closest nut-free swap"), ("tahini", "sesame; bitter edge"),
                               ("almond butter", "tree nut"), ("soy nut butter", "soy")]},
    "soy sauce": {"roles": ["salt/umami"],
                  "pool": [("tamari", "soy-based; gluten-free often"), ("coconut aminos", "soy-free, sweeter"),
                           ("worcestershire sauce", "umami/acid"), ("salt", "salt only")]},
    "breadcrumb": {"roles": ["binder", "coating", "texture"],
                   "pool": [("rolled oat", "binder"), ("cracker", "crumbs"), ("cornflake", "crumbs"), ("almond", "ground nuts")]},
    "buttermilk": {"roles": ["acid", "moisture", "tenderness"],
                   "pool": [("milk", "with lemon juice or vinegar"), ("yogurt", "thinned"), ("soy milk", "with lemon juice or vinegar"),
                            ("lemon juice", "acid"), ("vinegar", "acid")]},
    "pasta": {"roles": ["starch base"],
              "pool": [("rice noodle", "gluten-free"), ("rice", "starch base"), ("quinoa", "gluten-free"), ("zucchini", "low-carb noodles")]},
    "wine": {"roles": ["acid", "liquid", "flavour"],
             "pool": [("chicken broth", "savoury liquid"), ("vegetable broth", "savoury liquid"), ("vinegar", "acid"),
                      ("grape juice", "sweet; reduce")]},
}
ALIASES = {"heavy cream": "cream", "light cream": "cream", "whipping cream": "cream", "half and half": "cream",
           "heavy whipping cream": "cream", "cream": "cream", "eggs": "egg", "bread crumb": "breadcrumb",
           "bread crumbs": "breadcrumb", "all-purpose flour": "flour", "wheat flour": "flour", "plain flour": "flour",
           "evaporated milk": "milk", "whole milk": "milk", "unsalted butter": "butter"}


# Candidates that only fix PART of the role (acid, fat, thickener...) or need different amounts/processing. They are
# shown as options but never auto-applied at the original amount.
ADJUNCT: dict[str, set[str]] = {
    "egg": {"vegetable oil", "cornstarch"}, "butter": {"applesauce", "avocado"}, "milk": {"water", "vegetable broth"},
    "cream": {"olive oil", "cornstarch", "vegetable broth"}, "sour cream": {"lemon juice"},
    "cheese": {"nutritional yeast", "salt", "cashew", "tofu"},
    "flour": {"cornstarch", "almond flour", "potato starch", "tapioca flour"}, "honey": {"sugar"},
    "soy sauce": {"salt", "worcestershire sauce"}, "breadcrumb": {"almond"},
    "buttermilk": {"lemon juice", "vinegar", "milk"}, "wine": {"vinegar", "grape juice"},
}

CHEESE_TOKENS = {"parmesan", "parmigiano", "mozzarella", "cheddar", "ricotta", "feta", "romano", "gruyere", "provolone",
                 "mascarpone", "gouda", "brie", "gorgonzola", "asiago", "fontina", "havarti", "emmental", "paneer", "cheese"}


def lexicon_key(canon: str) -> str | None:
    if canon in ALIASES:
        return ALIASES[canon]
    toks = set(canon.split())
    if toks & CHEESE_TOKENS and "cream" not in toks:
        return "cheese"
    if "yogurt" in toks or "yoghurt" in toks:
        return "sour cream"
    best = None
    for key in LEX:
        kt = set(key.split())
        if kt <= toks and (best is None or len(kt) > len(set(best.split()))):
            best = key
    return best


# --------------------------------------------------------------------------------- recipe + targets
def _clean_lines(lines: list[str]) -> list[str]:
    """Drop the dataset's 'ADVERTISEMENT' scraping noise from display copies (the DB keeps the original lines)."""
    out = []
    for ln in lines:
        c = re.sub(r"\s*ADVERTISEMENT\s*", " ", ln or "").strip()
        if c:
            out.append(c)
    return out


def load_recipe(store: Store, recipe_id: str | None = None, recipe_text: dict | None = None) -> dict | None:
    if recipe_id:
        r = store.recipe(recipe_id, full=True)
        if not r:
            return None
        return {"id": r["id"], "title": r["title"], "ingredients": _clean_lines(r["ingredients"]),
                "instructions": r["instructions"] or "", "source": "dataset"}
    if recipe_text:
        ing = recipe_text.get("ingredients") or []
        if isinstance(ing, str):
            ing = [x.strip() for x in ing.splitlines() if x.strip()]
        return {"id": None, "title": recipe_text.get("title") or "user-supplied recipe", "ingredients": _clean_lines(ing),
                "instructions": recipe_text.get("instructions") or "", "source": "user"}
    return None


def find_targets(store: Store, recipe: dict, remove_terms: list[str], categories_to_remove: list[str]) -> list[dict]:
    cats: list[str] = []
    for c in categories_to_remove:
        cats += resolve_category(c)
    for t in remove_terms:
        cats += resolve_category(t)
    term_forms = {t: {c for c, _ in store.resolve_term(t, "broad")} | {normalize_term(t)} for t in remove_terms}
    out = []
    for idx, line in enumerate(recipe["ingredients"]):
        ing = parse_ingredient(line)
        if not ing:
            continue
        reasons = []
        for t, forms in term_forms.items():
            if ing.canon in forms or set(normalize_term(t).split()) <= set(ing.canon.split()):
                reasons.append(f"matches '{t}'")
        definite, _ = categories(ing.canon)
        hit = definite & set(cats)
        if hit:
            reasons.append("contains " + "/".join(sorted(hit)))
        if reasons:
            out.append({"index": idx, "raw": line, "canon": ing.canon, "qty": ing.qty, "unit": ing.unit,
                        "why": "; ".join(reasons)})
    return out


# --------------------------------------------------------------------------------- local evidence
SIZE_WORDS = {"large", "small", "medium", "extra", "whole", "fresh", "dried", "ground", "frozen", "canned", "plain", "more"}
_OR_SPLIT = re.compile(r"\s+or\s+", re.I)


def _violates(canon: str, cats: set[str], banned_forms: set[str]) -> bool:
    # conservative for candidates: an item that only POSSIBLY contains the restricted allergen (margarine for dairy,
    # soy sauce for gluten...) is not offered as a safe replacement
    definite, possible = categories(canon)
    return bool((definite | possible) & cats) or canon in banned_forms


def discover_or_alternatives(store: Store, term: str, cats: set[str], banned: set[str], limit_rows: int = 4000) -> dict:
    """Mine ingredient lines like '1 cup milk or soy milk' for alternatives to `term` (direct evidence)."""
    base = normalize_term(term)
    if not base:
        return {}
    like = f"%{base.split()[-1]}%"
    rows = store.q("SELECT rid, raw FROM or_lines WHERE raw LIKE ? LIMIT ?", (like, limit_rows))
    store._load_vocab()
    vocab = store._vocab or {}
    word = re.compile(rf"\b{re.escape(base.split()[-1])}s?\b")
    alts: dict[str, dict] = {}
    table = store.meta_table()
    for rid, raw in rows:
        m = table.get(rid)
        if not m:
            continue
        parts = _OR_SPLIT.split(raw.split(",")[0])
        if len(parts) != 2 or not word.search(raw.lower()):
            continue
        left_ing = parse_ingredient(parts[0])
        right_canon = _clean_name(parts[1].strip().lower())
        if not left_ing or not right_canon:
            continue
        pair = None
        if base in left_ing.canon.split() or left_ing.canon == base or set(base.split()) <= set(left_ing.canon.split()):
            pair = right_canon
        elif set(base.split()) <= set(right_canon.split()):
            pair = left_ing.canon
        if not pair or pair == base or set(base.split()) <= set(pair.split()):
            continue
        if _violates(pair, cats, banned) or vocab.get(pair, 0) < 30 or set(pair.split()) <= SIZE_WORDS:
            continue
        d = alts.setdefault(pair, {"count": 0, "examples": []})
        d["count"] += 1
        if len(d["examples"]) < 3:
            d["examples"].append({"id": m[0], "line": raw.strip()})
    return alts


def instead_of_snippets(store: Store, term: str, limit: int = 12) -> list[dict]:
    """Sentences in recipe instructions that explicitly talk about replacing `term`."""
    base = normalize_term(term).split()[-1] if normalize_term(term) else term.lower()
    seen, out = set(), []
    table = store.meta_table()
    b = re.escape(base)
    rows = store.q("SELECT rid, sentence FROM subst_sentences WHERE sentence LIKE ?", (f"%{base}%",))
    for rid, sent in rows:
        m = table.get(rid)
        if not m or m[5] != rid or rid in seen:   # skip incomplete recipes and duplicate copies
            continue
        s = sent.lower()
        # the removed ingredient must sit within a few words of the replace/substitute keyword
        if (re.search(rf"(instead of|in place of|replac\w*)\W+(?:\w+\W+){{0,2}}?\b{b}s?\b", s)
                or re.search(rf"substitut\w*\W+(?:\w+\W+){{0,6}}?for\W+(?:\w+\W+){{0,2}}?\b{b}s?\b", s)
                or re.search(rf"\b{b}s?\b\W+(?:\w+\W+){{0,4}}?(?:may|can|could) be (?:substituted|replaced)", s)):
            seen.add(rid)
            out.append({"id": m[0], "title": m[1], "sentence": sent.strip()[:300]})
            if len(out) >= limit:
                break
    return out


def similar_without(store: Store, recipe: dict, f_restrict: Filters, n: int = 60) -> dict:
    """Recipes similar to the target that satisfy the restriction (the 'lacking' group) and, for contrast,
    similar recipes without any restriction (the 'baseline' group)."""
    ing_canons = [i.canon for i in (parse_ingredient(x) for x in recipe["ingredients"]) if i]
    words = [w for w in re.findall(r"[a-z]+", (recipe["title"] or "").lower()) if len(w) > 2]
    # anchor on the dish type: the most specific title phrase (last two words, then last word) that still has
    # enough restriction-compliant recipes; otherwise fall back to a pure similarity search.
    phrases = []
    if len(words) >= 2:
        phrases.append(" ".join(words[-2:]))
    if words:
        phrases.append(words[-1])
    basis, lacking, phrase = "similarity search only (no dish-type phrase had enough compliant recipes)", None, None
    for ph in phrases:
        g = f_restrict.copy()
        g.title_contains = [ph]
        if store.count(g) >= 8:
            lacking = store.search(f"{recipe['title']} " + " ".join(ing_canons[:6]), g, k=n)
            basis = f"recipes whose title contains '{ph}' and satisfy the restriction ({lacking['filtered_total']} exist)"
            phrase = ph
            break
    query = f"{recipe['title']} " + " ".join(ing_canons[:6])
    if lacking is None:
        lacking = store.search(query, f_restrict, k=n)
    return {"query": query, "basis": basis, "phrase": phrase, "lacking": lacking["results"],
            "lacking_pool": lacking["filtered_total"]}


def candidate_support(store: Store, forms: list[str], rids: list[int]) -> list[int]:
    if not forms or not rids:
        return []
    out: list[int] = []
    for i in range(0, len(rids), 400):
        chunk = rids[i:i + 400]
        rows = store.q(f"SELECT DISTINCT rid FROM recipe_ingredients WHERE canon IN ({','.join('?' * len(forms))}) "
                       f"AND rid IN ({','.join('?' * len(chunk))})", [*forms, *chunk])
        out += [r[0] for r in rows]
    return out


def typical_amount(store: Store, forms: list[str], rids: list[int]) -> dict | None:
    """Median quantity/unit of the candidate in the supporting recipes (derived from the dataset)."""
    if not forms or not rids:
        return None
    vals: list[tuple[float, str]] = []
    for i in range(0, len(rids), 400):
        chunk = rids[i:i + 400]
        rows = store.q(f"SELECT qty, unit FROM recipe_ingredients WHERE canon IN ({','.join('?' * len(forms))}) "
                       f"AND qty IS NOT NULL AND rid IN ({','.join('?' * len(chunk))})", [*forms, *chunk])
        vals += [(r[0], r[1] or "") for r in rows]
    if not vals:
        return None
    unit = Counter(u for _, u in vals).most_common(1)[0][0]
    qs = [q for q, u in vals if u == unit]
    return {"median_qty": round(statistics.median(qs), 3), "unit": unit or "(count)", "n": len(qs)}


def prep_snippet(store: Store, forms: list[str], rids: list[int], base: str) -> dict | None:
    """An instruction sentence from a supporting recipe that mentions the candidate (how it is prepared/used)."""
    key = base.split()[-1]
    table = store.meta_table()
    for rid in rids[:25]:
        row = store.q("SELECT instructions FROM recipes WHERE rid = ?", (rid,))
        text = row[0][0] if row else ""
        for sent in re.split(r"(?<=[.!?])\s+|\n", text or ""):
            if key in sent.lower() and len(sent) < 260:
                m = table.get(rid)
                return {"id": m[0] if m else str(rid), "sentence": sent.strip()}
    return None


# --------------------------------------------------------------------------------- main entry
def substitution_evidence(recipe: dict, remove_terms: list[str], restrict_categories: list[str],
                          store: Store | None = None, max_candidates: int = 4) -> dict:
    store = store or get_store()
    remove_terms = [t for t in remove_terms if t]
    cats: set[str] = set()
    for c in restrict_categories:
        cats |= set(resolve_category(c))
    for t in remove_terms:
        cats |= set(resolve_category(t))
    targets = find_targets(store, recipe, remove_terms, restrict_categories)
    out: dict = {"recipe": {"id": recipe["id"], "title": recipe["title"], "source": recipe["source"]},
                 "restriction": {"remove": remove_terms, "categories": sorted(cats)},
                 "targets": targets, "safety_note": SAFETY_NOTE}
    if not targets:
        out["status"] = "nothing_to_substitute"
        out["message"] = ("No ingredient line in this recipe matches the requested removal, so no change was made. "
                          "Check the recipe text or the ingredient name.")
        return out

    banned_forms: set[str] = set()
    for t in remove_terms:
        banned_forms |= {c for c, _ in store.resolve_term(t, "broad")} | {normalize_term(t)}

    f_restrict = Filters(exclude=list(remove_terms), exclude_categories=sorted(cats))
    sim = similar_without(store, recipe, f_restrict)
    lacking_rids = [r["rid"] for r in sim["lacking"]]
    out["similar_recipes"] = {"query_used": sim["query"], "basis": sim["basis"],
                              "similar_without_restricted": len(lacking_rids),
                              "restriction_filter": f_restrict.describe(),
                              "examples": [{"id": r["id"], "title": r["title"]} for r in sim["lacking"][:5]]}

    plans = []
    for tgt in targets:
        key = lexicon_key(tgt["canon"])
        if key is None:
            for t in remove_terms:   # only fall back to the requested term when this very line is that ingredient
                if set(normalize_term(t).split()) <= set(tgt["canon"].split()):
                    key = lexicon_key(normalize_term(t))
        entry = LEX.get(key or "", None)
        with_rids: list[int] = []
        if sim.get("phrase") is not None or sim.get("basis", "").startswith("similarity"):
            wf = Filters(include=[tgt["canon"]], title_contains=[sim["phrase"]] if sim.get("phrase") else [])
            with_rids = [r["rid"] for r in store.search(sim["query"], wf, k=60)["results"]]
        role = entry["roles"] if entry else ["unknown (not in the role lexicon)"]
        pool = list(entry["pool"]) if entry else []
        discovered = discover_or_alternatives(store, tgt["canon"], cats, banned_forms)
        instead = instead_of_snippets(store, tgt["canon"])
        names: dict[str, str] = {}
        ratios: dict[str, tuple | None] = {}
        for pool_entry in pool:
            names.setdefault(pool_entry[0], pool_entry[1])
            ratios[pool_entry[0]] = pool_entry[2] if len(pool_entry) > 2 else None
        pool_names = set(names)
        if entry is None:   # unknown role: trust vetted 'or' alternatives discovered in the dataset
            for nm in sorted(discovered, key=lambda k: -discovered[k]["count"])[:6]:
                names.setdefault(nm, "found as an 'or' alternative in the dataset")
        other_seen = [{"name": k, "lines": v["count"], "examples": v["examples"][:2]}
                      for k, v in sorted(discovered.items(), key=lambda kv: -kv[1]["count"])[:5] if k not in pool_names]
        cands = []
        in_recipe = {i.canon for i in (parse_ingredient(x) for x in recipe["ingredients"]) if i}
        skipped_present = []
        for nm, note in names.items():
            if _violates(normalize_term(nm), cats, banned_forms):
                continue
            forms = [c for c, _ in store.resolve_term(nm, "strict")]
            if set(forms) & in_recipe or normalize_term(nm) in in_recipe:
                # already part of the dish: similar recipes would "support" it only because the dish defines it
                skipped_present.append(nm)
                continue
            if not forms:
                cands.append({"name": nm, "in_dataset": False, "evidence": "none", "external_note": note,
                              "ratio_per_original": ratios.get(nm), "adjunct": nm in ADJUNCT.get(key or "", set()),
                              "label": "EXTERNAL / UNVERIFIED: not found in the dataset's ingredient vocabulary"})
                continue
            sup = candidate_support(store, forms, lacking_rids)
            # popularity control: how common is the candidate in recipes that KEEP the ingredient, vs those without it?
            rate_without = len(sup) / len(lacking_rids) if lacking_rids else 0.0
            rate_with = (len(candidate_support(store, forms, with_rids)) / len(with_rids)) if len(with_rids) >= 5 else None
            lift = round((rate_without + 0.02) / (rate_with + 0.02), 2) if rate_with is not None else None
            direct = discovered.get(normalize_term(nm)) or next((v for k, v in discovered.items() if set(normalize_term(nm).split()) <= set(k.split())), None)
            n_direct = direct["count"] if direct else 0
            if n_direct >= 2:
                tier = "direct"
            elif len(sup) >= 3 and (lift is None or lift >= 1.5):
                tier = "moderate"
            elif len(sup) >= 1 or n_direct == 1:
                tier = "weak"
            else:
                tier = "none"
            cands.append({
                "name": nm, "in_dataset": True, "forms_found": forms[:4], "evidence": tier,
                "direct_alternative_lines": n_direct,
                "direct_examples": (direct or {}).get("examples", []),
                "similar_recipes_using_it": len(sup),
                "of_similar_recipes_without_restricted": len(lacking_rids),
                "rate_in_restricted_recipes": round(rate_without, 3),
                "rate_in_similar_recipes_keeping_ingredient": None if rate_with is None else round(rate_with, 3),
                "lift": lift,
                "lift_note": ("no more common than in similar recipes that keep the ingredient: weak signal"
                              if lift is not None and lift < 1.5 else None),
                "supporting_recipe_ids": [store.meta_table()[r][0] for r in sup[:6] if r in store.meta_table()],
                "typical_amount_in_those_recipes": typical_amount(store, forms, sup),
                "prep_in_local_recipe": prep_snippet(store, forms, sup, nm) if sup else None,
                "external_note": note, "ratio_per_original": ratios.get(nm),
                "adjunct": nm in ADJUNCT.get(key or "", set()),
                "label": ("local evidence" if tier != "none" else "EXTERNAL / UNVERIFIED: no local recipe evidence found"),
            })
        order = {"direct": 0, "moderate": 1, "weak": 2, "none": 3}
        cands.sort(key=lambda c: (order[c["evidence"]], -(c.get("direct_alternative_lines") or 0),
                                  -(c.get("lift") or 0), -(c.get("similar_recipes_using_it") or 0)))
        top = cands[:max_candidates]
        supported = [c for c in cands if c["evidence"] in {"direct", "moderate"}]
        sup_counts = [c.get("similar_recipes_using_it", 0) for c in cands if c.get("similar_recipes_using_it", 0) >= 2]
        conflicting = len(sup_counts) >= 3 and (max(sup_counts) / max(sum(sup_counts), 1)) < 0.4
        small = len(lacking_rids) < 15
        if supported and not conflicting:
            confidence = "good" if any(c["evidence"] == "direct" for c in supported) else "moderate (indirect)"
            if small:
                confidence += f"; small sample: only {len(lacking_rids)} comparable recipes satisfy the restriction"
        elif conflicting:
            confidence = "conflicting: several different approaches appear with no clear consensus"
        elif any(c["evidence"] == "weak" for c in cands):
            confidence = "weak: only 1-2 supporting recipes"
        else:
            confidence = "none locally: suggestions are general culinary knowledge, not verified in the dataset"
        plans.append({"target": tgt, "role": role, "confidence": confidence, "candidates": top,
                      "other_or_alternatives_seen_unvetted": other_seen,
                      "skipped_because_already_in_recipe": skipped_present,
                      "instead_of_mentions": instead[:5],
                      "indirect_evidence_note": ("Evidence is indirect: it comes from different recipes of the same dish type "
                                                 "that happen to lack the restricted item, not from tested swaps.")})
    out["status"] = "ok"
    out["plans"] = plans
    return out


# --------------------------------------------------------------------------------- adaptation builder
def build_adaptation(recipe: dict, evidence: dict, choices: dict[int, str] | None = None,
                     allow_unverified: bool = True) -> dict:
    """Deterministically apply the best-supported candidate per target line and re-verify the result.

    choices: optional {ingredient_index: candidate_name} to override the default (first candidate).
    """
    if evidence.get("status") != "ok":
        return {"status": evidence.get("status"), "message": evidence.get("message")}
    cats = set(evidence["restriction"]["categories"])
    new_lines = list(recipe["ingredients"])
    changes, notes = [], []
    replaced_words: list[tuple[str, str]] = []
    for plan in evidence["plans"]:
        tgt = plan["target"]
        cand = None
        if choices and tgt["index"] in choices:
            cand = next((c for c in plan["candidates"] if c["name"] == choices[tgt["index"]]), None)
        def buildable(c: dict) -> bool:
            # a count-based original (3 eggs) can only be re-sized with a known ratio; otherwise keep the unit.
            # Adjunct items (salt for cheese, oil for cream...) are never swapped in at the original amount.
            if c.get("adjunct") or not c.get("in_dataset", True):
                return False
            return tgt["unit"] is not None or tgt["qty"] is None or bool(c.get("ratio_per_original"))

        if cand is None:
            cand = next((c for c in plan["candidates"] if c["evidence"] in {"direct", "moderate", "weak"} and buildable(c)), None)
        if cand is None and allow_unverified:
            cand = next((c for c in plan["candidates"] if c["evidence"] == "none" and buildable(c)), None)
        if cand is None:
            options = [c["name"] for c in plan["candidates"] if c.get("adjunct")]
            changes.append({"index": tgt["index"], "original": tgt["raw"], "status": "no_supported_substitute",
                            "partial_options_not_applied": options,
                            "message": "No substitute with enough support to apply automatically. A reliable adaptation "
                                       "cannot be asserted; consider omitting it" +
                                       (f" or using a partial fix ({', '.join(options)}) at an amount you choose." if options else ".")})
            notes.append(f"Line '{tgt['raw']}' left unchanged: no reliable substitute found locally.")
            continue
        qty = tgt["qty"]
        unit = tgt["unit"]
        extra_line = None
        amount_note = "same amount as the original (ratio not verified in the dataset)"
        ratio = cand.get("ratio_per_original")
        if tgt["unit"] is None and qty is not None and ratio:
            rq, ru = ratio[0], ratio[1]
            qty, unit = qty * rq, ru
            amount_note = (f"EXTERNAL rule of thumb: {rq:g} {ru} per original unit (general culinary knowledge, "
                           "not verified in the dataset)")
            if len(ratio) > 2:
                eq, eu, en = ratio[2]
                extra_line = f"{tgt['qty'] * eq:g} {eu} {en}"
        parts = []
        if qty is not None:
            parts.append(f"{qty:g}")
        if unit:
            parts.append(unit)
        parts.append(cand["name"])
        new_line = " ".join(parts)
        if extra_line:
            new_line += f" (mixed with {extra_line}, let sit a few minutes)"
        new_lines[tgt["index"]] = new_line
        changes.append({"index": tgt["index"], "original": tgt["raw"], "replacement": new_line,
                        "substitute": cand["name"], "evidence": cand["evidence"], "amount": amount_note,
                        "caveat": ("UNVERIFIED: no local recipe evidence; based on general culinary knowledge"
                                   if cand["evidence"] == "none" else None),
                        "supporting_recipe_ids": cand.get("supporting_recipe_ids", []),
                        "role_kept": plan["role"]})
        replaced_words.append((tgt["canon"], cand["name"]))
        if cand.get("prep_in_local_recipe"):
            p = cand["prep_in_local_recipe"]
            notes.append(f"How '{cand['name']}' is used in recipe {p['id']}: \"{p['sentence']}\"")
        if cand.get("external_note") and cand["evidence"] == "none":
            notes.append(f"{cand['name']}: {cand['external_note']} (EXTERNAL general knowledge, unverified)")
    # revised instructions: swap mentions of the removed ingredient, then append adaptation notes
    text = recipe["instructions"] or ""
    for old, new in replaced_words:
        head = old.split()[-1]
        text = re.sub(rf"\b{re.escape(head)}s?\b", new, text, flags=re.I)
    revised_steps = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", text) if s.strip()]
    if changes:
        revised_steps.append("Adaptation note: ingredients were substituted as listed above; expect differences in "
                             "texture/flavour and adjust bake or cook time by sight if needed.")
    # programmatic re-verification of the adapted list
    violations = []
    for line in new_lines:
        ing = parse_ingredient(line)
        if not ing:
            continue
        definite, _ = categories(ing.canon)
        hit = definite & cats
        if hit:
            violations.append({"line": line, "categories": sorted(hit)})
    banned = {normalize_term(t) for t in evidence["restriction"]["remove"]}
    for line in new_lines:
        ing = parse_ingredient(line)
        if ing and ing.canon in banned:
            violations.append({"line": line, "categories": ["removed ingredient still present"]})
    unresolved = [c["original"] for c in changes if c.get("status") == "no_supported_substitute"]
    message = None
    if unresolved:
        message = ("A reliable adaptation could NOT be supported for: " + "; ".join(unresolved) +
                   ". These lines were left unchanged, so the restricted ingredient is still present. "
                   "Omit them, choose a substitute yourself, or request an external (web) check.")
    return {"status": "ok" if not violations else "violations_remaining", "message": message, "unresolved_lines": unresolved,
            "changes": changes, "updated_ingredients": new_lines, "revised_instructions": revised_steps,
            "notes": notes, "verification": {"restricted_categories_checked": sorted(cats),
                                             "violations": violations,
                                             "result": "no restricted ingredient detected in the adapted list (text-based check only)"
                                             if not violations else "RESTRICTED INGREDIENT STILL PRESENT"},
            "safety_note": SAFETY_NOTE}
