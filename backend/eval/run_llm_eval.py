"""Live evaluation: scripted conversations through the real agent (Gemini + tools) vs a plain one-shot Gemini answer.

Needs GEMINI_API_KEY.   python eval/run_llm_eval.py [--md ../docs/EVAL_LLM.md]

For each case we record: tools the agent called, automatic checks on its reply, the ground truth computed by the tools,
and the one-shot (no tools, no dataset) answer to the same question, so grounded vs ungrounded behaviour can be compared.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import chat  # noqa: E402
from app.llm import MODEL, PROVIDER, plain_completion  # noqa: E402
from app.session import Session  # noqa: E402
from app.stats import run_stats  # noqa: E402
from app.store import Filters, get_store  # noqa: E402


def one_shot(question: str) -> str:
    return plain_completion(question)


def digits(s: str) -> str:
    return re.sub(r"[^0-9.]", "", s)


def run_case(case: dict) -> dict:
    s = Session()
    turns = []
    tools_used: list[str] = []
    for q in case["turns"]:
        t = time.time()
        res = chat(s, q)
        tools_used += [c["name"] for c in res["tool_calls"]]
        turns.append({"q": q, "reply": res["reply"], "calls": res["tool_calls"], "secs": round(time.time() - t, 1)})
    final = re.sub(r"[*`]", "", turns[-1]["reply"])      # checks run on the reply with markdown emphasis removed
    checks = [(name, bool(fn(final, turns, tools_used))) for name, fn in case["checks"]]
    base = one_shot(case.get("baseline_q") or case["turns"][-1]) if case.get("baseline", True) else None
    return {"id": case["id"], "desc": case["desc"], "turns": turns, "tools": tools_used, "checks": checks,
            "truth": case.get("truth", lambda: "")(), "baseline": base}


def build_cases() -> list[dict]:
    store = get_store()
    exact = run_stats("percentage", Filters(include=["garlic", "onion"]), match_mode="exact", store=store)["result"]
    strict = run_stats("count", Filters(include=["garlic", "onion"]), match_mode="strict", store=store)["result"]
    rank = run_stats("rank_by_steps", Filters(include=["tomato"]), limit=5, store=store)["result"]
    ids = [r["id"] for r in rank]
    return [
        {"id": "L1", "desc": "Aggregate statistic must come from computation (one-shot LLM can only guess)",
         "turns": ["How many recipes contain both garlic and onion? What percentage of the dataset is that?"],
         "truth": lambda: f"exact-name count {exact['count']} of {exact['denominator']} usable recipes = {exact['percent']}%; strict (incl. variants) {strict}",
         "checks": [("calls dataset_stats", lambda r, t, u: "dataset_stats" in u),
                    ("reply contains the exact computed count or the strict count",
                     lambda r, t, u: str(exact["count"]) in digits(r) or str(strict) in digits(r)),
                    ("reply states the denominator", lambda r, t, u: str(exact["denominator"]) in digits(r))]},
        {"id": "L2", "desc": "Ranking by steps shows recipe ids and the counting rule",
         "turns": ["Among recipes containing tomato, which five have the fewest instruction steps?"],
         "truth": lambda: "expected ids: " + ", ".join(ids),
         "checks": [("calls dataset_stats", lambda r, t, u: "dataset_stats" in u),
                    ("reply cites >= 3 of the true ids", lambda r, t, u: sum(i[:12] in r for i in ids) >= 3),
                    ("mentions ties or the step rule", lambda r, t, u: bool(re.search(r"tie|sentence|rule", r, re.I)))]},
        {"id": "L3", "desc": "Multi-turn refinement: 'I cannot have dairy' applies to the ACTIVE penne search",
         "turns": ["Find a penne pasta recipe.", "I cannot have dairy."],
         "checks": [("second turn calls search_recipes", lambda r, t, u: u.count("search_recipes") >= 2),
                    ("dairy restriction passed as a filter, query not re-invented",
                     lambda r, t, u: any("dairy" in str(c["args"]) for c in t[1]["calls"])),
                    ("reply mentions penne", lambda r, t, u: "penne" in r.lower())]},
        {"id": "L4", "desc": "Unavailable attribute: nutrition must be declined, not invented",
         "turns": ["Find a banana bread recipe.", "How many calories does the first one have?"],
         "baseline_q": "How many calories are in the recipe 'Banana Banana Bread' (id ar:kq.naD.8G19M4UU9dVvJgHtpfo.l/eC) "
                       "from the AllRecipes part of the Recipe Box dataset?",
         "checks": [("reply says the data is not available", lambda r, t, u: bool(re.search(r"not (available|in|provided|include)|no nutrition|does not (contain|include|have)|unavailable", r, re.I))),
                    ("no calorie number is asserted as fact", lambda r, t, u: not re.search(r"\b\d{2,4}\s*(kcal|calories)\b", r, re.I) or bool(re.search(r"estimate|external", r, re.I)))]},
        {"id": "L5", "desc": "Ingredient matching with a follow-up exclusion",
         "turns": ["I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many as possible, with no peanuts and at most six instruction steps. Prefer recipes that need the fewest additional ingredients.",
                   "Exclude eggs too."],
         "checks": [("uses match_by_ingredients on both turns", lambda r, t, u: u.count("match_by_ingredients") >= 2),
                    ("second call adds only the new restriction (no full re-spec needed)",
                     lambda r, t, u: any("egg" in str(c["args"]).lower() for c in t[1]["calls"])),
                    ("reply lists matched/missing ingredients", lambda r, t, u: bool(re.search(r"missing", r, re.I)))]},
        {"id": "L6", "desc": "Substitution with weak local evidence must say so",
         "turns": ["Take a banana bread recipe and remove eggs. Use only ingredients that appear in the dataset. What replacements have evidence from related baked recipes, and how might the texture change?"],
         "checks": [("calls adapt_recipe", lambda r, t, u: "adapt_recipe" in u),
                    ("reports evidence strength honestly", lambda r, t, u: bool(re.search(r"weak|indirect|small sample|limited|moderate|unverified", r, re.I))),
                    ("labels texture effects as predictions", lambda r, t, u: bool(re.search(r"predict|expect|likely|may|might", r, re.I))),
                    ("cites recipe ids", lambda r, t, u: bool(re.search(r"`?(ar|epi|fn):", r)))]},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(Path(__file__).resolve().parents[2] / "docs" / "EVAL_LLM.md"))
    ap.add_argument("--only", help="comma-separated case ids to (re)run, e.g. L2,L5; others keep their saved result")
    args = ap.parse_args()
    store_path = Path(args.md).with_suffix(".json")
    saved: dict[str, dict] = {}
    if store_path.exists():
        saved = {r["id"]: r for r in json.loads(store_path.read_text(encoding="utf-8"))}
    only = {x.strip() for x in args.only.split(",")} if args.only else None
    cases = build_cases()
    for case in cases:
        if only and case["id"] not in only:
            continue
        print("running", case["id"], case["desc"])
        try:
            r = run_case(case)
            for t in r["turns"]:      # keep the saved file small: the full tool output lives in the app trace
                t["calls"] = [{"name": c["name"], "args": c["args"]} for c in t["calls"]]
            saved[case["id"]] = r
        except Exception as e:  # noqa: BLE001
            print("  failed:", e)
            if case["id"] not in saved or not only:
                saved[case["id"]] = {"id": case["id"], "desc": case["desc"], "turns": [], "tools": [],
                                     "checks": [("ran", False)], "truth": "", "baseline": f"error: {e}"}
    results = [saved[c["id"]] for c in cases if c["id"] in saved]
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    lines = [f"# Live LLM evaluation ({PROVIDER}: {MODEL})", "",
             "Agent = the chosen LLM + tools over the dataset. One-shot = the same final question sent to the same model with no tools/dataset.", ""]
    ok_all = True
    for r in results:
        passed = all(ok for _, ok in r["checks"])
        ok_all &= passed
        lines += [f"## {r['id']} - {r['desc']}", "", f"Agent checks: **{'PASS' if passed else 'FAIL'}**", ""]
        lines += [f"- {'[x]' if ok else '[ ]'} {name}" for name, ok in r["checks"]]
        lines += ["", f"Tools called: `{' -> '.join(r['tools']) or 'none'}`", ""]
        if r["truth"]:
            lines += [f"Ground truth (computed): {r['truth']}", ""]
        for t in r["turns"]:
            lines += [f"**User:** {t['q']}", "", f"**Agent ({t['secs']}s):**", "", t["reply"], ""]
        if r["baseline"] is not None:
            lines += [f"**One-shot {MODEL} (no tools, no dataset):**", "", r["baseline"], ""]
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(lines), encoding="utf-8")
    print("wrote", args.md)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
