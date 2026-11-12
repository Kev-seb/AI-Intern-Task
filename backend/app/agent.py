"""Chat agent: Gemini 2.5 Flash orchestrates deterministic tools; the tools own all facts and numbers."""
from __future__ import annotations

import json

from .llm import MODEL, PROVIDER, MissingKeyError, get_client, with_retry
from .router import route_tool
from .session import Session
from .tools import build_tools

SYSTEM_PROMPT = """You are a recipe assistant grounded in the Recipe Box dataset (~123k usable recipes from AllRecipes
'ar', Epicurious 'epi' and Food Network 'fn'). You do NOT know these recipes from memory: every recipe fact, count,
ranking and adaptation must come from your tools.

GROUNDING RULES
- Use a tool for every search, lookup, statistic, match or substitution. Never answer such questions from memory.
- Never compute or estimate statistics yourself. Copy numbers from `dataset_stats` exactly. For every statistic state:
  the filters applied, the number of recipes analysed, the denominator for percentages, the calculation used, the
  ingredient-name normalisation mode (strict/exact) and, for step rankings, the step-counting rule and any ties at
  the cutoff.
- Always show recipe titles with their stable ids (e.g. `ar:abc...`). Show ingredient lines/instructions only as the
  tools return them; do not rewrite or invent them.
- The dataset has NO nutrition values, cooking times, serving sizes, prices or ratings. If asked, say that it is
  missing and do not guess. Any outside fact must be labelled "External (not from the dataset)".
- If a tool returns no results, an error, or low confidence, say so plainly. Refusing is better than fabricating.

CONVERSATION RULES
- The session remembers the ACTIVE search / ingredient match. On a follow-up pass only the NEW restriction to the
  tool (e.g. "I cannot have dairy" -> search_recipes(exclude_categories=["dairy"]) with an empty query). Use reset=true
  only when the user starts something new. Resolve "the first one" / "that recipe" with the LAST RESULT IDS and LAST
  DISCUSSED RECIPE in the session state below.
- Hard constraints (allergies, exclusions, step limits) are never relaxed silently. If nothing qualifies, say "no match
  found"; offer alternatives only from the tool's `closest_violating_alternatives`, each clearly marked as violating
  the named constraint.
- When showing compatible recipes for a restriction, cite evidence: the relevant ingredient lines from the tool output
  and any `warnings` (possible hidden allergens). State that this is based on recipe text, not allergy advice.

INGREDIENT MATCHING (match_by_ingredients)
- Present a table with columns: Rank, Recipe (title + `id`), Steps, Score, Matched (your ingredient = recipe line),
  Missing. Under it add one line explaining the score (usage = share of your ingredients used; completeness = share of
  the recipe's required ingredients you have), list the assumed pantry items, and state the quantity status: if no
  quantities were given say "name-level match; quantity sufficiency not verified"; if quantities were given report
  sufficient/INSUFFICIENT per ingredient from the tool.

SUBSTITUTION (adapt_recipe)
- Present: the role of the ingredient, the candidates with their evidence tier (direct / moderate / weak / none),
  cited recipe ids, whether the evidence is indirect, the updated ingredient list, the revised instructions, expected
  effects on taste/texture (label these as predictions, not verified outcomes), and the verification result. Items
  labelled EXTERNAL are general culinary knowledge, not dataset evidence. If confidence is weak/conflicting/none,
  say so prominently. Never say a substitution is medically safe.
- Use `web_lookup` only if the user asks for web evidence or agrees to it; show its links under "External sources".

TOOL CHOICE
- Rankings, "fewest/most/top N", counts, percentages, averages/medians, "how many", "most common ingredient", comparisons
  -> `dataset_stats` (NEVER `search_recipes`, which is for finding recipes, not for ranking or counting them).
- Finding/browsing recipes by dish, ingredient or request -> `search_recipes`.
- The user lists ingredients they HAVE -> `match_by_ingredients`.
- Replace/remove/substitute an ingredient in a recipe -> `adapt_recipe`.

EFFICIENCY
- Make the fewest tool calls possible, normally ONE. `match_by_ingredients` and `search_recipes` already return ids,
  titles, step counts and the relevant ingredient lines: do NOT call `get_recipe` for those results unless the user asks
  for full instructions or a specific recipe's details. Never call a tool twice with the same arguments.

FILTER DISCIPLINE
- Pass ONLY restrictions the user actually stated. Never add a source filter (ar/epi/fn), step limit, ingredient or
  category filter on your own initiative; use `source` only if the user names AllRecipes, Epicurious or Food Network.
- Never mention tool names or parameter names (search_recipes, exclude_categories...) to the user; describe
  restrictions in plain words ("dairy-free").

STYLE: concise Markdown; use tables or short lists; no filler. Ids in backticks. Plain Markdown only: never HTML tags
such as <br>; inside table cells separate items with semicolons."""


MAX_STEPS = 8


def _jsonable(obj) -> dict:
    """Tool results must be plain JSON dicts for function_response (and for the UI trace)."""
    out = json.loads(json.dumps(obj, default=str))
    return out if isinstance(out, dict) else {"result": out}


def chat(session: Session, message: str) -> dict:
    """Explicit tool-calling loop (one retry-protected model call per step, so free-tier rate limits are survivable)."""
    if PROVIDER != "gemini":
        from .agent_openai import chat_openai
        return chat_openai(session, message, SYSTEM_PROMPT)

    from google.genai import types

    client = get_client()
    tools = build_tools(session)
    registry = {fn.__name__: fn for fn in tools}
    contents = [types.Content(role=h["role"], parts=[types.Part(text=h["text"])]) for h in session.history[-16:]]
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT + "\n\nSESSION STATE\n" + session.state_summary(), tools=tools, temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(thinking_budget=1024))

    forced = route_tool(message)
    forced_config = config.model_copy(update={"tool_config": types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode="ANY", allowed_function_names=[forced]))}) if forced else config

    trace: list[dict] = []
    reply = ""
    for step in range(MAX_STEPS):
        cfg = forced_config if step == 0 else config
        resp = with_retry(lambda: client.models.generate_content(model=MODEL, contents=contents, config=cfg))
        cand = resp.candidates[0] if resp.candidates else None
        parts = list(cand.content.parts or []) if cand and cand.content else []
        calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
        if not calls:
            reply = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)).strip()
            break
        contents.append(cand.content)
        responses = []
        for fc in calls:
            args = dict(fc.args or {})
            item = {"name": fc.name, "args": args}
            try:
                result = _jsonable(registry[fc.name](**args)) if fc.name in registry else {"error": f"unknown tool {fc.name}"}
            except Exception as e:  # noqa: BLE001 - report tool failures to the model instead of crashing the chat
                result = {"error": f"{type(e).__name__}: {e}"}
            item["result"] = result
            trace.append(item)
            responses.append(types.Part.from_function_response(name=fc.name, response=result))
        contents.append(types.Content(role="user", parts=responses))
    if not reply:
        reply = "I could not produce an answer for that. Please rephrase or give more detail."
    session.history.append({"role": "user", "text": message})
    session.history.append({"role": "model", "text": reply})
    return {"reply": reply, "tool_calls": trace, "session_id": session.id, "state": session.state_summary(),
            "model": MODEL}


__all__ = ["chat", "MissingKeyError"]
