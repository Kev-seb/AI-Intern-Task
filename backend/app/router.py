"""Deterministic first-step tool routing for questions that can only be answered by computation.

Models sometimes answer "which five have the fewest steps" by browsing with search_recipes. For clearly statistical
phrasing we force `dataset_stats` on the first model step (later steps stay automatic). Anything ambiguous is left to
the model.
"""
from __future__ import annotations

import re

STATS_PATTERNS = [
    r"\bhow many (?:recipes|dishes)\b",
    r"\bwhat (?:percentage|percent|proportion|share)\b",
    r"\b(?:percentage|percent) of (?:the )?(?:dataset|recipes)\b",
    r"\b(?:average|mean|median)\b.{0,40}\b(?:ingredients?|steps?|instructions?)\b",
    r"(?<!at )\b(?:fewest|least|most|lowest|highest)\b.{0,30}\b(?:instruction )?(?:steps?|ingredients?)\b",
    r"\bmost common (?:ingredients?)\b|\bcommon ingredients?\b",
    r"\b(?:rank|ranking)\b.{0,40}\b(?:steps?|ingredients?)\b",
    r"\bwhich (?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b.{0,60}(?<!at )\b(?:fewest|least|most|steps?)\b",
]
NOT_STATS = re.compile(r"\b(?:calories?|nutrition|nutritional|protein grams?|cook(?:ing)? time|price|cost)\b", re.I)


def route_tool(message: str) -> str | None:
    """Return the tool name to force on the first step, or None to let the model choose."""
    text = (message or "").lower()
    if NOT_STATS.search(text):
        return None
    if any(re.search(p, text) for p in STATS_PATTERNS):
        return "dataset_stats"
    return None
