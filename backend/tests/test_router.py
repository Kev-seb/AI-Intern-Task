import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.router import route_tool  # noqa: E402


@pytest.mark.parametrize("msg", [
    "How many recipes contain both garlic and onion? What percentage of the dataset is that?",
    "Among recipes containing tomato, which five have the fewest instruction steps?",
    "What is the average number of ingredients in chicken recipes?",
    "What are the most common ingredients in cookies?",
    "What is the median number of steps for pasta recipes?",
])
def test_statistical_questions_are_routed(msg):
    assert route_tool(msg) == "dataset_stats"


@pytest.mark.parametrize("msg", [
    "Find a penne pasta recipe.",
    "I cannot have dairy.",
    "How many calories does the first one have?",
    "How long does it take to cook the first recipe?",
    "I have eggs, onions, tomatoes, and rice. Suggest five recipes with at most six steps.",
    "Take a banana bread recipe and remove eggs.",
    "Exclude eggs too.",
    "I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many of these as possible, with no peanuts "
    "and at most six instruction steps. Prefer recipes that need the fewest additional ingredients.",
    "Which recipes need the fewest missing ingredients if I have chicken and rice?",
    "Replace cream in this pasta recipe without dairy and keep the most ingredients the same.",
])
def test_other_questions_are_left_to_the_model(msg):
    assert route_tool(msg) is None
