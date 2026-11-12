import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.normalize import categories, count_steps, match_level, parse_ingredient, parse_ingredients, resolve_category  # noqa: E402


def canon(line: str) -> str:
    ing = parse_ingredient(line)
    assert ing is not None
    return ing.canon


def test_parsing_strips_quantity_prep_and_noise():
    assert canon("4 skinless, boneless chicken breast halves ADVERTISEMENT") == "chicken breast"
    assert canon("1 onion, finely diced") == "onion"
    assert canon("3 cloves garlic, minced") == "garlic"
    assert canon("2 large eggs") == "egg"
    assert canon("1 cup heavy whipping cream") == "heavy cream"


def test_quantity_and_unit():
    ing = parse_ingredient("1 1/2 cups all-purpose flour")
    assert (ing.qty, ing.unit, ing.canon) == (1.5, "cups", "all-purpose flour")


def test_section_headers_are_ignored():
    assert parse_ingredient("For the sauce:") is None


def test_salt_and_pepper_split():
    assert [i.canon for i in parse_ingredients("Salt and pepper to taste")] == ["salt", "black pepper"]
    assert [i.canon for i in parse_ingredients("kosher salt and freshly ground black pepper")] == ["salt", "black pepper"]


def test_products_are_not_merged():
    assert match_level("butter", "peanut butter") is None
    assert match_level("tomato", "tomato sauce") is None
    assert match_level("milk", "coconut milk") is None
    assert match_level("chicken", "chicken fat") is None
    assert match_level("onion", "green onion") is None
    assert match_level("onion", "red onion") == "variant"
    assert match_level("milk", "low fat milk") == "variant"


def test_allergen_rules():
    assert categories("butter")[0] == {"dairy"}
    assert not categories("coconut milk")[0]
    assert not categories("cream of tartar")[0]
    assert "peanut" in categories("peanut butter")[0]
    assert "dairy" not in categories("peanut butter")[0]
    assert "dairy" in categories("margarine")[1]       # only POSSIBLE: surfaced as a warning, not excluded
    assert not categories("rice noodle")[0]
    assert "gluten" in categories("all-purpose flour")[0]


def test_category_resolution():
    assert set(resolve_category("vegan")) >= {"meat", "dairy", "egg", "honey"}
    assert resolve_category("dairy-free") == ["dairy"]


def test_step_counting_rule():
    assert count_steps("Mix well. Bake at 350 degrees for 1.5 hours. Serve.") == 3
    assert count_steps("Mix\nBake it now") == 2
    assert count_steps("") == 0 and count_steps(None) == 0
