"""Ingredient-line parsing, normalization, matching and allergen/diet categories.

Design (kept deliberately rule-based so every decision is explainable):

* ``parse_ingredient`` turns "2 (10.75 ounce) cans condensed cream of chicken soup ADVERTISEMENT"
  into quantity / unit / a canonical name ("condensed cream of chicken soup").
* Pure preparation words (chopped, minced, softened...) are dropped. Product-form words
  (canned, dried, sauce, paste, flour, "coconut milk"...) are KEPT, because they change what the
  ingredient is.
* ``match_level`` compares two canonical names token-wise: exact, or a "variant" when the only
  difference is a harmless modifier (red onion vs onion). Different products never match
  (peanut butter vs butter, tomato sauce vs tomato).
* ``categories`` flags allergen / diet groups. "definite" hits are exclusion-grade; "possible"
  hits (e.g. chocolate may contain milk) are only surfaced as warnings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

UNICODE_FRACTIONS = {
    "½": " 1/2", "¼": " 1/4", "¾": " 3/4", "⅓": " 1/3", "⅔": " 2/3",
    "⅛": " 1/8", "⅜": " 3/8", "⅝": " 5/8", "⅞": " 7/8",
}

UNITS = {
    "cup", "cups", "c", "tablespoon", "tablespoons", "tbsp", "tbsps", "tbs", "tbl", "teaspoon",
    "teaspoons", "tsp", "tsps", "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs", "pinch",
    "pinches", "dash", "dashes", "can", "cans", "package", "packages", "pkg", "packet", "packets",
    "clove", "cloves", "slice", "slices", "stick", "sticks", "quart", "quarts", "qt", "pint", "pints",
    "pt", "gram", "grams", "g", "kg", "kilogram", "ml", "milliliter", "liter", "liters", "litre",
    "gallon", "gallons", "jar", "jars", "bunch", "bunches", "head", "heads", "sprig", "sprigs",
    "stalk", "stalks", "bottle", "bottles", "container", "containers", "box", "boxes", "bag", "bags",
    "fluid", "fl", "drop", "drops", "piece", "pieces", "handful", "handfuls", "cube", "cubes",
    "envelope", "envelopes", "square", "squares", "inch", "pat", "pats", "knob", "scoop", "scoops",
    "dollop", "splash", "rib", "ribs", "fillet", "fillets", "loaf", "loaves", "sheet", "sheets",
}

# Words that only describe preparation / size / state and never change the product.
PREP_WORDS = {
    "chopped", "minced", "diced", "sliced", "slices", "grated", "shredded", "fresh", "freshly",
    "finely", "coarsely", "roughly", "thinly", "thickly", "large", "small", "medium", "big", "ripe",
    "peeled", "seeded", "cored", "drained", "rinsed", "divided", "softened", "melted", "cooked",
    "uncooked", "packed", "sifted", "beaten", "skinless", "boneless", "trimmed", "halved", "quartered",
    "cubed", "cut", "torn", "washed", "pitted", "stemmed", "deveined", "optional", "taste", "to",
    "about", "approximately", "heaping", "level", "scant", "rounded", "additional", "extra",
    "plus", "more", "for", "garnish", "and", "into", "halves", "half", "pieces", "chunks", "strips", "rings",
    "lightly", "well", "firmly", "loosely", "room", "temperature", "chilled", "cold", "warm",
    "boiling", "thawed", "virgin", "defrosted", "mashed", "crumbled", "julienned", "wedges", "inch", "such",
    "as", "like", "your", "choice", "favorite", "good", "quality", "needed", "ground-up", "the",
    "a", "an", "with", "in", "on", "or", "each", "per", "whole-ish", "bite-size", "bite", "sized",
    "size", "added",
}
# Words in PREP_WORDS are dropped unless that would empty the name (guarded in ``_clean_name``).
KEEP_IF_ONLY: set[str] = set()

LEADING_ADJ = PREP_WORDS | {"skinless", "boneless", "seedless", "lean", "bone-in", "skin-on", "fresh", "dried", "frozen"}

GROUND_KEEP_NEXT = {"beef", "turkey", "pork", "chicken", "lamb", "veal", "meat", "sausage", "bison", "venison"}

SYNONYMS = {
    "scallion": "green onion", "scallions": "green onion", "spring onion": "green onion",
    "spring onions": "green onion", "green onions": "green onion",
    "aubergine": "eggplant", "courgette": "zucchini", "capsicum": "bell pepper",
    "coriander leaf": "cilantro", "coriander leaves": "cilantro", "fresh coriander": "cilantro",
    "confectioners sugar": "powdered sugar", "confectioner's sugar": "powdered sugar",
    "icing sugar": "powdered sugar", "caster sugar": "superfine sugar", "castor sugar": "superfine sugar",
    "plain flour": "all-purpose flour", "all purpose flour": "all-purpose flour",
    "ap flour": "all-purpose flour", "garbanzo bean": "chickpea", "garbanzo": "chickpea",
    "bicarbonate of soda": "baking soda", "double cream": "heavy cream", "whipping cream": "heavy cream",
    "heavy whipping cream": "heavy cream", "single cream": "light cream", "prawn": "shrimp",
    "cornflour": "cornstarch", "corn flour": "cornstarch", "rocket": "arugula", "mince": "ground beef",
    "yoghurt": "yogurt", "pepperoncini": "pepperoncini", "chile": "chili", "chilli": "chili",
    "chiles": "chili", "chillies": "chili", "tomatoe": "tomato", "potatoe": "potato",
    "egg yolks": "egg yolk", "egg whites": "egg white", "eggs": "egg", "large egg": "egg",
    "extra virgin olive oil": "olive oil", "extra-virgin olive oil": "olive oil",
    "vegetable oil": "vegetable oil", "canola oil": "vegetable oil", "veg oil": "vegetable oil",
    "kosher salt": "salt", "sea salt": "salt", "table salt": "salt", "coarse salt": "salt",
    "black pepper": "black pepper", "ground black pepper": "black pepper", "pepper": "black pepper",
    "freshly ground black pepper": "black pepper", "ground pepper": "black pepper",
    "white sugar": "sugar", "granulated sugar": "sugar", "granulated white sugar": "sugar",
    "white granulated sugar": "sugar", "caster": "superfine sugar",
    "garlic clove": "garlic", "garlic cloves": "garlic", "clove garlic": "garlic",
    "cloves garlic": "garlic", "clove of garlic": "garlic", "cloves of garlic": "garlic",
    "garlic clove,": "garlic",
}

IRREGULAR_SINGULAR = {
    "tomatoes": "tomato", "potatoes": "potato", "leaves": "leaf", "halves": "half", "loaves": "loaf",
    "berries": "berry", "cherries": "cherry", "anchovies": "anchovy", "radishes": "radish",
    "peaches": "peach", "dishes": "dish", "mangoes": "mango", "avocados": "avocado",
    "molasses": "molasses", "hummus": "hummus", "asparagus": "asparagus", "couscous": "couscous",
    "swiss": "swiss", "watercress": "watercress", "cress": "cress", "grits": "grits", "oats": "oat",
    "chives": "chive", "olives": "olive", "capers": "caper", "beans": "bean", "peas": "pea",
    "lentils": "lentil", "chickpeas": "chickpea", "noodles": "noodle", "walnuts": "walnut",
    "almonds": "almond", "pecans": "pecan", "cashews": "cashew", "peanuts": "peanut",
    "raisins": "raisin", "dates": "date", "figs": "fig", "prunes": "prune", "apples": "apple",
    "carrots": "carrot", "onions": "onion", "mushrooms": "mushroom", "bananas": "banana",
    "lemons": "lemon", "limes": "lime", "oranges": "orange", "strawberries": "strawberry",
    "blueberries": "blueberry", "raspberries": "raspberry", "cranberries": "cranberry",
    "shallots": "shallot", "cloves": "clove", "peppers": "pepper", "chilies": "chili",
    "chiles": "chili", "shrimps": "shrimp", "tortillas": "tortilla", "biscuits": "biscuit",
    "breasts": "breast", "thighs": "thigh", "legs": "leg", "wings": "wing", "fillets": "fillet",
    "crackers": "cracker", "cookies": "cookie", "sprouts": "sprout", "seeds": "seed",
    "greens": "green", "sausages": "sausage", "pickles": "pickle", "buns": "bun", "rolls": "roll",
    "squash": "squash", "bass": "bass", "citrus": "citrus", "hazelnuts": "hazelnut",
    "pistachios": "pistachio", "pine nuts": "pine nut", "nuts": "nut", "chips": "chip",
    "gnocchi": "gnocchi", "brussels": "brussels", "lemongrass": "lemongrass", "glass": "glass",
    "tomatillos": "tomatillo", "zucchinis": "zucchini", "artichokes": "artichoke",
}

# Modifiers that make an ingredient a *variant* of the base item rather than a different product.
HARMLESS_MODIFIERS = {
    "roma", "plum", "cherry", "grape", "vine", "beefsteak", "yellow", "red", "white", "sweet",
    "extra", "virgin", "light", "dark", "low", "reduced", "sodium", "free", "organic", "kosher",
    "unsalted", "salted", "whole", "lean", "plain", "regular", "instant", "large", "small", "medium",
    "raw", "fresh", "dried", "canned", "frozen", "smoked", "toasted", "roasted", "boneless", "skinless",
    "granulated", "pure", "natural", "unsweetened", "sweetened", "baby", "mini", "jumbo", "ripe",
    "firm", "soft", "hard", "mild", "sharp", "aged", "young", "russet", "idaho", "yukon", "gold",
    "gala", "fuji", "granny", "smith", "golden", "delicious", "navel", "valencia", "hass", "italian",
    "yolk", "flat", "curly", "leaf", "bulk", "mixed", "assorted", "dry", "wet", "pasteurized",
}
# Subset of the above that signals a *form* difference worth flagging to the user.
FORM_MODIFIERS = {"yolk", "white", "dried", "canned", "frozen", "smoked", "toasted", "roasted", "raw", "fresh", "dry"}

_NUM = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
QTY_RE = re.compile(rf"^\s*({_NUM})(?:\s*(?:-|to)\s*({_NUM}))?\s*")
PAREN_RE = re.compile(r"\([^)]*\)")
HEADER_RE = re.compile(r"^[\w \-/&']+:\s*$")


def _to_float(s: str) -> float:
    s = s.strip()
    total = 0.0
    for part in s.split():
        if "/" in part:
            a, b = part.split("/")
            total += float(a) / float(b) if float(b) else 0.0
        else:
            total += float(part)
    return total


@dataclass
class Ingredient:
    raw: str
    canon: str
    qty: float | None
    unit: str | None
    alts: tuple[str, ...] = ()

    @property
    def tokens(self) -> frozenset[str]:
        return frozenset(self.canon.split())


def is_header(line: str) -> bool:
    s = line.strip()
    return bool(s) and bool(HEADER_RE.match(s)) and not re.search(r"\d", s)


def _singular(tok: str) -> str:
    if tok in IRREGULAR_SINGULAR:
        return IRREGULAR_SINGULAR[tok]
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def _clean_name(name: str) -> str:
    name = name.lower()
    name = PAREN_RE.sub(" ", name)
    name = re.sub(r"\badvertisement\b", " ", name)
    segs = [x.strip() for x in name.split(",")]
    while len(segs) > 1 and segs[0] and all(t in LEADING_ADJ for t in segs[0].split()):
        segs = segs[1:]
    name = segs[0]
    name = re.split(r"\s+-\s+|\s+--\s+|;|:", name)[0]
    name = re.sub(r"[^a-z0-9\-'/ ]+", " ", name)
    name = name.replace("'s", "s").replace("'", "")
    toks = name.split()
    out: list[str] = []
    for i, t in enumerate(toks):
        if t == "ground" and i + 1 < len(toks) and toks[i + 1] in GROUND_KEEP_NEXT:
            out.append(t)
            continue
        if t == "ground":
            continue
        if t in PREP_WORDS and t not in KEEP_IF_ONLY:
            continue
        out.append(t)
    if len(out) > 0:
        out = [t for t in out if t not in {"no", "added"}] or out
    if not out:  # everything was dropped (e.g. "salt", "water"): fall back to raw tokens
        out = [t for t in toks if t not in {"to", "taste", "for", "garnish", "and", "of"}]
    # "or" alternatives are handled before this function; strip stray unit words at the head
    while out and out[0] in UNITS:
        out = out[1:]
    out = [_singular(t) if i == len(out) - 1 else t for i, t in enumerate(out)]
    phrase = " ".join(out).strip(" -/")
    phrase = SYNONYMS.get(phrase, phrase)
    # synonym replace on tails like "fresh garlic cloves"
    for k in ("garlic cloves", "garlic clove"):
        if phrase.endswith(k):
            phrase = "garlic"
    return phrase


def _pre_comma(text: str) -> str:
    segs = [x.strip() for x in text.split(",")]
    while len(segs) > 1 and segs[0] and all(t in LEADING_ADJ for t in re.sub(r"[^a-z\- ]", " ", segs[0]).split()):
        segs = segs[1:]
    return segs[0]


SALT_PEPPER_RE = re.compile(
    r"^(?:[\d/ .]*\s*)?(?:(?:pinch|dash)(?:es)?\s+(?:of\s+)?)?"
    r"(?:(?:kosher|sea|table|coarse|fine|flaky|fresh|freshly)\s+)*salt\s*(?:,|and|&)\s*"
    r"(?:(?:freshly|fresh|ground|black|white|cracked)\s+)*pepper\b", re.I)


def parse_ingredients(raw: str) -> list[Ingredient]:
    """Like ``parse_ingredient`` but splits the common 'salt and pepper' line into two ingredients."""
    if raw and SALT_PEPPER_RE.match(raw.replace("ADVERTISEMENT", "").strip()):
        return [Ingredient(raw=raw.strip(), canon="salt", qty=None, unit=None),
                Ingredient(raw=raw.strip(), canon="black pepper", qty=None, unit=None)]
    ing = parse_ingredient(raw)
    return [ing] if ing else []


def parse_ingredient(raw: str) -> Ingredient | None:
    if not raw:
        return None
    s = raw.replace("ADVERTISEMENT", " ").strip()
    for u, a in UNICODE_FRACTIONS.items():
        s = s.replace(u, a)
    s = re.sub(r"\s+", " ", s).strip()
    if not s or is_header(s):
        return None
    qty = None
    m = QTY_RE.match(s)
    rest = s
    if m:
        try:
            qty = _to_float(m.group(1))
        except (ValueError, ZeroDivisionError):
            qty = None
        rest = s[m.end():]
    rest = PAREN_RE.sub(" ", rest) if re.match(r"^\s*\(", rest) else rest
    # a second number block like "2 (10 ounce) cans": after dropping parens, take the unit
    m2 = re.match(r"^\s*(\d+(?:\.\d+)?)\s+", rest)
    if m2:
        rest = rest[m2.end():]
    toks = rest.strip().split()
    unit = None
    if toks and toks[0].lower().strip(".") in UNITS:
        unit = toks[0].lower().strip(".")
        toks = toks[1:]
        if toks and toks[0].lower() in {"of", "fluid"}:
            toks = toks[1:]
            if toks and toks[0].lower() in {"ounce", "ounces", "oz"}:
                unit = "fl oz"
                toks = toks[1:]
    name_part = " ".join(toks)
    # "x or y" alternatives
    parts = re.split(r"\s+or\s+", _pre_comma(name_part.lower()), maxsplit=1)
    canon = _clean_name(parts[0])
    alts = tuple(a for a in (_clean_name(p) for p in parts[1:]) if a)
    if not canon:
        return None
    return Ingredient(raw=raw.replace("ADVERTISEMENT", "").strip(), canon=canon, qty=qty, unit=unit, alts=alts)


def normalize_term(term: str) -> str:
    """Normalize a user-typed ingredient name ("Tomatoes", "chopped onions") to canonical form."""
    term = (term or "").strip().lower()
    return _clean_name(term)


def match_level(query: str, ing: str) -> str | None:
    """Compare canonical names. Returns 'exact', 'variant' (harmless modifier differs) or None."""
    if not query or not ing:
        return None
    if query == ing:
        return "exact"
    qt, it = set(query.split()), set(ing.split())
    if qt == it:
        return "exact"

    def harmless(extra: set[str], whole: set[str]) -> bool:
        # "fat" only counts as a modifier in "low fat"/"reduced fat"/"fat free"; "chicken fat" is a product.
        if "fat" in extra and whole & {"low", "reduced", "free", "non", "nonfat"}:
            extra = extra - {"fat"}
        return extra <= HARMLESS_MODIFIERS

    if qt < it and harmless(it - qt, it):
        return "variant"
    if it < qt and harmless(qt - it, qt):
        return "variant"
    if (qt - HARMLESS_MODIFIERS) == (it - HARMLESS_MODIFIERS) and (qt - HARMLESS_MODIFIERS):
        return "variant"
    return None


def variant_note(query: str, ing: str) -> str:
    diff = (set(query.split()) ^ set(ing.split()))
    forms = diff & FORM_MODIFIERS
    if forms:
        return f"form differs ({', '.join(sorted(forms))})"
    return "variety/modifier differs"


# ---------------------------------------------------------------------------------------------
# Allergen / diet categories
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    triggers: frozenset[str]          # single tokens that trigger
    phrases: tuple[str, ...] = ()      # multi-token phrases that trigger
    exceptions: frozenset[str] = frozenset()   # tokens that cancel the trigger (e.g. coconut milk)
    except_phrases: tuple[str, ...] = ()
    possible: frozenset[str] = frozenset()     # tokens that only *possibly* contain it


NONDAIRY_QUALIFIERS = frozenset({"coconut", "almond", "soy", "oat", "rice", "cashew", "peanut", "cocoa",
                                 "shea", "hemp", "macadamia", "hazelnut", "pea", "flax", "vegan", "nondairy"})

RULES: dict[str, Rule] = {
    "dairy": Rule(
        triggers=frozenset({"milk", "cream", "butter", "cheese", "yogurt", "ghee", "buttermilk", "whey",
                            "casein", "mascarpone", "ricotta", "parmesan", "parmigiano", "mozzarella",
                            "cheddar", "feta", "brie", "gouda", "gruyere", "provolone", "romano",
                            "kefir", "custard", "gorgonzola", "camembert", "paneer", "halloumi",
                            "swiss", "colby", "monterey", "asiago", "fontina", "havarti", "emmental",
                            "velveeta", "creme", "crema", "half-and-half", "evaporated", "condensed",
                            "ice"} - {"ice", "evaporated", "condensed", "swiss", "monterey", "creme"}),
        phrases=("sour cream", "cream cheese", "half and half", "creme fraiche", "ice cream",
                 "whipped topping", "cottage cheese", "evaporated milk", "condensed milk",
                 "monterey jack", "swiss cheese"),
        exceptions=NONDAIRY_QUALIFIERS,
        except_phrases=("cream of tartar", "butter bean", "butter lettuce", "butter bean", "cream of coconut",
                        "butternut", "buttercup", "butterscotch", "creamed corn", "cream of rice",
                        "cream of wheat", "cocoa butter", "peanut butter", "almond butter", "nut butter",
                        "apple butter", "seed butter", "sunflower butter", "tahini", "milk chocolate"),
        possible=frozenset({"chocolate", "margarine", "caramel", "toffee", "biscuit", "nougat", "pudding",
                            "frosting", "waffle", "pancake", "brioche", "challah", "cake", "cookie"}),
    ),
    "egg": Rule(
        triggers=frozenset({"egg", "mayonnaise", "mayo", "meringue", "eggnog", "albumen"}),
        except_phrases=("eggplant", "egg noodle", "egg replacer", "egg substitute", "vegan egg"),
        exceptions=frozenset({"eggplant"}),
        possible=frozenset({"pasta", "noodle", "brioche", "challah", "aioli", "custard", "cake", "cookie",
                            "marshmallow", "hollandaise", "bearnaise", "waffle", "pancake", "wonton"}),
    ),
    "peanut": Rule(
        triggers=frozenset({"peanut", "peanuts", "groundnut"}),
        phrases=("peanut butter", "peanut oil"),
        possible=frozenset({"satay", "marzipan", "nougat", "praline"}),
    ),
    "tree_nut": Rule(
        triggers=frozenset({"almond", "walnut", "pecan", "cashew", "pistachio", "hazelnut", "macadamia",
                            "filbert", "praline", "marzipan", "pesto", "nutella", "amaretto", "frangelico"}),
        phrases=("pine nut", "brazil nut", "mixed nuts", "nut butter", "chestnut"),
        exceptions=frozenset({"water"}),
        except_phrases=("water chestnut", "coconut"),
        possible=frozenset({"granola", "nougat", "baklava", "frangipane"}),
    ),
    "gluten": Rule(
        triggers=frozenset({"flour", "wheat", "bread", "breadcrumb", "pasta", "spaghetti", "penne", "macaroni",
                            "noodle", "couscous", "barley", "rye", "semolina", "bulgur", "farro", "seitan",
                            "cracker", "tortilla", "bun", "biscuit", "pita", "croissant", "lasagna", "lasagne",
                            "linguine", "fettuccine", "orzo", "ziti", "rigatoni", "cake", "crouton", "panko",
                            "bagel", "baguette", "ciabatta", "pizza", "dough", "pretzel", "malt", "spelt",
                            "kamut", "triticale", "cornbread", "stuffing", "vermicelli", "tagliatelle",
                            "rotini", "rotelle", "farfalle", "shell", "ravioli", "tortellini", "gnocchi", "udon"}),
        exceptions=frozenset({"rice", "corn", "cornstarch", "almond", "coconut", "potato", "tapioca", "gluten-free",
                              "chickpea", "buckwheat", "oat", "cornmeal", "masa"}),
        except_phrases=("rice noodle", "rice flour", "corn tortilla", "almond flour", "coconut flour",
                        "gluten free", "gluten-free", "corn flour", "potato flour", "rice paper",
                        "buckwheat flour", "chickpea flour", "tapioca flour", "oat flour"),
        possible=frozenset({"soy sauce", "beer", "broth", "stock", "seasoning", "soup", "gravy", "sausage",
                            "oat", "oatmeal", "malt", "ale", "lager", "teriyaki", "worcestershire"}),
    ),
    "soy": Rule(
        triggers=frozenset({"soy", "soybean", "tofu", "tempeh", "edamame", "miso", "tamari", "shoyu"}),
        phrases=("soy sauce", "soy milk"),
        possible=frozenset({"margarine", "teriyaki", "worcestershire", "hoisin"}),
    ),
    "fish": Rule(
        triggers=frozenset({"fish", "salmon", "tuna", "cod", "tilapia", "trout", "halibut", "anchovy",
                            "sardine", "mackerel", "haddock", "bass", "snapper", "catfish", "swordfish",
                            "mahi", "herring", "flounder", "sole", "perch", "pollock", "pike", "carp"}),
        phrases=("fish sauce", "caesar dressing"),
        possible=frozenset({"worcestershire", "caesar", "surimi"}),
    ),
    "shellfish": Rule(
        triggers=frozenset({"shrimp", "crab", "lobster", "prawn", "crawfish", "crayfish", "clam", "oyster",
                            "scallop", "mussel", "squid", "calamari", "octopus", "shellfish", "krill", "langoustine",
                            "abalone", "conch", "escargot", "snail"}),
        except_phrases=("sea scallop-free",),
        possible=frozenset({"surimi"}),
    ),
    "meat": Rule(
        triggers=frozenset({"beef", "pork", "chicken", "turkey", "lamb", "veal", "bacon", "ham", "sausage",
                            "salami", "pepperoni", "prosciutto", "chorizo", "venison", "bison", "duck", "goose",
                            "rabbit", "steak", "brisket", "ribs", "meatball", "meat", "pancetta", "hotdog",
                            "mutton", "goat", "rib", "quail", "pheasant", "oxtail", "liver", "gelatin", "lard",
                            "tallow", "suet", "bologna", "mortadella", "jerky", "hot dog", "frankfurter"}),
        exceptions=frozenset({"vegetable", "mushroom", "vegan", "veggie", "tofu", "soy", "plant", "nut", "squash", "nutmeg"}),
        except_phrases=("chicken of the woods", "meatless", "bacon bits-free"),
        possible=frozenset({"broth", "stock", "bouillon", "consomme", "worcestershire", "gelatin", "marshmallow", "caesar"}),
    ),
    "pork": Rule(
        triggers=frozenset({"pork", "bacon", "ham", "prosciutto", "pancetta", "lard", "chorizo", "salami",
                            "pepperoni", "sausage", "bologna", "mortadella", "gammon", "hock"}),
        phrases=("hot dog",),
        possible=frozenset({"gelatin", "marshmallow"}),
    ),
    "alcohol": Rule(
        triggers=frozenset({"wine", "beer", "rum", "vodka", "whiskey", "whisky", "brandy", "bourbon", "liqueur",
                            "sherry", "gin", "tequila", "cognac", "port", "champagne", "ale", "lager", "stout",
                            "vermouth", "amaretto", "kahlua", "schnapps", "marsala", "mirin", "sake"}),
        exceptions=frozenset({"vinegar"}),
        except_phrases=("wine vinegar", "ginger ale", "ginger beer", "root beer", "apple cider vinegar",
                        "port salut", "cream sherry-free"),
    ),
    "honey": Rule(triggers=frozenset({"honey"})),
}

# diet = union of categories that are *excluded*
DIETS: dict[str, tuple[str, ...]] = {
    "vegetarian": ("meat", "fish", "shellfish"),
    "vegan": ("meat", "fish", "shellfish", "dairy", "egg", "honey"),
    "pescatarian": ("meat",),
    "gluten_free": ("gluten",),
    "dairy_free": ("dairy",),
    "nut_free": ("peanut", "tree_nut"),
    "egg_free": ("egg",),
    "peanut_free": ("peanut",),
    "shellfish_free": ("shellfish",),
    "soy_free": ("soy",),
    "pork_free": ("pork",),
    "alcohol_free": ("alcohol",),
}
# vegetarian bacon/ham style items are still meat: that's intended (definite hits only).
CATEGORY_ALIASES = {
    "milk": "dairy", "lactose": "dairy", "dairy-free": "dairy_free", "dairy free": "dairy_free",
    "eggs": "egg", "nuts": "tree_nut", "tree nut": "tree_nut", "tree nuts": "tree_nut", "treenut": "tree_nut",
    "peanuts": "peanut", "wheat": "gluten", "shellfish": "shellfish", "seafood": "fish",
    "vegetarian": "vegetarian", "vegan": "vegan", "gluten-free": "gluten_free", "gluten free": "gluten_free",
    "nut-free": "nut_free", "nut free": "nut_free", "meat": "meat", "pork": "pork", "alcohol": "alcohol",
    "soy": "soy", "fish": "fish", "egg": "egg", "dairy": "dairy", "gluten": "gluten", "honey": "honey",
    "peanut": "peanut", "tree_nut": "tree_nut",
}


def resolve_category(term: str) -> list[str]:
    """Map a user phrase ('dairy', 'vegan', 'gluten free') to the list of base categories to exclude."""
    t = (term or "").strip().lower().replace("_", " ")
    key = CATEGORY_ALIASES.get(t) or CATEGORY_ALIASES.get(t.replace(" ", "_")) or t.replace(" ", "_")
    if key in DIETS:
        return list(DIETS[key])
    if key in RULES:
        return [key]
    return []


def categories(canon: str) -> tuple[set[str], set[str]]:
    """Return (definite, possible) category sets for a canonical ingredient name."""
    text = f" {canon} "
    toks = set(canon.split())
    definite: set[str] = set()
    possible: set[str] = set()
    for cat, r in RULES.items():
        if any(p in text for p in r.except_phrases):
            continue
        hit = bool(toks & r.triggers) or any(f" {p} " in text for p in r.phrases)
        if hit:
            if toks & r.exceptions:   # a qualifier cancels the trigger ("coconut milk", "rice noodle")
                continue
            definite.add(cat)
        elif toks & r.possible or any(f" {p} " in text for p in (x for x in r.possible if " " in x)):
            possible.add(cat)
    # "gluten": flour of a non-wheat grain handled by exceptions; plain oats are "possible"
    return definite, possible - definite


def count_steps(instructions: str | None) -> int:
    """Step-counting rule (documented in README): one step per sentence; each non-empty line also
    starts a new step; fragments with fewer than 3 letters are ignored.

    Sentences split on '.', '!' or '?' followed by whitespace and a capital/digit, so decimals
    ("1.5 hours") do not split.
    """
    if not instructions:
        return 0
    n = 0
    for line in instructions.replace("\r", "\n").split("\n"):
        line = re.sub(r"\s+", " ", line.strip())
        if not line:
            continue
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", line):
            if len(re.findall(r"[A-Za-z]", sent)) >= 3:
                n += 1
    return n
