"""pytest wrapper: every deterministic evaluation case must pass (requires storage/recipes.db; see README)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

if not (ROOT / "storage" / "recipes.db").exists():
    pytest.skip("build the database first: python scripts/build_db.py", allow_module_level=True)

from cases import CASES  # noqa: E402


@pytest.mark.parametrize("cid,desc,fn", CASES, ids=[c[0] for c in CASES])
def test_case(cid, desc, fn):
    res = fn()
    failed = [name for name, ok in res["checks"] if not ok]
    assert res["passed"], f"{cid} {desc}: failed checks {failed}\n{res['excerpt']}"
