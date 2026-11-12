"""Download and unpack the Recipe Box dataset (three JSON files) into ../data."""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://eightportions.com/recipes_raw.zip"
DATA = Path(__file__).resolve().parents[2] / "data"
EXPECTED = ["recipes_raw_nosource_ar.json", "recipes_raw_nosource_epi.json", "recipes_raw_nosource_fn.json"]


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if all((DATA / f).exists() for f in EXPECTED):
        print("dataset already present in", DATA)
        return
    zpath = DATA / "recipes_raw.zip"
    print("downloading", URL)
    urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA)
    missing = [f for f in EXPECTED if not (DATA / f).exists()]
    if missing:
        sys.exit(f"missing after extraction: {missing}")
    print("ok:", ", ".join(EXPECTED))


if __name__ == "__main__":
    main()
