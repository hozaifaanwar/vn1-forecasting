"""Build data/processed/vn1_long.parquet from the raw wide CSVs.

Reproducible entry point for the Session 2 pipeline — doesn't depend on
manually running notebook cells in order. Run from anywhere; it resolves
the repo root itself.

    python scripts/build_dataset.py
"""

import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    current = start
    while not (current / "CONTEXT.md").is_file():
        if current == current.parent:
            raise FileNotFoundError("Could not find CONTEXT.md in any parent directory")
        current = current.parent
    return current


ROOT = find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(ROOT))

from src.data import load_long, clean, validate_long  # noqa: E402

SALES = ROOT / "data/raw/Phase_0_Sales.csv"
PRICE = ROOT / "data/raw/Phase_0_Price.csv"
OUT = ROOT / "data/processed/vn1_long.parquet"


def main():
    df = load_long(SALES, PRICE)
    cdf = clean(df)
    validate_long(cdf)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cdf.to_parquet(OUT, index=False)

    print(f"saved  : {OUT.relative_to(ROOT)}")
    print(f"shape  : {cdf.shape}")
    print(f"dates  : {cdf['date'].min().date()} -> {cdf['date'].max().date()}")


if __name__ == "__main__":
    main()
