"""Build data/processed/vn1_long.parquet from the raw wide CSVs.

Reproducible entry point for the Session 2 pipeline — doesn't depend on
manually running notebook cells in order. Run from anywhere; it resolves
the repo root itself.

    python scripts/build_dataset.py                # Phase 0 only -> vn1_long.parquet
    python scripts/build_dataset.py --through phase1  # Phase 0 + 1 -> vn1_long_p0p1.parquet

The Phase 0 file stays as it was: the frozen Phase 1 pipeline and
scripts/verify_phase1_freeze.py read it. Phase 0 + 1 goes to a separate file.
"""

import argparse
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

RAW = ROOT / "data/raw"
PHASES = {
    "phase0": ([RAW / "Phase_0_Sales.csv"], [RAW / "Phase_0_Price.csv"],
               ROOT / "data/processed/vn1_long.parquet"),
    "phase1": ([RAW / "Phase_0_Sales.csv", RAW / "Phase 1 - Sales.csv"],
               [RAW / "Phase_0_Price.csv", RAW / "Phase 1 - Price.csv"],
               ROOT / "data/processed/vn1_long_p0p1.parquet"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=PHASES, default="phase0")
    sales, price, out = PHASES[parser.parse_args().through]

    df = load_long(sales, price)
    cdf = clean(df)
    validate_long(cdf)

    out.parent.mkdir(parents=True, exist_ok=True)
    cdf.to_parquet(out, index=False)

    print(f"saved  : {out.relative_to(ROOT)}")
    print(f"shape  : {cdf.shape}")
    print(f"dates  : {cdf['date'].min().date()} -> {cdf['date'].max().date()} ({cdf['date'].nunique()} weeks)")


if __name__ == "__main__":
    main()
