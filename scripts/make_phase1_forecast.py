"""Generate and freeze the Phase 1 forecast (DECISIONS.md D012).

Trains the frozen pipeline (src/pipeline.py) on all 170 Phase 0 weeks and
forecasts the 13 Phase 1 weeks (2023-10-09 -> 2024-01-01). Writes the
submission plus a provenance file recording exactly what produced it.
Refuses to run with uncommitted changes to src/ or scripts/, so the
recorded commit hash really is the code that generated the forecast.

    python scripts/make_phase1_forecast.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
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

import lightgbm as lgb  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import KEY, validate_long  # noqa: E402
from src.pipeline import FROZEN_CONFIG, frozen_forecast, to_submission  # noqa: E402

DATA = ROOT / "data/processed/vn1_long.parquet"
RAW_SALES = ROOT / "data/raw/Phase_0_Sales.csv"
OUT_DIR = ROOT / "submissions/phase1"
EXPECTED_DATES = pd.date_range("2023-10-09", "2024-01-01", freq="W-MON")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dirty = git("status", "--porcelain", "--", "src", "scripts")
    if dirty:
        raise SystemExit(f"Uncommitted changes in src/ or scripts/ — commit first:\n{dirty}")
    commit = git("rev-parse", "HEAD")

    df = pd.read_parquet(DATA)
    validate_long(df)
    last = df["date"].nunique() - 1
    assert df["date"].max() == pd.Timestamp("2023-10-02"), "expected Phase 0 to end 2023-10-02"

    forecast, model, info = frozen_forecast(df, last)
    assert forecast["target"].isna().all(), "Phase 1 rows must have no targets"
    assert sorted(forecast["date"].unique()) == list(EXPECTED_DATES)

    key_order = pd.read_csv(RAW_SALES, usecols=KEY)
    submission = to_submission(forecast, key_order)
    assert len(submission) == 15_053
    assert list(submission.columns[3:]) == [d.strftime("%Y-%m-%d") for d in EXPECTED_DATES]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = OUT_DIR / "phase1_forecast.csv"
    submission.to_csv(sub_path, index=False)

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": commit,
        "training_cutoff": "2023-10-02",
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in EXPECTED_DATES],
        "method": "0.5 * LightGBM (direct multi-horizon) + 0.5 * MA4, clipped >= 0",
        "frozen_config": FROZEN_CONFIG,
        **info,
        "lightgbm_version": lgb.__version__,
        "pandas_version": pd.__version__,
        "input_sha256": {"Phase_0_Sales.csv": sha256(RAW_SALES),
                         "Phase_0_Price.csv": sha256(ROOT / "data/raw/Phase_0_Price.csv")},
        "submission_sha256": sha256(sub_path),
        "forecast_totals": {c: float(forecast[c].sum()) for c in ("lgbm", "ma4", "forecast")},
    }
    (OUT_DIR / "phase1_provenance.json").write_text(json.dumps(provenance, indent=2, default=str))

    print(f"saved      : {sub_path.relative_to(ROOT)} ({submission.shape})")
    print(f"commit     : {commit}")
    print(f"best_iter  : {info['best_iteration']}")
    print(f"totals     : {provenance['forecast_totals']}")


if __name__ == "__main__":
    main()
