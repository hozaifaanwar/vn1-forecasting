"""Verify the frozen Phase 1 forecast and write a separate verification addendum.

Never modifies the frozen files (phase1_forecast.csv, the two secondaries,
phase1_provenance.json). It checks that their hashes still match the
provenance, that the raw inputs are unchanged, that the frozen pipeline
regenerates every forecast value from the current processed data, and
that no Phase 1 actuals are present in data/raw/. Results go to
submissions/phase1/phase1_verification_addendum.json, which records what
the original provenance lacks: the processed-parquet and environment-lock
hashes.

    python scripts/verify_phase1_freeze.py
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import KEY  # noqa: E402
from src.pipeline import frozen_forecast  # noqa: E402

FREEZE_TAG = "phase1-freeze-v1"
OUT_DIR = ROOT / "submissions/phase1"
DATA = ROOT / "data/processed/vn1_long.parquet"
RAW = ROOT / "data/raw"
FILES = {"forecast": "phase1_forecast.csv", "blend": "phase1_secondary_blend.csv",
         "ma4": "phase1_secondary_ma4.csv"}


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dirty = git("status", "--porcelain", "--", "src", "scripts", "environment.lock.yml")
    if dirty:
        raise SystemExit(f"Uncommitted changes — commit first so verified_at_commit is accurate:\n{dirty}")

    provenance = json.loads((OUT_DIR / "phase1_provenance.json").read_text())
    freeze_commit = git("rev-parse", f"{FREEZE_TAG}^{{commit}}")

    # 1. Frozen files unchanged since the freeze commit, and hashes match provenance.
    frozen = [*FILES.values(), "phase1_provenance.json"]
    changed = git("diff", "--name-only", freeze_commit, "--", *[f"submissions/phase1/{f}" for f in frozen])
    assert not changed, f"frozen files changed since {FREEZE_TAG}: {changed}"
    for name in FILES.values():
        assert sha256(OUT_DIR / name) == provenance["output_sha256"][name], f"{name} hash mismatch"

    # 2. Raw inputs unchanged.
    for name, expected in provenance["input_sha256"].items():
        assert sha256(RAW / name) == expected, f"{name} differs from the file used at generation"

    # 3. No Phase 1 actuals present: nothing in data/raw/ covers Phase 1 dates
    #    except the organizers' random placeholder submission.
    placeholder = "Submission Phase 1 - Random-3.csv"
    with_phase1_dates = sorted(
        p.name for p in RAW.glob("*.csv")
        if "2023-10-09" in p.open(encoding="utf-8").readline() and p.name != placeholder
    )
    assert not with_phase1_dates, f"possible Phase 1 actuals present: {with_phase1_dates}"

    # 4. Regenerate from the current processed data and compare every value.
    df = pd.read_parquet(DATA)
    forecast, _, info = frozen_forecast(df, df["date"].nunique() - 1)
    assert info["best_iteration"] == provenance["best_iteration"]
    max_abs_diff = {}
    for column, name in FILES.items():
        frozen_wide = pd.read_csv(OUT_DIR / name).set_index(KEY)
        regen = forecast.pivot(index=KEY, columns="date", values=column).reindex(frozen_wide.index)
        diff = np.abs(regen.to_numpy() - frozen_wide.to_numpy()).max()
        assert diff < 1e-9, f"{name}: regenerated forecast differs by {diff}"
        max_abs_diff[name] = float(diff)

    addendum = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verified_at_commit": git("rev-parse", "HEAD"),
        "freeze_tag": FREEZE_TAG,
        "freeze_commit": freeze_commit,
        "generation_commit": provenance["code_commit"],
        "forecast_sha256_match_provenance": {n: provenance["output_sha256"][n] for n in FILES.values()},
        "raw_inputs_match_provenance": provenance["input_sha256"],
        "processed_parquet_sha256": sha256(DATA),
        "environment_lock_sha256": sha256(ROOT / "environment.lock.yml"),
        "regenerated_max_abs_diff": max_abs_diff,
        "regenerated_best_iteration": info["best_iteration"],
        "lightgbm_version": lgb.__version__,
        "phase1_actuals_present_in_data_raw": False,
        "note": "Separate from phase1_provenance.json, which stays exactly as frozen. Records the "
                "processed-data and environment hashes the original provenance lacked.",
    }
    (OUT_DIR / "phase1_verification_addendum.json").write_text(json.dumps(addendum, indent=2))
    print(json.dumps(addendum, indent=2))


if __name__ == "__main__":
    main()
