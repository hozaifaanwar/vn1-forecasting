"""Score the frozen Phase 1 forecasts against the real Phase 1 actuals.

Read-only with respect to the freeze: it refuses to run unless the frozen
files are byte-identical to tag phase1-freeze-v1, and unless the learner's
predictions (notes/phase1_predictions.md) were committed first. Scores the
LightGBM primary, the blend and MA4 secondaries, and the organizers' MA12
benchmark with the official metric, cross-checked against the organizers'
own formula. Writes tables to results/.

    python scripts/score_phase1.py
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

import pandas as pd  # noqa: E402

from src.features import classify_segment  # noqa: E402
from src.scoring import load_wide, ma12_benchmark, score_table, validate_actuals  # noqa: E402

FREEZE_TAG = "phase1-freeze-v1"
RAW = ROOT / "data/raw"
SUB_DIR = ROOT / "submissions/phase1"
OUT_DIR = ROOT / "results"
PREDICTIONS = "notes/phase1_predictions.md"
FROZEN = {"LightGBM (primary)": "phase1_forecast.csv", "Blend (secondary)": "phase1_secondary_blend.csv",
          "MA4 (secondary)": "phase1_secondary_ma4.csv"}
EXPECTED_DATES = pd.date_range("2023-10-09", "2024-01-01", freq="W-MON")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_actuals():
    candidates = [p for p in RAW.glob("*.csv")
                  if "phase" in p.name.lower() and "1" in p.name and "sales" in p.name.lower()
                  and "submission" not in p.name.lower()]
    if len(candidates) != 1:
        raise SystemExit(f"Expected exactly one Phase 1 sales file in data/raw/, found: {[p.name for p in candidates]}")
    return candidates[0]


def main():
    # Predictions must exist and be committed before the reveal.
    if not git("log", "--oneline", "--", PREDICTIONS) or git("status", "--porcelain", "--", PREDICTIONS):
        raise SystemExit(f"{PREDICTIONS} must be written and committed before scoring.")
    if "FILL IN" in (ROOT / PREDICTIONS).read_text(encoding="utf-8"):
        raise SystemExit(f"{PREDICTIONS} still has unfilled 'FILL IN' fields.")

    # Frozen files must be exactly what was tagged.
    frozen_paths = [f"submissions/phase1/{f}" for f in [*FROZEN.values(), "phase1_provenance.json"]]
    changed = git("diff", "--name-only", FREEZE_TAG, "--", *frozen_paths)
    if changed:
        raise SystemExit(f"Frozen files differ from {FREEZE_TAG}:\n{changed}")
    provenance = json.loads((SUB_DIR / "phase1_provenance.json").read_text())
    for name in FROZEN.values():
        assert sha256(SUB_DIR / name) == provenance["output_sha256"][name], f"{name} hash mismatch"

    history = load_wide(RAW / "Phase_0_Sales.csv")
    actuals_path = find_actuals()
    actuals = load_wide(actuals_path)
    validate_actuals(actuals, history.index, EXPECTED_DATES)

    submissions = {name: load_wide(SUB_DIR / f) for name, f in FROZEN.items()}
    submissions["MA12 (official benchmark)"] = ma12_benchmark(history, actuals.columns)

    # Segments from Phase 0 history only (same thresholds as Session 4).
    segments = classify_segment((history == 0).mean(axis=1))

    scores = score_table(submissions, actuals, segments)
    weekly = pd.DataFrame({"actual": actuals.sum(), **{n: s.sum() for n, s in submissions.items()}})
    weekly.index = weekly.index.strftime("%Y-%m-%d")

    OUT_DIR.mkdir(exist_ok=True)
    scores.to_csv(OUT_DIR / "phase1_scores.csv", index=False)
    weekly.to_csv(OUT_DIR / "phase1_weekly_totals.csv", index_label="week")
    meta = {
        "scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scored_at_commit": git("rev-parse", "HEAD"),
        "freeze_tag": FREEZE_TAG,
        "predictions_commit": git("log", "-1", "--format=%H", "--", PREDICTIONS),
        "actuals_file": actuals_path.name,
        "actuals_sha256": sha256(actuals_path),
        "frozen_sha256": {f: provenance["output_sha256"][f] for f in FROZEN.values()},
    }
    (OUT_DIR / "phase1_scoring.json").write_text(json.dumps(meta, indent=2))

    overall = scores[scores["slice"] == "all"].sort_values("score")
    pd.set_option("display.width", 200)
    fmt = {"score": "{:.4f}".format, "mae_pct": "{:.4f}".format, "bias_pct": "{:+.4f}".format,
           "total_actual": "{:,.0f}".format, "total_predicted": "{:,.0f}".format, "demand_share": "{:.3f}".format}
    print(f"actuals: {actuals_path.name}  ({actuals.shape[0]:,} series x {actuals.shape[1]} weeks)\n")
    print(overall.drop(columns=["slice", "demand_share"]).to_string(index=False, formatters=fmt))
    print("\nBy segment (Phase 0 zero-rate segments):")
    by_seg = scores[scores["slice"] != "all"].sort_values(["slice", "score"])
    print(by_seg.drop(columns=["total_actual", "total_predicted"]).to_string(index=False, formatters=fmt))
    print("\nWeekly totals:")
    print(weekly.round(0).astype(int).to_string())


if __name__ == "__main__":
    main()
