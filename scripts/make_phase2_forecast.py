"""Generate and freeze the Phase 2 forecast (DECISIONS.md D014).

Forecast origin = the last Phase 1 week (2024-01-01, index 182); forecast
weeks 2024-01-08 -> 2024-04-01. The primary is whatever
results/phase2_eval/decision.json selected under D014's rule; every other
candidate is frozen alongside it as a declared secondary, so all can be
scored later. Refuses to run with uncommitted changes to src/ or scripts/,
and refuses to overwrite an existing frozen forecast.

    python scripts/make_phase2_forecast.py
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "CONTEXT.md").is_file():
    if ROOT == ROOT.parent:
        raise FileNotFoundError("Could not find CONTEXT.md in any parent directory")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import KEY, validate_long  # noqa: E402
from src.pipeline import frozen_forecast, to_submission  # noqa: E402
from src.pipeline_v2 import Data, jan_adjusted_ma4, jan_adjusted_multiplier, v2_forecast  # noqa: E402

DATA = ROOT / "data/processed/vn1_long_p0p1.parquet"
OUT_DIR = ROOT / "submissions/phase2"
EXPECTED_DATES = pd.date_range("2024-01-08", "2024-04-01", freq="W-MON")
H = 13


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    dirty = git("status", "--porcelain", "--", "src", "scripts", "results/phase2_eval", "results/phase2_ablation")
    if dirty:
        raise SystemExit(f"Uncommitted changes — commit first:\n{dirty}")
    if (OUT_DIR / "phase2_forecast.csv").exists():
        raise SystemExit("submissions/phase2 already holds a frozen forecast — not overwriting it")
    commit = git("rev-parse", "HEAD")
    decision = json.loads((ROOT / "results/phase2_eval/decision.json").read_text())
    spec = decision["lightgbm_v2"]

    df = pd.read_parquet(DATA)
    validate_long(df)
    data = Data(df)
    o = df["date"].nunique() - 1
    assert df["date"].max() == pd.Timestamp("2024-01-01") and o == 182

    S = data.sales
    n = S.shape[0]
    fc, _, info_frozen = frozen_forecast(df, o)
    fv, _, info_v2 = v2_forecast(data, o, groups=tuple(spec["groups"]), gate_weeks=spec["gate_weeks"])
    assert sorted(fc["date"].unique()) == list(EXPECTED_DATES) and fc["target"].isna().all()
    assert (fc[KEY].to_numpy() == fv[KEY].to_numpy()).all()
    flat = lambda m: m.T.reshape(-1)  # (n, H) -> horizon-major long order  # noqa: E731
    adj = flat(jan_adjusted_ma4(S, o, H))
    cols = {
        "MA12": flat(np.repeat(S[:, o - 11: o + 1].mean(1)[:, None], H, 1)),
        "MA4": fc["ma4"].to_numpy(),
        "lgbm_frozen": fc["lgbm"].to_numpy(),
        "incumbent": fc["blend"].to_numpy(),
        "lgbm_v2": fv["lgbm"].to_numpy(),
        "adj_MA4": adj,
        "blend_v2_adj": 0.5 * fv["lgbm"].to_numpy() + 0.5 * adj,
    }
    assert np.allclose(cols["MA4"], flat(np.repeat(S[:, o - 3: o + 1].mean(1)[:, None], H, 1)))
    long = fc[KEY + ["date"]].copy()
    for k, v in cols.items():
        assert np.isfinite(v).all() and (v >= 0).all(), k
        long[k] = v

    key_order = pd.read_csv(ROOT / "data/raw/Phase 1 - Sales.csv", usecols=KEY)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    primary = decision["primary"]
    files = {primary: "phase2_forecast.csv"}
    files.update({k: f"phase2_secondary_{k}.csv" for k in cols if k != primary})
    for column, name in files.items():
        sub = to_submission(long, key_order, column=column)
        assert len(sub) == 15_053
        assert list(sub.columns[3:]) == [d.strftime("%Y-%m-%d") for d in EXPECTED_DATES]
        sub.to_csv(OUT_DIR / name, index=False)

    mult, n_years = jan_adjusted_multiplier(S, o)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": commit,
        "training_cutoff": "2024-01-01",
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in EXPECTED_DATES],
        "primary": f"{primary} -> phase2_forecast.csv (DECISIONS.md D014, results/phase2_eval/decision.json)",
        "secondaries": {v: k for k, v in files.items() if k != primary},
        "lightgbm_v2_spec": spec,
        "frozen_info": {k: info_frozen[k] for k in ("fit_origins", "val_origins", "best_iteration")},
        "v2_info": {k: info_v2[k] for k in ("fit_origins", "val_origins", "best_iteration", "feature_cols")},
        "jan_multiplier": {"n_years": n_years, "mean": float(mult.mean()), "median": float(np.median(mult))},
        "ma4_level_total": float(S[:, o - 3: o + 1].mean(1).sum()),
        "forecast_weekly_mean_totals": {k: float(v.sum() / H) for k, v in cols.items()},
        "lightgbm_version": lgb.__version__,
        "pandas_version": pd.__version__,
        "input_sha256": {"vn1_long_p0p1.parquet": sha256(DATA)},
        "output_sha256": {name: sha256(OUT_DIR / name) for name in files.values()},
    }
    (OUT_DIR / "phase2_provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    print(json.dumps({k: provenance[k] for k in ("primary", "forecast_weekly_mean_totals",
                                                 "ma4_level_total", "jan_multiplier")}, indent=2))


if __name__ == "__main__":
    main()
