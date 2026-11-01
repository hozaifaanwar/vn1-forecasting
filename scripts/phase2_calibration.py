"""Phase 2 benchmark-only calibration (inputs to D014 thresholds).

Scores ONLY the benchmarks — MA4, MA12 and the incumbent (frozen-style 50/50
LightGBM + MA4 blend) — on every comparison D014 will use, and records the
exact folds. No challenger is built or scored here. Outputs go to
results/phase2_calibration/ and are committed before D014 and before any
challenger runs (notes/d014_inputs.md, design choice 7).

    python scripts/phase2_calibration.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "CONTEXT.md").is_file():
    if ROOT == ROOT.parent:
        raise FileNotFoundError("Could not find CONTEXT.md in any parent directory")
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import KEY  # noqa: E402
from src.features import build_direct_horizon_matrix, rolling_origin_folds  # noqa: E402
from src.metrics import vn1_score_breakdown  # noqa: E402
from src.models.lightgbm_model import train_lgbm  # noqa: E402
from src.pipeline import FROZEN_CONFIG, HORIZONS, SEGMENT_DTYPE, frozen_forecast  # noqa: E402

OUT = ROOT / "results" / "phase2_calibration"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(ROOT / "data/processed/vn1_long_p0p1.parquet")
sales = df.pivot(index=KEY, columns="date", values="sales")
dates = sales.columns
vals = sales.to_numpy(dtype=float)
assert vals.shape == (15053, 183)
H = max(HORIZONS)


def ma(o, k):
    return vals[:, o - k + 1: o + 1].mean(axis=1)


def parts(actual, pred):
    b = vn1_score_breakdown(actual, pred)
    return {"score": b["score"], "mae_part": b["abs_error_sum"] / b["total_actual"],
            "bias_part": b["bias_term"] / b["total_actual"],
            "bias_pct": b["signed_bias"] / b["total_actual"]}


rows = []


def add(comparison, o, model, actual, pred, fold=None):
    rows.append({"comparison": comparison, "fold": fold, "origin_idx": o, "origin": str(dates[o].date()),
                 "model": model, **parts(actual, pred)})


def ma_rows(comparison, origins, fold=None):
    for o in origins:
        actual = vals[:, o + 1: o + 1 + H]
        for name, k in [("MA4", 4), ("MA12", 12)]:
            add(comparison, o, name, actual, np.repeat(ma(o, k)[:, None], H, axis=1), fold)


# 1. All-season rolling CV: exact folds, MA benchmarks + incumbent (per-fold LightGBM, as Session 5/6)
pool = list(range(52, len(dates) - H, 4))
folds = rolling_origin_folds(pool, HORIZONS, n_folds=3)
fold_record = []
for fi, (fit_o, val_o) in enumerate(folds):
    fold_record.append({"fold": fi, "fit_origins": fit_o, "val_origins": val_o,
                        "val_dates": [str(dates[o].date()) for o in val_o]})
    ma_rows("all_season_cv", val_o, fi)
    fm = build_direct_horizon_matrix(df, fit_o, HORIZONS)
    vm = build_direct_horizon_matrix(df, val_o, HORIZONS)
    for m in (fm, vm):
        m["segment"] = m["segment"].astype(SEGMENT_DTYPE)
    feats = [c for c in fm.columns if c not in (*KEY, "origin_date", "target_date", "target")]
    _, lgbm = train_lgbm(fm, vm, feats, categorical_feature=["segment"], origin_col="origin_date",
                         random_state=FROZEN_CONFIG["random_state"], **FROZEN_CONFIG["lgb_params"])
    blend = 0.5 * lgbm + 0.5 * vm["roll_mean_4"].to_numpy()
    for o in val_o:
        mask = (vm["origin_date"] == dates[o]).to_numpy()
        add("all_season_cv", o, "incumbent", vm.loc[mask, "target"].to_numpy(), blend[mask], fi)
    print(f"fold {fi} done", flush=True)

# 2. Non-overlapping origins (13 weeks apart) for benchmark-vs-benchmark variation
non_overlap = list(range(52, len(dates) - H, H))
ma_rows("non_overlapping", non_overlap)

# 3. January season-matched: MA family everywhere; incumbent only where it can be trained
jan = {"jan_2022": list(range(76, 81)), "jan_2023": list(range(128, 133))}
for comp, origins in jan.items():
    ma_rows(comp, origins)
for o in jan["jan_2023"]:
    fc, _, _ = frozen_forecast(df, o)
    add("jan_2023", o, "incumbent", fc["target"].to_numpy(), fc["blend"].to_numpy())
    print(f"jan_2023 origin {o} done", flush=True)

# 4. Phase 1 replay (origin 169): must reproduce the scored Phase 1 result
ma_rows("phase1_replay", [169])
fc, _, _ = frozen_forecast(df, 169)
add("phase1_replay", 169, "incumbent", fc["target"].to_numpy(), fc["blend"].to_numpy())

res = pd.DataFrame(rows)
res.to_csv(OUT / "benchmark_scores.csv", index=False)
inc_p1 = res.query("comparison == 'phase1_replay' and model == 'incumbent'")["score"].iloc[0]
ma4_p1 = res.query("comparison == 'phase1_replay' and model == 'MA4'")["score"].iloc[0]
assert abs(inc_p1 - 0.5704) < 5e-4, f"Phase 1 replay incumbent {inc_p1:.4f} != scored 0.5704"
assert abs(ma4_p1 - 0.6411) < 5e-4, f"Phase 1 replay MA4 {ma4_p1:.4f} != scored 0.6411"

# Summaries
summary = res.groupby(["comparison", "model"])[["score", "mae_part", "bias_part"]].mean().round(4)
piv = res.pivot_table(index=["comparison", "origin_idx"], columns="model", values="score")
var = {}
no = piv.loc["non_overlapping"]
d = no["MA4"] - no["MA12"]
var["non_overlapping_MA4_minus_MA12"] = {"n": int(d.size), "mean": d.mean(), "sd": d.std(ddof=1),
                                         "mean_abs": d.abs().mean()}
cv = res.query("comparison == 'all_season_cv'").pivot_table(index=["fold", "origin_idx"], columns="model", values="score")
for a, b in [("incumbent", "MA4"), ("MA4", "MA12")]:
    dd = cv[a] - cv[b]
    var[f"cv_{a}_minus_{b}"] = {"n": int(dd.size), "mean": dd.mean(), "sd_origins": dd.std(ddof=1),
                                "fold_means": dd.groupby(level="fold").mean().round(4).to_dict(),
                                "share_origins_a_wins": float((dd < 0).mean())}

(OUT / "folds.json").write_text(json.dumps({"origin_pool": pool, "folds": fold_record,
                                            "non_overlapping_origins": non_overlap, "january": jan,
                                            "phase1_replay": [169]}, indent=2))
(OUT / "variation.json").write_text(json.dumps(var, indent=2, default=float))
print("\nMean scores by comparison and model:\n", summary)
print("\nVariation:\n", json.dumps(var, indent=2, default=lambda x: round(float(x), 4)))
print("\nPhase 1 replay reproduced: incumbent", round(inc_p1, 4), "MA4", round(ma4_p1, 4))
