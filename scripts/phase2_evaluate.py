"""Phase 2 candidate evaluation and mechanical selection (DECISIONS.md D014).

Candidates: (1) MA12, (2) MA4, (3) frozen LightGBM, (4) incumbent = 0.5 frozen
LightGBM + 0.5 MA4, (5) LightGBM-v2 (spec from results/phase2_ablation/),
(6) January-adjusted MA4, (7) 0.5 LightGBM-v2 + 0.5 January-adjusted MA4.

Benchmark scores (1)-(4) are re-computed here with the same frozen procedure
and checked against the committed calibration. The D014 rule is applied in
code; this script was committed before it was run.

    python scripts/phase2_evaluate.py
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

from src.features import classify_segment  # noqa: E402
from src.metrics import vn1_score_breakdown  # noqa: E402
from src.pipeline import frozen_forecast  # noqa: E402
from src.pipeline_v2 import Data, jan_adjusted_ma4, jan_adjusted_multiplier, v2_forecast  # noqa: E402

H = 13
OUT = ROOT / "results" / "phase2_eval"
OUT.mkdir(parents=True, exist_ok=True)

# ---- D014 constants (do not edit after results exist) ----
T1, T2, T3, P1_TOL, BIAS_FLAG = 0.010, 0.60, 0.03, 0.10, 0.10
CV_ORIGINS = list(range(92, 169, 4))
BLOCK = {o: ("A" if o <= 116 else "B" if o <= 140 else "C") for o in CV_ORIGINS}
JAN = {"jan_2022": list(range(76, 81)), "jan_2023": list(range(128, 133))}
REPLAY = 169
MA_FAMILY = {"MA12", "MA4", "adj_MA4"}
LGBM_FAMILY = {"lgbm_frozen", "lgbm_v2", "blend_v2_adj"}
CHALLENGERS_RESERVED = {"lgbm_v2", "blend_v2_adj"}  # must win block C under T1
INCUMBENT = "incumbent"
CANDIDATES = ["MA12", "MA4", "lgbm_frozen", "incumbent", "lgbm_v2", "adj_MA4", "blend_v2_adj"]

df = pd.read_parquet(ROOT / "data/processed/vn1_long_p0p1.parquet")
data = Data(df)
S = data.sales
spec = json.loads((ROOT / "results/phase2_ablation/lightgbm_v2_spec.json").read_text())
V2 = dict(groups=tuple(spec["groups"]), gate_weeks=spec["gate_weeks"])
print("LightGBM-v2 spec:", V2, flush=True)


def forecasts(o):
    """All candidate forecasts at origin o as (n_series, H) arrays (series in sorted KEY order)."""
    n = S.shape[0]
    out = {"MA4": np.repeat(S[:, o - 3: o + 1].mean(1)[:, None], H, 1),
           "MA12": np.repeat(S[:, o - 11: o + 1].mean(1)[:, None], H, 1),
           "adj_MA4": jan_adjusted_ma4(S, o, H)}
    if o < 92:  # Jan 2022: no LightGBM model can be trained (D014)
        return out
    to_mat = lambda v: v.reshape(H, n).T  # long frames are horizon-major  # noqa: E731
    fc, _, _ = frozen_forecast(df, o)
    assert np.allclose(to_mat(fc["ma4"].to_numpy()), out["MA4"])
    out["lgbm_frozen"] = to_mat(fc["lgbm"].to_numpy())
    out["incumbent"] = to_mat(fc["blend"].to_numpy())
    fv, _, _ = v2_forecast(data, o, **V2)
    out["lgbm_v2"] = to_mat(fv["lgbm"].to_numpy())
    out["blend_v2_adj"] = 0.5 * out["lgbm_v2"] + 0.5 * out["adj_MA4"]
    return out


def parts(actual, pred):
    b = vn1_score_breakdown(actual, pred)
    return {"score": b["score"], "mae_part": b["abs_error_sum"] / b["total_actual"],
            "bias_part": b["bias_term"] / b["total_actual"], "bias_pct": b["signed_bias"] / b["total_actual"]}


rows, seg_rows = [], []
comparisons = [("all_season", o) for o in CV_ORIGINS] + [(c, o) for c, os in JAN.items() for o in os] + \
              [("phase1_replay", REPLAY)]
cache = {}
for comp, o in comparisons:
    if o not in cache:
        cache[o] = forecasts(o)
        print(f"origin {o} done", flush=True)
    actual = S[:, o + 1: o + 1 + H]
    seg = classify_segment(pd.Series((S[:, : o + 1] == 0).mean(1))).to_numpy()
    zero_series = actual.sum(1) == 0
    for m, pred in cache[o].items():
        rows.append({"comparison": comp, "origin_idx": o, "block": BLOCK.get(o), "model": m,
                     **parts(actual, pred), "volume_on_zero_series": float(pred[zero_series].sum()),
                     "forecast_total": float(pred.sum()), "actual_total": float(actual.sum()),
                     "n_years_jan_ratio": jan_adjusted_multiplier(S, o)[1]})
        if comp in ("all_season", "phase1_replay"):
            for sg in ("active", "intermittent", "sparse"):
                k = seg == sg
                seg_rows.append({"comparison": comp, "origin_idx": o, "model": m, "segment": sg,
                                 **parts(actual[k], pred[k])})
res = pd.DataFrame(rows)
res.to_csv(OUT / "candidate_scores.csv", index=False)
pd.DataFrame(seg_rows).to_csv(OUT / "segment_scores.csv", index=False)

# Benchmarks must match the committed calibration
cal = pd.read_csv(ROOT / "results/phase2_calibration/benchmark_scores.csv").replace({"all_season_cv": "all_season"})
chk = res.merge(cal, on=["comparison", "origin_idx", "model"], suffixes=("", "_cal"))
assert len(chk) >= 100 and np.allclose(chk["score"], chk["score_cal"], atol=1e-9), "benchmarks != calibration"
print(f"benchmarks match calibration on {len(chk)} rows", flush=True)

# ---- D014 rule, applied mechanically ----
pv = res.pivot_table(index=["comparison", "origin_idx"], columns="model", values="score")
cv = pv.loc["all_season"]
inc = cv[INCUMBENT]
j22, j23 = pv.loc["jan_2022"].mean(), pv.loc["jan_2023"].mean()
rep = res.query("comparison == 'phase1_replay'").set_index("model")
decision = {}
for m in CANDIDATES:
    if m == INCUMBENT:
        continue
    d = cv[m] - inc
    blocks = d.groupby(pd.Series(BLOCK)).mean()
    better_blocks = sorted(blocks[blocks < 0].index)
    t1 = bool(-d.mean() >= T1 and len(better_blocks) >= 2 and
              (m not in CHALLENGERS_RESERVED or "C" in better_blocks))
    t2 = bool((d < 0).mean() >= T2)
    if m in MA_FAMILY:
        d22 = j22[m] - j22["MA12"]
        d23 = j23[m] - j23[INCUMBENT]
        t3 = bool(d22 <= T3 and d23 <= T3 and (d22 + d23) / 2 < 0)
        jan_detail = {"jan_2022_vs_MA12": d22, "jan_2023_vs_incumbent": d23}
    else:
        d23 = j23[m] - j23[INCUMBENT]
        t3 = bool(d23 <= T3)
        jan_detail = {"jan_2023_vs_incumbent": d23}
    p1 = rep.loc[m, "score"] - rep.loc[INCUMBENT, "score"]
    gate = bool(p1 <= P1_TOL)
    decision[m] = {"all_season_mean": cv[m].mean(), "diff_vs_incumbent": d.mean(),
                   "block_diffs": blocks.round(4).to_dict(), "wins": int((d < 0).sum()), "n": len(d),
                   **jan_detail, "phase1_replay_diff": p1,
                   "phase1_bias_flag": bool(abs(rep.loc[m, "bias_pct"]) > BIAS_FLAG),
                   "T1": t1, "T2": t2, "T3": t3, "phase1_gate": gate, "passes_all": t1 and t2 and t3 and gate}
passing = [m for m, v in decision.items() if v["passes_all"]]
primary = min(passing, key=lambda m: decision[m]["all_season_mean"]) if passing else INCUMBENT
summary = {"primary": primary, "passing": passing, "incumbent_all_season_mean": inc.mean(),
           "thresholds": {"T1": T1, "T2": T2, "T3": T3, "phase1_tol": P1_TOL}, "lightgbm_v2": V2,
           "candidates": decision}
(OUT / "decision.json").write_text(json.dumps(summary, indent=2, default=float))

means = res.groupby(["comparison", "model"])[["score", "mae_part", "bias_part", "bias_pct"]].mean().round(4)
print("\nMean scores:\n", means.to_string())
print("\nDecision:\n", json.dumps(summary, indent=2, default=lambda x: round(float(x), 4)))
