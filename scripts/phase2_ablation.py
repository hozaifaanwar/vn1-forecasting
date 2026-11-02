"""LightGBM-v2 forward feature-group ablation (DECISIONS.md D014).

Ablation origins 92-128 only (targets end by week 141; the reserved block's
first target is week 145). Each model is the frozen procedure run at its own
origin (`src/pipeline_v2.py::v2_forecast`): early stopping on its own purged
internal origins, never on a scored target.

Fixed order: annual, activity, aggregates, trend, identity, then the dormancy
gate (N = 13, post-processing). A group is kept if, against the current set,
it improves the mean-of-origin VN1 score of LightGBM by >= 0.005 AND wins
>= 6 of the 10 origins. Writes results/phase2_ablation/.

    python scripts/phase2_ablation.py
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

from src.features_v2 import GROUPS, dormant_mask, v2_extra_features  # noqa: E402
from src.metrics import vn1_score_breakdown  # noqa: E402
from src.pipeline_v2 import Data, v2_forecast  # noqa: E402

ABLATION_ORIGINS = list(range(92, 129, 4))
KEEP_MIN_IMPROVEMENT = 0.005
KEEP_MIN_WINS = 6
GATE_WEEKS = 13
H = 13
OUT = ROOT / "results" / "phase2_ablation"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(ROOT / "data/processed/vn1_long_p0p1.parquet")
data = Data(df)
assert max(ABLATION_ORIGINS) + H < 145, "ablation targets would reach the reserved block"

# Leakage check on the real data, every group, at the latest ablation origin
o = max(ABLATION_ORIGINS)
corrupted = data.sales.copy()
corrupted[:, o + 1:] = 1e9
pd.testing.assert_frame_equal(v2_extra_features(data.keys, data.sales, [o], range(1, 14), GROUPS),
                              v2_extra_features(data.keys, corrupted, [o], range(1, 14), GROUPS))
print("real-data leakage check passed", flush=True)

calib = pd.read_csv(ROOT / "results/phase2_calibration/benchmark_scores.csv")


def score(fc, col):
    b = vn1_score_breakdown(fc["target"].to_numpy(), fc[col].to_numpy())
    return {"score": b["score"], "mae_part": b["abs_error_sum"] / b["total_actual"],
            "bias_part": b["bias_term"] / b["total_actual"]}


def run(groups):
    """Per-origin LightGBM forecasts (long frames) for a feature set."""
    out = {}
    for o in ABLATION_ORIGINS:
        fc, _, info = v2_forecast(data, o, groups=groups)
        out[o] = (fc, info["best_iteration"])
    return out


rows, steps = [], []


def record(step, label, preds, col="lgbm"):
    sc = {}
    for o, (fc, it) in preds.items():
        s = score(fc, col)
        sc[o] = s["score"]
        rows.append({"step": step, "variant": label, "origin_idx": o, "best_iteration": it, **s})
    return pd.Series(sc)


current = ()
cur_preds = run(current)
cur_scores = record(0, "frozen features", cur_preds)
# The base must reproduce the calibration's frozen LightGBM exactly
ref = calib.query("comparison == 'all_season_cv' and model == 'lgbm_frozen'").set_index("origin_idx")["score"]
assert np.allclose(cur_scores.to_numpy(), ref.loc[ABLATION_ORIGINS].to_numpy(), atol=1e-9), "base != calibration"
print(f"base mean {cur_scores.mean():.4f}", flush=True)

for i, g in enumerate(list(GROUPS) + ["dormancy_gate"], start=1):
    if g == "dormancy_gate":
        preds = {}
        for o, (fc, it) in cur_preds.items():
            fc = fc.copy()
            mask = np.tile(dormant_mask(data.sales, o, GATE_WEEKS), H)
            fc["lgbm"] = np.where(mask, 0.0, fc["lgbm"].to_numpy())
            preds[o] = (fc, it)
    else:
        preds = run(current + (g,))
    s = record(i, f"+{g}", preds)
    diff = s - cur_scores
    improvement = -diff.mean()
    wins = int((diff < 0).sum())
    keep = bool(improvement >= KEEP_MIN_IMPROVEMENT and wins >= KEEP_MIN_WINS)
    steps.append({"step": i, "group": g, "against": list(current) or ["frozen features"],
                  "mean_before": cur_scores.mean(), "mean_after": s.mean(), "improvement": improvement,
                  "wins": wins, "n": len(diff), "kept": keep})
    print(f"{g:14s} mean {s.mean():.4f} vs {cur_scores.mean():.4f}  improvement {improvement:+.4f}  "
          f"wins {wins}/{len(diff)}  -> {'KEEP' if keep else 'drop'}", flush=True)
    if keep:
        cur_scores, cur_preds = s, preds
        if g != "dormancy_gate":
            current = current + (g,)

gate_kept = any(st["kept"] for st in steps if st["group"] == "dormancy_gate")
final = {"groups": list(current), "gate_weeks": GATE_WEEKS if gate_kept else None,
         "ablation_origins": ABLATION_ORIGINS, "rule": {"min_improvement": KEEP_MIN_IMPROVEMENT,
                                                        "min_wins": KEEP_MIN_WINS}, "steps": steps}
pd.DataFrame(rows).to_csv(OUT / "ablation_scores.csv", index=False)
(OUT / "lightgbm_v2_spec.json").write_text(json.dumps(final, indent=2, default=float))
print("\nLightGBM-v2:", json.dumps({k: final[k] for k in ("groups", "gate_weeks")}))
