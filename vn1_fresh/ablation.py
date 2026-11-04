"""Post-hoc ablation of the vn1_fresh pipeline (DECISIONS.md D017/D018).

Diagnostic only: it scores on the Phase 2 actuals, so nothing here selects a model.
A. Post-processing on the committed Phase 2 run: LightGBM 1 seed -> 3 seeds -> ensemble -> dormancy.
B. Leave-one-out inside LightGBM (1 seed, 300 rounds): remove one ingredient, retrain at
   the Jan 2023 fold (origin 130, actuals in history) and at the Phase 2 origin (182).

    python vn1_fresh/ablation.py --data "path/to/vn1-forecasting-competition"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vn1_fresh as V  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"

_Dataset = V.lgb.Dataset


class _SafeDataset(_Dataset):
    """Drop categorical names that a variant removed from the features."""
    def __init__(self, *a, feature_name="auto", categorical_feature="auto", **k):
        if isinstance(categorical_feature, list) and isinstance(feature_name, list):
            categorical_feature = [c for c in categorical_feature if c in feature_name] or "auto"
        super().__init__(*a, feature_name=feature_name, categorical_feature=categorical_feature, **k)


V.lgb.Dataset = _SafeDataset
FOLD = 130
ROUND = 300

FEATURE_GROUPS = {
    "AMSI index features": ["amsi4", "amsi13", "amsi_idx"],
    "last-year features": ["ly", "ly3", "ly2", "ly_index", "woy"],
    "zero/recency features": ["zero13", "zero52", "zero_life", "since_last", "age"],
    "client id": ["client"],
    "rolling stats": ["mean4", "mean8", "mean13", "mean26", "mean52", "std13", "trend"],
}


def untrimmed(P):
    """Same panel, but every series 'launches' at week 0 (leading zeros count as history)."""
    Q = V.Panel.__new__(V.Panel)
    Q.__dict__.update(P.__dict__)
    Q.first = np.zeros(P.n, dtype=int)
    Q.launched = np.ones_like(P.launched)
    Q.Yn = P.Y.copy()
    z = np.zeros((P.n, 1))
    Q.C = np.concatenate([z, np.cumsum(P.Y, 1)], 1)
    Q.C2 = np.concatenate([z, np.cumsum(P.Y ** 2, 1)], 1)
    Q.CM = np.concatenate([z, np.cumsum(np.ones_like(P.Y), 1)], 1)
    Q.CZ = np.concatenate([z, np.cumsum((P.Y == 0).astype(float), 1)], 1)
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    data = Path(ap.parse_args().data)
    hist = V.load_history(data, 2)
    act2 = V.read_weekly(data / "Phase 2 - Sales.csv").loc[hist.index].to_numpy(float)
    dates = V.woy_fix(hist.columns)
    P = V.Panel(hist.to_numpy(), hist.index.get_level_values("Client").to_numpy())
    L = P.T - 1
    act_fold = P.Y[:, FOLD + 1: FOLD + 1 + V.H]
    rep = json.loads((OUT / "phase2_report.json").read_text())
    rows = []

    # ---- A. post-processing, from the committed run
    comps = {m: pd.read_csv(OUT / f"phase2_component_{m}.csv").set_index(V.INDEX).loc[hist.index].to_numpy()
             for m in V.COMPONENTS}
    w = np.array([rep["ensemble_weights"][m] for m in V.COMPONENTS])
    ens = V.combine(comps, w)
    final = pd.read_csv(OUT / "phase2_forecast.csv").set_index(V.INDEX).loc[hist.index].to_numpy()
    assert np.allclose(V.dormancy(P, L, ens, rep["dormancy_n"]), final, atol=1e-6), "cannot rebuild the forecast"

    # ---- B. leave-one-out LightGBM
    V.LGB_SEEDS, V.ROUNDS = (7,), (ROUND,)
    train, feats = V.build_train(P, dates, L)
    print(f"train table {len(train):,} rows", flush=True)

    def lgbm(train_, feats_, panel, params=None):
        saved = dict(V.LGB_PARAMS)
        if params:
            V.LGB_PARAMS.clear(); V.LGB_PARAMS.update(params)
        try:
            res = {o: V.lgbm_fit_predict(train_, feats_, panel, o, dates)[0][ROUND] for o in (FOLD, L)}
        finally:
            V.LGB_PARAMS.clear(); V.LGB_PARAMS.update(saved)
        return res

    def add(part, name, fold_fc, p2_fc):
        e = p2_fc - act2
        rows.append({"part": part, "variant": name,
                     "jan2023_fold": V.vn1_score(act_fold, fold_fc) if fold_fc is not None else np.nan,
                     "phase2": V.vn1_score(act2, p2_fc), "phase2_bias": e.sum() / act2.sum()})
        print(rows[-1], flush=True)

    full = lgbm(train, feats, P)
    add("B", "LightGBM full (1 seed)", full[FOLD], full[L])
    add("A", "+ 3 seeds averaged", None, comps["lgbm"])
    add("A", "+ ensemble weights", None, ens)
    add("A", "+ dormancy N=8 (final)", None, final)
    add("A", "LightGBM 3 seeds + dormancy only", None, V.dormancy(P, L, comps["lgbm"], rep["dormancy_n"]))
    add("A", "ensemble without naive (weights renormalised)", None,
        V.combine(comps, np.where(np.array(V.COMPONENTS) == "naive", 0, w) * w.sum() / (w.sum() - w[V.COMPONENTS.index("naive")])))

    for name, cols in FEATURE_GROUPS.items():
        f = [c for c in feats if c not in cols]
        r = lgbm(train, f, P)
        add("B", f"- {name}", r[FOLD], r[L])
    r = lgbm(train, feats, P, {**V.LGB_PARAMS, "objective": "regression"})
    add("B", "- Tweedie (L2 loss instead)", r[FOLD], r[L])
    Q = untrimmed(P)
    tq, fq = V.build_train(Q, dates, L)
    r = lgbm(tq, fq, Q)
    add("B", "- leading-zero trimming", r[FOLD], r[L])
    del tq
    sparse = train[train["_t"] % 6 == V.T0 % 6]
    r = lgbm(sparse, feats, P)
    add("B", "- half the training origins (step 6)", r[FOLD], r[L])

    out = pd.DataFrame(rows)
    base = out.iloc[0]
    out["delta_phase2_vs_full"] = out["phase2"] - base["phase2"]
    out["delta_jan2023_vs_full"] = out["jan2023_fold"] - base["jan2023_fold"]
    out.to_csv(OUT / "ablation_phase2.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
