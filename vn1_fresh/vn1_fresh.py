"""VN1 fresh pipeline: winners' ideas, one honest run.

Standalone. Needs only the official VN1 data folder (the CSVs as published) and
numpy, pandas, scipy, lightgbm, statsforecast. It never opens the sales of the
phase it forecasts: phase 1 is forecast from phase 0 only, phase 2 from phase 0
and phase 1 only. Score the result afterwards with the official vn1_score.py.

    python vn1_fresh.py --data "path/to/vn1-forecasting-competition" --phase 2 --out out

What it does (every choice is made on backtests, never on the scored phase):

1. Components, each forecast at an origin from history up to that origin:
   - naive, ma13                 benchmarks
   - amsi4, amsi13               anchored multiplicative seasonal index (1st place):
                                 recent anchor level x last years' index for the next
                                 13 weeks; index = ratio of sums over past years,
                                 shrunk from the series to its client to the portfolio
   - theta                       DynamicOptimizedTheta, season 52 (2nd place)
   - lgbm                        global direct LightGBM, Tweedie loss (2nd, 3rd, 5th),
                                 leading zeros removed (4th), recency/zero/seasonal
                                 features incl. the AMSI index (target-aligned)
2. Four non-overlapping 13-week backtest folds covering the last year (2nd place);
   the season-matched fold (one year before the forecast origin) counts double.
3. LightGBM boosting rounds picked on the folds.
4. Ensemble weights (non-negative) fitted on the folds to the VN1 metric (2nd, 5th).
5. Dormancy rule (5th): zero the forecast of series with no sale in the last N
   weeks; N (or no rule) picked on the folds.
6. Refit at the real origin, apply weights and rule, write the forecast.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
INDEX = ["Client", "Warehouse", "Product"]
H = 13
SEASON = 52
T0 = 60            # first training origin (needs a year of history for seasonal features)
STEP = 3           # weeks between training origins
ROUNDS = (150, 300, 500, 800, 1200)
FOLD_WEIGHTS = (1.0, 1.0, 1.0, 2.0)   # folds at L-13, L-26, L-39, L-52 (season-matched)
DORMANCY_N = (None, 8, 13, 26, 52)
COMPONENTS = ("lgbm", "amsi4", "amsi13", "theta", "ma13", "naive")
LGB_PARAMS = dict(objective="tweedie", tweedie_variance_power=1.5, learning_rate=0.03,
                  num_leaves=127, min_data_in_leaf=200, feature_fraction=0.7,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, max_bin=255,
                  num_threads=0, verbose=-1)
LGB_SEEDS = (7, 11, 23)   # same model, three seeds, averaged


# ---------------------------------------------------------------- data

def read_weekly(path):
    table = pd.read_csv(path).set_index(INDEX)
    table.columns = pd.to_datetime(table.columns)
    return table


def load_history(data, phase):
    files = ["Phase 0 - Sales.csv"] + (["Phase 1 - Sales.csv"] if phase == 2 else [])
    parts = [read_weekly(Path(data) / f) for f in files]
    for p in parts[1:]:
        assert p.index.equals(parts[0].index), "phase files list different series"
    hist = pd.concat(parts, axis=1)
    assert hist.columns.is_unique and hist.columns.is_monotonic_increasing
    assert (np.diff(hist.columns.values).astype("timedelta64[D]").astype(int) == 7).all()
    assert not hist.isna().any().any()
    return hist


def vn1_score(actual, forecast):
    err = forecast - actual
    return (np.abs(err).sum() + abs(err.sum())) / actual.sum()


class Panel:
    """Sales matrix (n series x T weeks) with cumulative sums for fast window stats.
    A series 'launches' at its first sale; weeks before it are not history (NaN)."""

    def __init__(self, Y, clients):
        self.Y = Y.astype(np.float64)
        self.n, self.T = Y.shape
        pos = self.Y > 0
        idx = np.arange(self.T)
        self.first = np.where(pos.any(1), pos.argmax(1), self.T)
        self.launched = idx[None, :] >= self.first[:, None]
        self.Yn = np.where(self.launched, self.Y, np.nan)
        z = np.zeros((self.n, 1))
        cum = lambda a: np.concatenate([z, np.cumsum(a, 1)], 1)  # noqa: E731
        y0 = np.where(self.launched, self.Y, 0.0)
        self.C, self.C2 = cum(y0), cum(y0 ** 2)
        self.CM = cum(self.launched.astype(float))
        self.CZ = cum(((self.Y == 0) & self.launched).astype(float))
        self.last = np.maximum.accumulate(np.where(pos, idx[None, :], -1), 1)
        _, self.client = np.unique(clients, return_inverse=True)
        self.n_clients = self.client.max() + 1

    def wsum(self, arr, t, w):
        return arr[:, t + 1] - arr[:, max(t + 1 - w, 0)]

    def rmean(self, t, w):
        if t < 0:
            return np.full(self.n, np.nan)
        m = self.wsum(self.CM, t, w)
        return np.where(m > 0, self.wsum(self.C, t, w) / np.maximum(m, 1), np.nan)


# ---------------------------------------------------------------- components

def anchor(P, t, kind):
    w4 = P.Y[:, t - 3: t + 1] @ np.array([0.1, 0.2, 0.3, 0.4])
    if kind == "w4":
        return w4
    return 0.5 * w4 + 0.5 * P.Y[:, max(t - 12, 0): t + 1].mean(1)


def amsi(P, o, kind, k_own=50.0, k_client=300.0):
    """Anchored multiplicative seasonal index at origin o -> (forecast, index), (n, H)."""
    num, den = np.zeros((P.n, H)), np.zeros(P.n)
    for y in (1, 2):
        a = o - SEASON * y
        if a - 12 < 0:
            continue
        ok = P.first <= a - 12          # launched before last year's anchor window
        num += np.where(ok[:, None], P.Y[:, a + 1: a + 1 + H], 0.0)
        den += np.where(ok, anchor(P, a, kind), 0.0)
    if den.sum() <= 0:
        return np.repeat(anchor(P, o, kind)[:, None], H, 1), np.ones((P.n, H))
    port = num.sum(0) / den.sum()
    cnum = np.stack([np.bincount(P.client, num[:, h], minlength=P.n_clients) for h in range(H)], 1)
    cden = np.bincount(P.client, den, minlength=P.n_clients)
    cidx = np.where(cden[:, None] > 0, cnum / np.maximum(cden, 1e-9)[:, None], port[None, :])
    cw = (cden / (cden + k_client))[:, None]
    cidx = cw * cidx + (1 - cw) * port[None, :]
    own = np.where(den[:, None] > 0, num / np.maximum(den, 1e-9)[:, None], 0.0)
    sw = (den / (den + k_own))[:, None]
    idx = np.clip(sw * own + (1 - sw) * cidx[P.client], 0.0, 4.0)
    return anchor(P, o, kind)[:, None] * idx, idx


def theta(P, o, dates):
    from statsforecast import StatsForecast
    from statsforecast.models import DynamicOptimizedTheta, HistoricAverage
    ii, tt = np.nonzero(P.launched[:, : o + 1])
    df = pd.DataFrame({"unique_id": ii, "ds": dates[tt], "y": P.Y[ii, tt]})
    sf = StatsForecast(models=[DynamicOptimizedTheta(season_length=SEASON)],
                       fallback_model=HistoricAverage(), freq="W-MON", n_jobs=-1)
    fc = sf.forecast(df=df, h=H).reset_index()
    out = np.zeros((P.n, H))
    fc = fc.sort_values(["unique_id", "ds"])
    vals = fc["DynamicOptimizedTheta"].to_numpy().reshape(-1, H)
    out[fc["unique_id"].to_numpy()[::H].astype(int)] = vals
    return np.clip(np.nan_to_num(out), 0, None)


def flat(v):
    return np.repeat(np.asarray(v, float)[:, None], H, 1)


def simple_components(P, o, dates):
    return {"naive": flat(P.Y[:, o]), "ma13": flat(P.Y[:, o - 12: o + 1].mean(1)),
            "amsi4": amsi(P, o, "w4")[0], "amsi13": amsi(P, o, "blend")[0],
            "theta": theta(P, o, dates)}


# ---------------------------------------------------------------- LightGBM table

def origin_rows(P, t, dates, with_target):
    """Feature rows at origin t for every launched series and h = 1..H."""
    s = np.flatnonzero(P.first <= t)
    base = {f"lag{k}": P.Yn[s, t - k] for k in range(13)}
    for w in (4, 8, 13, 26, 52):
        base[f"mean{w}"] = P.rmean(t, w)[s]
    m13 = P.wsum(P.CM, t, 13)[s]
    s13, q13 = P.wsum(P.C, t, 13)[s], P.wsum(P.C2, t, 13)[s]
    with np.errstate(invalid="ignore", divide="ignore"):
        base["std13"] = np.sqrt(np.maximum(q13 / m13 - (s13 / m13) ** 2, 0))
        base["zero13"] = P.wsum(P.CZ, t, 13)[s] / m13
        base["zero52"] = P.wsum(P.CZ, t, 52)[s] / P.wsum(P.CM, t, 52)[s]
        base["zero_life"] = P.CZ[s, t + 1] / P.CM[s, t + 1]
        base["trend"] = base["mean4"] / base["mean26"]
    base["age"] = (t - P.first[s]).astype(float)
    base["since_last"] = np.where(P.last[s, t] >= 0, t - P.last[s, t], np.nan).astype(float)
    base["client"] = P.client[s].astype(float)
    a4, i4 = amsi(P, t, "w4")
    a13, _ = amsi(P, t, "blend")
    ly_anchor = P.rmean(t - SEASON, 4)[s]
    woy = dates.isocalendar().week.to_numpy()
    frames = []
    for h in range(1, H + 1):
        f = dict(base)
        f["h"] = np.full(len(s), h, float)
        tgt = t + h
        f["woy"] = np.full(len(s), woy[tgt], float)
        def at(k):  # noqa: E306
            return P.Yn[s, k] if 0 <= k < P.T else np.full(len(s), np.nan)
        f["ly"] = at(tgt - 52)
        with np.errstate(invalid="ignore"):
            f["ly3"] = np.nanmean(np.stack([at(tgt - 53), at(tgt - 52), at(tgt - 51)]), 0)
        f["ly2"] = at(tgt - 104)
        with np.errstate(invalid="ignore", divide="ignore"):
            f["ly_index"] = f["ly3"] / ly_anchor
        f["amsi4"], f["amsi13"], f["amsi_idx"] = a4[s, h - 1], a13[s, h - 1], i4[s, h - 1]
        if with_target:
            f["y"] = P.Y[s, tgt]
        f["_s"] = s
        frames.append(pd.DataFrame(f))
    out = pd.concat(frames, ignore_index=True)
    out["_t"] = t
    return out


def woy_fix(dates):
    """Week-of-year for every index up to T + H (extends the calendar beyond history)."""
    ext = pd.date_range(dates[0], periods=len(dates) + H + 1, freq="W-MON")
    return ext


def build_train(P, dates, last_origin):
    parts = [origin_rows(P, t, dates, True) for t in range(T0, last_origin - H + 1, STEP)]
    df = pd.concat(parts, ignore_index=True)
    feats = [c for c in df.columns if c not in ("y", "_s", "_t")]
    return df, feats


def lgbm_fit_predict(train, feats, P, o, dates):
    tr = train[train["_t"] + H <= o]
    ds = lgb.Dataset(tr[feats].to_numpy(np.float32), tr["y"].to_numpy(np.float32),
                     feature_name=feats, categorical_feature=["client"], free_raw_data=False)
    X = origin_rows(P, o, dates, False)
    Xf = X[feats].to_numpy(np.float32)
    rows, hs = X["_s"].to_numpy(), X["h"].to_numpy().astype(int) - 1
    out = {r: np.zeros((P.n, H)) for r in ROUNDS}
    imp = dict.fromkeys(feats, 0.0)
    for seed in LGB_SEEDS:
        model = lgb.train({**LGB_PARAMS, "seed": seed}, ds, num_boost_round=max(ROUNDS))
        for r in ROUNDS:
            out[r][rows, hs] += np.clip(model.predict(Xf, num_iteration=r), 0, None) / len(LGB_SEEDS)
        for f, g in zip(feats, model.feature_importance("gain")):
            imp[f] += round(float(g)) / len(LGB_SEEDS)
    return out, len(tr), imp


# ---------------------------------------------------------------- selection

def combine(comps, w):
    return sum(wi * comps[m] for m, wi in zip(COMPONENTS, w))


def fit_weights(objective, k):
    """Non-negative weights minimising the (non-smooth) VN1 objective.

    The bias term makes single-weight moves useless near the optimum (they change the total
    level), so the search uses level-preserving moves (shift weight from one component to
    another) plus a scale move, with a shrinking step, from several starts."""
    rng = np.random.default_rng(0)
    starts = [np.full(k, 1.0 / k)] + [np.eye(k)[i] for i in range(k)] + list(rng.dirichlet(np.ones(k), 8))
    best_w, best_f = None, np.inf
    for w in starts:
        w, f = w.copy(), objective(w)
        for step in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005):
            improved = True
            while improved:
                improved = False
                moves = [(i, j) for i in range(k) for j in range(k) if i != j] + [("scale", 1), ("scale", -1)]
                for i, j in moves:
                    cand = w.copy()
                    if i == "scale":
                        cand = cand * (1 + j * step)
                    else:
                        d = min(step, cand[j])
                        if d <= 0:
                            continue
                        cand[i] += d
                        cand[j] -= d
                    fc = objective(cand)
                    if fc < f - 1e-7:
                        w, f, improved = cand, fc, True
        if f < best_f:
            best_w, best_f = w, f
    return best_w


def dormancy(P, o, fc, n_weeks):
    if n_weeks is None:
        return fc
    dead = (P.last[:, o] < 0) | (o - P.last[:, o] >= n_weeks)
    return np.where(dead[:, None], 0.0, fc)


def run(data, phase, out_dir):
    t_start = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = load_history(data, phase)
    dates = woy_fix(hist.columns)
    P = Panel(hist.to_numpy(), hist.index.get_level_values("Client").to_numpy())
    L = P.T - 1
    folds = [L - 13, L - 26, L - 39, L - 52]
    fw = np.array(FOLD_WEIGHTS)
    log = lambda *a: print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)  # noqa: E731
    log(f"phase {phase}: {P.n} series, {P.T} weeks, last week {hist.columns[-1].date()}, folds {folds}")

    train, feats = build_train(P, dates, L)
    log(f"training table {len(train):,} rows x {len(feats)} features")

    comps, rounds_pred, info = {}, {}, {}
    for o in folds + [L]:
        comps[o] = simple_components(P, o, dates)
        rounds_pred[o], n_rows, imp = lgbm_fit_predict(train, feats, P, o, dates)
        info[o] = {"train_rows": n_rows}
        log(f"origin {o}: components done ({n_rows:,} training rows)")
    del train

    actual = {o: P.Y[:, o + 1: o + 1 + H] for o in folds}
    wscore = lambda f: float(np.dot(fw, [f(o) for o in folds]) / fw.sum())  # noqa: E731

    rscores = {r: wscore(lambda o: vn1_score(actual[o], rounds_pred[o][r])) for r in ROUNDS}
    best_r = min(rscores, key=rscores.get)
    for o in folds + [L]:
        comps[o]["lgbm"] = rounds_pred[o][best_r]
    log(f"LightGBM rounds: {best_r}  {json.dumps({k: round(v, 4) for k, v in rscores.items()})}")

    fold_scores = {m: {int(o): vn1_score(actual[o], comps[o][m]) for o in folds} for m in COMPONENTS}

    def objective(w):
        return wscore(lambda o: vn1_score(actual[o], combine(comps[o], w)))
    weights = fit_weights(objective, len(COMPONENTS))
    log("ensemble weights " + json.dumps(dict(zip(COMPONENTS, weights.round(3).tolist()))))

    dscores = {str(n): wscore(lambda o: vn1_score(actual[o], dormancy(P, o, combine(comps[o], weights), n)))
               for n in DORMANCY_N}
    best_n = min(DORMANCY_N, key=lambda n: dscores[str(n)])
    log(f"dormancy N = {best_n}  {json.dumps({k: round(v, 4) for k, v in dscores.items()})}")

    final = {o: dormancy(P, o, combine(comps[o], weights), best_n) for o in folds + [L]}
    fold_scores["ENSEMBLE"] = {int(o): vn1_score(actual[o], final[o]) for o in folds}

    fdates = pd.date_range(hist.columns[-1], periods=H + 1, freq="W-MON")[1:]
    cols = [d.strftime("%Y-%m-%d") for d in fdates]
    def write(name, m):  # noqa: E306
        assert m.shape == (P.n, H) and np.isfinite(m).all() and (m >= 0).all()
        pd.DataFrame(m, index=hist.index, columns=cols).to_csv(out_dir / name)
    write(f"phase{phase}_forecast.csv", final[L])
    for m in COMPONENTS:
        write(f"phase{phase}_component_{m}.csv", comps[L][m])

    summary = {
        "phase": phase, "history_last_week": str(hist.columns[-1].date()), "forecast_weeks": cols,
        "folds": folds, "fold_weights": list(FOLD_WEIGHTS),
        "lgbm_rounds": best_r, "lgbm_seeds": list(LGB_SEEDS), "lgbm_round_scores": rscores, "lgbm_params": LGB_PARAMS,
        "ensemble_weights": dict(zip(COMPONENTS, weights.tolist())),
        "dormancy_n": best_n, "dormancy_scores": dscores,
        "fold_scores": fold_scores,
        "fold_weighted_mean": {m: wscore(lambda o: v[int(o)]) for m, v in fold_scores.items()},
        "forecast_weekly_total": {"ensemble": float(final[L].sum() / H),
                                  **{m: float(comps[L][m].sum() / H) for m in COMPONENTS}},
        "last_week_total": float(P.Y[:, L].sum()), "ma4_total": float(P.Y[:, L - 3:].mean(1).sum()),
        "lgbm_top_features": dict(sorted(imp.items(), key=lambda kv: -kv[1])[:15]),
        "runtime_s": round(time.time() - t_start),
    }
    (out_dir / f"phase{phase}_report.json").write_text(json.dumps(summary, indent=2, default=float))
    table = pd.DataFrame(fold_scores).T
    table["weighted"] = [summary["fold_weighted_mean"][m] for m in table.index]
    log("fold scores (lower is better):\n" + table.round(4).to_string())
    log(f"wrote {out_dir / f'phase{phase}_forecast.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="folder with the official VN1 CSVs")
    ap.add_argument("--phase", type=int, choices=(1, 2), default=2)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    run(a.data, a.phase, a.out)


if __name__ == "__main__":
    main()
