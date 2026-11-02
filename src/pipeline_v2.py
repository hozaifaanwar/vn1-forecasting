"""Phase 2 forecasting procedures (DECISIONS.md D014).

`v2_forecast` is the frozen procedure (`pipeline.frozen_forecast`) with
optional extra feature groups and the dormancy gate. With no groups and no
gate it reproduces `frozen_forecast` exactly (tested). Every LightGBM model
is trained only on data at or before its origin and early-stops on its own
three purged internal origins, never on a scored target.

`jan_adjusted_ma4` is challenger (6).
"""
import numpy as np
import pandas as pd

from .data import KEY
from .features import build_direct_horizon_matrix, purge_fit_origins
from .features_v2 import GROUPS, dormant_mask, sales_matrix, v2_extra_features
from .models.lightgbm_model import train_lgbm
from .pipeline import FROZEN_CONFIG, HORIZONS, _segment_as_category, origin_pool

CATEGORICAL_V2 = ["segment", "segment_fs", "id_client", "id_warehouse", "id_product"]


class Data:
    """df plus the sales matrix, computed once."""

    def __init__(self, df):
        self.df = df
        self.dates = np.sort(df["date"].unique())
        self.keys, self.sales = sales_matrix(df)


def _matrix(data, origins, groups, require_target=True):
    m = _segment_as_category(
        build_direct_horizon_matrix(data.df, origins, HORIZONS, require_target=require_target))
    extra = v2_extra_features(data.keys, data.sales, origins, HORIZONS, groups)
    assert len(extra) == len(m)
    # row alignment check: same series order as the frozen builder
    n = len(data.keys)
    assert (m[KEY].iloc[:n].to_numpy() == data.keys[KEY].to_numpy()).all()
    for c in extra.columns:
        m[c] = extra[c].to_numpy() if not isinstance(extra[c].dtype, pd.CategoricalDtype) else extra[c].values
    return m


def v2_forecast(data, forecast_origin, groups=(), gate_weeks=None, config=FROZEN_CONFIG):
    groups = tuple(g for g in GROUPS if g in groups)
    origins = origin_pool(forecast_origin, HORIZONS, config["origin_start"], config["origin_step"])
    val_origins = origins[-config["n_val_origins"]:]
    fit_origins = purge_fit_origins(origins, val_origins, HORIZONS)
    assert fit_origins, "no training origin"
    assert max(fit_origins) + max(HORIZONS) < min(val_origins)
    assert max(val_origins) + max(HORIZONS) <= forecast_origin

    fit_mat = _matrix(data, fit_origins, groups)
    val_mat = _matrix(data, val_origins, groups)
    eval_mat = _matrix(data, [forecast_origin], groups, require_target=False)
    feature_cols = [c for c in fit_mat.columns if c not in (*KEY, "origin_date", "target_date", "target")]
    cats = [c for c in CATEGORICAL_V2 if c in feature_cols]

    model, _ = train_lgbm(
        fit_mat, val_mat, feature_cols, objective=config["objective"], categorical_feature=cats,
        random_state=config["random_state"], stopping_rounds=config["stopping_rounds"],
        origin_col="origin_date", **config["lgb_params"])

    lgbm = np.clip(model.predict(eval_mat[feature_cols]), 0, None)
    if gate_weeks is not None:
        dormant = np.tile(dormant_mask(data.sales, forecast_origin, gate_weeks), len(HORIZONS))
        lgbm = np.where(dormant, 0.0, lgbm)
    ma4 = eval_mat["roll_mean_4"].to_numpy()
    w = config["w_lgbm"]
    fc = eval_mat[KEY + ["target_date", "target"]].rename(columns={"target_date": "date"})
    fc["lgbm"] = lgbm
    fc["ma4"] = ma4
    fc["blend"] = w * lgbm + (1 - w) * ma4
    assert fc[["lgbm", "ma4", "blend"]].notna().all().all() and (fc["lgbm"] >= 0).all()
    info = {"fit_origins": fit_origins, "val_origins": val_origins,
            "best_iteration": int(model.best_iteration_), "feature_cols": feature_cols,
            "groups": list(groups), "gate_weeks": gate_weeks}
    return fc, model, info


def jan_adjusted_multiplier(sales, o, own_min_level=5.0, h=13):
    """Per-series multiplier for challenger (6), D014.

    For each prior year k (origin p = o - 52k, MA4 defined at p, window p+1..p+h
    fully observed by o): portfolio ratio = window mean / MA4 level (totals);
    own ratio = the same per series, only where the series' MA4 at p >= 5.
    Ratios are averaged over the years where defined; multiplier = 0.5 own +
    0.5 portfolio, portfolio alone where no own ratio exists, 1 where no prior
    year exists. Returns (multiplier, n_years).
    """
    port, own = [], []
    k = 1
    while o - 52 * k - 3 >= 0:
        p = o - 52 * k
        assert p + h <= o
        lvl = sales[:, p - 3: p + 1].mean(axis=1)
        win = sales[:, p + 1: p + 1 + h].mean(axis=1)
        port.append(win.sum() / lvl.sum())
        r = np.full(len(lvl), np.nan)
        ok = lvl >= own_min_level
        r[ok] = win[ok] / lvl[ok]
        own.append(r)
        k += 1
    if not port:
        return np.ones(sales.shape[0]), 0
    p_ratio = float(np.mean(port))
    own_arr = np.vstack(own)
    n_def = (~np.isnan(own_arr)).sum(axis=0)
    own_mean = np.divide(np.nansum(own_arr, axis=0), n_def, out=np.full(own_arr.shape[1], np.nan),
                         where=n_def > 0)
    mult = np.where(np.isnan(own_mean), p_ratio, 0.5 * own_mean + 0.5 * p_ratio)
    return mult, len(port)


def jan_adjusted_ma4(sales, o, h=13):
    """(n_series, h) forecast: MA4 level x multiplier, flat over the window."""
    mult, _ = jan_adjusted_multiplier(sales, o, h=h)
    lvl = sales[:, o - 3: o + 1].mean(axis=1)
    return np.repeat((lvl * mult)[:, None], h, axis=1)
