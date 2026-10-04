"""Frozen Session 6 forecasting pipeline (DECISIONS.md D012).

One function produces both the development-holdout evaluation and the real
Phase 1 forecast, so the thing that was evaluated and the thing that was
submitted are the same code path — only `forecast_origin` differs.
"""
import numpy as np
import pandas as pd

from .data import KEY
from .features import build_direct_horizon_matrix, purge_fit_origins
from .models.lightgbm_model import train_lgbm

HORIZONS = list(range(1, 14))
SEGMENT_DTYPE = pd.CategoricalDtype(["active", "intermittent", "sparse"])

FROZEN_CONFIG = {
    "objective": "regression",  # D007/D009
    "lgb_params": dict(n_estimators=2000, learning_rate=0.05, num_leaves=127, min_child_samples=50),  # D009
    "random_state": 42,
    "stopping_rounds": 50,
    "origin_start": 52,  # enough lookback for lag_52
    "origin_step": 4,
    "n_val_origins": 3,
    "w_lgbm": 0.5,  # equal-weight blend with MA4, not tuned
    # D012: which column is the submitted forecast. LightGBM alone for an
    # Oct-Jan horizon (season-matched backtest); the blend won the
    # all-season rolling CV, so it is frozen alongside as a secondary.
    "primary": "lgbm",
}


def origin_pool(forecast_origin, horizons=HORIZONS, start=52, step=4):
    """Training origins whose full target window ends at or before
    `forecast_origin` — nothing after the forecast origin is ever a label."""
    return list(range(start, forecast_origin - max(horizons) + 1, step))


def _segment_as_category(mat):
    mat["segment"] = mat["segment"].astype(SEGMENT_DTYPE)
    return mat


def frozen_forecast(df, forecast_origin, config=FROZEN_CONFIG, horizons=HORIZONS):
    """Train the frozen LightGBM configuration using only data at or before
    `forecast_origin` (an index into df's sorted dates), then forecast the
    next len(horizons) weeks with LightGBM, MA4 and their weighted blend.

    Returns `(forecast, model, info)`. `forecast` is long-form: KEY + date +
    `lgbm`, `ma4`, `blend`, and `forecast` (= the `config["primary"]`
    column), plus `target` (NaN when the forecast weeks are beyond `df`).
    """
    origins = origin_pool(forecast_origin, horizons, config["origin_start"], config["origin_step"])
    val_origins = origins[-config["n_val_origins"]:]
    fit_origins = purge_fit_origins(origins, val_origins, horizons)
    assert max(fit_origins) + max(horizons) < min(val_origins), "fit/val overlap (D008)"
    assert max(val_origins) + max(horizons) <= forecast_origin, "a label lies after the forecast origin"

    fit_mat = _segment_as_category(build_direct_horizon_matrix(df, fit_origins, horizons))
    val_mat = _segment_as_category(build_direct_horizon_matrix(df, val_origins, horizons))
    eval_mat = _segment_as_category(
        build_direct_horizon_matrix(df, [forecast_origin], horizons, require_target=False)
    )
    feature_cols = [
        c for c in fit_mat.columns if c not in (*KEY, "origin_date", "target_date", "target")
    ]

    model, _ = train_lgbm(
        fit_mat, val_mat, feature_cols,
        objective=config["objective"],
        categorical_feature=["segment"],
        random_state=config["random_state"],
        stopping_rounds=config["stopping_rounds"],
        origin_col="origin_date",
        **config["lgb_params"],
    )

    lgbm = np.clip(model.predict(eval_mat[feature_cols]), 0, None)
    ma4 = eval_mat["roll_mean_4"].to_numpy()
    w = config["w_lgbm"]

    forecast = eval_mat[KEY + ["target_date", "target"]].rename(columns={"target_date": "date"})
    forecast["lgbm"] = lgbm
    forecast["ma4"] = ma4
    forecast["blend"] = w * lgbm + (1 - w) * ma4
    forecast["forecast"] = forecast[config["primary"]]

    assert forecast["forecast"].notna().all() and (forecast["forecast"] >= 0).all()
    assert not forecast.duplicated(KEY + ["date"]).any()

    info = {
        "forecast_origin_date": str(pd.Timestamp(eval_mat["origin_date"].iloc[0]).date()),
        "fit_origins": fit_origins,
        "val_origins": val_origins,
        "best_iteration": int(model.best_iteration_),
        "tracked_metric": list(model.evals_result_["valid_0"]),
        "feature_cols": feature_cols,
        "lgbm_params": {k: v for k, v in model.get_params().items() if not callable(v)},
    }
    return forecast, model, info


def to_submission(forecast, key_order, column="forecast"):
    """Pivot a long forecast to the official wide format: KEY columns then one
    column per forecast week, rows in `key_order` (the organizers' scorer
    asserts the submission index equals the actuals' index exactly)."""
    order = pd.MultiIndex.from_frame(key_order[KEY])
    assert not order.duplicated().any(), "key_order has duplicate series"
    wide = forecast.pivot(index=KEY, columns="date", values=column)
    # Exact key-set match: reindex alone would silently drop extra series.
    assert set(wide.index) == set(order), "forecast series differ from key_order's series"
    wide = wide.reindex(order)
    assert not wide.isna().any().any(), "submission has missing values"
    assert np.isfinite(wide.to_numpy()).all(), "submission has non-finite values"
    assert (wide.to_numpy() >= 0).all(), "submission has negative values"
    wide.columns = [pd.Timestamp(c).strftime("%Y-%m-%d") for c in wide.columns]
    return wide.reset_index()
