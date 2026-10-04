import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from src.data import KEY
from src.pipeline import FROZEN_CONFIG, frozen_forecast, origin_pool, to_submission

FAST_CONFIG = {**FROZEN_CONFIG, "lgb_params": dict(n_estimators=30, learning_rate=0.1, num_leaves=7, min_child_samples=5)}


def _toy_df(n_series=20, n_weeks=130, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")
    frames = []
    for p in range(n_series):
        sales = rng.poisson(rng.uniform(0.2, 6), size=n_weeks).astype(float)
        price = np.where(sales > 0, rng.uniform(5, 15, size=n_weeks), np.nan)
        frames.append(pd.DataFrame({
            "Client": p % 3, "Warehouse": 0, "Product": p, "date": dates, "sales": sales,
            "price": pd.Series(price).ffill().to_numpy(), "price_was_missing": np.isnan(price),
        }))
    return pd.concat(frames, ignore_index=True)


def test_origin_pool_labels_never_pass_the_forecast_origin():
    assert origin_pool(156) == list(range(52, 144, 4))  # Session 5 dev-holdout pool
    assert origin_pool(169)[-1] == 156                   # Phase 1: last fully-labeled origin
    assert max(origin_pool(169)) + 13 <= 169


def test_frozen_forecast_ignores_everything_after_the_forecast_origin():
    df = _toy_df()
    origin = 116
    cutoff = df["date"].unique()[origin]
    corrupted = df.copy()
    future = corrupted["date"] > cutoff
    corrupted.loc[future, "sales"] = 999.0
    corrupted.loc[future, "price"] = 999.0

    clean_fc, _, _ = frozen_forecast(df, origin, config=FAST_CONFIG)
    corrupt_fc, _, _ = frozen_forecast(corrupted, origin, config=FAST_CONFIG)
    np.testing.assert_array_equal(clean_fc["forecast"], corrupt_fc["forecast"])


def test_frozen_forecast_blends_lgbm_and_ma4_and_covers_every_series_week():
    df = _toy_df()
    last = df["date"].nunique() - 1
    fc, model, info = frozen_forecast(df, last, config=FAST_CONFIG)

    assert len(fc) == df.groupby(KEY).ngroups * 13
    assert fc["target"].isna().all()  # genuinely beyond the data
    assert fc["date"].min() == df["date"].max() + pd.Timedelta(weeks=1)
    np.testing.assert_allclose(fc["blend"], 0.5 * fc["lgbm"] + 0.5 * fc["ma4"])
    np.testing.assert_array_equal(fc["forecast"], fc[FAST_CONFIG["primary"]])

    last4 = df[df["date"] > df["date"].unique()[last - 4]].groupby(KEY)["sales"].mean()
    ma4 = fc.groupby(KEY)["ma4"].first()
    pd.testing.assert_series_equal(ma4, last4, check_names=False)
    assert info["tracked_metric"] == ["vn1_score_origin_mean"]


def test_to_submission_matches_official_wide_format_and_row_order():
    df = _toy_df()
    fc, _, _ = frozen_forecast(df, df["date"].nunique() - 1, config=FAST_CONFIG)
    key_order = df[KEY].drop_duplicates().iloc[::-1].reset_index(drop=True)  # deliberately not sorted
    sub = to_submission(fc, key_order)

    assert list(sub.columns[:3]) == KEY
    assert len(sub.columns) == 3 + 13
    pd.testing.assert_frame_equal(sub[KEY], key_order)
    assert not sub.isna().any().any()
