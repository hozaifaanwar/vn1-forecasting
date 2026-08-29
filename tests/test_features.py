import numpy as np
import pandas as pd
import pytest

from src.data import KEY
from src.features import (
    build_direct_horizon_matrix,
    classify_segment,
    purge_fit_origins,
    zero_rate_table,
)


def _toy_df(n_series=2, n_weeks=120, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")
    for p in range(n_series):
        sales = rng.integers(0, 5, size=n_weeks).astype(float)
        price = np.where(sales > 0, rng.uniform(5, 15, size=n_weeks), np.nan)
        price_was_missing = np.isnan(price)
        # forward-fill price like src.data.clean() does
        price = pd.Series(price).ffill().to_numpy()
        frames.append(pd.DataFrame({
            "Client": 0, "Warehouse": 0, "Product": p,
            "date": dates, "sales": sales, "price": price,
            "price_was_missing": price_was_missing,
        }))
    return pd.concat(frames, ignore_index=True)


def test_classify_segment_thresholds():
    zero_rate = pd.Series([0.0, 0.29, 0.3, 0.89, 0.9, 1.0])
    seg = classify_segment(zero_rate, active_max=0.3, sparse_min=0.9)
    assert seg.tolist() == ["active", "active", "intermittent", "intermittent", "sparse", "sparse"]


def test_zero_rate_table_matches_manual_calculation():
    df = _toy_df(n_series=1, n_weeks=10, seed=1)
    df.loc[df.index[:4], "sales"] = 0  # force a known zero_rate
    df.loc[df.index[4:], "sales"] = 1
    table = zero_rate_table(df)
    assert table["zero_rate"].iloc[0] == pytest.approx(0.4)
    assert table["weeks"].iloc[0] == 10


def test_build_direct_horizon_matrix_shape_and_target():
    df = _toy_df(n_series=3, n_weeks=120)
    origin = 60
    horizons = [1, 5, 13]
    result = build_direct_horizon_matrix(df, origins=[origin], horizons=horizons)

    assert len(result) == 3 * len(horizons)  # n_series * n_horizons
    assert set(result["horizon"]) == set(horizons)

    dates = np.sort(df["date"].unique())
    for h in horizons:
        row = result[(result["horizon"] == h) & (result["Product"] == 0)].iloc[0]
        expected_target = df[(df["Product"] == 0) & (df["date"] == dates[origin + h])]["sales"].iloc[0]
        assert row["target"] == pytest.approx(expected_target)
        assert row["origin_date"] == dates[origin]
        assert row["target_date"] == dates[origin + h]


def test_build_direct_horizon_matrix_lag_and_rolling_correctness():
    df = _toy_df(n_series=1, n_weeks=120)
    origin = 60
    result = build_direct_horizon_matrix(df, origins=[origin], horizons=[1])

    dates = np.sort(df["date"].unique())
    sales = df[df["Product"] == 0].sort_values("date")["sales"].to_numpy()

    row = result.iloc[0]
    assert row["lag_1"] == pytest.approx(sales[origin])       # value at origin
    assert row["lag_2"] == pytest.approx(sales[origin - 1])   # one week before origin
    assert row["roll_mean_4"] == pytest.approx(sales[origin - 3 : origin + 1].mean())
    assert row["zero_rate_12"] == pytest.approx((sales[origin - 11 : origin + 1] == 0).mean())


def test_build_direct_horizon_matrix_does_not_leak_future_values():
    """Corrupting data strictly after an origin must not change that
    origin's engineered features — the key leakage guarantee of the
    direct/horizon-as-feature strategy (DECISIONS.md D006)."""
    df = _toy_df(n_series=2, n_weeks=120, seed=2)
    origin = 60
    horizons = [1, 7, 13]

    baseline = build_direct_horizon_matrix(df, origins=[origin], horizons=horizons)

    corrupted = df.copy()
    dates = np.sort(df["date"].unique())
    future_mask = corrupted["date"] > dates[origin]
    corrupted.loc[future_mask, "sales"] = 999999.0
    corrupted.loc[future_mask, "price"] = 999999.0

    result = build_direct_horizon_matrix(corrupted, origins=[origin], horizons=horizons)

    feature_cols = [
        c for c in baseline.columns
        if c not in (
            "target", "target_date", "target_month",
            "target_week_sin", "target_week_cos", "target_quarter",
        )
    ]
    pd.testing.assert_frame_equal(
        baseline[feature_cols].reset_index(drop=True),
        result[feature_cols].reset_index(drop=True),
    )
    # targets, by contrast, SHOULD differ for any row where target_date is
    # in the corrupted future — confirms the test actually exercises the
    # future period rather than trivially passing on an all-past horizon set.
    assert (result["target"] == 999999.0).any()


def test_build_direct_horizon_matrix_rejects_origin_too_close_to_start():
    df = _toy_df(n_series=1, n_weeks=60)
    with pytest.raises(AssertionError, match="too early"):
        build_direct_horizon_matrix(df, origins=[10], horizons=[1])  # needs lookback 52


def test_build_direct_horizon_matrix_rejects_origin_too_close_to_end():
    df = _toy_df(n_series=1, n_weeks=60)
    with pytest.raises(AssertionError, match="exceeds available dates"):
        build_direct_horizon_matrix(df, origins=[55], horizons=[13])


def test_purge_fit_origins_drops_overlapping_origins():
    # Reproduces the exact overlap the review caught: origins 120/124/128
    # have targets reaching 133/137/141, all >= min(val_origins)=132.
    origins = list(range(52, 144, 4))
    val_origins = [132, 136, 140]
    horizons = list(range(1, 14))

    purged = purge_fit_origins(origins, val_origins, horizons)

    assert max(purged) + max(horizons) < min(val_origins)
    assert 120 not in purged and 124 not in purged and 128 not in purged
    assert 116 in purged  # the last origin that should legitimately survive


def test_purge_fit_origins_keeps_everything_when_already_disjoint():
    origins = [10, 20, 30]
    val_origins = [100]
    horizons = [1, 2, 3]
    assert purge_fit_origins(origins, val_origins, horizons) == origins
