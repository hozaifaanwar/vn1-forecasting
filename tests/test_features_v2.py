import numpy as np
import pandas as pd
import pytest

from src.data import KEY
from src.features_v2 import GROUPS, dormant_mask, first_last_sale, sales_matrix, v2_extra_features
from src.pipeline import frozen_forecast
from src.pipeline_v2 import Data, jan_adjusted_multiplier, v2_forecast

HORIZONS = list(range(1, 14))


def _toy_df(n_weeks=200, seed=0):
    """6 series over 2 clients, 2 warehouses, 3 products; one launches late,
    one goes dormant, one never sells."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-06", periods=n_weeks, freq="W-MON")
    frames = []
    for i, (c, w, p) in enumerate([(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 0), (0, 0, 1), (1, 0, 1)]):
        s = rng.poisson(3 + i, size=n_weeks).astype(float) * (rng.random(n_weeks) < 0.7)
        if i == 3:
            s[:120] = 0  # launches late
        if i == 4:
            s[90:] = 0  # goes dormant
        if i == 5:
            s[:] = 0  # never sells
        frames.append(pd.DataFrame({"Client": c, "Warehouse": w, "Product": p, "date": dates,
                                    "sales": s, "price": 10.0, "price_was_missing": False}))
    return pd.concat(frames, ignore_index=True).sort_values(KEY + ["date"], ignore_index=True)


def test_v2_features_do_not_leak():
    df = _toy_df()
    keys, sales = sales_matrix(df)
    o = 130
    base = v2_extra_features(keys, sales, [o], HORIZONS, GROUPS)
    corrupted = sales.copy()
    corrupted[:, o + 1:] = 999999.0
    after = v2_extra_features(keys, corrupted, [o], HORIZONS, GROUPS)
    pd.testing.assert_frame_equal(base, after)
    assert base.shape[1] >= 25


def test_annual_features_are_target_aligned_at_every_horizon():
    df = _toy_df()
    keys, sales = sales_matrix(df)
    o = 150
    x = v2_extra_features(keys, sales, [o], HORIZONS, ("annual",))
    n = len(keys)
    for h in HORIZONS:
        rows = x.iloc[(h - 1) * n: h * n]
        np.testing.assert_allclose(rows["ly_same_week"], sales[:, o + h - 52])
        np.testing.assert_allclose(rows["ly2_same_week"], sales[:, o + h - 104])


def test_activity_counts_from_first_sale_and_dormancy():
    df = _toy_df()
    keys, sales = sales_matrix(df)
    o = 130
    first, last = first_last_sale(sales, o)
    x = v2_extra_features(keys, sales, [o], [1], ("activity",))
    seg = x["segment_fs"].astype(str).to_numpy()
    never = np.isnan(first)
    assert (seg[never] == "unlaunched").all() and never.sum() == 1
    assert np.isnan(x["weeks_since_last_sale"].to_numpy()[never]).all()
    # the late launcher's zero rate excludes its 120 pre-launch weeks
    late = (first >= 120) & ~never
    assert late.sum() == 1
    j = np.flatnonzero(late)[0]
    s = sales[j, int(first[j]): o + 1]
    assert x["zero_rate_since_first_sale"].iloc[j] == pytest.approx((s == 0).mean())
    d = dormant_mask(sales, o, 13)
    assert d.sum() == 1 and (o - last[d] >= 13).all() and not d[never].any()


def test_jan_multiplier_uses_only_past_complete_windows():
    df = _toy_df()
    _, sales = sales_matrix(df)
    mult, n = jan_adjusted_multiplier(sales, 54)
    assert n == 0 and (mult == 1).all()  # 54-52-3 < 0: no prior year
    o = 160
    mult, n = jan_adjusted_multiplier(sales, o)
    assert n == 3  # p = 108, 56, 4 (MA4 needs p - 3 >= 0)
    corrupted = sales.copy()
    corrupted[:, o + 1:] = 999999.0
    np.testing.assert_allclose(jan_adjusted_multiplier(corrupted, o)[0], mult)
    # hand-check the portfolio part for a series with no own ratio (never sells)
    port = np.mean([sales[:, p + 1: p + 14].mean(1).sum() / sales[:, p - 3: p + 1].mean(1).sum()
                    for p in (o - 52, o - 104, o - 156)])
    assert mult[np.flatnonzero(sales.sum(1) == 0)[0]] == pytest.approx(port)


def test_v2_without_groups_reproduces_frozen_forecast():
    df = _toy_df()
    fc0, _, _ = frozen_forecast(df, 150)
    fc1, _, _ = v2_forecast(Data(df), 150)
    np.testing.assert_allclose(fc0["lgbm"].to_numpy(), fc1["lgbm"].to_numpy())
    np.testing.assert_allclose(fc0["blend"].to_numpy(), fc1["blend"].to_numpy())


def test_v2_all_groups_and_gate_run():
    df = _toy_df()
    fc, _, info = v2_forecast(Data(df), 150, groups=GROUPS, gate_weeks=13)
    assert info["groups"] == list(GROUPS)
    keys, sales = sales_matrix(df)
    d = np.tile(dormant_mask(sales, 150, 13), 13)
    assert (fc["lgbm"].to_numpy()[d] == 0).all()
