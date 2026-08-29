import pandas as pd
import pytest

from src.backtest import (
    ma12_forecast,
    score_forecast,
    seasonal_naive_forecast,
    split_last_horizon,
    zero_forecast,
)
from src.data import KEY


def _weekly_series(client, warehouse, product, start, sales_values):
    dates = pd.date_range(start, periods=len(sales_values), freq="W-MON")
    return pd.DataFrame({
        "Client": client,
        "Warehouse": warehouse,
        "Product": product,
        "date": dates,
        "sales": sales_values,
    })


def test_split_last_horizon_boundary():
    df = _weekly_series(0, 0, 0, "2020-01-06", list(range(20)))
    train, test = split_last_horizon(df, horizon=5)
    assert len(train) == 15
    assert len(test) == 5
    assert train["date"].max() < test["date"].min()
    assert test["date"].nunique() == 5


def test_ma12_forecast_computes_correct_mean():
    df = _weekly_series(0, 0, 0, "2020-01-06", [10, 20, 30, 40])
    forecast = ma12_forecast(df, window=2)
    assert forecast.loc[0, "forecast"] == pytest.approx((30 + 40) / 2)


def test_ma12_forecast_window_1_is_naive_last():
    df = _weekly_series(0, 0, 0, "2020-01-06", [10, 20, 30])
    forecast = ma12_forecast(df, window=1)
    assert forecast.loc[0, "forecast"] == pytest.approx(30)


def test_ma12_forecast_is_per_series():
    a = _weekly_series(0, 0, 0, "2020-01-06", [10, 10])
    b = _weekly_series(0, 0, 1, "2020-01-06", [100, 100])
    df = pd.concat([a, b], ignore_index=True)
    forecast = ma12_forecast(df, window=2).set_index(KEY)
    assert forecast.loc[(0, 0, 0), "forecast"] == pytest.approx(10)
    assert forecast.loc[(0, 0, 1), "forecast"] == pytest.approx(100)


def test_zero_forecast_is_zero_for_every_series():
    df = _weekly_series(0, 0, 0, "2020-01-06", [10, 20])
    forecast = zero_forecast(df)
    assert (forecast["forecast"] == 0).all()
    assert len(forecast) == 1  # one row per series, not per week


def test_seasonal_naive_forecast_looks_up_correct_lag():
    values = list(range(60))  # 60 weeks so a 52-week lookback lands inside history
    df = _weekly_series(0, 0, 0, "2020-01-06", values)
    test = df.iloc[-5:][KEY + ["date"]].reset_index(drop=True)
    forecast = seasonal_naive_forecast(df, test, lag_weeks=52)
    # test's first row is week index 55; 52 weeks earlier is week index 3
    assert forecast.loc[0, "forecast"] == pytest.approx(values[55 - 52])


def test_score_forecast_detects_row_count_mismatch():
    df = _weekly_series(0, 0, 0, "2020-01-06", [10, 20, 30])
    train, test = split_last_horizon(df, horizon=1)
    forecast = ma12_forecast(train, window=1)
    bad_forecast = pd.concat([forecast, forecast], ignore_index=True)  # duplicate key
    with pytest.raises(AssertionError):
        score_forecast(test, bad_forecast, horizon=1)


def test_score_forecast_returns_merged_frame_on_valid_input():
    df = _weekly_series(0, 0, 0, "2020-01-06", [10, 20, 30])
    train, test = split_last_horizon(df, horizon=1)
    forecast = ma12_forecast(train, window=1)
    scored = score_forecast(test, forecast, horizon=1)
    assert scored.loc[0, "forecast"] == pytest.approx(20)
    assert scored.loc[0, "sales"] == 30
