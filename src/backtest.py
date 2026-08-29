import pandas as pd

from .data import KEY

HORIZON = 13


def split_last_horizon(df, horizon=HORIZON):
    """Hold out the last `horizon` weeks as the test set; everything before is train."""
    dates = df["date"].sort_values().unique()
    cutoff = dates[-horizon]
    train = df[df["date"] < cutoff]
    test = df[df["date"] >= cutoff]
    return train, test


def ma12_forecast(train, window=12):
    """Flat per-series forecast: mean of the last `window` training weeks.

    Despite the name, `window` is not fixed to 12 — window=1 gives the naive
    "repeat last observed value" baseline, window=4/8/26 give other moving
    averages. The name stays MA12 because that's the project's named
    benchmark (window=12, the default).
    """
    forecast = (
        train.sort_values("date")
        .groupby(KEY)["sales"]
        .apply(lambda s: s.tail(window).mean())
        .rename("forecast")
        .reset_index()
    )
    forecast["forecast"] = forecast["forecast"].clip(lower=0)
    return forecast


def zero_forecast(train):
    """Trivial per-series forecast: always predict 0."""
    forecast = train[KEY].drop_duplicates().reset_index(drop=True)
    forecast["forecast"] = 0.0
    return forecast


def seasonal_naive_forecast(df, test, lag_weeks=52):
    """Forecast each test week as the actual value `lag_weeks` earlier.

    Unlike ma12_forecast/zero_forecast, this needs the full history (`df`),
    not just train, because each of the horizon weeks looks back to a
    different historical week — the result is one row per series+date, not
    one flat value per series.
    """
    lookup = test[KEY + ["date"]].copy()
    lookup["lookup_date"] = lookup["date"] - pd.Timedelta(weeks=lag_weeks)

    merged = lookup.merge(
        df[KEY + ["date", "sales"]].rename(columns={"date": "lookup_date"}),
        on=KEY + ["lookup_date"],
        how="left",
    )
    merged["forecast"] = merged["sales"].clip(lower=0)
    return merged[KEY + ["date", "forecast"]]


def score_forecast(test, forecast, horizon=HORIZON):
    """Merge a forecast onto actuals, verify alignment, return the scored frame.

    `forecast` may be flat (one row per series — KEY + ["forecast"]) or
    per-date (one row per series+date — KEY + ["date", "forecast"]);
    the merge key is picked accordingly. Raises if the merge left any test
    row unmatched, duplicated a key, or dropped/added rows — a plain
    isna()-only check misses the last two.
    """
    merge_cols = KEY + ["date"] if "date" in forecast.columns else KEY
    scored = test.merge(forecast, on=merge_cols, how="left")

    assert scored["forecast"].isna().sum() == 0, "every test row should have a forecast"
    assert len(scored) == len(test), "merge changed row count — check for duplicate keys in forecast"
    assert not scored.duplicated(KEY + ["date"]).any(), "duplicate key+date rows after merge"
    assert scored.groupby(KEY).size().eq(horizon).all(), (
        "not every series has exactly `horizon` test rows"
    )

    return scored
