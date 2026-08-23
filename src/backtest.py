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
    """Flat per-series forecast: mean of the last `window` training weeks."""
    return (
        train.sort_values("date")
        .groupby(KEY)["sales"]
        .apply(lambda s: s.tail(window).mean())
        .rename("forecast")
        .reset_index()
    )
