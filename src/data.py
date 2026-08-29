import numpy as np
import pandas as pd

KEY = ["Client", "Warehouse", "Product"]


def load_long(sales_path, price_path):
    """Read the two wide CSVs, melt to long, merge into one faithful long frame."""
    sales = pd.read_csv(sales_path).melt(id_vars=KEY, var_name="date", value_name="sales")
    price = pd.read_csv(price_path).melt(id_vars=KEY, var_name="date", value_name="price")

    # validate="one_to_one" only rules out duplicate keys on either side — it
    # does not catch a key+date present in one file but missing from the
    # other. Check that explicitly before committing to the left merge.
    check = sales.merge(price, on=KEY + ["date"], how="outer", indicator=True)
    mismatched = check["_merge"] != "both"
    assert not mismatched.any(), (
        f"{mismatched.sum()} Client|Warehouse|Product|date rows appear in only "
        "one of sales/price — raw files don't cover the same series and weeks"
    )

    df = sales.merge(price, on=KEY + ["date"], how="left", validate="one_to_one")

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(KEY + ["date"]).reset_index(drop=True)


def clean(df):
    """Forward-fill price within each series; flag where it was originally missing."""
    df = df.copy()
    df["price_was_missing"] = df["price"].isna()
    df["price"] = df.groupby(KEY)["price"].ffill()
    return df


def validate_long(df):
    """Sanity-check the long frame: complete grid, no duplicate keys, valid sales.

    Raises AssertionError with a descriptive message on the first violation.
    """
    assert not df.duplicated(KEY + ["date"]).any(), "duplicate Client|Warehouse|Product|date rows"

    n_dates = df["date"].nunique()
    assert df.groupby(KEY)["date"].nunique().eq(n_dates).all(), (
        "not every series has an observation for every date (incomplete grid)"
    )

    assert df["sales"].notna().all(), "sales has NaN"
    assert (df["sales"] >= 0).all(), "negative sales"
    assert np.isfinite(df["sales"]).all(), "non-finite sales"

    if "price" in df.columns:
        priced = df["price"].notna()
        assert (df.loc[priced, "price"] >= 0).all(), "negative price"
        assert np.isfinite(df.loc[priced, "price"]).all(), "non-finite price"

    if "price_was_missing" in df.columns:
        assert df["price_was_missing"].dtype == bool, "price_was_missing is not boolean"

    # Sorted within each series, not globally — dates repeat across series
    # since the frame is sorted by key then date.
    assert df.groupby(KEY)["date"].apply(lambda s: s.is_monotonic_increasing).all(), (
        "dates not sorted within at least one series"
    )
