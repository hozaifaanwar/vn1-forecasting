import numpy as np
import pandas as pd
import pytest

from src.data import KEY, clean, load_long, validate_long


def _toy_long(rows):
    """rows: list of (Client, Warehouse, Product, date, sales, price) tuples."""
    return pd.DataFrame(rows, columns=KEY + ["date", "sales", "price"])


def test_clean_does_not_leak_price_across_series():
    # Series (0,0,0) has a valid price on its last row; series (0,0,1),
    # sorted right after it, starts with a NaN price. A groupby-ffill must
    # not carry (0,0,0)'s price into (0,0,1)'s leading NaN.
    df = _toy_long([
        (0, 0, 0, pd.Timestamp("2020-01-06"), 1, 10.0),
        (0, 0, 0, pd.Timestamp("2020-01-13"), 1, 20.0),
        (0, 0, 1, pd.Timestamp("2020-01-06"), 0, np.nan),
        (0, 0, 1, pd.Timestamp("2020-01-13"), 0, np.nan),
    ])
    cdf = clean(df)
    b_prices = cdf.loc[cdf["Product"] == 1, "price"]
    assert b_prices.isna().all(), "series (0,0,1)'s leading NaN should not be filled from (0,0,0)"


def test_clean_preserves_leading_nan_and_fills_interior():
    df = _toy_long([
        (0, 0, 0, pd.Timestamp("2020-01-06"), 0, np.nan),  # leading
        (0, 0, 0, pd.Timestamp("2020-01-13"), 1, 10.0),
        (0, 0, 0, pd.Timestamp("2020-01-20"), 0, np.nan),  # interior -> ffilled to 10
        (0, 0, 0, pd.Timestamp("2020-01-27"), 1, 20.0),
    ])
    cdf = clean(df)
    assert cdf["price"].isna().sum() == 1
    assert pd.isna(cdf.loc[0, "price"])
    assert cdf.loc[2, "price"] == 10.0


def test_clean_price_was_missing_reflects_original_nan_not_post_fill():
    df = _toy_long([
        (0, 0, 0, pd.Timestamp("2020-01-06"), 0, np.nan),
        (0, 0, 0, pd.Timestamp("2020-01-13"), 1, 10.0),
        (0, 0, 0, pd.Timestamp("2020-01-20"), 0, np.nan),
    ])
    cdf = clean(df)
    assert cdf["price_was_missing"].tolist() == [True, False, True]
    assert cdf["price_was_missing"].dtype == bool


def test_validate_long_passes_on_a_clean_frame():
    df = _toy_long([
        (0, 0, 0, pd.Timestamp("2020-01-06"), 1, 10.0),
        (0, 0, 0, pd.Timestamp("2020-01-13"), 2, 20.0),
        (0, 0, 1, pd.Timestamp("2020-01-06"), 0, np.nan),
        (0, 0, 1, pd.Timestamp("2020-01-13"), 3, 5.0),
    ])
    validate_long(df)  # should not raise


def test_validate_long_catches_duplicate_key_date():
    df = _toy_long([
        (0, 0, 0, pd.Timestamp("2020-01-06"), 1, 10.0),
        (0, 0, 0, pd.Timestamp("2020-01-06"), 1, 10.0),  # duplicate key+date
    ])
    with pytest.raises(AssertionError, match="duplicate"):
        validate_long(df)


def test_validate_long_catches_incomplete_grid():
    df = _toy_long([
        (0, 0, 0, pd.Timestamp("2020-01-06"), 1, 10.0),
        (0, 0, 0, pd.Timestamp("2020-01-13"), 1, 10.0),
        (0, 0, 1, pd.Timestamp("2020-01-06"), 1, 10.0),
        # (0,0,1) is missing the 2020-01-13 week — incomplete grid
    ])
    with pytest.raises(AssertionError, match="incomplete grid"):
        validate_long(df)


def test_validate_long_catches_negative_sales():
    df = _toy_long([(0, 0, 0, pd.Timestamp("2020-01-06"), -1, 10.0)])
    with pytest.raises(AssertionError, match="negative sales"):
        validate_long(df)


def test_validate_long_catches_negative_price():
    df = _toy_long([(0, 0, 0, pd.Timestamp("2020-01-06"), 1, -5.0)])
    with pytest.raises(AssertionError, match="negative price"):
        validate_long(df)


def test_load_long_catches_mismatched_raw_files(tmp_path):
    sales_path = tmp_path / "sales.csv"
    price_path = tmp_path / "price.csv"
    # price file is missing the 2020-01-13 column entirely
    sales_path.write_text("Client,Warehouse,Product,2020-01-06,2020-01-13\n0,0,0,1,2\n")
    price_path.write_text("Client,Warehouse,Product,2020-01-06\n0,0,0,10.0\n")

    with pytest.raises(AssertionError, match="only one of sales/price"):
        load_long(sales_path, price_path)
