import pandas as pd

KEY = ["Client", "Warehouse", "Product"]


def load_long(sales_path, price_path):
    """Read the two wide CSVs, melt to long, merge into one faithful long frame."""
    sales = pd.read_csv(sales_path).melt(id_vars=KEY, var_name="date", value_name="sales")
    price = pd.read_csv(price_path).melt(id_vars=KEY, var_name="date", value_name="price")

    df = sales.merge(price, on=KEY + ["date"], how="left", validate="one_to_one")

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(KEY + ["date"]).reset_index(drop=True)


def clean(df):
    """Forward-fill price within each series; flag where it was originally missing."""
    df = df.copy()
    df["price_was_missing"] = df["price"].isna()
    df["price"] = df.groupby(KEY)["price"].ffill()
    return df
