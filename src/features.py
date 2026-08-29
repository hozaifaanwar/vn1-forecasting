import numpy as np
import pandas as pd

from .data import KEY


def zero_rate_table(df):
    """Per-series mean_sales/zero_rate/weeks — the Session 3 EDA table, formalized.

    Callers should pass train-only data if the result feeds a model or any
    decision that must not see the holdout period.
    """
    return df.groupby(KEY)["sales"].agg(
        mean_sales="mean",
        zero_rate=lambda s: (s == 0).mean(),
        weeks="count",
    )


def classify_segment(zero_rate, active_max=0.3, sparse_min=0.9):
    """Bucket a zero_rate Series into active/intermittent/sparse.

    Purely a threshold on an observed rate — an analytical convenience for
    grouping series with similar behavior, not a claim about *why* a series
    behaves that way (same positional-not-causal discipline as Session 3's
    zero-position labels).
    """
    return pd.Series(
        np.where(
            zero_rate >= sparse_min, "sparse",
            np.where(zero_rate < active_max, "active", "intermittent"),
        ),
        index=zero_rate.index,
        name="segment",
    )


def build_direct_horizon_matrix(
    df,
    origins,
    horizons,
    lookbacks=(1, 2, 3, 4, 8, 12, 26, 52),
    roll_windows=(4, 8, 12, 26),
    segment_active_max=0.3,
    segment_sparse_min=0.9,
):
    """Build a direct multi-horizon supervised table (see DECISIONS.md D006).

    One row per (series, origin, horizon). Every feature is computed from
    data at or before `origin` only — `horizon` is an explicit feature
    rather than folding it into a lag computed on the continuous series,
    which is what would leak real holdout values into "lag" features for
    horizon > 1. Only `target` (and calendar features of `target_date`,
    which are knowable in advance regardless of origin) may reference a
    date after the origin.

    `origins`/`horizons` are integer indices into the sorted unique dates
    in `df`, not dates themselves — e.g. origin=143, horizon=13 means
    "13 weeks after the date at sorted-date-index 143". Caller is
    responsible for keeping origin+max(horizons) within whatever period
    is meant to be visible (see notebooks/04_session4.ipynb for how the
    train/eval split is enforced this way).
    """
    dates = np.sort(df["date"].unique())
    n_dates = len(dates)

    sales_wide = df.pivot(index=KEY, columns="date", values="sales").reindex(columns=dates)
    price_wide = df.pivot(index=KEY, columns="date", values="price").reindex(columns=dates)
    missing_wide = (
        df.pivot(index=KEY, columns="date", values="price_was_missing").reindex(columns=dates)
    )

    keys_df = sales_wide.index.to_frame(index=False)
    sales_arr = sales_wide.to_numpy()
    price_arr = price_wide.to_numpy()
    missing_arr = missing_wide.to_numpy()

    max_lookback = max(lookbacks + roll_windows)
    blocks = []

    for o in origins:
        assert o - max_lookback + 1 >= 0, f"origin {o} too early for the largest lookback"
        assert o + max(horizons) < n_dates, f"origin {o} + max horizon exceeds available dates"

        # Expanding (origin-relative, not full-series) zero rate — using
        # only history up to and including the origin, so this can't leak
        # the series' true future behavior into the segment label.
        expanding_zero_rate = (sales_arr[:, : o + 1] == 0).mean(axis=1)
        segment = classify_segment(
            pd.Series(expanding_zero_rate), segment_active_max, segment_sparse_min
        ).to_numpy()

        lag_feats = {f"lag_{k}": sales_arr[:, o - k + 1] for k in lookbacks}
        roll_means = {
            f"roll_mean_{w}": sales_arr[:, o - w + 1 : o + 1].mean(axis=1) for w in roll_windows
        }
        roll_std_12 = sales_arr[:, o - 11 : o + 1].std(axis=1)
        zero_rate_12 = (sales_arr[:, o - 11 : o + 1] == 0).mean(axis=1)
        price_at_origin = price_arr[:, o]
        missing_at_origin = missing_arr[:, o]

        for h in horizons:
            t = o + h
            target_date = pd.Timestamp(dates[t])
            block = pd.DataFrame({
                **{c: keys_df[c].to_numpy() for c in KEY},
                "origin_date": dates[o],
                "horizon": h,
                "target_date": target_date,
                "target": sales_arr[:, t],
                **lag_feats,
                **roll_means,
                "roll_std_12": roll_std_12,
                "zero_rate_12": zero_rate_12,
                "price_at_origin": price_at_origin,
                "price_was_missing_at_origin": missing_at_origin,
                "segment": segment,
                "target_month": target_date.month,
                # Cyclical encoding, not a raw 1-53 integer: week 52 and week 1
                # are calendar-adjacent, but a raw integer feature puts them
                # as far apart as possible, which is the wrong inductive bias.
                "target_week_sin": np.sin(2 * np.pi * target_date.isocalendar()[1] / 52),
                "target_week_cos": np.cos(2 * np.pi * target_date.isocalendar()[1] / 52),
                "target_quarter": target_date.quarter,
            })
            blocks.append(block)

    return pd.concat(blocks, ignore_index=True)


def purge_fit_origins(origins, val_origins, horizons):
    """Drop any fit origin whose target range would reach into or past the
    first validation origin.

    Standard "purged" time-series validation (cf. de Prado): without this,
    a fit example's target can be the actual value at a calendar date at or
    after where a validation origin starts predicting from — the model is
    then trained on labels the validation metric is nominally checking
    generalization to, so early stopping / model selection against that
    validation set is not a clean proxy for genuine out-of-sample
    performance, even though it never touches the real final holdout.
    """
    max_h = max(horizons)
    min_val = min(val_origins)
    return [o for o in origins if o + max_h < min_val]
