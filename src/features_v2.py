"""LightGBM-v2 feature groups (DECISIONS.md D014).

Each group adds columns to the frozen feature matrix
(`build_direct_horizon_matrix`). Every value is computed from sales at or
before the origin only; the leakage tests corrupt everything after the origin
and require every column to stay identical.

Groups (ablation order, fixed in D014):
  annual     target-aligned annual history: y[o+h-52], y[o+h-51], y[o+h-53],
             their mean, and y[o+h-104] where it exists
  activity   weeks since last / first sale, non-zero counts, mean non-zero
             size, sold in the same window last year, zero rate and segment
             counted from the first sale
  aggregates client / warehouse / product totals: level, growth, and the
             group's target-week movement last year
  trend      MA4/MA12, last 13 weeks vs. the same 13 weeks last year
             (series and portfolio)
  identity   Client, Warehouse, Product as categorical features
The dormancy gate (group 6) is post-processing, see `dormant_mask`.

Row order matches `build_direct_horizon_matrix`: origins, then horizons, then
series in sorted KEY order.
"""
import numpy as np
import pandas as pd

from .data import KEY

GROUPS = ("annual", "activity", "aggregates", "trend", "identity")
SEGMENT_FS_DTYPE = pd.CategoricalDtype(["active", "intermittent", "sparse", "unlaunched"])
AGG_LEVELS = ("Client", "Warehouse", "Product")


def _div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=b != 0)
    return out


def sales_matrix(df):
    """(keys, sales array) in the same sorted-KEY order the frozen builder uses."""
    dates = np.sort(df["date"].unique())
    wide = df.pivot(index=KEY, columns="date", values="sales").reindex(columns=dates)
    return wide.index.to_frame(index=False), wide.to_numpy(dtype=float)


def first_last_sale(sales, o):
    """Index of the first and last week with a sale at or before origin o (NaN if none)."""
    sold = sales[:, : o + 1] > 0
    any_sale = sold.any(axis=1)
    first = np.where(any_sale, sold.argmax(axis=1), np.nan)
    last = np.where(any_sale, o - sold[:, ::-1].argmax(axis=1), np.nan)
    return first, last


def dormant_mask(sales, o, n_weeks=13):
    """Series that have sold before but not in the last `n_weeks` weeks up to o.
    Never-sold (not yet launched) series are not dormant."""
    _, last = first_last_sale(sales, o)
    return ~np.isnan(last) & (o - np.nan_to_num(last, nan=-1e9) >= n_weeks)


def _per_origin(sales, keys, o, groups, group_codes):
    """Origin-level features (identical for every horizon) and a function
    giving horizon-specific features."""
    n = sales.shape[0]
    base = {}
    if "activity" in groups:
        first, last = first_last_sale(sales, o)
        base["weeks_since_last_sale"] = o - last
        base["weeks_since_first_sale"] = o - first
        for w in (4, 13, 26, 52):
            base[f"nz_count_{w}"] = (sales[:, o - w + 1: o + 1] > 0).sum(axis=1).astype(float)
        win = sales[:, o - 51: o + 1]
        base["mean_nz_size_52"] = _div(win.sum(axis=1), (win > 0).sum(axis=1))
        base["sold_same_window_ly"] = (sales[:, o - 51: o - 38] > 0).any(axis=1).astype(float)
        idx = np.arange(o + 1)
        since = idx[None, :] >= np.nan_to_num(first, nan=o + 1)[:, None]
        n_since = since.sum(axis=1)
        zeros_since = ((sales[:, : o + 1] == 0) & since).sum(axis=1)
        zr_fs = _div(zeros_since, n_since)
        base["zero_rate_since_first_sale"] = zr_fs
        seg = np.where(np.isnan(zr_fs), "unlaunched",
                       np.where(zr_fs >= 0.9, "sparse", np.where(zr_fs < 0.3, "active", "intermittent")))
        base["segment_fs"] = pd.Categorical(seg, dtype=SEGMENT_FS_DTYPE)
    if "trend" in groups:
        ma4 = sales[:, o - 3: o + 1].mean(axis=1)
        ma12 = sales[:, o - 11: o + 1].mean(axis=1)
        base["ma4_over_ma12"] = _div(ma4, ma12)
        if o - 64 >= 0:
            now13 = sales[:, o - 12: o + 1].sum(axis=1)
            ly13 = sales[:, o - 64: o - 51].sum(axis=1)
            base["yoy_13w"] = _div(now13, ly13)
            base["portfolio_yoy_13w"] = np.full(n, _div(now13.sum(), ly13.sum()))
        else:
            base["yoy_13w"] = np.full(n, np.nan)
            base["portfolio_yoy_13w"] = np.full(n, np.nan)
    agg_now = {}
    if "aggregates" in groups:
        for lvl in AGG_LEVELS:
            codes, gs = group_codes[lvl]
            g4 = gs[:, o - 3: o + 1].mean(axis=1)
            g13 = gs[:, o - 12: o + 1].mean(axis=1)
            base[f"{lvl.lower()}_ma4"] = g4[codes]
            base[f"{lvl.lower()}_growth_4_13"] = _div(g4, g13)[codes]
            ref = gs[:, o - 64: o - 51].mean(axis=1) if o - 64 >= 0 else None
            agg_now[lvl] = (codes, gs, ref)
    if "identity" in groups:
        for lvl in AGG_LEVELS:
            base[f"id_{lvl.lower()}"] = pd.Categorical(keys[lvl].to_numpy(),
                                                       categories=np.sort(keys[lvl].unique()))

    def horizon_feats(h):
        out = {}
        if "annual" in groups:
            y52, y51, y53 = sales[:, o + h - 52], sales[:, o + h - 51], sales[:, o + h - 53]
            out["ly_same_week"] = y52
            out["ly_week_after"] = y51
            out["ly_week_before"] = y53
            out["ly_3wk_mean"] = (y51 + y52 + y53) / 3
            out["ly2_same_week"] = sales[:, o + h - 104] if o + h - 104 >= 0 else np.full(n, np.nan)
        for lvl, (codes, gs, ref) in agg_now.items():
            out[f"{lvl.lower()}_ly_target_index"] = (
                _div(gs[:, o + h - 52], ref)[codes] if ref is not None else np.full(n, np.nan))
        return out

    return base, horizon_feats


def group_sales(keys, sales):
    """Per aggregation level: (series -> group code, group-total sales array)."""
    out = {}
    for lvl in AGG_LEVELS:
        codes, uniq = pd.factorize(keys[lvl], sort=True)
        gs = np.zeros((len(uniq), sales.shape[1]))
        np.add.at(gs, codes, sales)
        out[lvl] = (codes, gs)
    return out


def v2_extra_features(keys, sales, origins, horizons, groups):
    """Extra columns for `groups`, row-aligned with build_direct_horizon_matrix."""
    groups = tuple(g for g in GROUPS if g in groups)
    if not groups:
        return pd.DataFrame(index=range(len(origins) * len(horizons) * len(keys)))
    gc = group_sales(keys, sales) if "aggregates" in groups else None
    blocks = []
    for o in origins:
        assert o >= 52, "origin too early for the annual features (o+h-53 must be >= 0)"
        base, hf = _per_origin(sales, keys, o, groups, gc)
        for h in horizons:
            assert h <= 51, "annual features would read past the origin"
            blocks.append(pd.DataFrame({**base, **hf(h)}))
    out = pd.concat(blocks, ignore_index=True)
    for c in out.columns:
        if out[c].dtype == float:
            out[c] = out[c].astype(np.float32)
    return out
