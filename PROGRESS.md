# VN1 Demand Forecasting — Progress

## Current phase
VN1 forecasting pipeline development, Month 1.
Primary technical reference: Joseph & Tackes, *Modern Time Series
Forecasting with Python*.

---

## VN1 project progress

### Completed
- [DONE] Week 1 Session 1 — repo + conda environment (vn1, Python 3.11).
- [DONE] Week 1 Session 2 — data pipeline built and verified end-to-end:
  - Raw sales & price inspected; 15,053 Client|Warehouse|Product series.
  - Weekly grid validated (170 weeks, Mondays, 2020-07-06 → 2023-10-02).
  - Sales & price melted wide→long; dates → datetime; sorted by key+date.
  - Merged with one-to-one validation.
  - Price forward-filled within each series; leading NaNs preserved;
    price_was_missing flag kept.
  - Processed long frame saved: data/processed/vn1_long.parquet
    (2,559,010 rows; dtype preserved).
  - Basic sparsity / zero-rate analysis completed.

- [DONE] Week 1 Session 3 — zero-structure EDA + first scored baseline:
  - Per-series zero table (mean_sales, zero_rate, weeks) confirms CONTEXT
    facts exactly (mean 0.718, median 0.835, max 0.9941).
  - Dead-series check: 0 series with zero_rate == 1.0, as expected.
  - Lifecycle split of zero-rows: leading (pre-launch) 1,139,262 / trailing
    (discontinued) 230,029 / interior (intermittent) 467,146. Sums to
    1,836,437, matching Session 2's pre-ffill price-NaN count exactly
    (price is NaN precisely on zero-sales weeks) — good cross-check.
  - Local 13-week holdout built (train 157 wks: 2020-07-06 -> 2023-07-03;
    test 13 wks: 2023-07-10 -> 2023-10-02).
  - **MA12 baseline score (official VN1 metric): 0.5163** — first number
    to beat.
  - Notebook: notebooks/03_session3.ipynb.
  - Known issue: matplotlib patch rendering (ax.bar/ax.hist/ax.add_patch/
    savefig) crashes the kernel in the vn1 env (matplotlib 3.11.1 + numpy
    2.4.6 + freetype 2.14.3, conda-forge builds). Plain line plots
    (ax.plot) work. Worked around with a text-based bucketed distribution
    instead of a histogram. Not yet fixed — revisit when plotting is
    actually needed (e.g. residual/forecast plots).

---

## Learning progress

### Covered
Chapter 1 foundations; pandas time-series basics; datetime handling;
long vs wide; melt/pivot; sorting time series correctly; missing data vs
zero demand; basic imputation concepts; lag/shift features; resampling;
moving-average baseline concept; intermittent/sparse-demand intuition.

### Next (in service of Week-3 feature engineering)
Rolling windows (incl. why shift(1) before rolling avoids leakage), then
expanding windows, then remaining Chapter 2 concepts.

---

## Current benchmark
MA12 — mean of the latest 12 historical weeks per series, repeated as a
flat forecast across the 13-week horizon. Scored on the real 13-week
holdout with the official metric: **0.5163**. A project benchmark (not an
official rule) that later models must beat.

---

## Next VN1 work

### Immediate — Week 1 Session 4
1. Targeted EDA of active vs intermittent series (use the lifecycle split
   from Session 3 to compare behavior across segments).
2. Create forecasting features (lags, rolling windows with shift(1) first
   to avoid leakage, calendar features).
3. Train the first global ML model (LightGBM); compare against the MA12
   baseline (0.5163) on the same 13-week holdout.

---

## Important open questions
- How much zero demand is genuine intermittency vs censored (stockout)
  demand cannot be identified directly — VN1 has no inventory file.
- Price behavior is inferred from transaction structure; revisit if better
  price information appears.
- Best treatment of highly intermittent series is an empirical question to
  settle through validation.

---

## Deferred, not abandoned
Not required for the first VN1 delivery: deep learning, foundation
forecasting models, advanced ensembling, advanced probabilistic methods.
Part of the roadmap; introduce when conceptually relevant.

---

## Single next action
Session 4 — targeted EDA of active vs intermittent series, then build
lag/rolling features (shift(1) before rolling) and train the first global
LightGBM model against the MA12 baseline (0.5163).

## Session-end rule
End every session by updating this file: what got done, the latest score,
and the exact single next action.
