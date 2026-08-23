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

### Current — Week 1 Session 3
Zero-structure EDA + first scored baseline.

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
flat forecast across the 13-week horizon. A project benchmark (not an
official rule) that later models must beat.

---

## Next VN1 work

### Immediate — Session 3 (both parts)
1. Build the per-series zero table: mean_sales, zero_rate, weeks per
   series. Promote the prototype already in notebooks/01_practice.ipynb.
2. Sanity-check against CONTEXT facts: expect 0 dead series
   (zero_rate == 1.0; verified — every series has >=1 sale, max
   zero_rate = 0.9941).
3. Classify each series' zeros into four categories via first & last
   positive-sales week: leading/pre-launch, trailing/discontinued,
   interior/intermittent, always-zero.
4. Build & verify the local 13-week validation setup.
5. Score MA12 with the official metric on the holdout; record the number.

### After Session 3
6. Targeted EDA of active vs intermittent series.
7. Create forecasting features (Week 3).
8. Train the first global ML model (LightGBM); compare against MA12.

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
Session 3 — build the per-series zero table (promote from
01_practice.ipynb), then score MA12 on vn1_long.parquet with the official
metric and record the baseline number.

## Session-end rule
End every session by updating this file: what got done, the latest score,
and the exact single next action.
