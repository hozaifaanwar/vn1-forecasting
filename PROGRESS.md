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
  - Merged with one-to-one validation, plus an explicit completeness check
    (outer-merge + indicator) since validate="one_to_one" alone doesn't
    catch a key+date present in one raw file but missing from the other.
  - Price forward-filled within each series; leading NaNs preserved;
    price_was_missing flag kept.
  - Processed long frame saved: data/processed/vn1_long.parquet
    (2,559,010 rows; dtype preserved).
  - `src/data.py::validate_long()`: duplicate-key, complete-grid, sales
    non-negative/finite, price non-negative/finite when present,
    price_was_missing is bool, sorted-within-series checks. Run in the
    notebook and in the build script below.
  - `scripts/build_dataset.py` — reproducible one-command build
    (load_long → clean → validate_long → save parquet), not dependent on
    running notebook cells in order. Resolves the repo root itself.
  - Basic sparsity / zero-rate analysis completed.

- [DONE] Week 1 Session 3 — zero-structure EDA + scored baselines:
  - Per-series zero table (mean_sales, zero_rate, weeks) confirms CONTEXT
    facts exactly (mean 0.718, median 0.835, max 0.9941).
  - Dead-series check: 0 series with zero_rate == 1.0, as expected.
  - Zero-position split (positional only, deliberately not called
    "lifecycle" — leading/interior/trailing describe *where* a zero sits
    relative to a series' first/last sale, not *why* it happened; VN1 has
    no launch/discontinuation/inventory data to justify a causal label):
    leading 1,139,262 / interior 467,146 / trailing 230,029 / always-zero
    0. Sums to 1,836,437, matching Session 2's pre-ffill price-NaN count
    exactly (price is NaN precisely on zero-sales weeks) — good cross-check.
  - Local 13-week holdout built (train 157 wks: 2020-07-06 -> 2023-07-03;
    test 13 wks: 2023-07-10 -> 2023-10-02).
  - **MA12 baseline score (official VN1 metric): 0.5163** — the tracked
    project benchmark. Verified against the CONTEXT.md formula
    term-for-term (src/metrics.py) and guarded with merge/alignment
    assertions (src/data.py::validate_long, plus src/backtest.py::
    score_forecast's row-count/duplicate/HORIZON-per-series checks) so
    the number isn't just "a function ran and printed something."
    `vn1_score_breakdown()` decomposes it: total_actual 3,508,827,
    total_predicted 3,354,248, signed_bias +154,579 (net under-forecast),
    abs_error_sum 1,657,060.
  - **Baseline suite** (same holdout, same scoring path):
    MA4 0.4987 < MA8 0.5114 < **MA12 0.5163** < naive-last 0.5247 <
    MA26 0.5618 < seasonal-naive-52 1.0498 < zero 2.0000. Zero-forecast
    landing exactly at 2.0000 is a useful sanity anchor on the metric
    formula itself. MA12 remains the tracked benchmark (matches the real
    VN1 competition's own baseline, so comparable externally), but MA4
    actually scores better here — a signal that recency may matter more
    than long averaging for this dataset; worth keeping in mind when
    designing Session 4 lag/rolling features (don't assume longer windows
    are automatically better).
  - Notebook: notebooks/03_session3.ipynb.
  - Both Session 2 and 3 notebooks now resolve the repo root at the top
    (walk up to the CONTEXT.md marker, with a guard against spinning
    forever if it's never found) so they run correctly regardless of
    whether Jupyter was launched from the repo root or notebooks/.
  - Known issue: matplotlib patch rendering (ax.bar/ax.hist/ax.add_patch/
    savefig) crashes the kernel in the vn1 env (matplotlib 3.11.1 + numpy
    2.4.6 + freetype 2.14.3, conda-forge builds). Plain line plots
    (ax.plot) work. Worked around with a text-based bucketed distribution
    instead of a histogram. Not yet fixed — revisit when plotting is
    actually needed (e.g. residual/forecast plots).
  - `pytest` added to `environment.yml` and installed in `vn1`. 24 unit
    tests added (tests/test_data.py, test_metrics.py, test_backtest.py),
    all passing: merge-completeness/duplicate-key rejection, within-series
    price forward-fill + preserved leading NaNs, validate_long's
    price/dtype checks, backtest boundary + forecast alignment, zero/
    MA12/seasonal-naive forecasts, and vn1_score/vn1_score_breakdown
    against hand-calculated examples. Run with `pytest tests/`.
  - Deferred (reasonable but disproportionate for this stage): parquet
    provenance metadata (hashes/timestamps), rolling-origin multi-window
    backtesting (revisit once actually comparing candidate models, not
    baselines).

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
holdout with the official metric: **0.5163**. Matches the real VN1
competition's own baseline (a project benchmark, not an official scoring
rule), so it's the one tracked for external comparability — even though
the Session 3 baseline suite found MA4 scores slightly better (0.4987) on
this specific holdout. Later models must beat 0.5163.

---

## Next VN1 work

### Immediate — Week 1 Session 4
1. Targeted EDA of active vs intermittent series (use the zero-position
   split from Session 3 to compare behavior across segments). Consider
   reporting the baseline suite / VN1 score by segment too, not just
   overall — the official aggregate score stays primary, segment scores
   are diagnostic.
2. Create forecasting features (lags, rolling windows with shift(1) first
   to avoid leakage, calendar features).
   - **Price leakage caution:** the processed frame's `price` column is
     forward-filled and causal, but a model must still not consume the
     *contemporaneous* price of a holdout/forecast-origin week as a
     feature unless it was genuinely known at forecast time. Use lagged
     price or "last known price as of forecast origin" — not the
     current-row price during holdout feature-building.
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
