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

- [DONE] Week 1 Session 4 — segment EDA, leakage-safe features, first
  global LightGBM. **Note:** the first pass had two real methodology
  problems, caught by external review, not self-caught — both are
  corrected below, and the corrected (honest) numbers are what's reported.
  See DECISIONS.md D008 for the full account.
  - Segment EDA computed from TRAIN-only stats (never touches the
    holdout): active (zero_rate<0.3, n=2,006, mean_sales 65.7) /
    intermittent (0.3-0.9, n=6,450, mean_sales 7.9) / sparse (>=0.9,
    n=6,597, mean_sales 0.26). Explicit thresholds, analytical grouping
    only — not a causal claim, same discipline as Session 3's zero
    positions.
  - **DECISIONS.md D006** — direct/horizon-as-feature multi-step strategy
    (`src/features.py::build_direct_horizon_matrix`): one row per
    (series, origin, horizon), every lag/rolling/price feature computed
    from data at-or-before the origin only, horizon (1-13) an explicit
    feature. Chosen over recursive forecasting to avoid a compounding-
    error feedback loop and because it's easier to verify leakage-free by
    construction. A dedicated test (`test_build_direct_horizon_matrix_
    does_not_leak_future_values`) corrupts all data after an origin and
    asserts that origin's features are byte-identical — the strongest
    check available for this guarantee.
  - **DECISIONS.md D008, correction #1 — purged fit/validation split.**
    The first pass sampled "the last 3 origins" as internal validation
    without checking for overlap: fit-origin targets reached index 141
    (origin 128 + horizon 13) while validation started at origin 132 —
    fit was trained on labels from calendar dates the validation set was
    nominally checking generalization to. Fixed with
    `src/features.py::purge_fit_origins()`, which drops any fit origin
    where `origin + max(horizons) >= min(val_origins)` (here: drops
    origins 120/124/128, leaving fit_origins at 52-116), with
    `max(fit_origins) + max(horizons) < min(val_origins)` asserted
    explicitly. Two new tests cover it. This never touched the real
    holdout, but it did mean early stopping wasn't validated against a
    genuinely clean signal.
  - Training origins (post-purge): every 4th week, index 52-116 (17
    origins; enough lookback for a 52-week lag). Validation: origins
    132/136/140. Eval: origin 156 (last training week), verified to
    predict exactly the 13 real test dates. 3.3M fit rows / 587K
    validation rows / 195,689 eval rows (15,053 series x 13 horizons).
  - Features (21, one changed from the first pass): horizon;
    lag_1/2/3/4/8/12/26/52; roll_mean_4/8/12/26; roll_std_12;
    zero_rate_12; price_at_origin (already-causal from Session 2's
    ffill — safe to read directly); price_was_missing_at_origin;
    segment (expanding zero-rate up to origin, not full-series);
    target_month, **target_week_sin/target_week_cos** (cyclical, not a
    raw 1-53 integer — week 52 and week 1 are calendar-adjacent, a raw
    integer put them maximally far apart), target_quarter.
  - **DECISIONS.md D007, correction #2 — objective selected on internal
    validation only, not the holdout.** The first pass picked `regression`
    (L2) over `tweedie`/`regression_l1` by comparing scores computed
    directly on the true 13-week holdout (0.7414 / 0.6145 / 0.4955) —
    that uses the holdout for model selection, which optimistically
    biases the reported number. Redone against the purged internal
    validation set only: tweedie 0.8253, regression_l1 0.6430,
    **regression 0.5052** — same ranking, L2 still wins, so the choice
    was right, just reached the wrong way. `random_state=42` and full
    `LGBMRegressor` params now recorded for reproducibility
    (lightgbm 4.7.0).
  - **LightGBM score (official VN1 metric): 0.4989** — a genuinely honest
    single look at the holdout with the internally-selected model,
    labeled a *development-holdout* result rather than a pristine one
    (the holdout was already inspected during the flawed first pass's
    3-way comparison — see D008). This is worse than the first pass's
    0.4955, exactly as expected once selection bias is removed. **Revised
    headline: LightGBM (0.4989) is essentially tied with MA4 (0.4987),
    not conclusively better than it** — the earlier "beats the whole
    suite" claim doesn't survive the correction. Breakdown: total_actual
    3,508,827, total_predicted 3,423,361, signed_bias +85,466 (~2.4% net
    under-forecast — still tighter than MA12's own +154,579), abs_error_
    sum 1,665,184.
  - Segment analysis, now properly quantified (not just a bare score —
    the official score is not a simple average of segment scores):
    active — 45.4% of demand, 34.9% of abs error, bias −61,726 (net
    over-forecast), score 0.4031. intermittent — 44.0% of demand, 43.5%
    of abs error, bias +57,864, score 0.5065. sparse — 10.6% of demand,
    **21.6% of abs error** (a real, supported ~2x overrepresentation,
    not eyeballed), bias +89,328, score 1.2115. Sparse series are
    disproportionately hard, flagged as a Session 5 candidate rather
    than fixed here.
  - Top feature importances: price_at_origin, lag_1, target_week_sin,
    target_week_cos, target_month, lag_12, lag_26, lag_52, lag_2, lag_8.
    Both cyclical week components and month rank highly, and lag_52 is
    still present — **consistent with a seasonality hypothesis, not a
    confirmed finding**: seasonal-naive-52 itself scores poorly (1.0498),
    so this shouldn't be overclaimed.
  - New: `src/features.py` (zero_rate_table, classify_segment,
    build_direct_horizon_matrix, purge_fit_origins). New:
    `tests/test_features.py`, 9 tests, all passing. Full suite: 33/33
    passing.
  - `lightgbm` added to `environment.yml` and installed in `vn1`.
  - Notebook: notebooks/04_session4.ipynb, rewritten and executed
    end-to-end for real after the corrections above. README.md updated
    (was stale, still said "Scores to follow").
  - **Process lesson (DECISIONS.md D008, permanent, not revisitable):**
    never use the true final holdout for model/objective/hyperparameter
    selection, even once — only for a single final evaluation per major
    decision. Use purged internal validation for everything else.

---

## Learning progress

### Covered
Chapter 1 foundations; pandas time-series basics; datetime handling;
long vs wide; melt/pivot; sorting time series correctly; missing data vs
zero demand; basic imputation concepts; lag/shift features; resampling;
moving-average baseline concept; intermittent/sparse-demand intuition;
rolling windows and why shift(1)-before-rolling avoids leakage; multi-step
forecasting strategies (recursive vs. direct/horizon-as-feature, and why
naive lag features break for horizon>1 on a continuous series); regression
framing of a forecasting problem; gradient boosting basics via LightGBM;
global (all-series-one-model) forecasting models; loss-function choice as
an empirical question, not a fixed rule (Tweedie vs. L2 vs. L1); purged
time-series cross-validation (why overlapping direct-multi-horizon targets
contaminate an unpurged internal split); the difference between "clean
internal validation for selection" and "one honest look at a true holdout"
— and how easy it is to blur that line without external review.

### Next
Expanding windows; per-segment/hurdle modeling for intermittent demand
(ties directly into the sparse-segment open question above); LightGBM
hyperparameter tuning fundamentals; remaining Chapter 2 concepts.

---

## Current benchmark
**LightGBM (direct multi-horizon, L2): 0.4989** — a development-holdout
result (see DECISIONS.md D008), essentially tied with MA4 (0.4987), not
conclusively better. MA12 (0.5163) stays tracked separately for external
comparability (matches the real VN1 competition's own baseline). Later
work must beat 0.4989 — and now that a full session's headline number has
already been revised once by external review, treat "beats the baseline"
claims cautiously until independently re-verified.

---

## Next VN1 work

### Immediate — Week 1 Session 5
1. The sparse segment (zero_rate>=0.9, 6,597 series, 10.6% of demand but
   21.6% of total abs error) scores far worse (1.2115) than active
   (0.4031) or intermittent (0.5065) — investigate whether a dedicated
   approach helps: a hurdle/two-stage model (classify zero-vs-nonzero,
   then regress the nonzero magnitude), a segment-specific model, or
   accepting this as an inherent property of very sparse demand and
   focusing effort elsewhere.
2. Hyperparameter tuning of the current LightGBM (num_leaves, learning
   rate, min_child_samples, n_estimators) — the first model used
   reasonable-but-unturned defaults, and must be tuned against internal
   validation only (D008), never the true holdout.
3. Consider a denser origin sample (currently every 4 weeks, post-purge
   52-116) or an expanded lookback/rolling feature set, now that the
   pipeline and leakage tests exist to check any change safely.
4. Longer-term: revisit D006 (recursive vs. direct) and D007 (Tweedie vs.
   L2) if either the segment split or a richer feature set changes the
   picture — both were evidence-based calls on the *first* model, not
   permanent. D008 (purge discipline, never select on the true holdout)
   is permanent and applies to all of the above.

---

## Important open questions
- How much zero demand is genuine intermittency vs censored (stockout)
  demand cannot be identified directly — VN1 has no inventory file.
- Price behavior is inferred from transaction structure; revisit if better
  price information appears.
- Best treatment of highly intermittent series is an empirical question to
  settle through validation. Session 4 gave a first concrete answer:
  the sparse segment (zero_rate>=0.9) scores 1.2115 vs. 0.40-0.51 for the
  other two segments, and contributes 21.6% of total absolute error from
  only 10.6% of demand — a real, evidence-based sign this deserves
  dedicated treatment rather than being folded into one global model
  as-is.

---

## Deferred, not abandoned
Not required for the first VN1 delivery: deep learning, foundation
forecasting models, advanced ensembling, advanced probabilistic methods.
Part of the roadmap; introduce when conceptually relevant.

---

## Single next action
Session 5 — investigate the sparse segment specifically (hurdle/two-stage
model or segment-specific approach), and/or tune the current LightGBM's
hyperparameters (against internal validation only, per D008), using the
0.4989 direct-multi-horizon model as the baseline to beat.

## Session-end rule
End every session by updating this file: what got done, the latest score,
and the exact single next action.
