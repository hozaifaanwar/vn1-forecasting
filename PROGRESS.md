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
  - **LightGBM score (official VN1 metric): 0.4989.** Because the final
    holdout influenced earlier development (the flawed first pass's
    3-way objective comparison scored directly against it — see D008),
    **0.4989 is a development-holdout result, not a fully untouched
    estimate of generalization.** It is worse than the first pass's
    0.4955, exactly as expected once selection bias is removed. **Revised
    headline: LightGBM (0.4989) is essentially tied with MA4 (0.4987) —
    it has not demonstrated reliable forecast value yet**, and the
    earlier "beats the whole suite" claim doesn't survive the correction.
    Breakdown: total_actual 3,508,827, total_predicted 3,423,361,
    signed_bias +85,466 (~2.4% net under-forecast — still tighter than
    MA12's own +154,579), abs_error_sum 1,665,184.
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

- [DONE] Week 1 Session 5 — rolling-origin CV, hyperparameter tuning,
  sparse-segment investigation, LightGBM-vs-MA4 origin-level comparison
  (DECISIONS.md D009/D010). **Went through three external review passes
  before anything was committed — each caught a real, distinct problem.**
  Pass 1: no rolling-origin MA4 comparison existed (the "clearly ahead"
  claim rested on one development-holdout point). Pass 2: early stopping
  used `eval_metric="l1"` instead of the VN1 score. Pass 3, the most
  consequential: `eval_metric=vn1_lgb_eval` alone does NOT make early
  stopping exclusive to that metric — LightGBM's own built-in metric for
  the objective (e.g. "l2") stays registered alongside it unless
  `metric="None"` + `first_metric_only=True` are also set. Fixing pass 3
  changed several conclusions from passes 1-2, not just the headline
  number — the numbers below are the final, three-times-corrected ones.
  - `src/features.py::rolling_origin_folds()` — 3 walk-forward, purged
    validation folds, replacing Session 4's single 3-origin split.
  - `src/metrics.py`: `vn1_score_by_group` (scores a group, e.g. a
    rolling-CV origin, separately rather than pooling — pooling lets one
    group's bias cancel another's); `vn1_lgb_eval` (custom LightGBM eval
    computing the real VN1 score); `make_vn1_origin_mean_eval` (a factory
    building an eval function that early-stops on the MEAN of per-origin
    scores, not one score pooled across a multi-origin validation set —
    closing the same bias-cancellation gap during early stopping itself,
    not just at model-selection time).
  - `src/models/lightgbm_model.py::train_lgbm()` — new, centralizes
    correct LightGBM training: `metric="None"`, `first_metric_only=True`,
    asserts `evals_result_` tracked exactly the intended metric. Built
    specifically so the pass-3 bug (verified directly: without this,
    `evals_result_` contained both `"l2"` and `"vn1_score"`) can't be
    silently reintroduced at a future call site.
  - 12 new tests across `test_metrics.py`/`test_models.py`, including one
    that documents the bug directly (asserts `"l2"` IS present without
    the fix) and one confirming the fix (asserts only the intended metric
    is tracked). Full suite: 46/46 passing.
  - **Objective**: regression (mean-of-origin 0.5510) vs. tweedie (0.5490)
    — **nearly tied**, not the large gap earlier passes reported (as far
    apart as 0.55 vs 0.78 before the pass-3 fix). Origin-by-origin:
    regression wins 9/15, tweedie 6/15, mean diff only -0.0020. Kept
    `regression` — wins the origin-count comparison and is simpler — but
    the earlier "clear win" framing does not survive correction.
    `regression_l1` remains clearly worse (0.6503).
  - **Hyperparameters**: `num_leaves=200` (0.5459) nominally beat `127`
    (0.5482) but is the search-boundary value; origin-by-origin, 200
    wins only 9/15, mean diff -0.0023 — practically tied. **Kept the
    simpler `num_leaves=127`.**
  - **Sparse-segment investigation**: dedicated sparse-only model 1.2929
    vs. global model 1.2990 on the same sparse rows — **now nearly tied**,
    not the clear negative result earlier passes reported (1.4694 vs
    1.3952 before the pass-3 fix). **Decision unchanged in practice**
    (kept one global model — the gap is too small to justify a second
    model's complexity) but the earlier "clear negative result" claim
    does not survive correction. (Scored at the fold-pooled level, not
    re-verified origin-by-origin like the two comparisons above — a
    known, minor gap in rigor, unlikely to change the conclusion given
    how small the difference is.)
  - **LightGBM vs. MA4, origin by origin, across all 15 rolling-CV
    validation origins** (MA4 = each origin's own `roll_mean_4` feature):
    LightGBM wins **8/15**, MA4 wins **7/15** (median diff -0.0006,
    essentially even). Mean diff favors LightGBM (-0.0352), but a
    **leave-one-out sensitivity check** shows most of that depends on one
    origin (2022-12-19, a year-end forecast origin where MA4 performed
    exceptionally poorly — MA4 scores 1.10 there, LightGBM 0.58;
    seasonality is a plausible hypothesis, not established by this
    comparison alone): excluding it, the mean diff nearly vanishes
    (-0.0004). The one
    **robust, consistent** advantage — confirmed with absolute/normalized
    bias, not a signed mean that can hide cancellation across origins —
    is bias control: mean |bias| 87,270 vs. MA4's 270,009 (~3.1x), mean
    normalized |bias| (share of that origin's demand) 0.0286 vs. 0.0850
    (~3.0x). Both models skew toward net under-forecasting (10/15 and
    12/15 origins respectively) but LightGBM less consistently so.
  - **Final development-holdout evaluation** (fully corrected config):
    LightGBM **0.4996**, MA4 **0.4987** — **MA4 very slightly wins this
    single point**, the opposite of the rolling-CV win-rate. Breakdown:
    total_actual 3,508,827, total_predicted 3,467,481, signed_bias
    +41,346 (~1.2% net under-forecast), abs_error_sum 1,711,494.
  - **Honest conclusion (revised twice — "clearly ahead of MA4", then
    "modest edge", now this):** evidence is mixed. LightGBM provides
    substantial protection during one extreme period and controls
    portfolio bias more tightly (a real, robustly-measured advantage),
    but LightGBM wins slightly more origins while MA4 wins the
    development-holdout point, and the two perform about as well as
    each other at the median origin. Not proof of a reliable edge — a
    real, narrow, concentrated
    difference whose practical significance the actual Phase 1 result
    will help settle.
  - Notebook: notebooks/05_session5.ipynb, rewritten a third time after
    the pass-3 correction and executed end-to-end for real.

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

Expanding windows; remaining Chapter 2 concepts.

### Covered (Session 5)
Rolling-origin (walk-forward) cross-validation for time series; scoring a
hyperparameter search against a multi-fold mean instead of a single split;
testing a segment-specialization hypothesis directly rather than assuming
it; leave-one-out sensitivity as a way to check whether a mean advantage
is real or driven by one outlier; absolute/normalized bias vs. signed
bias (signed bias can hide within-model cancellation that absolute
measures reveal); and a hard lesson in how an eval-metric implementation
detail (a framework's own built-in metric silently riding alongside a
custom one during early stopping) can distort several conclusions at
once, not just the headline number — verified directly rather than
assumed, three times, across three review passes.

---

## Current benchmark
No single number is "the" benchmark anymore — the evidence itself is
mixed (DECISIONS.md D009, final version). On the development holdout
(DECISIONS.md D008 — looked at across three sessions and three review
passes of Session 5 alone): **MA4 0.4987 vs. LightGBM 0.4996** — MA4
slightly wins this one point. On the 15-origin rolling-CV comparison:
LightGBM wins 8/15 (vs. MA4's 7/15), with a real, robustly-measured
advantage in bias control (~3x tighter), but most of its mean-score
edge depends on a single extreme origin. MA12 (0.5163) stays tracked
separately for external comparability. Whichever model is used going
forward, treat "beats the baseline" claims as provisional until checked
origin-by-origin — a single point (development holdout or otherwise)
is not sufficient evidence on its own, as this session demonstrated
three times over.

---

## Next VN1 work

### Immediate — Week 1 Session 6: freeze and generate the Phase 1 forecast
Per the agreed freeze-before-reveal protocol (CONTEXT.md's Phase section,
extending D008): the modeling decisions are now settled (D006 direct
multi-horizon, D007 regression objective, D009 tuned hyperparameters +
no segment split, corrected methodology) via internal validation only.
Real Phase 1 actuals have not been obtained or added yet — nothing here
should wait on them.

**One open judgment call worth deciding before freezing**: D009's final
origin-level evidence shows LightGBM's edge over MA4 is close to a coin
flip on win-rate (8/15 vs. 7/15), and MA4 actually wins the single
development-holdout point (0.4987 vs. 0.4996). "The SupChains Way"
reference (added this session) rates ensembling as A-tier — "nearly
guaranteed to deliver better results." Given genuinely mixed single-model
evidence — LightGBM's real advantage (tighter bias, resilience during
one extreme period, wins slightly more origins) and MA4's real
advantage (wins the holdout point) look complementary rather than one
dominating — an
ensemble of LightGBM + MA4 is a reasonable option to consider for the
frozen Phase 1 forecast rather than picking one outright. This would be
new scope: not yet built, and would need its own rolling-origin CV
evaluation before being trusted (per D008/D010 — the same discipline
that changed every other conclusion in this session). Decide explicitly
rather than defaulting to LightGBM alone by inertia.

1. Retrain the frozen configuration on **all 170 weeks of Phase 0**
   (no holdout withheld this time — there's no more Phase 0 data to
   reserve once forecasting into genuinely unknown territory).
2. Generate the 13-week forecast for the real Phase 1 dates
   (2023-10-09 -> 2024-01-01, matching the sample submission's format).
3. Clip to >=0, validate against the submission format (no missing
   values, correct KEY + date columns).
4. Save and commit: the forecast file, full model params, feature list,
   training cutoff (2023-10-02), generation timestamp, and the code
   commit hash it was produced from.
5. Do not open, request, or add real Phase 1 actuals until after this
   frozen forecast is committed.

### Longer-term (after the Phase 1 freeze, or in parallel on Phase 0 only)
- Revisit D006 (recursive vs. direct) and D007 (Tweedie vs. L2) if a
  richer feature set changes the picture — both were evidence-based
  calls on early models, not permanent.
- A two-stage hurdle model where *both* stages still train on all data
  (not just sparse rows) remains untested — D009's negative result was
  specific to naive segment-restricted training data, not every
  possible segment-aware design.
- Denser origin sampling or an expanded lookback/rolling feature set.
- D008 (purge discipline, never select on the true holdout) is
  permanent and applies to all future comparisons.

---

## Important open questions
- How much zero demand is genuine intermittency vs censored (stockout)
  demand cannot be identified directly — VN1 has no inventory file.
- Price behavior is inferred from transaction structure; revisit if better
  price information appears.
- Best treatment of highly intermittent series is an empirical question,
  and the answer so far runs counter to intuition. Session 4 found the
  sparse segment (zero_rate>=0.9) scores far worse than the other two
  segments and contributes 21.6% of total absolute error from only 10.6%
  of demand — a real, evidence-based sign it's the hardest part of the
  problem. But Session 5 directly tested "give it dedicated treatment"
  (a model trained only on sparse rows) and found that made it *worse*,
  not better (D009) — the global model's cross-series learning matters
  more than segment specialization, at least for this simple version of
  specialization. Still open: whether a *different* kind of segment-aware
  design (e.g. a two-stage hurdle model where both stages still see all
  data) would behave differently.

---

## Deferred, not abandoned
Not required for the first VN1 delivery: deep learning, foundation
forecasting models, advanced ensembling, advanced probabilistic methods.
Part of the roadmap; introduce when conceptually relevant.

---

## Single next action
Session 6 — first decide explicitly (LightGBM alone, MA4 alone, or an
ensemble, tested via rolling-origin CV per D008/D010) given D009's final
mixed evidence, then retrain the frozen configuration on all 170 weeks
of Phase 0, generate the real Phase 1 forecast (2023-10-09 to
2024-01-01), and commit it with full provenance (params, feature list,
training cutoff, commit hash, timestamp) — before requesting or opening
any real Phase 1 actuals.

## Session-end rule
End every session by updating this file: what got done, the latest score,
and the exact single next action.
