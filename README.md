# VN1 Forecasting

Demand forecasting on the VN1 Forecasting Accuracy Challenge
(Vandeput, DataSource.ai, 2024). Benchmarking baselines, gradient
boosting, and time-series foundation models under one backtest harness.

**Status:** Week 1, Session 5 — data pipeline, EDA, baseline suite, and a
global LightGBM model tuned via rolling-origin cross-validation, scored
on a real 13-week holdout with the official VN1 metric. Session 5 went
through three rounds of external review before committing, each catching
a real methodology gap — see [PROGRESS.md](PROGRESS.md) for current state
and [DECISIONS.md](DECISIONS.md) for the reasoning behind key choices,
including what changed between review passes and why.

## Setup
```bash
conda env create -f environment.yml
conda activate vn1
```

## Results
Official VN1 metric, scored on a 13-week development holdout (2023-07-10
to 2023-10-02) — but read this table with real caution, see below:

| Model | Score |
|---|---|
| MA4 | 0.4987 |
| LightGBM (direct multi-horizon, L2, tuned) | 0.4996 |
| MA8 | 0.5114 |
| MA12 (tracked benchmark) | 0.5163 |
| naive-last | 0.5247 |
| MA26 | 0.5618 |
| seasonal-naive-52 | 1.0498 |
| zero | 2.0000 |

**Evidence is mixed, not a clean win for either model.** On this single
holdout point, MA4 edges out LightGBM. On a 15-origin rolling-CV
comparison, LightGBM wins slightly more often (8/15 vs. 7/15) with a
real, robustly-measured advantage in bias control (~3x tighter, using
absolute not signed bias, which can hide cancellation) — but most of its
mean-score edge over MA4 depends on a single extreme origin (a year-end
forecast origin where MA4 performed exceptionally poorly — seasonality
is a plausible hypothesis, not established by this comparison alone);
excluding it, the advantage nearly vanishes. Neither model dominates.

This project's Session 5 went through three rounds of external review
before committing anything, and each one caught a real, distinct
problem: no rolling-origin baseline comparison existed at first; then
early stopping used a proxy metric (L1) instead of the real one; then —
most consequential — a custom LightGBM eval metric doesn't exclusively
drive early stopping unless `metric="None"` is also set, which was
silently letting LightGBM's own built-in metric influence model
selection throughout. Fixing that changed multiple conclusions, not just
the headline number (see DECISIONS.md D009/D010 and PROGRESS.md Session
5 for the full, three-times-corrected account). Session 4 had its own
earlier correction too: an initial 0.4955 was optimistic because the
objective had been selected by peeking at the holdout.
