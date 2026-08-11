# Project context

**Goal:** VN1 forecasting → VN2 inventory planning portfolio project, 4 months.
Career pivot toward supply chain data science.

**Data:** VN1 Forecasting Accuracy Challenge (DataSource.ai, Vandeput 2024).
Phase 0: Sales + Price only, weekly, 15,053 series (Client×Warehouse×Product),
170 weeks (2020-07-06 to 2023-10-02). Wide format. Sales 71.8% zeros;
Price NaN exactly when sales=0. No inventory file.

**Task:** forecast 13 weeks, all series. Submission is wide, key-indexed, no NaNs.
Metric (official, provided): score = (sum|err| + |sum err|) / total_actual.
Punishes accuracy AND systematic bias. Lower is better.
Baseline to beat: MA12 (mean of last 12 weeks, repeated flat 13 weeks).

**Validation:** competition finished, so hold out last 13 weeks of Phase 0,
predict them, score with the official function.

**Plan:**
- M1 W1-4: setup, data, metric, backtest harness, first global LightGBM
- M2 W5-8: global model tuning, censored demand, freeze model, write-up
- M3 W9-12: inventory theory, quantile forecasts, VN2 ordering policy
- M4 W13-16: polish, optional foundation-model benchmark, write-up, buffer

**Current position:** Week 1, Session 1 — setup and repo.

**Working rules:** hand-roll the backtest harness before using skforecast.
Notebooks explore, src/ keeps. Ship something at every month boundary.
