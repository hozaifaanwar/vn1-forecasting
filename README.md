# VN1 Forecasting

Demand forecasting on the VN1 Forecasting Accuracy Challenge
(Vandeput, DataSource.ai, 2024). Benchmarking baselines, gradient
boosting, and time-series foundation models under one backtest harness.

**Status:** Session 6. The Phase 1 forecast (13 weeks, 2023-10-09 to
2024-01-01) is frozen and committed with full provenance in
[submissions/phase1/](submissions/phase1/), *before* any Phase 1 actuals
were obtained. It hasn't been scored against real Phase 1 demand yet. See
[PROGRESS.md](PROGRESS.md) for current state and
[DECISIONS.md](DECISIONS.md) for the reasoning behind each choice.

## Setup
```bash
conda env create -f environment.yml
conda activate vn1
```

## Reproduce the Phase 1 forecast
```bash
python scripts/build_dataset.py        # raw CSVs -> data/processed/vn1_long.parquet
python scripts/make_phase1_forecast.py # refuses to run with uncommitted code changes
pytest tests/
```

## Results so far
All scores use the official VN1 metric, `(sum|error| + |sum error|) / sum(actual)`.
Lower is better.

**Which model is best depends on the season of the forecast horizon.**

| Model | Rolling CV, all seasons (15 origins) | Season-matched Oct–Jan backtest (5 origins) | Development holdout, Jul–Oct |
|---|---|---|---|
| LightGBM (direct multi-horizon, tuned) | 0.5482 | **0.7127** | 0.4996 |
| 50/50 LightGBM + MA4 blend | **0.5387** | 0.7536 | **0.4839** |
| MA4 | 0.5834 | 0.8384 | 0.4987 |
| MA12 (official benchmark) | — | — | 0.5163 |

- **Averaged over all seasons, the blend is best.** It wins 11 of 15
  rolling-CV origins against each of its two components.
- **For an Oct–Jan horizon, LightGBM alone is best.** It wins all 5
  backtests run at the same calendar point one year earlier. Phase 0
  demand rises 26–37% in Oct–Jan every year, and a flat moving average
  can't anticipate that, which drags the blend down with it.

Phase 1 is Oct–Jan, so **LightGBM alone is the primary Phase 1
forecast**, with the blend and MA4 frozen alongside as declared
secondaries ([DECISIONS.md D012](DECISIONS.md)). The development holdout
has been inspected since Session 3, so treat it as a development result,
not a clean test. The real test is the frozen Phase 1 forecast.

## Methodology guardrails
Each guardrail came out of a real mistake, caught in review:
- Purged rolling-origin validation. The final holdout is never used for
  selection (D008).
- LightGBM early stopping is driven only by the VN1 metric. A custom eval
  metric alone doesn't guarantee that, because LightGBM's own built-in
  metric keeps influencing early stopping until it is switched off (D010).
- No post-competition outcome knowledge shapes a frozen forecast (D011).
- Forecasts are frozen and committed with a code hash before actuals are
  seen (D012).
