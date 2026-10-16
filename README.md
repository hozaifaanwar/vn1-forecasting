# VN1 Forecasting

Demand forecasting on the VN1 Forecasting Accuracy Challenge
(Vandeput, DataSource.ai, 2024). Benchmarking baselines, gradient
boosting, and time-series foundation models under one backtest harness.

**Status:** Session 7. The Phase 1 forecast was frozen and publicly
tagged (`phase1-freeze-v1`) before the actuals were obtained, then scored
against them. **The model we chose as primary, LightGBM, came last among the four frozen forecasts evaluated.** The
simple 50/50 blend we kept as a backup won. See below, and
[DECISIONS.md D013](DECISIONS.md) for what went wrong and what changed.

## Setup
```bash
conda env create -f environment.yml
conda activate vn1
```

## Reproduce the Phase 1 forecast
```bash
python scripts/build_dataset.py        # raw CSVs -> data/processed/vn1_long.parquet
python scripts/make_phase1_forecast.py # refuses uncommitted code changes and won't overwrite the frozen files
python scripts/verify_phase1_freeze.py # regenerates in memory and checks against the frozen files
pytest tests/
```

## Phase 1 result: real actuals, forecast frozen beforehand
Official VN1 metric, `(sum|error| + |sum error|) / sum(actual)`. Lower is better.

| Model | Score | Total forecast vs. actual |
|---|---|---|
| 50/50 LightGBM + MA4 blend (backup) | **0.5704** | +0.6% |
| MA4 (backup) | 0.6411 | −9.9% |
| MA12 (official benchmark) | 0.6597 | −11.8% |
| LightGBM (our primary) | 0.7341 | +11.1% |

**What happened:**
- Demand did rise in Oct–Jan, by 11%.
- LightGBM had learned a 26–37% rise from earlier years and forecast +24%.
- It also predicted a peak that was too broad, running through Christmas.
- It spread demand across thousands of series that sold nothing.
- MA4 missed the rise entirely.
- Averaging the two cancelled most of each one's error: the blend's total was within 1% of actual.

**What we got wrong in process:** a planned selection rule picked the
blend. We overrode it after the fact, based on one past season in which
LightGBM had won. The replacement rule is that the pre-written rule
decides, and evidence from a single season can't override it.

### Earlier evidence, inside Phase 0
| Model | Rolling CV, all seasons (15 origins) | Oct–Jan backtest, 2022 (5 overlapping origins) | Development holdout, Jul–Oct |
|---|---|---|---|
| LightGBM | 0.5482 | **0.7127** | 0.4996 |
| 50/50 blend | **0.5387** | 0.7536 | **0.4839** |
| MA4 | 0.5834 | 0.8384 | 0.4987 |
| MA12 | — | — | 0.5163 |

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
- The primary is chosen by a rule written in advance; no post-hoc
  overrides (D013).
