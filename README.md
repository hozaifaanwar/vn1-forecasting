# VN1 Forecasting

Demand forecasting on the VN1 Forecasting Accuracy Challenge
(Vandeput, DataSource.ai, 2024). Benchmarking baselines, gradient
boosting, and time-series foundation models under one backtest harness.

**Status:** Week 1, Session 4 — data pipeline, EDA, baseline suite, and a
first global LightGBM model, all scored on a real 13-week holdout with the
official VN1 metric. See [PROGRESS.md](PROGRESS.md) for current state and
[DECISIONS.md](DECISIONS.md) for the reasoning behind key choices.

## Setup
```bash
conda env create -f environment.yml
conda activate vn1
```

## Results
Official VN1 metric, scored on a 13-week holdout (2023-07-10 to 2023-10-02):

| Model | Score |
|---|---|
| MA4 | 0.4987 |
| **LightGBM** (direct multi-horizon, L2) | **0.4989** |
| MA8 | 0.5114 |
| MA12 (tracked benchmark) | 0.5163 |
| naive-last | 0.5247 |
| MA26 | 0.5618 |
| seasonal-naive-52 | 1.0498 |
| zero | 2.0000 |

Lower is better. LightGBM is essentially tied with MA4, not conclusively
better — see PROGRESS.md Session 4 for the full account, including a
methodology correction (the objective was initially selected by peeking at
the holdout; redone using internal validation only, which changed the
reported score from an optimistic 0.4955 to this honest 0.4989).
