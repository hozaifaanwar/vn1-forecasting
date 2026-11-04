# VN1 fresh pipeline (post-competition replay)

One standalone script, `vn1_fresh.py`, that rebuilds a VN1 forecast from the methods the top 5
described after the competition (webinar, November 2024; Vandeput's Medium article, January 2025).

**Honesty label.** This was written after seeing the Phase 2 actuals, the leaderboard and the
winners' write-ups (DECISIONS.md D016). Its Phase 2 score is a *post-competition replay*, not a
competition result. To keep it as fair as possible, every choice is made inside the script on
backtest folds from the training history. The script never reads the sales of the phase it forecasts,
and the Phase 2 file was produced and scored once, after the code was committed.

## Run it

Python 3.10+ and `pip install -r requirements.txt`. Point `--data` at the official VN1 folder (the
one with `Phase 0 - Sales.csv`, `Phase 1 - Sales.csv`, `vn1_score.py`).

    python vn1_fresh.py --data "path/to/vn1-forecasting-competition" --phase 1 --out out
    python vn1_fresh.py --data "path/to/vn1-forecasting-competition" --phase 2 --out out

    cd "path/to/vn1-forecasting-competition"
    python vn1_score.py "path/to/out/phase2_forecast.csv"
    python vn1_score.py "path/to/out/phase1_forecast.csv" --phase 1

Each run takes roughly 10–30 minutes on a 16-core laptop. It writes:
- `phaseN_forecast.csv`: the ensemble, in the official layout.
- `phaseN_component_<model>.csv`: each component alone.
- `phaseN_report.json`: fold scores, chosen rounds, weights and dormancy rule.

## What is inside, and where the idea came from

| Step | What | Source |
|---|---|---|
| Leading zeros | A series starts at its first sale; earlier weeks are not history | 4th (Schwartz) |
| AMSI | Recent anchor level × last years' ratio for the next 13 weeks. Ratio of sums, shrunk series → client → portfolio | 1st (Stubbs & Figura) |
| Seasonal Theta | `DynamicOptimizedTheta`, season 52 | 2nd (Furlotte); Vandeput's B tier |
| LightGBM | Global, direct, Tweedie 1.5. Lags, rolling means, zero shares, weeks since last sale, last-year values, AMSI index | 2nd, 3rd, 5th; Vandeput's A tier |
| Backtest | 4 non-overlapping 13-week folds over the last year; the season-matched fold counts double | 2nd (Furlotte); our D015 lesson |
| Ensemble | Non-negative weights fitted to the VN1 metric on the folds | 2nd, 5th |
| Dormancy | Zero for series with no sale in the last N weeks (N picked on folds) | 5th (Hoang) |
