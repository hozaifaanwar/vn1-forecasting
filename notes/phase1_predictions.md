# Phase 1 predictions — written before the actuals are scored

Written by Hozaifa, without agent input, before any Phase 1 score is seen.
Official metric, lower is better.

## Expected ranking (best to worst)

1. Lightgbm
2. Blenden
3. MA4
4. MA12

## Rough score range for each

| Model | Expected score range |
|---|---|
| LightGBM (primary) | 0.4-0.60 |
| Blend (secondary) | 0.60-0.70 |
| MA4 (secondary) | 0.65-0.80|
| MA12 (official benchmark) | 0.70-0.85 |

## Why

Lightgbm beats the all other model as it covers the sesonality as oct-jan has christmas effect.
Blend score 2nd as it cover the both sessonality for the lower tail using lightgbm and for upper tail it uses the MA4 which covers the same demand pattern
MA4 is only good for if there is not much variation in the demand and no seasonality present.
MA12 might perform as MA4 as it avaregaes the all past 12 weeks, therefore it can't address any variation like intermettency and spike and lumphy demand,

## What result would overturn my reasoning
There might not seasonality in the real horizon, then our overforecasts will result poor.

