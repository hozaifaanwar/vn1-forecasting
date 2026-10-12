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
| LightGBM (primary) | 0.45 |
| Blend (secondary) | .70 |
| MA4 (secondary) | .75|
| MA12 (official benchmark) | .70 |

## Why

Lightgbm beats the all other model as it covers the sesonality as oct-jan has christmas effect

## What result would overturn my reasoning
There might not seasonality in the real horizon, then our overforecasts will result poor.

