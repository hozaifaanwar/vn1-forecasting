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

## After scoring

### What did I predict, and what happened?

I predicted that LightGBM would rank first because it would capture the October–January seasonality. Demand did increase, but LightGBM predicted an increase that was too large and lasted too long. The blend ranked first, while LightGBM ranked last among the four forecasts.

### Which part of my reasoning was right?

I was right that the forecast period contained seasonality and that a flat moving average would miss some of the increase. I was also partly right about the risk of overforecasting: seasonality existed, but it was weaker and ended earlier than LightGBM expected.

### Which part was wrong, and why?

I assumed that recognizing the seasonal direction would be enough for LightGBM to win. Our feature engineering did not represent the seasonal magnitude, timing, and allocation properly. Although we had a `lag_52` feature, it was calculated from the forecast origin and repeated across the 13 horizons instead of being aligned with each target week. We should test the seasonal rolling-window features described by Manu Joseph and Jeffrey Tackes.

### What did the simple models do better than LightGBM?

MA4 and MA12 were more conservative and did not spread such a large seasonal increase across thousands of inactive series. They underestimated total demand, while LightGBM overestimated it. The blend performed best because these opposite errors partly cancelled.

### What will I do differently for Phase 2?

Before modelling, I will examine the total demand and its breakdown by product, client, warehouse, and activity segment. I will study the historical January–April trend, when the year-end seasonality ends, and which intermittent series become active. I will then propose features and candidates based on those findings before evaluating their performance.
### Additions after review (Claude)

- **Numbers:** LightGBM forecast a +23.7% rise over the last 13 Phase 0 weeks; the actual rise was +11.3%. Scores: blend 0.5704, MA4 0.6411, MA12 0.6597, LightGBM 0.7341.
- **Another correct call:** MA4 was ranked above MA12, and that is how they finished.
- **Why MA4 allocated better:** a series with no sales in its last 4 weeks gets an MA4 forecast of exactly 0. MA4 put about 20,500 units on the 4,522 series that sold nothing, against LightGBM's 143,600, and had the best error part of all four models (0.542).
- **The process lesson:** the selection rule written before the results picked the blend. It was overridden after looking at the forecast, on the strength of one past season (five overlapping origins). The rule's pick won and the replacement came last. From Phase 2 on, the rule is written first and is not overridden (DECISIONS.md D013).
- **Evidence strength:** five origins from one season behave like one data point. Season-specific evidence needs at least two independent seasons before it can decide anything.
