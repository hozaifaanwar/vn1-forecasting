# D014 inputs — choices fixed before calibration

Recorded before any fold, calibration or challenger result was computed. These are the
inputs to DECISIONS.md D014; the thresholds (T1, T3 and the Phase 1 gate tolerance) are
still open and will be set from the benchmark-only calibration.

## Learner's answers to the six questions (`notes/session8_findings.md`)

| # | Question | Answer |
|---|---|---|
| 1 | Check every candidate specifically against the post-holiday January drop in the January backtests? | **Yes** (learner) |
| 2 | Use last year's pattern for *which* series take part, but be cautious about *how much*? | **Accept** (learner) |
| 3 | Count features from each series' first sale; add "weeks since first sale"? | **Yes** (learner) |
| 4 | Test a dormancy gate (forecast 0 after N silent weeks), kept only if the full score improves? | **Yes, N = 13** (learner) |
| 5 | Spikes: diagnostic only, or a model feature? | **Diagnostic only.** Delegated by the learner to Claude: spikes hold <1% of volume and the in-window signal (ISO weeks 6, 13) is weak, so a feature would add complexity and selection risk for little gain |
| 6 | Replace the portfolio-only January multiplier with a 50/50 mix of each series' own past January ratio and the portfolio ratio (portfolio only where history is insufficient), as one candidate? | **Yes** (learner) |

## Design choices (agreed after external review)

1. **January adjustment:** no damping toward 1. The one adjusted-MA4 candidate uses the 50/50 own/portfolio mix (answer 6), with ratios averaged over the prior years available at each origin. Each prior-year window must be fully observed before the current origin. The multiplier changes each series' level; allocation is the job of the activity, first-sale and seasonal features. Both are judged by the full official score.
2. **Dormancy gate:** N = 13, tested against the same model without it. In historical tests, "silent after selling" is distinguished from "not launched yet".
3. **No earlier year at an origin:** adjustment = 1 (plain MA4). Those origins are kept and reported separately.
4. **Folds:** three purged rolling folds over the origin pool extended through Phase 1. Exact origins recorded before any candidate runs.
5. **Jan 2022 (thin history):** keep the model specification; require no severe failure in either January season **and** positive combined January evidence.
6. **CatBoost:** skipped for Phase 2 (scope control). IDs are tested inside LightGBM (feature group 5), which does not answer whether CatBoost would help.
7. **Thresholds:** practical decision tolerances, not significance tests. Calibrated on **benchmarks only** (MA4, MA12, incumbent blend) using non-overlapping origins for variation. Calibration outputs and the chosen tolerances are committed before any challenger produces a forecast. T2 = 60% of origins won, with per-fold reporting. T3 is a separate January tolerance in score units. The Phase 1 gate is primarily full-score deterioration against the incumbent, with a tolerance; ±10% bias is a secondary flag. The incumbent was chosen partly because it won Phase 1, so challengers fail only if clearly worse there.

## Integrity notes carried into D014

- Session 8 (Q2 and the addendum) used actual Jan–Apr 2022 and 2023 outcomes, which are the windows of the January season-matched backtests. Those backtests are therefore not untouched for the January-adjusted MA4 and the series-level ratios; the all-season cross-validation gets more weight.
- The January finding matches the off-limits retrospective (D011). It is used only because it comes from Phase 0 data, and it must win the pre-written backtests.
