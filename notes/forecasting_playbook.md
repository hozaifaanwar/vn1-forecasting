# Forecasting playbook — lessons from VN1

Portable lessons for any demand-forecasting project. Each one has the rule,
the VN1 mistake that taught it, and how to apply it. Project details live in
DECISIONS.md; this file should still make sense on a different dataset.

Book references were checked against the PDF text in `references/`:
J&T = Joseph & Tackes, *Modern Time Series Forecasting with Python* (2024);
DFBP = Vandeput, *Demand Forecasting Best Practices* (2023).

---

## A. Understand demand before modelling

**1. Plot total demand over time before building anything.**
- *Mistake:* the 26–37% Oct–Jan rise in every year was found by accident in Session 6, five sessions in. One line plot would have shown it in Session 2.
- *Apply:* start every project, and every modelling session, with one demand question and the plot that answers it: the total over time, then by segment, client, warehouse and product.
- *Source:* DFBP's chapter order puts demand definitions, granularity and horizon (ch. 2–7) before metrics, benchmarks, FVA and models (ch. 8–15).

**2. Check whether a pattern is broad or concentrated.**
- *Mistake:* LightGBM applied the Q4 uplift to 84% of series and put 143,593 units on series that sold nothing.
- *Apply:* before modelling a seasonal effect, measure what share of series and of volume actually carries it.

**3. Direction is not magnitude.**
- *Mistake:* history said Q4 rises 26–37%. Phase 1 rose 11%. The model learned history's size of rise and forecast +24%.
- *Apply:* sanity-check every forecast's implied change in the total against the history *and* the recent trend. When they disagree, hedge instead of trusting either.

## B. Features

**4. In direct multi-horizon models, seasonal features must line up with the target week.**
- *Mistake:* `lag_52` was measured back from the origin, so it held one value for all 13 horizons. It matched "the same week last year" only for horizon 1, and by horizon 13 it pointed 12 weeks too early.
- *Apply:* for horizon *h* from origin *o*, use `y[o + h − 52]` (known at the origin while *h* < 52). Do the same for `−104` and for windows around those weeks.
- *Source:* J&T ch. 6, "Seasonal rolling window aggregations" (p. 134): windows of `y[T−M], y[T−2M], …`.

**5. Calendar features give timing, not size.**
- *Mistake:* week-of-year sin/cos plus month let the model learn "December means more" for every series.
- *Apply:* add measured seasonal indices (at series, product, client and warehouse level) and say which series take part in the season.

**6. Global models need identity and hierarchy to allocate correctly.**
- *Mistake:* Client, Warehouse and Product IDs were dropped, so series with similar recent numbers got similar forecasts whatever they were.
- *Apply:* test IDs as categorical features and aggregate demand by product, client and warehouse. Sparse series can borrow from denser relatives that way.
- *Source:* J&T's global-model chapter adds "meta features" (categorical IDs) to a global LightGBM.

**7. Feature importance is not feature value.**
- *Mistake:* `price_at_origin` ranked top, but price is recorded only when something sells, so it largely acts as an activity signal.
- *Apply:* judge a feature group by an ablation (add it, re-validate), not by its importance score.

## C. Intermittent demand and the metric

**8. Know what your metric rewards on zero-heavy series.**
- For a series that is usually zero, the error part (MAE) is minimised by forecasting zero. That forecast is useless and badly biased.
- An unbiased forecast is small and positive everywhere, which costs error on every series that stays at zero.
- VN1's metric (MAE + |bias|) weighs both.
- *Apply:* the lever is *allocation*: predicting which series will sell. Rewriting the formula as `P(sale) × size` alone doesn't help, because that still estimates the mean.
- *Source:* DFBP §9.2 "Intermittent demand" (p. 80): the all-zero forecast has the best MAE and −100% bias.

**9. Read the two parts of the score separately.**
- *Lesson:* the blend won Phase 1 almost entirely on bias (0.006). MA4 had the better error part (0.542 against 0.564).
- *Apply:* always report the error part, the bias part, per-segment scores, the weekly shape and the volume placed on series that later sell nothing.

## D. Choosing between models

**10. Write the decision rule before computing results, and never override it afterwards.**
- *Mistake:* the pre-registered rule picked the blend. An analysis run after looking at the forecast overrode it. The model selected by the original rule (the blend) won, while the post-hoc replacement came last.
- *Apply:* rules written in advance decide. Analyses run afterwards are diagnostics only.

**11. Count independent seasons, not origins.**
- *Mistake:* five weekly origins in one season, sharing two trained models, were treated as strong evidence.
- *Apply:* overlapping origins are about one data point. Require at least two independent seasons before letting season-specific evidence decide anything.
- *Also:* rolling origins 4 weeks apart with 13-week targets share most of their target weeks, so "15 origins" is real evidence but not 15 independent experiments. Look at the size and concentration of wins, not just the count: LightGBM's 8–7 win count over MA4 hid a near-zero median gap, and one origin drove most of its mean advantage.

**12. When reliable evidence disagrees, test a combination.**
- *Lesson:* LightGBM forecast too high, MA4 too low, and their average was within 0.6% of actual.
- *Apply:* when reliable comparisons disagree, treat a pre-declared fixed-weight combination as a hedge worth testing. Combine only complementary forecasts, and require the combination to pass the same acceptance criteria as any other candidate. Weights fitted on validation data are a selection risk.

**13. Simple benchmarks are hard to beat; measure Forecast Value Added.**
- *Lesson:* the official MA12 beat our tuned LightGBM on real data.
- *Apply:* compare each step with the step before it. A sophisticated step with negative value added gets diagnosed or removed, not defended.
- *Source:* DFBP ch. 12 (p. 101); Vandeput, *Data Science for Supply Chain Forecasting*, ch. 27.

## E. Validation discipline

**14. Never select on the final holdout.**
- *Mistake:* Session 4 picked the training objective by scoring on the holdout, which gave an optimistic score.
- *Apply:* the holdout gets one evaluation of a frozen pipeline, after every decision is made.

**15. Purge overlapping targets in time-series validation.**
- *Mistake:* training targets reached into the validation period.
- *Apply:* require `max(fit_origin) + max(horizon) < min(val_origin)`, and assert it rather than assume it.

**16. Verify framework defaults; don't assume them.**
- *Mistake:* passing a custom LightGBM eval metric did not make it the only one driving early stopping. The built-in metric stayed active, and three conclusions were distorted.
- *Apply:* assert what was actually tracked, and test the bug directly.

**17. Validate in the season you'll forecast *and* across seasons.**
- *Lesson:* an all-season average can hide the regime that matters, and one matched season can overstate it.
- *Apply:* run both comparisons, set up before seeing results, under rules 10 and 11.

## F. Integrity and reproducibility

**18. Freeze, tag and push a forecast before the actuals exist on your machine.**
- *Apply:*
  - generate the forecast from committed code;
  - record the commit, the input and output hashes and the environment;
  - tag and push;
  - verify with a separate script;
  - score only afterwards, and never edit frozen files.

**19. Write down your own prediction before every reveal.**
- *Lesson:* predicting "LightGBM first" and seeing it come last taught more than reading any analysis.
- *Apply:* before each experiment, write what you expect, why, and what result would change your mind. Then interpret the result yourself, before anyone else.

**20. Describe exactly what a model was trained on.**
- *Mistake:* the record said "trained on all 170 weeks", but the trees never saw the last 24 labelled weeks.
- *Apply:* state which data fit the model, which only chose its stopping point, and which only fed the features.

**21. Make the environment reproducible.**
- *Apply:* pin core versions, export a lock file, and write code that works across library versions (e.g. LightGBM's `eval_set` vs. `eval_X`).

**22. Don't use hindsight that real competitors couldn't have had.**
- *Apply:* post-event write-ups of the outcome are off-limits when designing a forecast meant to show a fair result. General, timeless practice is fine.

## G. Why validation missed the lag_52 problem

**23. Test what a feature means, not just that the code runs.**
- *Mistake:* the only lag test checked `lag_1` and `lag_2` at horizon 1. Nothing checked `lag_52`, or any lag at horizons 2–13. The leakage test passes equally for the right and the wrong seasonal lag, so it couldn't tell them apart.
- *Apply:* write each feature's definition in words ("sales in the same week of last year as the *target* week"), then test that definition at several horizons, including the last one.

**24. Name features by what they actually measure.**
- *Mistake:* the name `lag_52` suggested "52 weeks before the target". It was really "51 weeks before the origin" (the origin week itself is `lag_1`).
- *Apply:* make the reference point part of the name, e.g. `origin_lag_52` vs. `target_lag_52`, so a misreading is visible in the name.

**25. Run the diagnostics during validation, not only after the reveal.**
- *Mistake:* error by horizon, volume placed on series that later sell nothing, share of series over-forecast and per-segment scores were only examined after Phase 1. In validation the combined score looked competitive.
- *Apply:* make these part of every validation report, so an allocation or horizon problem shows up before a forecast is frozen.

**26. An advantage that comes from one origin is a warning, not a strength.**
- *Mistake:* the leave-one-out check showed LightGBM's mean edge over MA4 came almost entirely from one year-end origin. We read that as "LightGBM handles seasonality"; it was also a sign the advantage was fragile.
- *Apply:* when one origin drives a result, treat the result as unproven and look for a second independent case before relying on it.

## H. Demand analysis lessons (Session 8)

**27. Separate "not launched yet" from "didn't sell".**
- *Finding:* 62% of all Phase 0 zeros came before a series' first sale. Counting them, 40% of series got the wrong active/intermittent/sparse label, and the weekly zero share appeared to fall from 72% to 56% when, among series already selling, it actually rose.
- *Apply:* find each series' first sale before computing zero rates, segments or "same week last year" features. Count from the first sale, and add a "weeks since first sale" feature.

**28. Ask whether a seasonal effect is broad or concentrated.**
- *Finding:* the top 10% of series made 92% of the Q4 2023 rise, and 62% of series didn't rise at all.
- *Apply:* before modelling a seasonal effect, rank series by their uplift and compute the cumulative share. A concentrated effect needs features that say *which* series take part.

**29. Last year predicts *which* items peak better than *how much*.**
- *Finding:* 45% of 2023's top risers were also 2022's (about 9× chance), but those items rose about 70% less in 2023.
- *Apply:* use last year's pattern to rank participation, and damp or check its size against recent behaviour.

**30. When a seasonal rise looks weak, check whether the season or the starting point changed.**
- *Finding:* summer 2023 grew normally (+36% year on year) but Q4 2023 grew only +13%. The season broke the trend, not the baseline.
- *Apply:* compare both windows year on year, not just the rise within one year.

**31. Measure what a simple benchmark would do at the forecast origin, in past seasons.**
- *Finding:* at the first Monday of January in 2021–2023, an MA4 level was 11–19% above actual Jan–Apr demand in total.
- *Apply:* for the season you'll forecast, compute the benchmark's implied total against past actuals. It shows the bias risk before any model is built. (This checks the total, the bias part only; the per-series test is a backtest.)

**32. Long-silent series produce almost no future demand.**
- *Finding:* series quiet for 13+ weeks sold again 23% of the time but produced only 0.55% of next-13-week demand (52+ weeks: 0.05%).
- *Apply:* near-zero forecasts for long-dormant series cost little and avoid spreading volume across them. Test this as a rule; don't assume it.

**33. Rare spikes deserve a check, not a model.**
- *Finding:* isolated spikes hit 5% of series and held under 1% of units, consistently across thresholds. Some recur in fixed calendar weeks.
- *Apply:* measure the volume they carry under several thresholds before building anything for them. Watch for spikes inside a moving-average window at the origin.

**34. Gross and net are different numbers; say which one.**
- *Mistake:* "the peak added 155,825 units a week" counted only rising series; net of falling series it was about 117,000.
- *Apply:* when summing changes across items, report risers, fallers and net separately.

## I. Working practice (tooling lessons from Session 8)

**35. A crash blamed on one library may come from a shared dependency.**
- *Mistake:* for five sessions, "matplotlib bar charts crash" was recorded as a matplotlib limitation. The real cause was numpy's MKL maths library crashing every `np.dot` (Windows code 0xC06D007F), which also broke all plotting and would have broken any linear-algebra model.
- *Apply:* when a native crash appears, get the crashing Python line (`python -X faulthandler`) and the Windows error code. Then test the lowest-level call that fails before accepting a workaround.

**36. Pin a notebook to the right kernel.**
- *Mistake:* notebooks set to the generic `python3` kernel ran in base Anaconda, whose older pyarrow couldn't read the data.
- *Apply:* set the notebook's kernelspec to the project environment (`vn1`) and execute with `--ExecutePreprocessor.kernel_name=vn1`.

**37. Never let a pipeline hide a failure before a commit.**
- *Mistake:* a script with a broken path string was committed and pushed, because `command | tail` reported the exit code of `tail`, not of the script.
- *Apply:* check the real exit code (`set -o pipefail`, or capture to a file and test `$?`). Parse-check generated code (`ast.parse`) before writing it into a file or notebook. Only commit after a clean run.

**38. Save, then verify from the saved file.**
- *Mistake:* several times the notebook on disk didn't contain code that had been run in the editor.
- *Apply:* save before asking for a review, and review outputs from the saved file. Re-run a notebook top to bottom before committing it, so its outputs come from one in-order run.

## J. Phase 2 selection (Session 8, D014–D015)

**39. Every scored target must be invisible to the model that is scored, including through early stopping.**
- *Mistake:* the first calibration early-stopped LightGBM on the very origins it was scored on. The learner caught it. Fixing it (one frozen-procedure model per origin) moved the incumbent from 0.5354 to 0.5425.
- *Apply:* evaluate with exactly the procedure that will make the real forecast. Purge the ablation's targets away from the reserved block.

**40. An average ratio of ratios is biased upward.**
- *Mistake:* the January-adjusted MA4 averaged per-series ratios. Series that were tiny at the prior origin produced huge ratios, so the mixed multiplier was 1.01 when the portfolio ratio was 0.87, and 1.58 in Q4 against an actual 1.11.
- *Apply:* aggregate as total ÷ total (level-weighted), or shrink and cap per-series ratios. Check the level-weighted multiplier against the portfolio ratio before trusting it.

**41. A gain in the mean is not a gain if it loses most origins.**
- *Finding:* the dormancy gate improved the mean by 0.006 but won only 4 of 10 origins. LightGBM alone won Jan 2023 by 0.07 but lost 16 of 20 origins overall.
- *Apply:* require both size and consistency, and decide that before seeing results.

**42. Fix the specific defect before adding new ideas.**
- *Finding:* of six feature groups, only the target-aligned annual lag (the fix for the Phase 1 failure) helped. The other four groups and the gate did not pass, and IDs badly overfit (0/10).
- *Apply:* first repair what the post-mortem found, then test new features one at a time.
