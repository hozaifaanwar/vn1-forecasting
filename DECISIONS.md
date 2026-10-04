# Decision Log

Append-only. One entry per material choice. Newest at the bottom.

## D001 — Sales zeros are real observations

**Decision:** Do not interpolate or fill observed sales zeros.

**Reason:** The dataset has a complete weekly grid. A recorded zero is an observation ("nothing sold"), not missing data.

**Revisit if:** stockout, availability, missed-sales, or other censoring information becomes available that allows us to distinguish genuine zero demand from constrained demand.

## D002 — Price forward-fill, per series

**Decision:** Forward-fill price within each Client | Warehouse | Product series only; leave leading NaNs unchanged.

**Reason:** Price appears transaction-derived and sticky — it is observed on sale weeks, and no rows were found where sales > 0 while price was NaN. The last observed price is therefore used as the working estimate for subsequent sale-less weeks.

Leading NaNs are preserved because no historical price was yet observed. Do not backward-fill them, because doing so would use future information.

Preserve a `price_was_missing` flag before forward-filling.

**Revisit if:** a non-transaction price source, promotion file, price history, or other better price information becomes available.

## D003 — Merge validated one-to-one

**Decision:** Sales ↔ price merge uses `validate="one_to_one"`.

**Reason:** Client | Warehouse | Product | date should uniquely identify a row on both sides. Fail loudly on duplicates rather than silently multiplying rows.

**Revisit if:** the underlying data model changes and multiple records per series-date become legitimate.

## D004 — Local validation for a finished competition

**Decision:** Hold out the final 13 weeks of Phase-0 history, generate forecasts for those weeks using only earlier information, and score them using the official competition metric.

**Reason:** The competition is closed, so the local validation setup should reproduce the real forecasting task as faithfully as possible.

**Revisit if:** competition documentation or available historical phases suggest a more faithful rolling or multi-window validation design.

## D005 — Advanced model scope deferred

**Decision:** Deep learning, foundation forecasting models, and advanced ensembling are not required for the first portfolio delivery.

**Reason:** The initial milestone is time-boxed and can demonstrate strong forecasting practice using baselines and global machine-learning models.

These topics are NOT removed from the learning roadmap. Introduce them when conceptually relevant and revisit them for deeper implementation later.

**Revisit if:** the simpler/global ML approaches plateau, the project finishes ahead of schedule, or an advanced method offers a clear learning or performance benefit.

## D006 — Direct (horizon-as-feature) multi-step strategy, not recursive

**Decision:** The first global LightGBM model uses a direct multi-step strategy: one model, `horizon` (1..13) included as a feature, with every lag/rolling/price feature computed relative to a fixed forecast origin rather than relative to the target row.

**Reason:** A 13-week-ahead forecast cannot use lag features built by shifting the continuous historical series and then slicing out the holdout rows — for horizon h>1 that silently leaks real holdout actuals into the "lag" features (e.g. a naive lag_1 feature for holdout week 2 would be the true sales value from holdout week 1, which a real forecaster standing at the origin would not yet know). Two standard fixes exist: recursive (predict h=1, feed the prediction back in as if actual, repeat) or direct (reframe training so every example's features are origin-relative and horizon is explicit). Direct is chosen for the first model because it has no compounding-error feedback loop to get wrong, is simpler to verify as leakage-free by construction, and is the standard approach for global panel-forecasting models (vs. classical single-series recursive methods).

**Revisit if:** direct-strategy accuracy plateaus and recursive (or a hybrid, or per-horizon direct models instead of horizon-as-feature) is worth the added complexity to try.

## D007 — LightGBM objective: plain regression (L2), not Tweedie

**Decision:** The first global LightGBM model uses `objective="regression"` (L2/squared error), not `"tweedie"`.

**Reason:** Tweedie loss is common practitioner guidance for zero-inflated, right-skewed demand data (tier-3 evidence, e.g. widely used in M5-competition-style solutions), so it was tried alongside L1 and L2. **Correction (caught by external review, see D008):** the first pass selected the winner by comparing scores computed directly on the true 13-week holdout — Tweedie 0.7414, L1 0.6145, L2 0.4955 — which uses the holdout for model selection and optimistically biases the reported number. Redone correctly: all three were compared on a purged internal validation set only (D008) — Tweedie 0.8253, L1 0.6430, L2 0.5052 — L2 still wins, same ranking, so the *choice* was right, just reached the wrong way the first time. The one honest look at the true holdout with the L2 model then scored **0.4989** (not 0.4955) — worse than the peeked number, exactly as expected once selection bias is removed. This is a development-holdout result, not a pristine one (see D008): the holdout was already inspected during the flawed first pass.

**Revisit if:** the feature set or model architecture changes substantially (e.g. per-segment models, a hurdle/two-stage model for the sparse segment specifically), since Tweedie's zero-inflation handling may earn back its advantage under a different setup than tried here.

**Second correction (Session 5, D009/D010 — appended, not rewritten, per the append-only log policy since this entry was already published):** two things above don't hold up under fully corrected methodology. (1) "The one honest look" wording is historically imprecise — even at the time this was written, the holdout had already been inspected during the flawed first Session 4 pass described above; later documentation correctly calls it a development holdout, and this entry's phrase should be read as historical, not literal. (2) The Tweedie-vs-L2 gap reported here (0.8253 vs. 0.5052 on internal validation) turned out to be substantially a methodology artifact: Session 5 found that validation scoring itself was contaminated by LightGBM's own built-in metric riding alongside the intended one during early stopping (D010). Re-run with that fixed, Tweedie (mean-of-origin 0.5490) is nearly tied with regression (0.5510) — regression still wins the origin-count comparison (9/15) and remains the simpler choice, so the *decision* is unchanged, but the large gap originally reported here was not real.

## D008 — Purge fit/validation overlap; never use the true holdout for model/objective selection

**Decision:** (1) Any internal train/validation split built from overlapping direct-multi-horizon origins must be purged: `max(fit_origins) + max(horizons) < min(val_origins)`, enforced by `src/features.py::purge_fit_origins()` and asserted explicitly rather than assumed. (2) Never use the final holdout for model, objective, feature, or hyperparameter selection. Evaluate it only after the complete modeling decision is frozen. ("One look per major decision" is deliberately not the rule — that phrasing could be gamed into repeated holdout-guided decisions dressed up as separate "major" ones. The rule is one final evaluation of a frozen pipeline, full stop.)

**Reason:** Both violations were made in Session 4's first pass and caught by external review, not self-caught. (1) Sampling "the last few origins" as validation without purging let fit-origin targets (via horizons reaching up to 13 weeks out) land on or past calendar dates the validation set was nominally checking generalization to — contaminating the early-stopping signal, though it never touched the real holdout. (2) Comparing Tweedie/L1/L2 by their scores on the true holdout turned that holdout into a de facto model-selection set: the reported 0.4955 was optimistic by construction, confirmed empirically once corrected (0.4989 — worse, as expected). D004 established holding out the final 13 weeks; this decision makes explicit the corollary D004 didn't spell out — holding it out means not using it for *any* selection, only a single final evaluation.

**Revisit if:** never, as a general rule — this is a permanent process constraint, not an empirical finding to be overturned by evidence. Re-affirm it explicitly whenever a new model/feature/hyperparameter comparison is designed in a later session.

## D009 — Session 5: rolling-origin CV, tuned hyperparameters, no segment split (final, after three review passes)

**Decision:** (1) Use `rolling_origin_folds()` (3 walk-forward, purged folds) for any model/hyperparameter comparison going forward, scored by the **mean of origin-level scores** (`vn1_score_by_group`), not one pooled fold-aggregate — pooling lets one origin's bias cancel another's and hides how consistently a config wins. (2) LightGBM hyperparameters: `num_leaves=127, learning_rate=0.05, min_child_samples=50, n_estimators=2000` (early-stopped), replacing Session 4's untuned `num_leaves=63`. (3) Keep `objective="regression"` over Tweedie/L1. (4) Keep one global model — do not split into segment-specific models. (5) Early stopping must be exclusively driven by the real VN1 score, not a proxy or an unintentionally-still-registered built-in metric — see D010, a separate permanent process rule this session's review also produced.

**This entry went through three review passes before anything was committed** — each caught a real, distinct problem, and the third one (D010's early-stopping exclusivity bug) turned out to have distorted more of this session's findings than just the headline number. The numbers below are the final, fully corrected ones; see D007's appended correction and PROGRESS.md Session 5 for the fuller history of what changed between passes and why.

- **Objective**: regression (mean-of-origin 0.5510) vs. tweedie (0.5490) vs. regression_l1 (0.6503). L1 is clearly worse. **Tweedie is nearly tied with regression** — compared origin-by-origin, regression wins 9/15, tweedie wins 6/15, mean diff only −0.0020. Kept `regression` (wins the origin-count comparison, simpler — no variance-power hyperparameter) — but the large gap reported in earlier passes (as big as 0.55 vs 0.78) was substantially a D010 methodology artifact, not a real difference between the objectives.
- **Hyperparameters**: `num_leaves=200` (mean-of-origin 0.5459) nominally beat `127` (0.5482), but is the largest value tested (a search-boundary result). Origin-by-origin: 200 wins only 9/15, mean diff −0.0023 — practically tied. **Kept the simpler `num_leaves=127`**, not the boundary value.
- **Sparse-segment investigation**: a model trained only on sparse-segment rows scored 1.2929 (mean) vs. the global model's 1.2990 on the same sparse rows — **now nearly tied**, not clearly worse as earlier passes reported (as far apart as 1.4694 vs 1.3952 pre-D010-fix). **Decision unchanged in practice** (kept one global model — the difference is too small to justify a second model's complexity) but the earlier claim of a clear negative result does not survive the corrected methodology and should not be repeated as strong evidence. (This comparison was scored at the fold-pooled level, not re-verified origin-by-origin like the two comparisons above — a known, minor gap in rigor, not expected to change the conclusion given how small the difference is.)
- **LightGBM vs. MA4, origin by origin, across all 15 rolling-CV validation origins** (MA4 = each origin's own `roll_mean_4` feature): LightGBM wins **8/15**, MA4 wins **7/15** (median diff −0.0006, essentially even). Mean diff favors LightGBM (−0.0352) but a **leave-one-out sensitivity check** shows most of that mean advantage depends on one origin (2022-12-19, a year-end forecast origin where MA4 performed exceptionally poorly — MA4 scores 1.10 there, LightGBM 0.58 — seasonality is a plausible hypothesis for why, but not established by this comparison alone): excluding it, the mean diff nearly vanishes (−0.0004). The one **robust, consistent** advantage, confirmed with absolute/normalized bias measures rather than a signed mean that can hide cancellation across origins: bias control is genuinely tighter — mean |bias| 87,270 vs. MA4's 270,009 (~3.1x), mean normalized |bias| (as a share of that origin's demand) 0.0286 vs. 0.0850 (~3.0x) — both LightGBM and MA4 skew toward net under-forecasting (10/15 and 12/15 origins respectively) but LightGBM does so less consistently.
- **Final development-holdout evaluation** (fully corrected configuration): LightGBM **0.4996**, MA4 **0.4987** — **MA4 very slightly wins this single point**, the opposite of the rolling-CV win-rate. Consistent with "evidence is mixed, margin is small either way" — this single development-holdout point (already inspected across Sessions 3, 4, and three passes of Session 5) should not be read as stronger evidence than the 15-origin rolling comparison; if anything it's the noisier of the two estimates.

**Honest conclusion (revised twice now — first from "clearly ahead of MA4" to "modest edge," now to this):** evidence is mixed. LightGBM provides substantial protection during one extreme period and controls portfolio bias more tightly (a real, robustly-measured advantage), but LightGBM wins slightly more origins while MA4 wins the development-holdout point, and the two perform about as well as each other at the median origin. This is not proof that LightGBM adds reliable value over MA4 — it is a real, if narrow and concentrated, difference whose practical significance the actual Phase 1 result will help settle.

**Revisit if:** a richer feature set or different model architecture (e.g. a two-stage hurdle model where both stages still train on all data, not just sparse rows) might change whether segment specialization helps. Also revisit the 127-vs-200-leaves and regression-vs-tweedie ties, and the sparse-segment fold-pooled (not origin-level) comparison, if a denser origin sample or richer features later separates any of them more clearly.

## D010 — LightGBM early stopping must be exclusively driven by the real target metric

**Decision:** Whenever a custom LightGBM eval metric drives model/iteration selection, set `metric="None"` on the estimator and `first_metric_only=True` on the early-stopping callback, and assert `list(model.evals_result_["valid_0"]) == [expected_metric_name]` after fitting. Never assume that passing `eval_metric=<callable>` alone makes that the only metric considered. `src/models/lightgbm_model.py::train_lgbm()` centralizes this so it can't be forgotten at a future call site.

**Reason:** Verified directly, not assumed: with `objective="regression"` and `eval_metric=vn1_lgb_eval` alone, `model.evals_result_["valid_0"]` contained **both** `"l2"` (LightGBM's built-in metric for the `regression` objective, registered automatically) and `"vn1_score"`. Without `first_metric_only=True`, early stopping requires every registered metric to stop improving before halting, so `best_iteration_` was not guaranteed to be the iteration that actually minimizes the intended metric. Caught by external review of Session 5's second pass, not self-caught. The impact was larger than expected: re-running the objective comparison, hyperparameter search, and sparse-segment investigation with the fix in place changed several previously-reported conclusions (D009) — the earlier "clear" Tweedie-vs-regression gap and the earlier "clear" negative sparse-segment result were both substantially methodology artifacts, not purely real differences, even though the final decisions (regression, no segment split) happened to survive the correction.

**Revisit if:** never, as a general rule — like D008, this is a permanent process constraint for any future custom-eval-metric LightGBM training, not an empirical finding.

## D011 — Do not use post-competition outcome knowledge to shape the frozen Phase 1/Phase 2 forecast

**Decision:** When designing, selecting, or freezing the Phase 1 or Phase 2 forecasting model/strategy, do not use information from sources published after VN1 closed (October 2024) that reveals facts specific to VN1's actual outcome — including which model classes won, what accuracy competitors achieved, or any description of the real demand pattern in Phase 1/Phase 2. General, timeless forecasting/ML practitioner knowledge that doesn't depend on knowing VN1's specific outcome (e.g. "LightGBM suits tabular panel data," "ensembling generally helps," "moving averages can be fooled by seasonal transitions" as a general principle) remains legitimate tier-3 evidence, same as any 2024 competitor could have used. Outcome-specific hindsight does not.

**Reason:** A fair "we genuinely beat the other competitors" claim requires our modeling decisions to use no more information than a real 2024 competitor had. This is a distinct category from the temporal leakage D004/D006/D008 already guard against — leakage of *outcome knowledge*, not future rows of the time series. `references/The SupChains Way for AI Agents.md` (added Session 5) contains a January-2025 retrospective on VN1 specifically, including a concrete outcome-level fact: "moving averages absorbed Phase-1 year-end sales and over-forecasted Phase 2." Letting that shape our own Phase 2 approach later would be leakage of the answer laundered through prose instead of a CSV. Checked against what's already frozen: Session 5's D009 findings (no segment split, hyperparameter choices) were reached independently via our own rolling-CV *before* this document was consulted, which only corroborated them afterward — a legitimate sanity check, not the source of the decision, so nothing currently frozen is compromised. The risk is forward-looking: Session 6 (Phase 1 freeze) and especially the eventual Phase 2 freeze, which is exactly the transition this document's retrospective describes the outcome of.

**Revisit if:** never, as a general rule — like D008 and D010, this is a permanent integrity constraint, not an empirical finding to be weighed against evidence.

## D012 — Phase 1 primary forecast: LightGBM alone, chosen by a season-matched backtest; blend and MA4 frozen as secondaries

**Decision:** The frozen Phase 1 submission (`submissions/phase1/phase1_forecast.csv`) is the D009 LightGBM configuration alone, trained on all 170 Phase 0 weeks via `src/pipeline.py::frozen_forecast`. An equal-weight LightGBM+MA4 blend and MA4 alone are frozen alongside it as declared secondaries (`phase1_secondary_blend.csv`, `phase1_secondary_ma4.csv`). All three were committed before any Phase 1 actuals were obtained, and all three get scored when they are.

**Reason — two comparisons, answering two different questions:**

- **All-season rolling CV** (Session 5's 3 purged folds, 15 origins, pre-registered rule: blend chosen only if its mean-of-origin score beats both components and it wins ≥8/15 against each). Blend 0.5387 vs. LightGBM 0.5482 vs. MA4 0.5834; blend wins 11/15 origins against each, and its edge over LightGBM survives dropping its single best origin (LOO mean diff −0.0043). The rule was satisfied, so averaged over seasons the blend is the best of the three. Its cost is bias: mean normalized |bias| 0.046 vs. LightGBM's 0.029. Per-fold early stopping on the validation origins flatters LightGBM by at most ~0.004 where this could be checked, far below these margins.
- **Season-matched backtest** — the question that actually matches Phase 1. Inspecting the Phase 1 forecast before committing showed LightGBM forecasting ~24% more total demand than MA4 for Oct–Jan. Phase 0 itself shows a rise of that size every year: Oct–Jan demand was 26–37% above the preceding 13 weeks (2020 +37%, 2021 +26%, 2022 +33%). MA4 is flat by construction and cannot anticipate it. So the exact frozen pipeline was re-run at the same calendar point one year earlier (origins 2022-09-19 … 2022-10-17; Phase 1's origin is 2023-10-02), with "lowest mean score wins" fixed beforehand: **LightGBM 0.7127 vs. blend 0.7536 vs. MA4 0.8384; LightGBM wins 5/5.** The blend under-forecasts by 14–20% of demand there, LightGBM by 4–10%.

Phase 1 is an Oct–Jan horizon, so the season-matched result governs the Phase 1 choice, and the all-season rule is superseded *for Phase 1*. This was a post-hoc analysis, prompted by looking at the forecast rather than planned up front — recorded as such rather than presented as if it had been the plan all along. It is not leakage. It used only Phase 0 data, which any 2024 competitor had, so it is not outcome hindsight in the D011 sense. It also came before any Phase 1 actuals were seen, so it is not selection on the holdout in the D008 sense either.

**Evidence limits:** the 5 season-matched origins cover a single season (Q4 2022) and reuse only two distinct trained models (step-4 origin grid), so they are close to one strong data point, not five independent ones. They agree with three years of Q4 rises and with the mechanism, which is why they were trusted over the all-season average for this horizon. One final development-holdout evaluation of the frozen pipeline (origin 2023-07-03, a Jul–Oct horizon, run once after the all-season rule was applied) gave LightGBM 0.4996 and MA4 0.4987, reproducing Session 5 exactly, and blend **0.4839**. That is consistent with the all-season result, and per D008 it played no part in any choice.

**Process rule carried forward:** for any future frozen forecast (Phase 2 included), run both comparisons — all-season rolling CV *and* a season-matched backtest at the same calendar origin(s) one year earlier — before freezing, and choose the primary by the season-matched result when the two disagree. Decide that before looking at the forecast, not after. This rule comes from this session's own Phase 0 finding, not from any post-competition source (D011).

**Revisit if:** Phase 1 actuals show the season-matched backtest misjudged the horizon (e.g. the blend or MA4 beats LightGBM on Phase 1), or a model that captures seasonality without giving up the blend's per-series error advantage (e.g. a blend whose MA component is seasonally adjusted) is tested and wins both comparisons.

**Correction (appended after the freeze, per the append-only policy; caught by external review):** "trained on all 170 Phase 0 weeks" above is inaccurate. All 170 weeks are *used*, but not all as fitted labels. LightGBM's trees are fit on the purged training origins 52–132, whose targets end at week 145. Origins 148/152/156, whose targets run through week 169, are used only to choose the early-stopping iteration (223). The forecast's input features come from week 169. So the most recent 24 weeks of labelled examples never fit a tree. This is the same procedure that was backtested (dev holdout and season-matched origins), so the frozen forecast is consistent with its evidence; the Phase 1 freeze (tag `phase1-freeze-v1`, commit 1202691) is not reopened. **Declared Phase 2 candidate:** choose the iteration count on the purged validation block, then refit on every fully labelled origin with that count. It has not been backtested yet, so it enters Phase 2 as a candidate under the two comparisons above, not as an automatic upgrade — a tree count tuned on fewer rows need not transfer to more.
