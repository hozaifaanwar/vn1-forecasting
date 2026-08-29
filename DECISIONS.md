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

## D008 — Purge fit/validation overlap; never use the true holdout for model/objective selection

**Decision:** (1) Any internal train/validation split built from overlapping direct-multi-horizon origins must be purged: `max(fit_origins) + max(horizons) < min(val_origins)`, enforced by `src/features.py::purge_fit_origins()` and asserted explicitly rather than assumed. (2) Never use the final holdout for model, objective, feature, or hyperparameter selection. Evaluate it only after the complete modeling decision is frozen. ("One look per major decision" is deliberately not the rule — that phrasing could be gamed into repeated holdout-guided decisions dressed up as separate "major" ones. The rule is one final evaluation of a frozen pipeline, full stop.)

**Reason:** Both violations were made in Session 4's first pass and caught by external review, not self-caught. (1) Sampling "the last few origins" as validation without purging let fit-origin targets (via horizons reaching up to 13 weeks out) land on or past calendar dates the validation set was nominally checking generalization to — contaminating the early-stopping signal, though it never touched the real holdout. (2) Comparing Tweedie/L1/L2 by their scores on the true holdout turned that holdout into a de facto model-selection set: the reported 0.4955 was optimistic by construction, confirmed empirically once corrected (0.4989 — worse, as expected). D004 established holding out the final 13 weeks; this decision makes explicit the corollary D004 didn't spell out — holding it out means not using it for *any* selection, only a single final evaluation.

**Revisit if:** never, as a general rule — this is a permanent process constraint, not an empirical finding to be overturned by evidence. Re-affirm it explicitly whenever a new model/feature/hyperparameter comparison is designed in a later session.
