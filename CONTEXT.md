# VN1 Demand Forecasting — Project Context

Stable facts only. Current status lives in [PROGRESS.md](PROGRESS.md). Decisions and their reasons live in [DECISIONS.md](DECISIONS.md).

## Verification rule

- Lines tagged **[VERIFIED]** come from reproducible notebook/script output or official competition materials.
- **[DERIVED]** = interpretation or reasoning based on verified facts, not a raw measurement.
- **[DECISION]** = a modeling or workflow choice made for this project.
- Do not promote conversational estimates or assumptions into [VERIFIED]. If this file conflicts with the actual repo, dataset, notebook output, competition rules, or current reproducible results, the current evidence wins.

## Goal

Build a strong forecasting and inventory-optimization portfolio for a transition into AI-driven supply chain demand planning and supply-chain data science.

- VN1 = forecasting accuracy project. FIRST.
- VN2 = inventory / ordering decision project. AFTER VN1.

Both are finished competitions with public data and reference/leaderboard information that can be used for self-evaluation.

The broader learning objective is not only to achieve a good competition score, but to understand the full decision chain:

> Demand → Forecast → Forecast uncertainty → Supply / inventory decision → Sales / service / cost outcome

Keep these distinctions clear:

- forecast quantity ≠ order quantity
- sales ≠ unconstrained demand
- forecast ≠ target or budget
- forecast accuracy ≠ business value

## Dataset

- **[VERIFIED]** 15,053 time series.
- **[VERIFIED]** Each series is uniquely keyed by: Client | Warehouse | Product.
- **[VERIFIED]** 170 weekly observations per series.
- **[VERIFIED]** Date range: 2020-07-06 → 2023-10-02.
- **[VERIFIED]** Weekly timestamps are Monday-anchored.
- **[VERIFIED]** Long-form dataset contains: 15,053 × 170 = 2,559,010 rows.
- **[VERIFIED]** Sales: 71.8% of observations are zero; 0% are NaN.
- **[VERIFIED]** Price: 71.8% of observations are NaN; 0% are zero.
- **[VERIFIED]** Number of rows where sales > 0 AND price is NaN = 0.
- **[VERIFIED]** Zero-rate distribution across series:
  - mean zero rate = 0.718
  - median zero rate = 0.835
  - approximately 40% of series have >90% zero-sales weeks
  - approximately 994 series have <10% zero-sales weeks
  - max zero-rate = 0.9941
  - every series has at least one positive sale (0 series with total or mean sales = 0; minimum total sales per series = 1)
- **[VERIFIED]** The weekly grid is complete.
- **[VERIFIED]** No inventory file is currently available.
- **[VERIFIED]** No static product/customer metadata is currently available.
- **[DERIVED]** Price appears to be transaction-derived and sticky: it is recorded on sale weeks and missing on many zero-sale weeks.
- **[DERIVED]** The portfolio appears to contain at least two broad behavioral groups: a relatively small active-demand head, and a large intermittent/sparse-demand tail. This segmentation is useful conceptually, but should not be treated as a formal classification until explicitly engineered and validated.

## Demand interpretation

Observed sales and underlying demand must not automatically be treated as the same quantity. A zero-sales observation can represent genuine zero demand, or constrained/censored demand caused by stockout or supply shortage. At present, there is no inventory file in VN1, so stockout censoring cannot be directly identified from inventory data.

- **[DECISION]** Do not automatically reinterpret observed zeros as missing demand.
- **[DECISION]** Do not interpolate observed sales zeros.

Revisit if additional stock, shortage, missed-sales, or availability information becomes available.

## Official VN1 forecasting task

- **[VERIFIED]** Forecast horizon = 13 weeks.
- **[VERIFIED]** Forecast all 15,053 series.
- **[VERIFIED]** Output is evaluated using the official competition metric.

### Official metric

**[VERIFIED]**

```
score = (total absolute error + absolute total bias) / total actual demand
```

Lower is better. The metric penalizes both individual forecast error and systematic aggregate bias. For VN1, the official competition metric is the final optimization target, even when textbook forecasting KPIs suggest additional useful diagnostics.

**[VERIFIED]** `data/raw/Functions for participants/Functions for participants.py` (gitignored, part of the original competition download, not committed) is the organizers' own reference scoring/dummy-submission code. Confirmed directly against it: `src/metrics.py::vn1_score` is formula-identical to their `score = (abs_err + abs(err)) / objective.sum().sum()`, and `src/backtest.py::ma12_forecast`'s default window matches their own MA12 reference submission exactly.

## Competition phases and current data status

**[VERIFIED]** The real competition had three stages: Phase 0 (historical, given) → forecast Phase 1 (13 weeks) → Phase 1 actuals released → forecast Phase 2 (13 weeks, training on Phase 0+Phase 1 combined) → competition ends, no further leaderboard feedback.

**[VERIFIED]** `data/raw/` currently contains only `Phase_0_Sales.csv` and `Phase_0_Price.csv` (the 170 historical weeks this project already uses) plus `Submission Phase 1 - Random-3.csv` — confirmed by direct inspection to be a placeholder/template submission (dates 2023-10-09 to 2024-01-01, small-integer filler values), not real Phase 1 demand. **No Phase 1 or Phase 2 actual sales/price files are present.** Everything scored so far (MA12, the baseline suite, LightGBM) is a local backtest inside Phase 0 (train on its first 157 weeks, test on its final 13), not a score against real post-Phase-0 competition data.

**[DECISION]** If/when real Phase 1 actuals are added: freeze the full modeling pipeline (features, objective, hyperparameters, clipping) using Phase 0-only validation first, generate and commit the Phase 1 forecast, and only then load the actuals to score — never let them influence the frozen forecast. See DECISIONS.md D008, which this directly extends.

## Benchmark

**[DECISION]** Initial baseline = MA12.

For each series: calculate the mean demand over the most recent 12 historical weeks; repeat that value as a flat forecast across the full 13-week horizon.

Purpose: establish a simple benchmark, confirm the validation pipeline works, create a minimum performance level that later ML models must beat.

MA12 is a project benchmark, not an official competition rule.

## Validation principle

Because the competition is already finished, local backtesting should reproduce the original forecasting task as closely as possible.

**[DECISION]** Hold out the final 13 historical weeks. Train / calculate features using only data available before the holdout. Forecast those 13 weeks. Score predictions using the official competition metric.

No future information may enter training features. Prevent target leakage in all feature engineering, e.g.:

- lag features must use past values only
- rolling features should normally shift before rolling
- future prices/promotions may only be used if they would genuinely have been known at forecast time

## Preprocessing principles

### Sales

**[DECISION]** Preserve observed zeros. Do not interpolate or replace them merely because demand is sparse. Missing demand and zero demand are different concepts.

### Price

**[DECISION]** Forward-fill price within each Client | Warehouse | Product series only. Reason: price appears sticky and transaction-derived.

**[DECISION]** Preserve leading NaN prices before the first observed price. Do not backward-fill them, because that would introduce future information.

Where implemented, preserve a flag such as `price_was_missing` so the model can distinguish observed price from imputed price.

### Merge integrity

**[DECISION]** Sales-price merges should use one-to-one validation where appropriate. The pipeline should fail loudly if duplicate keys unexpectedly multiply rows.

## Source and evidence hierarchy

When reasoning or making recommendations, distinguish among four types of information:

1. **Verified project fact** — evidence from the actual dataset, repo, notebook/script output, official competition rules, or official metric.
2. **Book-derived guidance** — concepts or recommendations from Joseph & Tackes or Nicolas Vandeput. Name the source when it materially affects a decision.
3. **Competition / practitioner best practice** — evidence or lessons from winning solutions, forecasting competitions, strong industry practice, or supply-chain forecasting practice.
4. **Model / assistant recommendation** — a reasoned recommendation or interpretation. Do not present this as a verified fact.

## Project architecture

Each project file has one job:

- **Project Instructions** — how Claude should teach, reason, explain, and integrate the books.
- **CONTEXT.md** (this file) — slow-changing project facts and architecture.
- **PROGRESS.md** — fast-changing project state: what was completed, current session, current score, current experiment, next action.
- **DECISIONS.md** — append-only record of important choices: decision, reason, evidence, revisit condition.
- **Repo and data** — ultimate source of truth for implementation. If documentation conflicts with reproducible current output, current output wins.
- **Books** — retrieved reference knowledge used when relevant. The books are not separate courses that must all be completed before building models.

## Repo

Primary repo: `D:\vn1-forecasting`

Current structure includes:

```
src/
  data.py
  metrics.py
  backtest.py
  models/
notebooks/
data/raw/          # raw competition files, gitignored
data/processed/    # vn1_long.parquet, generated / gitignored
references/        # four forecasting / inventory books, gitignored
CONTEXT.md
PROGRESS.md
DECISIONS.md
environment.yml
.vscode/settings.json
```

Environment: conda environment `vn1`, Python 3.11. A separate GPU environment may be used later on the home PC.

Repo/current executable output is the implementation source of truth.

## Book roles

### Joseph & Tackes — *Modern Time Series Forecasting with Python*

PRIMARY technical sequence and coding spine.

Covers (not all needed for the first delivery — see Learning principle): time-series foundations, preprocessing, validation, feature engineering, regression framing, machine learning, gradient boosting, global forecasting models, ensembling, deep learning, probabilistic forecasting, intermittent demand, multi-step forecasting, forecast evaluation.

### Nicolas Vandeput — *Demand Forecasting Best Practices*

PRIMARY demand-planning process and decision-context reference.

Use for: why forecasts exist, demand vs sales, unconstrained demand, shortage censoring, granularity, forecasting horizon, forecast KPIs, portfolio metrics, benchmarks, ABC/XYZ and prioritization, Forecast Value Added, human review, judgmental forecasting, planner + AI interaction.

Use this book to connect technical forecasting to actual demand-planning decisions.

### Nicolas Vandeput — *Data Science for Supply Chain Forecasting*

SUPPORTING reference.

Use for: supply-chain forecasting intuition, classical statistical forecasting, forecast KPIs, demand drivers, machine learning, feature importance, gradient boosting, judgmental forecasting, Forecast Value Added, practical supply-chain forecasting implementation.

Joseph & Tackes remains the primary technical coding sequence where the two overlap.

### Nicolas Vandeput — *Inventory Optimization: Models and Simulations*

PRIMARY decision-layer reference for VN2. Also introduce its concepts earlier whenever forecasting naturally connects to inventory decisions.

Use for: inventory policies, reorder point, order-up-to policies, review periods, lead time, safety stock, service levels, fill rate, stochastic demand, stochastic lead time, cost/service optimization, non-normal demand, newsvendor, discrete demand, simulation, multi-echelon inventory optimization.

Important connection: forecast horizon should eventually be related to lead time and review period. Forecast uncertainty should eventually feed inventory decisions rather than stopping at point forecasts.

### Book integration rule

Do not teach the four books independently chapter-by-chapter. Use Joseph & Tackes as the technical sequence. Bring Vandeput material into the lesson when it:

- clarifies the business objective,
- improves a forecasting decision,
- prevents a bad modeling assumption,
- explains demand-planning practice,
- connects forecasting to inventory,
- or improves interpretation of results.

When sources disagree: explain the assumptions behind each position; do not silently choose one. For VN1 evaluation, the official competition rules/metric win. For operational supply-chain practice, explain whether a different method or KPI may be more appropriate.

## Learning principle

Nothing important should be permanently excluded merely because it is advanced. Use progressive depth:

> Encounter → Understand → Use simply → Revisit → Use deeply

One primary new concept at a time, together with only the dependencies needed to understand it. Advanced topics may be deferred from the first VN1/VN2 delivery while remaining part of the long-term learning roadmap — e.g. deep learning, foundation forecasting models, advanced ensembling, advanced probabilistic methods, modern neural forecasting architectures.

Deferred ≠ abandoned.

## Overall target

The goal is not simply to build a model with low error. The target capability is:

> Data → Understand demand → Build reliable baselines → Engineer forecasting features → Train scalable global models → Validate correctly → Quantify uncertainty → Interpret forecasts → Support planner judgment → Translate forecasts into inventory decisions → Measure operational and business value
