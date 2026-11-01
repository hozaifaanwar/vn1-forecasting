# Phase 2 strategy — DRAFT

Status: draft for discussion, **updated after Session 8** (demand analysis). The binding
version becomes DECISIONS.md D014, written **before** any Phase 2 candidate forecast is run
(D013 rule 1). Items marked **[decide]** are open; items marked **[proposed, Session 8]** are
changes suggested by the demand findings, for the learner to accept or reject in D014.

## The task

- **Data:** Phase 0 + Phase 1 = 183 weekly observations per series, 2020-07-06 → 2024-01-01 (index 0–182).
- **Forecast:** origin 2024-01-01 (index 182), 13 weeks, 2024-01-08 → 2024-04-01, all 15,053 series.
- **Incumbent:** the frozen-style 50/50 LightGBM–MA4 blend procedure, retrained on Phase 0 + 1. Its Phase 1 score (0.5704) is historical context, **not a Phase 2 target**, because demand difficulty differs between phases. The official MA12 stays as the external yardstick.
- **Allowed information:** Phase 1 actuals (competitors had them). Not allowed: anything published after VN1 closed about Phase 2's outcome (D011). Don't open the VN1 section of `references/The SupChains Way for AI Agents.md` while this is being designed.
- **D011 transparency note (Session 8):** the Q2 finding that a January MA4 level over-forecasts Jan–Apr matches what the off-limits retrospective reports for Phase 2. Here it comes only from Phase 0 data (2021–2023 seasons) that any 2024 competitor had. Anything built on it (the seasonally adjusted MA4 below) must win the pre-written January backtests to be chosen; it is never adopted on the strength of the finding alone.

## Sequence

1. [DONE] **Learner:** five-sentence interpretation of Phase 1 (`notes/phase1_predictions.md`).
2. [DONE] **Dataset:** Phase 0 + Phase 1 (`data/processed/vn1_long_p0p1.parquet`, 183 weeks).
3. [DONE] **Learner-led demand session:** Q1–Q6 answered and verified (`notes/session8_findings.md`). Still open: the learner's own interpretations and answers to its five "Questions for D014".
4. **Write D014:** shortlist, feature groups, validation design, decision rule, thresholds. The learner writes predicted winner, reasons and what would change their mind. Commit and push.
5. **Build:** features with leakage tests, then the candidates.
6. **Feature ablation** (LightGBM only), applying D014's keep/drop rule mechanically.
7. **Main comparison**, applying D014's decision rule mechanically.
8. **Freeze:** generate from committed code, provenance, verify, tag `phase2-freeze-v1`, push.
9. **Score** when Phase 2 actuals are added. Learner interprets first.

## Session 8 findings that shape this plan

Full fact sheet: `notes/session8_findings.md` (every number checked with a second implementation).

| Finding | Consequence for the plan |
|---|---|
| **Q1** Q4 2023 peak concentrated: top 10% of series = 92% of the rise | Features must say *which* series take part, not lift everything |
| **Q2** Jan–Apr fell below the January MA4 level in 2021–2023 (MA4 total 11–19% too high). At the Phase 2 origin the last 4 weeks include the Dec 11 peak (MA4 level ≈ 285,600) | Concrete design for the seasonally adjusted MA4; extra bias diagnostic |
| **Q3** Q4 2023 broke the growth trend; the same items peak each year (45% overlap, ~9× chance) but last year's top risers rose ~70% less | Last year predicts *which*, not *how much*: damp seasonal size; trend/regime features matter |
| **Q4** Spikes rare (0.9% of units); weeks 6 and 13 inside the Phase 2 window mildly elevated | Diagnostic only |
| **Q5** 62% of Phase 0 zeros are pre-launch; 40% of series change segment once excluded; 25% of series have under a year of history at the origin | Count features from each series' first sale; add "weeks since first sale" |
| **Q6** Series silent 13+ weeks produce ~0.5% of future demand | Test a dormancy gate |

## Validation design

| Comparison | Origins | What it answers | Caveat |
|---|---|---|---|
| All-season rolling CV | Rolling purged folds over the extended origin pool (targets ≤ week 182) **[decide: fold count/size]** | Best on average across seasons | Includes the Phase 1 period, which is now development data |
| Season-matched, Jan 2023 | ~5 weekly origins around index 130 (2023-01-02) | Best for a Jan–Apr horizon | One season; overlapping origins ≈ one data point |
| Season-matched, Jan 2022 | ~5 weekly origins around index 78 (2022-01-03) | Second independent Jan–Apr season | **Few training origins exist before it** (52 → ~64), so ML candidates are handicapped; the annual lag `o+h−104` doesn't exist yet |
| Phase 1 replay | Origin 169 (2023-10-02) | Development robustness test: do candidates repeat the Phase 1 failure? | **It does influence selection** (it is a gate in the decision rule), so its acceptance measures are frozen in D014 before any candidate runs. It is never tuned against repeatedly. It is one season, so it can disqualify a candidate but not crown one |

**[decide]** How to handle the thin training history at the Jan 2022 origins: accept and note it, or relax the 52-week lookback for a variant without `lag_52`.

## Candidates (fixed in D014; no additions afterwards)

**Benchmarks, fully eligible to win.** A challenger replaces them only by showing positive forecast value under the pre-written rule:
1. MA12 (official)
2. MA4
3. Frozen-procedure LightGBM (same features and settings, retrained on Phase 0 + 1)
4. Frozen-style 50/50 blend of (3) and (2) — **the incumbent**

**Challengers:**

5. **LightGBM-v2:** the feature set kept by the ablation (below), plus the refit-on-all procedure if it wins its own backtest.
6. **Seasonally adjusted MA4:** `MA4 × adjustment for the forecast window`.
   - Caution: a full-strength historical index would have overshot Phase 1 just as LightGBM did (history +26–37% vs. actual +11%). It fixes *allocation*, not *magnitude*.
   - **[proposed, Session 8]** One pre-declared variant: multiply each series' MA4 by the **average past ratio of (actual 13-week window ÷ MA4 level at the same calendar origin)**, computed on the **portfolio total** from the prior years available at each origin (for Phase 2: the 2021, 2022 and 2023 January ratios ≈ 0.90, 0.84, 0.87). Portfolio-level, not per segment: Q2's segment ratios are unstable (sparse swung −59% to +208%) because late starters contaminate the segment labels (Q5); only "active" was steady.
   - At backtest origins the ratio uses only years before that origin (Jan 2022: the 2021 ratio alone; Jan 2023: 2021 and 2022). Origins with no earlier year at the same calendar point use a ratio of 1, i.e. plain MA4. **[decide]** whether to accept that.
   - Expected weak spot, stated in advance: in the Phase 1 replay (October origin), past October ratios reflect the strong 2020–2022 Q4 rises and would overshoot 2023 just as LightGBM did. The Phase 1 robustness gate is there to catch this.
   - **[decide]** Whether to damp the ratio toward 1 (e.g. halfway), given that Q3 showed seasonal size does not repeat reliably. Decide before any backtest runs; one variant only.

7. **Fixed 50/50 blend of (5) and (6).**
8. **Optional: CatBoost with Client/Warehouse/Product as categorical features,** only if (5)'s ablation shows the ID group adds value, or if there's time **[decide]**.

**Excluded, with reasons:**
- XGBoost: same model family, no specific hypothesis.
- A standalone two-stage `P(sale) × size` model: it still estimates the mean (playbook 8). Sale-probability ideas enter through activity features instead.
- Fitted ensemble weights: a selection risk.

## LightGBM-v2 feature groups (ablation order)

Add one group at a time to the frozen feature set, and keep a group only if it passes D014's keep rule: an improvement of at least a threshold set the same way as T1 (from benchmark noise, before any ablation runs) **and** a majority of origins won.

1. **Target-aligned annual history:** `y[o+h−52]`, `y[o+h−51]`, `y[o+h−53]`, mean of the three; plus `y[o+h−104]` where available.
2. **Activity / recency:** weeks since last sale; count of non-zero weeks in the last 4, 13, 26 and 52; mean size of non-zero sales; whether the series sold in the same season last year. **[proposed, Session 8]** Also **weeks since first sale**, and the zero rate and segment label **counted only from the first sale** (Q5: 40% of series are mislabelled otherwise). Group 1's target-aligned lags need this companion: for the 25% of series launched within the last year, "same week last year" is a pre-launch zero.
3. **Aggregates (all ≤ origin):** product, client and warehouse totals, recent growth, and the target-week seasonal index at each level.
4. **Trend and regime:** `MA4/MA12`; recent 13 weeks vs. the same 13 weeks last year (this year's rise against last year's — the Phase 1 magnitude problem). **[proposed, Session 8]** Higher priority than before: Q3 showed seasonal *size* broke trend in 2023, and this is the only group that can see a change in regime.
5. **Identities:** Client, Warehouse, Product as categorical features.
6. **[proposed, Session 8] Dormancy gate** (post-processing, tested in the ablation like a group): forecast zero for series with no sale in the last **N** weeks at the origin (N fixed in D014, e.g. 13). Q6: such series produce ~0.5% of future demand; LightGBM placed 143,600 units on series that sold nothing in Phase 1. Plain MA4 already forecasts zero after 4 silent weeks, so the gate mainly matters for LightGBM.

Leakage tests for each group, in the existing style: corrupting all data after the origin must leave every feature unchanged.

## Decision rule (draft for D014)

An incumbent/challenger framework: the incumbent (4) stays primary unless a candidate passes **every** acceptance criterion below.

1. **All-season improvement:** the candidate beats the incumbent on the all-season mean-of-origin score by at least threshold **T1**, and wins at least share **T2** of origins against it.
2. **Season check (both Jan seasons):** no severe failure in either season (worse than the incumbent by more than **T3**), and positive aggregate evidence across the two seasons combined. A candidate may be *described* as season-specifically better only if it wins both seasons, but that is not required to be chosen, because Jan 2022 structurally handicaps ML candidates (little training history).
3. **Phase 1 robustness gate:** in the Phase 1 replay, total forecast within ±**T4** of actual, and volume on series that sold nothing well below frozen LightGBM's (criterion fixed in D014).
4. **Choose:** among candidates passing 1–3 (benchmarks included), the one with the best all-season mean becomes primary.
5. **Fallback:** if no candidate passes, **retain the incumbent**. A failing candidate, combination or not, is never the fallback.
6. **Defects:** a candidate may be invalidated or fixed once for a defect demonstrable independently of its score (D013 rule 1). Remaining candidates are never reordered by observed performance.
7. **No overrides after results are seen.** Anything found later is a recorded diagnostic.

**How T1–T4 are set, before any challenger runs:**
- from benchmark-vs-benchmark variability (e.g. MA4 vs. MA12, and the incumbent vs. MA4, across the same origins);
- from origin-to-origin noise in those benchmark comparisons;
- from operational significance: a gap too small to change a planning decision doesn't count.

The thresholds and the calculation that produced them are committed in D014 together. They are never adjusted after challenger scores exist.

## What every result report includes (D013 rule 4)

- Combined score, error part, bias part.
- Per-segment scores (active / intermittent / sparse).
- Weekly totals against actuals, and weekly scores.
- Volume placed on series with zero actual demand in the window.
- Per-fold and per-season breakdown, so one origin or one season can't dominate.
- **[proposed, Session 8]** Whether the origin's last 4 weeks contain a spike or peak week that inflates MA4-based forecasts (the Phase 2 origin does: Dec 11 = 375,764).
- **[proposed, Session 8]** Each candidate's implied total change over the window against the past January ratios (0.84–0.90 of the MA4 level), as a quick bias check.

## After the Phase 2 freeze

- Quantile forecasts, with calibration over cumulative lead-time demand (not per week, because of the zeros).
- A first inventory simulation as the bridge to VN2.
- Trim PROGRESS.md and move session history into `notes/archive/`.
