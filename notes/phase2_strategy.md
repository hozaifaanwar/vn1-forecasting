# Phase 2 strategy — DRAFT

Status: draft for discussion. The binding version becomes DECISIONS.md D014,
written **after** the learner-led demand session and **before** any Phase 2
candidate forecast is run (D013 rule 1). Items marked **[decide]** are open.

## The task

- **Data:** Phase 0 + Phase 1 = 183 weekly observations per series, 2020-07-06 → 2024-01-01 (index 0–182).
- **Forecast:** origin 2024-01-01 (index 182), 13 weeks, 2024-01-08 → 2024-04-01, all 15,053 series.
- **Incumbent:** the frozen-style 50/50 LightGBM–MA4 blend procedure, retrained on Phase 0 + 1. Its Phase 1 score (0.5704) is historical context, **not a Phase 2 target**, because demand difficulty differs between phases. The official MA12 stays as the external yardstick.
- **Allowed information:** Phase 1 actuals (competitors had them). Not allowed: anything published after VN1 closed about Phase 2's outcome (D011). Don't open the VN1 section of `references/The SupChains Way for AI Agents.md` while this is being designed.

## Sequence

1. **Learner:** five-sentence interpretation of Phase 1 (prediction vs. outcome).
2. **Dataset:** extend `scripts/build_dataset.py` to Phase 0 + Phase 1, with the same validation (complete grid, keys match Phase 0, price ffill + flag). Check that Phase 1 keys and order match Phase 0 exactly.
3. **Learner-led demand session** (questions below). Its findings decide which candidates are worth having.
4. **Write D014:** shortlist, feature groups, validation design, decision rule, thresholds. The learner writes predicted winner, reasons and what would change their mind. Commit and push.
5. **Build:** features with leakage tests, then the candidates.
6. **Feature ablation** (LightGBM only), applying D014's keep/drop rule mechanically.
7. **Main comparison**, applying D014's decision rule mechanically.
8. **Freeze:** generate from committed code, provenance, verify, tag `phase2-freeze-v1`, push.
9. **Score** when Phase 2 actuals are added. Learner interprets first.

## Demand questions for step 3

1. Which clients, warehouses and products drove the Nov 20 and Dec 11 peaks? What share of series carried most of the Q4 rise?
2. What happens in Jan–Apr in past years (2021, 2022, 2023), in total and by segment? Rising, falling or flat after the holidays?
3. How did the 2023 Q4 rise (+11%) differ from 2020–2022 (+26–37%)? Is the change visible in some series or all of them?
4. Spikes: how often do series have isolated weeks far above their usual level? Do those weeks cluster in particular weeks of the year?
5. Pre-launch zeros: how many series start selling partway through? Do late starters behave differently?
6. Activity: how many series have sold nothing in the last 13 / 26 / 52 weeks? How often do such series start selling again?

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
6. **Seasonally adjusted MA4:** `MA4 × seasonal index for the target week`.
   - Caution: a full-strength historical index would have overshot Phase 1 just as LightGBM did (history +26–37% vs. actual +11%). It fixes *allocation*, not *magnitude*.
   - **[decide]** One pre-declared damping rule: index estimated from the same weeks in prior years, shrunk toward 1 or scaled by recent-vs-last-year growth. One variant only.
7. **Fixed 50/50 blend of (5) and (6).**
8. **Optional: CatBoost with Client/Warehouse/Product as categorical features,** only if (5)'s ablation shows the ID group adds value, or if there's time **[decide]**.

**Excluded, with reasons:**
- XGBoost: same model family, no specific hypothesis.
- A standalone two-stage `P(sale) × size` model: it still estimates the mean (playbook 8). Sale-probability ideas enter through activity features instead.
- Fitted ensemble weights: a selection risk.

## LightGBM-v2 feature groups (ablation order)

Add one group at a time to the frozen feature set, and keep a group only if it passes D014's keep rule: an improvement of at least a threshold set the same way as T1 (from benchmark noise, before any ablation runs) **and** a majority of origins won.

1. **Target-aligned annual history:** `y[o+h−52]`, `y[o+h−51]`, `y[o+h−53]`, mean of the three; plus `y[o+h−104]` where available.
2. **Activity / recency:** weeks since last sale; count of non-zero weeks in the last 4, 13, 26 and 52; mean size of non-zero sales; whether the series sold in the same season last year.
3. **Aggregates (all ≤ origin):** product, client and warehouse totals, recent growth, and the target-week seasonal index at each level.
4. **Trend and regime:** `MA4/MA12`; recent 13 weeks vs. the same 13 weeks last year (this year's rise against last year's — the Phase 1 magnitude problem).
5. **Identities:** Client, Warehouse, Product as categorical features.

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

## After the Phase 2 freeze

- Quantile forecasts, with calibration over cumulative lead-time demand (not per week, because of the zeros).
- A first inventory simulation as the bridge to VN2.
- Trim PROGRESS.md and move session history into `notes/archive/`.
