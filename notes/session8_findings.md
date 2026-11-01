# Session 8 — demand findings (Phase 0 + Phase 1)

Verified facts from `notebooks/08_session8_demand.ipynb` (Q1–Q4, learner-run) and an independent
script run of Q5–Q6. Every number below was recomputed with a second implementation and matched.
Data: `data/processed/vn1_long_p0p1.parquet`, 183 weeks, 2020-07-06 → 2024-01-01.

The learner's own interpretation belongs in the notebook's "What I see / What it means" cells.
The "Questions for D014" here are open questions, not decisions.

## Q1 — What drove the Q4 2023 peak?

Uplift = mean of peak weeks (2023-11-20, 2023-12-11) minus mean of 2023-10-09 → 2023-11-13.

- Only 5,682 of 15,053 series (37.7%) rose. The risers added 155,825 units/week; fallers removed
  38,705, so the net peak-week change was about +117,120 units/week.
- Highly concentrated: the top 1% of series (150) produced 48.6% of the rise, the top 5% 81.1%,
  the top 10% 92.3%, the top 20% 98.3%.
- LightGBM over-forecast 84.4% of series in Phase 1 (D013), while only ~38% of series rose.

## Q2 — What happens in January–April? (the Phase 2 horizon)

Origin = first Monday of January; window = next 13 weeks.

| Year | Jan–Apr avg/week | prev 13 wks | last 4 wks (MA4 level) | vs prev 13 | MA4 too high by |
|---|---|---|---|---|---|
| 2021 | 132,816 | 145,332 | 147,986 | −8.6% | +11.4% |
| 2022 | 180,992 | 203,618 | 214,858 | −11.1% | +18.7% |
| 2023 | 247,196 | 265,520 | 284,740 | −6.9% | +15.2% |

- Within each year, Jan–Apr is roughly flat (no repeating within-window shape); levels grew
  strongly year on year. 2021 has a one-off dip in mid/late Feb 2021 (cause unknown).
- In all three years, an MA4 forecast at the January origin would have been 11–19% too high **in
  total** (bias part only; per-series accuracy needs the planned January backtests).
- This matches the off-limits retrospective's account of Phase 2 (D011). It is legitimate here
  because it comes only from Phase 0 data any 2024 competitor had; recorded for transparency.

## Q3 — Why was the 2023 Q4 rise weaker?

Origin = first Monday of October; Jul–Sep (13 wks) vs Oct–Dec (13 wks).

| Year | Q4 rise | Jul–Sep vs last year | Oct–Dec vs last year |
|---|---|---|---|
| 2020 | +28.5% | — | — |
| 2021 | +35.3% | +33.1% | +40.1% |
| 2022 | +33.4% | +32.3% | +30.4% |
| 2023 | +11.3% | +35.6% | +13.2% |

- The summer of 2023 grew normally; **Q4 2023 broke the trend**. Had Q4 grown like the summer it
  would have averaged ~360,000/week instead of 300,447.
- Series-level (2022 vs 2023): the share of series rising was similar (37.2% vs 38.6%); total
  rise from risers fell 99,400 → 75,376/week. Of the top 750 risers in 2023, 340 (45%) were also
  top 750 in 2022 (~37 expected by chance, so about 9× chance).
- 2022's top 750 risers added 79,685/week in Q4 2022 but only 23,679 in Q4 2023; 221 of them fell.
  All other series went from −13,278 to +6,858.
- By segment (net, all series): intermittent −32,507/week, active −8,778, sparse +5,415 (≈91% of
  the shortfall is intermittent). Segment labels mix in late starters (see Q5), so segment-level
  percentages are not comparable across years.
- Sales ≠ demand: the weak Q4 could be weaker demand or supply limits; VN1 has no inventory data.

## Q4 — Spikes

Spike = ≥5× the series' median non-zero week, ≥20 units, both neighbouring weeks <2× that level.

- 1,183 spike weeks in 795 series (5.3%), holding 0.9% of all units. Robust to the threshold:
  k=3 → 12.5% of series / 1.8% of units; k=10 → 2.0% / 0.5%.
- Share of series with a spike: active 10.5%, intermittent 7.0%, sparse 1.6%.
- Spikes recur in the same ISO weeks across years: ~28 (2022-07-11: 21, 2023-07-10: 31), ~41
  (2022-10-10: 17, 2023-10-09: 27), ~47 (2022-11-21: 23, 2023-11-20: 29). Cause unknown.
- Inside the Phase 2 window (ISO weeks 2–14), two weeks are mildly elevated in more than one year:
  week 6 (2022: 14, 2023: 17 spikes vs. typical weeks of 5 and 9) and week 13 (≈1.4–2× the year's
  typical week in 2021–2023). Spike counts also rise over time as the portfolio grows.

## Q5 — Late starters

- Only 2,224 series (14.8%) sold in the first data week; 61.0% first sold more than 12 months
  after the data starts. New first sales arrive steadily (≈600–1,300 series per quarter). The
  dataset contains only series that sold in Phase 0, so no series first sells in Phase 1.
- **62.0% of all Phase 0 zeros are pre-launch zeros** (before a series' first sale) — matching
  Session 3's leading-zero count (1,139,262 of 1,836,437).
- The segment label (zero rate over all Phase 0 weeks) misclassifies many series: counting only
  weeks after the first sale, **6,074 series (40.4%) change segment**. 757 "sparse" and 2,638
  "intermittent" series are active once they exist.
- Of 9,315 series that first sold after 2021-01-01 (with 26 weeks of follow-up): about half sell in
  any given week afterwards (51% at week 1, 44% at week 26); 11.4% never sold again within 26 weeks.
- Phase 1 demand by first-sale year: 2020 40.7%, 2021 22.5%, 2022 21.0%, 2023 15.8%.
- **3,727 series (24.8%) first sold within the 52 weeks before the Phase 2 origin**, carrying 15.7%
  of Phase 1 demand. For them a target-aligned "same week last year" lag is a pre-launch zero.

## Q6 — Activity

- At the Phase 2 origin, sold nothing in the last 13 weeks: 4,522 series (30.0%) — the same 4,522
  that sold nothing in Phase 1 (D013); last 26 weeks: 3,054 (20.3%); last 52 weeks: 1,726 (11.5%).
- Reactivation (origins every 4 weeks, series that had started and then went quiet): of series
  silent for 13 weeks, 23.4% sell something in the next 13 weeks, but they produce only 0.55% of
  next-13-week demand; silent 26 weeks → 16.2% / 0.26%; silent 52 weeks → 10.6% / 0.05%.
- Why the weekly zero share fell from 71.8% (Phase 0) to 56.2% (Phase 1): **an artefact of
  pre-launch zeros.** Among series that had already started selling, the zero share actually rose
  over time (22.1% in the first 13 weeks → 54.4% in Jul–Sep 2023 → 56.2% in Phase 1). Phase 0 looks
  emptier only because most series had not launched yet.

## Questions for D014 (to be answered by the learner, then fixed before any Phase 2 run)

1. Q2: in three seasons out of three, Jan–Apr fell below the January MA4 level. Should a seasonally
   adjusted MA4 (and any LightGBM variant) be checked specifically against this post-holiday drop
   in the Jan 2022 / Jan 2023 backtests?
2. Q1 + Q3: which products peak is fairly persistent (45% overlap, ~9× chance) but the size is not
   (last year's top risers rose ~70% less). Does that argue for target-aligned seasonal features
   that inform *which* series, with caution about *how much*?
3. Q5: 40% of series change segment once pre-launch zeros are excluded, and 24.8% have no full year
   of history at the Phase 2 origin. Should features use "weeks since first sale" and zero rates
   counted only after the first sale?
4. Q6: series silent for 13+ weeks produce ~0.5% of future demand. Does that justify forecasting
   (near-)zero for long-dormant series, and how much would that cut the volume LightGBM placed on
   non-selling series?
5. Q4: spikes are rare (<2% of volume). The big recurring spike weeks (28, 41, 47) are outside the
   Phase 2 window, but weeks 6 and 13 inside it are mildly elevated in more than one year. Worth a
   diagnostic in the January backtests, or low priority?

## Addendum — series-level January behaviour and where the error is (after the learner's idea)

The learner proposed using each series' own past January behaviour, and treating new products with
their own (short) history. Checked on Phase 0 + Phase 1 data:

- **Series-level January ratios persist.** Ratio = Jan–Apr avg ÷ last-4-weeks level at the January
  origin. Among series with a last-4 level of ≥5 / ≥20 / ≥50 units/week in both 2022 and 2023
  (1,868 / 1,051 / 593 series), the correlation of log ratios 2022 vs. 2023 is 0.66 / 0.75 / 0.79,
  and 73% / 77% / 81% fall on the same side of the portfolio ratio both years.
- **One-year test (2022 ratio → forecast Jan–Apr 2023), 1,051 series with level ≥20:** absolute
  error, portfolio ratio 69,039; own 2022 ratio 54,732; 50/50 mix of own and portfolio ratio 51,947
  (≈25% below portfolio). Bias: −10,259 / −25,054 / −17,657. One year only; higher-volume series only.
- **Where Phase 1 error came from, by behaviour after the first sale** (blend; LightGBM similar):
  truly active series = 88% of demand and 78% of absolute error (0.50 error per unit); truly
  intermittent = 12% / 20% (0.94); truly sparse = 0.2% / 1.7% (5.8). 61% of LightGBM's net
  over-forecast came from truly active series. The old labels (pre-launch zeros included) put
  "intermittent" at 49% of demand, which overstated its weight.

**Integrity note for D014:** this addendum and Q2 used actual Jan–Apr 2022 and 2023 outcomes, which
are the windows of the planned January season-matched backtests. Those backtests are therefore not
untouched for designs motivated here (the January-adjusted MA4, series-level ratios). D014 must state
this and weight the all-season cross-validation accordingly.

**Question for D014 (6):** replace the portfolio-only January multiplier with a 50/50 mix of each
series' own past January ratio and the portfolio ratio (portfolio ratio only where a series lacks
enough history), keeping it one candidate?
