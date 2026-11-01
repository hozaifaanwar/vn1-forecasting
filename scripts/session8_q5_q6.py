"""Session 8 Q5 (late starters) and Q6 (activity), with independent cross-checks.

Run: python scripts/session8_q5_q6.py  (vn1 env). Findings: notes/session8_findings.md.
Plots go to reports/figures/.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parent
while not (ROOT / "CONTEXT.md").is_file():
    if ROOT == ROOT.parent:
        raise FileNotFoundError("Could not find CONTEXT.md in any parent directory")
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from src.data import KEY
from src.features import classify_segment

SP = str(ROOT / "reports" / "figures")
os.makedirs(SP, exist_ok=True)
df = pd.read_parquet("data/processed/vn1_long_p0p1.parquet")
PHASE1_START = pd.Timestamp("2023-10-09")
sales = df.pivot(index=KEY, columns="date", values="sales")
weeks = sales.columns
segment = classify_segment((sales.loc[:, weeks < PHASE1_START] == 0).mean(axis=1))
vals = sales.to_numpy()
assert sales.shape == (15053, 183)

# ---------------- Q5: late starters ----------------
print("=" * 70, "\nQ5 LATE STARTERS")
assert (vals > 0).any(axis=1).all()
first_idx = (vals > 0).argmax(axis=1)
first_sale = pd.Series(weeks[first_idx], index=sales.index)
# cross-check first sale via long frame
chk = df[df["sales"] > 0].groupby(KEY)["date"].min().reindex(sales.index)
assert (chk == first_sale).all(), "first-sale cross-check failed"
# every series' first sale is inside Phase 0 (CONTEXT says every series sold in Phase 0)
assert (first_sale < PHASE1_START).all()

from_start = (first_idx == 0)
print(f"Selling in the first week (2020-07-06): {from_start.sum():,} ({from_start.mean():.1%})")
for m in [3, 6, 12, 24]:
    late = first_sale > weeks[0] + pd.DateOffset(months=m)
    print(f"First sale > {m:>2} months after start: {late.sum():,} ({late.mean():.1%})")
by_year = first_sale.dt.year.value_counts().sort_index()
print("First sale by calendar year:", by_year.to_dict())
by_q = first_sale.dt.to_period("Q").value_counts().sort_index()

# Q5b: zero rate after first sale (Phase 0 only, same window as the segment label)
p0 = sales.loc[:, weeks < PHASE1_START].to_numpy()
alive = np.arange(p0.shape[1]) >= first_idx[:, None]
zr_alive = ((p0 == 0) & alive).sum(axis=1) / alive.sum(axis=1)
# cross-check with a plain loop on a sample
rng = np.random.default_rng(0)
for i in rng.choice(len(p0), 200, replace=False):
    assert np.isclose(zr_alive[i], (p0[i, first_idx[i]:] == 0).mean())
seg_alive = classify_segment(pd.Series(zr_alive, index=sales.index))
ct = pd.crosstab(segment.rename("label: all Phase 0 weeks"), seg_alive.rename("after first sale"), margins=True)
print("\nSegment label vs. segment counted only after first sale:\n", ct)
moved = (segment != seg_alive)
print(f"Series whose segment changes once pre-launch zeros are excluded: {moved.sum():,} ({moved.mean():.1%})")

# Q5c: behaviour in the 26 weeks after first sale (series starting after 2021-01-01)
n_after = 26
elig = np.where((first_sale > pd.Timestamp("2021-01-01")).to_numpy() & (vals.shape[1] - first_idx > n_after))[0]
aligned = np.vstack([vals[i, first_idx[i]: first_idx[i] + n_after + 1] for i in elig])
share_selling = (aligned > 0).mean(axis=0)
print(f"\nLate starters with 26 weeks of follow-up: {len(elig):,}")
print("Share selling k weeks after first sale:", {k: f"{share_selling[k]:.0%}" for k in [0, 1, 2, 4, 8, 13, 26]})
never_again = (aligned[:, 1:] == 0).all(axis=1).mean()
print(f"Share that sold NOTHING in the 26 weeks after their first sale: {never_again:.1%}")

# Q5d: Phase 1 demand share by first-sale year; series without a full year of history at the Phase 2 origin
p1_units = sales.loc[:, weeks >= PHASE1_START].sum(axis=1)
cohort = first_sale.dt.year
tab = pd.DataFrame({"series": cohort.value_counts().sort_index(),
                    "share of Phase 1 demand": (p1_units.groupby(cohort).sum() / p1_units.sum()).round(3)})
print("\nBy first-sale year:\n", tab)
assert np.isclose(tab["share of Phase 1 demand"].sum(), 1.0, atol=0.01)
origin2 = pd.Timestamp("2024-01-01")
recent = first_sale > origin2 - pd.Timedelta(weeks=52)
print(f"First sold within 52 weeks before the Phase 2 origin: {recent.sum():,} ({recent.mean():.1%}), "
      f"{p1_units[recent].sum() / p1_units.sum():.1%} of Phase 1 demand")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(by_q.index.astype(str), by_q.values, marker="o")
ax.set_title("Q5: series by quarter of first sale"); ax.set_ylabel("series"); ax.grid(alpha=0.3)
plt.xticks(rotation=45); fig.tight_layout(); fig.savefig(f"{SP}/\q5_first_sale.png", dpi=80)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(n_after + 1), share_selling, marker="o")
ax.set_title("Q5: after the first sale, share of late starters selling each week")
ax.set_xlabel("weeks since first sale"); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(f"{SP}/\q5_ramp.png", dpi=80)

# ---------------- Q6: activity ----------------
print("\n" + "=" * 70, "\nQ6 ACTIVITY")
def dormant_at(origin_idx, n):
    """True if the series sold nothing in the n weeks up to and including origin_idx."""
    return (vals[:, origin_idx - n + 1: origin_idx + 1] == 0).all(axis=1)

o2 = weeks.get_loc(origin2)  # Phase 2 origin = last week, index 182
for n in [13, 26, 52]:
    d = dormant_at(o2, n)
    # cross-check with pandas
    d2 = (sales.iloc[:, o2 - n + 1: o2 + 1].sum(axis=1) == 0).to_numpy()
    assert (d == d2).all()
    print(f"At the Phase 2 origin, sold nothing in the last {n:>2} weeks: {d.sum():,} series ({d.mean():.1%})")

# Reactivation: at many past origins, of series dormant for n weeks, what share sold in the next 13 weeks,
# and how much volume did they produce (share of all next-13-week demand)?
origins = range(60, len(weeks) - 13, 4)
rows = []
for o in origins:
    nxt = vals[:, o + 1: o + 14]
    total_next = nxt.sum()
    for n in [13, 26, 52]:
        if o - n + 1 < 0:
            continue
        d = dormant_at(o, n)
        # exclude series that had not started yet (no sale ever before origin): they are pre-launch, not dormant
        started = first_idx <= o
        d = d & started
        if d.sum() == 0:
            continue
        woke = (nxt[d] > 0).any(axis=1)
        rows.append({"origin": weeks[o], "dormant_weeks": n, "n_dormant": int(d.sum()),
                     "share_reactivating": woke.mean(), "share_of_next13_demand": nxt[d].sum() / total_next})
react = pd.DataFrame(rows)
summary = react.groupby("dormant_weeks")[["n_dormant", "share_reactivating", "share_of_next13_demand"]].mean()
print("\nAverage over origins every 4 weeks (series that had already started, then went quiet):\n",
      summary.assign(share_reactivating=summary.share_reactivating.map("{:.1%}".format),
                     share_of_next13_demand=summary.share_of_next13_demand.map("{:.2%}".format),
                     n_dormant=summary.n_dormant.round(0).astype(int)))

# Why did the weekly zero share fall from 71.8% (Phase 0) to 56.2% (Phase 1)?
zero_share_week = (sales == 0).mean()
started_by_week = pd.Series((first_idx[:, None] <= np.arange(len(weeks))).sum(axis=0), index=weeks)
alive_mask = first_idx[:, None] <= np.arange(len(weeks))
zero_among_started = pd.Series(((vals == 0) & alive_mask).sum(axis=0) / alive_mask.sum(axis=0), index=weeks)
p0m, p1m = weeks < PHASE1_START, weeks >= PHASE1_START
print(f"\nZero share, all series:           Phase 0 {(vals[:, p0m] == 0).mean():.1%}   Phase 1 {(vals[:, p1m] == 0).mean():.1%}")
print(f"Zero share, only series already started: Phase 0 "
      f"{((vals[:, p0m] == 0) & alive_mask[:, p0m]).sum() / alive_mask[:, p0m].sum():.1%}   "
      f"Phase 1 {((vals[:, p1m] == 0) & alive_mask[:, p1m]).sum() / alive_mask[:, p1m].sum():.1%}")
print(f"Pre-launch zeros as a share of all Phase 0 zeros: "
      f"{((vals[:, p0m] == 0) & ~alive_mask[:, p0m]).sum() / (vals[:, p0m] == 0).sum():.1%}")
for label, w in [("first 13 wks of data", weeks[:13]), ("Jul-Sep 2023 (last 13 Phase 0)", weeks[p0m][-13:]), ("Phase 1", weeks[p1m])]:
    print(f"  {label:32s} zero share all {zero_share_week[w].mean():.1%} | among started {zero_among_started[w].mean():.1%} | started series {started_by_week[w].mean():,.0f}")

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(weeks, zero_share_week.values, label="all series")
ax.plot(weeks, zero_among_started.values, label="only series that had already started selling")
ax.axvline(PHASE1_START, color="red", ls="--", label="Phase 1 start")
ax.set_title("Q6: share of series with zero sales each week"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{SP}/\q6_zero_share.png", dpi=80)
print("\nplots saved")
