# Lesson 1 — Step 3: first sale and silent series

Merged version (Claude + Codex), tested on the real data on 2026-11-15 (pandas 2.3.3).
Run in `learn/01_eda.ipynb` after Step 2 (`hist`, `plt` already defined).

**Why:** pre-launch zeros are "product did not exist", not "no sales"; silent series
get over-forecast. In the VN1 ablation (D018) trimming leading zeros was worth +0.016
and zero/recency features +0.025.

## Tiny practice first

```python
tiny = pd.DataFrame(
    [[0, 0, 3, 0],      # A: launches in week 3
     [5, 0, 2, 0],      # B: sells from the start
     [0, 0, 0, 0]],     # C: never sells
    columns=pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22"]),
)

t_sold     = tiny.gt(0)                 # True/False table, labels kept
t_has_sale = t_sold.any(axis=1)         # [T, T, F]
t_first    = t_sold.idxmax(axis=1)      # first True's COLUMN NAME (date); C wrongly gets 2024-01-01
t_first.where(t_has_sale)               # C -> NaT

first_week = tiny.columns[0]
never  = ~t_has_sale
start  = t_has_sale & t_first.eq(first_week)
later  = t_has_sale & t_first.gt(first_week)
assert never.sum() + start.sum() + later.sum() == len(tiny)

t_silent      = ~t_sold.iloc[:, -2:].any(axis=1)   # nothing in the last 2 weeks
t_went_silent = t_has_sale & t_silent              # sold before, silent now
```

Before running: write down what each line returns for A, B, C.

## Step 3a — when did each series start selling?

```python
sold = hist.gt(0)
has_sale = sold.any(axis=1)

# idxmax gives the first True's date; all-zero rows would wrongly get week 1, so mask them.
first_sale = sold.idxmax(axis=1).where(has_sale)

first_week = hist.columns[0]
never_sold     = ~has_sale
from_start     = has_sale & first_sale.eq(first_week)
launched_later = has_sale & first_sale.gt(first_week)
assert never_sold.sum() + from_start.sum() + launched_later.sum() == len(hist)

print(f"Total series:      {len(hist):,}")
print(f"Never sold:        {never_sold.sum():,}")
print(f"Sold from week 1:  {from_start.sum():,}")
print(f"Launched later:    {launched_later.sum():,}  "
      f"({launched_later.mean():.1%} of all, {launched_later[has_sale].mean():.1%} of sold)")

first_sale.dt.to_period("M").value_counts().sort_index().plot(
    kind="bar", figsize=(14, 3), title="First-sale month (series count)")
plt.show()
```

## Step 3b — which series are silent now?

```python
DEAD_WEEKS = 8

silent      = ~sold.iloc[:, -DEAD_WEEKS:].any(axis=1)
went_silent = has_sale & silent          # sold before, nothing in the last DEAD_WEEKS

print(f"Silent last {DEAD_WEEKS} weeks: {silent.sum():,} ({silent.mean():.1%})")
print(f"  of which sold before:   {went_silent.sum():,}")
print(f"  of which never sold:    {(silent & never_sold).sum():,}")
```

## Prediction (write before running)

1. Launched later: what % of sold series?
2. Silent for 8 weeks: what %?
3. First-sale plot: spread out, or a spike in one month? (Hint: think of the 7, 7, 7, 7 values seen in Step 1 — a data artifact.)

## Notes

- pandas vs numpy: EDA and reports in pandas (labels, dates); heavy repeated math in numpy (as in `vn1_fresh.py`).
- `idxmax` returns a label, numpy `argmax` a position; both give the first column for an all-False row.
- `.loc` slices by label, `.iloc` by position. `assert` stops with an error when a check fails.
