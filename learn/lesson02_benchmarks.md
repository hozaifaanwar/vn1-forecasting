# Lesson 2 — Benchmark ladder

Notebook: `learn/02_benchmarks.ipynb` (vn1 kernel). Code tested on the real data on 2026-11-17;
naive at the Phase 2 origin reproduces the official 0.506970.

**Why:** in VN1 our selection had no naive candidate; naive scored 0.5070 against our primary 0.7128 (D016).
A model that cannot beat the simple benchmarks adds no value (FVA, Vandeput, DFBP ch. 10–11).

| Name | Forecast for the next 13 weeks |
|---|---|
| naive | last week's sales |
| ma4 / ma13 / ma52 | average of the last 4 / 13 / 52 weeks |
| snaive | the same weeks one year (52 weeks) earlier |

## Step 1 — setup

```python
import numpy as np
import pandas as pd

DATA  = r"D:\vn1-forecasting\data\raw\vn1-forecasting-competition"
INDEX = ["Client", "Warehouse", "Product"]
H = 13

def read_weekly(path):
    t = pd.read_csv(path).set_index(INDEX)
    t.columns = pd.to_datetime(t.columns)
    return t

hist = pd.concat([read_weekly(DATA + r"\Phase 0 - Sales.csv"),
                  read_weekly(DATA + r"\Phase 1 - Sales.csv")], axis=1)
Y = hist.to_numpy()

def vn1_score(actual, forecast):
    error = forecast - actual
    return (np.abs(error).sum() + abs(error.sum())) / actual.sum()
```

## Step 2 — tiny practice

```python
tiny = np.array([[1, 2, 3, 4, 5, 6, 7, 8],
                 [0, 0, 5, 5, 0, 5, 5, 0]])
o = 5
h = 2
tiny[:, [o]]                                     # (a) why [o] and not o?
np.repeat(tiny[:, [o]], h, axis=1)               # (b) naive
tiny[:, o-3:o+1].mean(axis=1, keepdims=True)     # (c) ma4
tiny[:, o+1-4:o+1-4+h]                           # (d) snaive with season = 4
```

`tiny[:, o]` is 1D; `tiny[:, [o]]` keeps a column. `keepdims=True` keeps the column after `mean`.
A slice end is excluded, so "up to o" is `o+1`.

## Step 3 — benchmark function (fill the TODOs)

```python
def benchmarks(Y, o):
    """Forecasts made at origin column o (last known week), for weeks o+1 … o+H."""
    fc = {}
    fc["naive"] = ____      # TODO 1: last week repeated H times
    for k in (4, 13, 52):
        fc[f"ma{k}"] = ____ # TODO 2: average of the last k weeks, repeated (Y[:, o+1-k : o+1])
    fc["snaive"] = ____     # TODO 3: the same weeks 52 earlier, starting at o+1-52
    return fc
```

## Step 4 — score at the Jan 2023 origin

```python
o = hist.columns.get_loc(pd.Timestamp("2022-12-26"))
actual = Y[:, o+1 : o+1+H]
print(hist.columns[o+1].date(), "→", hist.columns[o+H].date())   # 2023-01-02 → 2023-03-27

rows = []
for name, f in benchmarks(Y, o).items():
    rows.append({"model": name,
                 "score": vn1_score(actual, f),
                 "bias":  (f - actual).sum() / actual.sum()})
pd.DataFrame(rows).sort_values("score").round(3)
```

## Step 5 — check against the official number

```python
p2 = read_weekly(DATA + r"\Phase 2 - Sales.csv").loc[hist.index]   # same row order
L = Y.shape[1] - 1                                                 # 2024-01-01
print(round(vn1_score(p2.to_numpy(), benchmarks(Y, L)["naive"]), 6))   # must be 0.50697
```

Phase 2 actuals were already seen (D016), so this is a code check only, not model selection.

## Prediction (before running)

At the Jan 2023 origin, which wins: naive or ma4, and why? (Think about the last week of December in the Lesson 1 plot.)
