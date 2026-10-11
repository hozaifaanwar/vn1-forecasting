import numpy as np
import pandas as pd
import pytest

from src.data import KEY
from src.scoring import ma12_benchmark, organizers_score, score_table, validate_actuals


def _wide(values, start="2023-10-09"):
    values = np.asarray(values, dtype=float)
    index = pd.MultiIndex.from_tuples([(0, 0, p) for p in range(len(values))], names=KEY)
    return pd.DataFrame(values, index=index, columns=pd.date_range(start, periods=values.shape[1], freq="W-MON"))


def test_score_table_matches_hand_calculation_and_organizers_formula():
    actuals = _wide([[10, 0], [5, 5]])
    sub = _wide([[8, 2], [5, 9]])
    # errors (actual - pred): 2, -2, 0, -4 -> abs sum 8, bias -4 -> (8 + 4) / 20
    table = score_table({"m": sub}, actuals)
    row = table.iloc[0]
    assert row["score"] == pytest.approx(0.6)
    assert row["mae_pct"] == pytest.approx(0.4)
    assert row["bias_pct"] == pytest.approx(-0.2)  # over-forecast
    assert organizers_score(sub, actuals) == pytest.approx(0.6)


def test_score_table_segments_cover_all_demand():
    actuals = _wide([[10, 0], [5, 5], [1, 0]])
    sub = _wide([[8, 2], [5, 9], [0, 0]])
    segments = pd.Series(["active", "active", "sparse"], index=actuals.index)
    table = score_table({"m": sub}, actuals, segments)
    assert set(table["slice"]) == {"all", "active", "sparse"}
    assert table.loc[table["slice"] != "all", "demand_share"].sum() == pytest.approx(1.0)


def test_ma12_benchmark_is_flat_mean_of_last_12_weeks():
    history = _wide([list(range(20)), [3] * 20], start="2023-05-22")
    columns = pd.date_range("2023-10-09", periods=13, freq="W-MON")
    bench = ma12_benchmark(history, columns)
    assert bench.shape == (2, 13)
    assert (bench.iloc[0] == np.mean(range(8, 20))).all()
    assert (bench.iloc[1] == 3).all()


def test_validate_actuals_rejects_wrong_order_weeks_and_nan():
    actuals = _wide([[10, 0], [5, 5]])
    dates = list(actuals.columns)
    validate_actuals(actuals, actuals.index, dates)
    with pytest.raises(AssertionError, match="row order"):
        validate_actuals(actuals.iloc[::-1], actuals.index, dates)
    with pytest.raises(AssertionError, match="weeks differ"):
        validate_actuals(actuals, actuals.index, dates[:1])
    bad = actuals.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(AssertionError, match="NaN"):
        validate_actuals(bad, actuals.index, dates)
