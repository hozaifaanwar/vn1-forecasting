"""Score frozen wide-format submissions against released actuals.

Kept separate from the forecasting pipeline: nothing here can change a
forecast, it only reads frozen files and actuals and reports scores.
"""
import numpy as np
import pandas as pd

from .data import KEY
from .metrics import vn1_score_breakdown


def load_wide(path):
    """Read an official wide file (KEY columns + one column per week)."""
    wide = pd.read_csv(path).set_index(KEY)
    wide.columns = pd.to_datetime(wide.columns)
    return wide


def validate_actuals(actuals, reference_index, expected_dates):
    """Fail loudly unless the actuals cover exactly the expected series (in
    the organizers' row order) and weeks, with valid sales values."""
    assert actuals.index.equals(reference_index), "actuals' series or row order differ from Phase 0"
    assert list(actuals.columns) == list(expected_dates), "actuals' weeks differ from the forecast weeks"
    values = actuals.to_numpy(dtype=float)
    assert np.isfinite(values).all(), "actuals contain NaN or non-finite values"
    assert (values >= 0).all(), "actuals contain negative sales"


def organizers_score(submission, objective):
    """The organizers' own formula, transcribed from `data/raw/Functions for
    participants` — an independent cross-check on src.metrics.vn1_score."""
    assert (submission.index == objective.index).all()
    assert (submission.columns == objective.columns).all()
    abs_err = np.nansum(abs(submission - objective))
    err = np.nansum(submission - objective)
    return (abs_err + abs(err)) / objective.sum().sum()


def ma12_benchmark(history, columns):
    """The organizers' MA12 reference submission: mean of the last 12
    historical weeks, repeated flat across the forecast weeks."""
    level = np.nanmean(history.to_numpy()[:, -12:], axis=1).reshape(-1, 1)
    return pd.DataFrame(np.repeat(level, len(columns), axis=1), index=history.index, columns=columns)


def score_table(submissions, actuals, segments=None):
    """Score each named submission. Returns one row per (model, slice) where
    slice is "all" or, if `segments` (a Series indexed like actuals) is
    given, each segment. Segment scores are not additive: the bias term is
    an absolute value taken within each slice."""
    slices = {"all": np.ones(len(actuals), dtype=bool)}
    if segments is not None:
        assert segments.index.equals(actuals.index)
        slices.update({s: (segments == s).to_numpy() for s in sorted(segments.unique())})

    rows = []
    for name, sub in submissions.items():
        assert sub.index.equals(actuals.index) and list(sub.columns) == list(actuals.columns), name
        check = organizers_score(sub, actuals)
        for slice_name, mask in slices.items():
            b = vn1_score_breakdown(actuals.to_numpy()[mask], sub.to_numpy()[mask])
            if slice_name == "all":
                assert np.isclose(b["score"], check), f"{name}: vn1_score disagrees with the organizers' formula"
            rows.append({
                "model": name, "slice": slice_name, "score": b["score"],
                "mae_pct": b["abs_error_sum"] / b["total_actual"],
                "bias_pct": b["signed_bias"] / b["total_actual"],  # positive = under-forecast
                "total_actual": b["total_actual"], "total_predicted": b["total_predicted"],
                "demand_share": b["total_actual"] / actuals.to_numpy().sum(),
            })
    return pd.DataFrame(rows)
