import numpy as np


def vn1_score_breakdown(y_true, y_pred):
    """Official VN1 metric, decomposed into its components for diagnosis.

    score = (sum|error| + |sum error|) / sum(actual)

    Splitting it out matters because two very different forecasts can land
    on the same score: one dominated by scattered error (high abs_error_sum,
    low bias_term), another dominated by systematic over/under-forecasting
    (the reverse). y_true/y_pred are array-like of matching shape, flattened
    internally so wide or long inputs both work.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    assert y_true.shape == y_pred.shape, f"shape mismatch: {y_true.shape} vs {y_pred.shape}"
    assert np.isfinite(y_true).all() and np.isfinite(y_pred).all(), "non-finite values in input"

    total_actual = y_true.sum()
    assert total_actual != 0, "total actual demand is zero; score is undefined"

    error = y_true - y_pred  # positive = under-forecast, negative = over-forecast
    abs_error_sum = np.abs(error).sum()
    signed_bias = error.sum()
    bias_term = np.abs(signed_bias)

    return {
        "total_actual": total_actual,
        "total_predicted": y_pred.sum(),
        "signed_bias": signed_bias,
        "abs_error_sum": abs_error_sum,
        "bias_term": bias_term,
        "score": (abs_error_sum + bias_term) / total_actual,
    }


def vn1_score(y_true, y_pred):
    """Official VN1 metric: (sum|error| + |sum error|) / sum(actual).

    Penalizes individual forecast error and aggregate bias together.
    Lower is better. Use vn1_score_breakdown() for the component values.
    """
    return vn1_score_breakdown(y_true, y_pred)["score"]
