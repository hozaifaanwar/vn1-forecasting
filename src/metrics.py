import numpy as np
import pandas as pd


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


def vn1_score_by_group(y_true, y_pred, groups):
    """vn1_score computed separately within each distinct value of `groups`,
    returned as a Series indexed by group value.

    Pooling everything into one aggregate score lets one group's bias
    cancel another's, and hides how consistently a model wins or loses
    group by group — e.g. scoring a rolling-CV fold as one number hides
    per-origin behavior; grouping by origin here answers "how many origins
    did we actually win, not just the pooled average."
    """
    df = pd.DataFrame({
        "y_true": np.asarray(y_true, dtype=float),
        "y_pred": np.asarray(y_pred, dtype=float),
        "group": np.asarray(groups),
    })
    return df.groupby("group").apply(
        lambda g: vn1_score(g["y_true"], g["y_pred"]), include_groups=False
    )


def vn1_lgb_eval(y_true, y_pred):
    """Custom LightGBM eval metric: the official VN1 score itself, on
    clipped predictions (matching how predictions are actually used
    downstream), not a proxy like L1/MAE.

    Pass as `eval_metric=vn1_lgb_eval` so early stopping picks the
    iteration that minimizes the real target metric. Using "l1" instead
    is not leakage, but it is objective misalignment: LightGBM would then
    pick the iteration with the lowest MAE, which is not necessarily the
    lowest VN1 score (VN1 also penalizes aggregate bias).

    Returns the (name, value, is_higher_better) tuple LightGBM expects.
    """
    y_pred_clipped = np.clip(y_pred, 0, None)
    return "vn1_score", vn1_score(y_true, y_pred_clipped), False


def make_vn1_origin_mean_eval(groups):
    """Factory for a custom LightGBM eval function that early-stops on the
    MEAN of per-group VN1 scores (e.g. per rolling-CV origin), not one
    pooled score across every row in the validation set.

    vn1_lgb_eval alone still pools the whole validation set into a single
    score — exactly the bias-cancellation issue vn1_score_by_group exists
    to avoid at model-selection time, just not yet during early stopping
    itself. This closes that gap: `groups` is captured once (e.g. the
    validation set's origin_date column) and every call during training
    scores each group separately, then reports the mean.

    Must be paired with `metric="None"` on the estimator and
    `first_metric_only=True` on the early-stopping callback — otherwise
    LightGBM's own built-in metric for the objective (e.g. "l2") stays
    registered alongside this one and can influence which iteration early
    stopping settles on.
    """
    groups = np.asarray(groups)

    def _eval(y_true, y_pred):
        y_pred_clipped = np.clip(y_pred, 0, None)
        scores = vn1_score_by_group(y_true, y_pred_clipped, groups)
        return "vn1_score_origin_mean", scores.mean(), False

    return _eval
