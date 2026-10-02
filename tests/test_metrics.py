import numpy as np
import pytest

from src.metrics import (
    make_vn1_origin_mean_eval,
    vn1_lgb_eval,
    vn1_score,
    vn1_score_breakdown,
    vn1_score_by_group,
)


def test_vn1_score_hand_calculated_example():
    # error = actual - forecast = [-2, 2, 5]; abs sum = 9; signed sum = 5
    # total_actual = 60 -> score = (9 + 5) / 60
    y_true = [10, 20, 30]
    y_pred = [12, 18, 25]
    assert vn1_score(y_true, y_pred) == pytest.approx(14 / 60)


def test_vn1_score_zero_error_gives_zero_score():
    y_true = [5, 10, 15]
    assert vn1_score(y_true, y_true) == 0.0


def test_vn1_score_all_zero_forecast_gives_exactly_two():
    # error == actual everywhere, so abs_error_sum == |signed bias| == total_actual
    # score = (total_actual + total_actual) / total_actual = 2 — a sanity anchor.
    y_true = [3, 7, 0, 12]
    y_pred = [0, 0, 0, 0]
    assert vn1_score(y_true, y_pred) == pytest.approx(2.0)


def test_vn1_score_breakdown_components_match_hand_calculation():
    y_true = [10, 20, 30]
    y_pred = [12, 18, 25]
    b = vn1_score_breakdown(y_true, y_pred)
    assert b["total_actual"] == 60
    assert b["total_predicted"] == 55
    assert b["signed_bias"] == pytest.approx(5)
    assert b["abs_error_sum"] == pytest.approx(9)
    assert b["bias_term"] == pytest.approx(5)
    assert b["score"] == pytest.approx(14 / 60)
    assert b["score"] == vn1_score(y_true, y_pred)


def test_vn1_score_rejects_shape_mismatch():
    with pytest.raises(AssertionError, match="shape mismatch"):
        vn1_score([1, 2, 3], [1, 2])


def test_vn1_score_rejects_zero_total_actual():
    with pytest.raises(AssertionError, match="zero"):
        vn1_score([0, 0, 0], [1, 2, 3])


def test_vn1_score_rejects_non_finite_input():
    with pytest.raises(AssertionError, match="non-finite"):
        vn1_score([1, 2, np.inf], [1, 2, 3])


def test_vn1_score_by_group_matches_per_group_hand_calculation():
    # group A: y_true=[10,20], y_pred=[12,18] -> error=[-2,2], score=(4+0)/30
    # group B: y_true=[5],     y_pred=[0]     -> error=[5],    score=(5+5)/5 = 2.0
    y_true = [10, 20, 5]
    y_pred = [12, 18, 0]
    groups = ["A", "A", "B"]
    result = vn1_score_by_group(y_true, y_pred, groups)
    assert result["A"] == pytest.approx(4 / 30)
    assert result["B"] == pytest.approx(2.0)


def test_vn1_score_by_group_can_diverge_from_pooled_score():
    # Pooling can look perfect (biases cancel) while every group is wrong.
    y_true = [10, 10]
    y_pred = [15, 5]  # group A over-forecasts, group B under-forecasts by the same amount
    groups = ["A", "B"]
    pooled = vn1_score(y_true, y_pred)  # errors cancel: signed_bias -> 0, score = (10+0)/20 = 0.5
    per_group = vn1_score_by_group(y_true, y_pred, groups)
    assert pooled < per_group.mean()  # pooled hides the per-group error the split reveals
    assert per_group["A"] == pytest.approx(1.0)  # error=-5, abs_err=bias_term=5, /total_actual=10
    assert per_group["B"] == pytest.approx(1.0)  # error=+5, symmetric


def test_vn1_lgb_eval_matches_vn1_score_on_clipped_predictions():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, -5.0, 25.0])  # a negative prediction, as an unclipped model could emit
    name, value, is_higher_better = vn1_lgb_eval(y_true, y_pred)
    assert name == "vn1_score"
    assert is_higher_better is False
    assert value == pytest.approx(vn1_score(y_true, np.clip(y_pred, 0, None)))
    assert value != pytest.approx(vn1_score(y_true, y_pred))  # confirms clipping actually matters here


def _toy_regression_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = np.clip(X[:, 0] * 5 + rng.normal(scale=0.5, size=200) + 10, 0, None)
    return X[:150], X[150:], y[:150], y[150:]


def test_vn1_lgb_eval_without_metric_none_lets_l2_ride_along():
    """Documents the bug this project actually hit: without metric="None",
    LightGBM's own built-in metric for the objective (here "l2") stays
    registered alongside the custom eval_metric, so early stopping is not
    exclusively driven by the VN1 score, contrary to what passing only
    `eval_metric=vn1_lgb_eval` would suggest."""
    lgb = pytest.importorskip("lightgbm")
    X_train, X_val, y_train, y_val = _toy_regression_data()

    model = lgb.LGBMRegressor(objective="regression", n_estimators=200, verbosity=-1, random_state=0)
    model.fit(
        X_train, y_train, eval_X=X_val, eval_y=y_val,
        eval_metric=vn1_lgb_eval,
        callbacks=[lgb.early_stopping(10, verbose=False)],
    )
    tracked = list(model.evals_result_["valid_0"])
    assert "l2" in tracked  # the unwanted, auto-added built-in metric
    assert "vn1_score" in tracked
    assert tracked != ["vn1_score"]  # confirms it is NOT exclusively VN1


def test_vn1_lgb_eval_with_metric_none_is_exclusively_vn1():
    """The fix: metric="None" + first_metric_only=True makes early
    stopping exclusively driven by the real VN1 score."""
    lgb = pytest.importorskip("lightgbm")
    X_train, X_val, y_train, y_val = _toy_regression_data()

    model = lgb.LGBMRegressor(
        objective="regression", metric="None", n_estimators=200, verbosity=-1, random_state=0
    )
    model.fit(
        X_train, y_train, eval_X=X_val, eval_y=y_val,
        eval_metric=vn1_lgb_eval,
        callbacks=[lgb.early_stopping(10, first_metric_only=True, verbose=False)],
    )
    assert list(model.evals_result_["valid_0"]) == ["vn1_score"]
    assert model.best_iteration_ is not None
    assert model.best_iteration_ > 0


def test_make_vn1_origin_mean_eval_scores_mean_not_pooled():
    # Two groups: one perfect (score 0), one all-zero-forecast (score 2.0).
    # Pooled score would NOT be a simple average of 0 and 2.0; the mean-of-
    # groups eval function must report exactly (0 + 2.0) / 2 = 1.0.
    y_true = np.array([10.0, 20.0, 5.0, 5.0])
    y_pred = np.array([10.0, 20.0, 0.0, 0.0])
    groups = np.array(["A", "A", "B", "B"])

    eval_fn = make_vn1_origin_mean_eval(groups)
    name, value, is_higher_better = eval_fn(y_true, y_pred)

    assert name == "vn1_score_origin_mean"
    assert is_higher_better is False
    assert value == pytest.approx(1.0)
    assert value != pytest.approx(vn1_score(y_true, y_pred))  # pooled score would differ


def test_make_vn1_origin_mean_eval_plugs_into_lightgbm_exclusively():
    lgb = pytest.importorskip("lightgbm")
    X_train, X_val, y_train, y_val = _toy_regression_data()
    groups = np.where(np.arange(len(y_val)) < len(y_val) // 2, "early", "late")

    model = lgb.LGBMRegressor(
        objective="regression", metric="None", n_estimators=200, verbosity=-1, random_state=0
    )
    model.fit(
        X_train, y_train, eval_X=X_val, eval_y=y_val,
        eval_metric=make_vn1_origin_mean_eval(groups),
        callbacks=[lgb.early_stopping(10, first_metric_only=True, verbose=False)],
    )
    assert list(model.evals_result_["valid_0"]) == ["vn1_score_origin_mean"]
