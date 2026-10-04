import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from src.models.lightgbm_model import train_lgbm


def _toy_matrices(n=200, n_origins=2, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = np.clip(x1 * 5 + rng.normal(scale=0.5, size=n) + 10, 0, None)
    origin_date = np.where(np.arange(n) % n_origins == 0, "2020-01-06", "2020-02-03")
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y, "origin_date": origin_date})
    return df.iloc[: n // 2].reset_index(drop=True), df.iloc[n // 2 :].reset_index(drop=True)


def test_train_lgbm_pooled_mode_tracks_only_vn1_score():
    fit_mat, val_mat = _toy_matrices()
    model, preds = train_lgbm(
        fit_mat, val_mat, feature_cols=["x1", "x2"], n_estimators=100, origin_col=None,
    )
    assert list(model.evals_result_["valid_0"]) == ["vn1_score"]
    assert len(preds) == len(val_mat)
    assert (preds >= 0).all()  # predictions are clipped


def test_train_lgbm_origin_mode_tracks_only_origin_mean_score():
    fit_mat, val_mat = _toy_matrices()
    model, preds = train_lgbm(
        fit_mat, val_mat, feature_cols=["x1", "x2"], n_estimators=100, origin_col="origin_date",
    )
    assert list(model.evals_result_["valid_0"]) == ["vn1_score_origin_mean"]
    assert len(preds) == len(val_mat)


def test_train_lgbm_is_deterministic_given_random_state():
    fit_mat, val_mat = _toy_matrices()
    _, preds_a = train_lgbm(fit_mat, val_mat, feature_cols=["x1", "x2"], n_estimators=100, random_state=7)
    _, preds_b = train_lgbm(fit_mat, val_mat, feature_cols=["x1", "x2"], n_estimators=100, random_state=7)
    np.testing.assert_array_equal(preds_a, preds_b)


def test_eval_kwargs_match_the_installed_lightgbm_api():
    import inspect

    from src.models.lightgbm_model import _eval_kwargs

    params = inspect.signature(lgb.LGBMRegressor.fit).parameters
    kwargs = _eval_kwargs("X", "y")
    assert set(kwargs) <= set(params)
    assert kwargs == ({"eval_X": "X", "eval_y": "y"} if "eval_X" in params else {"eval_set": [("X", "y")]})
