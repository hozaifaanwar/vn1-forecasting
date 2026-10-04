import inspect

import numpy as np
import lightgbm as lgb

from ..metrics import make_vn1_origin_mean_eval, vn1_lgb_eval

# LightGBM 4.7 deprecates fit(eval_set=...) in favour of eval_X/eval_y, which
# older versions don't accept. Use whichever this installation supports;
# both pass the same validation data, so results are identical.
_HAS_EVAL_XY = "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters


def _eval_kwargs(X, y):
    return {"eval_X": X, "eval_y": y} if _HAS_EVAL_XY else {"eval_set": [(X, y)]}


def train_lgbm(
    fit_mat,
    val_mat,
    feature_cols,
    target_col="target",
    objective="regression",
    categorical_feature=None,
    random_state=42,
    stopping_rounds=50,
    origin_col="origin_date",
    **lgb_params,
):
    """Fit an LGBMRegressor whose early stopping is driven *exclusively*
    by the real VN1 metric — never by a proxy like L1/MAE, and never
    diluted by LightGBM's own built-in metric for the objective riding
    alongside it (see DECISIONS.md D009/D010).

    Two things make that exclusivity real rather than assumed:
    - `metric="None"` on the estimator disables the objective's built-in
      metric (e.g. "l2" for objective="regression"), which LightGBM
      registers automatically otherwise.
    - `first_metric_only=True` on the early-stopping callback, so if a
      second metric ever does end up registered, it can't affect which
      iteration is chosen.

    If `val_mat` has an `origin_col` (e.g. multiple rolling-CV origins),
    early stopping uses the MEAN of per-origin VN1 scores
    (`make_vn1_origin_mean_eval`) rather than one score pooled across
    every row — pooling lets one origin's bias cancel another's during
    early stopping itself, not just at model-selection time. Pass
    `origin_col=None` to fall back to a single pooled score (e.g. when
    `val_mat` genuinely has only one origin).

    Returns `(model, clipped_predictions_on_val_mat)`. Asserts that only
    the intended metric was actually tracked, rather than assuming the
    LightGBM configuration above worked.
    """
    if origin_col is not None:
        eval_fn = make_vn1_origin_mean_eval(val_mat[origin_col].to_numpy())
        expected_metric_name = "vn1_score_origin_mean"
    else:
        eval_fn = vn1_lgb_eval
        expected_metric_name = "vn1_score"

    model = lgb.LGBMRegressor(
        objective=objective, metric="None", random_state=random_state, verbosity=-1, **lgb_params
    )
    model.fit(
        fit_mat[feature_cols], fit_mat[target_col],
        **_eval_kwargs(val_mat[feature_cols], val_mat[target_col]),
        eval_metric=eval_fn,
        categorical_feature=categorical_feature or [],
        callbacks=[lgb.early_stopping(stopping_rounds, first_metric_only=True, verbose=False)],
    )

    tracked = list(model.evals_result_["valid_0"])
    assert tracked == [expected_metric_name], (
        f"expected only {expected_metric_name!r} to be tracked for early stopping, "
        f"got {tracked} — metric='None'/first_metric_only did not fully exclude "
        "LightGBM's own built-in metric"
    )

    preds = np.clip(model.predict(val_mat[feature_cols]), 0, None)
    return model, preds
