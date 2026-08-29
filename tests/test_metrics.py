import numpy as np
import pytest

from src.metrics import vn1_score, vn1_score_breakdown


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
