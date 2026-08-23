import numpy as np


def vn1_score(y_true, y_pred):
    """Official VN1 metric: (sum|error| + |sum error|) / sum(actual).

    Penalizes individual forecast error and aggregate bias together.
    Lower is better. y_true/y_pred are array-like of matching shape,
    flattened internally so wide or long inputs both work.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    return (np.abs(error).sum() + np.abs(error.sum())) / y_true.sum()
