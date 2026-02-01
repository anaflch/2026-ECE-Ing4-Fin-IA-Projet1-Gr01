import numpy as np
from FCC.src.fairness import compute_fairness

def test_fairness_outputs():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    A = np.array([0, 0, 1, 1])

    mf, gaps = compute_fairness(y_true, y_pred, A)
    assert "dp_diff" in gaps and "eo_diff" in gaps