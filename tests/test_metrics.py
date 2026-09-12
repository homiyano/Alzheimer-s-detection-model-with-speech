import numpy as np

from alzspeech.metrics import bootstrap_ci, classification_metrics, rmse


def test_classification_metrics_perfect():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["specificity"] == 1.0


def test_classification_metrics_all_wrong():
    y_true = [0, 1, 0, 1]
    y_pred = [1, 0, 1, 0]
    m = classification_metrics(y_true, y_pred)
    assert m["accuracy"] == 0.0
    assert m["specificity"] == 0.0


def test_rmse_zero_for_identical():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_rmse_known_value():
    assert abs(rmse([0.0, 0.0], [3.0, 4.0]) - 3.5355339) < 1e-5


def test_bootstrap_ci_bounds_contain_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=50)
    y_pred = y_true.copy()
    y_pred[:10] = 1 - y_pred[:10]  # inject some errors
    point, lower, upper = bootstrap_ci(y_true, y_pred, n_bootstrap=200)
    assert lower <= point <= upper


def test_bootstrap_ci_empty_input_returns_degenerate():
    point, lower, upper = bootstrap_ci([], [])
    assert point == lower == upper
