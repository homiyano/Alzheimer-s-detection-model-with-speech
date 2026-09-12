"""Classification/regression metrics, with bootstrap confidence intervals.

Small-N statistical power is a documented pitfall (docs/research/sota_papers.md,
pitfall #8): ADReSSo test sets are 46-71 speakers, so a 3-4 point accuracy
swing is easily noise. `bootstrap_ci` is used throughout evaluation so that
headline numbers are never reported without an uncertainty estimate.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accuracy = float((y_true == y_pred).mean())
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
    }


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bootstrap_ci(
    y_true,
    y_pred,
    metric_fn=lambda yt, yp: classification_metrics(yt, yp)["accuracy"],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (point_estimate, lower, upper) via percentile bootstrap over samples."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    point = metric_fn(y_true, y_pred)

    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = (1 - ci) / 2
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1 - alpha))
    return point, lower, upper
