"""Detection metrics, with the faulty class as the positive class."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def detection_metrics(faulty: np.ndarray, score: np.ndarray, tau: float, prefix: str = "") -> dict[str, float]:
    """Metrics of a detector that flags a window as faulty when its score is below ``tau``.

    Args:
        faulty: True labels (1 = faulty, 0 = healthy).
        score: Detector scores (high = healthy).
        tau: Decision threshold.
        prefix: Prefix of the returned keys.

    Returns:
        Accuracy, balanced accuracy, precision, recall, F1, false-alarm rate,
        confusion matrix (``tn``, ``fp``, ``fn``, ``tp``) and the area under
        the ROC curve of ``-score`` (threshold-free; ``nan`` with a single class).
    """
    y = np.asarray(faulty, dtype=bool)
    pred = np.asarray(score) < tau
    tp, fn = int(np.sum(pred & y)), int(np.sum(~pred & y))
    fp, tn = int(np.sum(pred & ~y)), int(np.sum(~pred & ~y))
    recall = tp / (tp + fn) if tp + fn else np.nan
    tnr = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    both = y.any() and (~y).any()
    out = {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": 0.5 * (recall + tnr) if both else np.nan,
        "precision": precision, "recall": recall, "f1": f1,
        "false_alarm_rate": fp / (tn + fp) if tn + fp else np.nan,
        "auc": float(roc_auc_score(y, -np.asarray(score))) if both else np.nan,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    return {f"{prefix}{k}": v for k, v in out.items()}

