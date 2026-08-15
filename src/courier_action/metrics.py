"""Aggregate classification metrics used by the evaluation protocol."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(
    y_true: Iterable[int], probabilities: Iterable[float], threshold: float = 0.5
) -> dict[str, float]:
    """Compute fold metrics from probabilities at one pre-specified threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    y = np.asarray(list(y_true), dtype=np.int8)
    proba = np.asarray(list(probabilities), dtype=float)
    if y.ndim != 1 or proba.ndim != 1 or len(y) != len(proba) or len(y) == 0:
        raise ValueError("y_true and probabilities must be non-empty one-dimensional arrays")
    if not np.isfinite(proba).all():
        raise ValueError("probabilities must be finite")
    if np.unique(y).size != 2:
        raise ValueError("AUC requires both target classes in every validation fold")

    predictions = (proba >= threshold).astype(np.int8)
    return {
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "auc": float(roc_auc_score(y, proba)),
        "accuracy": float(accuracy_score(y, predictions)),
    }


def aggregate_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Aggregate fold metrics as mean and sample standard deviation."""

    if not fold_metrics:
        raise ValueError("at least one fold is required")
    names = ("f1", "auc", "accuracy")
    output: dict[str, dict[str, float]] = {}
    for name in names:
        values = np.asarray([metric[name] for metric in fold_metrics], dtype=float)
        output[name] = {
            "mean": round(float(values.mean()), 8),
            "std": round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 8),
        }
    return output
