from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def _validated_arrays(y_true: Iterable[int], scores: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(y_true), dtype=np.int8)
    s = np.asarray(list(scores), dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s):
        raise ValueError("y_true and scores must be one-dimensional and have equal length")
    if len(y) == 0 or set(np.unique(y)) != {0, 1}:
        raise ValueError("both binary classes must be present")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or infinity")
    return y, s


def binary_metrics(
    y_true: Iterable[int], scores: Iterable[float], target_fpr: float = 0.05
) -> dict[str, float | int]:
    """Return AUROC and both achievable/interpolated TPR at a target FPR."""
    y, s = _validated_arrays(y_true, scores)
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    valid = fpr <= target_fpr + 1e-12
    achievable_tpr = float(tpr[valid].max()) if valid.any() else 0.0
    interpolated_tpr = float(np.interp(target_fpr, fpr, tpr))
    return {
        "n": int(len(y)),
        "n_negative": int((y == 0).sum()),
        "n_positive": int((y == 1).sum()),
        "auroc": float(roc_auc_score(y, s)),
        "tpr_at_fpr_5pct": achievable_tpr,
        "tpr_at_fpr_5pct_interpolated": interpolated_tpr,
    }


def bootstrap_intervals(
    y_true: Iterable[int],
    scores: Iterable[float],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260806,
) -> dict[str, dict[str, float | int]]:
    """Stratified non-parametric bootstrap intervals for the competition metrics."""
    y, s = _validated_arrays(y_true, scores)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {
        "auroc": [],
        "tpr_at_fpr_5pct": [],
        "tpr_at_fpr_5pct_interpolated": [],
    }
    for _ in range(n_bootstrap):
        sample = np.concatenate(
            [rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)]
        )
        metrics = binary_metrics(y[sample], s[sample])
        for key in collected:
            collected[key].append(float(metrics[key]))

    alpha = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, float | int]] = {}
    for key, values in collected.items():
        array = np.asarray(values)
        result[key] = {
            "confidence": confidence,
            "low": float(np.quantile(array, alpha)),
            "high": float(np.quantile(array, 1.0 - alpha)),
            "n_bootstrap": n_bootstrap,
        }
    return result


def fold_metrics(
    y_true: Iterable[int], scores: Iterable[float], folds: Iterable[int]
) -> list[dict[str, float | int]]:
    y, s = _validated_arrays(y_true, scores)
    fold_array = np.asarray(list(folds), dtype=np.int16)
    if len(fold_array) != len(y):
        raise ValueError("folds must have the same length as y_true")
    results = []
    for fold in sorted(np.unique(fold_array)):
        mask = fold_array == fold
        metrics = binary_metrics(y[mask], s[mask])
        metrics["fold"] = int(fold)
        results.append(metrics)
    return results

