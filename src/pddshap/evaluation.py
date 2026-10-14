import time
from typing import Any, Callable

import numpy as np
from numpy import typing as npt
from scipy import stats
from sklearn.metrics import r2_score


def compare_values(reference: npt.NDArray, approx: npt.NDArray) -> dict[str, float]:
    """
    Compare an array of approximate values (e.g. PDD-SHAP values, or predictions)
    against a reference array of the same shape (e.g. ground-truth or baseline-explainer
    values), flattening both before computing summary statistics.

    :return: dict with keys "r2", "pearson", "spearman".
    """
    ref_flat = np.asarray(reference).ravel()
    approx_flat = np.asarray(approx).ravel()
    if ref_flat.shape != approx_flat.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs approx {approx.shape}"
        )
    return {
        "r2": r2_score(ref_flat, approx_flat),
        "pearson": stats.pearsonr(ref_flat, approx_flat).statistic,
        "spearman": stats.spearmanr(ref_flat, approx_flat).statistic,
    }


def time_call(fn: Callable[..., Any], *args, **kwargs) -> tuple[Any, float]:
    """Call ``fn(*args, **kwargs)``, returning ``(result, elapsed_seconds)``."""
    start = time.time()
    result = fn(*args, **kwargs)
    return result, time.time() - start
