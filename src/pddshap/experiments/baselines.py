import numpy as np
from numpy import typing as npt
from shap import KernelExplainer, PermutationExplainer, SamplingExplainer

EXPLAINERS = ["permutation", "kernel", "sampling"]


def compute_shapley_values(
    pred_fn, X_bg: npt.NDArray, X: npt.NDArray, explainer: str
) -> npt.NDArray:
    """
    Compute baseline Shapley values for ``pred_fn`` on ``X`` using one of the
    `shap` package's explainers, normalized to shape ``(rows, features, outputs)``.
    """
    if explainer == "permutation":
        values = PermutationExplainer(pred_fn, X_bg)(X).values
    elif explainer == "kernel":
        values = KernelExplainer(pred_fn, X_bg).shap_values(X)
        if isinstance(values, list):
            return np.stack(values, axis=-1)
        return np.expand_dims(values, -1)
    elif explainer == "sampling":
        values = SamplingExplainer(pred_fn, X_bg)(X).values
    else:
        raise ValueError(f"Unknown explainer '{explainer}'")
    return values if values.ndim == 3 else np.expand_dims(values, -1)
