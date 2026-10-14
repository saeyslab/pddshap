import h5py
import numpy as np
import pandas as pd
from numpy import typing as npt


def save_predictions(arr: npt.NDArray, path: str) -> None:
    """Save a 1-D or 2-D prediction array as a CSV with named columns."""
    if arr.ndim == 1:
        df = pd.DataFrame(arr, columns=["prediction"])
    else:
        df = pd.DataFrame(arr, columns=[f"output_{i}" for i in range(arr.shape[1])])
    df.to_csv(path, index=False)


def save_shap_values(arr: npt.NDArray, feature_names: list[str], path: str) -> None:
    """Save SHAP values (rows, features, outputs) to an HDF5 file."""
    if arr.ndim != 3:
        raise ValueError(
            f"Expected SHAP values to have 3 dimensions, got shape {arr.shape}."
        )

    _, num_features, _ = arr.shape
    if num_features != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(num_features)]

    with h5py.File(path, "w") as h5f:
        h5f.create_dataset("values", data=arr)
        str_dtype = h5py.string_dtype(encoding="utf-8")
        h5f.create_dataset(
            "feature_names", data=np.array(feature_names, dtype=str_dtype)
        )


def load_shap_values(path: str) -> tuple[npt.NDArray, list[str]]:
    """Load SHAP values previously saved by :func:`save_shap_values`."""
    with h5py.File(path, "r") as h5f:
        values = h5f["values"][()]
        feature_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in h5f["feature_names"][()]]
    return values, feature_names
