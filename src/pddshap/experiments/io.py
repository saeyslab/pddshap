import h5py
import numpy as np
import pandas as pd
from numpy import typing as npt


def save_predictions(arr: npt.NDArray, path: str) -> None:
    """
    Save a 1-D or 2-D prediction array as a CSV with named columns.

    Parameters
    ----------
    arr : numpy.ndarray
        Predictions to save, 1-D (single output) or 2-D (multiple outputs).
    path : str
        Destination CSV path.
    """
    if arr.ndim == 1:
        df = pd.DataFrame(arr, columns=["prediction"])
    else:
        df = pd.DataFrame(arr, columns=[f"output_{i}" for i in range(arr.shape[1])])
    df.to_csv(path, index=False)


def save_shap_values(arr: npt.NDArray, feature_names: list[str], path: str) -> None:
    """
    Save SHAP values to an HDF5 file.

    Parameters
    ----------
    arr : numpy.ndarray of shape (rows, features, outputs)
        SHAP values to save.
    feature_names : list of str
        Name of each feature column; must match `arr.shape[1]`, otherwise
        placeholder names are generated instead.
    path : str
        Destination HDF5 path.
    """
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
    """
    Load SHAP values previously saved by `save_shap_values`.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.

    Returns
    -------
    tuple of (numpy.ndarray, list of str)
        SHAP values of shape (rows, features, outputs), and the feature names.
    """
    with h5py.File(path, "r") as h5f:
        values = h5f["values"][()]
        feature_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in h5f["feature_names"][()]]
    return values, feature_names
