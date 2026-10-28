import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy import typing as npt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: npt.NDArray
    y_test: npt.NDArray
    pred_type: str


def load_dataset(
    data_dir: str,
    openml_data_id: int | None = None,
    test_size: float = 0.2,
    train_test_split_random_state: int = 42
) -> DatasetSplit:
    """Load a train/test split for an OpenML dataset, fetching and caching it if needed.

    If no cached copy of the dataset matching ``test_size`` and
    ``train_test_split_random_state`` exists under ``data_dir``, the dataset is downloaded
    from OpenML, cleaned, encoded, split, and cached to disk. Otherwise, the cached CSV
    files are read directly.

    Parameters
    ----------
    data_dir : str
        Root directory under which cached datasets are stored, one subdirectory per
        ``openml_data_id``.
    openml_data_id : int or None, optional
        The OpenML dataset ID to fetch (passed as ``data_id`` to
        :func:`sklearn.datasets.fetch_openml`).
    test_size : float, optional
        Fraction of the dataset to hold out as the test set, by default 0.2.
    train_test_split_random_state : int, optional
        Random state used for the train/test split, by default 42.

    Returns
    -------
    DatasetSplit
        The train/test feature and label arrays, along with the inferred prediction
        type (``"regression"`` or ``"classification"``).
    """
    ds_dir = os.path.join(data_dir, str(openml_data_id))

    # If the dataset is not found under the ``data_dir``,
    # or if it is found but the test size or random state are different,
    # then it must be re-fetched from openml and re-split into train and test set.
    meta_json_path = os.path.join(ds_dir, "meta.json")
    needs_fetch = True
    if os.path.isfile(meta_json_path):
        with open(meta_json_path, "r") as fp:
            meta = json.load(fp)
        meta_test_size = meta.get("test_size")
        meta_random_state = meta.get("train_test_split_random_state")
        if meta_test_size == test_size and meta_random_state == train_test_split_random_state:
            needs_fetch = False

    #################################
    # Download dataset if necessary #
    #################################
    if needs_fetch:
        ds = fetch_openml(data_id=openml_data_id)
        pred_type = "regression" if pd.api.types.is_numeric_dtype(ds["target"]) else "classification"
        # Drop nan rows
        y = ds.target[ds.data.notnull().all(axis=1)].to_numpy()
        df = ds.data.dropna().reset_index(drop=True)
        if pred_type == "classification":
            y = LabelEncoder().fit_transform(y)
        else:
            y = StandardScaler().fit_transform(y.reshape(-1, 1))

        col_dfs = []
        for feat_name in df.columns:
            if df.dtypes[feat_name] in ["object", "bool", "category"]:
                encoder = OrdinalEncoder()
                dtype = "int8"
            elif df.dtypes[feat_name] in ["int64", "float64", "int32", "float32"]:
                encoder = StandardScaler()
                dtype = "float32"
            else:
                raise ValueError("Unrecognized dtype in dataframe")
            col_dfs.append(
                pd.DataFrame(
                    encoder.fit_transform(df[[feat_name]]),
                    columns=[feat_name],
                    dtype=dtype,
                )
            )
        df = pd.concat(col_dfs, axis=1)

        os.makedirs(ds_dir, exist_ok=True)
        df.to_csv(os.path.join(ds_dir, "data.csv"), index=False)
        np.savetxt(os.path.join(ds_dir, "labels.csv"), y)

        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=test_size, random_state=train_test_split_random_state
        )
        X_train.to_csv(os.path.join(ds_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(ds_dir, "X_test.csv"), index=False)
        np.savetxt(os.path.join(ds_dir, "y_train.csv"), y_train)
        np.savetxt(os.path.join(ds_dir, "y_test.csv"), y_test)

        with open(os.path.join(ds_dir, "meta.json"), "w") as fp:
            json.dump({
                "pred_type": pred_type,
                "openml_data_id": openml_data_id,
                "test_size": test_size,
                "train_test_split_random_state": train_test_split_random_state
            }, fp)

        return DatasetSplit(X_train, X_test, y_train, y_test, pred_type)

    ##############################################
    # Otherwise, read the CSV files and metadata #
    ##############################################
    meta_json_path = os.path.join(ds_dir, "meta.json")
    assert os.path.isfile(meta_json_path), "meta.json file missing"
    with open(meta_json_path, "r") as fp:
        pred_type = json.load(fp).get("pred_type")
        assert pred_type in ("regression", "classification"), f"Invalid pred_type: {pred_type}"

    X_train = pd.read_csv(os.path.join(ds_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(ds_dir, "X_test.csv"))
    y_train = np.loadtxt(os.path.join(ds_dir, "y_train.csv"))
    y_test = np.loadtxt(os.path.join(ds_dir, "y_test.csv"))
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel()
    return DatasetSplit(X_train, X_test, y_train, y_test, pred_type)
