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


def load(
    name: str,
    data_dir: str,
    *,
    pred_type: str | None = None,
    data_id: int | None = None,
    download: bool = False,
) -> DatasetSplit:
    """
    Load a cached dataset's train/test split as a :class:`DatasetSplit`.

    If ``download`` is True and the dataset isn't cached yet under ``data_dir/name``,
    it is fetched from OpenML first (requires ``data_id`` and ``pred_type``).
    ``pred_type`` can otherwise be omitted if the dataset was already cached (it's recorded in ``meta.json``).
    """
    ds_dir = os.path.join(data_dir, name)
    cached = os.path.isfile(os.path.join(ds_dir, "X_train.csv"))

    if not cached:
        if not download:
            raise FileNotFoundError(
                f"Dataset '{name}' not found under {data_dir}. Pass download=True to fetch it."
            )
        if data_id is None or pred_type is None:
            raise ValueError(
                f"Dataset '{name}' is not cached yet; data_id and pred_type are required to download it."
            )
        _fetch_and_cache(name, data_id, pred_type, data_dir)

    resolved_pred_type = pred_type or _read_meta(ds_dir).get("pred_type")
    if resolved_pred_type is None:
        raise ValueError(
            f"pred_type for '{name}' is unknown (no meta.json and none was passed in)."
        )

    X_train = pd.read_csv(os.path.join(ds_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(ds_dir, "X_test.csv"))
    y_train = np.loadtxt(os.path.join(ds_dir, "y_train.csv"))
    y_test = np.loadtxt(os.path.join(ds_dir, "y_test.csv"))
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    return DatasetSplit(X_train, X_test, y_train, y_test, resolved_pred_type)

def _read_meta(ds_dir: str) -> dict:
    meta_path = os.path.join(ds_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return {}
    with open(meta_path, "r") as fp:
        return json.load(fp)


def _fetch_and_cache(
    name: str, data_id: int, pred_type: str, data_dir: str, test_size: float = 0.2, random_state: int = 42
) -> None:
    """
    Download a dataset from OpenML, encode it (ordinal-encode categorical columns,
    standard-scale numerical ones and the regression target), split it into train/test,
    and cache everything under ``data_dir/name``.

    Writes ``data.csv``/``labels.csv`` (the full encoded dataset), a train/test split
    (``X_train.csv``, ``X_test.csv``, ``y_train.csv``, ``y_test.csv``), and ``meta.json``
    (``{"pred_type": ...}``) so later calls to :func:`pddshap.datasets.load` don't need
    ``pred_type`` repeated.

    :param name: name under which to cache the dataset (``data_dir/name/...``).
    :param data_id: OpenML dataset id to fetch.
    :param pred_type: ``"classification"`` or ``"regression"``.
    :param data_dir: root directory under which datasets are cached.
    """
    ds_dir = os.path.join(data_dir, name)

    ds = fetch_openml(data_id=data_id)
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
        df, y, test_size=test_size, random_state=random_state
    )
    X_train.to_csv(os.path.join(ds_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(ds_dir, "X_test.csv"), index=False)
    np.savetxt(os.path.join(ds_dir, "y_train.csv"), y_train)
    np.savetxt(os.path.join(ds_dir, "y_test.csv"), y_test)

    with open(os.path.join(ds_dir, "meta.json"), "w") as fp:
        json.dump({"pred_type": pred_type}, fp)

