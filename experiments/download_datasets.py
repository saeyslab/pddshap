import os
import pathlib

import numpy as np
import pandas as pd
import tqdm
import yaml
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

"""
OpenML Benchmarks:
- https://www.openml.org/search?type=benchmark&sort=tasks_included&study_type=task&id=99
- https://www.openml.org/search?type=benchmark&sort=tasks_included&study_type=task&id=269
- https://www.openml.org/search?type=benchmark&sort=tasks_included&study_type=task&id=297
- https://www.openml.org/search?type=benchmark&sort=tasks_included&study_type=task&id=299

UCI:
- http://archive.ics.uci.edu/ml/datasets/pen-based+recognition+of+handwritten+digits
- https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations
- https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
- https://archive.ics.uci.edu/ml/datasets/BlogFeedback
- https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1998+Data
- https://archive.ics.uci.edu/ml/datasets/Page+Blocks+Classification

"""


def download_dataset(
    ds_name: str, data_id: int, pred_type: str, data_dir: pathlib.Path
):
    ds_dir = data_dir / ds_name

    # Get dataset from OpenML
    ds = fetch_openml(data_id=data_id)
    # Drop nan rows
    y = ds.target[ds.data.notnull().all(axis=1)].to_numpy()
    df = ds.data.dropna().reset_index(drop=True)
    # Encode labels
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
        df, y, test_size=0.2, random_state=42
    )
    X_train.to_csv(os.path.join(ds_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(ds_dir, "X_test.csv"), index=False)
    np.savetxt(os.path.join(ds_dir, "y_train.csv"), y_train)
    np.savetxt(os.path.join(ds_dir, "y_test.csv"), y_test)


if __name__ == "__main__":
    # Read configuration
    with open("experiments/config/config.yaml", "r") as fp:
        config_dict = yaml.safe_load(fp)
        data_dir = pathlib.Path(config_dict["data_dir"])

    # Read datasets.yaml file
    with open("experiments/config/datasets.yaml", "r") as fp:
        datasets_dict = yaml.safe_load(fp)

    for ds_name in tqdm.tqdm(datasets_dict):
        data_id = datasets_dict[ds_name]["data_id"]
        pred_type = datasets_dict[ds_name]["pred_type"]
        download_dataset(ds_name, data_id, pred_type, data_dir)
