import os
import pathlib

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from pddshap import PartialDependenceDecomposition
from pddshap.sampling.collocation_method import RandomSubsampleCollocation
from pddshap.sampling.conditioning_method import IndependentConditioningMethod


def save_predictions(arr: np.ndarray, path: str) -> None:
    """Save a 1-D or 2-D prediction array as a CSV with named columns."""
    if arr.ndim == 1:
        df = pd.DataFrame(arr, columns=["prediction"])
    else:
        df = pd.DataFrame(arr, columns=[f"output_{i}" for i in range(arr.shape[1])])
    df.to_csv(path, index=False)


if __name__ == "__main__":
    # Load shared config and experiment-specific config
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    shared_config_path = repo_root / "experiments" / "config" / "config.yaml"
    experiment_config_path = repo_root / "experiments" / "reuse_decomposition" / "config.yaml"
    datasets_config_path = repo_root / "experiments" / "config" / "datasets.yaml"

    with open(shared_config_path, "r") as fp:
        shared_cfg = yaml.safe_load(fp)
    with open(experiment_config_path, "r") as fp:
        experiment_cfg = yaml.safe_load(fp)
    with open(datasets_config_path, "r") as fp:
        datasets_cfg = yaml.safe_load(fp)

    cfg = {**shared_cfg, **experiment_cfg}

    data_dir = pathlib.Path(cfg["data_dir"])
    out_dir = pathlib.Path(cfg["out_dir"])
    ds_name = cfg["dataset"]
    pred_type = datasets_cfg[ds_name]["pred_type"]
    pdd_cfg = cfg["pdd"]

    # Load dataset from CSV files
    ds_data_dir = data_dir / ds_name
    X_train = pd.read_csv(ds_data_dir / "X_train.csv")
    X_test = pd.read_csv(ds_data_dir / "X_test.csv")
    y_train = np.loadtxt(ds_data_dir / "y_train.csv")
    y_test = np.loadtxt(ds_data_dir / "y_test.csv")
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    # Create the output directory for this dataset
    ds_out_dir = out_dir / ds_name
    os.makedirs(ds_out_dir, exist_ok=True)

    # Extract a background dataset from the training set and save it to disk
    bg_size = int(pdd_cfg.get("background_size", 100))
    bg_size = min(bg_size, len(X_train))
    X_bg = X_train.sample(n=bg_size, random_state=42)
    X_bg.to_csv(ds_out_dir / "X_bg.csv", index=False)

    # Train a gradient boosting model
    print(f"Training GradientBoosting ({pred_type}) on '{ds_name}'...")
    if pred_type == "classification":
        model = GradientBoostingClassifier()
    else:
        model = GradientBoostingRegressor()
    model.fit(X_train.to_numpy(), y_train)

    # Compute and save model predictions
    pred_fn = model.predict_proba if pred_type == "classification" else model.predict

    print("Saving model predictions...")
    save_predictions(pred_fn(X_train.to_numpy()), str(ds_out_dir / "gb_predictions_train.csv"))
    save_predictions(pred_fn(X_test.to_numpy()), str(ds_out_dir / "gb_predictions_test.csv"))

    # Build and fit the partial dependence decomposition on the background set
    print("Fitting PDD...")
    pdd = PartialDependenceDecomposition(
        model=pred_fn,
        collocation_method=RandomSubsampleCollocation(frac=pdd_cfg["collocation_frac"]),
        conditioning_method=IndependentConditioningMethod(X_bg.to_numpy()),
        estimator_type="knn",
        est_kwargs={"k": 3},
    )
    pdd.fit(
        X_train,
        X_bg,
        feature_set_selection=pdd_cfg["feature_set_selection"],
        max_size=pdd_cfg["max_size"],
    )

    # Compute and save PDD predictions
    print("Saving PDD predictions...")
    save_predictions(pdd(X_train.to_numpy()), str(ds_out_dir / "pdd_predictions_train.csv"))
    save_predictions(pdd(X_test.to_numpy()), str(ds_out_dir / "pdd_predictions_test.csv"))

    print("Done.")