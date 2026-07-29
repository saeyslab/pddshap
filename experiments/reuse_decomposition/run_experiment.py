import json
import os
import pathlib

import h5py
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


def save_shap_values(arr: np.ndarray, feature_names: list[str], path: str) -> None:
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


def random_partial_ordering(
    num_elements: int, rng: np.random.Generator
) -> list[list[str]]:
    """Create a random partial ordering as a list of variable-length feature groups."""
    shuffled = list(rng.permutation(range(num_elements)))
    groups: list[list[str]] = []
    current: list[str] = []
    for feature in shuffled:
        current.append(int(feature))
        # End the current rank with moderate probability to get variable group sizes.
        if len(current) > 1 and rng.random() < 0.35:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    # Avoid degenerate single-group ordering for multi-feature datasets.
    if len(groups) == 1 and len(shuffled) > 1:
        split = int(rng.integers(1, len(shuffled)))
        groups = [shuffled[:split], shuffled[split:]]

    return groups


if __name__ == "__main__":
    # Load shared config and experiment-specific config
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    shared_config_path = repo_root / "experiments" / "config" / "config.yaml"
    experiment_config_path = (
        repo_root / "experiments" / "reuse_decomposition" / "config.yaml"
    )
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
    random_seed = int(pdd_cfg.get("random_seed", 42))
    bg_size = int(pdd_cfg.get("background_size", 100))
    bg_size = min(bg_size, len(X_train))
    X_bg = X_train.sample(n=bg_size, random_state=random_seed)
    X_bg.to_csv(ds_out_dir / "X_bg.csv", index=False)

    # Extract the rows for which explanations are generated and save them to disk
    num_explanations = int(pdd_cfg.get("num_explanations", 100))
    num_explanations = min(num_explanations, len(X_test))
    X_explain = X_test.sample(n=num_explanations, random_state=random_seed)
    X_explain.to_csv(ds_out_dir / "X_explain.csv", index=False)

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
    save_predictions(
        pred_fn(X_train.to_numpy()), str(ds_out_dir / "gb_predictions_train.csv")
    )
    save_predictions(
        pred_fn(X_test.to_numpy()), str(ds_out_dir / "gb_predictions_test.csv")
    )

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
        n_jobs=-1,
    )
    with open(ds_out_dir / "data_signature.json", "w") as fp:
        fp.write(pdd.data_signature.to_json())

    # Compute and save PDD predictions
    print("Saving PDD predictions...")
    save_predictions(
        pdd(X_train.to_numpy()), str(ds_out_dir / "pdd_predictions_train.csv")
    )
    save_predictions(
        pdd(X_test.to_numpy()), str(ds_out_dir / "pdd_predictions_test.csv")
    )

    # Generate and save SHAP values with the fitted decomposition
    print("Saving PDD-SHAP explanations...")
    shap_values = pdd.shapley_values(X_explain)
    save_shap_values(
        shap_values, list(X_explain.columns), str(ds_out_dir / "pdd_shap.h5")
    )

    # Generate and save SHAP values using a random partial ordering
    print("Saving PDD-SHAP explanations with partial ordering...")
    rng = np.random.default_rng(random_seed)
    partial_ordering = random_partial_ordering(len(X_explain.columns), rng)
    with open(ds_out_dir / "partial_ordering.json", "w") as fp:
        json.dump(partial_ordering, fp, indent=2)

    shap_values_partial_order = pdd.shapley_values(
        X_explain, partial_ordering=partial_ordering
    )
    save_shap_values(
        shap_values_partial_order,
        list(X_explain.columns),
        str(ds_out_dir / "pdd_shap_partial_order.h5"),
    )

    print("Done.")
