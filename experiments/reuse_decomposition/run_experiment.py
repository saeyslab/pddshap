import json
import os
import pathlib

import numpy as np

from pddshap import PartialDependenceDecomposition
from pddshap.datasets import load
from pddshap.experiments import load_experiment_config, make_model, predict_fn, sample_rows, save_predictions, save_shap_values
from pddshap.sampling.collocation_method import RandomSubsampleCollocation
from pddshap.sampling.conditioning_method import IndependentConditioningMethod
from pddshap.util import random_partial_ordering


if __name__ == "__main__":
    cfg = load_experiment_config(pathlib.Path(__file__).parent)
    datasets_cfg = cfg["datasets"]
    data_dir = cfg["data_dir"]
    out_dir = cfg["out_dir"]
    ds_name = cfg["dataset"]
    pdd_cfg = cfg["pdd"]

    # Load dataset
    split = load(
        ds_name, data_dir,
        pred_type=datasets_cfg[ds_name]["pred_type"],
        data_id=datasets_cfg[ds_name]["data_id"],
        download=True,
    )

    # Create the output directory for this dataset
    ds_out_dir = pathlib.Path(out_dir) / ds_name
    os.makedirs(ds_out_dir, exist_ok=True)

    # Extract a background dataset from the training set and save it to disk
    random_seed = int(pdd_cfg.get("random_seed", 42))
    bg_size = int(pdd_cfg.get("background_size", 100))
    X_bg = sample_rows(split.X_train, bg_size, random_state=random_seed, label="background")
    X_bg.to_csv(ds_out_dir / "X_bg.csv", index=False)

    # Extract the rows for which explanations are generated and save them to disk
    num_explanations = int(pdd_cfg.get("num_explanations", 100))
    X_explain = sample_rows(split.X_test, num_explanations, random_state=random_seed, label="explanations")
    X_explain.to_csv(ds_out_dir / "X_explain.csv", index=False)

    # Train a gradient boosting model
    print(f"Training GradientBoosting ({split.pred_type}) on '{ds_name}'...")
    model = make_model(split.pred_type, "gradientboosting")
    model.fit(split.X_train.to_numpy(), split.y_train)
    pred_fn = predict_fn(model, split.pred_type)

    print("Saving model predictions...")
    save_predictions(
        pred_fn(split.X_train.to_numpy()), str(ds_out_dir / "gb_predictions_train.csv")
    )
    save_predictions(
        pred_fn(split.X_test.to_numpy()), str(ds_out_dir / "gb_predictions_test.csv")
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
        split.X_train, X_bg,
        max_size=pdd_cfg["max_size"],
        n_jobs=-1,
    )
    with open(ds_out_dir / "data_signature.json", "w") as fp:
        fp.write(pdd.data_signature.to_json())

    # Compute and save PDD predictions
    print("Saving PDD predictions...")
    save_predictions(
        pdd(split.X_train.to_numpy()), str(ds_out_dir / "pdd_predictions_train.csv")
    )
    save_predictions(
        pdd(split.X_test.to_numpy()), str(ds_out_dir / "pdd_predictions_test.csv")
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
